from __future__ import annotations

import logging
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Loss helpers
# ──────────────────────────────────────────────────────────────────────────

def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    if neg.dim() == 1:
        neg = neg.unsqueeze(1)
    return -F.logsigmoid(pos.unsqueeze(1) - neg).mean()


def info_nce(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Listwise NCE: log softmax over [pos | negs]."""
    if neg.dim() == 1:
        neg = neg.unsqueeze(1)
    logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def sampled_softmax_loss(
    pos: torch.Tensor,
    neg: torch.Tensor,
    temperature: float = 0.7,
) -> torch.Tensor:
    """
    Disease-wise sampled softmax for ranking.

    [fix-6] This function receives RAW (non-temperature-scaled) scores.
    The caller must pass scores obtained with with_temperature=False so that
    `temperature` here is the sole temperature applied to the listwise loss,
    independent of the learnable log_tau used by BPR/InfoNCE.
    """
    if neg.dim() == 1:
        neg = neg.unsqueeze(1)
    tau = max(float(temperature), 1e-6)
    logits = torch.cat([pos.unsqueeze(1), neg], dim=1) / tau
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def node_contrastive_loss(
    node_repr: torch.Tensor,
    pos_pairs: torch.Tensor,
    batch_size: int = 1024,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    In-batch node contrastive loss over positive node pairs.

    Each row of `pos_pairs` is (anchor_global_idx, positive_global_idx). The
    diagonal pair is treated as positive; the other positives in the batch act
    as negatives. The loss is symmetric anchor→positive and positive→anchor.
    """
    if pos_pairs is None or pos_pairs.numel() == 0:
        return node_repr.new_tensor(0.0)
    n = pos_pairs.size(0)
    take = min(batch_size, n)
    idx = torch.randint(0, n, (take,), device=pos_pairs.device)
    pairs = pos_pairs[idx]
    z_a = F.normalize(node_repr[pairs[:, 0]], dim=-1)
    z_b = F.normalize(node_repr[pairs[:, 1]], dim=-1)
    logits = (z_a @ z_b.t()) / max(float(temperature), 1e-6)
    labels = torch.arange(take, dtype=torch.long, device=node_repr.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.t(), labels))


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────

def compute_ranking_metrics(ranks, ks=(10, 20, 50, 100)):
    # [fix-10] Return zeros for empty rank lists instead of NaN + RuntimeWarning.
    if len(ranks) == 0:
        out = {'MRR': 0.0, 'MeanRank': 0.0, 'MedianRank': 0.0}
        for k in ks:
            out[f'Hits@{k}'] = 0.0
        return out
    arr = np.array(ranks, dtype=np.float64)
    out = {
        'MRR':        float(np.mean(1.0 / arr)),
        'MeanRank':   float(np.mean(arr)),
        'MedianRank': float(np.median(arr)),
    }
    for k in ks:
        out[f'Hits@{k}'] = float(np.mean(arr <= k))
    return out


def _flat_pair_index(pi: torch.Tensor, pj: torch.Tensor, n: int) -> torch.Tensor:
    pi, pj = pi.long(), pj.long()
    return pi * (2*n - pi - 1) // 2 + (pj - pi - 1)


# ──────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model, kg_graph, config: dict, device: torch.device):
        self.model  = model.to(device)
        self.kg     = kg_graph.to(device)
        self.config = config
        self.device = device

        if hasattr(self.model.kge, 'set_graph'):
            self.model.kge.set_graph(self.kg.edge_index, self.kg.edge_type)
            logger.info("[v8] Bound KG edge_index/edge_type to CompGCN encoder.")

        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[v8] Total trainable parameters: {n:,}")
        self._contrastive_pairs: Optional[torch.Tensor] = None

    def build_contrastive_pairs(
        self,
        max_pairs_per_bucket: int = 12,
        max_total_pairs: int = 250_000,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Build node-positive pairs from KG neighborhoods.

        Drug positives:
          - share a target gene
          - share an indicated disease

        Disease positives:
          - share a gene
          - share an HPO term
          - are connected by comorbidity / disease-disease edges

        This is intentionally lightweight: it adds a contrastive signal to KGE
        pretraining without requiring any external ontology or text embedding.
        """
        rng = random.Random(seed)
        edge_index = self.kg.edge_index.detach().cpu()
        node_type = self.kg.node_type_mask.detach().cpu()

        drug_type, disease_type, gene_type, hpo_type = 0, 1, 2, 3
        buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        direct_pairs = set()

        src_all = edge_index[0].tolist()
        dst_all = edge_index[1].tolist()
        for src, dst in zip(src_all, dst_all):
            st = int(node_type[src])
            dt = int(node_type[dst])
            if st == drug_type and dt in (gene_type, disease_type):
                buckets[(st, dst)].append(src)
            elif st == disease_type and dt in (gene_type, hpo_type):
                buckets[(st, dst)].append(src)
            elif st == disease_type and dt == disease_type and src != dst:
                a, b = (src, dst) if src < dst else (dst, src)
                direct_pairs.add((a, b))

        pairs = set(direct_pairs)
        for nodes in buckets.values():
            uniq = list(dict.fromkeys(nodes))
            if len(uniq) < 2:
                continue
            cap = min(max_pairs_per_bucket, len(uniq) * (len(uniq) - 1) // 2)
            if len(uniq) <= 8:
                candidates = []
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        candidates.append((uniq[i], uniq[j]))
                rng.shuffle(candidates)
                for a, b in candidates[:cap]:
                    pairs.add((a, b) if a < b else (b, a))
            else:
                attempts = 0
                added = 0
                while added < cap and attempts < cap * 20:
                    attempts += 1
                    a, b = rng.sample(uniq, 2)
                    if a == b:
                        continue
                    pair = (a, b) if a < b else (b, a)
                    before = len(pairs)
                    pairs.add(pair)
                    added += int(len(pairs) > before)
            if len(pairs) >= max_total_pairs:
                break

        pairs_list = list(pairs)
        rng.shuffle(pairs_list)
        pairs_list = pairs_list[:max_total_pairs]
        if not pairs_list:
            out = torch.empty(0, 2, dtype=torch.long, device=self.device)
        else:
            out = torch.tensor(pairs_list, dtype=torch.long, device=self.device)
        self._contrastive_pairs = out
        logger.info(
            f"[v8] Built {out.size(0):,} contrastive node pairs "
            f"(max_per_bucket={max_pairs_per_bucket}, max_total={max_total_pairs:,})"
        )
        return out

    # ──────────────────────────────────────────────────────────────────
    # Pair index for full-space ranking — built lazily, cached
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _build_pair_index(self):
        if hasattr(self, '_pair_i'):
            return self._pair_i, self._pair_j
        n = self.kg.n_drug
        idx = torch.triu_indices(n, n, offset=1, device=self.device)
        self._pair_i = idx[0].to(torch.int32)
        self._pair_j = idx[1].to(torch.int32)
        logger.info(
            f"[v8] Pair index built: {n} drugs → {self._pair_i.numel():,} pairs")
        return self._pair_i, self._pair_j

    # ──────────────────────────────────────────────────────────────────
    # Phase A — KGE pretraining
    # ──────────────────────────────────────────────────────────────────

    def train_kge(self,
                  n_epochs:    int = 50,
                  batch_size:  int = 16384,
                  neg_per_pos: int = 64,
                  margin:      float = 9.0,
                  adv_temp:    float = 1.0,
                  lr:          float = 5e-3,
                  weight_decay:float = 1e-6,
                  contrastive_lambda: float = 0.0,
                  contrastive_batch: int = 1024,
                  contrastive_temp: float = 0.2,
                  contrastive_max_pairs_per_bucket: int = 12,
                  contrastive_max_total_pairs: int = 250_000,
                  contrastive_warmup_epochs: int = 3):
        """
        Phase A: train the KGE encoder on the KG.

        DistMult keeps the V1 forward-only behavior. CompGCN uses forward +
        inverse KG edges because message passing and asymmetric relation types
        benefit from direction-specific supervision.
        """
        logger.info("=" * 60)
        kge_name = getattr(self.model, 'kge_model', self.model.kge.__class__.__name__)
        logger.info(f"[v8] Phase A: {kge_name} KGE pretraining")
        logger.info("=" * 60)

        ei  = self.kg.edge_index    # [2, 2E]
        et  = self.kg.edge_type     # [2E]
        n_rel_fwd = self.kg.n_relations
        if getattr(self.model.kge, 'uses_inverse_edges', False):
            h = ei[0].long()
            r = et.long()
            t = ei[1].long()
            edge_scope = "forward+inverse"
        else:
            fwd_mask = et < n_rel_fwd
            h = ei[0][fwd_mask].long()
            r = et[fwd_mask].long()
            t = ei[1][fwd_mask].long()
            edge_scope = "forward"
        n_edges = h.numel()
        logger.info(f"  {edge_scope} triplets: {n_edges:,}; "
                    f"batch={batch_size}; neg/pos={neg_per_pos}; "
                    f"γ={margin}; α={adv_temp}")

        contrastive_on = contrastive_lambda > 0
        contrastive_pairs = None
        if contrastive_on:
            contrastive_pairs = self.build_contrastive_pairs(
                max_pairs_per_bucket=contrastive_max_pairs_per_bucket,
                max_total_pairs=contrastive_max_total_pairs,
            )
            if contrastive_pairs.numel() == 0:
                logger.warning("[v8] No contrastive pairs built; disabling contrastive loss.")
                contrastive_on = False
        logger.info(
            f"  [v8] Contrastive={'on' if contrastive_on else 'off'}"
            + (f" (λ={contrastive_lambda}, B={contrastive_batch}, "
               f"τ={contrastive_temp}, warmup={contrastive_warmup_epochs} ep)"
               if contrastive_on else "")
        )

        opt = AdamW(self.model.kge.parameters(), lr=lr, weight_decay=weight_decay)
        sch = CosineAnnealingLR(opt, T_max=n_epochs)
        n_nodes = self.model.n_total

        for epoch in range(n_epochs):
            t0 = time.time()
            self.model.train()

            perm = torch.randperm(n_edges, device=self.device)
            ep_loss = 0.0
            n_steps = 0

            for s in range(0, n_edges, batch_size):
                e = min(s + batch_size, n_edges)
                idx = perm[s:e]
                hb, rb, tb = h[idx], r[idx], t[idx]
                B = hb.size(0)

                node_repr = None
                if hasattr(self.model.kge, 'encode_nodes'):
                    node_repr = self.model.kge.encode_nodes()

                # Positive scores
                if node_repr is not None and hasattr(self.model.kge, 'score_with_node_embeddings'):
                    pos = self.model.kge.score_with_node_embeddings(node_repr, hb, rb, tb)
                else:
                    pos = self.model.kge.score(hb, rb, tb)            # [B]

                # Sample neg_per_pos corruptions per pos: replace head OR tail
                # with prob 0.5 each.
                replace_head = torch.rand(B, neg_per_pos, device=self.device) < 0.5
                rand_node    = torch.randint(0, n_nodes,
                                             (B, neg_per_pos),
                                             device=self.device)
                hb_e = hb.unsqueeze(1).expand(B, neg_per_pos)
                tb_e = tb.unsqueeze(1).expand(B, neg_per_pos)
                rb_e = rb.unsqueeze(1).expand(B, neg_per_pos)
                neg_h = torch.where(replace_head, rand_node, hb_e)
                neg_t = torch.where(replace_head, tb_e,    rand_node)

                if node_repr is not None and hasattr(self.model.kge, 'score_with_node_embeddings'):
                    neg = self.model.kge.score_with_node_embeddings(
                        node_repr,
                        neg_h.reshape(-1),
                        rb_e.reshape(-1),
                        neg_t.reshape(-1),
                    ).view(B, neg_per_pos)
                else:
                    neg = self.model.kge.score(
                        neg_h.reshape(-1), rb_e.reshape(-1), neg_t.reshape(-1)
                    ).view(B, neg_per_pos)

                # Self-adversarial weights (RotatE-style):
                #   weight_k = softmax(α · neg_k) over k (no grad)
                # Loss form (used because DistMult scores are unbounded
                # diagonal dot products, equivalent to logistic NCE):
                #   L = -log σ(pos) + Σ_k w_k · softplus(neg_k)
                with torch.no_grad():
                    w = F.softmax(adv_temp * neg, dim=1)
                pos_loss = -F.logsigmoid(pos).mean()
                neg_loss = (w * F.softplus(neg)).sum(1).mean()
                loss = pos_loss + neg_loss
                loss_cl = None
                if contrastive_on and (epoch + 1) > contrastive_warmup_epochs:
                    if node_repr is None:
                        node_repr = self.model.kge.node_emb.weight
                    loss_cl = node_contrastive_loss(
                        node_repr=node_repr,
                        pos_pairs=contrastive_pairs,
                        batch_size=contrastive_batch,
                        temperature=contrastive_temp,
                    )
                    loss = loss + contrastive_lambda * loss_cl
                _ = margin   # kept as a hyperparameter handle (for future RotatE swap)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.kge.parameters(), 1.0)
                opt.step()

                ep_loss += loss.item()
                n_steps += 1

            sch.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                cl_part = ""
                if contrastive_on:
                    cl_part = " | cl=on" if (epoch + 1) > contrastive_warmup_epochs else " | cl=warmup"
                logger.info(
                    f"  [v8] Ep {epoch+1:3d}/{n_epochs} | loss={ep_loss/n_steps:.4f} | "
                    f"LR={opt.param_groups[0]['lr']:.2e}{cl_part} | dt={time.time()-t0:.1f}s")

        logger.info("[v8] Phase A complete.")

    # ──────────────────────────────────────────────────────────────────
    # OHNM — Online Hard Negative Mining
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _mine_hard_negatives(self,
                             train_pos_set: Dict[int, set],
                             n_hard:        int = 8,
                             pool_factor:   int = 30) -> Dict[int, torch.Tensor]:
        """
        For each disease present in train_pos_set, score a random pool of
        candidate (i, j) drug pairs with the *current* model and keep the
        top-`n_hard` highest-scoring ones (after filtering known positives).

        Returns: dict[disease_idx → LongTensor of shape (n_hard, 2)] with
                 (i, j) drug indices.
        """
        self.model.eval()
        drug_emb, disease_emb = self.model.encode_all()
        h_drug_pre = self.model.scorer.precompute_drug_features(drug_emb)
        n_drugs = self.kg.n_drug
        device  = self.device

        hard_neg: Dict[int, torch.Tensor] = {}
        pool_target = max(n_hard * pool_factor, n_hard * 8)

        for d, pos_set in train_pos_set.items():
            # Sample with 2x headroom so filtering still leaves enough
            big = pool_target * 2
            ci = torch.randint(0, n_drugs, (big,), device=device)
            cj = torch.randint(0, n_drugs, (big,), device=device)
            keep = ci != cj
            ci, cj = ci[keep], cj[keep]
            ca = torch.minimum(ci, cj)
            cb = torch.maximum(ci, cj)

            # Vectorised filter against known positives
            if pos_set:
                pos_arr = torch.tensor(list(pos_set), dtype=torch.long, device=device)
                pos_enc = pos_arr[:, 0] * n_drugs + pos_arr[:, 1]
                ca_enc  = ca * n_drugs + cb
                fmask   = ~torch.isin(ca_enc, pos_enc)
                ca, cb  = ca[fmask], cb[fmask]
            if ca.numel() < n_hard:
                continue

            # Score with current model (vectorised)
            q, a_, b_, c_ = self.model.scorer.disease_weights(disease_emb[d])
            h_i = h_drug_pre[ca]
            h_j = h_drug_pre[cb]
            scores = ((q * (h_i * h_j)).sum(-1)
                    + (a_ * (h_i + h_j)).sum(-1)
                    + (b_ * (h_i - h_j).abs()).sum(-1)
                    + c_)
            k_take = min(n_hard, scores.numel())
            _, top_idx = torch.topk(scores, k=k_take)
            hard_neg[d] = torch.stack([ca[top_idx], cb[top_idx]], dim=1)

        self.model.train()
        return hard_neg

    # ──────────────────────────────────────────────────────────────────
    # Phase B — Combination scorer training
    # ──────────────────────────────────────────────────────────────────

    def train_scorer(self,
                     train_loader, val_loader, test_loader,
                     n_epochs:    int = 100,
                     lr:          float = 5e-4,
                     emb_lr:      float = 5e-5,
                     weight_decay:float = 1e-3,
                     neg_k:       int = 16,
                     loss_mix:    float = 0.5,
                     freeze_emb:  bool = False,
                     patience:    int = 12,
                     ckpt:        str = 'best_v8.pt',
                     # ── OHNM ──────────────────────────────────────────────
                     ohnm_interval:    int = 0,
                     ohnm_n_hard:      int = 8,
                     ohnm_pool_factor: int = 30,
                     ohnm_mix_ratio:   float = 0.5,
                     # ── drug_pairs auxiliary loss ─────────────────────────
                     drug_pairs_tensor: Optional[torch.Tensor] = None,
                     aux_pair_lambda:   float = 0.0,
                     aux_score_weight:  float = 0.0,
                     aux_pair_batch:    int = 256,
                     aux_pair_neg_k:    int = 16,
                     # ── first-order disease-drug auxiliary prior ──────────
                     aux_disease_drug_lambda: float = 0.0,
                     aux_disease_drug_score_weight: float = 0.0,
                     # ── v8 disease-wise sampled-softmax ranking ───────────
                     listwise_lambda: float = 0.0,
                     listwise_num_neg: int = 128,
                     listwise_temp: float = 0.7):
        """
        Phase B: train the CompactPairScorer with frozen-or-low-LR embeddings.

        loss = loss_mix · BPR + (1 - loss_mix) · InfoNCE
             + listwise_lambda · sampled_softmax(raw_pos, raw_same-disease_negs)
             + aux_pair_lambda · BPR(drug_pairs aux head)

        [fix-6] The listwise term uses raw scores (with_temperature=False) so
        listwise_temp is the sole temperature for that loss, decoupled from the
        learnable log_tau used by BPR and InfoNCE.

        With ohnm_interval > 0, hard negatives are mined every that many
        epochs and substituted for `int(K · ohnm_mix_ratio)` of the
        DataLoader's random negatives.
        """
        logger.info("=" * 60)
        logger.info("[v8] Phase B: Compact Pair Scorer training")
        logger.info("=" * 60)

        emb_params    = list(self.model.kge.parameters())
        scorer_params = (list(self.model.scorer.parameters()) +
                         list(self.model.aux_pair_head.parameters()) +
                         list(self.model.aux_disease_drug_head.parameters()))

        # [fix-8 note] emb_lr=0 now truly freezes embeddings instead of
        # wasting gradient computation with zero-update steps.
        if freeze_emb or emb_lr == 0.0:
            for p in emb_params:
                p.requires_grad_(False)
            param_groups = [{'params': scorer_params, 'lr': lr,
                             'weight_decay': weight_decay}]
            logger.info("  Embeddings frozen.")
        else:
            param_groups = [
                {'params': emb_params,    'lr': emb_lr,
                 'weight_decay': 0.0},                       # no L2 on embeddings
                {'params': scorer_params, 'lr': lr,
                 'weight_decay': weight_decay},
            ]
            logger.info(f"  [v8] Embedding LR={emb_lr:.2e}, scorer LR={lr:.2e}")

        opt = AdamW(param_groups)
        sch = CosineAnnealingLR(opt, T_max=n_epochs)

        # Build train_pos_set for eval-time masking and OHNM filtering
        train_pos_set: Dict[int, set] = defaultdict(set)
        for batch in train_loader:
            for row in batch['pos']:
                d, i, j = int(row[0]), int(row[1]), int(row[2])
                ci, cj = min(i, j), max(i, j)
                train_pos_set[d].add((ci, cj))
        self.train_pos_set = train_pos_set

        n_drugs = self.kg.n_drug
        device  = self.device

        ohnm_on    = ohnm_interval > 0
        aux_on     = drug_pairs_tensor is not None and aux_pair_lambda > 0
        aux_dd_on  = aux_disease_drug_lambda > 0
        listwise_on = listwise_lambda > 0 and listwise_num_neg > 0
        self.model.aux_score_weight = float(aux_score_weight if aux_on else 0.0)
        self.model.aux_disease_drug_score_weight = float(
            aux_disease_drug_score_weight if aux_dd_on else 0.0)
        logger.info(
            f"  [v8] OHNM={'on' if ohnm_on else 'off'}"
            + (f" (every {ohnm_interval} ep, n_hard={ohnm_n_hard}, "
               f"pool×{ohnm_pool_factor}, mix={ohnm_mix_ratio:.2f})" if ohnm_on else "")
            + f" | DrugPairsAux={'on' if aux_on else 'off'}"
            + (f" (λ={aux_pair_lambda}, B={aux_pair_batch}, "
               f"neg_k={aux_pair_neg_k}, score_w={self.model.aux_score_weight}, "
               f"|pairs|={drug_pairs_tensor.size(0):,})"
               if aux_on else "")
            + f" | DiseaseDrugAux={'on' if aux_dd_on else 'off'}"
            + (f" (λ={aux_disease_drug_lambda}, "
               f"score_w={self.model.aux_disease_drug_score_weight})"
               if aux_dd_on else "")
            + f" | Listwise={'on' if listwise_on else 'off'}"
            + (f" (λ={listwise_lambda}, neg={listwise_num_neg}, "
               f"τ={listwise_temp})" if listwise_on else "")
        )

        best_mrr   = 0.0
        no_improve = 0
        hard_neg_dict: Dict[int, torch.Tensor] = {}

        for epoch in range(n_epochs):
            t0 = time.time()
            self.model.train()
            ep_loss = ep_main = ep_aux = ep_aux_dd = ep_listwise = 0.0
            n_steps = 0

            # ── OHNM refresh ───────────────────────────────────────────
            if ohnm_on and (epoch % ohnm_interval == 0):
                t_mine = time.time()
                hard_neg_dict = self._mine_hard_negatives(
                    train_pos_set,
                    n_hard      = ohnm_n_hard,
                    pool_factor = ohnm_pool_factor,
                )
                logger.info(f"  [v8] OHNM: mined hard negs for {len(hard_neg_dict)} "
                            f"diseases in {time.time()-t_mine:.1f}s "
                            f"(epoch {epoch+1})")

            for batch in train_loader:
                pos  = batch['pos'].to(device)         # [B, 3]
                negs = batch['negs'].to(device)        # [B, K, 3]
                B, K, _ = negs.shape

                # ── OHNM mix ───────────────────────────────────────────
                if ohnm_on and hard_neg_dict and ohnm_mix_ratio > 0:
                    n_hard_use = max(1, int(K * ohnm_mix_ratio))
                    hard_rows = []
                    for b in range(B):
                        d_b   = int(pos[b, 0])
                        hn    = hard_neg_dict.get(d_b, None)
                        rows  = []
                        if hn is not None and hn.size(0) > 0:
                            take = min(n_hard_use, hn.size(0))
                            perm_h = torch.randperm(hn.size(0), device=device)[:take]
                            hn_take = hn[perm_h]
                            for k_ in range(take):
                                rows.append((d_b, int(hn_take[k_, 0]),
                                                  int(hn_take[k_, 1])))
                        k_pad = 0
                        while len(rows) < n_hard_use:
                            rows.append(tuple(negs[b, k_pad % K].tolist()))
                            k_pad += 1
                        hard_rows.append(torch.tensor(rows, dtype=torch.long,
                                                      device=device))
                    hard_t = torch.stack(hard_rows, 0)              # [B, n_hard, 3]
                    negs   = torch.cat([hard_t, negs[:, n_hard_use:]], dim=1)
                    K = negs.size(1)

                drug_emb, disease_emb = self.model.encode_all_grad()

                # Main triplet loss — uses learnable temperature (with_temperature=True)
                e_d   = disease_emb[pos[:, 0]]
                e_i   = drug_emb   [pos[:, 1]]
                e_j   = drug_emb   [pos[:, 2]]
                pos_s = self.model.score_triplets(
                    e_d, e_i, e_j, with_temperature=True)['final']

                neg_flat = negs.reshape(B*K, 3)
                e_dn = disease_emb[neg_flat[:, 0]]
                e_in = drug_emb   [neg_flat[:, 1]]
                e_jn = drug_emb   [neg_flat[:, 2]]
                neg_s = self.model.score_triplets(
                    e_dn, e_in, e_jn, with_temperature=True)['final'].reshape(B, K)

                loss_bpr  = bpr_loss(pos_s, neg_s)
                loss_nce  = info_nce(pos_s, neg_s)
                loss_main = loss_mix * loss_bpr + (1 - loss_mix) * loss_nce
                loss      = loss_main

                # ── v8 disease-wise sampled-softmax listwise loss ──────
                # [fix-6] Use raw scores (with_temperature=False) so listwise_temp
                # is the only temperature applied to this loss term.
                if listwise_on:
                    L = int(listwise_num_neg)
                    lw_i = torch.randint(0, n_drugs, (B, L), device=device)
                    lw_j = torch.randint(0, n_drugs, (B, L), device=device)

                    # [fix-5] Bounded loop: after max_iters, force resolve with offset.
                    bad = lw_i == lw_j
                    max_iters = max(n_drugs * 2, 20)
                    iters = 0
                    while bad.any() and iters < max_iters:
                        lw_j[bad] = torch.randint(
                            0, n_drugs, (int(bad.sum()),), device=device)
                        bad = lw_i == lw_j
                        iters += 1
                    if bad.any():
                        lw_j[bad] = (lw_i[bad] + 1) % n_drugs

                    lw_a = torch.minimum(lw_i, lw_j)
                    lw_b = torch.maximum(lw_i, lw_j)

                    lw_flat_i = lw_a.reshape(-1)
                    lw_flat_j = lw_b.reshape(-1)
                    lw_d = disease_emb[pos[:, 0]].repeat_interleave(L, dim=0)
                    lw_ei = drug_emb[lw_flat_i]
                    lw_ej = drug_emb[lw_flat_j]

                    # Raw scores for listwise — temperature applied inside
                    # sampled_softmax_loss via listwise_temp only (fix-6).
                    lw_raw_pos = self.model.score_triplets(
                        e_d, e_i, e_j, with_temperature=False)['final']
                    lw_neg_s = self.model.score_triplets(
                        lw_d, lw_ei, lw_ej, with_temperature=False)['final'].reshape(B, L)

                    loss_lw = sampled_softmax_loss(
                        lw_raw_pos, lw_neg_s, temperature=listwise_temp)
                    loss = loss + listwise_lambda * loss_lw
                    ep_listwise += float(loss_lw.item())

                # ── disease-drug auxiliary BPR loss ────────────────────
                if aux_dd_on:
                    h_i = self.model.scorer.precompute_drug_features(e_i)
                    h_j = self.model.scorer.precompute_drug_features(e_j)
                    pos_dd = (
                        self.model.aux_disease_drug_head.score(e_d, h_i)
                        + self.model.aux_disease_drug_head.score(e_d, h_j)
                    )
                    h_in = self.model.scorer.precompute_drug_features(e_in)
                    h_jn = self.model.scorer.precompute_drug_features(e_jn)
                    neg_dd = (
                        self.model.aux_disease_drug_head.score(e_dn, h_in)
                        + self.model.aux_disease_drug_head.score(e_dn, h_jn)
                    ).reshape(B, K)
                    loss_dd = bpr_loss(pos_dd, neg_dd)
                    loss = loss + aux_disease_drug_lambda * loss_dd
                    ep_aux_dd += float(loss_dd.item())

                # ── drug_pairs auxiliary BPR loss ──────────────────────
                if aux_on:
                    P = drug_pairs_tensor.size(0)
                    ap_B = min(aux_pair_batch, P)
                    ap_idx = torch.randint(0, P, (ap_B,), device=device)
                    ap_pos = drug_pairs_tensor[ap_idx]              # [ap_B, 2]
                    ap_pi, ap_pj = ap_pos[:, 0], ap_pos[:, 1]

                    h_pi = self.model.scorer.precompute_drug_features(drug_emb[ap_pi])
                    h_pj = self.model.scorer.precompute_drug_features(drug_emb[ap_pj])
                    pos_aux = self.model.aux_pair_head(h_pi, h_pj)  # [ap_B]

                    ap_ni = torch.randint(0, n_drugs, (ap_B, aux_pair_neg_k), device=device)
                    ap_nj = torch.randint(0, n_drugs, (ap_B, aux_pair_neg_k), device=device)
                    bad = ap_ni == ap_nj
                    # [fix-5] Bounded loop for aux pair negatives.
                    max_iters = max(n_drugs * 2, 20)
                    iters = 0
                    while bad.any() and iters < max_iters:
                        ap_nj[bad] = torch.randint(0, n_drugs,
                                                   (int(bad.sum()),), device=device)
                        bad = ap_ni == ap_nj
                        iters += 1
                    if bad.any():
                        ap_nj[bad] = (ap_ni[bad] + 1) % n_drugs

                    h_ni = self.model.scorer.precompute_drug_features(
                        drug_emb[ap_ni.flatten()]).reshape(ap_B, aux_pair_neg_k, -1)
                    h_nj = self.model.scorer.precompute_drug_features(
                        drug_emb[ap_nj.flatten()]).reshape(ap_B, aux_pair_neg_k, -1)
                    neg_aux = self.model.aux_pair_head(h_ni, h_nj)  # [ap_B, aux_pair_neg_k]

                    loss_aux = bpr_loss(pos_aux, neg_aux)
                    loss     = loss + aux_pair_lambda * loss_aux
                    ep_aux  += float(loss_aux.item())

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                ep_loss += loss.item()
                ep_main += float(loss_main.item())
                n_steps += 1

            sch.step()

            # Reset pad warnings (data.py infrastructure)
            try:
                pad_n = train_loader.dataset.reset_pad_warnings()
                if pad_n:
                    logger.warning(f"  [v8] Epoch {epoch+1}: {pad_n} pad samples")
            except AttributeError:
                pass

            if (epoch + 1) % 1 == 0 or epoch == 0:
                metrics = self.evaluate_full_space(
                    val_loader, train_pos_set=train_pos_set)
                mrr = metrics['MRR']
                test_metrics = self.evaluate_full_space(test_loader)
                test_mrr = test_metrics['MRR']
                aux_parts = []
                if aux_on:
                    aux_parts.append(f"pair={ep_aux/max(n_steps,1):.4f}")
                if aux_dd_on:
                    aux_parts.append(f"dd={ep_aux_dd/max(n_steps,1):.4f}")
                if listwise_on:
                    aux_parts.append(f"lw={ep_listwise/max(n_steps,1):.4f}")
                aux_str = " | aux_" + ",".join(aux_parts) if aux_parts else ""
                logger.info(
                    f"  [v8] Ep {epoch+1:3d}/{n_epochs} | "
                    f"loss={ep_loss/n_steps:.4f} (main={ep_main/n_steps:.4f}{aux_str}) | "
                    f"Val MRR={mrr:.4f} H@10={metrics['Hits@10']:.4f} "
                    f"H@50={metrics['Hits@50']:.4f} MedR={metrics['MedianRank']:.0f} | "
                    f"Test MRR={test_mrr:.4f} | "
                    f"Gap={mrr-test_mrr:+.4f} | "
                    f"LR={opt.param_groups[-1]['lr']:.2e} | dt={time.time()-t0:.1f}s")

                if mrr > best_mrr:
                    best_mrr = mrr
                    torch.save(self.model.state_dict(), ckpt)
                    no_improve = 0
                    logger.info(f"  → new best Val MRR={best_mrr:.4f} saved to {ckpt}")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info(f"  [v8] Early stop at epoch {epoch+1}.")
                        break

        if os.path.exists(ckpt):
            self.model.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True))
            logger.info(f"[v8] Restored best (Val MRR={best_mrr:.4f}).")

        logger.info("[v8] Phase B complete.")

    # ──────────────────────────────────────────────────────────────────
    # Full-space evaluation
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_full_space(self,
                            loader,
                            ks=(10, 20, 50, 100),
                            train_pos_set: Optional[Dict[int, set]] = None):
        """
        For each disease present in `loader`:
          1. Score every C(N_drug, 2) pair in one GPU sweep.
          2. Optionally mask known train positives to -inf.
          3. Compute rank of each ground-truth val/test pair.

        Reports disease-macro ranking metrics (MRR, Hits@K, MeanRank,
        MedianRank averaged across diseases) plus AUC and AP (AUPR).

        Fixes applied:
          [fix-7] AUC: n_neg_d counts only unmasked (non -inf) negatives.
          [fix-8] AUPR: standard Average Precision over ranked list.
          [fix-9] Dead stage_k branch removed; always does full ranking.
          [fix-10] compute_ranking_metrics handles empty list with zeros.
          [fix-11] total_pairs used consistently; n_total alias removed.
          [fix-12] Log prefix standardised to [v8].
        """
        self.model.eval()
        device  = self.device
        n_drugs = self.kg.n_drug
        total_pairs = n_drugs * (n_drugs - 1) // 2

        h_drug_pre = self.model.precompute_drug_pair_features()    # [N, d_v]
        pair_i, pair_j = self._build_pair_index()

        # Group ground truths by disease
        disease_to_pos: Dict[int, set] = defaultdict(set)
        all_records: List[Tuple[int, int, int]] = []
        seen = set()
        for batch in loader:
            for row in batch['pos']:
                d, i, j = int(row[0]), int(row[1]), int(row[2])
                ci, cj = min(i, j), max(i, j)
                disease_to_pos[d].add((ci, cj))
                key = (d, ci, cj)
                if key not in seen:
                    seen.add(key)
                    all_records.append(key)

        logger.info(f"  [v8] Eval: {len(all_records)} records | "
                    f"{len(disease_to_pos)} diseases | "
                    f"{total_pairs:,} pairs")

        ranks: Dict[Tuple[int, int, int], int] = {}
        disease_metric_list: List[Dict[str, float]] = []
        auc_list, aupr_list = [], []

        for dis_idx, pos_set in disease_to_pos.items():
            scores = self.model.score_full_space_for_disease(
                dis_idx, pair_i, pair_j, h_drug_pre, chunk=200_000)

            # Mask train positives to -inf
            if train_pos_set is not None:
                tp = train_pos_set.get(dis_idx, set()) - pos_set
                if tp:
                    tp_t  = torch.tensor(list(tp), dtype=torch.long, device=device)
                    flat  = _flat_pair_index(tp_t[:, 0], tp_t[:, 1], n_drugs)
                    scores[flat] = float('-inf')

            pos_list = list(pos_set)
            pos_t    = torch.tensor(pos_list, dtype=torch.long, device=device)
            pos_flat = _flat_pair_index(pos_t[:, 0], pos_t[:, 1], n_drugs)
            pos_scores = scores[pos_flat]

            # Filtered rank: count items scored strictly above each positive,
            # subtract other eval positives for the same disease that rank higher.
            raw_above = (scores.unsqueeze(1) > pos_scores.unsqueeze(0)).sum(0)
            eval_pos_above = (
                pos_scores.unsqueeze(1) > pos_scores.unsqueeze(0)
            ).sum(0)
            ranks_gpu = raw_above - eval_pos_above + 1
            for k_, pair in enumerate(pos_list):
                ranks[(dis_idx, pair[0], pair[1])] = int(ranks_gpu[k_].item())

            disease_metric_list.append(
                compute_ranking_metrics(ranks_gpu.detach().cpu().numpy(), ks)
            )

            # ── AUC [fix-7] ───────────────────────────────────────────
            # Only count unmasked items as negatives (exclude -inf train pos).
            n_pos_d = pos_flat.numel()
            valid_mask = ~torch.isinf(scores)
            n_valid = int(valid_mask.sum().item())
            n_neg_d = n_valid - n_pos_d  # [fix-7] was: total_pairs - n_pos_d

            if n_pos_d > 0 and n_neg_d > 0:
                valid_scores = scores[valid_mask]
                sorted_valid, _ = torch.sort(valid_scores)
                rank_lo = torch.searchsorted(sorted_valid, pos_scores, right=False).float()
                rank_hi = torch.searchsorted(sorted_valid, pos_scores, right=True).float()
                ranks_asc = (rank_lo + rank_hi) / 2.0 + 1
                auc_t = ((ranks_asc.sum() - n_pos_d * (n_pos_d + 1) / 2) /
                         (n_pos_d * n_neg_d)).clamp(0.0, 1.0)
                auc_list.append(float(auc_t.item()))

                # ── Average Precision (AUPR) [fix-8] ──────────────────
                # Standard AP: Σ_k precision@k * indicator(k is positive),
                # normalised by n_pos_d. Masked items sort to the end and
                # do not affect AP since they never appear above positives.
                sorted_desc_idx = torch.argsort(scores, descending=True)
                is_pos = torch.zeros(scores.numel(), dtype=torch.float32, device=device)
                is_pos[pos_flat] = 1.0
                is_pos_sorted = is_pos[sorted_desc_idx][:n_valid]  # trim masked tail
                cum_pos = torch.cumsum(is_pos_sorted, dim=0)
                ranks_k = torch.arange(1, n_valid + 1, dtype=torch.float32, device=device)
                precision_at_k = cum_pos / ranks_k
                ap = (precision_at_k * is_pos_sorted).sum() / max(n_pos_d, 1)
                aupr_list.append(float(ap.item()))

        if disease_metric_list:
            metric_names = disease_metric_list[0].keys()
            out = {
                name: float(np.mean([m[name] for m in disease_metric_list]))
                for name in metric_names
            }
        else:
            out = compute_ranking_metrics([], ks)   # [fix-10] returns zeros
        out['AUC']  = float(np.mean(auc_list))  if auc_list  else float('nan')
        out['AUPR'] = float(np.mean(aupr_list)) if aupr_list else float('nan')
        out['DiseaseCount'] = float(len(disease_metric_list))
        logger.info("  [v8] " +
                    " | ".join(f"{k}: {v:.4f}" for k, v in sorted(out.items())))
        return out

    # ──────────────────────────────────────────────────────────────────
    # Inference: rank for one disease
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def rank_all_pairs_for_disease(self, disease_id, disease2idx, drug_ids,
                                   top_k=100):
        if len(drug_ids) != self.kg.n_drug:
            raise ValueError(
                f"len(drug_ids)={len(drug_ids)} != kg.n_drug={self.kg.n_drug}")
        self.model.eval()
        h_drug_pre = self.model.precompute_drug_pair_features()
        pair_i, pair_j = self._build_pair_index()
        scores = self.model.score_full_space_for_disease(
            disease2idx[disease_id], pair_i, pair_j, h_drug_pre)
        topk_v, topk_idx = torch.topk(scores, k=min(top_k, scores.numel()))
        i_l = pair_i.long()[topk_idx]
        j_l = pair_j.long()[topk_idx]
        return [
            (drug_ids[i_l[k].item()], drug_ids[j_l[k].item()],
             float(topk_v[k].item()))
            for k in range(topk_v.numel())
        ]
