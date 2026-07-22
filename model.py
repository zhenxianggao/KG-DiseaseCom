from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — KGE encoders
# ─────────────────────────────────────────────────────────────────────────────

class DistMultKGE(nn.Module):
    """
    Bag-of-relations DistMult.

    Node ordering matches data.KGGraph: [drugs..., diseases..., genes...].
    The relation table covers `n_relations_total` (forward + inverse).
    """

    uses_inverse_edges = False

    def __init__(
        self,
        num_nodes:     int,
        num_relations: int,
        d_emb:         int = 64,
        init_scale:    float = 6.0,
    ):
        super().__init__()
        self.num_nodes     = num_nodes
        self.num_relations = num_relations
        self.d_emb         = d_emb

        self.node_emb = nn.Embedding(num_nodes, d_emb)
        self.rel_emb  = nn.Embedding(num_relations, d_emb)

        # RotatE-style uniform init on a controlled range so dot products
        # don't explode at the start of training. init_scale=6 follows the
        # convention of (k_emb / d) sqrt range for embeddings.
        bound = init_scale / math.sqrt(d_emb)
        nn.init.uniform_(self.node_emb.weight, -bound, bound)
        nn.init.uniform_(self.rel_emb.weight,  -bound, bound)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """h, r, t : [B] long → score [B] (dot product of e_h * w_r * e_t)."""
        return (self.node_emb(h) * self.rel_emb(r) * self.node_emb(t)).sum(-1)

    def score_pairs(self, e_h: torch.Tensor, w_r: torch.Tensor,
                    e_t: torch.Tensor) -> torch.Tensor:
        return (e_h * w_r * e_t).sum(-1)

    def get_drug_disease_embeddings(self, n_drug: int, n_disease: int):
        drug_emb    = self.node_emb.weight[:n_drug]
        disease_emb = self.node_emb.weight[n_drug : n_drug + n_disease]
        return drug_emb, disease_emb


class CompGCNKGE(nn.Module):
    """
    Lightweight CompGCN + RAAG encoder with a DistMult decoder.

    This implementation intentionally avoids external PyG dependencies. It
    performs relation-aware gated message passing using `index_add_` over KG edges:

        m_{u->v,r} = W_msg(e_u * r)
        alpha      = sigmoid(W_att([e_v, m_{u->v,r}]))
        e'_v       = LN(GELU(W_self(e_v) + mean(alpha * m_{u->v,r})))

    The final node embeddings are scored with a DistMult-style decoder. The
    public methods mirror DistMultKGE so the V1 trainer/scorer can stay small.

    [fix-1] _last_rel_for_score removed. _relation_for_score() now always
    computes rel_proj[-1](rel_emb.weight) on demand, which is device-safe,
    state_dict-safe, and always consistent with the current parameter values.
    """

    uses_inverse_edges = True

    def __init__(
        self,
        num_nodes:     int,
        num_relations: int,
        d_emb:         int = 64,
        init_scale:    float = 6.0,
        num_layers:    int = 1,
        dropout:       float = 0.1,
        use_raag:      bool = True,
        raag_chunk:    int = 50_000,
    ):
        super().__init__()
        self.num_nodes     = num_nodes
        self.num_relations = num_relations
        self.d_emb         = d_emb
        self.num_layers    = num_layers
        self.use_raag      = use_raag
        self.raag_chunk    = max(1, int(raag_chunk))

        self.node_emb = nn.Embedding(num_nodes, d_emb)
        self.rel_emb  = nn.Embedding(num_relations, d_emb)

        self.self_proj = nn.ModuleList([nn.Linear(d_emb, d_emb) for _ in range(num_layers)])
        self.msg_proj  = nn.ModuleList([nn.Linear(d_emb, d_emb, bias=False) for _ in range(num_layers)])
        self.rel_proj  = nn.ModuleList([nn.Linear(d_emb, d_emb, bias=False) for _ in range(num_layers)])
        self.raag_rel_proj = nn.ModuleList([nn.Linear(d_emb, d_emb, bias=False) for _ in range(num_layers)])
        self.raag_att = nn.ModuleList([nn.Linear(2 * d_emb, 1) for _ in range(num_layers)])
        self.norms     = nn.ModuleList([nn.LayerNorm(d_emb) for _ in range(num_layers)])
        self.dropout   = nn.Dropout(dropout)

        self.register_buffer('edge_index_buf', torch.empty(2, 0, dtype=torch.long), persistent=False)
        self.register_buffer('edge_type_buf',  torch.empty(0, dtype=torch.long), persistent=False)

        bound = init_scale / math.sqrt(d_emb)
        nn.init.uniform_(self.node_emb.weight, -bound, bound)
        nn.init.uniform_(self.rel_emb.weight,  -bound, bound)

    def set_graph(self, edge_index: torch.Tensor, edge_type: torch.Tensor):
        self.edge_index_buf = edge_index.long()
        self.edge_type_buf  = edge_type.long()

    def _aggregate_messages(self,
                            x: torch.Tensor,
                            src: torch.Tensor,
                            dst: torch.Tensor,
                            rel_type: torch.Tensor,
                            layer: int) -> torch.Tensor:
        if not self.use_raag:
            rel = self.rel_emb(rel_type)
            msg = self.msg_proj[layer](x[src] * rel)
            agg = x.new_zeros(x.shape)
            agg.index_add_(0, dst, msg)
            deg = x.new_zeros(x.size(0), 1)
            deg.index_add_(0, dst, torch.ones(dst.size(0), 1, device=x.device, dtype=x.dtype))
            return agg / deg.clamp_min(1.0)

        agg = x.new_zeros(x.shape)
        deg = x.new_zeros(x.size(0), 1)
        ones_cache = None

        for s in range(0, src.size(0), self.raag_chunk):
            e = min(s + self.raag_chunk, src.size(0))
            src_c = src[s:e]
            dst_c = dst[s:e]
            rel_c = rel_type[s:e]

            rel = self.raag_rel_proj[layer](self.rel_emb(rel_c))
            raw_msg = x[src_c] * rel
            gate_in = torch.cat([x[dst_c], raw_msg], dim=-1)
            alpha = torch.sigmoid(self.raag_att[layer](F.leaky_relu(gate_in, 0.2)))
            msg = self.msg_proj[layer](raw_msg) * alpha
            msg = self.dropout(msg)

            agg.index_add_(0, dst_c, msg)
            if ones_cache is None or ones_cache.size(0) != dst_c.size(0):
                ones_cache = torch.ones(dst_c.size(0), 1, device=x.device, dtype=x.dtype)
            deg.index_add_(0, dst_c, ones_cache)

        return agg / deg.clamp_min(1.0)

    def encode_nodes(self) -> torch.Tensor:
        x = self.node_emb.weight
        if self.edge_index_buf.numel() == 0 or self.num_layers <= 0:
            return x

        src, dst = self.edge_index_buf[0], self.edge_index_buf[1]
        rel_type = self.edge_type_buf

        for layer in range(self.num_layers):
            agg = self._aggregate_messages(x, src, dst, rel_type, layer)
            self_part = self.self_proj[layer](x)
            x_next = self.norms[layer](F.gelu(self_part + agg))
            x = self.dropout(x_next) + x

        return x

    def _relation_for_score(self) -> torch.Tensor:
        # [fix-1] Always recompute from the last layer's rel_proj so this is
        # device-safe and consistent after checkpoint loads. Previously stored
        # as a plain Python attribute (_last_rel_for_score) which was stale
        # after .to(device) and not included in state_dict.
        if self.num_layers > 0:
            return self.rel_proj[-1](self.rel_emb.weight)
        return self.rel_emb.weight

    def score_with_node_embeddings(
        self,
        node_emb: torch.Tensor,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        rel = self._relation_for_score()
        return (node_emb[h] * rel[r] * node_emb[t]).sum(-1)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        node_emb = self.encode_nodes()
        return self.score_with_node_embeddings(node_emb, h, r, t)

    def score_pairs(self, e_h: torch.Tensor, w_r: torch.Tensor,
                    e_t: torch.Tensor) -> torch.Tensor:
        return (e_h * w_r * e_t).sum(-1)

    def get_drug_disease_embeddings(self, n_drug: int, n_disease: int):
        node_emb = self.encode_nodes()
        drug_emb    = node_emb[:n_drug]
        disease_emb = node_emb[n_drug : n_drug + n_disease]
        return drug_emb, disease_emb


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Disease-conditioned compact pair scorer
# ─────────────────────────────────────────────────────────────────────────────

class CompactPairScorer(nn.Module):

    def __init__(self, d_emb: int = 64, d_v: int = 64,
                 dropout: float = 0.3, hidden_mult: int = 1):
        super().__init__()
        self.d_v = d_v

        # Drug-side projector (run once over all drugs each forward pass)
        h = max(d_v, d_emb) * hidden_mult
        self.pre_proj = nn.Sequential(
            nn.Linear(d_emb, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, d_v),
        )

        # Disease-side projector → 3 channel weight vectors + scalar bias
        self.dis_proj = nn.Sequential(
            nn.Linear(d_emb, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, 3 * d_v + 1),   # q | a | b | c (scalar)
        )

        # Optional learnable temperature for softer logits — helps with BPR
        self.log_tau = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def precompute_drug_features(self, drug_emb: torch.Tensor) -> torch.Tensor:
        """drug_emb : [N, d_emb] → [N, d_v]."""
        return self.pre_proj(drug_emb)

    def disease_weights(self, disease_emb: torch.Tensor):
        """
        disease_emb : [B, d_emb] (or [d_emb] for a single disease).
        Returns: q,a,b ∈ [B, d_v] ; c ∈ [B, 1]
        """
        if disease_emb.dim() == 1:
            disease_emb = disease_emb.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        out = self.dis_proj(disease_emb)
        q, a, b = out[:, : self.d_v], \
                  out[:, self.d_v : 2*self.d_v], \
                  out[:, 2*self.d_v : 3*self.d_v]
        c = out[:, 3*self.d_v: 3*self.d_v + 1]
        if squeeze:
            q, a, b, c = q.squeeze(0), a.squeeze(0), b.squeeze(0), c.squeeze(0)
        return q, a, b, c

    # ------------------------------------------------------------------
    # Per-triple scoring (training)
    # ------------------------------------------------------------------

    def score_triplets(self,
                        e_disease: torch.Tensor,
                        e_drug_i:  torch.Tensor,
                        e_drug_j:  torch.Tensor,
                        with_temperature: bool = True) -> torch.Tensor:
        """
        e_disease : [B, d_emb]      — already the *raw* disease embedding
        e_drug_i  : [B, d_emb]
        e_drug_j  : [B, d_emb]
        Returns   : [B] scores

        with_temperature=True  → divide by log_tau.exp() (default; used for
                                  BPR / InfoNCE losses).
        with_temperature=False → return raw logits (used for the listwise
                                  sampled-softmax loss so that listwise_temp
                                  is the sole temperature, avoiding the
                                  double-temperature bug present in the
                                  original v8_listwise_evl).
        """
        h_i = self.pre_proj(e_drug_i)         # [B, d_v]
        h_j = self.pre_proj(e_drug_j)         # [B, d_v]
        q, a, b, c = self.disease_weights(e_disease)  # [B, d_v] x3, [B,1]

        had = h_i * h_j                        # [B, d_v]
        s   = h_i + h_j
        df  = (h_i - h_j).abs()

        score = (q * had).sum(-1) + (a * s).sum(-1) + (b * df).sum(-1) + c.squeeze(-1)
        if with_temperature:
            return score / self.log_tau.exp()
        return score

    # ------------------------------------------------------------------
    # Vectorised full-space ranking
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_all_pairs_for_disease(
        self,
        disease_emb_one: torch.Tensor,    # [d_emb]
        h_drug_pre:      torch.Tensor,    # [N, d_v] precomputed
        pair_i:          torch.Tensor,    # [P] long
        pair_j:          torch.Tensor,    # [P] long
        chunk:           int = 200_000,
    ) -> torch.Tensor:
        """
        Returns raw scores (no temperature scaling) over the supplied pairs.
        Temperature does not affect ranking order, so it is omitted here.
        Computes in chunks of `chunk` pairs to bound peak memory at
        chunk * d_v floats (~50 MB for chunk=200k, d_v=64).
        """
        q, a, b, c = self.disease_weights(disease_emb_one)
        c          = c.view(())

        P = pair_i.size(0)
        out = h_drug_pre.new_empty(P)
        for s in range(0, P, chunk):
            e   = min(s + chunk, P)
            i   = pair_i[s:e].long()
            j   = pair_j[s:e].long()
            h_i = h_drug_pre[i]
            h_j = h_drug_pre[j]
            had = h_i * h_j
            sm  = h_i + h_j
            df  = (h_i - h_j).abs()
            sc  = (q * had).sum(-1) + (a * sm).sum(-1) + (b * df).sum(-1) + c
            out[s:e] = sc
        return out


class DrugPairAuxHead(nn.Module):

    def __init__(self, d_v: int):
        super().__init__()
        self.w_had = nn.Parameter(torch.empty(d_v))
        self.w_sum = nn.Parameter(torch.empty(d_v))
        self.bias  = nn.Parameter(torch.zeros(1))
        bound = 1.0 / max(d_v, 1) ** 0.5
        nn.init.normal_(self.w_had, std=bound)
        nn.init.normal_(self.w_sum, std=bound)

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """h_i, h_j : [..., d_v] → score [...]."""
        return ((self.w_had * (h_i * h_j)).sum(-1)
              + (self.w_sum * (h_i + h_j)).sum(-1)
              + self.bias)


class DiseaseDrugAuxHead(nn.Module):
    """
    Small first-order disease-drug relevance head.

    It complements the pair scorer with a reusable prior:

        s(d, g) = u(d) · h_g + b(d)

    where h_g is the shared drug feature from CompactPairScorer.pre_proj().
    The head is deliberately tiny so V6 gains first-order disease relevance
    without returning to the large V3 S2 MLP.
    """

    def __init__(self, d_emb: int, d_v: int, dropout: float = 0.3,
                 hidden_mult: int = 1):
        super().__init__()
        h = max(d_v, d_emb) * hidden_mult
        self.dis_proj = nn.Sequential(
            nn.Linear(d_emb, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, d_v + 1),
        )

    def disease_weights(self, disease_emb: torch.Tensor):
        squeeze = disease_emb.dim() == 1
        if squeeze:
            disease_emb = disease_emb.unsqueeze(0)
        out = self.dis_proj(disease_emb)
        u = out[:, :-1]
        b = out[:, -1]
        if squeeze:
            u, b = u.squeeze(0), b.squeeze(0)
        return u, b

    def score(self, disease_emb: torch.Tensor, h_drug: torch.Tensor) -> torch.Tensor:
        u, b = self.disease_weights(disease_emb)
        return (u * h_drug).sum(-1) + b

    @torch.no_grad()
    def score_all_drugs(self, disease_emb_one: torch.Tensor,
                        h_drug_pre: torch.Tensor) -> torch.Tensor:
        u, b = self.disease_weights(disease_emb_one)
        return h_drug_pre @ u + b


class DrugCombinationModel(nn.Module):
    """
    Two-layer drug-combination ranker.

    Phase A (handled by trainer): train DistMultKGE on KG triplets.
    Phase B (handled by trainer): train CompactPairScorer on (disease, drug_i, drug_j),
                                  with a low LR or freeze on the embedding tables.

    Public API mirrors v3 where useful so trainer code is small and predictable.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.n_drug    = config['n_drugs']
        self.n_disease = config['n_diseases']
        self.n_gene    = config['n_genes']
        self.n_hpo     = config.get('n_hpo', 0)
        self.n_mp      = config.get('n_mp', 0)
        self.n_other   = config.get('n_other', 0)
        self.n_total   = (
            self.n_drug + self.n_disease + self.n_gene
            + self.n_hpo + self.n_mp + self.n_other
        )

        d_emb   = config.get('d_emb', 64)
        d_v     = config.get('d_v',    d_emb)
        dropout = config.get('dropout', 0.3)
        hidden_mult = config.get('scorer_hidden_mult', 1)

        kge_model = str(config.get('kge_model', 'compgcn')).lower()
        if kge_model == 'distmult':
            self.kge = DistMultKGE(
                num_nodes     = self.n_total,
                num_relations = config['n_relations'],
                d_emb         = d_emb,
            )
        elif kge_model == 'compgcn':
            self.kge = CompGCNKGE(
                num_nodes     = self.n_total,
                num_relations = config['n_relations'],
                d_emb         = d_emb,
                num_layers    = int(config.get('compgcn_layers', 1)),
                dropout       = float(config.get('compgcn_dropout', 0.1)),
                use_raag      = bool(config.get('use_raag', True)),
                raag_chunk    = int(config.get('raag_chunk', 50_000)),
            )
        else:
            raise ValueError(f"Unsupported kge_model: {kge_model}")
        self.kge_model = kge_model

        self.scorer = CompactPairScorer(
            d_emb = d_emb, d_v = d_v,
            dropout = dropout, hidden_mult = hidden_mult,
        )

        self.aux_pair_head = DrugPairAuxHead(d_v)
        self.aux_disease_drug_head = DiseaseDrugAuxHead(
            d_emb=d_emb,
            d_v=d_v,
            dropout=dropout,
            hidden_mult=hidden_mult,
        )

        self.d_emb = d_emb
        self.d_v   = d_v
        self.aux_score_weight = float(config.get('aux_score_weight', 0.0))
        self.aux_disease_drug_score_weight = float(
            config.get('aux_disease_drug_score_weight', 0.0))


    @torch.no_grad()
    def encode_all(self, *_args, **_kwargs):
        """
        Returns (drug_emb, disease_emb), shapes [n_drug, d_emb], [n_disease, d_emb].
        For CompGCN this runs relation-aware KG message passing before slicing
        drug/disease rows. The signature is kept for API parity with v3/V1.

        *_args is accepted but ignored so trainer code that passes
        (drug_idx, disease_idx, gene_idx, node_type_mask, edge_index, edge_type)
        can keep using the same call site.
        """
        return self.kge.get_drug_disease_embeddings(self.n_drug, self.n_disease)

    def encode_all_grad(self):
        """
        Same as encode_all() but keeps grad — used in Phase B if you want to
        slowly fine-tune the embedding tables.
        """
        return self.kge.get_drug_disease_embeddings(self.n_drug, self.n_disease)

    def score_triplets(self,
                       e_disease: torch.Tensor,
                       e_drug_i:  torch.Tensor,
                       e_drug_j:  torch.Tensor,
                       with_temperature: bool = True):
        """
        Scores a batch of (disease, drug_i, drug_j) triplets.

        [fix-3] h_i / h_j are computed once even when both aux heads are
        active (the original code recomputed them in each branch).

        [fix-2] Temperature (log_tau) is applied here at the model level,
        not inside CompactPairScorer.score_triplets, so callers that need
        raw logits (e.g., the listwise loss) can pass with_temperature=False.
        """
        # Compute raw score from the scorer (no temperature applied inside)
        score = self.scorer.score_triplets(
            e_disease, e_drug_i, e_drug_j, with_temperature=False)

        # [fix-3] Compute h_i/h_j only once if any aux head is active.
        if self.aux_score_weight or self.aux_disease_drug_score_weight:
            h_i = self.scorer.precompute_drug_features(e_drug_i)
            h_j = self.scorer.precompute_drug_features(e_drug_j)
            if self.aux_score_weight:
                score = score + self.aux_score_weight * self.aux_pair_head(h_i, h_j)
            if self.aux_disease_drug_score_weight:
                dd = (self.aux_disease_drug_head.score(e_disease, h_i)
                      + self.aux_disease_drug_head.score(e_disease, h_j))
                score = score + self.aux_disease_drug_score_weight * dd

        if with_temperature:
            score = score / self.scorer.log_tau.exp()
        return {'final': score}

    def kge_score(self, h, r, t):
        return self.kge.score(h, r, t)

    @torch.no_grad()
    def precompute_drug_pair_features(self, drug_emb: torch.Tensor | None = None):
        """
        Returns [N_drug, d_v] tensor of pre-projected drug features.

        Accepts an optional pre-computed drug_emb to avoid a redundant
        encode_nodes() call when the caller already holds the embeddings.
        """
        if drug_emb is None:
            drug_emb, _ = self.kge.get_drug_disease_embeddings(
                self.n_drug, self.n_disease)
        return self.scorer.precompute_drug_features(drug_emb)

    @torch.no_grad()
    def score_full_space_for_disease(
        self,
        disease_idx: int,
        pair_i:      torch.Tensor,
        pair_j:      torch.Tensor,
        h_drug_pre:  torch.Tensor | None = None,
        chunk:       int = 200_000,
    ):
        """
        Convenience wrapper for inference / evaluation.
        If h_drug_pre is None it is computed; pass it in to share across
        many diseases and avoid recomputation.

        [fix-4] encode_nodes() is called at most once per invocation.
        Previously, get_drug_disease_embeddings() was called to obtain
        disease_emb, and then precompute_drug_pair_features() called it
        again internally — two full GCN passes for CompGCN. Now we call
        encode once and slice both drug and disease embeddings from the
        same result.
        """
        drug_emb, disease_emb = self.kge.get_drug_disease_embeddings(
            self.n_drug, self.n_disease)
        if h_drug_pre is None:
            # [fix-4] reuse already-computed drug_emb
            h_drug_pre = self.scorer.precompute_drug_features(drug_emb)

        scores = self.scorer.score_all_pairs_for_disease(
            disease_emb[disease_idx], h_drug_pre, pair_i, pair_j, chunk=chunk)

        if self.aux_score_weight:
            aux = scores.new_empty(scores.shape)
            for s in range(0, pair_i.size(0), chunk):
                e = min(s + chunk, pair_i.size(0))
                i = pair_i[s:e].long()
                j = pair_j[s:e].long()
                aux[s:e] = self.aux_pair_head(h_drug_pre[i], h_drug_pre[j])
            scores = scores + self.aux_score_weight * aux

        if self.aux_disease_drug_score_weight:
            dd_all = self.aux_disease_drug_head.score_all_drugs(
                disease_emb[disease_idx], h_drug_pre)
            scores = scores + self.aux_disease_drug_score_weight * (
                dd_all[pair_i.long()] + dd_all[pair_j.long()])
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Parameter count helper
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """Return a breakdown of trainable parameters by submodule."""
    counts = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters() if p.requires_grad)
        counts[name] = n
    counts['__total__'] = sum(counts.values())
    return counts
