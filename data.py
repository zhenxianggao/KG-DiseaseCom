import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CSV loaders
# ─────────────────────────────────────────────────────────────────────────────

def read_id_csv(path) -> List[str]:
    path = Path(path)
    if not path.exists():
        logger.warning(f"Optional ID file not found: {path}")
        return []
    df = pd.read_csv(path, dtype=str, header=0)
    col = df.iloc[:, 0].dropna().str.strip()
    ids = list(dict.fromkeys(v for v in col if v))
    return ids


def load_kg(kg_path: str) -> pd.DataFrame:

    df = pd.read_csv(kg_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {'node1', 'relation', 'node2'}.issubset(df.columns), \
        "KG CSV must have columns: node1, relation, node2"
    df['node1']    = df['node1'].str.strip()
    df['node2']    = df['node2'].str.strip()
    df['relation'] = df['relation'].str.strip()
    before = len(df)
    df = df.dropna(subset=['node1', 'relation', 'node2'])
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with missing values in KG")
    return df


def load_drug_pairs(path: str) -> pd.DataFrame:
    """
    Load drug-drug pairs for Phase 2 training.
    Expected columns: drug1_id, drug2_id
    All IDs are forced to str.
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ('drug1_id', 'drug2_id'):
        df[col] = df[col].str.strip()
    df = df.dropna(subset=['drug1_id', 'drug2_id'])
    before = len(df)
    df = df[df['drug1_id'] != df['drug2_id']]
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} self-pair rows in drug pair file")
    return df


def load_pairs(path: str) -> pd.DataFrame:
    """
    Load disease-drug-drug pairs.
    Expected columns: disease_id, drug1_id, drug2_id
    All IDs are forced to str.
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ('disease_id', 'drug1_id', 'drug2_id'):
        df[col] = df[col].str.strip()
    df = df.dropna(subset=['disease_id', 'drug1_id', 'drug2_id'])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# KG graph
# ─────────────────────────────────────────────────────────────────────────────

class KGGraph:
    """
    Processes KG into PyG-compatible format.

    Node ordering: [drugs..., diseases..., genes..., hpo..., mp..., other...]

    node_type_mask convention:
        0 = drug
        1 = disease
        2 = gene
        3 = HPO
        4 = MP
        5 = other KG node
    """

    def __init__(
        self,
        kg_df:       pd.DataFrame,
        drug_ids:    List[str],
        disease_ids: List[str],
        gene_ids:    Optional[List[str]] = None,
        hpo_ids:     Optional[List[str]] = None,
        mp_ids:      Optional[List[str]] = None,
    ):
        self.drug_ids    = drug_ids
        self.disease_ids = disease_ids
        self.gene_ids    = list(dict.fromkeys(gene_ids or []))
        self.hpo_ids     = list(dict.fromkeys(hpo_ids or []))
        self.mp_ids      = list(dict.fromkeys(mp_ids or []))

        # ── Build global node index ───────────────────────────────────────
        self.node2idx: Dict[str, int] = {}

        def add_nodes(nodes: List[str]) -> List[str]:
            added = []
            for nid in nodes:
                if nid and nid not in self.node2idx:
                    self.node2idx[nid] = len(self.node2idx)
                    added.append(nid)
            return added

        drug_ids = add_nodes(drug_ids)
        disease_ids = add_nodes(disease_ids)
        gene_nodes = add_nodes(self.gene_ids)
        hpo_nodes = add_nodes(self.hpo_ids)
        mp_nodes = add_nodes(self.mp_ids)

        all_kg_nodes = set(kg_df['node1'].tolist()) | set(kg_df['node2'].tolist())
        other_nodes = sorted(n for n in all_kg_nodes if n not in self.node2idx)
        other_nodes = add_nodes(other_nodes)

        self.drug_ids = drug_ids
        self.disease_ids = disease_ids
        self.gene_ids = gene_nodes
        self.hpo_ids = hpo_nodes
        self.mp_ids = mp_nodes
        self.other_ids = other_nodes

        self.n_drug    = len(drug_ids)
        self.n_disease = len(disease_ids)
        self.n_gene    = len(gene_nodes)
        self.n_hpo     = len(hpo_nodes)
        self.n_mp      = len(mp_nodes)
        self.n_other   = len(other_nodes)
        self.n_total   = len(self.node2idx)

        # Keep offsets explicit; drug/disease local indices are used by eval.
        self.drug_offset = 0
        self.disease_offset = self.n_drug
        self.gene_offset = self.disease_offset + self.n_disease
        self.hpo_offset = self.gene_offset + self.n_gene
        self.mp_offset = self.hpo_offset + self.n_hpo
        self.other_offset = self.mp_offset + self.n_mp

        missing_typed = all_kg_nodes - (
            set(drug_ids) | set(disease_ids) | set(gene_nodes) |
            set(hpo_nodes) | set(mp_nodes)
        )
        if missing_typed:
            logger.warning(
                f"{len(missing_typed)} KG nodes were not present in drug/disease/"
                f"gene/hpo/mp id lists; assigned to node_type=other")

        # ── Relation index ────────────────────────────────────────────────
        relations        = kg_df['relation'].unique().tolist()
        self.rel2idx     = {r: i for i, r in enumerate(relations)}
        self.n_relations = len(relations)
        logger.info(f"KG relations: {self.rel2idx}")

        # ── Build edge_index / edge_type (vectorised) ─────────────────────
        # [data-2] Replace iterrows() with vectorised map; ~100× faster.
        node1_idx = kg_df['node1'].map(self.node2idx)
        node2_idx = kg_df['node2'].map(self.node2idx)
        rel_idx   = kg_df['relation'].map(self.rel2idx)

        # Drop edges whose nodes are not in the vocab
        valid   = node1_idx.notna() & node2_idx.notna()
        skipped = (~valid).sum()
        if skipped:
            logger.warning(f"Skipped {skipped} KG edges (nodes not in vocab)")

        src_arr   = node1_idx[valid].astype(int).values
        dst_arr   = node2_idx[valid].astype(int).values
        etype_arr = rel_idx[valid].astype(int).values

        src_t   = torch.tensor(src_arr,   dtype=torch.long)
        dst_t   = torch.tensor(dst_arr,   dtype=torch.long)
        etype_t = torch.tensor(etype_arr, dtype=torch.long)

        # Undirected: forward + reverse edges
        rev_etype              = etype_t + self.n_relations
        self.n_relations_total = self.n_relations * 2

        self.edge_index = torch.stack([
            torch.cat([src_t, dst_t]),
            torch.cat([dst_t, src_t])
        ], dim=0)                                      # [2, 2E]
        self.edge_type = torch.cat([etype_t, rev_etype])  # [2E]

        # ── node_type_mask ────────────────────────────────────────────────
        self.node_type_mask = torch.full(
            (self.n_total,), fill_value=5, dtype=torch.long
        )
        self.node_type_mask[:self.n_drug] = 0
        self.node_type_mask[
            self.disease_offset : self.disease_offset + self.n_disease
        ] = 1
        self.node_type_mask[
            self.gene_offset : self.gene_offset + self.n_gene
        ] = 2
        self.node_type_mask[
            self.hpo_offset : self.hpo_offset + self.n_hpo
        ] = 3
        self.node_type_mask[
            self.mp_offset : self.mp_offset + self.n_mp
        ] = 4

        # Local indices for compatibility with older trainer code.
        self.gene_indices = torch.arange(self.n_gene, dtype=torch.long)
        self.hpo_indices = torch.arange(self.n_hpo, dtype=torch.long)
        self.mp_indices = torch.arange(self.n_mp, dtype=torch.long)

        # ── Drug-disease positive pairs for Phase 1 ───────────────────────
        self.drug_disease_pos_pairs = self._extract_drug_disease_pairs(kg_df)

        logger.info(
            f"KG Stats: {self.n_drug} drugs, {self.n_disease} diseases, "
            f"{self.n_gene} genes, {self.n_hpo} HPO, {self.n_mp} MP, "
            f"{self.n_other} other, {self.edge_index.size(1)} edges "
            f"(bidirectional), {self.n_relations_total} relation types"
        )

    def _extract_drug_disease_pairs(self, kg_df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Extract positive drug-disease pairs from KG edges.
        Returns: List[(drug_local_idx, disease_local_idx)]

        [data-3] Vectorised; replaces iterrows().
        """
        drug_set    = set(self.drug_ids)
        disease_set = set(self.disease_ids)

        # Forward: node1 = drug, node2 = disease
        fwd_mask = (
            kg_df['node1'].isin(drug_set) & kg_df['node2'].isin(disease_set)
        )
        fwd = kg_df[fwd_mask]
        fwd_pairs = list(zip(
            fwd['node1'].map(self.node2idx).astype(int),
            (fwd['node2'].map(self.node2idx) - self.n_drug).astype(int)
        ))

        # Reverse: node1 = disease, node2 = drug
        rev_mask = (
            kg_df['node1'].isin(disease_set) & kg_df['node2'].isin(drug_set)
        )
        rev = kg_df[rev_mask]
        rev_pairs = list(zip(
            rev['node2'].map(self.node2idx).astype(int),
            (rev['node1'].map(self.node2idx) - self.n_drug).astype(int)
        ))

        # Deduplicate: same (drug, disease) may appear under multiple relation types
        return list(set(fwd_pairs + rev_pairs))

    def to(self, device):
        self.edge_index     = self.edge_index.to(device)
        self.edge_type      = self.edge_type.to(device)
        self.node_type_mask = self.node_type_mask.to(device)
        self.gene_indices   = self.gene_indices.to(device)
        self.hpo_indices    = self.hpo_indices.to(device)
        self.mp_indices     = self.mp_indices.to(device)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DrugCombinationDataset(Dataset):

    def __init__(
        self,
        pairs_df:    pd.DataFrame,
        disease2idx: Dict[str, int],
        drug2idx:    Dict[str, int],
        n_drugs:     int,
        neg_ratio:   int = 5,
        mode:        str = 'train'
    ):
        self.n_drugs     = n_drugs
        self.neg_ratio   = neg_ratio
        self.mode        = mode
        self.disease2idx = disease2idx
        self.drug2idx    = drug2idx

        # Drop duplicate rows before indexing.
        pairs_df = pairs_df.drop_duplicates(
            subset=['disease_id', 'drug1_id', 'drug2_id'])

        # [data-7] Vectorised positive triplet construction (replaces iterrows).
        valid_mask = (
            pairs_df['disease_id'].isin(disease2idx) &
            pairs_df['drug1_id'].isin(drug2idx) &
            pairs_df['drug2_id'].isin(drug2idx)
        )
        skipped_unknown = int((~valid_mask).sum())
        valid_df = pairs_df[valid_mask].copy()

        d_idxs = valid_df['disease_id'].map(disease2idx).values.astype(np.int64)
        i_idxs = valid_df['drug1_id'].map(drug2idx).values.astype(np.int64)
        j_idxs = valid_df['drug2_id'].map(drug2idx).values.astype(np.int64)

        # Canonical ordering (i < j)
        i_can = np.minimum(i_idxs, j_idxs)
        j_can = np.maximum(i_idxs, j_idxs)

        # Remove duplicates that arise after canonical ordering
        # (e.g. swapped drug1/drug2 rows map to the same triplet)
        triplets = list(dict.fromkeys(
            zip(d_idxs.tolist(), i_can.tolist(), j_can.tolist())
        ))
        self.pos_triplets: List[Tuple[int, int, int]] = triplets

        # Per-disease positive set for false-negative-safe neg sampling
        self.disease_pos: Dict[int, set] = defaultdict(set)
        for d, i, j in self.pos_triplets:
            self.disease_pos[d].add((i, j))

        self._pad_warnings = 0  # count pads across epoch; log once per epoch

        logger.info(
            f"[{mode}] {len(self.pos_triplets)} positive triplets, "
            f"{len(set(d for d, _, _ in self.pos_triplets))} diseases"
        )
        if skipped_unknown:
            logger.info(
                f"[{mode}] skipped {skipped_unknown} rows with unknown "
                f"disease/drug ids"
            )

    def __len__(self) -> int:
        return len(self.pos_triplets)

    def __getitem__(self, idx: int) -> dict:
        d, i, j = self.pos_triplets[idx]

        if self.mode != 'train':
            return {'pos': torch.tensor([d, i, j], dtype=torch.long)}

        # [improve-neg] Three negative strategies sampled uniformly:
        #   (a) replace drug_i only  — hard negative sharing drug_j
        #   (b) replace drug_j only  — hard negative sharing drug_i
        #   (c) replace both drugs   — fully random pair, broader coverage
        # Adding strategy (c) prevents the model from relying solely on
        # one shared drug to distinguish positives from negatives.
        negs     = []
        attempts = 0
        while len(negs) < self.neg_ratio and attempts < 300:
            attempts += 1
            r = np.random.random()
            if r < 0.35:
                ni = np.random.randint(0, self.n_drugs)
                ni, nj = min(ni, j), max(ni, j)
            elif r < 0.70:
                nj = np.random.randint(0, self.n_drugs)
                ni, nj = min(i, nj), max(i, nj)
            else:
                # [improve-neg-c] Replace both drugs for broader negatives
                ni = np.random.randint(0, self.n_drugs)
                nj = np.random.randint(0, self.n_drugs)
                ni, nj = min(ni, nj), max(ni, nj)
            if (ni, nj) not in self.disease_pos[d] and ni != nj:
                negs.append((d, ni, nj))

        # [data-5] Pad with last valid negative; warn so users notice.
        if len(negs) < self.neg_ratio:
            self._pad_warnings += 1
            pad = negs[-1] if negs else (d, i, j)  # fallback to pos if truly stuck
            while len(negs) < self.neg_ratio:
                negs.append(pad)

        return {
            'pos':  torch.tensor([d, i, j], dtype=torch.long),
            'negs': torch.tensor(negs,      dtype=torch.long),  # [neg_ratio, 3]
        }

    def reset_pad_warnings(self) -> int:
        """Return and reset pad-warning counter (call once per epoch)."""
        count, self._pad_warnings = self._pad_warnings, 0
        return count


# ─────────────────────────────────────────────────────────────────────────────
# Collation & DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    pos = torch.stack([b['pos'] for b in batch])    # [B, 3]
    if 'negs' in batch[0]:
        negs = torch.stack([b['negs'] for b in batch])  # [B, neg_ratio, 3]
        return {'pos': pos, 'negs': negs}
    return {'pos': pos}


def build_dataloaders(
    data_dir:    str,
    drug_ids:    List[str],
    disease_ids: List[str],
    batch_size:  int = 512,
    neg_ratio:   int = 5,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    data_dir    = Path(data_dir)
    drug2idx    = {nid: i for i, nid in enumerate(drug_ids)}
    disease2idx = {nid: i for i, nid in enumerate(disease_ids)}
    n_drugs     = len(drug_ids)

    train_df = load_pairs(data_dir / 'pair_train.csv')
    val_df   = load_pairs(data_dir / 'pair_val.csv')
    test_df  = load_pairs(data_dir / 'pair_test.csv')

    train_ds = DrugCombinationDataset(train_df, disease2idx, drug2idx, n_drugs, neg_ratio, 'train')
    val_ds   = DrugCombinationDataset(val_df,   disease2idx, drug2idx, n_drugs, neg_ratio, 'val')
    test_ds  = DrugCombinationDataset(test_df,  disease2idx, drug2idx, n_drugs, neg_ratio, 'test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
