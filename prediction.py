from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Local imports (same directory as this script)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from data import KGGraph, read_id_csv, load_kg  # noqa: E402
from model import DrugCombinationModel    # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Rank drug combinations for a given disease.')
    p.add_argument('--data_dir',     required=True,
                   help='Directory with KG.csv, drug_ids.csv, disease_ids.csv, etc.')
    p.add_argument('--drug_ids',     default=None,
                   help='Path to drug_ids.csv (default: data_dir/drug_ids.csv).')
    p.add_argument('--disease_ids',  default=None,
                   help='Path to disease_ids.csv (default: data_dir/disease_ids.csv).')
    p.add_argument('--output_dir',   default=None,
                   help='Training output directory (contains config.json & best_v8.pt). '
                        'If omitted, inferred from --checkpoint parent directory.')
    p.add_argument('--checkpoint',   default=None,
                   help='Path to model checkpoint .pt. '
                        'Default: output_dir/best_v8.pt.')
    p.add_argument('--disease',     nargs='+', required=True,
                   help='One or more disease IDs to rank (e.g. C0006118).')
    p.add_argument('--top_k',       type=int, default=100,
                   help='Number of top-ranked drug pairs to return per disease.')
    p.add_argument('--out_file',    default=None,
                   help='Write results to this file (.tsv or .json). '
                        'Default: print to stdout as TSV.')
    p.add_argument('--id_name_csv', default=None,
                   help='Path to ID_NAME.csv (default: data_dir/ID_NAME.csv).')
    p.add_argument('--device',      default='auto',
                   help='"auto" picks CUDA if available, else CPU.')
    p.add_argument('--chunk',       type=int, default=200_000,
                   help='Pair-scoring chunk size (tune to fit GPU VRAM).')

    # ── Model hyperparams (only needed when config.json is absent) ──────────
    g = p.add_argument_group('model hyperparams (fallback when config.json is missing)')
    g.add_argument('--d_emb',              type=int,   default=128)
    g.add_argument('--d_v',               type=int,   default=128)
    g.add_argument('--kge_model',          type=str,   default='compgcn')
    g.add_argument('--compgcn_layers',     type=int,   default=2)
    g.add_argument('--compgcn_dropout',    type=float, default=0.1)
    g.add_argument('--no_raag',            action='store_true')
    g.add_argument('--raag_chunk',         type=int,   default=50_000)
    g.add_argument('--dropout',            type=float, default=0.3)
    g.add_argument('--scorer_hidden_mult', type=int,   default=1)
    g.add_argument('--aux_score_weight',        type=float, default=0.2)
    g.add_argument('--aux_disease_drug_score_weight', type=float, default=0.1)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_id_name(path: str) -> dict[str, str]:
    """Read ID_NAME.csv → {id: name}. Tolerates missing/extra columns."""
    mapping: dict[str, str] = {}
    if not os.path.isfile(path):
        logger.warning(f"ID_NAME.csv not found at {path}; names will be empty.")
        return mapping
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_col   = row.get('ID') or row.get('id') or ''
            name_col = row.get('Name') or row.get('name') or ''
            if id_col:
                mapping[id_col.strip()] = name_col.strip()
    logger.info(f"Loaded {len(mapping):,} ID→Name entries from {path}")
    return mapping


def build_pair_index(n_drug: int, device: torch.device):
    """Upper-triangular pair indices for all C(n_drug, 2) pairs."""
    idx = torch.triu_indices(n_drug, n_drug, offset=1, device=device)
    pair_i = idx[0].to(torch.int32)
    pair_j = idx[1].to(torch.int32)
    logger.info(f"Pair index: {n_drug} drugs → {pair_i.numel():,} pairs")
    return pair_i, pair_j


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir  = Path(args.data_dir)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None

    # Resolve output_dir: explicit > inferred from checkpoint parent > error
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif ckpt_path:
        output_dir = ckpt_path.parent
        logger.info(f"--output_dir not given; inferred from checkpoint: {output_dir}")
    else:
        logger.error("Provide --output_dir or --checkpoint so config.json can be located.")
        sys.exit(1)

    if ckpt_path is None:
        ckpt_path = output_dir / 'best_v8.pt'

    cfg_path     = output_dir / 'config.json'
    id_name_path = args.id_name_csv or str(data_dir / 'ID_NAME.csv')

    if not ckpt_path.exists():
        logger.error(f"Required file not found: {ckpt_path}  (checkpoint)")
        sys.exit(1)

    # ── Config ───────────────────────────────────────────────────────────────
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = json.load(f)
        logger.info(f"Config loaded from {cfg_path}")
        _config_from_file = True
    else:
        logger.warning(f"config.json not found at {cfg_path}; will build from CLI args "
                       f"after loading KG data. Ensure hyperparams match training.")
        config = None
        _config_from_file = False

    # ── ID lists ─────────────────────────────────────────────────────────────
    drug_ids    = read_id_csv(args.drug_ids    or str(data_dir / 'drug_ids.csv'))
    disease_ids = read_id_csv(args.disease_ids or str(data_dir / 'disease_ids.csv'))
    gene_ids    = read_id_csv(str(data_dir / 'gene_ids.csv'))

    hpo_path = data_dir / 'hpo_ids.csv'
    mp_path  = data_dir / 'mp_ids.csv'
    hpo_ids  = read_id_csv(str(hpo_path)) if hpo_path.exists() else []
    mp_ids   = read_id_csv(str(mp_path))  if mp_path.exists()  else []

    disease2idx = {nid: i for i, nid in enumerate(disease_ids)}
    drug2idx    = {nid: i for i, nid in enumerate(drug_ids)}

    # Validate requested diseases
    unknown = [d for d in args.disease if d not in disease2idx]
    if unknown:
        logger.error(f"Unknown disease ID(s): {unknown}")
        logger.error(f"Available diseases: {disease_ids[:10]} ...")
        sys.exit(1)

    # ── Drug name mapping ────────────────────────────────────────────────────
    id_name = load_id_name(id_name_path)

    # ── KG graph (needed for CompGCN edge_index / edge_type) ─────────────────
    logger.info("Building KG graph …")
    kg_df = load_kg(str(data_dir / 'KG.csv'))
    kg = KGGraph(kg_df, drug_ids, disease_ids, gene_ids, hpo_ids, mp_ids)
    kg = kg.to(device)
    logger.info(f"  Drugs: {kg.n_drug}  Diseases: {kg.n_disease}  "
                f"Genes: {kg.n_gene}  Relations: {kg.n_relations_total}")

    # ── Config (build from CLI args when config.json was absent) ─────────────
    if not _config_from_file:
        config = {
            'n_drugs':     kg.n_drug,
            'n_diseases':  kg.n_disease,
            'n_genes':     kg.n_gene,
            'n_hpo':       kg.n_hpo,
            'n_mp':        kg.n_mp,
            'n_other':     kg.n_other,
            'n_relations': kg.n_relations_total,
            'd_emb':       args.d_emb,
            'd_v':         args.d_v,
            'kge_model':   args.kge_model,
            'compgcn_layers':   args.compgcn_layers,
            'compgcn_dropout':  args.compgcn_dropout,
            'use_raag':    not args.no_raag,
            'raag_chunk':  args.raag_chunk,
            'dropout':     args.dropout,
            'scorer_hidden_mult': args.scorer_hidden_mult,
            'aux_score_weight':  args.aux_score_weight,
            'aux_disease_drug_score_weight': args.aux_disease_drug_score_weight,
        }
        logger.info(f"Config built from CLI args: d_emb={config['d_emb']}, "
                    f"d_v={config['d_v']}, kge={config['kge_model']}, "
                    f"layers={config['compgcn_layers']}")

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Building model …")
    model = DrugCombinationModelV4(config)
    if hasattr(model.kge, 'set_graph'):
        model.kge.set_graph(kg.edge_index, kg.edge_type)
    sd = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    logger.info(f"Checkpoint loaded from {ckpt_path}")

    # ── Precompute drug features (shared across all diseases) ─────────────────
    logger.info("Precomputing drug features …")
    with torch.no_grad():
        h_drug_pre = model.precompute_drug_pair_features()   # [N_drug, d_v]

    # ── Pair index ────────────────────────────────────────────────────────────
    pair_i, pair_j = build_pair_index(kg.n_drug, device)

    # ── Collect results ───────────────────────────────────────────────────────
    all_results: list[dict] = []

    for disease_id in args.disease:
        didx = disease2idx[disease_id]
        logger.info(f"Ranking pairs for disease {disease_id} (idx={didx}) …")

        with torch.no_grad():
            scores = model.score_full_space_for_disease(
                disease_idx = didx,
                pair_i      = pair_i,
                pair_j      = pair_j,
                h_drug_pre  = h_drug_pre,
                chunk       = args.chunk,
            )

        top_k = min(args.top_k, scores.numel())
        topk_scores, topk_idx = torch.topk(scores, k=top_k)

        i_arr = pair_i[topk_idx].long()
        j_arr = pair_j[topk_idx].long()

        for idx in range(top_k):
            di = drug_ids[i_arr[idx].item()]
            dj = drug_ids[j_arr[idx].item()]
            all_results.append({
                'disease_id':  disease_id,
                'rank':        idx + 1,
                'drug1_id':    di,
                'drug1_name':  id_name.get(di, ''),
                'drug2_id':    dj,
                'drug2_name':  id_name.get(dj, ''),
                'score':       round(float(topk_scores[idx].item()), 6),
            })

        logger.info(f"  Top-5 for {disease_id}:")
        for row in all_results[-top_k:][:5]:
            logger.info(f"    #{row['rank']:3d}  {row['drug1_id']} ({row['drug1_name']}) + "
                        f"{row['drug2_id']} ({row['drug2_name']})  score={row['score']:.4f}")

    # ── Output ────────────────────────────────────────────────────────────────
    TSV_COLS = ['disease_id', 'rank', 'drug1_id', 'drug1_name',
                'drug2_id', 'drug2_name', 'score']

    if args.out_file:
        out_path = Path(args.out_file)
        if out_path.suffix.lower() == '.json':
            with open(out_path, 'w') as f:
                json.dump(all_results, f, indent=2)
        else:
            # Default: TSV
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=TSV_COLS, delimiter='\t')
                writer.writeheader()
                writer.writerows(all_results)
        logger.info(f"Results saved to {out_path} ({len(all_results)} rows)")
    else:
        # Print TSV to stdout
        writer = csv.DictWriter(sys.stdout, fieldnames=TSV_COLS,
                                delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)


if __name__ == '__main__':
    main()
