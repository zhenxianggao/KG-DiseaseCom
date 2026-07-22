import argparse
import json
import logging
import os
from pathlib import Path

import torch

from data       import read_id_csv, load_kg, KGGraph, build_dataloaders
from model   import DrugCombinationModel, count_parameters
from trainer import Trainer


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'train_v8.log')
    fmt = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        for h in [logging.StreamHandler(), logging.FileHandler(log_path, mode='a')]:
            h.setFormatter(fmt)
            root.addHandler(h)
    logging.info(f"Logging to {log_path}")


def parse_args():
    p = argparse.ArgumentParser('Drug Combination Ranking — v8 listwise disease-macro eval')

    # Data
    p.add_argument('--data_dir',     type=str, default='./data_neoplastic')
    p.add_argument('--drug_ids',     type=str, required=True)
    p.add_argument('--disease_ids',  type=str, required=True)
    p.add_argument('--output_dir',   type=str, default='./output_neoplastic')

    # Model
    p.add_argument('--kge_model',    type=str, default='compgcn',
                   choices=['compgcn', 'distmult'],
                   help='KG embedding model. Defaults to lightweight CompGCN+RAAG; '
                        'distmult reproduces the V1 KGE family.')
    p.add_argument('--d_emb',        type=int,   default=64)
    p.add_argument('--d_v',          type=int,   default=64)
    p.add_argument('--dropout',      type=float, default=0.3)
    p.add_argument('--scorer_hidden_mult', type=int, default=1)
    p.add_argument('--compgcn_layers',  type=int,   default=1)
    p.add_argument('--compgcn_dropout', type=float, default=0.1)
    p.add_argument('--no_raag', action='store_true',
                   help='Disable relation-aware attention gate inside CompGCN.')
    p.add_argument('--raag_chunk', type=int, default=50000,
                   help='Number of KG edges processed per RAAG chunk.')

    # Phase A — KGE
    p.add_argument('--kge_epochs',     type=int,   default=50)
    p.add_argument('--kge_batch_size', type=int,   default=16384)
    p.add_argument('--kge_neg',        type=int,   default=64)
    p.add_argument('--kge_margin',     type=float, default=9.0)
    p.add_argument('--kge_adv_temp',   type=float, default=1.0)
    p.add_argument('--kge_lr',         type=float, default=5e-3)
    p.add_argument('--kge_wd',         type=float, default=1e-6)
    p.add_argument('--contrastive_lambda', type=float, default=0.03,
                   help='Weight of KG node contrastive loss in Phase A '
                        '(0 disables)')
    p.add_argument('--contrastive_batch', type=int, default=1024,
                   help='Positive node-pair batch size for contrastive loss')
    p.add_argument('--contrastive_temp', type=float, default=0.3,
                   help='Temperature for in-batch node contrastive loss')
    p.add_argument('--contrastive_max_pairs_per_bucket', type=int, default=12,
                   help='Max positive node pairs sampled per shared-neighbor bucket')
    p.add_argument('--contrastive_max_total_pairs', type=int, default=250000,
                   help='Max total positive node pairs cached for contrastive loss')
    p.add_argument('--contrastive_warmup_epochs', type=int, default=10,
                   help='KGE epochs before enabling contrastive loss')

    # Phase B — Scorer
    p.add_argument('--scorer_epochs', type=int,   default=100)
    p.add_argument('--scorer_lr',     type=float, default=5e-4)
    p.add_argument('--emb_lr',        type=float, default=5e-5,
                   help='LR for KGE embedding fine-tune in Phase B. '
                        'Set to 0 to freeze embeddings (equivalent to --freeze_emb).')
    p.add_argument('--scorer_wd',     type=float, default=1e-3)
    p.add_argument('--neg_ratio',     type=int,   default=16)
    p.add_argument('--loss_mix',      type=float, default=0.5,
                   help='loss = mix * BPR + (1-mix) * InfoNCE')
    p.add_argument('--listwise_lambda', type=float, default=0.2,
                   help='Weight of v8 disease-wise sampled-softmax ranking loss '
                        '(0 disables)')
    p.add_argument('--listwise_num_neg', type=int, default=128,
                   help='Same-disease random drug-pair negatives per positive '
                        'for v8 listwise loss')
    p.add_argument('--listwise_temp', type=float, default=0.7,
                   help='Temperature for v8 sampled-softmax listwise logits. '
                        'Applied to raw scores (not double-tempered with log_tau).')
    p.add_argument('--freeze_emb',    action='store_true',
                   help='Freeze KGE embeddings during Phase B')
    p.add_argument('--patience',      type=int,   default=12)
    p.add_argument('--batch_size',    type=int,   default=512)

    # ── OHNM ─────────────────────────────────────────────────────────────
    p.add_argument('--ohnm_interval',    type=int,   default=15,
                   help='Re-mine hard negatives every N epochs (0 disables)')
    p.add_argument('--ohnm_n_hard',      type=int,   default=8,
                   help='Number of hard negatives cached per disease')
    p.add_argument('--ohnm_pool_factor', type=int,   default=30,
                   help='Candidate pool size = ohnm_n_hard × pool_factor')
    p.add_argument('--ohnm_mix_ratio',   type=float, default=0.5,
                   help='Fraction of per-batch negatives replaced with hard negs')

    # ── drug_pairs.csv auxiliary loss ─────────────────────────────────────
    p.add_argument('--drug_pairs',       type=str,   default=None,
                   help='Path to drug_pairs.csv (auto-loaded from data_dir if exists)')
    p.add_argument('--no_drug_pairs',    action='store_true',
                   help='Disable drug_pairs auxiliary loss even if file exists')
    p.add_argument('--aux_pair_lambda',  type=float, default=0.3,
                   help='Weight of drug-pair aux BPR loss in Phase B (0 disables)')
    p.add_argument('--aux_score_weight', type=float, default=0.2,
                   help='Weight of disease-agnostic drug-pair prior added to score')
    p.add_argument('--aux_pair_batch',   type=int,   default=256,
                   help='Per-step drug-pair aux positive batch size')
    p.add_argument('--aux_pair_neg_k',   type=int,   default=16,
                   help='Negatives per positive for the drug-pair aux loss')
    p.add_argument('--aux_disease_drug_lambda', type=float, default=0.15,
                   help='Weight of v6 disease-drug auxiliary BPR loss (0 disables)')
    p.add_argument('--aux_disease_drug_score_weight', type=float, default=0.1,
                   help='Weight of v6 disease-drug prior added to final score')

    # Misc
    p.add_argument('--seed',          type=int, default=42)
    p.add_argument('--num_workers',   type=int, default=4)
    p.add_argument('--no_kge',        action='store_true',
                   help='Skip Phase A (use random embeddings; for ablation)')
    p.add_argument('--eval_only',     action='store_true')
    p.add_argument('--checkpoint',    type=str, default=None)
    p.add_argument('--top_k',         type=int, default=100)

    return p.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    data_dir = Path(args.data_dir)

    # [fix-13] Use pandas-based read_id_csv to skip CSV header rows correctly.
    drug_ids    = read_id_csv(args.drug_ids)
    disease_ids = read_id_csv(args.disease_ids)
    gene_ids    = read_id_csv(data_dir / 'gene_ids.csv')
    hpo_ids     = read_id_csv(data_dir / 'hpo_ids.csv')
    mp_ids      = read_id_csv(data_dir / 'mp_ids.csv')

    logging.info("Loading KG …")
    kg_df = load_kg(data_dir / 'KG.csv')
    kg    = KGGraph(kg_df, drug_ids, disease_ids, gene_ids, hpo_ids, mp_ids)
    logging.info(f"  Drugs:     {len(drug_ids)}")
    logging.info(f"  Diseases:  {len(disease_ids)}")
    logging.info(f"  Genes:     {kg.n_gene}")
    logging.info(f"  HPO:       {kg.n_hpo}")
    logging.info(f"  MP:        {kg.n_mp}")
    logging.info(f"  Other KG:  {kg.n_other}")
    logging.info(f"  Relations (forward + inverse): {kg.n_relations_total}")

    logging.info("Building dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir    = data_dir,
        drug_ids    = drug_ids,
        disease_ids = disease_ids,
        batch_size  = args.batch_size,
        neg_ratio   = args.neg_ratio,
        num_workers = args.num_workers,
    )

    config = {
        'n_drugs':     len(drug_ids),
        'n_diseases':  len(disease_ids),
        'n_genes':     kg.n_gene,
        'n_hpo':       kg.n_hpo,
        'n_mp':        kg.n_mp,
        'n_other':     kg.n_other,
        'n_relations': kg.n_relations_total,

        'd_emb':       args.d_emb,
        'd_v':         args.d_v,
        'kge_model':   args.kge_model,
        'compgcn_layers': args.compgcn_layers,
        'compgcn_dropout': args.compgcn_dropout,
        'use_raag':    not args.no_raag,
        'raag_chunk':  args.raag_chunk,
        'dropout':     args.dropout,
        'scorer_hidden_mult': args.scorer_hidden_mult,
        'aux_score_weight': args.aux_score_weight,
        'aux_disease_drug_score_weight': (
            args.aux_disease_drug_score_weight
            if args.aux_disease_drug_lambda > 0 else 0.0),
    }

    logging.info(f"Building model (KGE={args.kge_model}) …")
    model = DrugCombinationModel(config)
    counts = count_parameters(model)
    logging.info(f"  Parameter breakdown: {counts}")

    if args.checkpoint:
        logging.info(f"  Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint,
                                          map_location='cpu',
                                          weights_only=True))

    trainer = Trainer(model, kg, config, device)

    # ── Load drug_pairs.csv for the auxiliary head ───────────────────────
    drug_pairs_tensor = None
    if not args.no_drug_pairs and args.aux_pair_lambda > 0:
        from data import load_drug_pairs
        dp_path = Path(args.drug_pairs) if args.drug_pairs else (data_dir / 'drug_pairs.csv')
        if dp_path.exists():
            pairs_df = load_drug_pairs(dp_path)
            drug2idx = {nid: i for i, nid in enumerate(drug_ids)}
            valid = []
            for _, row in pairs_df.iterrows():
                d1, d2 = row['drug1_id'], row['drug2_id']
                if d1 in drug2idx and d2 in drug2idx:
                    a, b = drug2idx[d1], drug2idx[d2]
                    valid.append((min(a, b), max(a, b)))
            valid = list(dict.fromkeys(valid))   # dedup
            if valid:
                drug_pairs_tensor = torch.tensor(
                    valid, dtype=torch.long, device=device)
                logging.info(f"[v8] Loaded {len(valid):,} unique drug pairs "
                             f"from {dp_path} for auxiliary loss")
            else:
                logging.warning(f"[v8] {dp_path} produced 0 valid pairs after "
                                f"drug_id mapping — aux loss will be skipped")
        else:
            logging.warning(f"[v8] No drug_pairs file at {dp_path}; "
                            f"aux loss will be skipped")

    if args.eval_only and not args.checkpoint:
        raise ValueError("--eval_only requires --checkpoint.")

    if not args.eval_only:
        ckpt_path = os.path.join(args.output_dir, 'best_v8.pt')
        if not args.no_kge:
            trainer.train_kge(
                n_epochs    = args.kge_epochs,
                batch_size  = args.kge_batch_size,
                neg_per_pos = args.kge_neg,
                margin      = args.kge_margin,
                adv_temp    = args.kge_adv_temp,
                lr          = args.kge_lr,
                weight_decay= args.kge_wd,
                contrastive_lambda=args.contrastive_lambda,
                contrastive_batch=args.contrastive_batch,
                contrastive_temp=args.contrastive_temp,
                contrastive_max_pairs_per_bucket=args.contrastive_max_pairs_per_bucket,
                contrastive_max_total_pairs=args.contrastive_max_total_pairs,
                contrastive_warmup_epochs=args.contrastive_warmup_epochs,
            )

        trainer.train_scorer(
            train_loader, val_loader, test_loader,
            n_epochs    = args.scorer_epochs,
            lr          = args.scorer_lr,
            emb_lr      = args.emb_lr,       # [fix-14] 0.0 → freeze embeddings
            weight_decay= args.scorer_wd,
            neg_k       = args.neg_ratio,
            loss_mix    = args.loss_mix,
            freeze_emb  = args.freeze_emb,
            patience    = args.patience,
            ckpt        = ckpt_path,
            ohnm_interval     = args.ohnm_interval,
            ohnm_n_hard       = args.ohnm_n_hard,
            ohnm_pool_factor  = args.ohnm_pool_factor,
            ohnm_mix_ratio    = args.ohnm_mix_ratio,
            drug_pairs_tensor = drug_pairs_tensor,
            aux_pair_lambda   = args.aux_pair_lambda if drug_pairs_tensor is not None else 0.0,
            aux_score_weight  = args.aux_score_weight if drug_pairs_tensor is not None else 0.0,
            aux_pair_batch    = args.aux_pair_batch,
            aux_pair_neg_k    = args.aux_pair_neg_k,
            aux_disease_drug_lambda = args.aux_disease_drug_lambda,
            aux_disease_drug_score_weight = args.aux_disease_drug_score_weight,
            listwise_lambda = args.listwise_lambda,
            listwise_num_neg = args.listwise_num_neg,
            listwise_temp = args.listwise_temp,
        )

    logging.info("\n=== Final Test Evaluation (full-space ranking) ===")
    test_metrics = trainer.evaluate_full_space(test_loader)
    for k, v in sorted(test_metrics.items()):
        logging.info(f"  {k}: {v:.4f}")

    metrics_path = os.path.join(args.output_dir, 'test_metrics_v8.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logging.info(f"Metrics saved to {metrics_path}")

    # Example inference
    disease2idx    = {nid: i for i, nid in enumerate(disease_ids)}
    sample_disease = disease_ids[0]
    logging.info(f"\n=== Example Inference: disease={sample_disease} ===")
    results = trainer.rank_all_pairs_for_disease(
        sample_disease, disease2idx, drug_ids, top_k=args.top_k)
    logging.info("Top-5 drug combinations:")
    for d1, d2, s in results[:5]:
        logging.info(f"  ({d1}, {d2})  score={s:.4f}")

    rpath = os.path.join(args.output_dir, f'rankings_v8_{sample_disease}.json')
    with open(rpath, 'w') as f:
        json.dump(
            [{'drug1': d1, 'drug2': d2, 'score': s} for d1, d2, s in results],
            f, indent=2,
        )
    logging.info(f"Rankings saved to {rpath}")


if __name__ == '__main__':
    main()
