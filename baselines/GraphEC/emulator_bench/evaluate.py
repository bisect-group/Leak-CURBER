from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as ssp
from scipy.sparse.linalg import inv
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .results import (
    compact_evaluation_summary,
    compute_care_metrics,
    compute_graphec_metrics,
    compute_supplemental_classification_metrics,
    compute_supplemental_ranking_metrics,
    multilabel_targets,
    write_ranked_outputs,
)
from .train import build_model, load_active_sites, padding_ver1
from .utils import (
    BASELINE_ROOT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_RUNS_ROOT,
    autocast_kwargs,
    baseline_cwd,
    cache_dirs,
    choose_precision,
    ensure_dir,
    ensure_graphec_data_links,
    import_task_module,
    load_run_metadata,
    read_json,
    seed_results_root_for_split,
    seed_train_metadata_path_for_split,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GraphEC on an Leak-CURBER split")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--rank-top-k", type=int, default=50)
    diffusion = parser.add_mutually_exclusive_group()
    diffusion.add_argument("--label-diffusion", dest="label_diffusion", action="store_true")
    diffusion.add_argument("--no-label-diffusion", dest="label_diffusion", action="store_false")
    parser.set_defaults(label_diffusion=True)
    parser.add_argument("--diffusion-lambda", type=float, default=0.1)
    parser.add_argument("--identity-cutoff", type=float, default=0.1)
    return parser.parse_args()


def load_manifest_dataset(path: str | Path, label_to_idx: dict[str, int]) -> tuple[dict, list[str], np.ndarray]:
    manifest = pd.read_csv(path)
    dataset = {}
    names = []
    targets = []
    for _, row in manifest.iterrows():
        key = str(row["cache_key"])
        labels = [label for label in str(row["EC number"]).split(";") if label in label_to_idx]
        if not labels:
            continue
        dataset[key] = [str(row["Sequence"])]
        names.append(key)
        targets.append(multilabel_targets(labels, label_to_idx))
    if not dataset:
        raise ValueError(f"No usable rows in {path}")
    return dataset, names, np.stack(targets, axis=0)


def load_models(model_mod, train_metadata: dict, num_labels: int, device: torch.device) -> list[torch.nn.Module]:
    models = []
    for fold_result in train_metadata["fold_results"]:
        checkpoint = torch.load(fold_result["checkpoint"], map_location=device)
        model = build_model(model_mod, num_labels, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
    if not models:
        raise ValueError("No fold checkpoints found in train metadata")
    return models


def predict_scores(
    *,
    dataset: dict,
    active_sites: dict,
    data_mod,
    models: list[torch.nn.Module],
    batch_size: int,
    num_workers: int,
    precision: str,
) -> tuple[list[str], np.ndarray]:
    dataloader = DataLoader(
        data_mod.ProteinGraphDataset(dataset, range(len(dataset)), args=None, active_sites=active_sites),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    device = next(models[0].parameters()).device
    names = []
    scores = []
    for batch in tqdm(dataloader, desc="GraphEC evaluate", unit="batch"):
        batch_data, mask_data, batch_activate_site = padding_ver1(
            batch.node_feat, batch.batch, batch.node_feat.shape[1], batch.activate_site
        )
        batch = batch.to(device)
        batch_data = batch_data.to(device)
        mask_data = mask_data.to(device)
        batch_activate_site = batch_activate_site.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(**autocast_kwargs(precision)):
            fold_scores = [
                model(
                    batch.X,
                    batch.node_feat,
                    batch.edge_index,
                    batch.seq,
                    batch.batch,
                    batch_data,
                    mask_data,
                    batch_activate_site,
                ).sigmoid()
                for model in models
            ]
            probs = torch.stack(fold_scores, 0).mean(0)
        names.extend([str(name) for name in batch.name])
        scores.append(probs.detach().cpu().numpy())
    return names, np.concatenate(scores, axis=0)


def sparse_divide_nonzero(a, b):
    inv_b = b.copy()
    inv_b.data = 1 / inv_b.data
    return a.multiply(inv_b)


def jaccard(W):
    co = W.dot(W.T)
    clen = W.sum(axis=1)
    nonzero_mask = co.astype("bool")
    denominator = nonzero_mask.multiply(clen) + nonzero_mask.multiply(clen.T) - co + nonzero_mask
    return sparse_divide_nonzero(co, denominator)


def compute_laplacian(W):
    jw = jaccard(W).multiply(W)
    degree = 1.0 / jw.sum(axis=1)
    w_s2f = 0.5 * (jw.multiply(degree) + jw.multiply(degree.T))
    degree = w_s2f.sum(axis=1)
    d_s2f = ssp.spdiags(degree.T, 0, W.shape[0], W.shape[0])
    return d_s2f - w_s2f


def read_fasta_ids(fasta: Path) -> list[str]:
    ids = []
    with fasta.open() as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line[1:].strip())
    return ids


def homology_matrix(fasta: Path, diamond: Path, work_dir: Path, cutoff: float = 0.1):
    ensure_dir(work_dir)
    ids = read_fasta_ids(fasta)
    id_to_idx = dict(zip(ids, range(len(ids))))
    db_prefix = work_dir / "homology_db"
    out_tsv = work_dir / "homology.tsv"
    subprocess.run([str(diamond), "makedb", "--in", str(fasta), "-d", str(db_prefix), "--quiet"], check=True)
    subprocess.run(
        [
            str(diamond),
            "blastp",
            "-d",
            str(db_prefix) + ".dmnd",
            "-q",
            str(fasta),
            "-o",
            str(out_tsv),
            "--very-sensitive",
            "--quiet",
            "-p",
            "8",
        ],
        check=True,
    )
    homology = {}
    with out_tsv.open() as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) < 3:
                continue
            query_idx = id_to_idx[fields[0]]
            subject_idx = id_to_idx[fields[1]]
            identity = float(fields[2]) / 100.0
            homology.setdefault(query_idx, {})[subject_idx] = max(
                homology.get(query_idx, {}).get(subject_idx, 0.0), identity
            )
            homology.setdefault(subject_idx, {})[query_idx] = max(
                homology.get(subject_idx, {}).get(query_idx, 0.0), identity
            )
    rows, cols, data = [], [], []
    for row, values in homology.items():
        for col, identity in values.items():
            value = 1.0 if row == col else identity
            if value >= cutoff:
                rows.append(row)
                cols.append(col)
                data.append(value)
    return ssp.csr_matrix((np.array(data), (rows, cols)), shape=(len(ids), len(ids)))


def run_label_diffusion(
    *,
    scores: np.ndarray,
    train_manifest: str | Path,
    eval_manifest: str | Path,
    labels: list[str],
    work_dir: Path,
    lamda: float,
    identity_cutoff: float,
) -> np.ndarray:
    diamond = BASELINE_ROOT / "EC_number/tools/diamond"
    if not diamond.exists():
        raise FileNotFoundError(f"Missing GraphEC Diamond executable: {diamond}")
    ensure_dir(work_dir)
    train_df = pd.read_csv(train_manifest)
    eval_df = pd.read_csv(eval_manifest)
    label_to_idx = {label: index for index, label in enumerate(labels)}
    train_fasta = work_dir / "train.fasta"
    eval_fasta = work_dir / "eval.fasta"
    train_seed_fasta = work_dir / "train_seed.fasta"
    combined_fasta = work_dir / "train_seed_and_eval.fasta"
    blast_tsv = work_dir / "eval_vs_train.tsv"
    with train_fasta.open("w") as handle:
        for _, row in train_df.loc[:, ["Entry", "Sequence"]].drop_duplicates().iterrows():
            handle.write(f">{row['Entry']}\n{row['Sequence']}\n")
    with eval_fasta.open("w") as handle:
        for _, row in eval_df.loc[:, ["Entry", "Sequence"]].drop_duplicates().iterrows():
            handle.write(f">{row['Entry']}\n{row['Sequence']}\n")
    db_prefix = work_dir / "train_db"
    subprocess.run([str(diamond), "makedb", "--in", str(train_fasta), "-d", str(db_prefix), "--quiet"], check=True)
    subprocess.run(
        [
            str(diamond),
            "blastp",
            "-d",
            str(db_prefix) + ".dmnd",
            "-q",
            str(eval_fasta),
            "-o",
            str(blast_tsv),
            "--very-sensitive",
            "--quiet",
            "-p",
            "8",
        ],
        check=True,
    )
    train_seed_ids = set()
    with blast_tsv.open() as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) >= 3 and float(fields[2]) / 100.0 > identity_cutoff:
                train_seed_ids.add(fields[1])
    if not train_seed_ids:
        print("[emulator_bench] label diffusion skipped: no homologous train seeds", flush=True)
        return scores
    seed_df = train_df[train_df["Entry"].isin(train_seed_ids)].drop_duplicates("Entry")
    with train_seed_fasta.open("w") as handle:
        for row in seed_df.itertuples(index=False):
            handle.write(f">{row.Entry}\n{row.Sequence}\n")
    combined_fasta.write_text(train_seed_fasta.read_text() + eval_fasta.read_text())
    W = homology_matrix(combined_fasta, diamond, work_dir / "homology", cutoff=identity_cutoff)
    row_indices, col_indices = [], []
    for row_index, (_, row) in enumerate(seed_df.iterrows()):
        for label in str(row["EC number"]).split(";"):
            if label in label_to_idx:
                row_indices.append(row_index)
                col_indices.append(label_to_idx[label])
    train_seed_label = ssp.csr_matrix(
        ([1] * len(row_indices), (row_indices, col_indices)),
        shape=(len(seed_df), len(labels)),
    )
    initial_pred = ssp.vstack([train_seed_label, ssp.csr_matrix(scores)])
    laplacian = compute_laplacian(W)
    kernel = inv((ssp.identity(W.shape[0]) + laplacian.multiply(lamda)).tocsc())[len(seed_df) :]
    return kernel.dot(initial_pred).toarray()


def evaluate_split(args, metadata, train_metadata, eval_split: str, labels: list[str], label_to_idx: dict[str, int]):
    dataset, ordered_names, y_true = load_manifest_dataset(metadata["manifests"][eval_split], label_to_idx)
    ensure_graphec_data_links(dataset.keys(), args.cache_root)
    active_sites = load_active_sites(list(dataset.keys()), args.cache_root)
    result_root = ensure_dir(seed_results_root_for_split(args.split_group, args.seed, args.runs_root))
    precision = choose_precision(args.precision)
    with baseline_cwd():
        model_mod = import_task_module("EC_number", "model")
        data_mod = sys.modules["data"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = load_models(model_mod, train_metadata, len(labels), device)
        pred_names, raw_scores = predict_scores(
            dataset=dataset,
            active_sites=active_sites,
            data_mod=data_mod,
            models=models,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            precision=precision,
        )
    order = [pred_names.index(name) for name in ordered_names]
    raw_scores = raw_scores[order]
    final_scores = raw_scores
    if args.label_diffusion:
        final_scores = run_label_diffusion(
            scores=raw_scores,
            train_manifest=metadata["manifests"]["train"],
            eval_manifest=metadata["manifests"][eval_split],
            labels=labels,
            work_dir=result_root / f"{eval_split}_label_diffusion",
            lamda=args.diffusion_lambda,
            identity_cutoff=args.identity_cutoff,
        )
    np.savez_compressed(
        result_root / f"{eval_split}_scores.npz",
        names=np.array(ordered_names),
        raw_scores=raw_scores,
        final_scores=final_scores,
        labels=np.array(labels),
    )
    care_csv = result_root / f"{eval_split}_results_df.csv"
    care_df = write_ranked_outputs(
        metadata["manifests"][eval_split],
        final_scores,
        labels,
        care_csv,
        top_k=args.rank_top_k,
    )
    metrics = {
        "split_group": args.split_group,
        "seed": args.seed,
        "eval_split": eval_split,
        "label_diffusion": args.label_diffusion,
        "graphec": compute_graphec_metrics(y_true, final_scores),
        "care_task1": compute_care_metrics(care_df),
        "supplemental": {
            "classification": compute_supplemental_classification_metrics(
                y_true,
                (final_scores >= 0.5).astype(int),
            ),
            "ranking": compute_supplemental_ranking_metrics(care_df),
        },
        "artifacts": {
            "care_ranked_csv": str(care_csv),
            "scores_npz": str(result_root / f"{eval_split}_scores.npz"),
        },
    }
    metrics_path = write_json(result_root / f"{eval_split}_metrics.json", metrics)
    print(f"[emulator_bench] {eval_split} metrics: {metrics_path}", flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    metadata = load_run_metadata(args.split_group, args.runs_root)
    train_metadata = read_json(seed_train_metadata_path_for_split(args.split_group, args.seed, args.runs_root))
    vocab = read_json(metadata["vocab_path"])
    idx_to_label = {int(index): label for index, label in vocab["idx_to_label"].items()}
    labels = [idx_to_label[index] for index in sorted(idx_to_label)]
    label_to_idx = {label: index for index, label in enumerate(labels)}
    eval_splits = ["val", "test"] if args.eval_split == "both" else [args.eval_split]
    all_metrics = [
        evaluate_split(args, metadata, train_metadata, eval_split, labels, label_to_idx)
        for eval_split in eval_splits
    ]
    result_root = ensure_dir(seed_results_root_for_split(args.split_group, args.seed, args.runs_root))
    write_json(
        result_root / "evaluation_summary.json",
        compact_evaluation_summary(all_metrics, train_metadata=train_metadata),
    )


if __name__ == "__main__":
    main()
