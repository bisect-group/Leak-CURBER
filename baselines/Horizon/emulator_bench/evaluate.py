from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .results import compute_care_task2_metrics, compute_supplemental_ec_metrics
from .utils import (
    BASELINE_ROOT,
    DEFAULT_RUNS_ROOT,
    ensure_dir,
    load_run_metadata,
    read_json,
    run_command,
    seed_results_root,
    seed_train_metadata_path,
    write_json,
    write_yaml,
)


def _abs(path: str | Path) -> str:
    return str(Path(path).resolve())


def build_eval_config(metadata: dict, *, eval_split: str, seed_root: Path) -> dict:
    return {
        "seed": 42,
        "logging": {
            "log_dir": str(seed_root / "eval_logs"),
            "checkpoint_dir": str(seed_root / "eval_checkpoints"),
            "save_every_n_epochs": 10,
            "log_every_n_steps": 1,
        },
        "data": {
            "train_pairs_path": _abs(metadata["baseline_files"]["train"]["pairs"]),
            "test_pairs_path": _abs(metadata["baseline_files"][eval_split]["pairs"]),
            "train_reactions_path": _abs(metadata["baseline_files"]["train"]["reactions"]),
            "test_reactions_path": _abs(metadata["baseline_files"][eval_split]["reactions"]),
            "protein_embeds_path": _abs(metadata["baseline_files"]["protein_embeds"]),
            "train_batch_size": 16384,
            "retrieval_batch_size": 128,
            "num_workers": 0,
            "pin_memory": False,
            "rdkit_fp_dim": 1024,
            "drfp_dim": 1024,
            "standardize_reactions": True,
            "standardize_hypervalent": True,
            "standardize_remove_hs": True,
            "standardize_kekulize": False,
            "standardize_uncharge": True,
            "standardize_metals": True,
        },
        "model": {
            "name": "DualContrastiveModel",
            "query_encoder_dims": [2048, 4096, 4096, 512],
            "target_encoder_dims": [1024, 4096, 4096, 512],
            "embedding_dim": 512,
        },
        "training": {
            "max_epochs": 100,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "loss": {
                "name": "FullBatchMLNCELoss",
                "beta": 10.0,
                "learn_beta": False,
                "beta_min": 0.01,
                "beta_max": 100.0,
            },
            "metrics": {"top_k": [1, 10, 100, 1000]},
            "check_val_every_n_epoch": 10,
            "enable_progress_bar": True,
        },
    }


def _read_reactions(path: str | Path) -> dict[str, str]:
    reactions = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            reactions[row["reaction_id"]] = row["reaction_smiles"]
    return reactions


def _load_manifest_rows(metadata: dict, split: str) -> pd.DataFrame:
    return pd.read_csv(metadata["manifests"][split])


def _protein_ecs(metadata: dict, splits: tuple[str, ...]) -> dict[str, set[str]]:
    protein_ecs: dict[str, set[str]] = defaultdict(set)
    for split in splits:
        frame = _load_manifest_rows(metadata, split)
        for row in frame.itertuples(index=False):
            for ec in str(row.ec_number).split(";"):
                if ec:
                    protein_ecs[str(row.protein_id)].add(ec)
    return protein_ecs


def _build_ec_index(
    protein_ids: list[str],
    protein_ecs: dict[str, set[str]],
    *,
    device: str,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    ec_labels = sorted({ec for ecs in protein_ecs.values() for ec in ecs})
    ec_to_index = {ec: idx for idx, ec in enumerate(ec_labels)}
    protein_indices = []
    ec_indices = []
    for protein_idx, protein_id in enumerate(protein_ids):
        for ec in sorted(protein_ecs.get(protein_id, set())):
            protein_indices.append(protein_idx)
            ec_indices.append(ec_to_index[ec])
    if not protein_indices:
        raise ValueError("No EC labels available for CARE ranking")
    return (
        ec_labels,
        torch.tensor(protein_indices, dtype=torch.long, device=device),
        torch.tensor(ec_indices, dtype=torch.long, device=device),
    )


def _build_ec_centroids(
    target_embeds: torch.Tensor,
    ec_protein_indices: torch.Tensor,
    ec_indices: torch.Tensor,
    num_ecs: int,
) -> torch.Tensor:
    centroids = torch.zeros(
        num_ecs,
        target_embeds.shape[1],
        dtype=target_embeds.dtype,
        device=target_embeds.device,
    )
    counts = torch.zeros(num_ecs, dtype=target_embeds.dtype, device=target_embeds.device)
    centroids.index_add_(0, ec_indices, target_embeds.index_select(0, ec_protein_indices))
    counts.index_add_(0, ec_indices, torch.ones_like(ec_indices, dtype=target_embeds.dtype))
    return centroids / counts.clamp_min(1).unsqueeze(1)


def _maybe_normalize(values: torch.Tensor, similarity: str) -> torch.Tensor:
    if similarity == "cosine":
        return F.normalize(values, p=2, dim=-1)
    return values


def _truth_by_reaction(metadata: dict, split: str) -> dict[str, dict]:
    frame = _load_manifest_rows(metadata, split)
    truth: dict[str, dict] = {}
    for reaction_id, group in frame.groupby("reaction_id", sort=True):
        labels = sorted(
            {
                ec
                for value in group["ec_number"]
                for ec in str(value).split(";")
                if ec
            }
        )
        reaction_smiles = str(group["reaction_smiles"].iloc[0])
        truth[str(reaction_id)] = {
            "Reaction": reaction_smiles,
            "Reaction Text": reaction_smiles,
            "Mapped Reaction": reaction_smiles,
            "EC number": ";".join(labels),
            "reaction_id": str(reaction_id),
            "protein_count": int(group["protein_id"].nunique()),
        }
    return truth


def _create_fingerprint_dataset(reactions_path: str | Path):
    from horizyn.datasets.base import BaseDataset
    from horizyn.datasets.collection import MergeDataset
    from horizyn.datasets.csv import CSVDataset
    from horizyn.datasets.fingerprints import (
        DRFPFingerprintDataset,
        RDKitPlusFingerprintDataset,
    )
    from horizyn.datasets.transform import ConcatTensorTransform

    reactions = CSVDataset(
        file_path=str(reactions_path),
        key_column="reaction_id",
        columns=["reaction_smiles"],
    )
    augmented_keys = []
    augmented_data = []
    for rxn_id in reactions.keys:
        smiles = reactions[rxn_id]["reaction_smiles"]
        augmented_keys.append(f"{rxn_id}_f")
        augmented_data.append({"reaction_smiles": smiles})
        parts = smiles.split(">>")
        if len(parts) == 2:
            augmented_keys.append(f"{rxn_id}_r")
            augmented_data.append({"reaction_smiles": f"{parts[1]}>>{parts[0]}"})
    augmented = BaseDataset(keys=augmented_keys, array_data=augmented_data)
    rdkit_fp = RDKitPlusFingerprintDataset(
        reaction_dataset=augmented,
        vec_dim=1024,
        mol_fp_type="morgan",
        rxn_fp_type="struct",
        use_chirality=True,
        standardize=True,
        standardize_hypervalent=True,
        standardize_remove_hs=True,
        standardize_kekulize=False,
        standardize_uncharge=True,
        standardize_metals=True,
    )
    drfp_fp = DRFPFingerprintDataset(
        reaction_dataset=augmented,
        vec_dim=1024,
        radius=3,
        rings=True,
        standardize=True,
        standardize_hypervalent=True,
        standardize_remove_hs=True,
        standardize_kekulize=False,
        standardize_uncharge=True,
        standardize_metals=True,
    )
    merged = MergeDataset(datasets={"rdkit": rdkit_fp, "drfp": drfp_fp}, add_prefix=False)
    merged.append_transforms(ConcatTensorTransform(labels=["rdkit", "drfp"], dim=0))
    return merged


def write_care_ranked_csv(
    *,
    metadata: dict,
    checkpoint: str | Path,
    eval_split: str,
    output_csv: str | Path,
    device: str,
    batch_size: int,
    direction_aggregation: str,
    ec_candidate_splits: tuple[str, ...],
    ec_scoring: str,
    score_similarity: str,
    max_rank_columns: int | None,
) -> pd.DataFrame:
    from horizyn.datasets.hdf5 import EmbedDataset
    from horizyn.lightning_module import HorizynLitModule

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HorizynLitModule.load_from_checkpoint(str(checkpoint), map_location=device)
    model.eval()
    model.to(device)

    protein_embeds = EmbedDataset(metadata["baseline_files"]["protein_embeds"], in_memory=True)
    protein_ecs = _protein_ecs(metadata, ec_candidate_splits)
    protein_ids = [protein_id for protein_id in protein_embeds.keys if protein_ecs.get(protein_id)]
    ec_labels, ec_protein_indices, ec_indices = _build_ec_index(
        protein_ids,
        protein_ecs,
        device=device,
    )

    target_embeds = torch.zeros(
        len(protein_ids),
        model.model.target_encoder.output_dim,
        device=device,
    )
    with torch.no_grad():
        for start in tqdm(range(0, len(protein_ids), batch_size), desc="CARE targets"):
            end = min(start + batch_size, len(protein_ids))
            batch_keys = protein_ids[start:end]
            target_vecs = torch.stack([protein_embeds[key] for key in batch_keys]).to(device)
            target_embeds[start:end] = model.model.target_encoder(target_vecs)

    ec_centroids = None
    scoring_target_embeds = target_embeds
    if ec_scoring == "centroid":
        ec_centroids = _build_ec_centroids(
            target_embeds,
            ec_protein_indices,
            ec_indices,
            len(ec_labels),
        )
        ec_centroids = _maybe_normalize(ec_centroids, score_similarity)
    elif ec_scoring == "max":
        scoring_target_embeds = _maybe_normalize(target_embeds, score_similarity)
    else:
        raise ValueError(f"Unsupported EC scoring mode: {ec_scoring}")

    reaction_path = metadata["baseline_files"][eval_split]["reactions"]
    reactions = _read_reactions(reaction_path)
    fingerprints = _create_fingerprint_dataset(reaction_path)
    truth = _truth_by_reaction(metadata, eval_split)
    rows = []

    with torch.no_grad():
        for reaction_id in tqdm(sorted(truth), desc=f"CARE {eval_split} ranks"):
            query_scores = []
            for suffix in ("f", "r"):
                if direction_aggregation == "forward" and suffix == "r":
                    continue
                key = f"{reaction_id}_{suffix}"
                if key not in fingerprints.keys:
                    continue
                query_fp = fingerprints[key].unsqueeze(0).to(device)
                query_embed = model.model.query_encoder(query_fp)
                query_embed = _maybe_normalize(query_embed, score_similarity)
                if ec_scoring == "centroid":
                    assert ec_centroids is not None
                    query_scores.append(torch.matmul(query_embed, ec_centroids.T).squeeze(0))
                else:
                    query_scores.append(
                        torch.matmul(query_embed, scoring_target_embeds.T).squeeze(0)
                    )
            if not query_scores:
                raise ValueError(f"No fingerprint query available for reaction {reaction_id}")
            stacked = torch.stack(query_scores, dim=0)
            if direction_aggregation == "mean" and stacked.shape[0] > 1:
                scores = stacked.mean(dim=0)
            else:
                scores = stacked.max(dim=0).values

            if ec_scoring == "centroid":
                ec_scores = scores
            else:
                ec_scores = torch.full((len(ec_labels),), float("-inf"), device=device)
                ec_scores.scatter_reduce_(
                    0,
                    ec_indices,
                    scores.index_select(0, ec_protein_indices),
                    reduce="amax",
                    include_self=True,
                )
            if max_rank_columns is not None:
                rank_count = min(max_rank_columns, len(ec_labels))
                ranked_ec_indices = torch.topk(ec_scores, k=rank_count).indices.tolist()
            else:
                ranked_ec_indices = torch.argsort(ec_scores, descending=True).tolist()
            ranked_ecs = [ec_labels[idx] for idx in ranked_ec_indices]
            row = {
                **truth[reaction_id],
                "split_group": metadata["split_group"],
                "split": eval_split,
                "direction_aggregation": direction_aggregation,
                "ec_candidate_splits": ";".join(ec_candidate_splits),
                "ec_scoring": ec_scoring,
                "score_similarity": score_similarity,
            }
            for rank, ec in enumerate(ranked_ecs):
                row[str(rank)] = ec
            rows.append(row)

    care_df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    ensure_dir(output_csv.parent)
    care_df.to_csv(output_csv, index=False)
    return care_df


def run_native_evaluator(
    *,
    checkpoint: str | Path,
    config_path: str | Path,
    output_json: str | Path,
    device: str,
    batch_size: int,
) -> None:
    native_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if native_device == "auto":
        native_device = "cpu"
    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(config_path),
        "--output",
        str(output_json),
        "--device",
        native_device,
        "--batch-size",
        str(batch_size),
    ]
    run_command(command, cwd=BASELINE_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Horizon on Leak-CURBER split data")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval-split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--direction-aggregation",
        choices=["max", "mean", "forward"],
        default="max",
    )
    parser.add_argument(
        "--ec-scoring",
        choices=["max", "centroid"],
        default="max",
        help=(
            "How protein-level target scores become EC scores. 'max' preserves the "
            "converted Horizon metric; 'centroid' matches CARE-style EC prototypes."
        ),
    )
    parser.add_argument(
        "--score-similarity",
        choices=["dot", "cosine"],
        default="dot",
        help="Similarity used for query/reference ranking. CARE uses cosine.",
    )
    parser.add_argument(
        "--results-tag",
        default=None,
        help="Optional suffix for result subdirectories, e.g. test__care_style.",
    )
    parser.add_argument(
        "--max-rank-columns",
        type=int,
        default=None,
        help=(
            "Limit numeric EC rank columns in the CARE CSV. Use 50 to cover all "
            "implemented CARE k-values while avoiding full all-EC output."
        ),
    )
    parser.add_argument(
        "--ec-candidate-split",
        action="append",
        choices=["train", "val", "test"],
        default=None,
        help=(
            "Split(s) used to map proteins to EC candidates. Defaults to train-only "
            "for converted EC retrieval."
        ),
    )
    parser.add_argument(
        "--reuse-native-metrics",
        action="store_true",
        help="Reuse an existing native_metrics.json and rerun only converted EC ranking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_rank_columns is not None and args.max_rank_columns < 1:
        raise ValueError(f"--max-rank-columns must be >= 1, got {args.max_rank_columns}")
    metadata = load_run_metadata(args.split_group, args.runs_root)
    seed_root = seed_results_root(args.split_group, args.seed, args.runs_root).parent
    results_root = seed_results_root(args.split_group, args.seed, args.runs_root)
    ensure_dir(results_root)
    train_metadata_path = seed_train_metadata_path(args.split_group, args.seed, args.runs_root)
    if args.checkpoint is None:
        train_metadata = read_json(train_metadata_path)
        checkpoint = train_metadata["checkpoint"]
    else:
        checkpoint = args.checkpoint

    eval_splits = ["val", "test"] if args.eval_split == "both" else [args.eval_split]
    ec_candidate_splits = tuple(args.ec_candidate_split or ["train"])
    summary = {
        "split_group": args.split_group,
        "seed": int(args.seed),
        "checkpoint": str(checkpoint),
        "results_root": str(results_root),
        "eval_splits": eval_splits,
        "ec_candidate_splits": list(ec_candidate_splits),
        "ec_scoring": args.ec_scoring,
        "score_similarity": args.score_similarity,
        "direction_aggregation": args.direction_aggregation,
        "max_rank_columns": args.max_rank_columns,
        "results_tag": args.results_tag,
        "metrics": {},
        "artifacts": {},
    }
    for split in eval_splits:
        result_name = f"{split}__{args.results_tag}" if args.results_tag else split
        split_root = ensure_dir(results_root / result_name)
        config_path = split_root / "eval_config.yaml"
        native_json = split_root / "native_metrics.json"
        care_csv = split_root / "care_task2_ranked.csv"
        metrics_json = split_root / "metrics.json"

        write_yaml(config_path, build_eval_config(metadata, eval_split=split, seed_root=seed_root))
        if args.reuse_native_metrics and native_json.exists():
            print(f"[emulator_bench] reusing native metrics: {native_json}", flush=True)
        elif args.reuse_native_metrics and args.results_tag:
            source_native_json = results_root / split / "native_metrics.json"
            if source_native_json.exists():
                shutil.copy2(source_native_json, native_json)
                print(
                    f"[emulator_bench] copied native metrics from {source_native_json}",
                    flush=True,
                )
            else:
                run_native_evaluator(
                    checkpoint=checkpoint,
                    config_path=config_path,
                    output_json=native_json,
                    device=args.device,
                    batch_size=args.batch_size,
                )
        else:
            run_native_evaluator(
                checkpoint=checkpoint,
                config_path=config_path,
                output_json=native_json,
                device=args.device,
                batch_size=args.batch_size,
            )
        care_df = write_care_ranked_csv(
            metadata=metadata,
            checkpoint=checkpoint,
            eval_split=split,
            output_csv=care_csv,
            device=args.device,
            batch_size=args.batch_size,
            direction_aggregation=args.direction_aggregation,
            ec_candidate_splits=ec_candidate_splits,
            ec_scoring=args.ec_scoring,
            score_similarity=args.score_similarity,
            max_rank_columns=args.max_rank_columns,
        )
        native_metrics = read_json(native_json)
        metrics = {
            "split_group": args.split_group,
            "run_slug": metadata["run_slug"],
            "seed": int(args.seed),
            "eval_split": split,
            "result_name": result_name,
            "checkpoint": str(checkpoint),
            "ec_candidate_splits": list(ec_candidate_splits),
            "ec_scoring": args.ec_scoring,
            "score_similarity": args.score_similarity,
            "direction_aggregation": args.direction_aggregation,
            "max_rank_columns": args.max_rank_columns,
            "primary_native_horizon": native_metrics,
            "care_task2": compute_care_task2_metrics(care_df),
            "supplemental_ec_ranking": compute_supplemental_ec_metrics(care_df),
            "artifacts": {
                "native_metrics": str(native_json),
                "care_task2_ranked_csv": str(care_csv),
                "eval_config": str(config_path),
            },
        }
        write_json(metrics_json, metrics)
        summary["metrics"][result_name] = metrics
        summary["artifacts"][result_name] = metrics["artifacts"]
        print(f"[emulator_bench] metrics: {metrics_json}", flush=True)

    write_json(results_root / "evaluation_summary.json", summary)


if __name__ == "__main__":
    main()
