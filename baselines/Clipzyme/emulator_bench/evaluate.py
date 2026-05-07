from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

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
)


BASE_EVAL_CONFIG = BASELINE_ROOT / "configs" / "train" / "clip_esm.json"


def _one(value):
    return [value]


def _abs(path: str | Path) -> str:
    return str(Path(path).resolve())


def build_eval_config(
    metadata: dict,
    *,
    checkpoint: str | Path,
    eval_split: str,
    seed_root: Path,
    available_gpus: str,
    num_workers: int,
) -> dict:
    import json

    base = json.loads(BASE_EVAL_CONFIG.read_text())
    config = copy.deepcopy(base)
    config["script"] = "main"
    config["available_gpus"] = [available_gpus]
    hp = config["cartesian_hyperparams"]
    torch_hub_dir = ensure_dir(Path(metadata["cache_root"]) / "torch_hub")
    hp.update(
        {
            "dataset_file_path": _one(_abs(metadata["clipzyme_files"]["dataset_json"])),
            "ec2uniprot_path": _one(_abs(metadata["clipzyme_files"]["ec2uniprot"])),
            "uniprot2sequence_path": _one(_abs(metadata["clipzyme_files"]["uniprot2sequence"])),
            "from_checkpoint": _one(True),
            "checkpoint_path": _one(_abs(checkpoint)),
            "remove_duplicate_reactions": _one(False),
            "use_mapped_reaction": _one(True),
            "assign_splits": _one(False),
            "seed": _one(0),
            "max_protein_length": _one(int(metadata["max_sequence_length"])),
            "checkpoint_dir": _one(_abs(seed_root / "eval_checkpoints")),
            "pretrained_hub_dir": _one(_abs(torch_hub_dir)),
            "logger_name": _one("tensorboard"),
            "logger_tags": _one(f"clipzyme emulator eval {metadata['run_slug']}"),
            "project_name": _one("emulator_clipzyme"),
            "workspace": _one("emulator"),
            "inference_dir": _one(_abs(seed_root / "eval_inference")),
            "gpus": _one(1),
            "num_workers": _one(int(num_workers)),
            "train": _one(False),
            "dev": _one(eval_split == "val"),
            "test": _one(eval_split == "test"),
            "save_predictions": _one(False),
            "save_hiddens": _one(False),
            "lr": _one(1e-4),
            "weight_decay": _one(0.05),
        }
    )
    hp.pop("dataset_cache_path", None)
    return config


def _load_manifest(metadata: dict, split: str) -> pd.DataFrame:
    return pd.read_csv(metadata["manifests"][split])


def _truth_by_reaction(metadata: dict, split: str) -> dict[str, dict]:
    frame = _load_manifest(metadata, split)
    truth = {}
    for reaction_id, group in frame.groupby("reaction_id", sort=True):
        labels = sorted(set(group["ec_number"].astype(str)))
        reaction = str(group["reaction_smiles"].iloc[0])
        mapped = str(group["mapped_reaction"].iloc[0])
        truth[str(reaction_id)] = {
            "Reaction": reaction,
            "Mapped Reaction": mapped,
            "EC number": ";".join(labels),
            "Reaction Text": reaction,
            "EC3": ";".join(sorted({".".join(ec.split(".")[:3]) for ec in labels})),
            "EC2": ";".join(sorted({".".join(ec.split(".")[:2]) for ec in labels})),
            "EC1": ";".join(sorted({ec.split(".")[0] for ec in labels})),
            "Duplicated EC": int(len(labels) > 1),
            "members EC": int(len(labels)),
            "Reactions with a single EC": int(len(labels) == 1),
            "reaction_id": str(reaction_id),
            "split": split,
        }
    return truth


def _reference_proteins(metadata: dict) -> list[dict]:
    by_protein: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        frame = _load_manifest(metadata, split)
        for row in frame.itertuples(index=False):
            protein_id = str(row.protein_id)
            record = by_protein.setdefault(
                protein_id,
                {"protein_id": protein_id, "sequence": str(row.sequence), "ecs": set()},
            )
            record["ecs"].add(str(row.ec_number))
    records = []
    for record in by_protein.values():
        records.append(
            {
                "protein_id": record["protein_id"],
                "sequence": record["sequence"],
                "ecs": sorted(record["ecs"]),
            }
        )
    return sorted(records, key=lambda item: item["protein_id"])


def _move_to_device(value, device: str):
    if hasattr(value, "to") and not isinstance(value, (str, bytes)):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(val, device) for key, val in value.items()}
    if isinstance(value, list):
        return [_move_to_device(val, device) for val in value]
    return value


def _load_clipzyme_inference_model(checkpoint: str | Path, device: str):
    from argparse import Namespace

    import torch
    from clipzyme.lightning.clipzyme import CLIPZyme

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args = Namespace(
        checkpoint_path=str(checkpoint),
        use_as_protein_encoder=False,
        use_as_reaction_encoder=False,
        save_hiddens=False,
        save_predictions=False,
        inference_dir="emulator_bench/inference",
    )
    wrapper = CLIPZyme(args=args)
    wrapper.model.to(device)
    wrapper.model.eval()
    return wrapper, device


def _encode_protein_ec_centers(
    wrapper,
    *,
    proteins: list[dict],
    device: str,
    batch_size: int,
) -> tuple[list[str], object]:
    import torch
    from tqdm import tqdm
    from clipzyme.utils.loading import default_collate

    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
    with torch.no_grad():
        for start in tqdm(range(0, len(proteins), batch_size), desc="CARE protein centers"):
            batch_records = proteins[start : start + batch_size]
            batch = default_collate(
                [
                    {"sequence": record["sequence"], "sample_id": record["protein_id"]}
                    for record in batch_records
                ]
            )
            batch = _move_to_device(batch, device)
            protein_hiddens = wrapper.extract_protein_features(batch).detach().cpu()
            for hidden, record in zip(protein_hiddens, batch_records):
                for ec in record["ecs"]:
                    if ec not in sums:
                        sums[ec] = torch.zeros_like(hidden)
                    sums[ec] += hidden
                    counts[ec] += 1
    ecs = sorted(sums)
    centers = torch.stack([sums[ec] / max(counts[ec], 1) for ec in ecs], dim=0)
    centers = centers / centers.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return ecs, centers.to(device)


def _encode_reactions(
    wrapper,
    *,
    reactions: list[dict],
    device: str,
    batch_size: int,
):
    import torch
    from tqdm import tqdm
    from clipzyme.utils.loading import default_collate
    from clipzyme.utils.screening import process_mapped_reaction

    encoded = []
    ids = []
    with torch.no_grad():
        for start in tqdm(range(0, len(reactions), batch_size), desc="CARE reactions"):
            batch_records = reactions[start : start + batch_size]
            batch_items = []
            for record in batch_records:
                reactants, products = process_mapped_reaction(record["mapped_reaction"])
                batch_items.append(
                    {
                        "reactants": reactants,
                        "products": products,
                        "sample_id": record["reaction_id"],
                    }
                )
            batch = default_collate(batch_items)
            batch = _move_to_device(batch, device)
            hiddens = wrapper.extract_reaction_features(batch).detach().cpu()
            encoded.append(hiddens)
            ids.extend([record["reaction_id"] for record in batch_records])
    return ids, torch.cat(encoded, dim=0).to(device)


def write_care_ranked_csv(
    *,
    metadata: dict,
    checkpoint: str | Path,
    eval_split: str,
    output_csv: str | Path,
    device: str,
    batch_size: int,
    max_ranks: int | None,
) -> pd.DataFrame:
    import torch

    wrapper, device = _load_clipzyme_inference_model(checkpoint, device)
    proteins = _reference_proteins(metadata)
    if not proteins:
        raise ValueError("No reference proteins available for CARE ranking")
    ecs, ec_centers = _encode_protein_ec_centers(
        wrapper,
        proteins=proteins,
        device=device,
        batch_size=batch_size,
    )

    truth = _truth_by_reaction(metadata, eval_split)
    frame = _load_manifest(metadata, eval_split)
    reaction_records = (
        frame.loc[:, ["reaction_id", "mapped_reaction"]]
        .drop_duplicates("reaction_id")
        .sort_values("reaction_id")
        .to_dict("records")
    )
    reaction_ids, reaction_hiddens = _encode_reactions(
        wrapper,
        reactions=reaction_records,
        device=device,
        batch_size=batch_size,
    )
    max_ranks = len(ecs) if max_ranks is None else min(max_ranks, len(ecs))
    rows = []
    with torch.no_grad():
        scores = torch.matmul(reaction_hiddens, ec_centers.T).detach().cpu()
    for idx, reaction_id in enumerate(reaction_ids):
        order = sorted(
            range(len(ecs)),
            key=lambda ec_idx: (-float(scores[idx, ec_idx]), ecs[ec_idx]),
        )
        ranked_ecs = [ecs[ec_idx] for ec_idx in order[:max_ranks]]
        row = {
            **truth[reaction_id],
            "split_group": metadata["split_group"],
            "candidate_ec_count": int(len(ecs)),
            "reference_protein_count": int(len(proteins)),
        }
        for rank, ec in enumerate(ranked_ecs):
            row[str(rank)] = ec
        rows.append(row)

    care_df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    ensure_dir(output_csv.parent)
    care_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return care_df


def _collect_tensorboard_scalars(log_dir: str | Path) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}
    scalars = {}
    for event_file in Path(log_dir).rglob("events.out.tfevents*"):
        try:
            accumulator = EventAccumulator(str(event_file))
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                events = accumulator.Scalars(tag)
                if events:
                    scalars[tag] = float(events[-1].value)
        except Exception:
            continue
    return scalars


def _collect_native_log_metrics(log_dir: str | Path) -> dict[str, float]:
    metrics = {}
    for log_file in sorted(Path(log_dir).glob("*.txt"), key=lambda path: path.stat().st_mtime):
        for line in log_file.read_text(errors="replace").splitlines():
            if "│" not in line:
                continue
            parts = [part.strip() for part in line.split("│") if part.strip()]
            if len(parts) != 2:
                continue
            name, value = parts
            if name.lower() in {"test metric", "dataloader 0"}:
                continue
            try:
                metrics[name] = float(value)
            except ValueError:
                continue
    return metrics


def run_native_eval(
    *,
    metadata: dict,
    checkpoint: str | Path,
    eval_split: str,
    seed_root: Path,
    available_gpus: str,
    num_workers: int,
) -> dict:
    config = build_eval_config(
        metadata,
        checkpoint=checkpoint,
        eval_split=eval_split,
        seed_root=seed_root,
        available_gpus=available_gpus,
        num_workers=num_workers,
    )
    config_path = seed_root / "configs" / f"eval_{eval_split}_config.json"
    write_json(config_path, config)
    log_dir = seed_root / "eval_logs" / eval_split
    ensure_dir(log_dir)
    command = [sys.executable, "scripts/dispatcher.py", "-c", str(config_path), "-l", str(log_dir)]
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    run_command(command, cwd=BASELINE_ROOT, env=env)
    return {
        "config": str(config_path),
        "log_dir": str(log_dir),
        "metrics": _collect_native_log_metrics(log_dir),
        "tensorboard_scalars": _collect_tensorboard_scalars(log_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Clipzyme Leak-CURBER checkpoint")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval-split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--available-gpus", default="0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-ranks", type=int, default=None)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-care", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_run_metadata(args.split_group, args.runs_root)
    train_metadata = read_json(seed_train_metadata_path(args.split_group, args.seed, args.runs_root))
    checkpoint = Path(args.checkpoint or train_metadata["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    seed_root = Path(train_metadata["seed_run_root"])
    ensure_dir(seed_root / "configs")
    results_root = seed_results_root(args.split_group, args.seed, args.runs_root)
    ensure_dir(results_root)
    eval_splits = ["val", "test"] if args.eval_split == "both" else [args.eval_split]
    summary = {}
    for eval_split in eval_splits:
        split_root = ensure_dir(results_root / eval_split)
        native = {}
        if not args.skip_native:
            native = run_native_eval(
                metadata=metadata,
                checkpoint=checkpoint,
                eval_split=eval_split,
                seed_root=seed_root,
                available_gpus=args.available_gpus,
                num_workers=args.num_workers,
            )
            write_json(split_root / "native_metrics.json", native)

        care_metrics = {}
        supplemental_metrics = {}
        care_csv = None
        if not args.skip_care:
            care_csv = split_root / "care_task2_ranked_ec.csv"
            care_df = write_care_ranked_csv(
                metadata=metadata,
                checkpoint=checkpoint,
                eval_split=eval_split,
                output_csv=care_csv,
                device=args.device,
                batch_size=args.batch_size,
                max_ranks=args.max_ranks,
            )
            care_metrics = compute_care_task2_metrics(care_df)
            supplemental_metrics = compute_supplemental_ec_metrics(care_df)
            write_json(split_root / "care_task2_metrics.json", care_metrics)
            write_json(split_root / "supplemental_metrics.json", supplemental_metrics)

        summary[eval_split] = {
            "checkpoint": str(checkpoint),
            "native": native,
            "care_ranked_csv": str(care_csv) if care_csv is not None else None,
            "care_metrics": care_metrics,
            "supplemental_metrics": supplemental_metrics,
        }

    write_json(results_root / "evaluation_summary.json", summary)
    print(f"[emulator_bench] evaluation complete: {results_root}", flush=True)


if __name__ == "__main__":
    main()
