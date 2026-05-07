from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .results import (
    choose_f1_thresholds,
    compact_evaluation_summary,
    compute_care_metrics,
    compute_hitec_metrics,
    compute_supplemental_classification_metrics,
    compute_supplemental_ranking_metrics,
    sigmoid,
    write_care_ranked_csv,
)
from .train import HitecLightningModule
from .utils import (
    DEFAULT_RUNS_ROOT,
    ensure_dir,
    load_run_metadata,
    read_json,
    seed_results_root,
    seed_train_metadata_path,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a HITEC checkpoint")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval-split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--rank-limit", type=int, default=50)
    parser.add_argument("--force-thresholds", action="store_true")
    parser.add_argument("--care-results-root", default=None)
    return parser.parse_args()


class PredictionDataset(Dataset):
    def __init__(self, manifest_path, total_outputs, level_starts, output_dims):
        self.manifest_path = Path(manifest_path)
        self.records = pd.read_csv(self.manifest_path)
        self.total_outputs = int(total_outputs)
        self.level_starts = list(level_starts)
        self.output_dims = list(output_dims)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        cached = torch.load(Path(row["token_cache_path"]), map_location="cpu")
        tokens = cached["tokens"].long()
        target = torch.zeros(self.total_outputs, dtype=torch.float32)
        mask = torch.zeros(self.total_outputs, dtype=torch.float32)
        target_indices = json.loads(row["target_indices"])
        observed_levels = json.loads(row["observed_levels"])
        for level in observed_levels:
            start = self.level_starts[int(level)]
            end = start + self.output_dims[int(level)]
            mask[start:end] = 1.0
        for level_indices in target_indices:
            if level_indices:
                target[torch.tensor(level_indices, dtype=torch.long)] = 1.0
        return tokens, target, mask, int(index)


def choose_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(metadata, seed, checkpoint):
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    train_json = seed_train_metadata_path(metadata, seed)
    if not train_json.exists():
        raise FileNotFoundError("Missing train metadata: {}".format(train_json))
    train_metadata = read_json(train_json)
    return Path(train_metadata["checkpoint"])


def load_vocab(metadata):
    vocab_path = Path(metadata["vocab_path"])
    if not vocab_path.exists():
        raise FileNotFoundError("Missing vocab: {}".format(vocab_path))
    return read_json(vocab_path)


def predict_split(model, metadata, split_name, batch_size, num_workers, device):
    dataset = PredictionDataset(
        metadata["manifests"][split_name],
        total_outputs=sum(metadata["output_dims"]),
        level_starts=metadata["level_starts"],
        output_dims=metadata["output_dims"],
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    logits_parts = []
    target_parts = []
    mask_parts = []
    index_parts = []

    model.eval()
    with torch.no_grad():
        for tokens, target, mask, indices in tqdm(
            loader,
            desc="{} predictions".format(split_name),
            unit="batch",
        ):
            tokens = tokens.to(device)
            logits = model(tokens).detach().cpu().numpy()
            logits_parts.append(logits)
            target_parts.append(target.numpy())
            mask_parts.append(mask.numpy())
            index_parts.append(indices.numpy())

    return {
        "logits": np.concatenate(logits_parts, axis=0),
        "targets": np.concatenate(target_parts, axis=0),
        "masks": np.concatenate(mask_parts, axis=0),
        "indices": np.concatenate(index_parts, axis=0),
        "manifest": dataset.records,
    }


def save_prediction_npz(path, arrays):
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        logits=arrays["logits"],
        targets=arrays["targets"],
        masks=arrays["masks"],
        indices=arrays["indices"],
    )


def thresholds_paths(result_root):
    return result_root / "validation_thresholds.npy", result_root / "validation_thresholds.json"


def load_or_select_thresholds(model, metadata, result_root, args, device, vocab, checkpoint):
    thresholds_npy, thresholds_json = thresholds_paths(result_root)
    if thresholds_npy.exists() and thresholds_json.exists() and not args.force_thresholds:
        threshold_metadata = read_json(thresholds_json)
        if threshold_metadata.get("checkpoint") == str(checkpoint):
            thresholds = np.load(thresholds_npy)
            return thresholds, threshold_metadata, None

    val_arrays = predict_split(
        model,
        metadata,
        "val",
        args.batch_size,
        args.num_workers,
        device,
    )
    save_prediction_npz(result_root / "val_logits.npz", val_arrays)
    val_probs = sigmoid(val_arrays["logits"])
    thresholds, stats = choose_f1_thresholds(
        val_probs,
        val_arrays["targets"],
        val_arrays["masks"],
    )
    np.save(thresholds_npy, thresholds)
    threshold_metadata = {
        "source_split": "val",
        "checkpoint": str(checkpoint),
        "thresholds": str(thresholds_npy),
        "stats": stats,
        "classes": int(len(thresholds)),
    }
    write_json(thresholds_json, threshold_metadata)
    return thresholds, threshold_metadata, val_arrays


def external_care_path(care_results_root, metadata, seed, eval_split):
    return Path(care_results_root) / (
        "{}_seed{}_{}_results_df.csv".format(metadata["run_slug"], seed, eval_split)
    )


def evaluate_arrays(arrays, thresholds, metadata, vocab, result_root, split_name, seed, rank_limit, care_results_root):
    probs = sigmoid(arrays["logits"])
    predictions = (probs >= thresholds.reshape(1, -1)).astype(np.int32)
    hitec_metrics = compute_hitec_metrics(
        arrays["logits"],
        arrays["targets"],
        arrays["masks"],
        thresholds,
        vocab,
    )
    care_csv = result_root / "{}_results_df.csv".format(split_name)
    care_df = write_care_ranked_csv(
        arrays["manifest"],
        probs,
        vocab,
        care_csv,
        rank_limit=rank_limit,
    )
    external_csv = None
    if care_results_root:
        external_csv = external_care_path(care_results_root, metadata, seed, split_name)
        write_care_ranked_csv(
            arrays["manifest"],
            probs,
            vocab,
            external_csv,
            rank_limit=rank_limit,
        )

    metrics = {
        "split_group": metadata["split_group"],
        "run_slug": metadata["run_slug"],
        "eval_split": split_name,
        "seed": int(seed),
        "hitec": hitec_metrics,
        "care_task1": compute_care_metrics(care_df),
        "supplemental": {
            "classification": compute_supplemental_classification_metrics(
                arrays["targets"],
                predictions,
                arrays["masks"],
            ),
            "ranking": compute_supplemental_ranking_metrics(care_df),
        },
        "artifacts": {
            "care_ranked_csv": str(care_csv),
            "external_care_ranked_csv": str(external_csv) if external_csv else None,
            "logits_npz": str(result_root / "{}_logits.npz".format(split_name)),
        },
    }
    write_json(result_root / "{}_metrics.json".format(split_name), metrics)
    return metrics


def main():
    args = parse_args()
    metadata = load_run_metadata(args.split_group, args.runs_root)
    vocab = load_vocab(metadata)
    result_root = ensure_dir(seed_results_root(metadata, args.seed))
    checkpoint = resolve_checkpoint(metadata, args.seed, args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError("Missing checkpoint: {}".format(checkpoint))

    device = choose_device(args.device)
    model = HitecLightningModule.load_from_checkpoint(str(checkpoint), map_location=device)
    model.to(device)

    thresholds, threshold_metadata, cached_val_arrays = load_or_select_thresholds(
        model,
        metadata,
        result_root,
        args,
        device,
        vocab,
        checkpoint,
    )

    eval_splits = ["val", "test"] if args.eval_split == "both" else [args.eval_split]
    metrics = []
    for split_name in eval_splits:
        if split_name == "val" and cached_val_arrays is not None:
            arrays = cached_val_arrays
        else:
            arrays = predict_split(
                model,
                metadata,
                split_name,
                args.batch_size,
                args.num_workers,
                device,
            )
            save_prediction_npz(result_root / "{}_logits.npz".format(split_name), arrays)
        metrics.append(
            evaluate_arrays(
                arrays,
                thresholds,
                metadata,
                vocab,
                result_root,
                split_name,
                args.seed,
                args.rank_limit,
                args.care_results_root,
            )
        )

    write_json(
        result_root / "evaluation_summary.json",
        compact_evaluation_summary(
            metrics,
            checkpoint=str(checkpoint),
            thresholds=threshold_metadata,
        ),
    )
    print("[emulator_bench] evaluation summary: {}".format(result_root / "evaluation_summary.json"), flush=True)


if __name__ == "__main__":
    main()
