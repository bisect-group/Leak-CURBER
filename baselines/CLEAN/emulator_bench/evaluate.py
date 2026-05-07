from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .results import (
    compact_evaluation_summary,
    compute_care_metrics,
    compute_supplemental_classification_metrics,
    compute_supplemental_ranking_metrics,
    parse_clean_maxsep_csv,
    write_care_ranked_csv,
)
from .utils import (
    APP_DIR,
    DEFAULT_RUNS_ROOT,
    add_clean_to_path,
    ensure_dir,
    load_run_metadata,
    pushd,
    read_json,
    seed_results_root,
    seed_run_root,
    seed_train_metadata_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CLEAN checkpoint on Leak-CURBER splits")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--eval-split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--care-results-root", default=None)
    parser.add_argument("--seed", type=int, default=1234, help="Random seed used by the model")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(
    metadata: dict,
    model_name: str | None,
    checkpoint: str | None,
    seed: int,
) -> tuple[str, Path]:
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (APP_DIR / checkpoint_path).resolve()
        resolved_model_name = model_name or checkpoint_path.stem
        return resolved_model_name, checkpoint_path

    canonical_train_json = seed_train_metadata_path(metadata, seed)
    if canonical_train_json.exists():
        train_metadata = read_json(canonical_train_json)
        return train_metadata["model_name"], Path(train_metadata["checkpoint"])

    train_json = Path(metadata["run_root"]) / f"train_seed{seed}.json"
    if train_json.exists():
        train_metadata = read_json(train_json)
        return train_metadata["model_name"], Path(train_metadata["checkpoint"])

    train_data = metadata["clean_data"]["train"]
    resolved_model_name = model_name or f"{train_data}_triplet_seed{seed}"
    return resolved_model_name, APP_DIR / "data" / "model" / f"{resolved_model_name}.pth"


def external_care_ranked_csv(
    care_results_root: str | Path,
    metadata: dict,
    seed: int,
    eval_split: str,
) -> Path:
    return Path(care_results_root) / (
        f"{metadata['run_slug']}_seed{seed}_{eval_split}_results_df.csv"
    )


def evaluate_split(
    *,
    metadata: dict,
    eval_split: str,
    model_name: str,
    checkpoint: Path,
    seed: int,
    seed_run_root_path: Path,
    result_root: Path,
    hidden_dim: int,
    out_dim: int,
    device: torch.device,
    care_results_root: str | None,
) -> dict:
    add_clean_to_path()
    from CLEAN.distance_map import get_dist_map_test
    from CLEAN.evaluate import (
        get_eval_metrics,
        get_pred_labels,
        get_pred_probs,
        get_true_labels,
        write_max_sep_choices,
    )
    from CLEAN.model import LayerNormNet
    from CLEAN.utils import esm_embedding, get_ec_id_dict, model_embedding_test

    train_data = metadata["clean_data"]["train"]
    eval_data = metadata["clean_data"][eval_split]
    dtype = torch.float32

    with pushd(APP_DIR):
        id_ec_train, ec_id_dict_train = get_ec_id_dict(f"./data/{train_data}.csv")
        id_ec_eval, _ = get_ec_id_dict(f"./data/{eval_data}.csv")
        model = LayerNormNet(hidden_dim, out_dim, device, dtype)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        emb_eval = model_embedding_test(id_ec_eval, model, device, dtype)
        eval_dist = get_dist_map_test(
            emb_train, emb_eval, ec_id_dict_train, id_ec_eval, device, dtype
        )
        eval_df = pd.DataFrame.from_dict(eval_dist)

        native_prefix = f"results/emulator_bench/{metadata['run_slug']}/seed{seed}/{eval_split}"
        ensure_dir(APP_DIR / Path(native_prefix).parent)
        write_max_sep_choices(eval_df, native_prefix)
        native_maxsep_csv = APP_DIR / f"{native_prefix}_maxsep.csv"
        pred_label = get_pred_labels(native_prefix, pred_type="_maxsep")
        pred_probs = get_pred_probs(native_prefix, pred_type="_maxsep")
        true_label, all_label = get_true_labels(f"./data/{eval_data}")
        native_metric_values = get_eval_metrics(pred_label, pred_probs, true_label, all_label)

    care_csv = result_root / f"{eval_split}_results_df.csv"
    clean_eval_csv = APP_DIR / "data" / f"{eval_data}.csv"
    care_df = write_care_ranked_csv(eval_df, clean_eval_csv, care_csv)
    parsed_maxsep_csv = result_root / f"{eval_split}_maxsep_df.csv"
    parse_clean_maxsep_csv(native_maxsep_csv).to_csv(parsed_maxsep_csv, index=False)

    if care_results_root:
        external_csv = external_care_ranked_csv(
            care_results_root,
            metadata,
            seed,
            eval_split,
        )
        write_care_ranked_csv(eval_df, clean_eval_csv, external_csv)
    else:
        external_csv = None

    metrics = {
        "native_clean": {
            "weighted_precision": float(native_metric_values[0]),
            "weighted_recall": float(native_metric_values[1]),
            "weighted_f1": float(native_metric_values[2]),
            "weighted_auc": float(native_metric_values[3]),
            "exact_match_accuracy": float(native_metric_values[4]),
        },
        "care_task1": compute_care_metrics(care_df),
        "supplemental": {
            "classification": compute_supplemental_classification_metrics(true_label, pred_label),
            "ranking": compute_supplemental_ranking_metrics(care_df),
        },
        "artifacts": {
            "native_maxsep_csv": str(native_maxsep_csv),
            "parsed_maxsep_csv": str(parsed_maxsep_csv),
            "care_ranked_csv": str(care_csv),
            "external_care_ranked_csv": str(external_csv) if external_csv else None,
        },
        "eval_split": eval_split,
        "split_group": metadata["split_group"],
        "run_slug": metadata["run_slug"],
        "train_data": train_data,
        "eval_data": eval_data,
        "model_name": model_name,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "seed_run_root": str(seed_run_root_path),
    }
    metrics_path = result_root / f"{eval_split}_metrics.json"
    write_json(metrics_path, metrics)
    print(f"[emulator_bench] {eval_split} metrics: {metrics_path}", flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    metadata = load_run_metadata(args.split_group, args.runs_root)
    model_name, checkpoint = resolve_checkpoint(
        metadata,
        args.model_name,
        args.checkpoint,
        args.seed,
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    device = choose_device(args.device)
    seed_run_root_path = seed_run_root(metadata, args.seed)
    result_root = ensure_dir(seed_results_root(metadata, args.seed))

    eval_splits = ["val", "test"] if args.eval_split == "both" else [args.eval_split]
    all_metrics = [
        evaluate_split(
            metadata=metadata,
            eval_split=eval_split,
            model_name=model_name,
            checkpoint=checkpoint,
            seed=args.seed,
            seed_run_root_path=seed_run_root_path,
            result_root=result_root,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            device=device,
            care_results_root=args.care_results_root,
        )
        for eval_split in eval_splits
    ]
    write_json(
        result_root / "evaluation_summary.json",
        compact_evaluation_summary(
            all_metrics,
            checkpoint=str(checkpoint),
            model_name=model_name,
        ),
    )


if __name__ == "__main__":
    main()
