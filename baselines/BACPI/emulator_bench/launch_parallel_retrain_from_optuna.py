import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import optuna
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import AFFINITY_VALUE_TYPES, DEFAULT_SPLIT_GROUPS, discover_split_jobs, normalize_threshold_args, resolve_base_dir
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings


TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def load_best_hparams(args):
    if args.hparams_json:
        with open(args.hparams_json, "r") as handle:
            payload = json.load(handle)
        if "best_hparams" in payload:
            return payload["best_hparams"]
        if "defaults" in payload:
            return payload["defaults"]
        return payload

    if not args.storage:
        raise ValueError("Provide either --hparams_json or --storage.")
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    return dict(study.best_params)


def resolve_training_hparams(raw_hparams, args):
    def choose(key, fallback):
        override = getattr(args, key)
        if override is not None:
            return override
        return raw_hparams.get(key, fallback)

    return {
        "batch_size": int(choose("batch_size", 16)),
        "lr": float(choose("lr", 5e-4)),
        "weight_decay": float(choose("weight_decay", 0.0)),
        "step_size": int(choose("step_size", 10)),
        "gamma": float(choose("gamma", 0.5)),
        "grad_clip": float(choose("grad_clip", 0.0)),
        "patience": int(choose("patience", 0)),
        "scheduler": str(choose("scheduler", "step")),
        "min_lr": float(choose("min_lr", 1e-6)),
        "min_delta": float(choose("min_delta", 0.0)),
        "amsgrad": bool(args.amsgrad if args.amsgrad is not None else raw_hparams.get("amsgrad", True)),
    }


def normalize_value_types(value_types):
    if value_types is None:
        return ["ki"]
    if isinstance(value_types, str):
        raw_values = [value_types]
    else:
        raw_values = list(value_types)
    normalized = []
    seen = set()
    for value in raw_values:
        key = str(value).strip().lower()
        if not key:
            continue
        if key not in AFFINITY_VALUE_TYPES:
            raise ValueError("Unsupported value_type `%s`. Expected one of %s." % (value, ", ".join(AFFINITY_VALUE_TYPES)))
        if key not in seen:
            seen.add(key)
            normalized.append(key)
    if not normalized:
        raise ValueError("At least one value type is required.")
    return normalized


def resolve_base_dir_for_value(args, value_type):
    if args.base_dir:
        candidate = Path(args.base_dir).expanduser().resolve()
        candidate_child = candidate / value_type
        if candidate_child.exists():
            return candidate_child
        if len(args.value_type) > 1:
            raise ValueError(
                "When passing multiple --value_type values, --base_dir must point to the BACPI root "
                "containing per-value subdirectories, or be omitted."
            )
        return candidate
    return resolve_base_dir(None, args.base_root, value_type)


def resolve_value_path_template(path_value, value_type):
    if path_value is None:
        return None
    rendered = str(path_value).format(value_type=value_type)
    return Path(rendered).expanduser().resolve()


def resolve_cache_root(args):
    if args.base_dir:
        candidate = Path(args.base_dir).expanduser().resolve()
        if len(args.value_type) == 1:
            child = candidate / args.value_type[0]
            if child.exists():
                return candidate
            return candidate.parent if candidate.name == args.value_type[0] else candidate
        return candidate
    return Path(args.base_root or Path(resolve_base_dir(None, None, "ki")).parents[0]).expanduser().resolve()


def uses_shared_embeddings(args):
    if args.embeddings_dir:
        return "{value_type}" not in str(args.embeddings_dir)
    return len(args.value_type) > 1


def resolve_shared_embeddings_dir(args, cache_root):
    if args.embeddings_dir:
        return Path(args.embeddings_dir).expanduser().resolve()
    return cache_root / "embeddings_shared"


def resolve_embeddings_dir_for_value(args, base_dir, value_type, cache_root):
    if args.embeddings_dir and "{value_type}" in str(args.embeddings_dir):
        return resolve_value_path_template(args.embeddings_dir, value_type)
    if uses_shared_embeddings(args):
        return resolve_shared_embeddings_dir(args, cache_root)
    if args.embeddings_dir:
        return Path(args.embeddings_dir).expanduser().resolve()
    return base_dir / "embeddings"


def default_output_root(args, base_dir, value_type):
    if args.output_root:
        rendered = resolve_value_path_template(args.output_root, value_type)
        if len(args.value_type) > 1 and "{value_type}" not in str(args.output_root):
            return rendered / value_type
        return rendered
    if args.hparams_json:
        stem = Path(args.hparams_json).stem
        return base_dir / ("%s_retrain_runs" % stem)
    return base_dir / "bacpi_optuna_best_runs"


def build_value_type_configs(args):
    cache_root = resolve_cache_root(args)
    configs = []
    for value_type in args.value_type:
        base_dir = resolve_base_dir_for_value(args, value_type)
        embeddings_dir = resolve_embeddings_dir_for_value(args, base_dir, value_type, cache_root)
        output_root = default_output_root(args, base_dir, value_type)
        configs.append(
            {
                "value_type": value_type,
                "base_dir": base_dir,
                "embeddings_dir": embeddings_dir,
                "output_root": output_root,
            }
        )
    return configs, cache_root


def build_experiments(value_type, jobs, seeds, output_root, embeddings_dir):
    experiments = []
    for job in jobs:
        for seed in seeds:
            run_dir = output_root / job["split_group"] / job["split_name"] / ("seed_%s" % seed)
            experiments.append(
                {
                    "value_type": value_type,
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "difficulty": job["difficulty"],
                    "train_path": job["train_path"],
                    "val_path": job["val_path"],
                    "test_path": job["test_path"],
                    "embeddings_dir": embeddings_dir,
                    "seed": int(seed),
                    "run_dir": run_dir,
                }
            )
    return experiments


def train_command(exp, args, hparams, device):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_path",
        exp["train_path"],
        "--val_path",
        exp["val_path"],
        "--test_path",
        exp["test_path"],
        "--embeddings_dir",
        str(exp["embeddings_dir"]),
        "--out_dir",
        str(exp["run_dir"]),
        "--task_name",
        "%s_%s_%s_seed%s" % (exp["value_type"], exp["split_group"], exp["split_name"], exp["seed"]),
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--target_col",
        args.target_col,
        "--lr",
        str(hparams["lr"]),
        "--weight_decay",
        str(hparams["weight_decay"]),
        "--step_size",
        str(hparams["step_size"]),
        "--gamma",
        str(hparams["gamma"]),
        "--batch_size",
        str(hparams["batch_size"]),
        "--num_epochs",
        str(args.num_epochs),
        "--scheduler",
        hparams["scheduler"],
        "--min_lr",
        str(hparams["min_lr"]),
        "--grad_clip",
        str(hparams["grad_clip"]),
        "--patience",
        str(hparams["patience"]),
        "--min_delta",
        str(hparams["min_delta"]),
        "--val_every",
        str(args.val_every),
        "--gat_dim",
        str(args.gat_dim),
        "--num_head",
        str(args.num_head),
        "--dropout",
        str(args.dropout),
        "--alpha",
        str(args.alpha),
        "--comp_dim",
        str(args.comp_dim),
        "--prot_dim",
        str(args.prot_dim),
        "--latent_dim",
        str(args.latent_dim),
        "--window",
        str(args.window),
        "--layer_cnn",
        str(args.layer_cnn),
        "--layer_out",
        str(args.layer_out),
        "--device",
        device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--bucket_size_multiplier",
        str(args.bucket_size_multiplier),
        "--pad_to_multiple",
        str(args.pad_to_multiple),
        "--seed",
        str(exp["seed"]),
    ]
    if args.skip_smiles_validity_check:
        cmd.append("--skip_smiles_validity_check")
    if args.pin_memory:
        cmd.append("--pin_memory")
    else:
        cmd.append("--no_pin_memory")
    if args.persistent_workers:
        cmd.append("--persistent_workers")
    else:
        cmd.append("--no_persistent_workers")
    if args.preload_features:
        cmd.append("--preload_features")
    else:
        cmd.append("--no_preload_features")
    if args.torch_compile:
        cmd.append("--torch_compile")
    if hparams["amsgrad"]:
        cmd.append("--amsgrad")
    else:
        cmd.append("--no_amsgrad")
    return cmd


def run_experiment(exp, args, hparams, gpu_id):
    exp["run_dir"].mkdir(parents=True, exist_ok=True)
    metric_path = exp["run_dir"] / "final_results_test.csv"
    if metric_path.exists() and not args.overwrite:
        print(
            "[GPU %s] Skipping existing run %s/%s seed=%s"
            % (gpu_id, exp["value_type"], exp["split_name"], exp["seed"]),
            flush=True,
        )
        return {
            "status": "skipped_exists",
            "gpu_id": str(gpu_id),
            "value_type": exp["value_type"],
            "run_dir": str(exp["run_dir"]),
            "split_group": exp["split_group"],
            "split_name": exp["split_name"],
            "difficulty": exp["difficulty"],
            "seed": exp["seed"],
        }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    device = "cuda:0" if args.device.startswith("cuda") else args.device
    cmd = train_command(exp, args, hparams, device)
    print(
        "[GPU %s] Starting %s | split_group=%s split_name=%s seed=%s"
        % (gpu_id, exp["value_type"], exp["split_group"], exp["split_name"], exp["seed"]),
        flush=True,
    )
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    print(
        "[GPU %s] Finished %s | split_group=%s split_name=%s seed=%s"
        % (gpu_id, exp["value_type"], exp["split_group"], exp["split_name"], exp["seed"]),
        flush=True,
    )
    return {
        "status": "completed",
        "gpu_id": str(gpu_id),
        "value_type": exp["value_type"],
        "run_dir": str(exp["run_dir"]),
        "split_group": exp["split_group"],
        "split_name": exp["split_name"],
        "difficulty": exp["difficulty"],
        "seed": exp["seed"],
    }


def run_parallel(experiments, args, hparams):
    work_queue = queue.Queue()
    for exp in experiments:
        work_queue.put(exp)

    results = []
    result_lock = threading.Lock()

    def worker(gpu_id, slot_index):
        while True:
            try:
                exp = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_experiment(exp, args, hparams, gpu_id)
                result["slot_index"] = int(slot_index)
            except Exception as exc:
                print(
                    "[GPU %s] Failed %s | split_group=%s split_name=%s seed=%s | %s"
                    % (gpu_id, exp["value_type"], exp["split_group"], exp["split_name"], exp["seed"], exc),
                    flush=True,
                )
                result = {
                    "status": "failed",
                    "gpu_id": str(gpu_id),
                    "slot_index": int(slot_index),
                    "value_type": exp["value_type"],
                    "run_dir": str(exp["run_dir"]),
                    "split_group": exp["split_group"],
                    "split_name": exp["split_name"],
                    "difficulty": exp["difficulty"],
                    "seed": exp["seed"],
                    "error": str(exc),
                }
            with result_lock:
                results.append(result)
            work_queue.task_done()

    threads = []
    for gpu_id in args.gpus:
        for slot_index in range(args.trials_per_gpu):
            thread = threading.Thread(target=worker, args=(str(gpu_id), slot_index), daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()
    return results


def main():
    parser = argparse.ArgumentParser(description="Retrain BACPI split jobs in parallel from Optuna or JSON hyperparameters.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--trials_per_gpu", type=int, default=1)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--base_root", type=str, default=None)
    parser.add_argument("--value_type", nargs="+", default=["ki"])
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--bucket_size_multiplier", type=int, default=50)
    parser.add_argument("--pad_to_multiple", type=int, default=1)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--preload_features", dest="preload_features", action="store_true")
    parser.add_argument("--no_preload_features", dest="preload_features", action="store_false")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--cache_overwrite_vocabs", action="store_true")
    parser.add_argument("--clean_splits_in_place", dest="clean_splits_in_place", action="store_true")
    parser.add_argument("--no_clean_splits_in_place", dest="clean_splits_in_place", action="store_false")
    parser.add_argument("--skip_smiles_validity_check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--step_size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--min_delta", type=float, default=None)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--amsgrad", type=str, default=None)
    parser.add_argument("--study_name", type=str, default="bacpi_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--hparams_json", type=str, default=None)

    parser.add_argument("--gat_dim", type=int, default=50)
    parser.add_argument("--num_head", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--comp_dim", type=int, default=80)
    parser.add_argument("--prot_dim", type=int, default=80)
    parser.add_argument("--latent_dim", type=int, default=80)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--layer_cnn", type=int, default=3)
    parser.add_argument("--layer_out", type=int, default=3)
    parser.set_defaults(pin_memory=True, persistent_workers=True, preload_features=True, clean_splits_in_place=True)
    args = parser.parse_args()

    if args.trials_per_gpu <= 0:
        raise ValueError("--trials_per_gpu must be a positive integer")

    args.value_type = normalize_value_types(args.value_type)
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    raw_hparams = load_best_hparams(args)
    if isinstance(args.amsgrad, str):
        args.amsgrad = args.amsgrad.strip().lower() == "true"
    hparams = resolve_training_hparams(raw_hparams, args)
    value_type_configs, cache_root = build_value_type_configs(args)
    print("Resolved retraining hyperparameters: %s" % json.dumps(hparams, sort_keys=True), flush=True)

    if uses_shared_embeddings(args):
        cache_args = SimpleNamespace(**vars(args))
        cache_args.base_dir = str(cache_root)
        cache_args.embeddings_dir = str(resolve_shared_embeddings_dir(args, cache_root))
        print("Preparing shared cache at %s" % cache_args.embeddings_dir, flush=True)
        maybe_cache_embeddings(cache_args)
    else:
        for config in value_type_configs:
            cache_args = SimpleNamespace(**vars(args))
            cache_args.base_dir = config["base_dir"]
            cache_args.embeddings_dir = config["embeddings_dir"]
            cache_args.value_type = [config["value_type"]]
            print("Preparing cache for %s at %s" % (config["value_type"], config["embeddings_dir"]), flush=True)
            maybe_cache_embeddings(cache_args)

    experiments = []
    for config in value_type_configs:
        jobs = discover_split_jobs(config["base_dir"], split_groups=args.split_groups, thresholds=args.thresholds)
        if not jobs:
            raise FileNotFoundError("No split jobs found in %s" % config["base_dir"])
        print(
            "Discovered %s split jobs for %s under %s"
            % (len(jobs), config["value_type"], config["base_dir"]),
            flush=True,
        )
        config["output_root"].mkdir(parents=True, exist_ok=True)
        experiments.extend(
            build_experiments(
                config["value_type"],
                jobs,
                args.seeds,
                config["output_root"],
                config["embeddings_dir"],
            )
        )
    print(
        "Dispatching %s experiments across %s GPU worker slots"
        % (len(experiments), len(args.gpus) * args.trials_per_gpu),
        flush=True,
    )
    results = run_parallel(experiments, args, hparams)
    failed_results = [result for result in results if result.get("status") == "failed"]
    if failed_results:
        raise RuntimeError(
            "One or more retraining jobs failed. First failure: %s"
            % json.dumps(failed_results[0], sort_keys=True)
        )
    for config in value_type_configs:
        output_root = config["output_root"]
        value_results = [result for result in results if result.get("value_type") == config["value_type"]]
        results_path = output_root / "dispatch_results.csv"
        pd.DataFrame(value_results).to_csv(results_path, index=False)

        metric_rows = []
        for exp in experiments:
            if exp["value_type"] != config["value_type"]:
                continue
            metric_path = exp["run_dir"] / "final_results_test.csv"
            if metric_path.exists():
                metrics = pd.read_csv(metric_path).iloc[0].to_dict()
                row = {
                    "value_type": exp["value_type"],
                    "split_group": exp["split_group"],
                    "split_name": exp["split_name"],
                    "difficulty": exp["difficulty"],
                    "seed": exp["seed"],
                    "run_dir": str(exp["run_dir"]),
                }
                row.update(metrics)
                metric_rows.append(row)
        metrics_frame = pd.DataFrame(metric_rows)
        metrics_frame.to_csv(output_root / "summary_runs.csv", index=False)
        if not metrics_frame.empty:
            numeric_cols = [column for column in metrics_frame.columns if pd.api.types.is_numeric_dtype(metrics_frame[column])]
            summary = metrics_frame.groupby(
                ["value_type", "split_group", "split_name", "difficulty"], dropna=False
            )[numeric_cols].agg(["mean", "std"])
            summary.columns = ["%s_%s" % (column, stat) for column, stat in summary.columns]
            summary.reset_index().to_csv(output_root / "summary.csv", index=False)
        print("Saved dispatch results to %s" % results_path, flush=True)


if __name__ == "__main__":
    main()
