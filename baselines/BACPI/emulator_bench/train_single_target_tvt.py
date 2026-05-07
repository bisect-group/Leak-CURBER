import argparse
import datetime
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_RESULTS_DIRNAME,
    append_csv_row,
    enable_fast_torch_math,
    read_table,
    regression_metrics,
    require_columns,
    resolve_amp_dtype,
    resolve_base_dir,
    resolve_single_split_job,
    save_json,
    set_seed,
)
from emulator_bench.dataset import BACPIDataset, CompoundCacheStore, ProteinCacheStore, create_loader
from emulator_bench.feature_pipeline import BACPIFeaturizer, compound_cache_path, filter_invalid_smiles_rows, protein_cache_path
from emulator_bench.modeling_v2 import BACPIModelConfig, build_model


MINIMIZE_METRICS = {"rmse", "mse", "mae", "loss"}


def autocast_context(device: torch.device, dtype=None):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def move_batch_to_device(batch, device: torch.device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def forward_batch(model, batch):
    return model(
        batch["compounds"],
        batch["compound_mask"],
        batch["adjacencies"],
        batch["proteins"],
        batch["protein_mask"],
        batch["fingerprint"],
    )


def metric_direction(metric_name: str) -> str:
    return "minimize" if metric_name in MINIMIZE_METRICS else "maximize"


def monitor_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    residual = y_true - y_pred
    if metric_name == "loss" or metric_name == "mse":
        return float(np.mean(np.square(residual)))
    if metric_name == "rmse":
        return float(np.sqrt(np.mean(np.square(residual))))
    if metric_name == "mae":
        return float(np.mean(np.abs(residual)))
    if metric_name == "pearson":
        return float(regression_metrics(y_true, y_pred)["pearson"])
    if metric_name == "spearman":
        return float(regression_metrics(y_true, y_pred)["spearman"])
    raise ValueError("Unsupported monitor metric: %s" % metric_name)


def evaluate_loader(model, loader, device, autocast_dtype=None, desc="Evaluation", metric_name="rmse", show_progress=True):
    model.eval()
    preds = []
    truths = []
    total_loss = 0.0
    total_samples = 0
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False) if show_progress else loader
    with torch.inference_mode():
        for batch in iterator:
            batch = move_batch_to_device(batch, device)
            with autocast_context(device, autocast_dtype):
                prediction = forward_batch(model, batch)
                loss = F.mse_loss(prediction.float(), batch["targets"].float())
            batch_size = int(batch["targets"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            preds.append(prediction.detach().cpu().float())
            truths.append(batch["targets"].detach().cpu().float())
    pred_np = torch.cat(preds).numpy() if preds else np.array([], dtype=np.float32)
    truth_np = torch.cat(truths).numpy() if truths else np.array([], dtype=np.float32)
    metrics = regression_metrics(truth_np, pred_np)
    metrics["loss"] = round(total_loss / max(1, total_samples), 6)
    metrics[metric_name] = round(monitor_metric(metric_name, truth_np, pred_np), 6)
    return truth_np.reshape(-1), pred_np.reshape(-1), metrics


def train_one_epoch(model, loader, optimizer, device, scaler, autocast_dtype=None, grad_clip: float = 0.0, desc="Train"):
    model.train()
    total_loss = 0.0
    total_samples = 0
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False)
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, autocast_dtype):
            prediction = forward_batch(model, batch)
            loss = F.mse_loss(prediction.float(), batch["targets"].float())

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = int(batch["targets"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        iterator.set_postfix(loss="%.4f" % float(loss.item()))

    return {"loss": round(total_loss / max(1, total_samples), 6)}


def build_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.num_epochs), eta_min=args.min_lr)
    raise ValueError("Unsupported scheduler: %s" % args.scheduler)


def resolve_paths(args):
    if args.train_path and args.val_path and args.test_path:
        return Path(args.train_path), Path(args.val_path), Path(args.test_path), None, None
    base_dir = resolve_base_dir(args.base_dir, args.base_root, args.value_type)
    if args.split_group is None:
        raise ValueError("Provide --split_group when using --base_dir/--base_root.")
    job = resolve_single_split_job(base_dir, split_group=args.split_group, threshold=args.threshold)
    return Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"]), job, base_dir


def default_out_dir(args, job):
    if args.out_dir:
        return Path(args.out_dir)
    if job is None:
        raise ValueError("--out_dir is required when explicit train/val/test paths are used.")
    return Path(job["root_dir"]) / args.results_dirname / ("seed_%s" % args.seed)


def ensure_split_cache(embeddings_dir: Path, frame: pd.DataFrame, smiles_col: str, sequence_col: str):
    missing_compounds = []
    for smiles in sorted(set(frame[smiles_col].astype(str))):
        cache_path = compound_cache_path(embeddings_dir, smiles)
        if not cache_path.exists():
            missing_compounds.append(str(cache_path))
            if len(missing_compounds) >= 3:
                break
    missing_proteins = []
    for sequence in sorted(set(frame[sequence_col].astype(str))):
        cache_path = protein_cache_path(embeddings_dir, sequence)
        if not cache_path.exists():
            missing_proteins.append(str(cache_path))
            if len(missing_proteins) >= 3:
                break
    if missing_compounds or missing_proteins:
        messages = []
        if missing_compounds:
            messages.append("missing compound caches, e.g. %s" % ", ".join(missing_compounds))
        if missing_proteins:
            messages.append("missing protein caches, e.g. %s" % ", ".join(missing_proteins))
        raise FileNotFoundError(
            "Required caches were not found in %s (%s). Run emulator_bench/cache_embeddings.py first."
            % (embeddings_dir, "; ".join(messages))
        )


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray):
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)


def save_metrics(path: Path, metrics: dict):
    pd.DataFrame([metrics]).to_csv(path, index=False)


def prepare_split_frame(frame: pd.DataFrame, split_path: Path, smiles_col: str, sequence_col: str, target_col: str, skip_smiles_validity_check: bool):
    require_columns(frame, [sequence_col, smiles_col, target_col], split_path)
    if skip_smiles_validity_check:
        filtered_frame = frame.reset_index(drop=True)
        return filtered_frame, {
            "source": str(split_path),
            "rows_in": int(len(frame)),
            "rows_out": int(len(frame)),
            "rows_dropped": 0,
            "invalid_examples": [],
        }
    filtered_frame, stats = filter_invalid_smiles_rows(frame, smiles_col=smiles_col, source_name=str(split_path))
    if stats["rows_dropped"] > 0:
        print(
            "Dropped %s rows with invalid SMILES from %s"
            % (stats["rows_dropped"], split_path),
            flush=True,
        )
    if filtered_frame.empty:
        raise ValueError("No valid rows remain in %s after dropping invalid SMILES." % split_path)
    return filtered_frame, stats


def main():
    parser = argparse.ArgumentParser(description="Train BACPI directly on explicit train/val/test split files.")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--base_root", type=str, default=None)
    parser.add_argument("--value_type", type=str, default="ki")
    parser.add_argument("--split_group", type=str, default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--task_name", type=str, default="bacpi_retrain")

    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--scheduler", choices=["step", "cosine", "none"], default="step")
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--val_every", type=int, default=0)

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

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--bucket_size_multiplier", type=int, default=50)
    parser.add_argument("--pad_to_multiple", type=int, default=1)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--skip_smiles_validity_check", action="store_true")
    parser.add_argument("--monitor_metric", choices=["rmse", "pearson", "spearman", "mae", "mse", "loss"], default="rmse")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--preload_features", dest="preload_features", action="store_true")
    parser.add_argument("--no_preload_features", dest="preload_features", action="store_false")
    parser.add_argument("--amsgrad", dest="amsgrad", action="store_true")
    parser.add_argument("--no_amsgrad", dest="amsgrad", action="store_false")
    parser.set_defaults(pin_memory=True, persistent_workers=True, preload_features=True, amsgrad=True)
    args = parser.parse_args()

    set_seed(args.seed)
    enable_fast_torch_math()
    device = torch.device(args.device)
    autocast_dtype, precision_mode = resolve_amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and autocast_dtype == torch.float16))

    train_path, val_path, test_path, job, base_dir = resolve_paths(args)
    embeddings_dir = Path(args.embeddings_dir).expanduser().resolve() if args.embeddings_dir else ((base_dir / "embeddings") if base_dir is not None else None)
    if embeddings_dir is None:
        raise ValueError("--embeddings_dir is required when explicit train/val/test paths are used.")
    out_dir = default_out_dir(args, job)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_table(train_path)
    val_df = read_table(val_path)
    test_df = read_table(test_path)
    train_df, train_filter_stats = prepare_split_frame(
        train_df, train_path, args.smiles_col, args.sequence_col, args.target_col, args.skip_smiles_validity_check
    )
    val_df, val_filter_stats = prepare_split_frame(
        val_df, val_path, args.smiles_col, args.sequence_col, args.target_col, args.skip_smiles_validity_check
    )
    test_df, test_filter_stats = prepare_split_frame(
        test_df, test_path, args.smiles_col, args.sequence_col, args.target_col, args.skip_smiles_validity_check
    )
    for frame in (train_df, val_df, test_df):
        ensure_split_cache(embeddings_dir, frame, args.smiles_col, args.sequence_col)

    featurizer_path = embeddings_dir / "vocabs" / "featurizer.pkl"
    if not featurizer_path.exists():
        raise FileNotFoundError("Missing featurizer at %s. Run emulator_bench/cache_embeddings.py first." % featurizer_path)
    featurizer = BACPIFeaturizer.load(featurizer_path)

    all_sequences = pd.concat([train_df[args.sequence_col], val_df[args.sequence_col], test_df[args.sequence_col]], ignore_index=True).astype(str)
    all_smiles = pd.concat([train_df[args.smiles_col], val_df[args.smiles_col], test_df[args.smiles_col]], ignore_index=True).astype(str)
    unique_compounds = len(set(all_smiles.tolist()))
    unique_proteins = len(set(all_sequences.tolist()))
    print(
        "Split sizes | train=%s val=%s test=%s | unique compounds=%s unique proteins=%s"
        % (len(train_df), len(val_df), len(test_df), unique_compounds, unique_proteins),
        flush=True,
    )
    if args.preload_features:
        print("Preloading cached BACPI features into host memory...", flush=True)
    else:
        print("Using lazy on-demand cache loads from disk.", flush=True)
    compound_store = CompoundCacheStore(
        embeddings_dir,
        smiles_values=all_smiles.tolist(),
        preload=args.preload_features,
        preload_desc="Preloading compounds" if args.preload_features else None,
    )
    protein_store = ProteinCacheStore(
        embeddings_dir,
        sequences=all_sequences.tolist(),
        preload=args.preload_features,
        preload_desc="Preloading proteins" if args.preload_features else None,
    )

    train_dataset = BACPIDataset(train_df, compound_store, protein_store, args.smiles_col, args.sequence_col, args.target_col)
    val_dataset = BACPIDataset(val_df, compound_store, protein_store, args.smiles_col, args.sequence_col, args.target_col)
    test_dataset = BACPIDataset(test_df, compound_store, protein_store, args.smiles_col, args.sequence_col, args.target_col)

    pin_memory = bool(args.pin_memory and device.type == "cuda")
    persistent_workers = bool(args.persistent_workers and args.num_workers > 0)
    print(
        "Building DataLoaders | batch_size=%s workers=%s pin_memory=%s persistent_workers=%s"
        % (args.batch_size, args.num_workers, pin_memory, persistent_workers),
        flush=True,
    )
    train_loader, train_batch_sampler = create_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=persistent_workers,
        bucket_size_multiplier=args.bucket_size_multiplier,
        pad_to_multiple=args.pad_to_multiple,
        seed=args.seed,
    )
    val_loader, _ = create_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=persistent_workers,
        bucket_size_multiplier=args.bucket_size_multiplier,
        pad_to_multiple=args.pad_to_multiple,
        seed=args.seed,
    )
    test_loader, _ = create_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=persistent_workers,
        bucket_size_multiplier=args.bucket_size_multiplier,
        pad_to_multiple=args.pad_to_multiple,
        seed=args.seed,
    )

    model_config = BACPIModelConfig(
        gat_dim=args.gat_dim,
        num_head=args.num_head,
        dropout=args.dropout,
        alpha=args.alpha,
        comp_dim=args.comp_dim,
        prot_dim=args.prot_dim,
        latent_dim=args.latent_dim,
        window=args.window,
        layer_cnn=args.layer_cnn,
        layer_out=args.layer_out,
    )
    model = build_model(featurizer.n_atom_tokens, featurizer.n_amino_tokens, model_config).to(device)
    if args.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )
    scheduler = build_scheduler(optimizer, args)
    monitor_direction = metric_direction(args.monitor_metric)
    best_metric = float("inf") if monitor_direction == "minimize" else float("-inf")
    best_epoch = 0
    non_improving = 0
    started = time.time()

    log_path = out_dir / "logfile.csv"
    best_checkpoint_path = out_dir / "bestmodel.pth"
    best_state_dict_path = out_dir / "bestmodel_state_dict.pth"
    last_checkpoint_path = out_dir / "checkpoint_last.pt"

    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        capability = ".".join(map(str, torch.cuda.get_device_capability(device_index)))
        print("CUDA device: %s | compute capability: %s | precision: %s" % (gpu_name, capability, precision_mode), flush=True)
    else:
        print("Device: %s | precision: %s" % (device, precision_mode), flush=True)
    print("Starting training for %s epochs..." % args.num_epochs, flush=True)

    for epoch in range(1, args.num_epochs + 1):
        if train_batch_sampler is not None and hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
            grad_clip=args.grad_clip,
            desc="Epoch %s train" % epoch,
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "elapsed_seconds": round(time.time() - started, 3),
        }

        val_metrics = None
        if args.val_every > 0 and epoch % args.val_every == 0:
            _val_true, _val_pred, val_metrics = evaluate_loader(
                model,
                val_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                desc="Epoch %s val" % epoch,
                metric_name=args.monitor_metric,
                show_progress=False,
            )
            current_metric = float(val_metrics[args.monitor_metric])
            row.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_%s" % args.monitor_metric: current_metric,
                }
            )
            if monitor_direction == "minimize":
                improved = (best_metric - current_metric) > args.min_delta
            else:
                improved = (current_metric - best_metric) > args.min_delta
            if improved or best_epoch == 0:
                best_metric = current_metric
                best_epoch = epoch
                non_improving = 0
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "monitor_metric": args.monitor_metric,
                    "precision_mode": precision_mode,
                    "args": vars(args),
                }
                torch.save(checkpoint, best_checkpoint_path)
                torch.save(model.state_dict(), best_state_dict_path)
            else:
                non_improving += 1

        if scheduler is not None:
            scheduler.step()

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "monitor_metric": args.monitor_metric,
                "precision_mode": precision_mode,
                "args": vars(args),
            },
            last_checkpoint_path,
        )
        append_csv_row(log_path, row)

        if args.val_every > 0 and args.patience > 0 and non_improving >= args.patience:
            print("Early stopping at epoch %s after %s non-improving validation checks." % (epoch, non_improving), flush=True)
            break

    if best_checkpoint_path.exists():
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    else:
        best_checkpoint = torch.load(last_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    train_true, train_pred, train_final_metrics = evaluate_loader(
        model,
        train_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        desc="Final train",
        metric_name=args.monitor_metric,
        show_progress=True,
    )
    val_true, val_pred, val_final_metrics = evaluate_loader(
        model,
        val_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        desc="Final val",
        metric_name=args.monitor_metric,
        show_progress=True,
    )
    test_true, test_pred, test_final_metrics = evaluate_loader(
        model,
        test_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        desc="Final test",
        metric_name=args.monitor_metric,
        show_progress=True,
    )

    save_predictions(out_dir / "pred_label_val.csv", val_true, val_pred)
    save_predictions(out_dir / "pred_label_test.csv", test_true, test_pred)
    save_metrics(out_dir / "final_results_train.csv", train_final_metrics)
    save_metrics(out_dir / "final_results_val.csv", val_final_metrics)
    save_metrics(out_dir / "final_results_test.csv", test_final_metrics)

    summary = {
        "task_name": args.task_name,
        "started_at": datetime.datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 3),
        "precision_mode": precision_mode,
        "best_epoch": int(best_checkpoint.get("epoch", best_epoch or args.num_epochs)),
        "monitor_metric": args.monitor_metric,
        "best_metric": (None if args.val_every <= 0 else float(best_checkpoint.get("best_metric", best_metric))),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "embeddings_dir": str(embeddings_dir),
        "n_atom_tokens": int(featurizer.n_atom_tokens),
        "n_amino_tokens": int(featurizer.n_amino_tokens),
        "invalid_smiles_rows_dropped": {
            "train": int(train_filter_stats["rows_dropped"]),
            "val": int(val_filter_stats["rows_dropped"]),
            "test": int(test_filter_stats["rows_dropped"]),
        },
        "final_train_metrics": train_final_metrics,
        "final_val_metrics": val_final_metrics,
        "final_test_metrics": test_final_metrics,
        "args": vars(args),
    }
    save_json(out_dir / "run_summary.json", summary)
    pd.DataFrame(
        [
            {
                "task_name": args.task_name,
                "best_epoch": summary["best_epoch"],
                "best_metric": summary["best_metric"],
                "monitor_metric": args.monitor_metric,
                "precision_mode": precision_mode,
                "elapsed_seconds": summary["elapsed_seconds"],
                "train_path": str(train_path),
                "val_path": str(val_path),
                "test_path": str(test_path),
            }
        ]
    ).to_csv(out_dir / "run_summary.csv", index=False)


if __name__ == "__main__":
    main()
