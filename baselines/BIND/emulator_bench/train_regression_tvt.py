import argparse
import datetime
import gc
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HIDDEN_STATE_INDICES,
    DEFAULT_RESULTS_DIRNAME,
    append_csv_row,
    canonical_value_type,
    normalize_sequence,
    read_table,
    regression_metrics,
    require_columns,
    resolve_single_split_job,
    save_json,
    set_seed,
    truncate_sequence,
)
from emulator_bench.dataset import CachedBindDataset, collate_bind_samples
from emulator_bench.feature_pipeline import (
    LigandGraphStore,
    PROTEIN_MODEL_NAME,
    ProteinEmbeddingStore,
    esm_sequence_limit_from_pretrained,
    resolve_amp_dtype,
)
from emulator_bench.modeling import build_model


MINIMIZE_METRICS = {"rmse", "mse", "mae", "loss"}
PAPER_EFFECTIVE_EPOCHS = 10.365942049524907


def _autocast_context(device: torch.device, dtype=None):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _build_scheduler(optimizer, args, total_optimizer_steps: int):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(total_optimizer_steps)),
            eta_min=float(args.min_lr),
        )
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.lr_decay_factor),
            patience=int(args.lr_decay_patience),
            min_lr=float(args.min_lr),
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def _metric_direction(metric_name: str) -> str:
    return "minimize" if metric_name in MINIMIZE_METRICS else "maximize"


def _monitor_metric_from_arrays(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metrics = regression_metrics(y_true, y_pred)
    if metric_name == "loss":
        residual = np.asarray(y_true, dtype=np.float64).reshape(-1) - np.asarray(y_pred, dtype=np.float64).reshape(-1)
        return float(np.mean(np.square(residual))) if residual.size else float("nan")
    return float(metrics[metric_name])


def _prepare_batch(batch, device: torch.device):
    return batch.to(device)


def _make_loader(dataset, loader_kwargs: dict, shuffle: bool):
    return DataLoader(dataset, shuffle=shuffle, **loader_kwargs)


def _shutdown_loader(loader) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()
        loader._iterator = None
    del loader
    gc.collect()


def _resolve_target_optimizer_steps(args, train_size: int) -> int:
    if args.train_budget_mode == "fixed_optimizer_steps":
        return max(1, int(args.max_optimizer_steps))
    if args.train_budget_mode == "fixed_epochs":
        steps_per_epoch = max(1, math.ceil(int(train_size) / max(1, int(args.batch_size) * int(args.grad_accumulation_steps))))
        return max(1, int(args.epochs) * int(steps_per_epoch))
    effective_batch_size = max(1, int(args.batch_size) * int(args.grad_accumulation_steps))
    return max(1, int(math.ceil(float(args.paper_effective_epochs) * int(train_size) / effective_batch_size)))


def evaluate_loader(model, loader, device, criterion, target_head, autocast_dtype=None, desc="Evaluation", metric_name="rmse", show_progress=True):
    model.eval()
    preds = []
    truths = []
    metadatas = []
    total_loss = 0.0
    total_samples = 0
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False) if show_progress else loader
    with torch.no_grad():
        for batch in iterator:
            batch = _prepare_batch(batch, device)
            with _autocast_context(device, autocast_dtype):
                outputs = model(batch.graphs, batch.protein_layers, batch.attention_mask)
                prediction = outputs[target_head]
                loss = criterion(prediction, batch.targets)
            total_loss += float(loss.item()) * int(batch.targets.shape[0])
            total_samples += int(batch.targets.shape[0])
            preds.append(prediction.detach().cpu().float())
            truths.append(batch.targets.detach().cpu().float())
            metadatas.extend(batch.metadata)
    pred_np = torch.cat(preds).numpy() if preds else np.array([], dtype=np.float32)
    truth_np = torch.cat(truths).numpy() if truths else np.array([], dtype=np.float32)
    avg_loss = total_loss / max(1, total_samples)
    metrics = regression_metrics(truth_np, pred_np)
    metrics["loss"] = avg_loss
    metrics[metric_name] = avg_loss if metric_name == "loss" else _monitor_metric_from_arrays(metric_name, truth_np, pred_np)
    return truth_np, pred_np, metrics, metadatas


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    target_head,
    grad_accumulation_steps,
    max_optimizer_steps,
    optimizer_steps_completed=0,
    scheduler=None,
    autocast_dtype=None,
    clip_grad=None,
    desc="Train",
    max_consecutive_bad_batches: int = 50,
):
    model.train()
    total_loss = 0.0
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False)
    optimizer_steps_this_epoch = 0
    total_epochs = getattr(loader, "_bench_total_epochs", None)
    epoch_index = getattr(loader, "_bench_epoch_index", None)
    consecutive_bad_batches = 0
    for step_index, batch in enumerate(iterator, start=1):
        batch = _prepare_batch(batch, device)
        with _autocast_context(device, autocast_dtype):
            outputs = model(batch.graphs, batch.protein_layers, batch.attention_mask)
            prediction = outputs[target_head]
            loss = criterion(prediction, batch.targets)

        if not loss.isfinite():
            consecutive_bad_batches += 1
            if consecutive_bad_batches >= max_consecutive_bad_batches:
                raise RuntimeError(
                    f"Training aborted: {consecutive_bad_batches} consecutive non-finite loss batches. "
                    "Consider adding --clip_grad 1.0 or reducing --lr."
                )
            iterator.set_postfix({"loss": "non-finite", "skipped": consecutive_bad_batches})
            continue
        consecutive_bad_batches = 0

        if scaler.is_enabled():
            scaler.scale(loss / grad_accumulation_steps).backward()
        else:
            (loss / grad_accumulation_steps).backward()

        should_step = (step_index % grad_accumulation_steps == 0) or (step_index == len(loader))
        if should_step:
            if scaler.is_enabled():
                if clip_grad is not None and clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad is not None and clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps_completed += 1
            optimizer_steps_this_epoch += 1

        total_loss += float(loss.item()) * int(batch.targets.shape[0])
        total_samples += int(batch.targets.shape[0])
        postfix = {"loss": f"{float(loss.item()):.4f}"}
        if epoch_index is not None and total_epochs is not None:
            postfix["epoch"] = f"{epoch_index}/{total_epochs}"
        else:
            postfix["epoch"] = "?"
        iterator.set_postfix(postfix)
        if optimizer_steps_completed >= max_optimizer_steps:
            break

    return {
        "loss": total_loss / max(1, total_samples),
        "optimizer_steps_completed": int(optimizer_steps_completed),
        "optimizer_steps_this_epoch": int(optimizer_steps_this_epoch),
    }


def save_predictions(path: Path, metadata_rows, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rows = []
    for metadata, truth, pred in zip(metadata_rows, y_true.reshape(-1), y_pred.reshape(-1)):
        row = dict(metadata)
        row["y_true"] = float(truth)
        row["y_pred"] = float(pred)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_metrics(path: Path, metrics: dict) -> None:
    pd.DataFrame([metrics]).to_csv(path, index=False)


def _resolve_paths(args):
    if args.train_path and args.val_path and args.test_path:
        return Path(args.train_path), Path(args.val_path), Path(args.test_path), None
    if not args.base_dir or not args.value_type or not args.split_group:
        raise ValueError("Provide either explicit --train_path/--val_path/--test_path or --base_dir with --value_type and --split_group.")
    job = resolve_single_split_job(
        Path(args.base_dir),
        value_type=args.value_type,
        split_group=args.split_group,
        threshold=args.threshold,
    )
    return Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"]), job


def _default_out_dir(args, job):
    if args.out_dir:
        return Path(args.out_dir)
    if job is None:
        raise ValueError("--out_dir is required when explicit train/val/test paths are used.")
    return Path(job["root_dir"]) / args.results_dirname / f"seed_{args.seed}"


def _existing_cache_mask(frame, embeddings_dir, args):
    from emulator_bench.common import ligand_cache_path, protein_cache_path

    sequences = frame[args.sequence_col].astype(str).map(normalize_sequence)
    smiles = frame[args.smiles_col].astype(str).map(lambda value: str(value).strip())
    protein_exists = sequences.map(
        lambda sequence: protein_cache_path(
            embeddings_dir,
            sequence,
            model_name=args.protein_model_name,
            layer_indices=args.hidden_state_indices,
        ).exists()
    )
    ligand_exists = smiles.map(lambda smiles_value: ligand_cache_path(embeddings_dir, smiles_value).exists())
    return protein_exists & ligand_exists


def _filter_frame(frame, split_name: str, embeddings_dir: Path, args, model_sequence_limit: int):
    filtered = frame.copy()
    filtered = filtered.loc[
        filtered[args.sequence_col].notna()
        & filtered[args.smiles_col].notna()
        & filtered[args.target_col].notna()
    ].copy()
    filtered["original_sequence_length"] = filtered[args.sequence_col].astype(str).map(lambda value: len(normalize_sequence(value)))
    filtered[args.sequence_col] = filtered[args.sequence_col].astype(str).map(lambda value: truncate_sequence(value, args.protein_length_cap))
    filtered[args.smiles_col] = filtered[args.smiles_col].astype(str).map(lambda value: str(value).strip())
    filtered["sequence_length"] = filtered[args.sequence_col].map(len)
    truncated_rows = int((filtered["sequence_length"] < filtered["original_sequence_length"]).sum())
    if args.long_sequence_strategy == "drop":
        filtered = filtered.loc[filtered["sequence_length"] <= int(model_sequence_limit)].copy()
    filtered = filtered.loc[_existing_cache_mask(filtered, embeddings_dir, args)].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered, {
        "split_name": split_name,
        "rows_after_filter": int(len(filtered)),
        "rows_truncated_to_cap": truncated_rows,
        "max_original_sequence_length": int(filtered["original_sequence_length"].max()) if len(filtered) else 0,
        "max_sequence_length_kept": int(filtered["sequence_length"].max()) if len(filtered) else 0,
        "protein_length_cap": int(args.protein_length_cap),
        "model_sequence_limit": int(model_sequence_limit),
        "long_sequence_strategy": args.long_sequence_strategy,
    }


def main():
    parser = argparse.ArgumentParser(description="Train BIND regression on explicit train/val/test split files.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--value_type", type=str, default=None)
    parser.add_argument("--split_group", type=str, default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--task_name", type=str, default="bind_regression_retrain")

    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--target_head", type=str, default=None)
    parser.add_argument("--protein_model_name", type=str, default=PROTEIN_MODEL_NAME)
    parser.add_argument("--hidden_state_indices", nargs="+", type=int, default=DEFAULT_HIDDEN_STATE_INDICES)
    parser.add_argument("--protein_length_cap", type=int, default=2048)
    parser.add_argument("--long_sequence_strategy", choices=["drop", "direct", "sliding_window"], default="direct")

    parser.add_argument("--cross_attention_dropout", type=float, default=0.1)
    parser.add_argument("--leaky_relu_negative_slope", type=float, default=0.05)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=11)
    parser.add_argument("--train_budget_mode", choices=["paper_steps", "fixed_epochs", "fixed_optimizer_steps"], default="fixed_epochs")
    parser.add_argument("--paper_effective_epochs", type=float, default=PAPER_EFFECTIVE_EPOCHS)
    parser.add_argument("--max_optimizer_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_decay_patience", type=int, default=2)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--huber_delta", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--monitor_metric", choices=["rmse", "pearson", "spearman", "r2_score", "mae", "mse", "loss"], default="rmse")
    parser.add_argument("--validate_every_epochs", type=int, default=1)

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--sharing_strategy", choices=["file_descriptor", "file_system"], default="file_system")
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=512)
    parser.add_argument("--lazy_ligands", action="store_true")
    parser.add_argument("--preload_ligands", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy(args.sharing_strategy)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    set_seed(args.seed)
    args.hidden_state_indices = [int(index) for index in args.hidden_state_indices]
    args.value_type = canonical_value_type(args.value_type) if args.value_type else None
    target_head = canonical_value_type(args.target_head or args.value_type or "ki")

    device = torch.device(args.device)
    autocast_dtype, precision_mode = resolve_amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype == torch.float16))

    train_path, val_path, test_path, job = _resolve_paths(args)
    out_dir = _default_out_dir(args, job)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_table(train_path)
    val_df = read_table(val_path)
    test_df = read_table(test_path)
    for split_path, frame in ((train_path, train_df), (val_path, val_df), (test_path, test_df)):
        require_columns(frame, [args.sequence_col, args.smiles_col, args.target_col], split_path)

    model_sequence_limit = esm_sequence_limit_from_pretrained(args.protein_model_name)

    embeddings_dir = Path(args.embeddings_dir)
    train_df, train_filter = _filter_frame(train_df, "train", embeddings_dir, args, model_sequence_limit)
    val_df, val_filter = _filter_frame(val_df, "val", embeddings_dir, args, model_sequence_limit)
    test_df, test_filter = _filter_frame(test_df, "test", embeddings_dir, args, model_sequence_limit)

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("One or more filtered splits are empty after applying cache and sequence-length checks.")

    all_sequences = pd.concat(
        [train_df[args.sequence_col], val_df[args.sequence_col], test_df[args.sequence_col]],
        ignore_index=True,
    ).astype(str)
    all_smiles = pd.concat(
        [train_df[args.smiles_col], val_df[args.smiles_col], test_df[args.smiles_col]],
        ignore_index=True,
    ).astype(str)

    protein_store = ProteinEmbeddingStore(
        embeddings_dir,
        model_name=args.protein_model_name,
        layer_indices=args.hidden_state_indices,
        sequences=all_sequences.tolist(),
        preload=args.preload_proteins,
        max_items=args.protein_cache_items,
    )
    ligand_store = LigandGraphStore(
        embeddings_dir,
        smiles_values=all_smiles.tolist(),
        preload=(args.preload_ligands and not args.lazy_ligands),
    )

    train_dataset = CachedBindDataset(train_df, protein_store, ligand_store, args.sequence_col, args.smiles_col, args.target_col)
    val_dataset = CachedBindDataset(val_df, protein_store, ligand_store, args.sequence_col, args.smiles_col, args.target_col)
    test_dataset = CachedBindDataset(test_df, protein_store, ligand_store, args.sequence_col, args.smiles_col, args.target_col)

    pin_memory = args.pin_memory or device.type == "cuda"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_bind_samples,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    eval_loader_kwargs = dict(loader_kwargs)
    eval_loader_kwargs["persistent_workers"] = False

    train_loader = _make_loader(train_dataset, loader_kwargs, shuffle=True)

    model = build_model(
        cross_attention_dropout=args.cross_attention_dropout,
        leaky_relu_negative_slope=args.leaky_relu_negative_slope,
    ).to(device)
    if args.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        amsgrad=args.amsgrad,
    )
    target_optimizer_steps = _resolve_target_optimizer_steps(args, train_size=len(train_dataset))
    scheduler = _build_scheduler(optimizer, args, total_optimizer_steps=target_optimizer_steps)
    criterion = nn.HuberLoss(delta=args.huber_delta, reduction="mean")

    log_path = out_dir / "logfile.csv"
    best_checkpoint_path = out_dir / "bestmodel.pt"
    best_state_dict_path = out_dir / "bestmodel_state_dict.pth"
    last_checkpoint_path = out_dir / "checkpoint_last.pt"
    run_summary_path = out_dir / "run_summary.json"
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    started = time.time()
    monitor_direction = _metric_direction(args.monitor_metric)
    best_val_metric = float("inf") if monitor_direction == "minimize" else float("-inf")
    no_improve = 0
    force_validate_each_epoch = args.patience > 0 or args.scheduler == "plateau"
    validate_every_epochs = 1 if force_validate_each_epoch else max(0, int(args.validate_every_epochs))

    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        capability = ".".join(map(str, torch.cuda.get_device_capability(device_index)))
        print(f"CUDA device: {gpu_name} | compute capability: {capability} | precision: {precision_mode}", flush=True)
    else:
        print(f"Device: {device} | precision: {precision_mode}", flush=True)

    optimizer_steps_completed = 0
    epoch_index = 0
    if not args.no_resume and last_checkpoint_path.exists():
        _ckpt = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(_ckpt["model_state_dict"])
        optimizer.load_state_dict(_ckpt["optimizer_state_dict"])
        optimizer_steps_completed = int(_ckpt["optimizer_steps_completed"])
        epoch_index = int(_ckpt["epoch"])
        best_val_metric = float(_ckpt.get("best_val_metric", best_val_metric))
        no_improve = int(_ckpt.get("no_improve", 0))
        if scheduler is not None and "scheduler_state_dict" in _ckpt and _ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(_ckpt["scheduler_state_dict"])
        if scaler.is_enabled() and "scaler_state_dict" in _ckpt and _ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(_ckpt["scaler_state_dict"])
        print(f"Resumed from checkpoint: epoch {epoch_index}, optimizer steps {optimizer_steps_completed}/{target_optimizer_steps}", flush=True)

    total_epochs = int(args.epochs) if args.train_budget_mode == "fixed_epochs" else None
    while optimizer_steps_completed < target_optimizer_steps:
        epoch_index += 1
        train_loader._bench_epoch_index = epoch_index
        train_loader._bench_total_epochs = total_epochs
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            scaler=scaler,
            target_head=target_head,
            grad_accumulation_steps=max(1, int(args.grad_accumulation_steps)),
            max_optimizer_steps=target_optimizer_steps,
            optimizer_steps_completed=optimizer_steps_completed,
            scheduler=scheduler if args.scheduler == "cosine" else None,
            autocast_dtype=autocast_dtype,
            clip_grad=args.clip_grad,
            desc=f"Epoch {epoch_index} train",
        )
        optimizer_steps_completed = int(train_metrics["optimizer_steps_completed"])

        should_validate = (
            validate_every_epochs > 0 and epoch_index % validate_every_epochs == 0
        ) or optimizer_steps_completed >= target_optimizer_steps
        val_metrics = {"loss": float("nan"), args.monitor_metric: float("nan")}
        if should_validate:
            val_loader = _make_loader(val_dataset, eval_loader_kwargs, shuffle=False)
            _val_true, _val_pred, val_metrics, _val_metadata = evaluate_loader(
                model,
                val_loader,
                device=device,
                criterion=criterion,
                target_head=target_head,
                autocast_dtype=autocast_dtype,
                desc=f"Epoch {epoch_index} val",
                metric_name=args.monitor_metric,
                show_progress=False,
            )
            _shutdown_loader(val_loader)

        row = {
            "epoch": epoch_index,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            f"val_{args.monitor_metric}": val_metrics[args.monitor_metric],
            "optimizer_steps_completed": optimizer_steps_completed,
            "target_optimizer_steps": target_optimizer_steps,
            "elapsed_seconds": time.time() - started,
        }

        if should_validate:
            current_val_metric = float(val_metrics[args.monitor_metric])
            if monitor_direction == "minimize":
                improved = (best_val_metric - current_val_metric) > args.min_delta
            else:
                improved = (current_val_metric - best_val_metric) > args.min_delta
            if improved:
                best_val_metric = current_val_metric
                no_improve = 0
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch_index,
                    "best_val_metric": best_val_metric,
                    "monitor_metric": args.monitor_metric,
                    "target_head": target_head,
                    "args": vars(args),
                    "precision_mode": precision_mode,
                    "optimizer_steps_completed": optimizer_steps_completed,
                    "target_optimizer_steps": target_optimizer_steps,
                }
                torch.save(checkpoint, best_checkpoint_path)
                torch.save(model.state_dict(), best_state_dict_path)
            else:
                no_improve += 1

            if scheduler is not None and args.scheduler == "plateau":
                scheduler.step(val_metrics[args.monitor_metric] if args.monitor_metric in MINIMIZE_METRICS else val_metrics["loss"])

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "epoch": epoch_index,
                "no_improve": no_improve,
                "best_val_metric": best_val_metric,
                "monitor_metric": args.monitor_metric,
                "target_head": target_head,
                "args": vars(args),
                "precision_mode": precision_mode,
                "optimizer_steps_completed": optimizer_steps_completed,
                "target_optimizer_steps": target_optimizer_steps,
            },
            last_checkpoint_path,
        )
        append_csv_row(log_path, row)

        if should_validate and args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch_index} after {no_improve} non-improving validation checks.", flush=True)
            break

    if not best_checkpoint_path.exists():
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch_index,
                "best_val_metric": best_val_metric,
                "monitor_metric": args.monitor_metric,
                "target_head": target_head,
                "args": vars(args),
                "precision_mode": precision_mode,
                "optimizer_steps_completed": optimizer_steps_completed,
                "target_optimizer_steps": target_optimizer_steps,
            },
            best_checkpoint_path,
        )
        torch.save(model.state_dict(), best_state_dict_path)

    _shutdown_loader(train_loader)
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    train_loader = _make_loader(train_dataset, eval_loader_kwargs, shuffle=False)
    final_train_true, final_train_pred, final_train_metrics, final_train_metadata = evaluate_loader(
        model,
        train_loader,
        device=device,
        criterion=criterion,
        target_head=target_head,
        autocast_dtype=autocast_dtype,
        desc="Final train",
        metric_name=args.monitor_metric,
    )
    _shutdown_loader(train_loader)
    val_loader = _make_loader(val_dataset, eval_loader_kwargs, shuffle=False)
    final_val_true, final_val_pred, final_val_metrics, final_val_metadata = evaluate_loader(
        model,
        val_loader,
        device=device,
        criterion=criterion,
        target_head=target_head,
        autocast_dtype=autocast_dtype,
        desc="Final val",
        metric_name=args.monitor_metric,
    )
    _shutdown_loader(val_loader)
    test_loader = _make_loader(test_dataset, eval_loader_kwargs, shuffle=False)
    final_test_true, final_test_pred, final_test_metrics, final_test_metadata = evaluate_loader(
        model,
        test_loader,
        device=device,
        criterion=criterion,
        target_head=target_head,
        autocast_dtype=autocast_dtype,
        desc="Final test",
        metric_name=args.monitor_metric,
    )
    _shutdown_loader(test_loader)

    save_predictions(out_dir / "pred_label_train.csv", final_train_metadata, final_train_true, final_train_pred)
    save_predictions(out_dir / "pred_label_val.csv", final_val_metadata, final_val_true, final_val_pred)
    save_predictions(out_dir / "pred_label_test.csv", final_test_metadata, final_test_true, final_test_pred)
    save_metrics(out_dir / "final_results_train.csv", final_train_metrics)
    save_metrics(out_dir / "final_results_val.csv", final_val_metrics)
    save_metrics(out_dir / "final_results_test.csv", final_test_metrics)

    summary = {
        "task_name": args.task_name,
        "value_type": args.value_type,
        "target_head": target_head,
        "started_at": started_at,
        "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - started,
        "precision_mode": precision_mode,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "filters": [train_filter, val_filter, test_filter],
        "train_budget_mode": args.train_budget_mode,
        "paper_effective_epochs": float(args.paper_effective_epochs),
        "target_optimizer_steps": int(target_optimizer_steps),
        "optimizer_steps_completed": int(optimizer_steps_completed),
        "effective_batch_size": int(args.batch_size) * int(args.grad_accumulation_steps),
        "best_val_metric": float(best_val_metric),
        "monitor_metric": args.monitor_metric,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
        "args": vars(args),
    }
    save_json(run_summary_path, summary)
    print(f"Finished training. Results saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
