import argparse
import datetime
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_RESULTS_DIRNAME,
    append_csv_row,
    read_table,
    regression_metrics,
    require_columns,
    resolve_single_split_job,
    save_json,
    set_seed,
)
from emulator_bench.dataset import CachedStructureFreeDataset, collate_cached
from emulator_bench.feature_pipeline import EmbeddingStore, autocast_context, resolve_amp_dtype
from emulator_bench.modeling import CachedAffinityPredictor


MINIMIZE_METRICS = {"rmse", "mse", "mae", "loss"}


def parse_hidden_sizes(value: str):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_batch_schedule(value: str, default_batch_size: int):
    if value is None or not str(value).strip():
        return [(1, int(default_batch_size))]
    schedule = []
    for item in str(value).split(","):
        epoch_text, batch_text = item.split(":")
        schedule.append((int(epoch_text), int(batch_text)))
    schedule = sorted(schedule)
    if schedule[0][0] != 1:
        schedule.insert(0, (1, int(default_batch_size)))
    return schedule


def batch_size_for_epoch(schedule, epoch: int) -> int:
    current = schedule[0][1]
    for start_epoch, batch_size in schedule:
        if epoch >= start_epoch:
            current = batch_size
    return int(current)


def metric_direction(metric_name: str) -> str:
    return "minimize" if metric_name in MINIMIZE_METRICS else "maximize"


def hybrid_regression_loss(prediction: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    mse = F.mse_loss(prediction, target, reduction="sum")
    if alpha <= 0:
        return mse
    if prediction.numel() < 2:
        return mse
    cosine = 1.0 - F.cosine_similarity(prediction.view(1, -1), target.view(1, -1), dim=1).mean()
    return float(alpha) * cosine + (1.0 - float(alpha)) * mse


def prepare_batch(batch, device: torch.device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_loader(dataset, batch_size, args, shuffle):
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(args.num_workers),
        "pin_memory": args.pin_memory or torch.cuda.is_available(),
        "collate_fn": collate_cached,
        "drop_last": False,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def evaluate_loader(model, loader, device, autocast_dtype, loss_alpha, desc="Evaluation", show_progress=True):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds = []
    truths = []
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False) if show_progress else loader
    with torch.no_grad():
        for batch in iterator:
            batch = prepare_batch(batch, device)
            labels = batch.pop("labels")
            with autocast_context(device, autocast_dtype):
                output = model(batch)
                loss = hybrid_regression_loss(output.float(), labels.float(), alpha=loss_alpha)
            total_loss += float(loss.item()) * labels.numel()
            total_samples += labels.numel()
            preds.append(output.detach().cpu().float())
            truths.append(labels.detach().cpu().float())
    pred_np = torch.cat(preds).numpy() if preds else np.array([], dtype=np.float32)
    truth_np = torch.cat(truths).numpy() if truths else np.array([], dtype=np.float32)
    metrics = regression_metrics(truth_np, pred_np)
    metrics["loss"] = total_loss / max(1, total_samples)
    return truth_np, pred_np, metrics


def train_one_epoch(model, loader, optimizer, device, scaler, autocast_dtype, loss_alpha, clip_grad=None, desc="Train"):
    model.train()
    total_loss = 0.0
    total_samples = 0
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False)
    for batch in iterator:
        batch = prepare_batch(batch, device)
        labels = batch.pop("labels")
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, autocast_dtype):
            output = model(batch)
            loss = hybrid_regression_loss(output.float(), labels.float(), alpha=loss_alpha)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if clip_grad and clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        total_samples += labels.numel()
        iterator.set_postfix(loss="%.4f" % float(loss.item()))
    return {"loss": total_loss / max(1, total_samples)}


def build_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_decay_factor,
            patience=args.lr_decay_patience,
            min_lr=args.min_lr,
        )
    warmup_epochs = max(0, min(args.lr_warmup_epochs, args.epochs - 1))
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.min_lr)
    if args.scheduler == "cosine" and warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.lr_warmup_start_factor,
            end_factor=1.0,
            total_iters=max(1, warmup_epochs),
        )
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return cosine


def resolve_paths(args):
    if args.train_path and args.val_path and args.test_path:
        return Path(args.train_path), Path(args.val_path), Path(args.test_path), None
    job = resolve_single_split_job(Path(args.base_dir), args.split_group, args.threshold)
    return Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"]), job


def default_out_dir(args, job):
    if args.out_dir:
        return Path(args.out_dir)
    if job is None:
        raise ValueError("--out_dir is required when explicit paths are used.")
    return Path(job["root_dir"]) / args.results_dirname / f"seed_{args.seed}"


def save_predictions(path: Path, frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, args) -> None:
    out = frame[[args.sequence_col, args.smiles_col, args.target_col]].copy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out.to_csv(path, index=False)


def save_metrics(path: Path, metrics: dict) -> None:
    pd.DataFrame([metrics]).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train StructureFree-DTA head on explicit train/val/test split files using cached embeddings.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--split_group", type=str, default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--task_name", type=str, default="structurefree_retrain")

    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--protein_model_name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--molecule_model_name", type=str, default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--max_protein_length", type=int, default=1024)
    parser.add_argument("--max_molecule_length", type=int, default=128)

    parser.add_argument("--hidden_sizes", type=str, default="1024,768,512,256,1")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_schedule", type=str, default="1:64,20:512,60:2048")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_decay_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--lr_warmup_epochs", type=int, default=10)
    parser.add_argument("--lr_warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--loss_alpha", type=float, default=0.5)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--monitor_metric", choices=["rmse", "mse", "mae", "loss", "r2_score", "pearson", "spearman", "ci"], default="rmse")

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload_embeddings", action="store_true")
    parser.add_argument("--embedding_cache_items", type=int, default=20000)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    set_seed(args.seed)
    device = torch.device(args.device)
    autocast_dtype, precision_mode = resolve_amp_dtype(device)
    scaler = make_grad_scaler(enabled=(autocast_dtype == torch.float16))

    train_path, val_path, test_path, job = resolve_paths(args)
    out_dir = default_out_dir(args, job)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_table(train_path)
    val_df = read_table(val_path)
    test_df = read_table(test_path)
    for split_path, frame in ((train_path, train_df), (val_path, val_df), (test_path, test_df)):
        require_columns(frame, [args.sequence_col, args.smiles_col, args.target_col], split_path)

    all_sequences = pd.concat([train_df[args.sequence_col], val_df[args.sequence_col], test_df[args.sequence_col]], ignore_index=True).astype(str).tolist()
    all_smiles = pd.concat([train_df[args.smiles_col], val_df[args.smiles_col], test_df[args.smiles_col]], ignore_index=True).astype(str).tolist()
    protein_store = EmbeddingStore(
        Path(args.embeddings_dir),
        "proteins",
        args.protein_model_name,
        args.max_protein_length,
        values=all_sequences,
        preload=args.preload_embeddings,
        max_items=args.embedding_cache_items,
    )
    molecule_store = EmbeddingStore(
        Path(args.embeddings_dir),
        "molecules",
        args.molecule_model_name,
        args.max_molecule_length,
        values=all_smiles,
        preload=args.preload_embeddings,
        max_items=args.embedding_cache_items,
    )

    train_dataset = CachedStructureFreeDataset(train_df, protein_store, molecule_store, args.sequence_col, args.smiles_col, args.target_col)
    val_dataset = CachedStructureFreeDataset(val_df, protein_store, molecule_store, args.sequence_col, args.smiles_col, args.target_col)
    test_dataset = CachedStructureFreeDataset(test_df, protein_store, molecule_store, args.sequence_col, args.smiles_col, args.target_col)

    sample = train_dataset[0]
    model = CachedAffinityPredictor(
        protein_dim=int(sample["protein_embedding"].numel()),
        molecule_dim=int(sample["molecule_embedding"].numel()),
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
        dropout=args.dropout,
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
    scheduler = build_scheduler(optimizer, args)
    batch_schedule = parse_batch_schedule(args.batch_size_schedule, args.batch_size)

    monitor_direction = metric_direction(args.monitor_metric)
    best_val_metric = float("inf") if monitor_direction == "minimize" else float("-inf")
    no_improve = 0
    best_checkpoint_path = out_dir / "bestmodel.pth"
    last_checkpoint_path = out_dir / "checkpoint_last.pt"
    log_path = out_dir / "logfile.csv"
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    started = time.time()

    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        print(
            f"CUDA device: {torch.cuda.get_device_name(index)} | capability: {torch.cuda.get_device_capability(index)} | precision: {precision_mode}",
            flush=True,
        )
    else:
        print(f"Device: {device} | precision: {precision_mode}", flush=True)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        batch_size = batch_size_for_epoch(batch_schedule, epoch)
        train_loader = make_loader(train_dataset, batch_size, args, shuffle=True)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
            loss_alpha=args.loss_alpha,
            clip_grad=args.clip_grad,
            desc=f"Epoch {epoch} train bs={batch_size}",
        )
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "batch_size": batch_size, "train_loss": train_metrics["loss"], "elapsed_seconds": time.time() - started}

        val_metrics = None
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_loader = make_loader(val_dataset, min(batch_size, 4096), args, shuffle=False)
            _val_true, _val_pred, val_metrics = evaluate_loader(
                model,
                val_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                loss_alpha=args.loss_alpha,
                desc=f"Epoch {epoch} val",
                show_progress=False,
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            current = float(val_metrics[args.monitor_metric])
            improved = (best_val_metric - current) > args.min_delta if monitor_direction == "minimize" else (current - best_val_metric) > args.min_delta
            if improved:
                best_val_metric = current
                no_improve = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_val_metric": best_val_metric,
                        "monitor_metric": args.monitor_metric,
                        "args": vars(args),
                        "precision_mode": precision_mode,
                    },
                    best_checkpoint_path,
                )
            else:
                no_improve += 1
        else:
            no_improve += 1

        if scheduler is not None:
            if args.scheduler == "plateau" and val_metrics is not None:
                scheduler.step(val_metrics["loss"])
            elif args.scheduler != "plateau":
                scheduler.step()
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "args": vars(args)}, last_checkpoint_path)
        append_csv_row(log_path, row)
        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} after {no_improve} non-improving validation checks.", flush=True)
            break

    if not best_checkpoint_path.exists():
        raise RuntimeError("No best checkpoint was written. Check validation data and monitor metric.")
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    final_batch_size = min(max(batch_size_for_epoch(batch_schedule, int(best_checkpoint["epoch"])), args.batch_size), 4096)
    train_loader = make_loader(train_dataset, final_batch_size, args, shuffle=False)
    val_loader = make_loader(val_dataset, final_batch_size, args, shuffle=False)
    test_loader = make_loader(test_dataset, final_batch_size, args, shuffle=False)
    final_train_true, final_train_pred, final_train_metrics = evaluate_loader(model, train_loader, device, autocast_dtype, args.loss_alpha, "Final train")
    final_val_true, final_val_pred, final_val_metrics = evaluate_loader(model, val_loader, device, autocast_dtype, args.loss_alpha, "Final val")
    final_test_true, final_test_pred, final_test_metrics = evaluate_loader(model, test_loader, device, autocast_dtype, args.loss_alpha, "Final test")

    save_predictions(out_dir / "pred_label_train.csv", train_df, final_train_true, final_train_pred, args)
    save_predictions(out_dir / "pred_label_val.csv", val_df, final_val_true, final_val_pred, args)
    save_predictions(out_dir / "pred_label_test.csv", test_df, final_test_true, final_test_pred, args)
    save_metrics(out_dir / "final_results_train.csv", final_train_metrics)
    save_metrics(out_dir / "final_results_val.csv", final_val_metrics)
    save_metrics(out_dir / "final_results_test.csv", final_test_metrics)
    save_json(
        out_dir / "run_summary.json",
        {
            "task_name": args.task_name,
            "started_at": started_at,
            "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": time.time() - started,
            "precision_mode": precision_mode,
            "best_epoch": int(best_checkpoint["epoch"]),
            "monitor_metric": args.monitor_metric,
            "best_val_metric": float(best_checkpoint["best_val_metric"]),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "embeddings_dir": str(args.embeddings_dir),
            "final_train_metrics": final_train_metrics,
            "final_val_metrics": final_val_metrics,
            "final_test_metrics": final_test_metrics,
            "args": vars(args),
        },
    )


if __name__ == "__main__":
    main()
