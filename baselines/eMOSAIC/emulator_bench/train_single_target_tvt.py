import argparse
import datetime
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from emulator_bench.dataset import CachedEmosaicDataset, EmosaicDataLoader
from emulator_bench.feature_pipeline import LigandGraphStore, ProteinEmbeddingStore, resolve_amp_dtype
from emulator_bench.modeling import build_model


MINIMIZE_METRICS = {"rmse", "mse", "mae", "loss"}


def _autocast_context(device: torch.device, dtype=None):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _build_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int(args.cosine_t0)
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def _prepare_batch(batch, device: torch.device):
    batch = batch.to(device, non_blocking=True)
    if device.type != "cuda":
        batch.prot_emb = batch.prot_emb.float()  # float16 unsupported for BN on CPU
    # on CUDA: prot_emb stays float16, autocast handles dtype per-op
    # prot_mask stays bool — model does .float() in forward
    if hasattr(batch, "value"):
        batch.value = batch.value.float()
    return batch


def _metric_direction(metric_name: str) -> str:
    return "minimize" if metric_name in MINIMIZE_METRICS else "maximize"


def _monitor_metric_from_arrays(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return float("nan")

    residual = y_true - y_pred
    if metric_name in {"loss", "mse"}:
        return float(np.mean(np.square(residual)))
    if metric_name == "rmse":
        return float(np.sqrt(np.mean(np.square(residual))))
    if metric_name == "mae":
        return float(np.mean(np.abs(residual)))
    if metric_name == "r2_score":
        ss_res = float(np.sum(np.square(residual)))
        ss_tot = float(np.sum(np.square(y_true - y_true.mean())))
        return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    if metric_name == "pearson":
        if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        return float(np.corrcoef(y_true, y_pred)[0, 1])
    if metric_name == "spearman":
        try:
            from scipy import stats

            value = float(stats.spearmanr(y_true, y_pred).statistic)
            return 0.0 if math.isnan(value) else value
        except Exception:
            true_ranks = np.argsort(np.argsort(y_true))
            pred_ranks = np.argsort(np.argsort(y_pred))
            if np.std(true_ranks) == 0 or np.std(pred_ranks) == 0:
                return 0.0
            return float(np.corrcoef(true_ranks, pred_ranks)[0, 1])
    raise ValueError("Unsupported monitor_metric: %s" % metric_name)


def evaluate_loader(model, loader, device, autocast_dtype=None, desc="Evaluation", metric_name="rmse", full_metrics=False, show_progress=True):
    model.eval()
    preds = []
    truths = []
    total_loss = 0.0
    total_samples = 0
    loss_fn = torch.nn.MSELoss(reduction="mean")
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False) if show_progress else loader
    with torch.no_grad():
        for batch in iterator:
            batch = _prepare_batch(batch, device)
            with _autocast_context(device, autocast_dtype):
                prediction = model(batch)
                target = batch.value.view_as(prediction)
                loss = loss_fn(prediction, target)
            n = int(batch.num_graphs)
            total_loss += float(loss.item()) * n
            total_samples += n
            preds.append(prediction.detach().cpu().float())
            truths.append(target.detach().cpu().float())
            del batch, loss
    pred_np = torch.cat(preds).numpy() if preds else np.array([], dtype=np.float32)
    truth_np = torch.cat(truths).numpy() if truths else np.array([], dtype=np.float32)
    avg_loss = total_loss / max(1, total_samples)
    if full_metrics:
        metrics = regression_metrics(truth_np, pred_np)
        metrics["loss"] = avg_loss
    else:
        metrics = {
            "loss": avg_loss,
            metric_name: avg_loss if metric_name == "loss" else _monitor_metric_from_arrays(metric_name, truth_np, pred_np),
        }
    return truth_np, pred_np, metrics


def train_one_epoch(model, loader, optimizer, device, scaler, autocast_dtype=None, clip_grad=None, desc="Train"):
    model.train()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_samples = 0
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False)
    for batch in iterator:
        batch = _prepare_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, autocast_dtype):
            prediction = model(batch)
            target = batch.value.view_as(prediction)
            loss = loss_fn(prediction, target)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if clip_grad is not None and clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad is not None and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        n = int(batch.num_graphs)
        total_loss += float(loss.item()) * n
        total_samples += n
        iterator.set_postfix(loss="%.4f" % float(loss.item()))
        del batch, loss
    return {"loss": total_loss / max(1, total_samples)}


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)


def save_metrics(path: Path, metrics: dict) -> None:
    pd.DataFrame([metrics]).to_csv(path, index=False)


def _resolve_paths(args):
    if args.train_path and args.val_path and args.test_path:
        return Path(args.train_path), Path(args.val_path), Path(args.test_path), None
    if not args.base_dir or not args.split_group:
        raise ValueError("Provide either explicit --train_path/--val_path/--test_path or --base_dir with --split_group.")
    job = resolve_single_split_job(Path(args.base_dir), split_group=args.split_group, threshold=args.threshold)
    return Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"]), job


def _default_out_dir(args, job):
    if args.out_dir:
        return Path(args.out_dir)
    if job is None:
        raise ValueError("--out_dir is required when explicit train/val/test paths are used.")
    return Path(job["root_dir"]) / args.results_dirname / f"seed_{args.seed}"


def _filter_by_max_length(df: pd.DataFrame, sequence_col: str, max_length: int) -> pd.DataFrame:
    if max_length is None or max_length <= 0:
        return df
    df = df.copy()
    df[sequence_col] = df[sequence_col].astype(str).str[:max_length]
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Train eMOSAIC TrustAffinity regressor on explicit train/val/test split files.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--split_group", type=str, default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--task_name", type=str, default="emosaic_retrain")

    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")

    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--max_length", type=int, default=700)
    parser.add_argument("--prot_input_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--amsgrad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scheduler", choices=["none", "cosine_warm_restarts"], default="cosine_warm_restarts")
    parser.add_argument("--cosine_t0", type=int, default=10)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--val_every", type=int, default=1, help="Validate every N epochs. 0 = end of training only.")
    parser.add_argument("--monitor_metric", choices=["rmse", "pearson", "spearman", "r2_score", "mae", "mse", "loss"], default="rmse")

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=512)
    parser.add_argument("--lazy_ligands", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    set_seed(args.seed)
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

    train_df = _filter_by_max_length(train_df, args.sequence_col, args.max_length)
    val_df = _filter_by_max_length(val_df, args.sequence_col, args.max_length)
    test_df = _filter_by_max_length(test_df, args.sequence_col, args.max_length)

    all_sequences = pd.concat(
        [train_df[args.sequence_col], val_df[args.sequence_col], test_df[args.sequence_col]],
        ignore_index=True,
    ).astype(str)
    all_smiles = pd.concat(
        [train_df[args.smiles_col], val_df[args.smiles_col], test_df[args.smiles_col]],
        ignore_index=True,
    ).astype(str)

    protein_store = ProteinEmbeddingStore(
        Path(args.embeddings_dir),
        sequences=all_sequences.tolist(),
        preload=args.preload_proteins,
        max_items=args.protein_cache_items,
    )
    ligand_store = LigandGraphStore(
        Path(args.embeddings_dir),
        smiles_values=all_smiles.tolist(),
        preload=not args.lazy_ligands,
    )

    sample_sequence = str(train_df[args.sequence_col].iloc[0])
    sample_cached = protein_store.get(sample_sequence)
    detected_prot_dim = int(sample_cached["embedding"].shape[1])
    if args.prot_input_dim != detected_prot_dim:
        print(
            f"[info] Overriding --prot_input_dim {args.prot_input_dim} with detected {detected_prot_dim} from cache.",
            flush=True,
        )
        args.prot_input_dim = detected_prot_dim

    train_dataset = CachedEmosaicDataset(train_df, protein_store, ligand_store, args.max_length, args.sequence_col, args.smiles_col, args.target_col)
    val_dataset = CachedEmosaicDataset(val_df, protein_store, ligand_store, args.max_length, args.sequence_col, args.smiles_col, args.target_col)
    test_dataset = CachedEmosaicDataset(test_df, protein_store, ligand_store, args.max_length, args.sequence_col, args.smiles_col, args.target_col)

    pin_memory = args.pin_memory or device.type == "cuda"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = EmosaicDataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = EmosaicDataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = EmosaicDataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = build_model(
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        max_length=args.max_length,
        prot_input_dim=args.prot_input_dim,
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
    scheduler = _build_scheduler(optimizer, args)

    log_path = out_dir / "logfile.csv"
    best_checkpoint_path = out_dir / "bestmodel.pth"
    best_state_dict_path = out_dir / "bestmodel_state_dict.pth"
    last_checkpoint_path = out_dir / "checkpoint_last.pt"
    run_summary_path = out_dir / "run_summary.json"
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    started = time.time()
    monitor_direction = _metric_direction(args.monitor_metric)
    best_val_metric = float("inf") if monitor_direction == "minimize" else float("-inf")

    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        capability = ".".join(map(str, torch.cuda.get_device_capability(device_index)))
        print(f"CUDA device: {gpu_name} | compute capability: {capability} | precision: {precision_mode}", flush=True)
    else:
        print(f"Device: {device} | precision: {precision_mode}", flush=True)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
            clip_grad=args.clip_grad,
            desc=f"Epoch {epoch} train",
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "elapsed_seconds": time.time() - started,
        }

        val_metrics = None
        run_val = (epoch == args.epochs) if args.val_every == 0 else (epoch % args.val_every == 0 or epoch == args.epochs)
        if run_val:
            _val_true, _val_pred, val_metrics = evaluate_loader(
                model,
                val_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                desc=f"Epoch {epoch} val",
                metric_name=args.monitor_metric,
                full_metrics=False,
                show_progress=False,
            )
            row.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_%s" % args.monitor_metric: val_metrics[args.monitor_metric],
                }
            )

            current_val_metric = float(val_metrics[args.monitor_metric])
            if monitor_direction == "minimize":
                improved = current_val_metric < best_val_metric
            else:
                improved = current_val_metric > best_val_metric
            if improved:
                best_val_metric = current_val_metric
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_metric": best_val_metric,
                    "monitor_metric": args.monitor_metric,
                    "args": vars(args),
                    "precision_mode": precision_mode,
                }
                torch.save(checkpoint, best_checkpoint_path)
                torch.save(model.state_dict(), best_state_dict_path)

        if scheduler is not None:
            scheduler.step()

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
            last_checkpoint_path,
        )
        append_csv_row(log_path, row)

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    final_train_true, final_train_pred, final_train_metrics = evaluate_loader(
        model, train_loader, device=device, autocast_dtype=autocast_dtype, desc="Final train", metric_name=args.monitor_metric, full_metrics=True,
    )
    final_val_true, final_val_pred, final_val_metrics = evaluate_loader(
        model, val_loader, device=device, autocast_dtype=autocast_dtype, desc="Final val", metric_name=args.monitor_metric, full_metrics=True,
    )
    final_test_true, final_test_pred, final_test_metrics = evaluate_loader(
        model, test_loader, device=device, autocast_dtype=autocast_dtype, desc="Final test", metric_name=args.monitor_metric, full_metrics=True,
    )

    save_predictions(out_dir / "pred_label_train.csv", final_train_true, final_train_pred)
    save_predictions(out_dir / "pred_label_val.csv", final_val_true, final_val_pred)
    save_predictions(out_dir / "pred_label_test.csv", final_test_true, final_test_pred)
    save_metrics(out_dir / "final_results_train.csv", final_train_metrics)
    save_metrics(out_dir / "final_results_val.csv", final_val_metrics)
    save_metrics(out_dir / "final_results_test.csv", final_test_metrics)

    summary = {
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
    }
    save_json(run_summary_path, summary)
    pd.DataFrame([summary]).drop(columns=["final_train_metrics", "final_val_metrics", "final_test_metrics", "args"]).to_csv(
        out_dir / "run_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
