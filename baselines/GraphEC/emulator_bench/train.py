from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from tqdm import tqdm

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
    seed_run_root_for_split,
    seed_train_metadata_path_for_split,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphEC on an Leak-CURBER split")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--env-name", default="graphec")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    return parser.parse_args()


def load_records(path: str | Path, label_to_idx: dict[str, int]) -> tuple[dict, dict[str, list[int]]]:
    records = pd.read_csv(path)
    dataset = {}
    labels_by_key = {}
    for _, row in records.iterrows():
        key = str(row["cache_key"])
        labels = [
            label_to_idx[label]
            for label in str(row["EC number"]).split(";")
            if label in label_to_idx
        ]
        if not labels:
            continue
        dataset[key] = [str(row["Sequence"])]
        labels_by_key[key] = labels
    if not dataset:
        raise ValueError(f"No usable rows in {path}")
    return dataset, labels_by_key


def load_active_sites(keys: list[str], cache_root: str | Path) -> dict:
    active_dir = cache_dirs(cache_root)["active_sites"]
    active_sites = {}
    missing = []
    for key in keys:
        path = active_dir / f"{key}.pt"
        if not path.exists():
            missing.append(str(path))
        else:
            active_sites[key] = torch.load(path, map_location="cpu")
    if missing:
        raise FileNotFoundError(f"Missing active-site cache files, first missing: {missing[0]}")
    return active_sites


def targets_for_names(names: list[str], labels_by_key: dict[str, list[int]], num_labels: int, device) -> torch.Tensor:
    target = torch.zeros((len(names), num_labels), dtype=torch.float32, device=device)
    for row_idx, name in enumerate(names):
        target[row_idx, labels_by_key[str(name)]] = 1.0
    return target


def padding_ver1(x, batch_id, feature_dim, activate_site):
    batch_size = max(batch_id) + 1
    max_len = max(torch.unique(batch_id, return_counts=True)[1])
    batch_data = torch.zeros([batch_size, max_len, feature_dim])
    mask = torch.zeros([batch_size, max_len])
    batch_activate_site = torch.zeros([batch_size, max_len, 1])
    len_0 = 0
    len_1 = 0
    for i in range(batch_size):
        len_1 = len_0 + torch.unique(batch_id, return_counts=True)[1][i]
        batch_data[i][: torch.unique(batch_id, return_counts=True)[1][i]] = x[len_0:len_1]
        batch_activate_site[i][: torch.unique(batch_id, return_counts=True)[1][i]] = activate_site[
            len_0:len_1
        ]
        mask[i][: torch.unique(batch_id, return_counts=True)[1][i]] = 1
        len_0 += torch.unique(batch_id, return_counts=True)[1][i]
    return batch_data, mask, batch_activate_site


def resize_output_layer(model: nn.Module, num_labels: int) -> None:
    old_layer = model.output_block[-1]
    model.output_block[-1] = nn.Linear(old_layer.in_features, num_labels)
    model.FC_2 = nn.Linear(model.hidden_dim, num_labels)


def build_model(model_mod, num_labels: int, device: torch.device) -> nn.Module:
    model = model_mod.GraphEC(
        node_input_dim=1024 + 9 + 184,
        edge_input_dim=450,
        hidden_dim=256,
        num_layers=3,
        dropout=0.1,
        augment_eps=0,
        device=device,
    )
    resize_output_layer(model, num_labels)
    return model.to(device)


def run_epoch(
    model,
    dataloader,
    labels_by_key,
    num_labels,
    device,
    criterion,
    optimizer=None,
    scaler=None,
    precision="fp32",
    desc="train",
) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    progress = tqdm(dataloader, desc=desc, unit="batch", leave=False)
    for batch in progress:
        batch_data, mask_data, batch_activate_site = padding_ver1(
            batch.node_feat, batch.batch, batch.node_feat.shape[1], batch.activate_site
        )
        batch = batch.to(device)
        batch_data = batch_data.to(device)
        mask_data = mask_data.to(device)
        batch_activate_site = batch_activate_site.to(device)
        target = targets_for_names(list(batch.name), labels_by_key, num_labels, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(**autocast_kwargs(precision)):
            logits = model(
                batch.X,
                batch.node_feat,
                batch.edge_index,
                batch.seq,
                batch.batch,
                batch_data,
                mask_data,
                batch_activate_site,
            )
            loss = criterion(logits, target)
        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        progress.set_postfix(loss=f"{losses[-1]:.4f}")
    return float(sum(losses) / max(1, len(losses)))


def train_fold(
    *,
    fold: int,
    train_indices,
    val_indices,
    dataset,
    labels_by_key,
    active_sites,
    data_mod,
    model_mod,
    num_labels: int,
    args,
    run_root: Path,
    precision: str,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = data_mod.ProteinGraphDataset(dataset, train_indices, args=None, active_sites=active_sites)
    val_dataset = data_mod.ProteinGraphDataset(dataset, val_indices, args=None, active_sites=active_sites)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    model = build_model(model_mod, num_labels, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))
    checkpoint_dir = ensure_dir(run_root / "checkpoints")
    checkpoint_path = checkpoint_dir / f"fold{fold}.pt"
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            labels_by_key,
            num_labels,
            device,
            criterion,
            optimizer=optimizer,
            scaler=scaler,
            precision=precision,
            desc=f"fold {fold} train {epoch}/{args.epochs}",
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model,
                val_loader,
                labels_by_key,
                num_labels,
                device,
                criterion,
                precision=precision,
                desc=f"fold {fold} val {epoch}/{args.epochs}",
            )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"[emulator_bench] fold={fold} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "num_labels": num_labels,
                    "precision": precision,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if args.epochs > 1 and stale_epochs >= args.patience:
            break
    return {
        "fold": fold,
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    metadata = load_run_metadata(args.split_group, args.runs_root)
    vocab = read_json(metadata["vocab_path"])
    label_to_idx = {label: int(index) for label, index in vocab["label_to_idx"].items()}
    num_labels = int(vocab["num_labels"])
    train_dataset, train_labels = load_records(metadata["manifests"]["train"], label_to_idx)
    val_dataset, val_labels = load_records(metadata["manifests"]["val"], label_to_idx)
    dataset = {**train_dataset, **val_dataset}
    labels_by_key = {**train_labels, **val_labels}
    keys = list(dataset.keys())
    ensure_graphec_data_links(keys, args.cache_root)
    active_sites = load_active_sites(keys, args.cache_root)
    precision = choose_precision(args.precision)
    run_root = ensure_dir(seed_run_root_for_split(args.split_group, args.seed, args.runs_root))

    with baseline_cwd():
        model_mod = import_task_module("EC_number", "model")
        data_mod = sys.modules["data"]
        train_keys = list(train_dataset.keys())
        if len(train_keys) < 2:
            folds = [(0, list(range(len(train_keys))), list(range(len(train_keys))))]
        else:
            split_count = min(args.folds, len(train_keys))
            kfold = KFold(n_splits=split_count, shuffle=True, random_state=args.seed)
            folds = []
            for fold, (train_idx, val_idx) in enumerate(kfold.split(train_keys)):
                folds.append((fold, train_idx.tolist(), val_idx.tolist()))

        fold_results = []
        for fold, train_indices, val_indices in folds:
            fold_results.append(
                train_fold(
                    fold=fold,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    dataset=train_dataset,
                    labels_by_key=train_labels,
                    active_sites=active_sites,
                    data_mod=data_mod,
                    model_mod=model_mod,
                    num_labels=num_labels,
                    args=args,
                    run_root=run_root,
                    precision=precision,
                )
            )

    train_metadata = {
        "split_group": args.split_group,
        "seed": args.seed,
        "run_root": str(run_root),
        "num_labels": num_labels,
        "vocab_path": metadata["vocab_path"],
        "epochs": args.epochs,
        "folds": len(fold_results),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "precision": precision,
        "fold_results": fold_results,
    }
    train_json = write_json(seed_train_metadata_path_for_split(args.split_group, args.seed, args.runs_root), train_metadata)
    write_json(Path(metadata["run_root"]) / f"train_seed{args.seed}.json", train_metadata)
    print(f"[emulator_bench] train metadata: {train_json}", flush=True)


if __name__ == "__main__":
    main()
