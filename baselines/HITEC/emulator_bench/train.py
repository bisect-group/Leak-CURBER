from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from torch.utils.data import DataLoader, Dataset

from model.model import Transformer

from .utils import (
    DEFAULT_MODEL_DIMENSION,
    DEFAULT_RUNS_ROOT,
    choose_precision,
    ensure_dir,
    lightning_precision_arg,
    load_run_metadata,
    resolve_model_max_tokens,
    seed_run_root,
    seed_train_metadata_path,
    write_json,
)


DEFAULT_CONFIG = {
    "ah": 2,
    "dr": 0.1,
    "beta": 0.59,
    "lr": 8.75e-5,
    "dimension": DEFAULT_MODEL_DIMENSION,
    "vocab_size": 23,
    "swa_lrs": 3e-7,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train HITEC on an Leak-CURBER split")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--env-name", default="hitec")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--model-dimension", type=int, default=DEFAULT_CONFIG["dimension"])
    parser.add_argument("--no-swa", action="store_true")
    parser.add_argument("--limit-train-batches", default=None)
    parser.add_argument("--limit-val-batches", default=None)
    return parser.parse_args()


def _optional_float(value):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return float(value)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


class CachedHitecDataset(Dataset):
    def __init__(self, manifest_path, total_outputs, level_starts, output_dims):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError("Missing manifest: {}".format(self.manifest_path))
        self.records = pd.read_csv(self.manifest_path)
        self.total_outputs = int(total_outputs)
        self.level_starts = list(level_starts)
        self.output_dims = list(output_dims)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        cache_path = Path(row["token_cache_path"])
        if not cache_path.exists():
            raise FileNotFoundError("Missing token cache: {}".format(cache_path))
        cached = torch.load(cache_path, map_location="cpu")
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
        return tokens, target, mask


class HitecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        metadata,
        batch_size=2,
        num_workers=0,
    ):
        super().__init__()
        self.metadata = metadata
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.total_outputs = int(sum(metadata["output_dims"]))
        self.level_starts = metadata["level_starts"]
        self.output_dims = metadata["output_dims"]

    def setup(self, stage=None):
        self.train_dataset = CachedHitecDataset(
            self.metadata["manifests"]["train"],
            self.total_outputs,
            self.level_starts,
            self.output_dims,
        )
        self.val_dataset = CachedHitecDataset(
            self.metadata["manifests"]["val"],
            self.total_outputs,
            self.level_starts,
            self.output_dims,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


class HitecLightningModule(pl.LightningModule):
    def __init__(
        self,
        output_dims,
        ah=2,
        dr=0.1,
        beta=0.59,
        lr=8.75e-5,
        dimension=DEFAULT_MODEL_DIMENSION,
        vocab_size=23,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(vocab_size, dimension, ah, output_dims, dr, beta)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.lr = float(lr)

    def forward(self, tokens):
        return self.model(tokens.long(), mode="infer")

    def _masked_loss(self, batch):
        tokens, target, mask = batch
        logits = self(tokens)
        losses = self.criterion(logits, target.float())
        denom = torch.clamp(mask.sum(), min=1.0)
        return (losses * mask).sum() / denom

    def training_step(self, batch, batch_idx):
        loss = self._masked_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._masked_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt_parameters = []
        no_decay = ["norm", "bias"]
        init_lr = self.lr
        linear_lr = self.lr * 1.05

        for name, params in self.named_parameters():
            weight_decay = 0.0 if any(item in name for item in no_decay) else 0.01
            if name.startswith("model.embeddings") or name.startswith("model.enc_1"):
                lr = init_lr * (0.9 ** 3)
            elif name.startswith("model.enc_2"):
                lr = init_lr * (0.9 ** 2)
            elif name.startswith("model.enc_3"):
                lr = init_lr * 0.9
            elif name.startswith("model.enc_4"):
                lr = init_lr
            elif "linear" in name:
                lr = linear_lr
            else:
                lr = init_lr
            opt_parameters.append({"params": params, "weight_decay": weight_decay, "lr": lr})

        optimizer = torch.optim.AdamW(opt_parameters, lr=init_lr)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer,
            lr_lambda=lambda epoch: 0.95,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def resolve_accelerator(requested):
    if requested != "auto":
        return requested
    return "gpu" if torch.cuda.is_available() else "cpu"


def checkpoint_from_callback(callback):
    candidates = [callback.last_model_path, callback.best_model_path]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    ckpts = sorted(Path(callback.dirpath).glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError("Training finished but no checkpoint was found in {}".format(callback.dirpath))


def main():
    args = parse_args()
    set_seed(args.seed)
    metadata = load_run_metadata(args.split_group, args.runs_root)
    seed_root = seed_run_root(metadata, args.seed)
    checkpoint_dir = ensure_dir(seed_root / "checkpoints")
    selected_precision = choose_precision(args.precision)
    precision = lightning_precision_arg(selected_precision)
    accelerator = resolve_accelerator(args.accelerator)

    config = dict(DEFAULT_CONFIG)
    config["lr"] = float(args.learning_rate)
    config["dimension"] = int(args.model_dimension)
    config["output_dims"] = metadata["output_dims"]
    metadata_max_tokens = int(metadata.get("max_tokens", config["dimension"]))
    resolve_model_max_tokens(metadata_max_tokens, config["dimension"])

    model = HitecLightningModule(
        output_dims=metadata["output_dims"],
        ah=config["ah"],
        dr=config["dr"],
        beta=config["beta"],
        lr=config["lr"],
        dimension=config["dimension"],
        vocab_size=config["vocab_size"],
    )
    data_module = HitecDataModule(
        metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch-{epoch:03d}",
        every_n_epochs=1,
        save_top_k=-1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_callback]
    if not args.no_swa and args.epochs > 1:
        swa_start = max(1, args.epochs - 15)
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=DEFAULT_CONFIG["swa_lrs"],
                swa_epoch_start=swa_start,
            )
        )

    trainer_kwargs = {
        "max_epochs": int(args.epochs),
        "accelerator": accelerator,
        "devices": int(args.devices) if accelerator != "cpu" else None,
        "precision": precision,
        "callbacks": callbacks,
        "enable_progress_bar": True,
        "num_sanity_val_steps": 0,
        "default_root_dir": str(seed_root),
        "logger": False,
    }
    if trainer_kwargs["devices"] is None:
        trainer_kwargs.pop("devices")
    limit_train_batches = _optional_float(args.limit_train_batches)
    limit_val_batches = _optional_float(args.limit_val_batches)
    if limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = limit_train_batches
    if limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=data_module)
    checkpoint = checkpoint_from_callback(checkpoint_callback)

    train_metadata = {
        "split_group": args.split_group,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "precision_requested": args.precision,
        "precision_selected": selected_precision,
        "precision": str(precision),
        "accelerator": accelerator,
        "devices": int(args.devices),
        "model_dimension": int(config["dimension"]),
        "max_tokens": metadata_max_tokens,
        "checkpoint": str(checkpoint),
        "checkpoint_dir": str(checkpoint_dir),
        "seed_run_root": str(seed_root),
        "config": config,
        "train_rows": metadata["stats"]["train"]["rows_after_label_filter"],
        "val_rows": metadata["stats"]["val"]["rows_after_label_filter"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json(seed_train_metadata_path(metadata, args.seed), train_metadata)
    print("[emulator_bench] checkpoint: {}".format(checkpoint), flush=True)


if __name__ == "__main__":
    main()
