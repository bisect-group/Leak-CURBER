import argparse
import sys
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import enable_fast_torch_math, read_table, require_columns, resolve_amp_dtype, save_json
from emulator_bench.dataset import BACPIDataset, CompoundCacheStore, ProteinCacheStore, create_loader
from emulator_bench.feature_pipeline import BACPIFeaturizer, filter_invalid_smiles_rows
from emulator_bench.modeling_v2 import BACPIModelConfig, build_model
from emulator_bench.train_single_target_tvt import evaluate_loader


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved BACPI checkpoint on a single split file.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split_path", type=str, required=True)
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pad_to_multiple", type=int, default=1)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--skip_smiles_validity_check", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = Path(args.embeddings_dir).expanduser().resolve()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint.get("args", {})

    enable_fast_torch_math()
    device = torch.device(args.device)
    autocast_dtype, precision_mode = resolve_amp_dtype(device)

    featurizer = BACPIFeaturizer.load(embeddings_dir / "vocabs" / "featurizer.pkl")
    split_df = read_table(Path(args.split_path))
    require_columns(split_df, [args.sequence_col, args.smiles_col, args.target_col], Path(args.split_path))
    if args.skip_smiles_validity_check:
        split_df = split_df.reset_index(drop=True)
        filter_stats = {"rows_dropped": 0}
    else:
        split_df, filter_stats = filter_invalid_smiles_rows(split_df, smiles_col=args.smiles_col, source_name=str(Path(args.split_path)))
        if filter_stats["rows_dropped"] > 0:
            print("Dropped %s rows with invalid SMILES from %s" % (filter_stats["rows_dropped"], args.split_path), flush=True)
        if split_df.empty:
            raise ValueError("No valid rows remain in %s after dropping invalid SMILES." % args.split_path)

    print(
        "Predict split size=%s | unique compounds=%s unique proteins=%s"
        % (
            len(split_df),
            split_df[args.smiles_col].astype(str).nunique(),
            split_df[args.sequence_col].astype(str).nunique(),
        ),
        flush=True,
    )
    compound_store = CompoundCacheStore(
        embeddings_dir,
        smiles_values=split_df[args.smiles_col].astype(str).tolist(),
        preload=True,
        preload_desc="Preloading compounds",
    )
    protein_store = ProteinCacheStore(
        embeddings_dir,
        sequences=split_df[args.sequence_col].astype(str).tolist(),
        preload=True,
        preload_desc="Preloading proteins",
    )
    dataset = BACPIDataset(split_df, compound_store, protein_store, args.smiles_col, args.sequence_col, args.target_col)
    batch_size = args.batch_size or int(train_args.get("batch_size", 16))
    loader, _ = create_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=args.prefetch_factor,
        persistent_workers=(args.num_workers > 0),
        bucket_size_multiplier=50,
        pad_to_multiple=args.pad_to_multiple,
        seed=args.seed,
    )

    model_config = BACPIModelConfig(
        gat_dim=int(train_args.get("gat_dim", 50)),
        num_head=int(train_args.get("num_head", 3)),
        dropout=float(train_args.get("dropout", 0.1)),
        alpha=float(train_args.get("alpha", 0.1)),
        comp_dim=int(train_args.get("comp_dim", 80)),
        prot_dim=int(train_args.get("prot_dim", 80)),
        latent_dim=int(train_args.get("latent_dim", 80)),
        window=int(train_args.get("window", 5)),
        layer_cnn=int(train_args.get("layer_cnn", 3)),
        layer_out=int(train_args.get("layer_out", 3)),
    )
    model = build_model(featurizer.n_atom_tokens, featurizer.n_amino_tokens, model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred, metrics = evaluate_loader(
        model,
        loader,
        device=device,
        autocast_dtype=autocast_dtype,
        desc="Predict",
        metric_name=str(checkpoint.get("monitor_metric", "rmse")),
        show_progress=True,
    )
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    save_json(
        out_dir / "summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "split_path": str(Path(args.split_path).expanduser().resolve()),
            "precision_mode": precision_mode,
            "invalid_smiles_rows_dropped": int(filter_stats["rows_dropped"]),
            "metrics": metrics,
        },
    )


if __name__ == "__main__":
    main()
