from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .dataset_adapter import prepare_dataset
from .utils import (
    BASELINE_ROOT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_RUNS_ROOT,
    baseline_cwd,
    cache_dirs,
    ensure_dir,
    ensure_graphec_data_links,
    import_task_module,
    load_run_metadata,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GraphEC features for Leak-CURBER splits")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-group", action="append", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--max-seq-length", type=int, default=1022)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--prottrans-model-path", default=os.environ.get("PROTTRANS_MODEL_PATH"))
    parser.add_argument("--prottrans-batch-size", type=int, default=1)
    parser.add_argument("--active-site-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-feature-generation", action="store_true")
    return parser.parse_args()


def manifest_items(metadata: dict) -> list[dict]:
    seen = {}
    for split_name, manifest_path in metadata["manifests"].items():
        records = pd.read_csv(manifest_path)
        for row in records.itertuples(index=False):
            key = str(row.cache_key)
            if key not in seen:
                seen[key] = {"cache_key": key, "sequence": str(row.Sequence)}
    return list(seen.values())


def write_items_fasta(items: list[dict], fasta_path: Path) -> None:
    ensure_dir(fasta_path.parent)
    with fasta_path.open("w") as handle:
        for item in items:
            handle.write(f">{item['cache_key']}\n{item['sequence']}\n")


def missing_items(items: list[dict], directory: Path, suffix: str) -> list[dict]:
    return [item for item in items if not (directory / f"{item['cache_key']}{suffix}").exists()]


def generate_esmfold_features(items: list[dict], dirs: dict[str, Path]) -> None:
    if not items:
        return
    fasta = dirs["tmp"] / "missing_structures.fasta"
    write_items_fasta(items, fasta)
    with baseline_cwd():
        import Features.features as features

        print(f"[emulator_bench] generating ESMFold structures for {len(items)} entries", flush=True)
        subprocess.run(
            [
                sys.executable,
                "./Features/esmfold/esmfold.py",
                "-i",
                str(fasta),
                "-o",
                str(dirs["structures"]) + "/",
                "--chunk-size",
                "128",
            ],
            check=True,
        )
        for item in items:
            pdb_path = dirs["structures"] / f"{item['cache_key']}.pdb"
            if not pdb_path.exists():
                raise FileNotFoundError(f"Missing ESMFold output: {pdb_path}")
            with pdb_path.open() as handle:
                coord = features.get_pdb_xyz(handle.readlines())
            torch.save(torch.tensor(coord, dtype=torch.float32), dirs["structures"] / f"{item['cache_key']}.tensor")


def generate_dssp_features(items: list[dict], dirs: dict[str, Path]) -> None:
    if not items:
        return
    fasta = dirs["tmp"] / "missing_dssp.fasta"
    write_items_fasta(items, fasta)
    with baseline_cwd():
        import Features.features as features

        print(f"[emulator_bench] generating DSSP features for {len(items)} entries", flush=True)
        features.get_dssp(
            str(fasta),
            "./Features/dssp-2.0.4/",
            str(dirs["structures"]) + "/",
            str(dirs["dssp"]) + "/",
        )


def generate_prottrans_features(
    items: list[dict],
    dirs: dict[str, Path],
    *,
    model_path: str | None,
    batch_size: int,
) -> None:
    if not items:
        return
    if not model_path:
        raise ValueError(
            "Missing ProtTrans model path. Pass --prottrans-model-path or set PROTTRANS_MODEL_PATH."
        )
    from transformers import T5EncoderModel, T5Tokenizer

    import re

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path).eval().to(device)
    print(
        f"[emulator_bench] generating ProtTrans features for {len(items)} entries on {device}",
        flush=True,
    )
    for start in tqdm(range(0, len(items), batch_size), desc="ProtTrans", unit="batch"):
        batch = items[start : start + batch_size]
        ids = [item["cache_key"] for item in batch]
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", item["sequence"]))) for item in batch]
        tokens = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(tokens["input_ids"], device=device)
        attention_mask = torch.tensor(tokens["attention_mask"], device=device)
        with torch.no_grad():
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embeddings = embeddings.detach().cpu()
        for index, cache_key in enumerate(ids):
            seq_len = int((attention_mask[index] == 1).sum().item())
            torch.save(embeddings[index, : seq_len - 1].clone(), dirs["prottrans"] / f"{cache_key}.tensor")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_active_site_features(
    items: list[dict],
    dirs: dict[str, Path],
    *,
    batch_size: int,
    num_workers: int,
) -> None:
    missing = missing_items(items, dirs["active_sites"], ".pt")
    if not missing:
        return
    ensure_graphec_data_links([item["cache_key"] for item in missing], dirs["structures"].parents[0])
    with baseline_cwd():
        model_mod = import_task_module("Active_sites", "model")
        data_mod = __import__("data")
        from torch_geometric.loader import DataLoader
        import torch_geometric

        dataset_dict = {item["cache_key"]: [item["sequence"]] for item in missing}
        dataset = data_mod.ProteinGraphDataset(dataset_dict, range(len(dataset_dict)), args=None)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = []
        for fold in range(5):
            state_dict = torch.load(BASELINE_ROOT / "Active_sites/model" / f"fold{fold}.ckpt", device)
            model = model_mod.GraphEC_AS(
                1024 + 9 + 184,
                450,
                128,
                4,
                0.2,
                0.1,
                task="ActiveSite",
            ).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)

        print(
            f"[emulator_bench] generating active-site features for {len(missing)} entries",
            flush=True,
        )
        for batch in tqdm(dataloader, desc="GraphEC-AS", unit="batch"):
            batch = batch.to(device)
            with torch.no_grad():
                outputs = [model(batch.X, batch.node_feat, batch.edge_index, batch.seq, batch.batch).sigmoid() for model in models]
                outputs = torch.stack(outputs, 0).mean(0)
            for name, tensor in zip(batch.name, torch_geometric.utils.unbatch(outputs, batch.batch)):
                torch.save([tensor.detach().cpu()], dirs["active_sites"] / f"{name}.pt")


def prepare_and_cache(args: argparse.Namespace) -> list[dict]:
    metadata_list = prepare_dataset(
        dataset_root=args.dataset_root,
        split_groups=args.split_group,
        runs_root=args.runs_root,
        max_sequence_length=args.max_seq_length,
        limit_per_split=args.limit_per_split,
    )
    dirs = cache_dirs(args.cache_root)
    all_items = {}
    for metadata in metadata_list:
        for item in manifest_items(metadata):
            all_items.setdefault(item["cache_key"], item)
    items = list(all_items.values())
    print(f"[emulator_bench] unique cache entries: {len(items)}", flush=True)

    if not args.skip_feature_generation:
        generate_esmfold_features(missing_items(items, dirs["structures"], ".tensor"), dirs)
        generate_prottrans_features(
            missing_items(items, dirs["prottrans"], ".tensor"),
            dirs,
            model_path=args.prottrans_model_path,
            batch_size=args.prottrans_batch_size,
        )
        generate_dssp_features(missing_items(items, dirs["dssp"], ".tensor"), dirs)
        ensure_graphec_data_links([item["cache_key"] for item in items], args.cache_root)
        generate_active_site_features(
            items,
            dirs,
            batch_size=args.active_site_batch_size,
            num_workers=args.num_workers,
        )
    else:
        print("[emulator_bench] skipping feature generation by request", flush=True)

    for metadata in metadata_list:
        metadata["cache_root"] = str(Path(args.cache_root))
        metadata["feature_cache"] = {key: str(path) for key, path in dirs.items()}
        write_json(metadata["metadata_path"], metadata)
    return metadata_list


def main() -> None:
    args = parse_args()
    prepare_and_cache(args)


if __name__ == "__main__":
    main()
