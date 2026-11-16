"""Shared builders for the test suite."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra

from hydra import compose, initialize_config_dir
from src.data.baseline_regression_datamodule import BaselineRegressionDataModule
from src.data.components.embedders.store import ShardedEmbeddingStore
from src.data.components.embedders.types import CacheMetadata
from src.data.components.embedders.utils import sha256_short

REPO_ROOT = Path(__file__).resolve().parents[2]


class SilentLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


def load_script(name: str) -> ModuleType:
    """Import scripts/<name>.py once and register it in sys.modules under its name."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compose_config(config_name: str, overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(str(REPO_ROOT / "configs"), version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides or [])


# Embedding caches and split files ---------------------------------------------------------


def write_cache(
    root: Path,
    embedder: str | None = None,
    values: dict[str, list[float]] | None = None,
    *,
    dim: int = 1,
) -> Path:
    """Write a one-shard embedding cache.

    With `values`, each key gets its vector and a hashed index row. Without it, the cache holds
    one all-ones row of width `dim`; `embedder` defaults to the fourth path part from the end.
    """
    root.mkdir(parents=True, exist_ok=True)
    if values is None:
        embedder = embedder or (root.parts[-4] if len(root.parts) >= 4 else "test")
        array = np.ones((1, dim), dtype=np.float32)
        index = [{"key_hash": "abc", "shard_id": 0, "row_idx": 0}]
    else:
        dim = len(next(iter(values.values())))
        array = np.asarray(list(values.values()), dtype=np.float32)
        index = [
            {"key_hash": sha256_short(key), "shard_id": 0, "row_idx": idx}
            for idx, key in enumerate(values)
        ]
    np.save(root / "shard_000000.npy", array, allow_pickle=False)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "embedder": embedder,
                "model_name": "test",
                "version": "v1",
                "storage_dtype": "float32",
                "embedding_dim": dim,
                "max_shard_bytes": 1024,
                "key_type": "test",
                "key_field": "test",
            }
        )
    )
    pd.DataFrame(index).to_parquet(root / "index.parquet", index=False)
    return root


def write_tvt(split_dir: Path, *, labels=(1.0, 1.0, 1.0), include_pdb_ids: bool = True) -> None:
    """Write one-row train/val/test regression splits with the given label per split."""
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, label in zip(("train", "val", "test"), labels):
        payload = {"smiles": ["C"], "sequence": ["SEQ"]}
        if include_pdb_ids:
            payload["pdb_ids"] = [["P12345"]]
        payload["log10_value"] = [label]
        pd.DataFrame(payload).to_parquet(split_dir / f"{split_name}.parquet", index=False)


def write_classification_tvt(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        "train": [
            {"sequence": "SEQ1", "pdb_ids": ["P1"], "ec_number": "1.2.3.4"},
            {"sequence": "SEQ2", "pdb_ids": ["P2"], "ec_number": "2.7.7.7"},
        ],
        "val": [{"sequence": "SEQ1", "pdb_ids": ["P1"], "ec_number": "1.2.3.4"}],
        "test": [{"sequence": "SEQ2", "pdb_ids": ["P2"], "ec_number": "2.7.7.7"}],
    }
    for split_name, payload in rows.items():
        pd.DataFrame(payload).to_parquet(split_dir / f"{split_name}.parquet", index=False)


def write_retrieval_tvt(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"rxn_key": ["A>>B", "C>>D"], "ec_number": ["1.1.1.1", "2.2.2.2"]})
    for split_name in ("train", "val", "test"):
        frame.to_parquet(split_dir / f"{split_name}.parquet", index=False)


# Splitters --------------------------------------------------------------------------------

KEYS = [f"r{i}" for i in range(14)]


def make_embeddings(dim: int, binary: bool) -> dict[str, np.ndarray]:
    """r0-r1 all-zero, r2 missing, r3 unique bit (valid, similarity 0), rest random."""
    rng = np.random.default_rng(0)
    embs = {}
    for i, key in enumerate(KEYS):
        if i == 2:
            continue
        vec = np.zeros(dim, dtype=np.uint8 if binary else np.float32)
        if i == 3:
            vec[dim - 1] = 1
        elif i > 3:
            vec[rng.choice(dim - 1, 3, replace=False)] = 1 if binary else rng.random(3)
        embs[key] = vec
    return embs


def make_splitter(cls, tmp_path: Path, embs: dict, **attrs):
    splitter = object.__new__(cls)
    splitter.KEY_COLUMN = "k"
    splitter.TEST_FRAC = 0.3
    splitter.MATCH_VAL_TO_TEST = True
    splitter.VAL_FRAC = 0.3
    splitter.CACHE_OVERWRITE = False
    splitter.COSINE_BATCH_SIZE = 4
    splitter.unique_keys = KEYS
    splitter.full_dataset_df = pd.DataFrame({"k": KEYS})
    splitter.logger = MagicMock()
    splitter.MAX_SIM_CACHE_PATH = tmp_path / "cache.tsv"
    splitter._valid_embedding_mask = None
    splitter._max_tanimoto_array = splitter._max_tanimoto_to_dataset = None
    splitter._max_cosine_array = splitter._max_cosine_to_dataset = None
    splitter._load_per_key_embeddings = lambda: embs
    # Cache fingerprints read the embedding source and cosine parameters.
    splitter._embedding_source, splitter.EMBEDDING_COLUMN = "column", "emb"
    splitter.EMBEDDING_LOOKUP_MODE, splitter.DATAMOL_NUM_CONFORMERS = "direct", 1
    splitter.EMBEDDING_KEY_TEMPLATES = []
    for name, value in attrs.items():
        setattr(splitter, name, value)
    return splitter


def split_keys(splitter, threshold: float) -> dict[str, set]:
    splits, _ = splitter.dissimilarity_based_split(threshold)
    return {name: set(df["k"]) for name, df in splits.items()}


def reaction_split(tmp_path: Path, keys: list[str]) -> dict[str, set]:
    from src.data.components.splitters.reaction_tanimoto import (
        ReactionDRFPTanimotoSimilaritySplitter,
    )

    splitter = make_splitter(
        ReactionDRFPTanimotoSimilaritySplitter, tmp_path, {}, TEST_FRAC=0.3, VAL_FRAC=0.3
    )
    splitter.unique_keys = keys
    splitter.full_dataset_df = pd.DataFrame({"k": keys})
    splitter._valid_embedding_mask = np.ones(len(keys), dtype=bool)
    splitter._max_tanimoto_array = np.full(len(keys), 0.01)  # every key ties
    splitter._max_tanimoto_to_dataset = dict(zip(keys, splitter._max_tanimoto_array))
    splitter.compute_max_tanimoto_to_dataset = lambda: splitter._max_tanimoto_array
    splitter.get_max_tanimoto_for_keys = lambda ks: {k: 0.01 for k in ks}
    splits, _ = splitter.dissimilarity_based_split(0.1)
    return {name: set(df["k"]) for name, df in splits.items()}


# Embedding stores -------------------------------------------------------------------------


def make_store(
    root: Path, settings: str = "", shard_bytes: int = 1 << 20
) -> ShardedEmbeddingStore:
    metadata = CacheMetadata(
        embedder="t",
        model_name="m",
        version="v1",
        storage_dtype="float32",
        embedding_dim=None,
        max_shard_bytes=shard_bytes,
        key_type="t",
        key_field="t",
        settings=settings,
    )
    return ShardedEmbeddingStore(root, metadata, logging.getLogger("t"))


def store_metadata(max_shard_bytes: int = 1024 * 1024) -> CacheMetadata:
    return CacheMetadata(
        embedder="test",
        model_name="test",
        version="v1",
        storage_dtype="float32",
        embedding_dim=2,
        max_shard_bytes=max_shard_bytes,
        key_type="test",
        key_field="test",
    )


def write_shard(root: Path, shard_id: int, keys: list[str]) -> list[dict]:
    values = np.asarray(
        [[float(shard_id), float(row_idx)] for row_idx in range(len(keys))],
        dtype=np.float32,
    )
    np.save(root / f"shard_{shard_id:06d}.npy", values, allow_pickle=False)
    return [
        {"key_hash": sha256_short(key), "shard_id": shard_id, "row_idx": row_idx}
        for row_idx, key in enumerate(keys)
    ]


def write_store_files(root: Path, shard_keys: dict[int, list[str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps(store_metadata().to_dict()))
    rows: list[dict] = []
    for shard_id, keys in shard_keys.items():
        rows.extend(write_shard(root, shard_id, keys))
    pd.DataFrame(rows).to_parquet(root / "index.parquet", index=False)


# Pipeline and datamodule ------------------------------------------------------------------


def pipeline_args(tmp_path: Path) -> argparse.Namespace:
    pipeline = load_script("run_baseline_mlp_pipeline")
    return argparse.Namespace(
        output_root=tmp_path / "out",
        pipeline_id="test_pipeline",
        data_root=tmp_path / "data",
        embeddings_root=tmp_path / "embeddings",
        endpoints=list(pipeline.ENDPOINTS),
        splits=list(pipeline.SPLITS),
        seeds=[666, 777, 888],
        gpus=["0", "1"],
        max_slots_per_gpu=1,
        hpo_workers_per_endpoint=1,
        hpo_sampler_seed_base=42,
        hpo_sampler_seed_strategy="worker_offset",
        n_trials=50,
        max_epochs=200,
        final_max_epochs=None,
        final_params_source="hpo_csv",
        final_best_params_path=None,
        final_reuse_existing_successes=True,
        hpo_checkpoints=False,
        final_checkpoints=True,
        retry_failed=0,
        job_timeout_sec=0,
        skip_embeddings=False,
        skip_materialization=False,
        refresh_materialized_features=False,
        materialized_feature_dtype="float16",
        materialized_cache_root=tmp_path / "out" / "_cache" / "materialized_features",
        materialized_block_rows=65536,
        materialized_validation="metadata",
        refresh_embedding_superset=False,
        stream_all_jobs=False,
        modalities=list(pipeline.MODALITIES),
        wandb_project="EMULaToR",
        wandb_entity="bisect-group",
        wandb_offline=True,
        skip_wandb_check=True,
        dry_run=True,
        preflight_only=False,
        skip_preflight=False,
        fail_fast=False,
        progress_backend="plain",
        hpo_trial_log=False,
        no_progress=True,
        hpo_split="enzyme_structure_splits",
        storage_profile=pipeline.default_storage_profile(),
        cleanup=True,
        cleanup_dry_run=False,
        cleanup_existing_run=False,
        skip_hpo=False,
        skip_final=False,
        run_label="mlp",
        study_prefix="baseline_mlp",
        run_tags=["baseline_regression", "mlp"],
        hydra_profile={},
        run_config=None,
        print_resolved_run_config=False,
    )


def dispatch_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=tmp_path,
        pipeline_id="dispatch",
        gpus=["0"],
        materialized_cache_root=tmp_path / "cache",
        print_resolved_run_config=False,
        dry_run=True,
        cleanup_existing_run=False,
        skip_preflight=False,
        preflight_only=True,
    )


def superset_args(tmp_path: Path) -> argparse.Namespace:
    args = pipeline_args(tmp_path)
    split_dir = args.data_root / "baselines_ki" / "random_splits_grouped_smiles"
    split_dir.mkdir(parents=True)
    for name in ("train", "val", "test"):
        pd.DataFrame({"smiles": ["C"]}).to_parquet(split_dir / f"{name}.parquet", index=False)
    args.endpoints = ["ki"]
    args.splits = ["random_splits_grouped_smiles"]
    args.modalities = ["ecfp"]
    return args


def regression_datamodule(
    tmp_path: Path,
    split: str = "random_splits_grouped_smiles",
    labels=(1.0, 2.0, 3.0),
    **overrides,
) -> BaselineRegressionDataModule:
    data_root = tmp_path / "data"
    split_dir = data_root / "baselines_ki" / split
    write_tvt(split_dir, labels=labels)
    smited = write_cache(tmp_path / "embeddings" / "smited", "smited", {"C": [1.0, 2.0]})
    esmc = write_cache(tmp_path / "embeddings" / "esmc", "esmc", {"SEQ": [3.0]})
    kwargs = {
        "data_root": str(data_root),
        "endpoint": "ki",
        "split": split,
        "modalities": ["smited", "esmc"],
        "cache_roots": {"smited": str(smited), "esmc": str(esmc)},
        "batch_size": 2,
        "parquet_batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
    }
    kwargs.update(overrides)
    return BaselineRegressionDataModule(**kwargs)


def logged_metrics(module, stage: str) -> dict[str, float]:
    """Values passed to a mocked `module.log` for metrics starting with `stage`."""
    return {
        call.args[0]: call.args[1]
        for call in module.log.call_args_list
        if call.args[0].startswith(stage)
    }


def embedder_config(tmp_path: Path, items: list, **embeddings) -> OmegaConf:
    """Config for a BaseShardEmbedder with `items` pickled as its input file."""
    import pickle

    from omegaconf import OmegaConf

    input_path = tmp_path / "input.pkl"
    input_path.write_bytes(pickle.dumps(items))
    return OmegaConf.create(
        {
            "embeddings": {
                "log_dir": str(tmp_path / "logs"),
                "embeddings_path": str(tmp_path / "emb"),
                "input_pkl_path": str(input_path),
                **embeddings,
            }
        }
    )


def run_pandarallel_serially(monkeypatch) -> None:
    """Make `pandarallel.initialize` a no-op and `Series.parallel_apply` a plain `apply`."""
    monkeypatch.setattr("pandarallel.pandarallel.initialize", lambda **_kwargs: None)
    monkeypatch.setattr(pd.Series, "parallel_apply", pd.Series.apply, raising=False)
