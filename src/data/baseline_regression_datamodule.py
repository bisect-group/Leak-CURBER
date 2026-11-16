from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.data.components.baseline_base_datamodule import (
    BaseMaterializedFeatureBatchIterableDataset,
    BaseParquetFeatureBatchIterableDataset,
    BaselineBaseDataModule,
    _to_plain_dict,
)


# Regression endpoints and split families supported by the precomputed baseline
# parquet layout. These checks fail early when a config points to an unexpected
# dataset tree.
VALID_ENDPOINTS = {"ec50", "ic50", "kcat", "kd", "ki", "km"}
VALID_SPLITS = {
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "uniprot_time_splits",
    "substrate_splits",
    "enzyme_sequence_splits",
    "enzyme_structure_splits",
    "conformer_cosine_splits",
}

DEFAULT_KEY_COLUMNS = {
    "smited": "smiles",
    "ecfp": "smiles",
    "molr": "smiles",
    "esmc": "sequence",
    "esm3": "pdb_ids",
}


class ParquetFeatureBatchIterableDataset(BaseParquetFeatureBatchIterableDataset):
    """Parquet feature batches with a float32 regression target as ``y``."""

    LABEL_DTYPE = np.float32

    def _encode_label(self, value, row_idx: int) -> float:
        label = float(value)
        if not np.isfinite(label):
            raise ValueError(
                f"Non-finite target in {self.parquet_path} column '{self.label_col}' "
                f"at parquet batch row {row_idx}."
            )
        return label


class MaterializedFeatureBatchIterableDataset(BaseMaterializedFeatureBatchIterableDataset):
    """Memmapped cached features with float32 regression targets."""

    LABEL_DTYPE = np.float32


class BaselineRegressionDataModule(BaselineBaseDataModule):
    """Lightning data module for regression baselines over cached embeddings.

    Expected split layout:

    ``{data_root}/baselines_{endpoint}/{split}/{train,val,test}.parquet``

    Thresholded splits may instead live under:

    ``{data_root}/baselines_{endpoint}/{split}/threshold_{threshold}/...``

    Each split parquet must include the target column, ``log10_value`` by
    default, plus the key columns required by the selected modalities. Embedding
    cache roots must point to directories readable by ``ShardedEmbeddingReader``.

    The dataloaders stream parquet batches and concatenate all selected
    modalities in the order provided by ``modalities``.
    """

    LABEL_DTYPE = np.float32
    MATERIALIZED_DATASET = MaterializedFeatureBatchIterableDataset

    def __init__(
        self,
        data_root: str,
        endpoint: str,
        split: str,
        threshold: Optional[str] = None,
        label_col: str = "log10_value",
        modalities: Optional[list[str]] = None,
        cache_roots: Optional[dict[str, str]] = None,
        key_columns: Optional[dict[str, str]] = None,
        batch_size: int = 1024,
        parquet_batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        feature_dtype: str = "float32",
        shard_cache_size: int = 4,
        materialization_dir: Optional[str] = None,
        materialized_cache_root: Optional[str] = None,
        materialization_max_gb: float = 64.0,
        use_materialized_features: bool = False,
        materialize_if_missing: bool = True,
        refresh_materialized_features: bool = False,
        materialized_validation: str = "metadata",
        materialized_feature_dtype: str = "float16",
        materialized_block_rows: int = 65536,
        materialized_shuffle: str = "block",
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        pooled_list_cache_size: int = 0,
        seed: int = 42,
        **kwargs,
    ) -> None:
        """Create the data module.

        Args:
            data_root: Root containing ``baselines_{endpoint}`` directories.
            endpoint: Regression endpoint, for example ``"ki"`` or ``"kcat"``.
            split: Split family name, for example ``"random_splits_grouped_smiles"``.
            threshold: Optional threshold folder suffix. Values may be provided
                with or without the ``threshold_`` prefix.
            label_col: Numeric regression target column in each split parquet.
            modalities: Ordered list of embedding modalities to concatenate.
            cache_roots: Mapping from modality name to embedding cache directory.
            key_columns: Optional overrides for modality-to-parquet-key mapping.
            batch_size: Number of rows per yielded training batch when
                ``parquet_batch_size`` is not set.
            parquet_batch_size: Number of parquet rows to read at a time.
            num_workers: Number of PyTorch DataLoader workers.
            pin_memory: Forwarded to DataLoader.
            feature_dtype: NumPy dtype for returned feature tensors.
            shard_cache_size: Number of embedding shards cached per reader.
            materialization_dir: Default output directory for ``materialize_split``.
            materialized_cache_root: Shared cache root for validated materialized
                Torch feature matrices. Overrides ``materialization_dir``.
            materialization_max_gb: Safety cap for materialized feature arrays.
            use_materialized_features: Whether Torch dataloaders should read
                from validated memmapped feature matrices.
            materialize_if_missing: Build a missing/stale materialized cache on
                demand when materialized dataloaders are requested.
            refresh_materialized_features: Rebuild the materialized cache even
                when an existing complete cache validates.
            materialized_validation: Cache-hit validation policy. ``"metadata"``
                validates expected metadata and ``.npy`` headers without loading
                embedding indexes, ``"full"`` preserves reader-backed
                validation, and ``"trust_complete"`` validates only fingerprint
                metadata plus the completion marker.
            materialized_feature_dtype: Stored dtype for materialized X arrays.
            materialized_block_rows: Contiguous row block size used by the
                block-shuffled memmap training loader.
            materialized_shuffle: ``"block"`` for train block shuffling or
                ``"none"`` for sequential materialized training.
            persistent_workers: Forwarded to DataLoader when ``num_workers > 0``.
            prefetch_factor: Forwarded to DataLoader when ``num_workers > 0``.
            pooled_list_cache_size: Streaming-only bounded cache for list-valued
                modality pooling, keyed by the tuple of row keys.
            seed: Base seed for per-worker row shuffling.
            **kwargs: Extra Hydra/Lightning keys. Private keys beginning with
                ``_`` are ignored before hyperparameters are saved.
        """

        for key in [k for k in list(kwargs) if k.startswith("_")]:
            kwargs.pop(key, None)
        modalities = list(modalities or ["smited", "ecfp", "molr", "esmc", "esm3"])
        # Rebinding these locals keeps the hyperparameters saved below identical to before.
        cache_roots = _to_plain_dict(cache_roots)
        key_columns = _to_plain_dict(key_columns)
        super().__init__(
            data_root=Path(data_root),
            endpoint=endpoint,
            split=split,
            threshold=threshold,
            label_col=label_col,
            modalities=modalities,
            cache_roots=cache_roots,
            key_columns={**DEFAULT_KEY_COLUMNS, **key_columns},
            batch_size=batch_size,
            parquet_batch_size=parquet_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            feature_dtype=feature_dtype,
            shard_cache_size=shard_cache_size,
            materialization_dir=materialization_dir,
            materialized_cache_root=materialized_cache_root,
            materialization_max_gb=materialization_max_gb,
            use_materialized_features=use_materialized_features,
            materialize_if_missing=materialize_if_missing,
            refresh_materialized_features=refresh_materialized_features,
            materialized_validation=materialized_validation,
            materialized_feature_dtype=materialized_feature_dtype,
            materialized_block_rows=materialized_block_rows,
            materialized_shuffle=materialized_shuffle,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pooled_list_cache_size=pooled_list_cache_size,
            seed=seed,
        )
        self.save_hyperparameters(logger=False)

    def _make_parquet_dataset(self, split: str, *, shuffle_rows: bool, feature_dtype: str):
        return ParquetFeatureBatchIterableDataset(
            self._split_path(split),
            label_col=self.label_col,
            modality_specs=self.modality_specs,
            parquet_batch_size=self.parquet_batch_size,
            feature_dtype=feature_dtype,
            shuffle_rows=shuffle_rows,
            seed=self.seed,
            pooled_list_cache_size=self.pooled_list_cache_size,
        )

    def _resolve_split_dir(self) -> Path:
        """Find the directory that contains train/val/test parquet files."""

        if self.endpoint not in VALID_ENDPOINTS:
            raise ValueError(f"Unsupported endpoint '{self.endpoint}'. Valid: {sorted(VALID_ENDPOINTS)}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unsupported split '{self.split}'. Valid: {sorted(VALID_SPLITS)}")

        split_root = self.data_root / f"baselines_{self.endpoint}" / self.split
        if self._has_tvt(split_root):
            return split_root

        # Thresholded split folders are named threshold_<value>. Accept configs
        # with either the raw value or the full directory prefix.
        if self.threshold is not None:
            threshold_name = str(self.threshold)
            if not threshold_name.startswith("threshold_"):
                threshold_name = f"threshold_{threshold_name}"
            threshold_dir = split_root / threshold_name
            if not self._has_tvt(threshold_dir):
                raise FileNotFoundError(f"Threshold split does not contain train/val/test parquets: {threshold_dir}")
            return threshold_dir

        # If there is exactly one valid threshold folder, use it automatically.
        # More than one would make the experiment ambiguous.
        threshold_dirs = [
            path
            for path in sorted(split_root.glob("threshold_*"))
            if path.is_dir() and self._has_tvt(path)
        ]
        if len(threshold_dirs) == 1:
            return threshold_dirs[0]
        if len(threshold_dirs) > 1:
            raise ValueError(
                f"Multiple threshold folders found under {split_root}; set data.threshold explicitly."
            )
        raise FileNotFoundError(f"No train/val/test parquets found for split: {split_root}")
