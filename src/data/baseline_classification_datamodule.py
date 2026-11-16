from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pyarrow.parquet as pq

from src.data.components.baseline_base_datamodule import (
    BaseMaterializedFeatureBatchIterableDataset,
    BaseParquetFeatureBatchIterableDataset,
    BaselineBaseDataModule,
    _to_plain_dict,
)


VALID_ENDPOINTS = {"classification", "retrieval"}

DEFAULT_KEY_COLUMNS = {
    "drfp": "rxn_key",
    "ecfp": "smiles",
    "esm3": "pdb_ids",
    "esmc": "sequence",
    "molr": "smiles",
    "smited": "smiles",
}


def _clean_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text if text and text.lower() != "nan" else ""


@dataclass(frozen=True)
class LabelEncoding:
    classes: list[str]
    train_classes: list[str]
    class_to_index: dict[str, int]

    @classmethod
    def from_labels(cls, train_labels: Iterable[Any], all_labels: Iterable[Any]) -> "LabelEncoding":
        train_classes = sorted({_clean_label(value) for value in train_labels if _clean_label(value)})
        seen = set(train_classes)
        extra_classes = sorted({_clean_label(value) for value in all_labels if _clean_label(value)} - seen)
        classes = [*train_classes, *extra_classes]
        return cls(
            classes=classes,
            train_classes=train_classes,
            class_to_index={label: idx for idx, label in enumerate(classes)},
        )

    def encode(self, value: Any) -> int:
        return self.class_to_index.get(_clean_label(value), -1)


class ParquetClassificationFeatureBatchIterableDataset(BaseParquetFeatureBatchIterableDataset):
    """Stream parquet rows into dense feature batches and encoded EC labels."""

    LABEL_DTYPE = np.int64

    def __init__(self, parquet_path: Path, *, label_encoding: LabelEncoding, **kwargs) -> None:
        super().__init__(parquet_path, **kwargs)
        self.label_encoding = label_encoding

    def _encode_label(self, value, row_idx: int) -> int:
        return self.label_encoding.encode(value)


class MaterializedClassificationFeatureBatchIterableDataset(BaseMaterializedFeatureBatchIterableDataset):
    """Read cached dense features and int64 encoded class labels."""

    LABEL_DTYPE = np.int64


class BaselineClassificationDataModule(BaselineBaseDataModule):
    """Classification datamodule over cached embeddings for EC-number prediction."""

    LABEL_DTYPE = np.int64
    MATERIALIZED_DATASET = MaterializedClassificationFeatureBatchIterableDataset

    def __init__(
        self,
        endpoint: str,
        data_root: Optional[str] = None,
        data_roots: Optional[dict[str, str]] = None,
        split: str = "random_splits_grouped_sequence",
        threshold: Optional[str] = None,
        label_col: str = "ec_number",
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
        for key in [k for k in list(kwargs) if k.startswith("_")]:
            kwargs.pop(key, None)
        endpoint = str(endpoint)
        if endpoint not in VALID_ENDPOINTS:
            raise ValueError(f"Unsupported endpoint '{endpoint}'. Valid: {sorted(VALID_ENDPOINTS)}")
        super().__init__(
            data_root=self._resolve_data_root(endpoint, data_root, _to_plain_dict(data_roots)),
            endpoint=endpoint,
            split=str(split),
            threshold=threshold,
            label_col=label_col,
            modalities=list(modalities or (["drfp"] if endpoint == "retrieval" else ["esmc", "esm3"])),
            cache_roots=_to_plain_dict(cache_roots),
            key_columns={**DEFAULT_KEY_COLUMNS, **_to_plain_dict(key_columns)},
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
        self.label_encoding: LabelEncoding | None = None

    def _is_setup(self) -> bool:
        return bool(self.modality_specs) and self.label_encoding is not None

    def _finish_setup(self) -> None:
        self.label_encoding = self._build_label_encoding()

    def get_input_dim(self) -> int:
        info = self._existing_materialized_info_if_available()
        if info is not None:
            return int(info["input_dim"])
        return super().get_input_dim()

    def get_num_classes(self) -> int:
        info = self._existing_materialized_info_if_available()
        if info is not None:
            return int(info["num_classes"])
        self.setup()
        assert self.label_encoding is not None
        return len(self.label_encoding.train_classes)

    def get_class_labels(self) -> list[str]:
        info = self._existing_materialized_info_if_available()
        if info is not None:
            return list(info["classes"])
        self.setup()
        assert self.label_encoding is not None
        return list(self.label_encoding.classes)

    def _make_parquet_dataset(self, split: str, *, shuffle_rows: bool, feature_dtype: str):
        assert self.label_encoding is not None
        return ParquetClassificationFeatureBatchIterableDataset(
            self._split_path(split),
            label_col=self.label_col,
            label_encoding=self.label_encoding,
            modality_specs=self.modality_specs,
            parquet_batch_size=self.parquet_batch_size,
            feature_dtype=feature_dtype,
            shuffle_rows=shuffle_rows,
            seed=self.seed,
            pooled_list_cache_size=self.pooled_list_cache_size,
        )

    def _write_extra_cache_files(self, tmp_dir: Path, metadata: dict[str, Any]) -> None:
        (tmp_dir / "classes.json").write_text(json.dumps(metadata["classes"], indent=2) + "\n")

    def _metadata_extras(self) -> dict[str, Any]:
        assert self.label_encoding is not None
        return {
            "task": "ec_classification",
            "data_root": str(self.data_root.resolve()),
            "classes": self.label_encoding.classes,
            "train_classes": self.label_encoding.train_classes,
            "num_classes": len(self.label_encoding.train_classes),
            "num_metric_classes": len(self.label_encoding.classes),
            "unknown_label_index": -1,
        }

    def _materialized_metadata_from_caches(self, *, load_readers: bool = False) -> dict[str, Any]:
        self.setup()
        return super()._materialized_metadata_from_caches(load_readers=load_readers)

    @staticmethod
    def _resolve_data_root(endpoint: str, data_root: Optional[str], data_roots: dict[str, str]) -> Path:
        if endpoint in data_roots:
            return Path(data_roots[endpoint])
        if data_root is None:
            raise ValueError("Either data_root or data_roots[endpoint] must be configured.")
        return Path(data_root)

    def _resolve_split_dir(self) -> Path:
        split_root = self.data_root / self.split
        if self._has_tvt(split_root):
            return split_root
        if self.threshold is not None:
            threshold_name = str(self.threshold)
            candidates = [split_root / threshold_name]
            if not threshold_name.startswith("threshold_") and not threshold_name.startswith("L"):
                candidates.append(split_root / f"threshold_{threshold_name}")
            for candidate in candidates:
                if self._has_tvt(candidate):
                    return candidate
            raise FileNotFoundError(f"Missing train/val/test under {split_root} for threshold {self.threshold}")

        if not split_root.exists():
            raise FileNotFoundError(f"No split directory found: {split_root}")
        candidates = [path for path in sorted(split_root.iterdir()) if path.is_dir() and self._has_tvt(path)]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple split folders found under {split_root}; set data.threshold explicitly."
            )
        raise FileNotFoundError(f"No train/val/test parquets found for split: {split_root}")

    def _build_label_encoding(self) -> LabelEncoding:
        train_path = self._split_path("train")
        train_labels = pq.read_table(train_path, columns=[self.label_col]).column(self.label_col).to_pylist()
        all_labels = []
        for split_name in ("train", "val", "test"):
            all_labels.extend(
                pq.read_table(self._split_path(split_name), columns=[self.label_col])
                .column(self.label_col)
                .to_pylist()
            )
        encoding = LabelEncoding.from_labels(train_labels, all_labels)
        if not encoding.train_classes:
            raise ValueError(f"No valid labels found in {train_path} column '{self.label_col}'")
        return encoding
