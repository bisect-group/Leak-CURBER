from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, open_dict

from src.data.components.embedders.store import ShardedEmbeddingStore
from src.data.components.embedders.types import CacheMetadata
from src.data.components.unimol_embed import UnimolSDFEmbedder
from src.data.components.unimol_featurize import UnimolSDFFeaturizer
from src.utils.tqdmlogger import TqdmLogger


class UnimolSDFShardEmbedder:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        log_dir = Path(cfg.unimol_sdf_embeddings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = TqdmLogger(
            log_dir=log_dir,
            log_file_name=cfg.unimol_sdf_embeddings.unimol_sdf_log_file_name,
        ).get_logger()

        self.sdf_dir = Path(cfg.unimol_sdf_embeddings.unimol_sdf_sdf_dir)
        self.store_dir = Path(cfg.unimol_sdf_embeddings.unimol_sdf_embeddings_store_dir)
        self.raw_data_dir = Path(cfg.paths.raw_data_dir)
        self.num_conformers = int(cfg.embeddings.datamol_num_conformers)
        self.stems_per_batch = int(
            cfg.unimol_sdf_embeddings.get("unimol_sdf_stems_per_batch", 1_000_000)
        )

    def _sharded_sdf_path(self, group: str, smi_hash: str, stem: str) -> Path:
        return self.sdf_dir / group / smi_hash[:4] / f"{stem}.sdf"

    def _pubchem_sdf_path(self, smi_hash: str) -> Path:
        return self._sharded_sdf_path("pubchem", smi_hash, f"pubchem_{smi_hash}")

    def _datamol_sdf_path(self, smi_hash: str, conf_id: int) -> Path:
        return self._sharded_sdf_path(
            "datamol", smi_hash, f"datamol_{smi_hash}_conf{conf_id}"
        )

    def resolve_input_path(self, input_override: str | None = None) -> Path:
        configured_input = self.cfg.unimol_sdf_embeddings.get(
            "unimol_sdf_input_dataset_parquet_file_path",
            self.cfg.embeddings.get(
                "datamol_conformer_input_dataset_parquet_file_path",
                self.cfg.splits.dataset.cosine_split.input_dataset_parquet_file_path,
            ),
        )
        input_path = Path(input_override or configured_input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input parquet not found at {input_path}")
        return input_path

    def _load_dataset_smiles_df(self, input_path: Path) -> pd.DataFrame:
        df = pd.read_parquet(input_path)

        required_cols = {"smiles", "smiles_hash"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            missing = ", ".join(sorted(missing_cols))
            raise ValueError(
                f"Input dataset {input_path} is missing required columns: {missing}"
            )

        return (
            df[["smiles_hash", "smiles"]]
            .dropna(subset=["smiles_hash", "smiles"])
            .drop_duplicates(subset=["smiles_hash"], keep="first")
            .sort_values("smiles_hash")
            .reset_index(drop=True)
        )

    def _build_relevant_sdf_stems(self, smiles_df: pd.DataFrame) -> dict[str, Path]:
        stems: dict[str, Path] = {}
        for smi_hash in smiles_df["smiles_hash"].astype(str):
            pubchem_path = self._pubchem_sdf_path(smi_hash)
            if pubchem_path.exists():
                stems[pubchem_path.stem] = pubchem_path

            for conf_id in range(self.num_conformers):
                datamol_path = self._datamol_sdf_path(smi_hash, conf_id)
                if datamol_path.exists():
                    stems[datamol_path.stem] = datamol_path

        return stems

    def _build_store(self) -> ShardedEmbeddingStore:
        meta_path = self.store_dir / "meta.json"

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = CacheMetadata(**json.load(handle))
        else:
            metadata = CacheMetadata(
                embedder="unimol_sdf",
                model_name="unimolv2",
                version="v1",
                storage_dtype="float16",
                embedding_dim=None,
                max_shard_bytes=512 * 1024 * 1024,
                key_type="str",
                key_field="smiles",
            )

        return ShardedEmbeddingStore(self.store_dir, metadata, self.logger)

    def _prepare_store_arrays(
        self, arrays_by_key: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        prepared: dict[str, np.ndarray] = {}
        for key, value in arrays_by_key.items():
            arr = np.asarray(value, dtype=np.float16)
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            if arr.ndim != 1:
                raise ValueError(f"Unexpected embedding shape for key '{key}': {arr.shape}")
            prepared[key] = arr
        return prepared

    def _iter_batches(self, items: list[str], batch_size: int):
        for start in range(0, len(items), batch_size):
            yield start // batch_size, items[start : start + batch_size]

    def _process_batch(
        self,
        batch_index: int,
        batch_stems: list[str],
        stem_to_path: dict[str, Path],
        store: ShardedEmbeddingStore,
        input_path: Path,
        keep_workdir: bool,
    ) -> int:
        temp_root = self.raw_data_dir / "unimol_incremental"
        temp_root.mkdir(parents=True, exist_ok=True)
        workdir = Path(
            tempfile.mkdtemp(
                prefix=f"unimol_{Path(input_path).stem}_batch{batch_index:04d}_",
                dir=str(temp_root),
            )
        )
        raw_dir = workdir / "raw"

        self.logger.info(
            f"Batch {batch_index + 1}: processing {len(batch_stems):,} SDF stems"
        )
        self.logger.info(f"Temporary workdir: {workdir}")

        with open_dict(self.cfg.unimol_sdf_embeddings):
            self.cfg.unimol_sdf_embeddings.unimol_sdf_raw_dir = str(raw_dir)
            self.cfg.unimol_sdf_embeddings.unimol_sdf_input_sdf_paths = [
                str(stem_to_path[stem]) for stem in batch_stems
            ]

        succeeded = False
        try:
            featurizer = UnimolSDFFeaturizer(self.cfg)
            featurizer.featurize()

            embedder = UnimolSDFEmbedder(self.cfg)
            arrays_by_key = self._prepare_store_arrays(embedder.embed(save_npz=False))
            batch_set = set(batch_stems)
            missing_outputs = sorted(batch_set - set(arrays_by_key.keys()))
            if missing_outputs:
                self.logger.warning(
                    f"UniMol output missing {len(missing_outputs):,} requested stems in batch "
                    f"{batch_index + 1}. Examples: {missing_outputs[:10]}"
                )

            written = store.put_many(arrays_by_key, flush=True)
            self.logger.info(
                f"Persisted {written:,} incremental UniMol embeddings from batch "
                f"{batch_index + 1} into {store.root_dir}"
            )
            succeeded = True
            return written
        finally:
            if keep_workdir:
                self.logger.info(f"Keeping temporary workdir at {workdir}")
            elif not succeeded:
                self.logger.warning(
                    f"UniMol incremental batch failed; keeping temporary workdir at {workdir}"
                )
            else:
                shutil.rmtree(workdir, ignore_errors=True)
                self.logger.info(f"Removed temporary workdir {workdir}")

    def embed(self, *, input_override: str | None = None, keep_workdir: bool = False) -> None:
        input_path = self.resolve_input_path(input_override)
        smiles_df = self._load_dataset_smiles_df(input_path)
        stem_to_path = self._build_relevant_sdf_stems(smiles_df)

        store = self._build_store()
        relevant_stems = sorted(stem_to_path.keys())
        missing_stems = store.get_missing_keys(relevant_stems)

        self.logger.info(f"Dataset input: {input_path}")
        self.logger.info(f"SDF dir: {self.sdf_dir}")
        self.logger.info(f"Store dir: {store.root_dir}")
        self.logger.info(f"Unique smiles_hash keys: {len(smiles_df):,}")
        self.logger.info(f"Relevant SDF stems for dataset: {len(relevant_stems):,}")
        self.logger.info(f"Missing stems to compute incrementally: {len(missing_stems):,}")
        self.logger.info(f"Configured SDF stems per batch: {self.stems_per_batch:,}")

        if not missing_stems:
            self.logger.info(
                "Store already contains all dataset-relevant UniMol embeddings. Nothing to do."
            )
            return

        self.logger.info(
            f"Processing {len(missing_stems):,} missing SDF stems directly from {self.sdf_dir}"
        )
        total_written = 0
        total_batches = (len(missing_stems) + self.stems_per_batch - 1) // self.stems_per_batch
        for batch_index, batch_stems in self._iter_batches(
            missing_stems, self.stems_per_batch
        ):
            self.logger.info(
                f"Starting batch {batch_index + 1}/{total_batches} with "
                f"{len(batch_stems):,} stems"
            )
            total_written += self._process_batch(
                batch_index=batch_index,
                batch_stems=batch_stems,
                stem_to_path=stem_to_path,
                store=store,
                input_path=input_path,
                keep_workdir=keep_workdir,
            )

        self.logger.info(
            f"Persisted {total_written:,} total incremental UniMol embeddings into {store.root_dir}"
        )
