import json
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import DictConfig
from src.data.components.splitters.base import BaseThresholdedSimilaritySplitter
from src.data.components.embedders.utils import sha256_short
from src.utils.tqdmlogger import TqdmLogger


class EmbeddingCosineSimilaritySplitter(BaseThresholdedSimilaritySplitter):
    """Embedding-agnostic cosine similarity splitter.

    This splitter can operate on different embedding sources:
    - A dataframe embedding column (for pre-attached embeddings)
    - A sharded embedding store (meta.json + index.parquet + shard_*.npy)
    - An NPZ archive mapping embedding keys to vectors

    Supported lookup modes:
    - direct: dataset key directly matches embedding key
    - conformer: uses direct, pubchem_{key}, datamol_{key}_conf{i}
    - templates: uses splits.cosine_embedding_key_templates
    """

    def __init__(self, cfg: DictConfig):
        scfg = cfg.splits
        self.cfg = cfg
        ccfg = scfg.dataset.cosine_split

        self.KEY_COLUMN = ccfg.get("key_column", "smiles_hash")
        self.EMBEDDING_LOOKUP_MODE = str(ccfg.get("embedding_lookup_mode", "conformer")).lower()

        self.TEST_FRAC = float(ccfg.get("test_frac", scfg.test_frac or 0.1))
        self.MATCH_VAL_TO_TEST = bool(ccfg.get("match_val_to_test", True))
        self.VAL_FRAC = float(ccfg.get("val_frac", self.TEST_FRAC))

        if not (0.0 < self.TEST_FRAC < 0.5):
            raise ValueError("splits.dataset.cosine_split.test_frac must be in (0, 0.5).")
        if not self.MATCH_VAL_TO_TEST and not (0.0 < self.VAL_FRAC < 0.5):
            raise ValueError(
                "splits.dataset.cosine_split.val_frac must be in (0, 0.5) when match_val_to_test=false."
            )

        self.THRESHOLDS = list(
            ccfg.get("similarity_thresholds", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        )
        self.COSINE_BATCH_SIZE = int(ccfg.get("batch_size", 512))
        self.CACHE_OVERWRITE = bool(scfg.get("similarity_cache_overwrite", False))
        self.EMBEDDING_COLUMN = ccfg.get("embedding_col", None)

        self.DATAMOL_NUM_CONFORMERS = int(
            ccfg.get(
                "datamol_num_conformers",
                cfg.get("embeddings", {}).get("datamol_num_conformers", 10),
            )
        )
        self.EMBEDDING_KEY_TEMPLATES = list(ccfg.get("embedding_key_templates", []))

        npz_cfg = ccfg.get("embeddings_npz", None)
        self.EMBEDDINGS_NPZ = Path(npz_cfg) if npz_cfg else None

        store_cfg = ccfg.get("embeddings_store_dir", None)
        self.EMBEDDINGS_STORE_DIR = Path(store_cfg) if store_cfg else None

        self.INPUT_DATASET_PATH = Path(ccfg.input_dataset_parquet_file_path)
        self.OUTPUT_DIR = Path(ccfg.output_dir)
        self.UNIQUE_COSINE_SIMILARITY_PLOT_PATH = Path(ccfg.unique_similarity_plot_path)
        self.FULL_DATASET_COSINE_SIMILARITY_PLOT_PATH = Path(
            ccfg.full_dataset_similarity_plot_path
        )

        log_path = Path(scfg.log_dir)
        for path in [
            log_path,
            self.OUTPUT_DIR,
            self.UNIQUE_COSINE_SIMILARITY_PLOT_PATH.parent,
            self.FULL_DATASET_COSINE_SIMILARITY_PLOT_PATH.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=log_path,
            log_file_name=scfg.log_file_name,
        ).get_logger()

        if not self.INPUT_DATASET_PATH.exists():
            msg = f"Dataset parquet not found at {self.INPUT_DATASET_PATH}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        self.logger.info("Loading full dataset...")
        self.full_dataset_df = pd.read_parquet(self.INPUT_DATASET_PATH)
        self.logger.info(f"Full dataset shape: {self.full_dataset_df.shape}")

        if self.KEY_COLUMN not in self.full_dataset_df.columns:
            raise ValueError(
                f"Dataset must contain key column '{self.KEY_COLUMN}' for cosine splitting."
            )

        self.full_dataset_df[self.KEY_COLUMN] = self.full_dataset_df[self.KEY_COLUMN].astype("string")
        self.unique_keys = (
            self.full_dataset_df[self.KEY_COLUMN].dropna().astype(str).unique().tolist()
        )

        self._embedding_source, self._embedding_source_path = self._resolve_embedding_source()

        self.logger.info(f"Unique keys ({self.KEY_COLUMN}): {len(self.unique_keys):,}")
        self.logger.info(
            f"embedding_source={self._embedding_source} ({self._embedding_source_path}), "
            f"lookup_mode={self.EMBEDDING_LOOKUP_MODE}, "
            f"cosine_batch_size={self.COSINE_BATCH_SIZE}, "
            f"thresholds={self.THRESHOLDS}, "
            f"test_frac={self.TEST_FRAC}, "
            f"match_val_to_test={self.MATCH_VAL_TO_TEST}"
        )

        self._max_cosine_array: np.ndarray | None = None
        self._max_cosine_to_dataset: dict[str, float] | None = None
        self._valid_embedding_mask: np.ndarray | None = None
        self.MAX_SIM_CACHE_PATH = self.OUTPUT_DIR / "max_cosine_similarities.tsv"

    def _resolve_embedding_source(self) -> tuple[str, Path]:
        if (
            self.EMBEDDING_COLUMN
            and self.EMBEDDING_COLUMN in self.full_dataset_df.columns
        ):
            return "column", self.INPUT_DATASET_PATH

        if self.EMBEDDINGS_STORE_DIR is not None:
            meta_path = self.EMBEDDINGS_STORE_DIR / "meta.json"
            index_path = self.EMBEDDINGS_STORE_DIR / "index.parquet"
            if (
                self.EMBEDDINGS_STORE_DIR.exists()
                and meta_path.exists()
                and index_path.exists()
            ):
                return "store", self.EMBEDDINGS_STORE_DIR

        npz_candidates = []
        if self.EMBEDDINGS_NPZ is not None:
            npz_candidates.append(self.EMBEDDINGS_NPZ)
        npz_candidates.append(self.INPUT_DATASET_PATH.parent / "unimol_sdf_embeddings.npz")
        npz_candidates.append(self.INPUT_DATASET_PATH.parent.parent / "unimol_sdf_embeddings.npz")

        for candidate in npz_candidates:
            if candidate is not None and candidate.exists():
                return "npz", candidate

        checked = [str(p) for p in npz_candidates if p is not None]
        if self.EMBEDDINGS_STORE_DIR is not None:
            checked.append(str(self.EMBEDDINGS_STORE_DIR))

        raise FileNotFoundError(
            "No cosine embedding source found. Set one of: "
            "splits.cosine_embedding_col, splits.dataset.cosine_embeddings_store_dir, "
            "or splits.dataset.cosine_embeddings_npz. Checked: "
            + ", ".join(checked)
        )

    def _candidate_embedding_specs(self, dataset_key: str) -> list[tuple[str, int, bool]]:
        if self.EMBEDDING_LOOKUP_MODE == "direct":
            return [(dataset_key, 3, False)]

        if self.EMBEDDING_LOOKUP_MODE == "conformer":
            specs: list[tuple[str, int, bool]] = [
                (dataset_key, 3, False),
                (f"pubchem_{dataset_key}", 2, False),
            ]
            for i in range(self.DATAMOL_NUM_CONFORMERS):
                specs.append((f"datamol_{dataset_key}_conf{i}", 1, True))
            return specs

        if self.EMBEDDING_LOOKUP_MODE == "templates":
            if not self.EMBEDDING_KEY_TEMPLATES:
                raise ValueError(
                    "splits.cosine_embedding_key_templates must be provided when cosine_embedding_lookup_mode=templates."
                )

            specs = []
            for template in self.EMBEDDING_KEY_TEMPLATES:
                if "{i}" in template:
                    for i in range(self.DATAMOL_NUM_CONFORMERS):
                        specs.append((template.format(key=dataset_key, i=i), 1, True))
                else:
                    specs.append((template.format(key=dataset_key), 2, False))
            return specs

        raise ValueError(
            "Unsupported splits.cosine_embedding_lookup_mode. Use one of: direct, conformer, templates."
        )

    def _vector_from_raw(self, raw_value) -> np.ndarray | None:
        if raw_value is None:
            return None

        value = raw_value
        if isinstance(raw_value, str):
            stripped = raw_value.strip()
            if not stripped:
                return None
            try:
                value = json.loads(stripped)
            except Exception:
                return None

        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            return None
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            return arr.mean(axis=0)
        return None

    def _finalize_embeddings(
        self,
        direct_best: dict[str, tuple[int, np.ndarray]],
        pooled_candidates: dict[str, list[np.ndarray]],
    ) -> dict[str, np.ndarray]:
        merged: dict[str, np.ndarray] = {}

        for key, vectors in pooled_candidates.items():
            if not vectors:
                continue
            pooled = np.stack(vectors).mean(axis=0).astype(np.float32)
            merged[key] = pooled

        for key, (priority, vec) in direct_best.items():
            existing = merged.get(key)
            if existing is None:
                merged[key] = vec
                continue

            if priority >= 2:
                merged[key] = vec

        return merged

    def _log_embedding_coverage_summary(
        self,
        per_key_embs: dict[str, np.ndarray],
        *,
        source_label: str,
    ) -> None:
        key_set = set(self.unique_keys)
        matched_keys = key_set & set(per_key_embs.keys())
        row_count_dict = self.full_dataset_df[self.KEY_COLUMN].value_counts().to_dict()
        matched_rows = sum(row_count_dict.get(key, 0) for key in matched_keys)
        missing_keys = len(key_set) - len(matched_keys)
        missing_rows = len(self.full_dataset_df) - matched_rows

        self.logger.info(
            f"[Cosine] Coverage from {source_label}: "
            f"{len(matched_keys):,}/{len(key_set):,} keys matched, "
            f"{matched_rows:,}/{len(self.full_dataset_df):,} rows covered."
        )
        self.logger.info(
            f"[Cosine] Missing embeddings after {source_label}: "
            f"{missing_keys:,} keys, {missing_rows:,} rows."
        )

    def _log_lookup_match_breakdown(self, lookup: dict[str, np.ndarray]) -> None:
        direct = 0
        pubchem = 0
        datamol = 0
        unmatched = 0

        for dataset_key in self.unique_keys:
            if dataset_key in lookup:
                direct += 1
                continue
            if f"pubchem_{dataset_key}" in lookup:
                pubchem += 1
                continue
            if any(
                f"datamol_{dataset_key}_conf{i}" in lookup
                for i in range(self.DATAMOL_NUM_CONFORMERS)
            ):
                datamol += 1
                continue
            unmatched += 1

        self.logger.info(
            "[Cosine] Lookup match breakdown by preferred source: "
            f"direct={direct:,}, pubchem={pubchem:,}, "
            f"datamol_mean={datamol:,}, unmatched={unmatched:,}."
        )

    def _load_per_key_embeddings_from_column(self) -> dict[str, np.ndarray]:
        if not self.EMBEDDING_COLUMN:
            return {}

        direct_best: dict[str, tuple[int, np.ndarray]] = {}
        pooled_candidates: dict[str, list[np.ndarray]] = {}

        for row in self.full_dataset_df[[self.KEY_COLUMN, self.EMBEDDING_COLUMN]].itertuples(index=False):
            key, raw_vec = row
            if pd.isna(key):
                continue
            vec = self._vector_from_raw(raw_vec)
            if vec is None:
                continue
            key_str = str(key)
            pooled_candidates.setdefault(key_str, []).append(vec)
            direct_best.setdefault(key_str, (3, vec))

        merged = self._finalize_embeddings(direct_best, pooled_candidates)
        self.logger.info(
            f"Loaded {len(merged):,} embeddings from dataframe column '{self.EMBEDDING_COLUMN}'."
        )
        self._log_embedding_coverage_summary(
            merged, source_label=f"dataframe column '{self.EMBEDDING_COLUMN}'"
        )
        return merged

    def _build_candidate_maps(
        self,
    ) -> tuple[dict[str, tuple[str, int, bool]], dict[str, tuple[str, int, bool]]]:
        key_map: dict[str, tuple[str, int, bool]] = {}
        hash_map: dict[str, tuple[str, int, bool]] = {}

        for dataset_key in self.unique_keys:
            for embedding_key, priority, poolable in self._candidate_embedding_specs(dataset_key):
                if embedding_key not in key_map:
                    spec = (dataset_key, priority, poolable)
                    key_map[embedding_key] = spec
                    hash_map[sha256_short(embedding_key)] = spec

        return key_map, hash_map

    def _load_per_key_embeddings_from_npz(self) -> dict[str, np.ndarray]:
        self.logger.info(f"Loading embeddings from {self._embedding_source_path}...")
        data = np.load(str(self._embedding_source_path), allow_pickle=False)
        keys = list(data.files)
        self.logger.info(f"  {len(keys):,} embedding keys in NPZ.")
        self._log_lookup_match_breakdown(data)

        candidate_map, _ = self._build_candidate_maps()

        direct_best: dict[str, tuple[int, np.ndarray]] = {}
        pooled_candidates: dict[str, list[np.ndarray]] = {}

        for embedding_key in tqdm(keys, desc="Reading NPZ keys"):
            spec = candidate_map.get(embedding_key)
            if spec is None:
                continue

            dataset_key, priority, poolable = spec
            vec = self._vector_from_raw(data[embedding_key])
            if vec is None:
                continue

            if poolable:
                pooled_candidates.setdefault(dataset_key, []).append(vec)
            else:
                prev = direct_best.get(dataset_key)
                if prev is None or priority > prev[0]:
                    direct_best[dataset_key] = (priority, vec)

        merged = self._finalize_embeddings(direct_best, pooled_candidates)
        self.logger.info(f"Loaded NPZ embeddings for {len(merged):,} dataset keys.")
        self._log_embedding_coverage_summary(
            merged, source_label=f"NPZ {self._embedding_source_path}"
        )
        return merged

    def _load_per_key_embeddings_from_store(self) -> dict[str, np.ndarray]:
        store_dir = self._embedding_source_path
        meta_path = store_dir / "meta.json"
        index_path = store_dir / "index.parquet"

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.logger.info(
            "[Cosine] Store metadata: "
            f"embedder={meta.get('embedder')}, model_name={meta.get('model_name')}, "
            f"version={meta.get('version')}, key_field={meta.get('key_field')}, "
            f"embedding_dim={meta.get('embedding_dim')}."
        )

        index_df = pd.read_parquet(index_path)
        if index_df.empty:
            self.logger.warning(f"Embedding store index is empty: {index_path}")
            return {}
        self.logger.info(
            f"[Cosine] Store index rows: {len(index_df):,} before dedup, "
            f"{index_df['key_hash'].nunique():,} unique key hashes."
        )

        index_df = index_df.drop_duplicates(subset=["key_hash"], keep="last")
        lookup = {
            row.key_hash: (int(row.shard_id), int(row.row_idx))
            for row in index_df.itertuples(index=False)
        }

        _, candidate_hash_map = self._build_candidate_maps()
        present_hashes = [key_hash for key_hash in candidate_hash_map if key_hash in lookup]
        self.logger.info(
            f"[Cosine] Candidate hash matches in store: {len(present_hashes):,} / "
            f"{len(candidate_hash_map):,} candidate keys."
        )

        if not present_hashes:
            self.logger.warning("No dataset-relevant embeddings found in sharded store.")
            return {}

        by_shard: dict[int, list[tuple[str, int]]] = {}
        for key_hash in present_hashes:
            shard_id, row_idx = lookup[key_hash]
            by_shard.setdefault(shard_id, []).append((key_hash, row_idx))

        vectors_by_hash: dict[str, np.ndarray] = {}
        for shard_id, rows in by_shard.items():
            shard_path = store_dir / f"shard_{shard_id:06d}.npy"
            if not shard_path.exists():
                self.logger.warning(f"[Cosine] Missing shard file referenced by index: {shard_path}")
                continue
            arr = np.load(shard_path, mmap_mode="r")
            for key_hash, row_idx in rows:
                if 0 <= row_idx < len(arr):
                    vectors_by_hash[key_hash] = np.asarray(arr[row_idx], dtype=np.float32)
                else:
                    self.logger.warning(
                        f"[Cosine] Row index {row_idx} out of bounds for shard {shard_path} "
                        f"(len={len(arr)})."
                    )

        direct_best: dict[str, tuple[int, np.ndarray]] = {}
        pooled_candidates: dict[str, list[np.ndarray]] = {}

        for key_hash, vec in vectors_by_hash.items():
            dataset_key, priority, poolable = candidate_hash_map[key_hash]
            if poolable:
                pooled_candidates.setdefault(dataset_key, []).append(vec)
            else:
                prev = direct_best.get(dataset_key)
                if prev is None or priority > prev[0]:
                    direct_best[dataset_key] = (priority, vec)

        merged = self._finalize_embeddings(direct_best, pooled_candidates)
        self.logger.info(
            f"Loaded store embeddings for {len(merged):,} dataset keys from {store_dir}."
        )
        self.logger.info(
            "[Cosine] Store match breakdown after pooling: "
            f"direct/pubchem winners={len(direct_best):,}, "
            f"datamol pooled keys={len(pooled_candidates):,}."
        )
        self._log_embedding_coverage_summary(merged, source_label=f"store {store_dir}")
        return merged

    def _load_per_key_embeddings(self) -> dict[str, np.ndarray]:
        if self._embedding_source == "column":
            return self._load_per_key_embeddings_from_column()
        if self._embedding_source == "store":
            return self._load_per_key_embeddings_from_store()
        return self._load_per_key_embeddings_from_npz()

    def _compute_max_cosine_batched(self, embeddings: np.ndarray) -> np.ndarray:
        n_samples = len(embeddings)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        normed = (embeddings / norms).astype(np.float32)

        max_sims = np.full(n_samples, -1.0, dtype=np.float32)

        for start in tqdm(
            range(0, n_samples, self.COSINE_BATCH_SIZE),
            desc="Batched cosine similarity",
        ):
            end = min(start + self.COSINE_BATCH_SIZE, n_samples)
            batch = normed[start:end]
            sims = batch @ normed.T

            b = end - start
            row_idx = np.arange(b)
            col_idx = np.arange(start, end)
            sims[row_idx, col_idx] = -2.0

            max_sims[start:end] = sims.max(axis=1)

        return max_sims.astype(np.float64)

    def compute_max_cosine_to_dataset(self) -> np.ndarray:
        if self._max_cosine_array is not None:
            return self._max_cosine_array

        if not self.CACHE_OVERWRITE:
            loaded = self._load_similarity_cache(
                cache_path=self.MAX_SIM_CACHE_PATH,
                key_column=self.KEY_COLUMN,
                value_column="max_cosine_similarity",
                ordered_keys=self.unique_keys,
            )
            if loaded is not None:
                values, _ = loaded
                self._max_cosine_array = values
                self._max_cosine_to_dataset = {
                    key: float(values[i]) for i, key in enumerate(self.unique_keys)
                }
                self._valid_embedding_mask = None
                self.logger.info(
                    f"Using cached cosine similarities from {self.MAX_SIM_CACHE_PATH}"
                )
                zero_count = int(np.sum(values == 0.0))
                self.logger.info(
                    f"[Cosine] Cached zero-similarity keys: {zero_count:,}/{len(values):,}."
                )
                self.logger.info(
                    "[Cosine] Cached similarities do not preserve explicit embedding-validity flags; "
                    "threshold diagnostics will treat zero-cosine keys as the suspicious bucket."
                )
                return self._max_cosine_array

        per_key_embs = self._load_per_key_embeddings()

        n_unique = len(self.unique_keys)
        available = [key for key in self.unique_keys if key in per_key_embs]
        self.logger.info(
            f"Embeddings available for {len(available):,}/{n_unique:,} unique keys."
        )

        if not per_key_embs:
            raise ValueError(
                "No embeddings available for cosine splitting. "
                f"Source={self._embedding_source} at {self._embedding_source_path}."
            )

        emb_dim = next(iter(per_key_embs.values())).shape[0]
        emb_matrix = np.zeros((n_unique, emb_dim), dtype=np.float32)
        valid_mask = np.zeros(n_unique, dtype=bool)
        key_to_idx = {key: i for i, key in enumerate(self.unique_keys)}

        for key, emb in per_key_embs.items():
            if key not in key_to_idx:
                continue
            idx = key_to_idx[key]
            emb_matrix[idx] = emb
            valid_mask[idx] = True

        valid_indices = np.where(valid_mask)[0]
        valid_embs = emb_matrix[valid_indices]
        self._valid_embedding_mask = valid_mask.copy()

        if len(valid_indices) == 0:
            raise ValueError(
                "No dataset keys were found in embedding source; cannot compute cosine similarities."
            )

        self.logger.info(
            f"[Cosine] Computing pairwise max for {len(valid_indices):,} keys "
            f"(D={emb_dim}, batch_size={self.COSINE_BATCH_SIZE})..."
        )

        if len(valid_indices) == 1:
            max_sims_valid = np.array([0.0], dtype=np.float64)
        else:
            max_sims_valid = self._compute_max_cosine_batched(valid_embs)

        max_sims = np.zeros(n_unique, dtype=np.float64)
        max_sims[valid_indices] = max_sims_valid

        self._max_cosine_array = max_sims
        self._max_cosine_to_dataset = {
            key: float(max_sims[i]) for i, key in enumerate(self.unique_keys)
        }

        self._save_similarity_cache(
            cache_path=self.MAX_SIM_CACHE_PATH,
            key_column=self.KEY_COLUMN,
            value_column="max_cosine_similarity",
            ordered_keys=self.unique_keys,
            values=max_sims,
        )

        valid_sims = max_sims[valid_indices]
        self.logger.info("Max cosine similarity statistics (over keys with embeddings):")
        self.logger.info(f"  Min:    {valid_sims.min():.4f}")
        self.logger.info(f"  Max:    {valid_sims.max():.4f}")
        self.logger.info(f"  Mean:   {valid_sims.mean():.4f}")
        self.logger.info(f"  Median: {np.median(valid_sims):.4f}")
        self.logger.info(
            f"  Missing embeddings (forced to train): {int((~valid_mask).sum()):,}"
        )
        self.logger.info(
            f"[Cosine] Zero-similarity keys in saved cache: {int(np.sum(max_sims == 0.0)):,}."
        )

        return max_sims

    def get_max_cosine_for_keys(self, keys: list[str]) -> dict[str, float]:
        if self._max_cosine_to_dataset is None:
            self.compute_max_cosine_to_dataset()

        sims = {key: self._max_cosine_to_dataset.get(key, 0.0) for key in keys}

        values = list(sims.values())
        if values:
            self.logger.info(
                f"  {len(values)} keys - min={min(values):.4f}, max={max(values):.4f}, "
                f"mean={np.mean(values):.4f}"
            )

        return sims

    def dissimilarity_based_split(self, similarity_threshold: float) -> tuple[dict, dict]:
        self.logger.info(
            f"Splitting with cosine threshold={similarity_threshold}, "
            f"max_test_frac={self.TEST_FRAC}"
        )

        max_cosine_to_dataset = self.compute_max_cosine_to_dataset()

        unique_keys = self.unique_keys
        n_unique = len(unique_keys)
        n_full = len(self.full_dataset_df)
        max_test_rows = int(self.TEST_FRAC * n_full)

        self.logger.info(f"Max test rows: {max_test_rows:,}")

        row_count_dict = self.full_dataset_df[self.KEY_COLUMN].value_counts().to_dict()
        suspicious_mask = (
            ~self._valid_embedding_mask
            if self._valid_embedding_mask is not None
            else (max_cosine_to_dataset == 0.0)
        )
        suspicious_keys = int(np.sum(suspicious_mask))
        suspicious_rows = sum(
            row_count_dict.get(unique_keys[i], 0)
            for i in np.where(suspicious_mask)[0]
        )
        self.logger.info(
            f"[Cosine] Suspicious low-information keys before thresholding: "
            f"{suspicious_keys:,} keys, {suspicious_rows:,} rows."
        )
        if self._valid_embedding_mask is None:
            self.logger.info(
                "[Cosine] Interpreting suspicious keys as zero-cosine keys from cache."
            )
        else:
            self.logger.info(
                "[Cosine] Interpreting suspicious keys as dataset keys without resolved embeddings."
            )

        dissimilar_mask = (~suspicious_mask) & (max_cosine_to_dataset < similarity_threshold)
        dissimilar_indices = np.where(dissimilar_mask)[0]
        dissimilar_suspicious_keys = int(np.sum(suspicious_mask[dissimilar_indices]))
        dissimilar_suspicious_rows = sum(
            row_count_dict.get(unique_keys[i], 0)
            for i in dissimilar_indices
            if suspicious_mask[i]
        )
        self.logger.info(
            f"  {len(dissimilar_indices):,} dissimilar keys (max cosine < {similarity_threshold})"
        )
        self.logger.info(
            f"[Cosine] Dissimilar pool composition at threshold {similarity_threshold}: "
            f"{dissimilar_suspicious_keys:,} suspicious keys "
            f"covering {dissimilar_suspicious_rows:,} rows."
        )
        self.logger.info(
            f"[Cosine] Suspicious keys excluded from test/val consideration: "
            f"{suspicious_keys:,} keys, {suspicious_rows:,} rows."
        )

        dissimilar_indices = dissimilar_indices[
            np.argsort(max_cosine_to_dataset[dissimilar_indices])
        ]

        test_indices = []
        test_row_count = 0
        for idx in tqdm(dissimilar_indices, desc="Filling test set", leave=False):
            key = unique_keys[idx]
            row_count = row_count_dict.get(key, 0)
            if test_row_count + row_count <= max_test_rows:
                test_indices.append(idx)
                test_row_count += row_count
            else:
                break

        self.logger.info(
            f"  Test set: {len(test_indices):,} keys, {test_row_count:,} rows"
        )
        test_suspicious_keys = int(np.sum(suspicious_mask[test_indices]))
        test_suspicious_rows = sum(
            row_count_dict.get(unique_keys[i], 0)
            for i in test_indices
            if suspicious_mask[i]
        )
        self.logger.info(
            f"[Cosine] Test set suspicious-key coverage: "
            f"{test_suspicious_keys:,} keys, {test_suspicious_rows:,} rows."
        )

        used_indices = set(test_indices)
        remaining_indices = [
            idx for idx in range(n_unique) if idx not in used_indices and not suspicious_mask[idx]
        ]
        remaining_sims = max_cosine_to_dataset[remaining_indices]
        remaining_indices = [remaining_indices[i] for i in np.argsort(remaining_sims)]

        target_val_rows = (
            test_row_count
            if self.MATCH_VAL_TO_TEST
            else int(self.VAL_FRAC * n_full)
        )

        val_indices = []
        val_row_count = 0
        for idx in tqdm(remaining_indices, desc="Filling val set", leave=False):
            key = unique_keys[idx]
            row_count = row_count_dict.get(key, 0)
            if val_row_count + row_count <= target_val_rows:
                val_indices.append(idx)
                val_row_count += row_count
            else:
                break

        self.logger.info(f"  Val set: {len(val_indices):,} keys, {val_row_count:,} rows")
        val_suspicious_keys = int(np.sum(suspicious_mask[val_indices]))
        val_suspicious_rows = sum(
            row_count_dict.get(unique_keys[i], 0)
            for i in val_indices
            if suspicious_mask[i]
        )
        self.logger.info(
            f"[Cosine] Val set suspicious-key coverage: "
            f"{val_suspicious_keys:,} keys, {val_suspicious_rows:,} rows."
        )

        val_set = set(val_indices)
        train_indices = [
            idx for idx in range(n_unique) if idx not in used_indices and idx not in val_set
        ]
        train_row_count = n_full - test_row_count - val_row_count
        self.logger.info(
            f"  Train set: {len(train_indices):,} keys, ~{train_row_count:,} rows"
        )
        train_suspicious_keys = int(np.sum(suspicious_mask[train_indices]))
        train_suspicious_rows = sum(
            row_count_dict.get(unique_keys[i], 0)
            for i in train_indices
            if suspicious_mask[i]
        )
        self.logger.info(
            f"[Cosine] Train set suspicious-key coverage: "
            f"{train_suspicious_keys:,} keys, {train_suspicious_rows:,} rows."
        )

        test_keys = [unique_keys[i] for i in test_indices]
        val_keys = [unique_keys[i] for i in val_indices]
        train_keys = [unique_keys[i] for i in train_indices]

        splits = {
            "train": self.full_dataset_df[
                self.full_dataset_df[self.KEY_COLUMN].isin(train_keys)
            ].reset_index(drop=True),
            "val": self.full_dataset_df[
                self.full_dataset_df[self.KEY_COLUMN].isin(val_keys)
            ].reset_index(drop=True),
            "test": self.full_dataset_df[
                self.full_dataset_df[self.KEY_COLUMN].isin(test_keys)
            ].reset_index(drop=True),
        }

        assert len(set(test_keys) & set(val_keys)) == 0, "Test and val overlap"
        assert len(set(test_keys) & set(train_keys)) == 0, "Test and train overlap"
        assert len(set(val_keys) & set(train_keys)) == 0, "Val and train overlap"

        test_similarities = self.get_max_cosine_for_keys(test_keys)
        val_similarities = self.get_max_cosine_for_keys(val_keys)
        similarities = {"test": test_similarities, "val": val_similarities}

        nan_mask = self.full_dataset_df[self.KEY_COLUMN].isna()
        if nan_mask.any():
            nan_rows = self.full_dataset_df[nan_mask].reset_index(drop=True)
            splits["train"] = pd.concat([splits["train"], nan_rows], ignore_index=True)
            self.logger.info(
                f"  Added {len(nan_rows):,} NaN {self.KEY_COLUMN} rows to train."
            )

        for split_name, split_df in splits.items():
            pct = 100 * len(split_df) / n_full
            self.logger.info(f"  {split_name}: {len(split_df):,} rows ({pct:.1f}%)")

        return splits, similarities

    def run_splits_across_thresholds(self) -> tuple[dict, dict]:
        self.logger.info("Computing max cosine similarities to full dataset (once)...")
        self.compute_max_cosine_to_dataset()

        all_splits = {}
        all_similarities = {}

        for threshold in tqdm(
            self.THRESHOLDS,
            desc="Processing thresholds",
            unit="threshold",
        ):
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"Cosine threshold: {threshold}")
            self.logger.info(f"{'=' * 50}")

            splits, similarities = self.dissimilarity_based_split(
                similarity_threshold=threshold
            )
            all_splits[threshold] = splits
            all_similarities[threshold] = similarities

        return all_splits, all_similarities

    def plot_cosine_distribution(self, all_similarities: dict) -> None:
        values_by_threshold = self._build_value_map_from_similarity_dicts(
            all_similarities
        )

        self._plot_threshold_split_distributions(
            values_by_threshold=values_by_threshold,
            bins=np.linspace(-1, 1, 41),
            output_path=self.UNIQUE_COSINE_SIMILARITY_PLOT_PATH,
            xlabel="Max Cosine Similarity to Dataset",
            ylabel="Count",
            title_fn=lambda threshold, val_values, test_values: (
                f"Threshold: {threshold}\n"
                f"Val: {len(val_values)}, Test: {len(test_values)}"
            ),
            stats_mode="summary",
            legend_loc="upper right",
            hist_range=(-1, 1),
            xlim=(-1, 1),
        )

    def plot_full_dataset_cosine_distribution(
        self,
        all_splits: dict,
        all_similarities: dict,
    ) -> None:
        full_dataset_size = len(self.full_dataset_df)
        values_by_threshold = self._build_value_map_from_split_frames(
            all_splits=all_splits,
            all_similarities=all_similarities,
            key_column=self.KEY_COLUMN,
        )

        self._plot_threshold_split_distributions(
            values_by_threshold=values_by_threshold,
            bins=np.linspace(-1, 1, 41),
            output_path=self.FULL_DATASET_COSINE_SIMILARITY_PLOT_PATH,
            xlabel="Max Cosine Similarity to Dataset",
            ylabel="Number of Rows",
            title_fn=lambda threshold, val_values, test_values: (
                f"Full Dataset Embedding Cosine Similarity\nThreshold {threshold}\n"
                f"Val: {len(val_values)} ({100 * len(val_values) / full_dataset_size:.1f}%), "
                f"Test: {len(test_values)} ({100 * len(test_values) / full_dataset_size:.1f}%)"
            ),
            stats_mode="detailed",
            legend_loc="upper left",
            hist_range=(-1, 1),
            xlim=(-1, 1),
        )

    def plot_unique_distribution(self, all_similarities: dict) -> None:
        self.plot_cosine_distribution(all_similarities)

    def plot_full_distribution(self, all_splits: dict, all_similarities: dict) -> None:
        self.plot_full_dataset_cosine_distribution(all_splits, all_similarities)

    def get_output_dir(self) -> Path:
        return self.OUTPUT_DIR


# Backward-compatible alias for existing imports.
ConformerCosineSimilaritySplitter = EmbeddingCosineSimilaritySplitter


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="data_processing")

    splitter = EmbeddingCosineSimilaritySplitter(cfg=cfg)
    all_splits, all_similarities = splitter.generate_splits()
