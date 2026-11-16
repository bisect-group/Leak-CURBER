from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import sys
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig
from rdkit import Chem, RDLogger

from src.data.components.embedders.store import ShardedEmbeddingStore
from src.data.components.embedders.types import CacheMetadata
from src.data.components.embedders.utils import load_pickle_items
from src.utils.rich_progress import component_logger, progress, write

warnings.filterwarnings("ignore")

# ── Worker helpers (module-level so Pool can pickle them) ─────────────────────


def _init_worker():
    """Silence logging and RDKit noise inside every worker process."""
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")


def _featurize_sdf(sdf_path: str):
    """Featurize every record in one SDF file."""
    from unimol_tools.data.conformer import mol2unimolv2

    stem = Path(sdf_path).stem
    try:
        supp = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
        feats = []
        valid_records = 0
        sample_error = None
        for mol in supp:
            if mol is None:
                continue
            valid_records += 1
            try:
                feats.append(mol2unimolv2(mol, max_atoms=sys.maxsize, remove_hs=True))
            except Exception as exc:
                if sample_error is None:
                    sample_error = str(exc)
        return stem, feats, valid_records, sample_error
    except OSError as exc:
        # Never delete the input; the empty result makes the caller log the stem as failed.
        return stem, [], 0, f"Unreadable SDF (kept): {exc}"


# ── Featurizer ─────────────────────────────────────────────────────────────────

class UnimolSDFFeaturizer:
    """
    CPU featurisation pipeline for UniMolV2 SDF embedding.

    Featurises every record of the given SDF files without atom cropping and
    sorts the records by size. The result goes to UnimolSDFEmbedder.embed().
    """

    def __init__(self, cfg: DictConfig):
        """
        Config keys expected (under cfg.embeddings):
          - unimol_num_workers: CPU worker count (0 = auto)
          - log_dir:            directory for log file
        """
        scfg = cfg.embeddings
        self.NUM_WORKERS = scfg.unimol_num_workers or os.cpu_count()

        LOG_DIR = Path(scfg.log_dir)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        self.logger = component_logger(
            log_dir=LOG_DIR,
            log_file_name="unimol_featurize.log",
        ).get_logger()
        self.logger.info(f"num_workers={self.NUM_WORKERS}")

    def featurize(self, sdf_paths: list[Path]) -> dict:
        """Featurise the SDF files and return the records sorted by atom count.

        Keys: feats_sorted, stems_sorted, recidx_sorted.
        """
        missing_paths = [str(p) for p in sdf_paths if not Path(p).exists()]
        if missing_paths:
            raise FileNotFoundError(
                f"SDF input paths are missing. Examples: {', '.join(missing_paths[:10])}"
            )
        if not sdf_paths:
            raise FileNotFoundError("No SDF input paths given.")
        sdf_strs = [str(p) for p in sdf_paths]
        self.logger.info(f"Found {len(sdf_strs):,} SDF files  ({self.NUM_WORKERS} workers).")

        # ── Phase 1: parallel CPU featurisation ───────────────────────────────
        self.logger.info("[Phase 1] Featurising…")
        per_stem: dict = {}
        empty_stems: list[str] = []
        sample_errors: list[str] = []
        readable_records = 0
        with Pool(self.NUM_WORKERS, initializer=_init_worker) as pool:
            for stem, feats, valid_records, sample_error in progress(
                pool.imap_unordered(_featurize_sdf, sdf_strs),
                total=len(sdf_strs),
                desc="Featurising",
            ):
                readable_records += valid_records
                if feats:
                    per_stem[stem] = feats
                else:
                    empty_stems.append(stem)
                    if sample_error and len(sample_errors) < 10:
                        sample_errors.append(f"{stem}: {sample_error}")

        total_mols = sum(len(v) for v in per_stem.values())
        self.logger.info(f"  {total_mols:,} records from {len(per_stem):,} files.")
        if empty_stems:
            self.logger.warning(
                f"  {len(empty_stems):,} SDF files produced zero UniMol-ready records "
                f"(failed stems, first 10: {empty_stems[:10]}; errors: {sample_errors})."
            )

        if total_mols == 0:
            sample_stems = ", ".join(empty_stems[:10]) if empty_stems else "n/a"
            error_block = "\n".join(sample_errors) if sample_errors else "No per-file exceptions captured."
            raise ValueError(
                "UniMol featurization produced zero valid records. "
                f"Readable RDKit records before featurization: {readable_records:,}. "
                f"Sample failing stems: {sample_stems}. "
                f"Sample exceptions:\n{error_block}"
            )

        # ── Phase 2: sort by atom count for efficient GPU batching ────────────
        self.logger.info("[Phase 2] Sorting by atom count…")
        flat = [
            (stem, i, feat)
            for stem, feats in per_stem.items()
            for i, feat in enumerate(feats)
        ]
        flat.sort(key=lambda t: len(t[2]["src_tokens"]))

        stems_sorted = [t[0] for t in flat]
        recidx_sorted = [t[1] for t in flat]
        feats_sorted = [t[2] for t in flat]
        self.logger.info(
            f"  Smallest mol: {len(feats_sorted[0]['src_tokens'])} atoms  |  "
            f"Largest mol: {len(feats_sorted[-1]['src_tokens'])} atoms"
        )

        return {
            "feats_sorted": feats_sorted,
            "stems_sorted": stems_sorted,
            "recidx_sorted": recidx_sorted,
        }


# Shared featurized data. Set in embed() before forking so workers inherit it
# via copy-on-write — the 220 GB list is never serialized over IPC.
_FEATS_SORTED: list = []


# ── GPU worker (module-level for fork-safe access) ───────────────────────────

def _embed_by_atom_count(feats: list, infer) -> list:
    """
    Run infer(feats_subset) -> embeddings once per atom count and return the
    embeddings in the order of feats.

    Uni-Mol2 does not mask padded atoms, so a mini-batch must not mix atom counts.
    Each call of infer receives molecules of one atom count; its own mini-batching
    then cannot pad.
    """
    groups: dict[int, list[int]] = {}
    for i, feat in enumerate(feats):
        groups.setdefault(len(feat["src_tokens"]), []).append(i)
    embs: list = [None] * len(feats)
    for idxs in groups.values():
        for i, emb in zip(idxs, infer([feats[i] for i in idxs])):
            embs[i] = emb
    return embs


def _gpu_inference_worker(
    rank: int,
    gpu_id: str,
    stride: int,
    model_size: str,
    batch_size: int,
) -> tuple:
    """
    Forked worker: takes every stride-th molecule starting at rank (interleaved),
    so each GPU processes an equal mix of small and large molecules.
    The stride slice allocates only a list of pointers; actual feat dicts stay on
    shared copy-on-write pages.
    """
    # Must be set before anything that could initialise CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    # Interleaved CoW slice — each GPU gets molecules rank, rank+stride, rank+2*stride, …
    feats_chunk = _FEATS_SORTED[rank::stride]

    from unimol_tools.predictor import UniMolRepr, MolDataset
    from unimol_tools.tasks.trainer import Trainer

    repr_obj = UniMolRepr(
        model_name="unimolv2",
        model_size=model_size,
        use_cuda=True,
        use_ddp=False,
        use_gpu="0",  # always "0" — CUDA_VISIBLE_DEVICES already remapped the card
        batch_size=batch_size,
    )
    trainer = Trainer(
        task="repr",
        batch_size=batch_size,
        use_cuda=True,
        use_amp=False,  # no GradScaler needed for inference
        use_ddp=False,
        use_gpu="0",
    )
    emb_list = _embed_by_atom_count(
        feats_chunk,
        lambda feats: trainer.inference_without_ddp(
            model=repr_obj.model,
            dataset=MolDataset(feats),
            model_name="unimolv2",
            return_repr=True,
            return_atomic_reprs=False,
        ),
    )
    return rank, emb_list


# ── Main class ─────────────────────────────────────────────────────────────────

class UnimolSDFEmbedder:
    """
    Multi-GPU inference pipeline for UniMolV2 SDF embedding.

    Takes the records from UnimolSDFFeaturizer.featurize(), splits
    molecules evenly across N GPUs (spawn-based, one process per GPU), and
    returns the final embeddings in memory.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize from Hydra config.

        Config keys expected (under cfg.embeddings):
            - unimol_gpu_ids:    list of CUDA device indices, e.g. [0, 1, 2]
            - unimol_batch_size: per-GPU inference batch size
            - unimol_model_size: model size (84m|164m|310m|570m|1.1B)
            - log_dir:           directory for log file
        """
        scfg = cfg.embeddings

        self.BATCH_SIZE = scfg.unimol_batch_size
        self.MODEL_SIZE = scfg.unimol_model_size

        raw_ids = scfg.unimol_gpu_ids
        self.GPU_IDS = (
            [str(g) for g in raw_ids]
            if isinstance(raw_ids, (list, ListConfig))
            else [str(raw_ids)]
        )

        LOG_DIR = Path(scfg.log_dir)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        self.logger = component_logger(
            log_dir=LOG_DIR,
            log_file_name="unimol_embed_gpus.log",
        ).get_logger()

        self.logger.info(
            f"GPU_IDS={self.GPU_IDS}  batch_size={self.BATCH_SIZE}  "
            f"model_size={self.MODEL_SIZE}  n_gpus={len(self.GPU_IDS)}"
        )

    def embed(self, data: dict) -> dict[str, np.ndarray]:
        """Run multi-GPU inference on featurized records and return {stem: embeddings}.

        Consumes data: feats_sorted is popped so that this method holds the only
        reference, which it drops after the workers fork.
        """
        feats_sorted = data.pop("feats_sorted")
        stems_sorted = data["stems_sorted"]
        recidx_sorted = data["recidx_sorted"]
        total_mols = len(feats_sorted)
        self.logger.info(f"  {total_mols:,} molecules")

        # ── Pre-download weights so spawned workers don't race ────────────────
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.GPU_IDS)

        from unimol_tools.config import MODEL_CONFIG_V2
        from unimol_tools.weights.weighthub import get_weight_dir, weight_download_v2

        weight_dir = get_weight_dir()
        pretrain = MODEL_CONFIG_V2["weight"][self.MODEL_SIZE]
        self.logger.info(f"Pre-checking weights ({pretrain})…")
        weight_download_v2(pretrain, weight_dir)  # no-op if already cached

        # ── Phase 3: split feats evenly across GPUs and dispatch ──────────────
        n_gpus = len(self.GPU_IDS)
        self.logger.info(f"[Phase 3] Dispatching to {n_gpus} GPU(s)…")
        worker_args = [
            (
                rank,
                self.GPU_IDS[rank],
                n_gpus,  # stride — each GPU takes every n_gpus-th molecule
                self.MODEL_SIZE,
                self.BATCH_SIZE,
            )
            for rank in range(n_gpus)
        ]
        self.logger.info(
            f"  ~{total_mols // n_gpus:,} molecules/GPU (interleaved, stride={n_gpus})"
        )

        # Populate the module-level global before forking so workers inherit the
        # full list via copy-on-write.  Only index ranges (cheap ints) are sent
        # over IPC — the 220 GB of feat dicts is never serialized.
        global _FEATS_SORTED
        _FEATS_SORTED = feats_sorted
        del feats_sorted  # drop local ref so we hold only one

        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(n_gpus) as pool:
            # Workers are forked here and inherit _FEATS_SORTED via CoW.
            # Release the parent's reference immediately: physical pages remain
            # alive (mapped by workers) but the main process is now ~220 GB lighter.
            _FEATS_SORTED = []
            self.logger.info(f"  Waiting for {n_gpus} GPU worker(s) to complete…")
            rank_results = pool.starmap(_gpu_inference_worker, worker_args)

        # Un-interleave: worker `rank` processed indices rank, rank+n_gpus, rank+2*n_gpus, …
        # so the j-th embedding from worker rank → original index rank + j*n_gpus.
        emb_list = [None] * total_mols
        for rank, chunk_embs in rank_results:
            for j, emb in enumerate(chunk_embs):
                emb_list[rank + j * n_gpus] = emb
        embed_dim = emb_list[0].shape[-1]
        self.logger.info(f"  {len(emb_list):,} embeddings, dim={embed_dim}")

        # ── Phase 4: reassemble {stem → (n_records, embed_dim)} ─────────────
        self.logger.info("[Phase 4] Reassembling results…")
        stem_counts = Counter(stems_sorted)
        stem_rows: dict = {stem: [None] * count for stem, count in stem_counts.items()}
        for stem, rec_idx, emb in zip(stems_sorted, recidx_sorted, emb_list):
            stem_rows[stem][rec_idx] = emb

        output = {stem: np.stack(rows) for stem, rows in stem_rows.items()}

        self.logger.info(f"\nDone.  {len(output):,} entries produced in memory.")

        return output


class UnimolSDFShardEmbedder:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        log_dir = Path(cfg.embeddings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = component_logger(
            log_dir=log_dir,
            log_file_name=cfg.embeddings.unimol_log_file_name,
        ).get_logger()

        self.sdf_dir = Path(cfg.embeddings.unimol_sdf_dir)
        self.store_dir = Path(cfg.embeddings.unimol_embeddings_store_dir)
        self.num_conformers = int(cfg.embeddings.datamol_num_conformers)
        self.stems_per_batch = int(
            cfg.embeddings.get("unimol_stems_per_batch", 1_000_000)
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
        input_path = Path(input_override or self.cfg.embeddings.datamol_conformer_input_pkl_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input pickle not found at {input_path}")
        return input_path

    def _load_smiles_df(self, input_path: Path) -> pd.DataFrame:
        keys = sorted(set(load_pickle_items(input_path, "smiles_key")))
        return pd.DataFrame(
            {
                "smiles_hash": [hashlib.sha256(k.encode()).hexdigest() for k in keys],
                "smiles_key": keys,
            }
        ).sort_values("smiles_hash", ignore_index=True)

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
    ) -> int:
        self.logger.info(
            f"Batch {batch_index + 1}: processing {len(batch_stems):,} SDF stems"
        )
        data = UnimolSDFFeaturizer(self.cfg).featurize(
            [stem_to_path[stem] for stem in batch_stems]
        )
        arrays_by_key = self._prepare_store_arrays(UnimolSDFEmbedder(self.cfg).embed(data))
        missing_outputs = sorted(set(batch_stems) - set(arrays_by_key))
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
        return written

    def embed(self, *, input_override: str | None = None) -> None:
        input_path = self.resolve_input_path(input_override)
        smiles_df = self._load_smiles_df(input_path)
        stem_to_path = self._build_relevant_sdf_stems(smiles_df)

        store = self._build_store()
        relevant_stems = sorted(stem_to_path.keys())
        missing_stems = store.get_missing_keys(relevant_stems)

        self.logger.info(f"SMILES input: {input_path}")
        self.logger.info(f"SDF dir: {self.sdf_dir}")
        self.logger.info(f"Store dir: {store.root_dir}")
        self.logger.info(f"Unique smiles_hash keys: {len(smiles_df):,}")
        self.logger.info(f"Relevant SDF stems: {len(relevant_stems):,}")
        self.logger.info(f"Missing stems to compute incrementally: {len(missing_stems):,}")
        self.logger.info(f"Configured SDF stems per batch: {self.stems_per_batch:,}")

        if not missing_stems:
            self.logger.info(
                "Store already contains all relevant UniMol embeddings. Nothing to do."
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
            )

        self.logger.info(
            f"Persisted {total_written:,} total incremental UniMol embeddings into {store.root_dir}"
        )
