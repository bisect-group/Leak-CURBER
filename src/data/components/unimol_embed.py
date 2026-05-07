import os
import pickle
import logging
import warnings
import numpy as np
import multiprocessing
from pathlib import Path
from collections import Counter
from src.utils.tqdmlogger import TqdmLogger
from omegaconf import DictConfig, ListConfig


try:
    import lz4.frame as _lz4
    _HAS_LZ4 = True
except ImportError:
    _HAS_LZ4 = False

warnings.filterwarnings("ignore")

# Shared featurized data. Set in embed() before forking so workers inherit it
# via copy-on-write — the 220 GB list is never serialized over IPC.
_FEATS_SORTED: list = []


# ── GPU worker (module-level for fork-safe access) ───────────────────────────

def _gpu_inference_worker(
    rank: int,
    gpu_id: str,
    stride: int,
    model_size: str,
    max_atoms: int,
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
        max_atoms=max_atoms,
        batch_size=batch_size,
    )
    dataset = MolDataset(feats_chunk)
    trainer = Trainer(
        task="repr",
        batch_size=batch_size,
        use_cuda=True,
        use_amp=False,  # no GradScaler needed for inference
        use_ddp=False,
        use_gpu="0",
    )
    emb_list = trainer.inference_without_ddp(
        model=repr_obj.model,
        dataset=dataset,
        model_name="unimolv2",
        return_repr=True,
        return_atomic_reprs=False,
    )
    return rank, emb_list


# ── Main class ─────────────────────────────────────────────────────────────────

class UnimolSDFEmbedder:
    """
    Multi-GPU inference pipeline for UniMolV2 SDF embedding.

    Loads the featurized checkpoint written by UnimolSDFFeaturizer, splits
    molecules evenly across N GPUs (spawn-based, one process per GPU), and
    saves the final embeddings as a compressed .npz file.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize from Hydra config.

        Config keys expected (under cfg.unimol_sdf_embeddings):
            - unimol_sdf_raw_dir:    directory containing featurized.pkl.lz4/.pkl
            - unimol_sdf_output_npz: path to write the output .npz file
            - unimol_sdf_gpu_ids:    list of CUDA device indices, e.g. [0, 1, 2]
            - unimol_sdf_batch_size: per-GPU inference batch size
            - unimol_sdf_model_size: model size (84m|164m|310m|570m|1.1B)
            - log_dir:               directory for log file
        """
        scfg = cfg.unimol_sdf_embeddings

        self.RAW_DIR = Path(scfg.unimol_sdf_raw_dir)
        self.OUTPUT_NPZ = Path(scfg.unimol_sdf_output_npz)
        self.BATCH_SIZE = scfg.unimol_sdf_batch_size
        self.MODEL_SIZE = scfg.unimol_sdf_model_size

        raw_ids = scfg.unimol_sdf_gpu_ids
        self.GPU_IDS = (
            [str(g) for g in raw_ids]
            if isinstance(raw_ids, (list, ListConfig))
            else [str(raw_ids)]
        )

        LOG_DIR = Path(scfg.log_dir)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_DIR,
            log_file_name="unimol_embed_gpus.log",
        ).get_logger()

        self.logger.info(
            f"GPU_IDS={self.GPU_IDS}  batch_size={self.BATCH_SIZE}  "
            f"model_size={self.MODEL_SIZE}  n_gpus={len(self.GPU_IDS)}"
        )

    def embed(self, *, save_npz: bool = True) -> dict[str, np.ndarray]:
        """Load featurized checkpoint, run multi-GPU inference, and optionally save embeddings."""
        # ── Load checkpoint ───────────────────────────────────────────────────
        # Probe for whichever format UnimolSDFFeaturizer wrote.
        for candidate in (
            self.RAW_DIR / "featurized.pkl.lz4",
            self.RAW_DIR / "featurized.pkl",
            self.RAW_DIR / "featurized.pkl.gz",  # legacy gzip
        ):
            if candidate.exists():
                save_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"No featurized data found in {self.RAW_DIR}. "
                "Run UnimolSDFFeaturizer.featurize() first."
            )

        self.logger.info(f"Loading featurized data from {save_path}…")
        if save_path.suffix == ".lz4":
            if not _HAS_LZ4:
                raise ImportError(
                    "Checkpoint is lz4-compressed but lz4 is not installed. "
                    "Run: pip install lz4"
                )
            with _lz4.open(save_path, "rb") as fh:
                checkpoint = pickle.load(fh)
        elif save_path.suffix == ".gz":
            import gzip
            with gzip.open(save_path, "rb") as fh:
                checkpoint = pickle.load(fh)
        else:
            with open(save_path, "rb") as fh:
                checkpoint = pickle.load(fh)

        max_atoms = checkpoint["max_atoms"]
        feats_sorted = checkpoint["feats_sorted"]
        stems_sorted = checkpoint["stems_sorted"]
        recidx_sorted = checkpoint["recidx_sorted"]
        total_mols = len(feats_sorted)
        self.logger.info(f"  {total_mols:,} molecules, MAX_ATOMS={max_atoms}")

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
                max_atoms,
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

        # ── Phase 4: reassemble {stem → (n_records, embed_dim)} and save ─────
        self.logger.info("[Phase 4] Reassembling results…")
        stem_counts = Counter(stems_sorted)
        stem_rows: dict = {stem: [None] * count for stem, count in stem_counts.items()}
        for stem, rec_idx, emb in zip(stems_sorted, recidx_sorted, emb_list):
            stem_rows[stem][rec_idx] = emb

        output = {stem: np.stack(rows) for stem, rows in stem_rows.items()}

        if save_npz:
            self.logger.info(f"Saving to {self.OUTPUT_NPZ}…")
            np.savez_compressed(str(self.OUTPUT_NPZ), **output)

            self.logger.info(f"\nDone.  {len(output):,} entries saved.")
            first_stem = next(iter(output))
            self.logger.info(f"Example: '{first_stem}' → shape {output[first_stem].shape}")
            self.logger.info(
                f"\nLoad with:\n  import numpy as np\n"
                f"  data = np.load('{self.OUTPUT_NPZ}')\n"
                f"  emb  = data['{first_stem}']   # shape: {output[first_stem].shape}"
            )
        else:
            self.logger.info(f"\nDone.  {len(output):,} entries produced in memory.")

        return output
