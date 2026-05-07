import os
import math
import pickle
import logging
import warnings

try:
    import lz4.frame as _lz4
    _HAS_LZ4 = True
except ImportError:
    _HAS_LZ4 = False
import numpy as np
from rdkit import Chem
from pathlib import Path
from rdkit import RDLogger
from tqdm.auto import tqdm
from multiprocessing import Pool
from omegaconf import DictConfig
from src.utils.tqdmlogger import TqdmLogger

warnings.filterwarnings("ignore")

# ── Worker helpers (module-level so Pool can pickle them) ─────────────────────


def _init_worker():
    """Silence logging and RDKit noise inside every worker process."""
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")


def _delete_invalid_sdf(sdf_path: str, exc: OSError) -> bool:
    path = Path(sdf_path).expanduser()
    if not path.exists():
        print(f"Invalid SDF file already absent {path}: {exc}", flush=True)
        return True

    delete_error = None
    for attempt in range(2):
        try:
            if attempt == 1:
                os.chmod(path, 0o666)
            os.remove(path)
            break
        except FileNotFoundError:
            return True
        except Exception as delete_exc:
            delete_error = delete_exc

    deleted = not path.exists()
    if deleted:
        print(f"Deleted invalid SDF file {path}: {exc}", flush=True)
        return True

    print(
        f"Failed to delete invalid SDF file {path}: {exc}. "
        f"Delete error: {delete_error}",
        flush=True,
    )
    return False


def _count_atoms_in_sdf(sdf_path: str) -> list:
    """Return heavy-atom count for every valid record in an SDF file."""
    try:
        supp = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
        return [mol.GetNumAtoms() for mol in supp if mol is not None]
    except OSError as exc:
        _delete_invalid_sdf(sdf_path, exc)
        return []


def _featurize_sdf(args):
    """Featurize every record in one SDF file."""
    from unimol_tools.data.conformer import mol2unimolv2

    sdf_path, max_atoms = args
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
                feats.append(mol2unimolv2(mol, max_atoms=max_atoms, remove_hs=True))
            except Exception as exc:
                if sample_error is None:
                    sample_error = str(exc)
        return stem, feats, valid_records, sample_error
    except OSError as exc:
        deleted = _delete_invalid_sdf(sdf_path, exc)
        status = "deleted" if deleted else "delete_failed"
        return stem, [], 0, f"Invalid SDF {status}: {exc}"


def _next_power_of_two(n: float) -> int:
    """Round n up to the nearest power of two (e.g. 177 → 256, 64 → 64)."""
    return 1 << math.ceil(math.log2(float(n)))


# ── Main class ─────────────────────────────────────────────────────────────────

class UnimolSDFFeaturizer:
    """
    CPU featurisation pipeline for UniMolV2 SDF embedding.

    Runs Phases 0-2 (atom-count survey, mol featurisation, sort by size) and
    saves a compressed checkpoint consumed later by UnimolSDFEmbedder or
    unimol_embed_gpus.py.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize from Hydra config.

        Config keys expected (under cfg.unimol_sdf_embeddings):
          - unimol_sdf_sdf_dir:         directory containing .sdf files
          - unimol_sdf_intermediate_dir: where to save featurized.pkl.gz
          - unimol_sdf_percentile:       percentile for MAX_ATOMS selection (50-100)
          - unimol_sdf_num_workers:      CPU worker count (0 = auto)
          - log_dir:                     directory for log file
        """
        scfg = cfg.unimol_sdf_embeddings

        self.SDF_DIR = Path(scfg.unimol_sdf_sdf_dir)
        self.RAW_DIR = Path(scfg.unimol_sdf_raw_dir)
        self.PERCENTILE = scfg.unimol_sdf_percentile
        self.NUM_WORKERS = scfg.unimol_sdf_num_workers or os.cpu_count()
        configured_sdf_paths = scfg.get("unimol_sdf_input_sdf_paths", None)
        self.INPUT_SDF_PATHS = (
            [Path(p) for p in configured_sdf_paths]
            if configured_sdf_paths
            else None
        )

        LOG_DIR = Path(scfg.log_dir)
        self.RAW_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_DIR,
            log_file_name="unimol_featurize.log",
        ).get_logger()

        if not self.SDF_DIR.exists():
            self.logger.error(f"SDF directory {self.SDF_DIR} does not exist.")
            raise FileNotFoundError(f"SDF directory {self.SDF_DIR} does not exist.")

        self.logger.info(
            f"SDF_DIR={self.SDF_DIR}, RAW_DIR={self.RAW_DIR}, "
            f"num_workers={self.NUM_WORKERS}, percentile={self.PERCENTILE}"
        )

    def featurize(self) -> None:
        """Run Phases 0-2 and save the checkpoint to disk."""
        if self.INPUT_SDF_PATHS is not None:
            sdf_files = [p for p in self.INPUT_SDF_PATHS if p.exists()]
            missing_paths = [
                str(p) for p in self.INPUT_SDF_PATHS if not p.exists()
            ]
            if missing_paths:
                sample_missing = ", ".join(missing_paths[:10])
                raise FileNotFoundError(
                    "Configured SDF input paths are missing. "
                    f"Examples: {sample_missing}"
                )
        else:
            sdf_files = list(self.SDF_DIR.glob("*.sdf"))
        if not sdf_files:
            raise FileNotFoundError(f"No .sdf files found under {self.SDF_DIR}")
        sdf_strs = [str(p) for p in sdf_files]
        self.logger.info(f"Found {len(sdf_files):,} SDF files  ({self.NUM_WORKERS} workers).")

        # ── Phase 0: atom-count survey ────────────────────────────────────────
        self.logger.info("[Phase 0] Surveying atom counts…")
        with Pool(self.NUM_WORKERS, initializer=_init_worker) as pool:
            per_file_counts = list(
                tqdm(
                    pool.imap_unordered(_count_atoms_in_sdf, sdf_strs),
                    total=len(sdf_strs),
                    desc="Counting atoms",
                )
            )

        all_counts = np.array([n for sub in per_file_counts for n in sub])
        if all_counts.size == 0:
            raise ValueError(
                f"No readable RDKit molecules were found under {self.SDF_DIR}. "
                "Cannot continue UniMol featurization."
            )
        self.logger.info("  Heavy-atom count distribution:")
        for p in (50, 90, 95, 99, 100):
            self.logger.info(f"    p{p:3d}: {np.percentile(all_counts, p):.0f}")

        pct_val = float(np.percentile(all_counts, self.PERCENTILE))
        MAX_ATOMS = _next_power_of_two(pct_val)
        n_crops = int((all_counts > MAX_ATOMS).sum())
        self.logger.info(
            f"  p{self.PERCENTILE} = {pct_val:.0f}  →  MAX_ATOMS = {MAX_ATOMS}  "
            f"({n_crops:,}/{len(all_counts):,} = "
            f"{100 * n_crops / len(all_counts):.1f}% will be atom-cropped)"
        )

        # ── Phase 1: parallel CPU featurisation ───────────────────────────────
        self.logger.info(f"[Phase 1] Featurising (MAX_ATOMS={MAX_ATOMS})…")
        work_args = [(s, MAX_ATOMS) for s in sdf_strs]
        per_stem: dict = {}
        empty_stems: list[str] = []
        sample_errors: list[str] = []
        readable_records = 0
        with Pool(self.NUM_WORKERS, initializer=_init_worker) as pool:
            for stem, feats, valid_records, sample_error in tqdm(
                pool.imap_unordered(_featurize_sdf, work_args),
                total=len(work_args),
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
                f"  {len(empty_stems):,} SDF files produced zero UniMol-ready records."
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

        # ── Save checkpoint ───────────────────────────────────────────────────
        # lz4 is ~10x faster than gzip with similar compression; fall back to
        # plain pickle if lz4 is not installed (install with: pip install lz4).
        if _HAS_LZ4:
            save_path = self.RAW_DIR / "featurized.pkl.lz4"
        else:
            save_path = self.RAW_DIR / "featurized.pkl"
            self.logger.warning(
                "lz4 not found — saving uncompressed (~200 GB). "
                "Install with: pip install lz4"
            )
        self.logger.info(f"Saving featurized data to {save_path}…")
        save_data = {
            "max_atoms": MAX_ATOMS,
            "feats_sorted": feats_sorted,
            "stems_sorted": stems_sorted,
            "recidx_sorted": recidx_sorted,
        }
        if _HAS_LZ4:
            with _lz4.open(save_path, "wb") as fh:
                pickle.dump(save_data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(save_path, "wb") as fh:
                pickle.dump(save_data, fh, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(
            f"Done.  {total_mols:,} molecules saved to {save_path}.\n"
            "Run UnimolSDFEmbedder.embed() next."
        )
