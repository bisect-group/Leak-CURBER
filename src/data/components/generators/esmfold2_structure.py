from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import sys
import traceback
import types
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig
from src.utils.rich_progress import progress, write

from src.data.components.embedders.utils import sha256_short, utc_timestamp
from src.utils.rich_progress import component_logger


SMOKE_TEST_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"

# Future complex-generation inputs should build StructurePredictionInput with
# multiple ProteinInput chains plus optional DNAInput/RNAInput/LigandInput,
# Modification, and CovalentBond objects. The bulk path here is intentionally
# protein-sequence only, with optional per-accession monomer MSA files.


def _ensure_transformers_vision_imports() -> None:
    try:
        from torchvision.transforms import InterpolationMode  # noqa: F401
    except Exception:
        class _InterpolationMode:
            NEAREST_EXACT = "nearest_exact"
            BOX = "box"
            BILINEAR = "bilinear"
            HAMMING = "hamming"
            BICUBIC = "bicubic"
            LANCZOS = "lanczos"

        torchvision = types.ModuleType("torchvision")
        torchvision.__spec__ = ModuleSpec("torchvision", None)
        transforms = types.ModuleType("torchvision.transforms")
        transforms.__spec__ = ModuleSpec("torchvision.transforms", None)
        transforms.InterpolationMode = _InterpolationMode
        torchvision.transforms = transforms
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.transforms"] = transforms


@dataclass(frozen=True)
class ProteinStructureRecord:
    acc_id: str
    sequence: str
    uniprot_ids: tuple[str, ...] = ()
    source_pdbs: tuple[str, ...] = ()
    pdb_type: str | None = None


@dataclass(frozen=True)
class ESMFold2WorkerConfig:
    model_id: str
    cif_path: str
    pdb_path: str
    msa_dir: str | None
    msa_max_sequences: int
    num_loops: int
    num_sampling_steps: int
    num_diffusion_samples: int
    seed: int | None
    write_pdb_compat: bool
    torch_dtype: str | None
    rows_dir: str | None = None


def _append_row_line(rows_dir: str | None, gpu_id: int, kind: str, row: dict) -> None:
    # Each row is appended to a per-GPU jsonl file as soon as it is done, so an OOM kill
    # keeps the manifest rows of finished structures.
    if rows_dir is None:
        return
    Path(rows_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(rows_dir) / f"gpu{gpu_id}.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": kind, "row": row}, default=str) + "\n")


def _merge_incremental_rows(rows_dir: Path, manifest_path: Path, failures_path: Path) -> tuple[int, int]:
    """Move the rows left in the per-GPU jsonl files into the manifest and failures files."""
    manifest_rows: list[dict] = []
    failure_rows: list[dict] = []
    files = sorted(rows_dir.glob("gpu*.jsonl")) if rows_dir.exists() else []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by a kill
            (manifest_rows if entry["kind"] == "manifest" else failure_rows).append(entry["row"])
    _append_rows(manifest_path, manifest_rows)
    _append_rows(failures_path, failure_rows)
    for path in files:
        path.unlink()
    return len(manifest_rows), len(failure_rows)


def _torch_dtype_from_name(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None or str(dtype_name).lower() in {"", "none", "null"}:
        return None
    dtype_name = str(dtype_name).lower()
    if dtype_name in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported ESMFold2 torch dtype: {dtype_name}")


def _sanitize_output_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._|\-]+", "_", str(value))


def _sorted_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    try:
        return tuple(sorted(str(item) for item in value if str(item)))
    except TypeError:
        item = str(value)
        return (item,) if item else ()


def _record_from_input_item(item: dict, input_path: Path) -> ProteinStructureRecord | None:
    if "acc_id" in item and "sequence" in item:
        acc_id = str(item["acc_id"])
        sequence = str(item["sequence"])
        if not acc_id or not sequence:
            return None
        uniprot_ids = _sorted_strings(item.get("uniprot_id", (acc_id,)))
        if not uniprot_ids:
            uniprot_ids = (acc_id,)
        return ProteinStructureRecord(
            acc_id=_sanitize_output_id(acc_id),
            sequence=sequence,
            uniprot_ids=uniprot_ids,
            source_pdbs=_sorted_strings(item.get("pdbs")),
            pdb_type=item.get("pdb_type"),
        )

    if "sequence" in item and "uniprot_id" in item:
        sequence = str(item["sequence"])
        uniprot_ids = _sorted_strings(item.get("uniprot_id"))
        source_pdbs = _sorted_strings(item.get("pdbs"))
        if not sequence:
            return None
        if uniprot_ids:
            acc_id = uniprot_ids[0]
        elif source_pdbs:
            acc_id = source_pdbs[0]
        else:
            raise ValueError(
                f"Input dictionaries in {input_path} with sequence-only schema must "
                "contain a non-empty 'uniprot_id' or 'pdbs' set."
            )
        return ProteinStructureRecord(
            acc_id=_sanitize_output_id(acc_id),
            sequence=sequence,
            uniprot_ids=uniprot_ids,
            source_pdbs=source_pdbs,
            pdb_type=item.get("pdb_type"),
        )

    raise ValueError(
        f"Input dictionaries in {input_path} must contain either 'acc_id' and "
        "'sequence', or 'sequence', 'uniprot_id', 'pdbs', and 'pdb_type'."
    )


def _load_protein_records(input_path: Path) -> list[ProteinStructureRecord]:
    with open(input_path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Unsupported input type {type(payload)} from {input_path}; expected list.")

    records: list[ProteinStructureRecord] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(
                f"Unsupported payload element type {type(item)} at index {idx}; expected dict."
            )
        record = _record_from_input_item(item, input_path)
        if record is not None:
            records.append(record)

    # Two raw IDs that sanitize to one name would write the same output files. Fail
    # with the clashing IDs instead of renaming (names stay unchanged otherwise).
    raw_by_name: dict[str, set[str]] = {}
    for record in records:
        for raw_id in record.uniprot_ids or (record.acc_id,):
            raw_by_name.setdefault(_sanitize_output_id(raw_id), set()).add(raw_id)
    clashes = {name: sorted(raw) for name, raw in raw_by_name.items() if len(raw) > 1}
    if clashes:
        raise ValueError(f"Raw IDs map to the same output name: {clashes}")

    return sorted(records, key=lambda record: record.acc_id)


def _merge_records_for_sequence(records: list[ProteinStructureRecord]) -> ProteinStructureRecord:
    if not records:
        raise ValueError("Cannot merge an empty record list.")

    sequence = records[0].sequence
    uniprot_ids = tuple(
        sorted(
            {
                uniprot_id
                for record in records
                for uniprot_id in (record.uniprot_ids or (record.acc_id,))
            }
        )
    )
    source_pdbs = tuple(
        sorted({source_pdb for record in records for source_pdb in record.source_pdbs})
    )
    pdb_types = [record.pdb_type for record in records if record.pdb_type is not None]
    pdb_type = pdb_types[0] if pdb_types else None
    if uniprot_ids:
        acc_id = _sanitize_output_id(uniprot_ids[0])
    else:
        acc_id = records[0].acc_id

    return ProteinStructureRecord(
        acc_id=acc_id,
        sequence=sequence,
        uniprot_ids=uniprot_ids,
        source_pdbs=source_pdbs,
        pdb_type=pdb_type,
    )


def _deduplicate_by_sequence(
    records: list[ProteinStructureRecord],
) -> tuple[list[ProteinStructureRecord], dict[str, list[ProteinStructureRecord]]]:
    by_sequence: dict[str, list[ProteinStructureRecord]] = {}
    for record in records:
        by_sequence.setdefault(record.sequence, []).append(record)

    unique_records = [
        _merge_records_for_sequence(records_for_seq)
        for records_for_seq in by_sequence.values()
    ]
    return unique_records, by_sequence


def _record_output_ids(record: ProteinStructureRecord) -> tuple[str, ...]:
    output_ids = record.uniprot_ids or (record.acc_id,)
    return tuple(dict.fromkeys(_sanitize_output_id(output_id) for output_id in output_ids))


def _manifest_metadata(record: ProteinStructureRecord) -> dict:
    return {
        "uniprot_ids": list(record.uniprot_ids),
        "source_pdbs": list(record.source_pdbs),
        "pdb_type": record.pdb_type,
    }


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _existing_structure_status_row(
    record: ProteinStructureRecord,
    *,
    cif_path: Path | None,
    pdb_path: Path | None,
    status: str,
    model_id: str,
) -> dict:
    return {
        "acc_id": record.acc_id,
        "sequence_hash": sha256_short(record.sequence),
        "sequence_length": len(record.sequence),
        **_manifest_metadata(record),
        "cif_path": str(cif_path) if cif_path is not None else None,
        "pdb_path": str(pdb_path) if pdb_path is not None and pdb_path.exists() else None,
        "source_acc_id": None,
        "model_id": model_id,
        "status": status,
        "created_at": utc_timestamp(),
    }


def _output_paths(record: ProteinStructureRecord, worker_cfg: ESMFold2WorkerConfig) -> tuple[Path, Path]:
    return (
        Path(worker_cfg.cif_path) / f"ESMFold2-{record.acc_id}.cif",
        Path(worker_cfg.pdb_path) / f"ESMFold2-{record.acc_id}.pdb",
    )


def _load_optional_msa(
    msa_dir: str | None,
    acc_id: str,
    max_sequences: int,
) -> tuple[Any | None, str | None]:
    if msa_dir is None or str(msa_dir).lower() in {"", "none", "null"}:
        return None, None

    msa_path = Path(msa_dir) / f"{acc_id}.a3m"
    if not msa_path.exists():
        return None, None

    from esm.utils.msa import MSA

    msa = MSA.from_a3m(
        path=msa_path,
        remove_insertions=True,
        max_sequences=max_sequences,
    )
    return msa, str(msa_path)


def _make_structure_prediction_input(sequence: str, msa: Any | None = None) -> Any:
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    return StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence=sequence, msa=msa)]
    )


def _write_pdb_compat(result: Any, pdb_path: Path) -> str | None:
    try:
        protein_complex = result.complex.to_protein_complex().normalize_chain_ids_for_pdb()
        protein_complex.to_pdb(pdb_path)
    except Exception as exc:
        return str(exc)
    return None


def _metric_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value, dtype=np.float32).mean())
    return float(value)


def _fold_records_on_gpu(
    gpu_id: int,
    records: list[ProteinStructureRecord],
    worker_cfg: ESMFold2WorkerConfig,
) -> tuple[list[dict], list[dict]]:
    _ensure_transformers_vision_imports()
    from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model

    if not records:
        return [], []

    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{gpu_id}")

    torch_dtype = _torch_dtype_from_name(worker_cfg.torch_dtype)
    model = EsmFold2Model.from_pretrained(worker_cfg.model_id, device=device, dtype=torch_dtype).eval()
    builder = ESMFold2InputBuilder()

    manifest_rows: list[dict] = []
    failure_rows: list[dict] = []

    for record in progress(
        records,
        position=gpu_id,
        desc=f"GPU {gpu_id} ESMFold2 structures",
        leave=False,
    ):
        cif_path, pdb_path = _output_paths(record, worker_cfg)
        cif_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            msa, msa_path = _load_optional_msa(
                worker_cfg.msa_dir,
                record.acc_id,
                worker_cfg.msa_max_sequences,
            )
            structure_input = _make_structure_prediction_input(record.sequence, msa=msa)
            fold_context = (
                torch.autocast(device_type="cuda", dtype=torch_dtype)
                if torch_dtype in {torch.float16, torch.bfloat16}
                else nullcontext()
            )
            with fold_context:
                result = builder.fold(
                    model,
                    structure_input,
                    num_loops=worker_cfg.num_loops,
                    num_sampling_steps=worker_cfg.num_sampling_steps,
                    num_diffusion_samples=worker_cfg.num_diffusion_samples,
                    seed=worker_cfg.seed,
                    complex_id=record.acc_id,
                )
            if isinstance(result, list):
                result = result[0]

            mmcif = result.complex.to_mmcif()
            # A kill during write_text left a truncated CIF that later runs treated as done.
            tmp_cif = cif_path.with_name(cif_path.name + ".tmp")
            tmp_cif.write_text(mmcif)
            tmp_cif.replace(cif_path)

            pdb_error = None
            if worker_cfg.write_pdb_compat:
                pdb_error = _write_pdb_compat(result, pdb_path)

            manifest_row = (
                {
                    "acc_id": record.acc_id,
                    "sequence_hash": sha256_short(record.sequence),
                    "sequence_length": len(record.sequence),
                    **_manifest_metadata(record),
                    "cif_path": str(cif_path),
                    "pdb_path": str(pdb_path) if worker_cfg.write_pdb_compat and pdb_error is None else None,
                    "pdb_error": pdb_error,
                    "plddt_mean": _metric_float(result.plddt),
                    "ptm": _metric_float(result.ptm),
                    "iptm": _metric_float(result.iptm),
                    "msa_path": msa_path,
                    "model_id": worker_cfg.model_id,
                    "num_loops": worker_cfg.num_loops,
                    "num_sampling_steps": worker_cfg.num_sampling_steps,
                    "num_diffusion_samples": worker_cfg.num_diffusion_samples,
                    "seed": worker_cfg.seed,
                    "status": "ok",
                    "created_at": utc_timestamp(),
                }
            )
            manifest_rows.append(manifest_row)
            _append_row_line(worker_cfg.rows_dir, gpu_id, "manifest", manifest_row)
        except Exception as exc:
            failure_row = (
                {
                    "acc_id": record.acc_id,
                    "sequence_hash": sha256_short(record.sequence),
                    "sequence_length": len(record.sequence),
                    **_manifest_metadata(record),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                    "status": "error",
                    "created_at": utc_timestamp(),
                }
            )
            failure_rows.append(failure_row)
            _append_row_line(worker_cfg.rows_dir, gpu_id, "failure", failure_row)
        finally:
            torch.cuda.empty_cache()

    return manifest_rows, failure_rows


def _esmfold2_worker(
    gpu_id: int,
    records: list[ProteinStructureRecord],
    worker_cfg: ESMFold2WorkerConfig,
    return_dict,
) -> None:
    try:
        manifest_rows, failure_rows = _fold_records_on_gpu(gpu_id, records, worker_cfg)
        return_dict[gpu_id] = {
            "manifest_rows": manifest_rows,
            "failure_rows": failure_rows,
            "fatal_error": None,
        }
    except Exception as exc:
        return_dict[gpu_id] = {
            "manifest_rows": [],
            "failure_rows": [],
            "fatal_error": f"{exc}\n{traceback.format_exc()}",
        }


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_df = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    appended_df = pd.concat([existing_df, pd.DataFrame(rows)], ignore_index=True)
    # Keep the latest row per ID.
    appended_df = appended_df.drop_duplicates("acc_id", keep="last", ignore_index=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    appended_df.to_parquet(tmp_path, index=False, compression="zstd", compression_level=3)
    tmp_path.replace(path)


def _split_by_length(records: list[ProteinStructureRecord], num_parts: int) -> list[list[ProteinStructureRecord]]:
    """Give each part similar total work (length squared) by assigning the longest record to the lightest part."""
    parts: list[list[ProteinStructureRecord]] = [[] for _ in range(num_parts)]
    loads = [0] * num_parts
    for record in sorted(records, key=lambda record: len(record.sequence), reverse=True):
        idx = loads.index(min(loads))
        parts[idx].append(record)
        loads[idx] += len(record.sequence) ** 2
    return parts


def _available_cuda_memory_gib(gpu_id: int) -> float:
    free_bytes, _total_bytes = torch.cuda.mem_get_info(gpu_id)
    return free_bytes / (1024**3)


class ESMFold2StructureGenerator:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.gpu_ids = list(cfg.embeddings.gpu_ids)
        self.input_path = Path(cfg.embeddings.esmfold2_input_pkl_path)
        self.cif_path = Path(cfg.embeddings.esmfold2_cif_path)
        self.pdb_path = Path(cfg.embeddings.esmfold2_pdb_path)
        self.manifest_path = Path(cfg.embeddings.esmfold2_manifest_path)
        self.failures_path = Path(cfg.embeddings.esmfold2_failures_path)

        log_dir = Path(cfg.embeddings.log_dir)
        self.logger = component_logger(
            log_dir=log_dir,
            log_file_name=cfg.embeddings.esmfold2_log_file_name,
        ).get_logger()

        for path in (log_dir, self.cif_path, self.pdb_path, self.manifest_path.parent):
            path.mkdir(parents=True, exist_ok=True)

        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

    @property
    def rows_dir(self) -> Path:
        return self.manifest_path.parent / ".esmfold2_rows"

    def _worker_config(self, *, smoke: bool = False) -> ESMFold2WorkerConfig:
        return ESMFold2WorkerConfig(
            model_id=str(_config_get(self.cfg.embeddings, "esmfold2_model_id", "biohub/ESMFold2")),
            cif_path=str(self.cif_path),
            pdb_path=str(self.pdb_path),
            msa_dir=_config_get(self.cfg.embeddings, "esmfold2_msa_dir", None),
            msa_max_sequences=int(_config_get(self.cfg.embeddings, "esmfold2_msa_max_sequences", 1000)),
            num_loops=1 if smoke else int(_config_get(self.cfg.embeddings, "esmfold2_num_loops", 3)),
            num_sampling_steps=1
            if smoke
            else int(_config_get(self.cfg.embeddings, "esmfold2_num_sampling_steps", 50)),
            num_diffusion_samples=int(_config_get(self.cfg.embeddings, "esmfold2_num_diffusion_samples", 1)),
            seed=_config_get(self.cfg.embeddings, "esmfold2_seed", 0),
            write_pdb_compat=bool(_config_get(self.cfg.embeddings, "esmfold2_write_pdb_compat", True)),
            torch_dtype=str(_config_get(self.cfg.embeddings, "esmfold2_torch_dtype", "bfloat16")),
            rows_dir=None if smoke else str(self.rows_dir),
        )

    def generate(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        # Rows left by a killed run go into the manifest before the skip logic runs.
        _merge_incremental_rows(self.rows_dir, self.manifest_path, self.failures_path)
        records = _load_protein_records(self.input_path)
        self.logger.info(f"Loaded {len(records):,} protein records from {self.input_path}")

        unique_records, _records_by_sequence = _deduplicate_by_sequence(records)
        skip_known_structures = bool(
            _config_get(self.cfg.embeddings, "esmfold2_skip_known_structures", False)
        )
        model_id = str(_config_get(self.cfg.embeddings, "esmfold2_model_id", "biohub/ESMFold2"))

        work_records: list[ProteinStructureRecord] = []
        manifest_rows: list[dict] = []
        for record in unique_records:
            existing_cif = self.cif_path / f"ESMFold2-{record.acc_id}.cif"
            existing_pdb = self.pdb_path / f"ESMFold2-{record.acc_id}.pdb"
            if existing_cif.exists():
                manifest_rows.append(
                    _existing_structure_status_row(
                        record,
                        cif_path=existing_cif,
                        pdb_path=existing_pdb,
                        status="existing_output",
                        model_id=model_id,
                    )
                )
                continue
            if skip_known_structures and record.pdb_type in {"predicted", "experimental"}:
                manifest_rows.append(
                    _existing_structure_status_row(
                        record,
                        cif_path=None,
                        pdb_path=None,
                        status="skipped_existing_structure",
                        model_id=model_id,
                    )
                )
                continue
            work_records.append(record)

        max_length = _config_get(self.cfg.embeddings, "esmfold2_max_sequence_length", None)
        if max_length is not None:
            n_missing = len(work_records)
            work_records = [r for r in work_records if len(r.sequence) <= max_length]
            self.logger.info(
                f"Left {n_missing - len(work_records):,} sequences longer than {max_length} "
                "unfolded; they are not in the manifest or failures."
            )

        self.logger.info(
            f"Prepared {len(unique_records):,} unique sequences; "
            f"{len(work_records):,} structures missing from disk"
        )
        _append_rows(self.manifest_path, manifest_rows)

        if work_records:
            self._run_workers(work_records, self._worker_config())
        else:
            self.logger.info("No new ESMFold2 structures to generate.")

        copy_rows = self._copy_uniprot_outputs(unique_records)
        _append_rows(self.manifest_path, copy_rows)

    def run_smoke_test(self, min_free_gib: float = 20.0) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for ESMFold2 smoke test.")

        gpu_id = 0
        available_gib = _available_cuda_memory_gib(gpu_id)
        if available_gib < min_free_gib:
            raise RuntimeError(
                f"GPU {gpu_id} has {available_gib:.1f} GiB free; "
                f"ESMFold2 smoke test requires at least {min_free_gib:.1f} GiB free."
            )

        smoke_record = ProteinStructureRecord(
            acc_id="ESMFOLD2-SMOKE",
            sequence=SMOKE_TEST_SEQUENCE,
        )
        manifest_rows, failure_rows = _fold_records_on_gpu(
            gpu_id,
            [smoke_record],
            self._worker_config(smoke=True),
        )
        _append_rows(self.manifest_path, manifest_rows)
        _append_rows(self.failures_path, failure_rows)
        if failure_rows:
            raise RuntimeError(f"ESMFold2 smoke test failed: {failure_rows[0]['error']}")
        self.logger.info("ESMFold2 smoke test completed successfully.")

    def _run_workers(
        self,
        records: list[ProteinStructureRecord],
        worker_cfg: ESMFold2WorkerConfig,
    ) -> None:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available for ESMFold2 generation.")

        chunks = _split_by_length(records, num_gpus)
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()
        processes = []

        try:
            for gpu_id, chunk in enumerate(chunks):
                process = ctx.Process(
                    target=_esmfold2_worker,
                    args=(gpu_id, chunk, worker_cfg, return_dict),
                )
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            fatal_errors = [
                result["fatal_error"]
                for result in (return_dict.get(gpu_id, {}) for gpu_id in range(num_gpus))
                if result.get("fatal_error")
            ]
            # The rows were written as each structure finished; merge them even after a crash.
            n_ok, n_failed = _merge_incremental_rows(
                self.rows_dir, self.manifest_path, self.failures_path
            )
            if fatal_errors:
                raise RuntimeError("\n\n".join(fatal_errors))

            self.logger.info(
                f"Generated {n_ok:,} ESMFold2 structures; recorded {n_failed:,} failures."
            )
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()

    def _copy_uniprot_outputs(
        self,
        records: list[ProteinStructureRecord],
    ) -> list[dict]:
        copy_rows: list[dict] = []
        model_id = str(_config_get(self.cfg.embeddings, "esmfold2_model_id", "biohub/ESMFold2"))

        for record in records:
            output_ids = _record_output_ids(record)
            if len(output_ids) <= 1:
                continue

            source_cif = self.cif_path / f"ESMFold2-{record.acc_id}.cif"
            source_pdb = self.pdb_path / f"ESMFold2-{record.acc_id}.pdb"
            if not source_cif.exists():
                continue

            for output_id in output_ids:
                if output_id == record.acc_id:
                    continue

                target_cif = self.cif_path / f"ESMFold2-{output_id}.cif"
                target_pdb = self.pdb_path / f"ESMFold2-{output_id}.pdb"
                if not target_cif.exists():
                    shutil.copyfile(source_cif, target_cif)
                if source_pdb.exists() and not target_pdb.exists():
                    shutil.copyfile(source_pdb, target_pdb)

                copy_rows.append(
                    {
                        "acc_id": output_id,
                        "sequence_hash": sha256_short(record.sequence),
                        "sequence_length": len(record.sequence),
                        **_manifest_metadata(record),
                        "cif_path": str(target_cif),
                        "pdb_path": str(target_pdb) if target_pdb.exists() else None,
                        "source_acc_id": record.acc_id,
                        "model_id": model_id,
                        "status": "copied_from_sequence_canonical",
                        "created_at": utc_timestamp(),
                    }
                )

        if copy_rows:
            self.logger.info(f"Copied {len(copy_rows):,} ESMFold2 UniProt alias outputs.")
        return copy_rows
