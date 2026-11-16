from __future__ import annotations

import argparse
import copy
import signal
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_baseline_mlp_pipeline as pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = ["classification", "retrieval"]
ENDPOINT_DATASET_DIRS = {
    "classification": "enzyme_classification_dataset",
    "retrieval": "enzyme_retrieval_dataset",
}
ENDPOINT_DEFAULT_MODALITIES = {
    "classification": ["esmc", "esm3"],
    "retrieval": ["drfp"],
}
ENDPOINT_HPO_SPLITS = {
    "classification": "enzyme_structure_splits",
    "retrieval": "reaction_drfp_tanimoto_splits",
}
SPLITS = [
    # The random splitter writes random_splits_grouped_<col>; each endpoint finds its own folder
    # (classification: sequence, retrieval: rxn_key) and the other name is skipped as missing.
    "random_splits_grouped_sequence",
    "random_splits_grouped_rxn_key",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "enzyme_structure_splits",
    "reaction_drfp_tanimoto_splits",
]
MODALITIES = ["drfp", "esmc", "esm3"]
MODALITY_KEY_COLUMNS = {
    "drfp": "rxn_key",
    "esmc": "sequence",
    "esm3": "pdb_ids",
}
MODALITY_INPUT_OVERRIDES = {
    "drfp": "embeddings.drfp_embeddings_input_pkl_path",
    "esmc": "embeddings.esmc_embeddings_input_pkl_path",
    "esm3": "embeddings.esm3_embeddings_input_pkl_path",
}
GPU_EMBEDDERS = {"esmc", "esm3"}
EMBEDDING_DATASET_CONFIG = "enzyme_retrieval"


def endpoint_modalities(args: argparse.Namespace, endpoint: str) -> list[str]:
    common = (pipeline.hydra_profile(args).get("common") or {}).get("data") or {}
    by_endpoint = common.get("endpoint_modalities") or {}
    if endpoint in by_endpoint:
        return [str(item) for item in by_endpoint[endpoint]]
    if len(getattr(args, "endpoints", []) or []) == 1 and getattr(args, "modalities", None):
        return [str(item) for item in args.modalities]
    return list(ENDPOINT_DEFAULT_MODALITIES[endpoint])


def default_data_root() -> Path:
    return REPO_ROOT / "outputs/baselines/MLP/data/classification"


def pipeline_arg_defaults() -> dict[str, Any]:
    defaults = pipeline.pipeline_arg_defaults()
    defaults.update(
        {
            "output_root": REPO_ROOT / "outputs/baselines/MLP",
            "data_root": default_data_root(),
            "endpoints": list(ENDPOINTS),
            "splits": list(SPLITS),
            "modalities": list(MODALITIES),
            "hpo_split": "auto",
            "run_label": "classification_mlp",
            "study_prefix": "baseline_classification",
            "run_tags": ["baseline_classification"],
        }
    )
    return defaults


def cache_roots(embeddings_root: Path) -> dict[str, Path]:
    return {
        "drfp": embeddings_root / "drfp/radius3_min0_len2048_rings0_h0/v1",
        "esmc": embeddings_root / "esmc/ESMC-6B/hf_v1",
        "esm3": embeddings_root / "esm3/open_small_structure/v1",
    }


def endpoint_root(args: argparse.Namespace, endpoint: str) -> Path:
    data_root = Path(args.data_root)
    direct = data_root / ENDPOINT_DATASET_DIRS[endpoint]
    if direct.exists():
        return direct
    if data_root.name == ENDPOINT_DATASET_DIRS[endpoint]:
        return data_root
    return direct


def has_tvt(path: Path) -> bool:
    return all((path / f"{split}.parquet").exists() for split in ("train", "val", "test"))


def resolve_split_dir(
    data_root: Path, endpoint: str, split: str, threshold: Optional[str] = None
) -> Path:
    args = pipeline._ACTIVE_ARGS or argparse.Namespace(data_root=data_root)
    root = endpoint_root(args, endpoint)
    split_root = root / split
    if has_tvt(split_root):
        return split_root
    if threshold is not None:
        threshold_name = str(threshold)
        candidates = [split_root / threshold_name]
        if not threshold_name.startswith("threshold_") and not threshold_name.startswith("L"):
            candidates.append(split_root / f"threshold_{threshold_name}")
        for candidate in candidates:
            if has_tvt(candidate):
                return candidate
        raise FileNotFoundError(
            f"Missing train/val/test under {split_root} for threshold {threshold}"
        )
    if not split_root.exists():
        raise FileNotFoundError(f"No split directory found: {split_root}")
    candidates = [path for path in sorted(split_root.iterdir()) if path.is_dir() and has_tvt(path)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Multiple split folders under {split_root}; pass an explicit threshold.")
    raise FileNotFoundError(f"No train/val/test parquets found under {split_root}")


def split_threshold(split_dir: Path) -> Optional[str]:
    return (
        split_dir.name
        if split_dir.name.startswith("threshold_") or split_dir.name.startswith("L")
        else None
    )


def discover_split_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = []
    for endpoint in args.endpoints:
        for split in args.splits:
            try:
                split_dir = resolve_split_dir(args.data_root, endpoint, split)
            except FileNotFoundError:
                continue
            specs.append(
                {
                    "endpoint": endpoint,
                    "split": split,
                    "threshold": split_threshold(split_dir),
                    "split_dir": split_dir,
                }
            )
    return specs


def discover_final_jobs(args: argparse.Namespace, run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for endpoint in args.endpoints:
        for split in args.splits:
            try:
                split_dir = resolve_split_dir(args.data_root, endpoint, split)
            except FileNotFoundError:
                continue
            for seed in args.seeds:
                rows.append(
                    {
                        "endpoint": endpoint,
                        "split": split,
                        "threshold": split_threshold(split_dir),
                        "seed": int(seed),
                        "output_dir": pipeline.final_dir(
                            args.output_root, endpoint, split, split_threshold(split_dir), seed
                        ),
                    }
                )
    return rows


def hpo_split_for_endpoint(args: argparse.Namespace, endpoint: str) -> str:
    split = str(getattr(args, "hpo_split", "auto"))
    return ENDPOINT_HPO_SPLITS[endpoint] if split == "auto" else split


def hpo_study_suffix_for_endpoint(args: argparse.Namespace, endpoint: str) -> str:
    split = hpo_split_for_endpoint(args, endpoint)
    if split == "enzyme_structure_splits":
        return "structure"
    if split == "reaction_drfp_tanimoto_splits":
        return "reaction_drfp"
    return split


def study_name_for_endpoint(args: argparse.Namespace, endpoint: str) -> str:
    # The data fingerprint keeps a rebuild from reusing old studies.
    fingerprint = pipeline.data_fingerprint(args, endpoint, hpo_split_for_endpoint(args, endpoint))
    return f"{getattr(args, 'study_prefix', 'baseline_classification')}_{endpoint}_{hpo_study_suffix_for_endpoint(args, endpoint)}_{fingerprint}"


def build_base_overrides(
    args: argparse.Namespace, endpoint: str, output_dir: Path, stage: str
) -> list[str]:
    scoped_args = copy.copy(args)
    scoped_args.modalities = endpoint_modalities(args, endpoint)
    overrides = pipeline.build_base_overrides(scoped_args, endpoint, output_dir, stage)
    data_roots = {
        key: str((Path(args.data_root) / dataset_dir).resolve())
        for key, dataset_dir in ENDPOINT_DATASET_DIRS.items()
    }
    extra = [
        f"++data.data_roots.classification={data_roots['classification']}",
        f"++data.data_roots.retrieval={data_roots['retrieval']}",
        "++data.label_col=ec_number",
    ]
    return [*overrides, *extra]


def build_hpo_jobs(args: argparse.Namespace, run_root: Path) -> list[pipeline.PipelineJob]:
    optuna_dir = run_root / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[pipeline.PipelineJob] = []
    for endpoint in args.endpoints:
        hpo_split_name = hpo_split_for_endpoint(args, endpoint)
        hpo_split_dir = resolve_split_dir(args.data_root, endpoint, hpo_split_name)
        hpo_threshold = split_threshold(hpo_split_dir)
        study_name = study_name_for_endpoint(args, endpoint)
        storage = pipeline.optuna_storage_url(run_root, study_name)
        for worker_index in range(args.hpo_workers_per_endpoint):
            output_dir = run_root / "hpo" / endpoint / f"worker_{worker_index}"
            trial_output_dir = f"{output_dir}/trials/trial_{pipeline.OPTUNA_TRIAL_NUMBER_TOKEN}"
            overrides = build_base_overrides(args, endpoint, Path(trial_output_dir), "hpo")
            overrides.extend(
                [
                    *pipeline.profile_hpo_overrides(args),
                    f"data.split={hpo_split_name}",
                    f"data.threshold={hpo_threshold}" if hpo_threshold else "data.threshold=null",
                    f"trainer.max_epochs={args.max_epochs}",
                    f"hydra.sweep.dir={output_dir}/multirun",  # one sweep dir per worker
                    f"hydra.sweeper.study_name={study_name}",
                    f"hydra.sweeper.storage={storage}",
                    f"hydra.sweeper.n_trials={args.n_trials}",
                    f"+wandb_name=hpo_{endpoint}_worker_{worker_index}",
                    *pipeline.profile_raw_overrides(args, "hpo"),
                    *pipeline.hpo_sampler_seed_override(args, worker_index),
                ]
            )
            jobs.append(
                pipeline.PipelineJob(
                    job_id=f"hpo_{endpoint}_worker_{worker_index}_{study_name.rsplit('_', 1)[-1]}",
                    stage="hpo",
                    endpoint=endpoint,
                    split=hpo_split_name,
                    threshold=hpo_threshold,
                    seed=None,
                    worker_index=worker_index,
                    command=[sys.executable, str(pipeline.TRAIN_SCRIPT), "-m", *overrides],
                    output_dir=output_dir,
                )
            )
    return jobs


def materialization_specs_for_hpo(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = []
    for endpoint in args.endpoints:
        split_name = hpo_split_for_endpoint(args, endpoint)
        split_dir = resolve_split_dir(args.data_root, endpoint, split_name)
        specs.append(
            {"endpoint": endpoint, "split": split_name, "threshold": split_threshold(split_dir)}
        )
    return specs


def materialization_specs_for_final(
    args: argparse.Namespace, run_root: Path
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, Optional[str]]] = set()
    specs = []
    for spec in discover_final_jobs(args, run_root):
        key = (spec["endpoint"], spec["split"], spec["threshold"])
        if key in seen:
            continue
        seen.add(key)
        specs.append({"endpoint": key[0], "split": key[1], "threshold": key[2]})
    return specs


def materialization_specs_for_final_endpoint(
    args: argparse.Namespace, run_root: Path, endpoint: str
) -> list[dict[str, Any]]:
    return [
        spec
        for spec in materialization_specs_for_final(args, run_root)
        if spec["endpoint"] == endpoint
    ]


def materialization_totals(status: dict[str, Any]) -> tuple[Any, Any]:
    """Return (rows, estimated_gb) from a cache status or from a built cache's info."""
    # ensure_materialized_cache returns estimated_total_gb and splits[...]['rows'].
    rows = status.get("rows")
    if rows is None and status.get("splits"):
        rows = sum(info["rows"] for info in status["splits"].values())
    estimated_gb = status.get("estimated_gb", status.get("estimated_total_gb"))
    return ("" if rows is None else rows), ("" if estimated_gb is None else estimated_gb)


def materialization_status_row(
    args: argparse.Namespace,
    spec: dict[str, Any],
    *,
    dry_run: bool,
    prior_row: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    import time

    started_at = time.monotonic()
    # Each row stores the data fingerprint; the status shortcut applies only when it still
    # matches. Other rows take the checked path.
    data_fp = pipeline.data_fingerprint(
        args, spec["endpoint"], spec["split"], spec.get("threshold")
    )
    if (
        not dry_run
        and not args.refresh_materialized_features
        and prior_row is not None
        and str(prior_row.get("data_fingerprint") or "") == data_fp
        and prior_row.get("status") in {"hit", "built", "fast_hit"}
        and str(prior_row.get("feature_dtype") or "") == str(args.materialized_feature_dtype)
        and str(prior_row.get("validation") or "") == str(args.materialized_validation)
    ):
        cache_dir = Path(str(prior_row.get("cache_dir") or ""))
        if (
            cache_dir
            and (cache_dir / ".complete").exists()
            and (cache_dir / "metadata.json").exists()
        ):
            elapsed_sec = time.monotonic() - started_at
            return {
                **spec,
                "stage": "materialize",
                "status": "hit",
                "reason": "status_fast_hit",
                "elapsed_sec": f"{elapsed_sec:.3f}",
                "rows": prior_row.get("rows", ""),
                "estimated_gb": prior_row.get("estimated_gb", ""),
                "cache_dir": str(cache_dir),
                "feature_dtype": args.materialized_feature_dtype,
                "validation": args.materialized_validation,
                "data_fingerprint": data_fp,
            }

    from src.data.baseline_classification_datamodule import BaselineClassificationDataModule

    dm = BaselineClassificationDataModule(
        data_root=str(args.data_root),
        data_roots={
            "classification": str(endpoint_root(args, "classification")),
            "retrieval": str(endpoint_root(args, "retrieval")),
        },
        endpoint=spec["endpoint"],
        split=spec["split"],
        threshold=spec.get("threshold"),
        modalities=endpoint_modalities(args, spec["endpoint"]),
        cache_roots=cache_roots(args.embeddings_root),
        batch_size=4096,
        parquet_batch_size=4096,
        num_workers=0,
        pin_memory=False,
        use_materialized_features=True,
        materialize_if_missing=not dry_run,
        refresh_materialized_features=args.refresh_materialized_features and not dry_run,
        materialized_cache_root=str(args.materialized_cache_root),
        materialized_feature_dtype=args.materialized_feature_dtype,
        materialized_block_rows=args.materialized_block_rows,
        materialized_validation=args.materialized_validation,
        materialization_max_gb=10_000.0,
        pooled_list_cache_size=100000,
        seed=666,
    )
    try:
        if (
            not dry_run
            and not args.refresh_materialized_features
            and prior_row is not None
            and prior_row.get("status") in {"hit", "built", "fast_hit"}
        ):
            status = dm.materialized_cache_status()
            if status.get("status") == "hit":
                elapsed_sec = time.monotonic() - started_at
                return {
                    **spec,
                    "stage": "materialize",
                    "status": "hit",
                    "reason": "fast_hit",
                    "elapsed_sec": f"{elapsed_sec:.3f}",
                    "rows": status.get("rows") or prior_row.get("rows", ""),
                    "estimated_gb": status.get("estimated_gb")
                    or prior_row.get("estimated_gb", ""),
                    "cache_dir": status["cache_dir"],
                    "feature_dtype": args.materialized_feature_dtype,
                    "validation": args.materialized_validation,
                    "data_fingerprint": data_fp,
                }
        status = (
            dm.materialized_cache_status()
            if dry_run
            else dm.ensure_materialized_cache(refresh=args.refresh_materialized_features)
        )
        elapsed_sec = time.monotonic() - started_at
        cache_status = status.get("cache_status") or status.get("status")
        rows, estimated_gb = materialization_totals(status)
        return {
            **spec,
            "stage": "materialize",
            "status": cache_status,
            "reason": status.get("reason", ""),
            "elapsed_sec": f"{elapsed_sec:.3f}",
            "rows": rows,
            "estimated_gb": estimated_gb,
            "cache_dir": status.get("cache_dir", ""),
            "feature_dtype": args.materialized_feature_dtype,
            "validation": args.materialized_validation,
            "data_fingerprint": data_fp,
        }
    except Exception as exc:
        elapsed_sec = time.monotonic() - started_at
        return {
            **spec,
            "stage": "materialize",
            "status": "failed",
            "reason": str(exc),
            "elapsed_sec": f"{elapsed_sec:.3f}",
            "rows": "",
            "estimated_gb": "",
            "cache_dir": "",
            "feature_dtype": args.materialized_feature_dtype,
            "validation": args.materialized_validation,
        }


def required_columns_for_modalities(modalities: Iterable[str]) -> set[str]:
    columns = {"ec_number"}
    columns.update(MODALITY_KEY_COLUMNS[modality] for modality in modalities)
    return columns


def preflight(args: argparse.Namespace, run_root: Path) -> None:
    issues: list[dict[str, Any]] = []
    package_rows = []
    for package in ("optuna", "wandb", "sqlalchemy", "pyarrow", "pytest"):
        try:
            __import__(package)
            package_rows.append({"check": f"package:{package}", "status": "ok", "detail": ""})
        except ImportError as exc:
            issues.append({"kind": "missing_package", "package": package, "error": str(exc)})
            package_rows.append(
                {"check": f"package:{package}", "status": "missing", "detail": str(exc)}
            )

    if args.skip_embeddings:
        roots = cache_roots(args.embeddings_root)
        for endpoint in args.endpoints:
            for name in endpoint_modalities(args, endpoint):
                if not (roots[name] / "meta.json").exists():
                    issues.append(
                        {
                            "kind": "missing_cache",
                            "modality": name,
                            "path": str(roots[name] / "meta.json"),
                        }
                    )
                if not pipeline.index_files(roots[name]):  # parts count as an index
                    issues.append(
                        {
                            "kind": "missing_cache",
                            "modality": name,
                            "path": str(roots[name] / "index.parquet"),
                        }
                    )

    schema_tasks: list[tuple[str, str, Path, str]] = []
    for spec in discover_split_specs(args):
        for tvt in ("train", "val", "test"):
            schema_tasks.append((spec["endpoint"], spec["split"], Path(spec["split_dir"]), tvt))

    for endpoint, split, split_dir, tvt in pipeline.progress_iter(
        args,
        schema_tasks,
        desc="Preflight schemas",
        total=len(schema_tasks),
        unit="file",
    ):
        path = split_dir / f"{tvt}.parquet"
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        missing = sorted(
            required_columns_for_modalities(endpoint_modalities(args, endpoint)) - schema
        )
        if missing:
            issues.append(
                {
                    "kind": "missing_columns",
                    "endpoint": endpoint,
                    "split": split,
                    "threshold": split_threshold(split_dir),
                    "tvt": tvt,
                    "path": str(path),
                    "missing": ",".join(missing),
                }
            )

    report_path = run_root / "preflight_report.csv"
    pipeline.write_dict_rows(report_path, issues)
    pipeline.render_summary_table(
        "Preflight summary",
        [
            *package_rows,
            {
                "check": "endpoints",
                "status": len(args.endpoints),
                "detail": ",".join(args.endpoints),
            },
            {"check": "split_specs", "status": len(discover_split_specs(args)), "detail": ""},
            {"check": "schema_files", "status": len(schema_tasks), "detail": ""},
            {"check": "issues", "status": len(issues), "detail": str(report_path)},
        ],
        [("check", "check"), ("status", "status"), ("detail", "detail")],
    )
    if issues:
        raise RuntimeError(f"Preflight failed with {len(issues)} issue(s). See {report_path}")
    pipeline.log_step("Preflight passed")


# This script supplies its hooks here; pipeline.main(TASK) runs the shared orchestrator with them.
TASK = pipeline.PipelineTask(
    modalities=MODALITIES,
    modality_key_columns=MODALITY_KEY_COLUMNS,
    modality_input_overrides=MODALITY_INPUT_OVERRIDES,
    gpu_embedders=GPU_EMBEDDERS,
    embedding_dataset_config=EMBEDDING_DATASET_CONFIG,
    pipeline_arg_defaults=pipeline_arg_defaults,
    cache_roots=cache_roots,
    resolve_split_dir=resolve_split_dir,
    split_threshold=split_threshold,
    discover_split_specs=discover_split_specs,
    discover_final_jobs=discover_final_jobs,
    study_name_for_endpoint=study_name_for_endpoint,
    build_base_overrides=build_base_overrides,
    build_hpo_jobs=build_hpo_jobs,
    materialization_specs_for_hpo=materialization_specs_for_hpo,
    materialization_specs_for_final=materialization_specs_for_final,
    materialization_specs_for_final_endpoint=materialization_specs_for_final_endpoint,
    materialization_status_row=materialization_status_row,
    required_columns_for_modalities=required_columns_for_modalities,
    preflight=preflight,
    hpo_split_for_endpoint=hpo_split_for_endpoint,
)


def main() -> None:
    pipeline.main(TASK)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, pipeline.handle_termination_signal)
    signal.signal(signal.SIGTERM, pipeline.handle_termination_signal)
    try:
        main()
    except KeyboardInterrupt:
        pipeline.cleanup_interrupted_run(
            "classification orchestrator interrupted by KeyboardInterrupt"
        )
        raise
