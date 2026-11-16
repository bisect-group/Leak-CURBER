from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.components.embedders.store import ShardedEmbeddingStore
from src.data.components.embedders.types import CacheMetadata
from tests.helpers.builders import REPO_ROOT, load_script, pipeline_args, write_cache, write_tvt

pipeline = load_script("run_baseline_mlp_pipeline")


def _write_full_split_tree(data_root: Path) -> None:
    threshold_splits = {
        "substrate_splits",
        "enzyme_sequence_splits",
        "enzyme_structure_splits",
        "conformer_cosine_splits",
    }
    for endpoint in pipeline.ENDPOINTS:
        for split in pipeline.SPLITS:
            root = data_root / f"baselines_{endpoint}" / split
            split_dir = root / "threshold_0.1" if split in threshold_splits else root
            write_tvt(split_dir)


def _write_all_caches(embeddings_root: Path) -> None:
    for root in pipeline.cache_roots(embeddings_root).values():
        write_cache(root)


def test_preflight_and_final_job_planning(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    run_root = pipeline.run_root_for(args)
    _write_full_split_tree(args.data_root)
    _write_all_caches(args.embeddings_root)

    pipeline.preflight(args, run_root)
    final_jobs = pipeline.build_final_jobs(
        args,
        run_root,
        params_by_endpoint={endpoint: {} for endpoint in args.endpoints},
    )

    assert len(final_jobs) == 6 * 7 * 3
    assert any(job.threshold == "threshold_0.1" for job in final_jobs)
    assert all("CUDA_VISIBLE_DEVICES" not in " ".join(job.command) for job in final_jobs)


def test_hpo_job_planning_creates_optuna_storage_dir(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.endpoints = ["ki"]
    args.splits = ["enzyme_structure_splits"]
    run_root = pipeline.run_root_for(args)
    write_tvt(args.data_root / "baselines_ki" / "enzyme_structure_splits")

    jobs = pipeline.build_hpo_jobs(args, run_root)

    assert (run_root / "optuna").is_dir()
    assert len(jobs) == 1
    command = " ".join(jobs[0].command)
    assert (
        f"hydra.sweeper.storage=sqlite:///{run_root / 'optuna' / (pipeline.study_name_for_endpoint(args, 'ki') + '.db')}"
        in command
    )
    assert (
        f"paths.output_dir={run_root / 'hpo' / 'ki' / 'worker_0'}/trials/trial_{pipeline.OPTUNA_TRIAL_NUMBER_TOKEN}"
        in command
    )
    assert "${hydra.job.num}" not in command
    assert command.count("hydra.sweeper.storage=") == 1
    assert "data.materialized_feature_dtype=float16" in command
    assert f"data.materialized_cache_root={args.materialized_cache_root}" in command


def test_run_profile_defaults_override_cli_defaults() -> None:
    profile_path = REPO_ROOT / "configs" / "pipeline" / "baseline_mlp_optuna100.yaml"

    defaults = pipeline.pipeline_arg_defaults()
    defaults.update(pipeline.load_run_config_defaults(profile_path))

    assert defaults["pipeline_id"] == "mlp_optuna100"
    assert defaults["n_trials"] == 100
    assert defaults["skip_embeddings"] is True
    assert defaults["skip_hpo"] is False
    assert defaults["final_params_source"] == "hpo_csv"
    assert (
        defaults["final_best_params_path"]
        == "outputs/baselines/MLP/_runs/mlp_optuna100/hpo_best_params.csv"
    )
    assert defaults["final_reuse_existing_successes"] is True
    assert defaults["materialized_feature_dtype"] == "float16"
    assert defaults["materialized_validation"] == "metadata"
    assert defaults["hpo_split"] == "enzyme_structure_splits"
    assert defaults["hpo_sampler_seed_base"] == 42
    assert defaults["hpo_sampler_seed_strategy"] == "worker_offset"
    assert defaults["hpo_checkpoints"] is False
    assert defaults["final_checkpoints"] is True
    assert defaults["run_label"] == "mlp"
    assert defaults["study_prefix"] == "baseline_mlp"
    assert defaults["hydra_profile"]["hpo"]["search_space"]["optimizer.lr"].startswith("choice(")
    assert defaults["hydra_profile"]["final"]["data"]["pin_memory"] is False
    assert defaults["hydra_profile"]["final"]["data"]["num_workers"] == 0
    assert defaults["storage_profile"]["checkpoints"]["prune_hpo_checkpoints"] is True


def test_ridge_profile_uses_fixed_final_params() -> None:
    profile_path = REPO_ROOT / "configs" / "pipeline" / "baseline_sklearn_ridge_optuna100.yaml"

    defaults = pipeline.pipeline_arg_defaults()
    defaults.update(pipeline.load_run_config_defaults(profile_path))

    assert defaults["pipeline_id"] == "ridge_optuna100"
    assert defaults["skip_hpo"] is True
    assert defaults["final_params_source"] == "fixed"
    assert defaults["final_best_params_path"] is None


def test_xgboost_profile_uses_gpu_chunked_prediction_and_resilient_hpo() -> None:
    profile_path = REPO_ROOT / "configs" / "pipeline" / "baseline_xgboost_optuna100.yaml"

    defaults = pipeline.pipeline_arg_defaults()
    defaults.update(pipeline.load_run_config_defaults(profile_path))

    assert defaults["pipeline_id"] == "xgboost_gpu_optuna100"
    assert defaults["endpoints"] == ["ic50"]
    assert defaults["gpus"] == [0]
    assert defaults["max_slots_per_gpu"] == 1
    assert defaults["hpo_workers_per_endpoint"] == 1
    assert defaults["final_reuse_existing_successes"] is True
    hydra = defaults["hydra_profile"]
    assert hydra["common"]["model"]["estimator"]["tree_method"] == "hist"
    assert hydra["common"]["model"]["estimator"]["device"] == "cuda"
    assert hydra["common"]["model"]["predict_batch_rows"] == 32768
    assert hydra["hpo"]["search_space"]["model.estimator.n_estimators"].startswith("choice(")
    assert hydra["hpo"]["search_space"]["model.estimator.max_bin"].startswith("choice(")
    assert "hydra.sweeper.sampler.constant_liar=true" in hydra["raw_overrides"]["hpo"]
    assert "hydra.sweeper.max_consecutive_failed_trials=5" in hydra["raw_overrides"]["hpo"]
    assert "hydra.sweeper.stale_trial_timeout_sec=43200" in hydra["raw_overrides"]["hpo"]
    assert not any(
        item.startswith("hydra.sweeper.sampler.seed=") for item in hydra["raw_overrides"]["hpo"]
    )


def test_xgboost_cpu_profile_hides_gpus_and_bounds_threads() -> None:
    profile_path = REPO_ROOT / "configs" / "pipeline" / "baseline_xgboost_cpu_optuna100.yaml"

    defaults = pipeline.pipeline_arg_defaults()
    defaults.update(pipeline.load_run_config_defaults(profile_path))

    assert defaults["pipeline_id"] == "xgboost_cpu_optuna100"
    assert defaults["endpoints"] == ["ec50", "kcat", "kd", "ki", "km"]
    assert defaults["gpus"] == ["-1"]
    estimator = defaults["hydra_profile"]["common"]["model"]["estimator"]
    assert estimator["device"] == "cpu"
    assert defaults["max_slots_per_gpu"] * estimator["n_jobs"] <= 48


def test_hpo_command_omits_model_checkpoint_when_disabled(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.endpoints = ["ki"]
    args.splits = ["enzyme_structure_splits"]
    args.hpo_checkpoints = False
    args.final_checkpoints = True
    args.hydra_profile = {
        "config_groups": {
            "data": "baseline_regression",
            "model": "baseline_regression",
            "callbacks": ["default", "model_checkpoint", "early_stopping"],
        }
    }
    run_root = pipeline.run_root_for(args)
    write_tvt(args.data_root / "baselines_ki" / "enzyme_structure_splits")

    hpo_command = " ".join(pipeline.build_hpo_jobs(args, run_root)[0].command)
    final_command = " ".join(
        pipeline.build_final_jobs(args, run_root, params_by_endpoint={"ki": {}})[0].command
    )

    assert "callbacks=[default,early_stopping]" in hpo_command
    assert "callbacks=[default,model_checkpoint,early_stopping]" in final_command
    assert "++trainer.enable_checkpointing=false" in hpo_command
    assert "enable_checkpointing" not in final_command

    args.hpo_checkpoints = True
    hpo_command = " ".join(pipeline.build_hpo_jobs(args, run_root)[0].command)
    assert "enable_checkpointing" not in hpo_command


def test_hpo_workers_get_deterministic_distinct_sampler_seeds(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.endpoints = ["ki"]
    args.hpo_workers_per_endpoint = 2
    args.hpo_sampler_seed_base = 42
    args.hpo_sampler_seed_strategy = "worker_offset"
    run_root = pipeline.run_root_for(args)
    write_tvt(args.data_root / "baselines_ki" / "enzyme_structure_splits")

    jobs = pipeline.build_hpo_jobs(args, run_root)

    assert jobs[0].command[-1] == "hydra.sweeper.sampler.seed=42"
    assert jobs[1].command[-1] == "hydra.sweeper.sampler.seed=43"


def test_computed_hpo_sampler_seed_overrides_raw_seed(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.endpoints = ["ki"]
    args.hydra_profile = {
        "raw_overrides": {"common": [], "hpo": ["hydra.sweeper.sampler.seed=999"], "final": []}
    }
    run_root = pipeline.run_root_for(args)
    write_tvt(args.data_root / "baselines_ki" / "enzyme_structure_splits")

    job = pipeline.build_hpo_jobs(args, run_root)[0]

    raw_index = job.command.index("hydra.sweeper.sampler.seed=999")
    computed_index = job.command.index("hydra.sweeper.sampler.seed=42")
    assert raw_index < computed_index


def test_profile_hydra_overrides_replace_search_space(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.hydra_profile = {
        "config_groups": {
            "data": "baseline_regression",
            "model": "baseline_regression",
        },
        "common": {
            "model": {
                "_target_": "src.models.baseline_regression_module.BaselineRegressionEstimatorModule",
                "backend": "sklearn",
                "estimator": {"_target_": "sklearn.linear_model.Ridge", "alpha": 1.0},
            }
        },
        "hpo": {
            "hparams_search": "baseline_regression_optuna4",
            "direction": "maximize",
            "optimized_metric": "val/r2",
            "search_space": {"model.estimator.alpha": "tag(log, interval(1e-4, 1e4))"},
        },
        "raw_overrides": {"common": [], "hpo": [], "final": []},
    }
    args.run_label = "ridge"
    args.study_prefix = "baseline_ridge"
    args.run_tags = ["baseline_regression", "sklearn", "ridge"]
    args.endpoints = ["ki"]
    run_root = pipeline.run_root_for(args)
    write_tvt(args.data_root / "baselines_ki" / "enzyme_structure_splits")

    jobs = pipeline.build_hpo_jobs(args, run_root)
    command = " ".join(jobs[0].command)

    assert "model=baseline_regression" in command
    assert (
        "++model._target_=src.models.baseline_regression_module.BaselineRegressionEstimatorModule"
        in command
    )
    assert "++model.estimator._target_=sklearn.linear_model.Ridge" in command
    assert "++model.estimator.alpha=1.0" in command
    assert "~hydra.sweeper.params" in command
    assert (
        '++hydra.sweeper.params.model.estimator.alpha="tag(log, interval(1e-4, 1e4))"' in command
    )
    assert (
        f"hydra.sweeper.storage=sqlite:///{run_root / 'optuna' / (pipeline.study_name_for_endpoint(args, 'ki') + '.db')}"
        in command
    )
    assert "baseline_regression,sklearn,ridge,hpo,ki" in command


def test_xgboost_hpo_command_disables_hpo_predictions(tmp_path: Path) -> None:
    defaults = pipeline.pipeline_arg_defaults()
    defaults.update(
        pipeline.load_run_config_defaults(
            REPO_ROOT / "configs" / "pipeline" / "baseline_xgboost_optuna100.yaml"
        )
    )
    args = argparse.Namespace(**defaults)
    args.gpus = ["0"]
    args.output_root = tmp_path / "out"
    args.data_root = tmp_path / "data"
    args.embeddings_root = tmp_path / "embeddings"
    args.materialized_cache_root = tmp_path / "out" / "_cache" / "materialized_features"
    args.endpoints = ["ki"]
    run_root = pipeline.run_root_for(args)
    write_tvt(args.data_root / "baselines_ki" / "enzyme_structure_splits")

    jobs = pipeline.build_hpo_jobs(args, run_root)
    command = " ".join(jobs[0].command)

    assert "++model.predict_batch_rows=32768" in command
    assert "++model.estimator.device=cuda" in command
    assert "++model.save_predictions=false" in command
    assert "hydra.sweeper.max_consecutive_failed_trials=5" in command


def test_resolved_run_config_is_written(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.pipeline_id = "resolved"
    args.output_root = tmp_path / "out"
    args.materialized_cache_root = tmp_path / "out" / "_cache" / "materialized_features"
    args.gpus = ["1"]
    run_root = pipeline.run_root_for(args)
    run_root.mkdir(parents=True)

    pipeline.write_resolved_run_config(args, run_root)

    text = (run_root / "run_config_resolved.yaml").read_text()
    assert "pipeline_id: resolved" in text
    assert "hpo:" in text
    assert "sampler_seed_base: 42" in text
    assert "sampler_seed_strategy: worker_offset" in text
    assert "hydra:" in text


def test_final_job_planning_uses_final_max_epochs_and_materialized_cache(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.endpoints = ["ki"]
    args.splits = ["random_splits_grouped_smiles"]
    args.seeds = [666]
    args.final_max_epochs = 1
    run_root = pipeline.run_root_for(args)
    write_tvt(args.data_root / "baselines_ki" / "random_splits_grouped_smiles")

    jobs = pipeline.build_final_jobs(args, run_root, params_by_endpoint={"ki": {}})

    command = " ".join(jobs[0].command)
    assert "trainer.max_epochs=1" in command
    assert f"data.materialized_cache_root={args.materialized_cache_root}" in command
    assert "data.use_materialized_features=true" in command


def test_final_job_planning_applies_final_only_overrides(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.endpoints = ["ki"]
    args.splits = ["random_splits_grouped_smiles"]
    args.seeds = [666]
    args.hydra_profile = {
        "common": {"data": {"pin_memory": True, "num_workers": 2}},
        "final": {
            "data": {
                "pin_memory": False,
                "num_workers": 0,
                "persistent_workers": False,
                "prefetch_factor": None,
            }
        },
    }
    run_root = pipeline.run_root_for(args)
    write_tvt(args.data_root / "baselines_ki" / "random_splits_grouped_smiles")

    jobs = pipeline.build_final_jobs(args, run_root, params_by_endpoint={"ki": {}})

    command = " ".join(jobs[0].command)
    assert "++data.pin_memory=true" in command
    assert "++data.num_workers=2" in command
    assert "++data.pin_memory=false" in command
    assert "++data.num_workers=0" in command
    assert "++data.persistent_workers=false" in command
    assert "++data.prefetch_factor=null" in command


def test_load_best_params_csv_requires_all_endpoints(tmp_path: Path) -> None:
    path = tmp_path / "hpo_best_params.csv"
    pd.DataFrame(
        [{"endpoint": "ki", "optimizer.lr": "1e-05", "model.hidden_dims": "[512,256]"}]
    ).to_csv(path, index=False)

    params = pipeline.load_best_params_csv(path, ["ki"])

    assert params == {"ki": {"optimizer.lr": "1e-05", "model.hidden_dims": "[512,256]"}}
    with pytest.raises(KeyError, match="missing endpoint"):
        pipeline.load_best_params_csv(path, ["ki", "km"])


def test_final_only_uses_best_params_csv_without_rewriting(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.endpoints = ["ki"]
    args.splits = ["random_splits_grouped_smiles"]
    args.seeds = [666]
    args.skip_hpo = True
    run_root = pipeline.run_root_for(args)
    write_tvt(args.data_root / "baselines_ki" / "random_splits_grouped_smiles")
    best_path = run_root / "hpo_best_params.csv"
    best_path.parent.mkdir(parents=True)
    original = 'endpoint,optimizer.lr,model.hidden_dims\nki,1e-05,"[512,256]"\n'
    best_path.write_text(original)

    params = pipeline.load_best_params_csv(
        pipeline.final_best_params_path(args, run_root), args.endpoints
    )
    jobs = pipeline.build_final_jobs(
        args,
        run_root,
        params_by_endpoint=params,
        write_best_params_csv=False,
    )

    command = " ".join(jobs[0].command)
    assert "++optimizer.lr=1e-05" in command
    assert "++model.hidden_dims=[512,256]" in command
    assert best_path.read_text() == original


def test_final_only_fixed_params_uses_profile_defaults_without_best_csv(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.endpoints = ["ki"]
    args.splits = ["random_splits_grouped_smiles"]
    args.seeds = [666]
    args.skip_hpo = True
    args.final_params_source = "fixed"
    args.hydra_profile = {
        "common": {
            "model": {
                "estimator": {
                    "alpha": 1.0,
                },
            },
        },
    }
    run_root = pipeline.run_root_for(args)
    write_tvt(args.data_root / "baselines_ki" / "random_splits_grouped_smiles")

    jobs = pipeline.build_final_jobs(
        args,
        run_root,
        params_by_endpoint={endpoint: {} for endpoint in args.endpoints},
        write_best_params_csv=False,
    )

    command = " ".join(jobs[0].command)
    assert "model.estimator.alpha=1.0" in command
    assert "++optimizer.lr=" not in command
    assert "++model.hidden_dims=" not in command
    assert not (run_root / "hpo_best_params.csv").exists()


def test_materialization_stage_reuses_cache_on_second_call(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.dry_run = False
    args.endpoints = ["ki"]
    args.splits = ["random_splits_grouped_smiles"]
    args.modalities = ["smited", "esmc"]
    args.materialized_cache_root = tmp_path / "materialized"
    run_root = pipeline.run_root_for(args)
    write_tvt(args.data_root / "baselines_ki" / "random_splits_grouped_smiles")
    write_cache(args.embeddings_root / "smited" / "materials_smi_ted_fork" / "v1", dim=1)
    write_cache(args.embeddings_root / "esmc" / "ESMC-6B" / "hf_v1", dim=1)
    specs = [{"endpoint": "ki", "split": "random_splits_grouped_smiles", "threshold": None}]

    first = pipeline.run_materialization_stage(
        args, specs, run_root, run_root / "materialization_status.csv"
    )
    second = pipeline.run_materialization_stage(
        args, specs, run_root, run_root / "materialization_status.csv"
    )

    assert first[0]["status"] == "built"
    assert second[0]["status"] == "hit"
    assert second[0]["reason"] == "fast_hit"


def test_preflight_reports_missing_pdb_ids(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    run_root = pipeline.run_root_for(args)
    _write_all_caches(args.embeddings_root)
    split_dir = args.data_root / "baselines_ki" / "random_splits_grouped_smiles"
    split_dir.mkdir(parents=True)
    frame = pd.DataFrame({"smiles": ["C"], "sequence": ["SEQ"], "log10_value": [1.0]})
    for name in ("train", "val", "test"):
        frame.to_parquet(split_dir / f"{name}.parquet", index=False)
    args.endpoints = ["ki"]
    args.splits = ["random_splits_grouped_smiles"]

    try:
        pipeline.preflight(args, run_root)
    except RuntimeError:
        report = (run_root / "preflight_report.csv").read_text()
        assert "missing_columns" in report
        assert "pdb_ids" in report
    else:
        raise AssertionError("preflight should fail when pdb_ids is absent")


def test_embedding_superset_and_plan_missing_cache(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.endpoints = ["ki"]
    args.splits = ["random_splits_grouped_smiles"]
    run_root = pipeline.run_root_for(args)
    split_dir = args.data_root / "baselines_ki" / "random_splits_grouped_smiles"
    split_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "smiles": ["C", "C", "CC"],
            "sequence": ["SEQ", "SEQ2", "SEQ"],
            "pdb_ids": [["P1", "P2"], ["P1"], ["P3"]],
            "log10_value": [1.0, 2.0, 3.0],
        }
    )
    for name in ("train", "val", "test"):
        frame.to_parquet(split_dir / f"{name}.parquet", index=False)

    pipeline.preflight(args, run_root)
    input_paths, keys_by_column = pipeline.build_embedding_superset(args, run_root)
    jobs = pipeline.build_embedding_jobs(args, run_root, input_paths, keys_by_column)

    assert keys_by_column["smiles"] == ["C", "CC"]
    assert keys_by_column["sequence"] == ["SEQ", "SEQ2"]
    assert keys_by_column["pdb_ids"] == ["P1", "P2", "P3"]
    assert {job.modality for job in jobs} == set(pipeline.MODALITIES)
    molr_job = next(job for job in jobs if job.modality == "molr")
    molr_command = " ".join(molr_job.command)
    assert "conda run" not in molr_command
    assert "embeddings.molr_use_gpu=true" in molr_command
    assert "embeddings.molr_gpu_ids=[0,1]" in molr_command
    esmc_job = next(job for job in jobs if job.modality == "esmc")
    esmc_command = " ".join(esmc_job.command)
    assert "embeddings.esmc_model_name=ESMC-6B" in esmc_command
    assert "embeddings.esmc_version=hf_v1" in esmc_command
    plan = pd.read_csv(run_root / "embedding_plan.csv")
    assert set(plan["modality"]) == set(pipeline.MODALITIES)
    assert int(plan.loc[plan["modality"] == "ecfp", "missing_keys"].iloc[0]) == 2
    assert "ESMC-6B/hf_v1" in plan.loc[plan["modality"] == "esmc", "cache_root"].iloc[0]


def test_embedding_superset_reuses_cache_without_parquet_reads(
    tmp_path: Path, monkeypatch
) -> None:
    args = pipeline_args(tmp_path)
    args.endpoints = ["ki"]
    args.splits = ["random_splits_grouped_smiles"]
    first_run_root = args.output_root / "first"
    split_dir = args.data_root / "baselines_ki" / "random_splits_grouped_smiles"
    split_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "smiles": ["C", "CC"],
            "sequence": ["SEQ", "SEQ2"],
            "pdb_ids": [["P1"], ["P2"]],
            "log10_value": [1.0, 2.0],
        }
    )
    for name in ("train", "val", "test"):
        frame.to_parquet(split_dir / f"{name}.parquet", index=False)

    first_input_paths, first_keys = pipeline.build_embedding_superset(args, first_run_root)

    def fail_read_table(*pipeline_args, **_kwargs):
        raise AssertionError("cached superset should not reread parquet files")

    monkeypatch.setattr(pipeline.pq, "read_table", fail_read_table)
    second_run_root = args.output_root / "second"
    second_input_paths, second_keys = pipeline.build_embedding_superset(args, second_run_root)

    assert second_keys == first_keys
    assert second_input_paths == first_input_paths
    manifest = pd.read_csv(second_run_root / "embedding_input_manifest.csv")
    assert set(manifest["cache_status"]) == {"hit"}


def test_skip_embeddings_requires_existing_cache(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.skip_embeddings = True
    args.endpoints = ["ki"]
    args.splits = ["random_splits_grouped_smiles"]
    run_root = pipeline.run_root_for(args)
    write_tvt(args.data_root / "baselines_ki" / "random_splits_grouped_smiles")

    try:
        pipeline.preflight(args, run_root)
    except RuntimeError:
        report = (run_root / "preflight_report.csv").read_text()
        assert "missing_cache" in report
    else:
        raise AssertionError(
            "preflight should fail when embedding generation is skipped and caches are absent"
        )


def test_child_runs_get_data_root_and_cache_roots(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    overrides = pipeline.build_base_overrides(args, "ki", tmp_path / "out", "final")
    assert f"data.data_root={(tmp_path / 'data').resolve()}" in overrides
    esmc_root = (args.embeddings_root / "esmc/ESMC-6B/hf_v1").resolve()
    assert f"data.cache_roots.esmc={esmc_root}" in overrides


def test_hpo_workers_get_their_own_sweep_dir(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.endpoints = ["ki"]
    args.hpo_workers_per_endpoint = 2
    write_tvt(args.data_root / "baselines_ki" / "enzyme_structure_splits")
    jobs = pipeline.build_hpo_jobs(args, pipeline.run_root_for(args))
    sweep_dirs = [next(c for c in job.command if c.startswith("hydra.sweep.dir=")) for job in jobs]
    assert len(set(sweep_dirs)) == 2 and all("worker_" in d for d in sweep_dirs)


def test_embedding_count_matches_any_pdb_source_variant(tmp_path: Path) -> None:
    from src.data.components.embedders.utils import PDB_SOURCE_KEY_TYPE

    pipeline = load_script("run_baseline_mlp_pipeline")

    metadata = CacheMetadata(
        embedder="t",
        model_name="m",
        version="v1",
        storage_dtype="float32",
        embedding_dim=None,
        max_shard_bytes=1 << 20,
        key_type=PDB_SOURCE_KEY_TYPE,
        key_field="acc_id",
    )
    store = ShardedEmbeddingStore(tmp_path, metadata, logging.getLogger("t"))
    store.put_many({"X1|alphafold": np.ones(2, dtype=np.float32)}, flush=True)
    assert pipeline.existing_embedding_count(tmp_path, ["X1", "X2"]) == 1


def test_final_dir_layout(tmp_path: Path) -> None:
    root = tmp_path / "MLP"
    assert pipeline.final_dir(root, "ki", "random_splits_grouped_smiles", None, 666) == (
        root / "ki" / "random_splits_grouped_smiles" / "all" / "seed_666"
    )
    assert pipeline.final_dir(root, "kcat", "enzyme_structure_splits", "threshold_0.9", 7) == (
        root / "kcat" / "enzyme_structure_splits" / "threshold_0.9" / "seed_7"
    )
    assert (
        pipeline.final_dir(root, "classification", "s", None, 1)
        == root / "enzyme_classification" / "s" / "all" / "seed_1"
    )
    assert (
        pipeline.final_dir(root, "retrieval", "s", "threshold_x", 1)
        == root / "enzyme_retrieval" / "s" / "threshold_x" / "seed_1"
    )


def test_final_job_metrics_scans_final_layout_only(tmp_path: Path) -> None:
    for endpoint, threshold in (("classification", None), ("ki", "threshold_0.9")):
        job_dir = pipeline.final_dir(tmp_path, endpoint, "split", threshold, 1)
        job_dir.mkdir(parents=True)
        (job_dir / "metrics.json").write_text("{}")
    stray = tmp_path / "_runs" / "run" / "a" / "b" / "seed_1"
    stray.mkdir(parents=True)
    (stray / "metrics.json").write_text("{}")

    found = [row[:4] for row in pipeline.final_job_metrics(tmp_path)]
    assert found == [("classification", "split", None, "1"), ("ki", "split", "threshold_0.9", "1")]
    assert [row[0] for row in pipeline.final_job_metrics(tmp_path, ["ki"])] == ["ki"]
