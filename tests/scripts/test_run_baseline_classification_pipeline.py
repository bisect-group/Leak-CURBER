from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from tests.helpers.builders import dispatch_args, load_script, write_classification_tvt

pipeline = load_script("run_baseline_mlp_pipeline")


classification_pipeline = load_script("run_baseline_classification_pipeline")


def test_classification_threshold_split_resolution(tmp_path: Path) -> None:
    class_root = tmp_path / "data" / "enzyme_classification_dataset"
    write_classification_tvt(class_root / "enzyme_structure_splits" / "threshold_0.9")

    resolved_threshold = classification_pipeline.resolve_split_dir(
        tmp_path / "data",
        "classification",
        "enzyme_structure_splits",
        "0.9",
    )

    assert resolved_threshold.name == "threshold_0.9"
    assert classification_pipeline.split_threshold(resolved_threshold) == "threshold_0.9"


def test_pipeline_uses_rxn_key_for_drfp() -> None:
    module = load_script("run_baseline_classification_pipeline")
    assert module.MODALITY_KEY_COLUMNS["drfp"] == "rxn_key"
    assert "rxn_key" in module.required_columns_for_modalities(["drfp"])


def test_materialization_totals_read_the_built_cache_keys() -> None:
    classification = load_script("run_baseline_classification_pipeline")
    info = {"estimated_total_gb": 1.5, "splits": {"train": {"rows": 3}, "val": {"rows": 1}}}
    assert classification.materialization_totals(info) == (4, 1.5)
    assert classification.materialization_totals({"rows": 2, "estimated_gb": 0.5}) == (2, 0.5)


def test_pipeline_looks_for_the_grouped_random_folders() -> None:
    classification = load_script("run_baseline_classification_pipeline")
    assert "random_splits" not in classification.SPLITS
    assert {"random_splits_grouped_sequence", "random_splits_grouped_rxn_key"} <= set(
        classification.SPLITS
    )


def _run_main_with_spy(monkeypatch, tmp_path: Path, main_call, task):
    """Run main() up to the preflight hook, with the heavy setup steps mocked."""
    for name in (
        "acquire_run_lock",
        "write_resolved_run_config",
        "render_disk_summary",
        "log_step",
    ):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "write_disk_report", lambda *a, **k: [])
    monkeypatch.setattr(pipeline.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "parse_args", lambda: dispatch_args(tmp_path))
    for name in ("_TASK", "_ACTIVE_ARGS", "_SHUTDOWN_CLEANUP_ARMED"):
        monkeypatch.setattr(pipeline, name, getattr(pipeline, name))
    spy = MagicMock()
    original = task.preflight
    object.__setattr__(task, "preflight", spy)  # frozen dataclass
    try:
        main_call()
    finally:
        object.__setattr__(task, "preflight", original)
    return spy


def test_classification_task_dispatches_classification_hooks(monkeypatch, tmp_path: Path) -> None:
    spy = _run_main_with_spy(
        monkeypatch, tmp_path, lambda: classification_pipeline.main(), classification_pipeline.TASK
    )
    assert pipeline._TASK is classification_pipeline.TASK
    spy.assert_called_once()
    task = classification_pipeline.TASK
    assert task.resolve_split_dir is classification_pipeline.resolve_split_dir
    assert task.build_hpo_jobs is classification_pipeline.build_hpo_jobs
    assert task.modalities == classification_pipeline.MODALITIES


def test_regression_default_dispatches_regression_hooks(monkeypatch, tmp_path: Path) -> None:
    spy = _run_main_with_spy(
        monkeypatch, tmp_path, lambda: pipeline.main(), pipeline.REGRESSION_TASK
    )
    assert pipeline._TASK is pipeline.REGRESSION_TASK
    spy.assert_called_once()
    task = pipeline.REGRESSION_TASK
    assert task.resolve_split_dir is pipeline.resolve_split_dir
    assert task.build_hpo_jobs is pipeline.build_hpo_jobs
    assert task is not classification_pipeline.TASK
    assert task.modalities != classification_pipeline.TASK.modalities
