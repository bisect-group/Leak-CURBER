from __future__ import annotations

import json
import sys
from pathlib import Path

import optuna
import pytest

from tests.helpers.builders import REPO_ROOT, load_script, pipeline_args

pipeline = load_script("run_baseline_mlp_pipeline")


def test_rich_progress_falls_back_when_rich_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    args = pipeline_args(tmp_path)
    args.no_progress = False
    args.progress_backend = "rich"

    monkeypatch.setattr(pipeline, "_CONSOLE", None)

    assert not pipeline.rich_enabled(args)


def test_collect_hpo_progress_reads_optuna_sqlite_study(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.output_root = tmp_path / "out"
    args.pipeline_id = "progress"
    args.n_trials = 3
    run_root = pipeline.run_root_for(args)
    study_name = pipeline.study_name_for_endpoint(args, "ki")
    storage = pipeline.optuna_storage_url(run_root, study_name)
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize")
    complete = study.ask()
    study.tell(complete, 0.42)
    failed = study.ask()
    study.tell(failed, state=optuna.trial.TrialState.FAIL)

    job = pipeline.PipelineJob(
        job_id="hpo_ki_worker_0",
        stage="hpo",
        endpoint="ki",
        split="enzyme_structure_splits",
        threshold=None,
        seed=None,
        worker_index=0,
        command=[sys.executable, "-c", "pass"],
        output_dir=run_root / "hpo" / "ki" / "worker_0",
    )

    snapshot = pipeline.collect_hpo_progress(args, [job])["ki"]

    assert snapshot["completed"] == 1
    assert snapshot["failed"] == 1
    assert snapshot["target"] == 3
    assert snapshot["best_value"] == "0.42"


def test_environment_pins_modern_rich() -> None:
    assert "rich==14." in (REPO_ROOT / "environment.yaml").read_text()


def test_final_dashboard_renders_active_job(tmp_path: Path) -> None:
    if pipeline.Table is None:
        pytest.skip("rich is not installed")
    args = pipeline_args(tmp_path)
    args.no_progress = False
    args.progress_backend = "rich"
    job = pipeline.PipelineJob(
        job_id="final_ki_random_seed_666",
        stage="final",
        endpoint="ki",
        split="random_splits_grouped_smiles",
        threshold=None,
        seed=666,
        worker_index=0,
        command=[sys.executable, "-c", "pass"],
        output_dir=tmp_path / "final",
    )
    active = {
        job.job_id: {
            "job": job,
            "gpu": "1",
            "slot": 0,
            "started_at": 1.0,
            "attempt": 1,
            "status": "running",
        }
    }

    renderable = pipeline.render_job_dashboard(args, "final", [job], [job], [], active, {})

    assert renderable is not None


def test_stream_display_state_sanitizes_carriage_return_progress() -> None:
    state = pipeline.StreamDisplayState(max_lines=3)

    state.feed("[embed/esm3] feature shards: 80%|\r")
    state.feed("[embed/esm3] feature shards: 90%|#########\r")
    state.feed("[embed/esm3] done\n")

    lines = state.render_lines()
    assert any("feature shards: 90%" in line for line in lines)
    assert lines[-1] == "[embed/esm3] done"


def test_split_dashboard_renders_stream_panel(tmp_path: Path) -> None:
    if pipeline.Table is None:
        pytest.skip("rich is not installed")
    args = pipeline_args(tmp_path)
    args.no_progress = False
    args.progress_backend = "rich"
    job = pipeline.PipelineJob(
        job_id="embed_esm3",
        stage="embed",
        endpoint="",
        split=None,
        threshold=None,
        seed=None,
        worker_index=0,
        command=[sys.executable, "-c", "pass"],
        output_dir=tmp_path / "embed_esm3",
        modality="esm3",
        stream_output=True,
    )
    state = pipeline.StreamDisplayState()
    state.feed("[embed/esm3] feature shards: 90%|#########\r")

    renderable = pipeline.render_job_dashboard(
        args, "embed", [job], [job], [], {}, {}, {job.job_id: state}
    )

    assert renderable is not None


def test_progress_callback_writes_state_and_events(tmp_path: Path) -> None:
    from src.callbacks.orchestrator_progress import OrchestratorProgressCallback

    callback = OrchestratorProgressCallback(
        tmp_path,
        stage="final",
        endpoint="ki",
        split="random_splits_grouped_smiles",
        seed="666",
        job_id="job",
        status_interval_sec=0,
    )

    class DummyTrainer:
        is_global_zero = True
        current_epoch = 2
        max_epochs = 10
        num_training_batches = 100
        callback_metrics = {"val/r2": 0.12345, "val/loss": 1.5}

    callback.train_batch_idx = 24
    callback.on_validation_epoch_end(DummyTrainer(), None)

    state = json.loads((tmp_path / "progress_state.json").read_text())
    assert state["stage"] == "final"
    assert state["endpoint"] == "ki"
    assert state["split"] == "random_splits_grouped_smiles"
    assert state["seed"] == "666"
    assert state["current_epoch"] == 2
    assert state["train_batch_idx"] == 24
    assert state["current_val_r2"] == pytest.approx(0.12345)
    assert (tmp_path / "progress_events.jsonl").read_text().strip()


def test_batch_progress_text_includes_resettable_bar() -> None:
    text = pipeline.batch_progress_text({"train_batch_idx": 24, "train_num_batches": 100})

    assert text == "25/100 [##--------]"


def test_hpo_monitor_reads_active_trial_progress(tmp_path: Path) -> None:
    if pipeline.Table is None or pipeline.Console is None:
        pytest.skip("rich is not installed")
    args = pipeline_args(tmp_path)
    run_root = tmp_path / "run"
    job = pipeline.PipelineJob(
        job_id="hpo_ki_worker_0",
        stage="hpo",
        endpoint="ki",
        split="enzyme_structure_splits",
        threshold=None,
        seed=None,
        worker_index=0,
        command=[sys.executable, "-c", "pass"],
        output_dir=run_root / "hpo" / "ki" / "worker_0",
    )
    trial_dir = job.output_dir / "trials" / "0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "progress_state.json").write_text(
        json.dumps(
            {"hydra_job_num": "0", "current_epoch": 4, "max_epochs": 20, "current_val_r2": 0.2}
        )
    )
    active = {
        job.job_id: {
            "job": job,
            "gpu": "1",
            "slot": 0,
            "started_at": 1.0,
            "attempt": 1,
            "status": "running",
        }
    }

    renderable = pipeline.render_job_dashboard(
        args, "hpo", [job], [job], [], active, {"ki": {"target": 1}}
    )
    console = pipeline.Console(record=True, width=200)
    console.print(renderable)
    text = console.export_text()

    assert renderable is not None
    assert "study best val/r2" in text
    assert "study last" in text
    assert pipeline.load_progress_state_for_job(job)["current_val_r2"] == 0.2


def test_rich_run_jobs_executes_short_job(tmp_path: Path) -> None:
    if pipeline.Table is None:
        pytest.skip("rich is not installed")
    args = pipeline_args(tmp_path)
    args.dry_run = False
    args.no_progress = False
    args.progress_backend = "rich"
    args.gpus = ["0"]
    marker = tmp_path / "rich_job.txt"
    job = pipeline.PipelineJob(
        job_id="rich_short",
        stage="final",
        endpoint="ki",
        split="random_splits_grouped_smiles",
        threshold=None,
        seed=666,
        worker_index=0,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ok'); "
            f"Path({str(tmp_path / 'rich_short')!r}).mkdir(exist_ok=True); "
            f"Path({str(tmp_path / 'rich_short' / 'metrics.json')!r}).write_text('{{}}')",
        ],
        output_dir=tmp_path / "rich_short",
    )

    rows = pipeline.run_jobs(args, [job], tmp_path / "status.csv")

    assert marker.read_text() == "ok"
    assert rows[0]["status"] == "success"
