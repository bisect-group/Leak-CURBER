from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import pytest

from tests.helpers.builders import load_script, pipeline_args

pipeline = load_script("run_baseline_mlp_pipeline")


def test_run_jobs_skips_previously_successful_jobs(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.dry_run = False
    args.gpus = ["0"]
    args.max_slots_per_gpu = 1
    status_path = tmp_path / "status.csv"
    skipped_marker = tmp_path / "skipped.txt"
    pending_marker = tmp_path / "pending.txt"

    skipped = pipeline.PipelineJob(
        job_id="done",
        stage="final",
        endpoint="ki",
        split="random_splits_grouped_smiles",
        threshold=None,
        seed=666,
        worker_index=0,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(skipped_marker)!r}).write_text('should-not-run'); raise SystemExit(99)",
        ],
        output_dir=tmp_path / "done",
    )
    pending = pipeline.PipelineJob(
        job_id="pending",
        stage="final",
        endpoint="ki",
        split="random_splits_grouped_smiles",
        threshold=None,
        seed=777,
        worker_index=0,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(pending_marker)!r}).write_text('ran'); "
            f"Path({str(tmp_path / 'pending')!r}).mkdir(exist_ok=True); "
            f"Path({str(tmp_path / 'pending' / 'metrics.json')!r}).write_text('{{}}')",
        ],
        output_dir=tmp_path / "pending",
    )
    pipeline.write_dict_rows(
        status_path,
        [
            {
                "job_id": "done",
                "stage": "final",
                "endpoint": "ki",
                "status": "success",
                "output_dir": str(skipped.output_dir),
            }
        ],
    )

    rows = pipeline.run_jobs(args, [skipped, pending], status_path)

    assert not skipped_marker.exists()
    assert pending_marker.read_text() == "ran"
    assert {row["job_id"] for row in rows} == {"done", "pending"}
    assert all(row["status"] == "success" for row in rows)


def test_run_jobs_resume_uses_latest_status_for_retried_job(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.dry_run = False
    args.gpus = ["0"]
    status_path = tmp_path / "hpo_status.csv"
    marker = tmp_path / "retried.txt"
    job = pipeline.PipelineJob(
        job_id="hpo_ic50_worker_0",
        stage="hpo",
        endpoint="ic50",
        split="enzyme_structure_splits",
        threshold="threshold_0.6",
        seed=None,
        worker_index=0,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ok')",
        ],
        output_dir=tmp_path / "hpo" / "ic50" / "worker_0",
    )
    pipeline.write_dict_rows(
        status_path,
        [
            {
                "job_id": job.job_id,
                "stage": "hpo",
                "endpoint": "ic50",
                "status": "failed",
                "output_dir": str(job.output_dir),
            }
        ],
    )

    rows = pipeline.run_jobs(args, [job], status_path)

    assert marker.read_text() == "ok"
    assert pipeline.jobs_succeeded(rows, [job])
    assert pipeline.count_successful_jobs(rows, [job]) == 1
    assert [row["status"] for row in rows if row["job_id"] == job.job_id] == ["failed", "success"]


def test_streaming_finals_start_after_endpoint_hpo_before_all_hpo_done(
    tmp_path: Path, monkeypatch
) -> None:
    args = pipeline_args(tmp_path)
    args.dry_run = False
    args.gpus = ["0", "1"]
    args.max_slots_per_gpu = 1
    args.endpoints = ["fast", "slow"]
    run_root = pipeline.run_root_for(args)
    hpo_jobs = [
        pipeline.PipelineJob(
            job_id="hpo_fast_worker_0",
            stage="hpo",
            endpoint="fast",
            split="enzyme_structure_splits",
            threshold=None,
            seed=None,
            worker_index=0,
            command=[sys.executable, "-c", "pass"],
            output_dir=run_root / "hpo" / "fast" / "worker_0",
        ),
        pipeline.PipelineJob(
            job_id="hpo_slow_worker_0",
            stage="hpo",
            endpoint="slow",
            split="enzyme_structure_splits",
            threshold=None,
            seed=None,
            worker_index=0,
            command=[sys.executable, "-c", "pass"],
            output_dir=run_root / "hpo" / "slow" / "worker_0",
        ),
    ]
    final_job = pipeline.PipelineJob(
        job_id="final_fast_split_seed_666",
        stage="final",
        endpoint="fast",
        split="random_splits_grouped_smiles",
        threshold=None,
        seed=666,
        worker_index=0,
        command=[sys.executable, "-c", "pass"],
        output_dir=pipeline.final_dir(
            args.output_root, "fast", "random_splits_grouped_smiles", None, 666
        ),
    )
    events: list[tuple[str, float]] = []

    def fake_execute(pipeline_args, job, gpu_id, slot_index, **_kwargs):
        events.append((f"start:{job.job_id}", time.monotonic()))
        if job.job_id == "hpo_slow_worker_0":
            time.sleep(0.3)
        events.append((f"end:{job.job_id}", time.monotonic()))
        return {
            "job_id": job.job_id,
            "stage": job.stage,
            "endpoint": job.endpoint,
            "split": job.split,
            "threshold": job.threshold,
            "seed": job.seed,
            "gpu": gpu_id,
            "slot": slot_index,
            "status": "success",
            "returncode": 0,
            "seconds": 0.0,
            "output_dir": str(job.output_dir),
            "log_path": str(job.output_dir / "pipeline_stdout_stderr.log"),
            "command": " ".join(job.command),
        }

    monkeypatch.setattr(pipeline, "execute_job", fake_execute)
    monkeypatch.setattr(
        pipeline,
        "endpoint_hpo_completed",
        lambda pipeline_args, _run_root, endpoint: endpoint == "fast",
    )
    monkeypatch.setattr(
        pipeline,
        "_TASK",
        dataclasses.replace(
            pipeline._TASK,
            materialization_specs_for_final_endpoint=lambda pipeline_args, _run_root, endpoint: [
                {"endpoint": endpoint, "split": "random_splits_grouped_smiles", "threshold": None}
            ],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "run_materialization_stage",
        lambda *pipeline_args, **_kwargs: [
            {
                "endpoint": "fast",
                "split": "random_splits_grouped_smiles",
                "threshold": None,
                "status": "hit",
            }
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "build_final_jobs_for_endpoint",
        lambda pipeline_args, _run_root, endpoint: (
            [final_job],
            [{"endpoint": endpoint, "optimizer.lr": "1e-3"}],
        ),
    )

    pipeline.run_hpo_with_streaming_finals(args, run_root, hpo_jobs)

    event_times = dict(events)
    assert event_times["start:final_fast_split_seed_666"] < event_times["end:hpo_slow_worker_0"]


def test_cleanup_job_output_deletes_only_job_local_materialized_features(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.dry_run = False
    args.cleanup = True
    shared_cache = (
        args.materialized_cache_root
        / "ki"
        / "enzyme_structure_splits"
        / "threshold_0.1"
        / "fingerprint"
    )
    shared_cache.mkdir(parents=True)
    (shared_cache / ".complete").write_text("ok")
    job_dir = tmp_path / "job"
    local_cache = job_dir / "materialized_features"
    local_cache.mkdir(parents=True)
    (local_cache / "train_X.npy").write_bytes(b"x" * 128)
    job = pipeline.PipelineJob(
        job_id="final_ki",
        stage="final",
        endpoint="ki",
        split="enzyme_structure_splits",
        threshold="threshold_0.1",
        seed=666,
        worker_index=0,
        command=[sys.executable, "-c", "pass"],
        output_dir=job_dir,
    )

    rows = pipeline.cleanup_job_output(args, job)

    assert not local_cache.exists()
    assert shared_cache.exists()
    assert rows[0]["kind"] == "job_materialized_features"
    assert rows[0]["action"] == "deleted"


def test_hpo_checkpoint_cleanup_prunes_checkpoints_and_keeps_logs(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.dry_run = False
    args.cleanup = True
    run_root = tmp_path / "run"
    ckpt = (
        run_root
        / "hpo"
        / "ki"
        / "worker_0"
        / "trials"
        / "trial_0"
        / "checkpoints"
        / "epoch=0.ckpt"
    )
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"checkpoint")
    log_path = run_root / "hpo" / "ki" / "worker_0" / "pipeline_stdout_stderr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("log")

    rows = pipeline.prune_checkpoints(args, run_root, "hpo")

    assert not ckpt.exists()
    assert log_path.exists()
    assert rows[0]["kind"] == "checkpoint"
    assert rows[0]["stage"] == "hpo"


def test_disk_report_is_written(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    run_root = pipeline.run_root_for(args)
    (run_root / "child").mkdir(parents=True)
    (run_root / "child" / "file.bin").write_bytes(b"x" * 16)

    rows = pipeline.write_disk_report(args, run_root, "before")

    assert rows
    assert (run_root / "disk_usage_before.csv").exists()


def test_streaming_job_tees_output_to_console_and_log(tmp_path: Path, capsys) -> None:
    args = pipeline_args(tmp_path)
    args.dry_run = False
    args.gpus = ["0"]
    job = pipeline.PipelineJob(
        job_id="embed_molr",
        stage="embed",
        endpoint="",
        split=None,
        threshold=None,
        seed=None,
        worker_index=0,
        command=[sys.executable, "-c", "print('molr-progress')"],
        output_dir=tmp_path / "embed_molr",
        modality="molr",
        requires_gpu=False,
        stream_output=True,
    )

    rows = pipeline.run_jobs(args, [job], tmp_path / "status.csv")

    captured = capsys.readouterr()
    assert "[embed/molr] molr-progress" in captured.out
    assert (
        "[embed/molr] molr-progress" in (job.output_dir / "pipeline_stdout_stderr.log").read_text()
    )
    assert rows[0]["gpu"] == "0"
    assert rows[0]["status"] == "success"


def test_molr_embedding_job_receives_all_allowed_gpus(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.dry_run = False
    args.gpus = ["1", "2", "3"]
    marker = tmp_path / "visible_gpus.txt"
    job = pipeline.PipelineJob(
        job_id="embed_molr",
        stage="embed",
        endpoint="",
        split=None,
        threshold=None,
        seed=None,
        worker_index=0,
        command=[
            sys.executable,
            "-c",
            f"import os; from pathlib import Path; Path({str(marker)!r}).write_text(os.environ.get('CUDA_VISIBLE_DEVICES', ''))",
        ],
        output_dir=tmp_path / "embed_molr",
        modality="molr",
        requires_gpu=False,
        stream_output=False,
    )

    rows = pipeline.run_jobs(args, [job], tmp_path / "status.csv")

    assert marker.read_text() == "1,2,3"
    assert rows[0]["gpu"] == "1,2,3"
    assert rows[0]["status"] == "success"


def test_molr_graph_build_uses_pysmiles_compatibility() -> None:
    featurizer = pytest.importorskip("MolR.featurizer")

    feature_encoder = featurizer.load_feature_encoder("gcn_1024")
    recovered = "[C]1[C][N]C(=Nn2cnc3ccccc32)[N]1"
    invalid_ez = "[C]=[C][C]N1[C]=[C]/C(=[C]\\[C]=C(C#N)C#N)c2ccccc21"
    charged = "[C]C(=N[N+]=C([N])[N])C1=[C]C([C])([C])[C][C][C]1"
    tellurium = (
        "[C]N([C])c1ccc2c(-c3ccc(C(=S)N4[C][C][C][C][C]4)s3)c3[c][c]c(=[N+]([C])[C])[c]c-3[te]c2c1"
    )

    for index, smiles in enumerate([recovered, invalid_ez]):
        result = featurizer.build_graph_result(index, smiles, feature_encoder)
        assert result.graph is not None
        assert result.smiles == smiles
        assert result.error is None

    for index, smiles in enumerate([charged, tellurium], start=10):
        result = featurizer.build_graph_result(index, smiles, feature_encoder)
        assert result.graph is None
        assert result.smiles == smiles
        assert result.error is not None
