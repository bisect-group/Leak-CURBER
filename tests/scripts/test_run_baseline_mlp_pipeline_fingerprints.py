from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tests.helpers.builders import (
    load_script,
    pipeline_args,
    superset_args,
    write_tvt,
)

pipeline = load_script("run_baseline_mlp_pipeline")


def test_superset_key_tracks_store_version_and_warns_on_pending(tmp_path: Path, capsys) -> None:
    args = superset_args(tmp_path)
    root = pipeline.cache_roots(args.embeddings_root)["ecfp"]
    root.mkdir(parents=True)
    meta = root / "meta.json"
    meta.write_text(json.dumps({"version": "v1"}))
    specs = pipeline._TASK.discover_split_specs(args)

    def key():
        return pipeline.embedding_superset_cache_key(
            pipeline.embedding_superset_fingerprint(args, specs, ["smiles"])
        )

    before = key()
    meta.write_text(json.dumps({"version": "v2"}))
    assert key() != before
    (root / "_pending_shard.json").write_text(json.dumps({"keys": ["a", "b", "c"]}))
    key()
    assert "3 pending rows" in capsys.readouterr().out


def test_changed_split_file_changes_study_name_and_job_ids(tmp_path: Path) -> None:
    args = pipeline_args(tmp_path)
    args.endpoints = ["ki"]
    split_dir = args.data_root / "baselines_ki" / "enzyme_structure_splits"
    write_tvt(split_dir)
    run_root = pipeline.run_root_for(args)
    before_study = pipeline.study_name_for_endpoint(args, "ki")
    before_job = pipeline.build_hpo_jobs(args, run_root)[0].job_id
    pd.DataFrame(
        {
            "smiles": ["C", "CC"],
            "sequence": ["S", "T"],
            "pdb_ids": [["P1"], ["P2"]],
            "log10_value": [1.0, 2.0],
        }
    ).to_parquet(split_dir / "train.parquet", index=False)
    assert pipeline.study_name_for_endpoint(args, "ki") != before_study
    assert pipeline.build_hpo_jobs(args, run_root)[0].job_id != before_job


def test_fingerprint_is_stable_for_unchanged_files_and_survives_an_empty_namespace(
    tmp_path: Path,
) -> None:
    import argparse

    args = pipeline_args(tmp_path)
    write_tvt(args.data_root / "baselines_ki" / "enzyme_structure_splits")
    first = pipeline.data_fingerprint(args, "ki", "enzyme_structure_splits")
    assert first == pipeline.data_fingerprint(args, "ki", "enzyme_structure_splits") != "nodata"
    assert (
        pipeline.data_fingerprint(argparse.Namespace(), "ki", "enzyme_structure_splits")
        == "nodata"
    )
