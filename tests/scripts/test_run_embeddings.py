import argparse
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from tests.helpers.builders import load_script

run_embeddings = load_script("run_embeddings")


def _args(**kwargs) -> argparse.Namespace:
    defaults = {
        "embedders": None,
        "skip": None,
        "continue_on_error": False,
        "overrides": None,
        "unimol_input_path": None,
    }
    return argparse.Namespace(**{**defaults, **kwargs})


def test_build_embedder_list_defaults_to_all_embedders():
    assert run_embeddings.build_embedder_list(_args()) == list(run_embeddings.EMBEDDER_SPECS)


def test_build_embedder_list_keeps_selection_order_and_drops_skipped():
    args = _args(embedders=["rxnfp", "ecfp", "drfp"], skip=["ecfp"])
    assert run_embeddings.build_embedder_list(args) == ["rxnfp", "drfp"]


def test_parse_args_rejects_unknown_embedder(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_embeddings.py", "--embedders", "nope"])
    with pytest.raises(SystemExit):
        run_embeddings.parse_args()


def test_parse_args_reads_lists_and_flags(monkeypatch):
    argv = ["run_embeddings.py", "--datasets", "a", "b", "--skip", "esmc", "--continue-on-error"]
    monkeypatch.setattr(sys, "argv", argv)
    args = run_embeddings.parse_args()
    assert args.datasets == ["a", "b"]
    assert args.skip == ["esmc"]
    assert args.continue_on_error is True


def test_main_lists_embedders_without_composing_config(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_embeddings.py", "--list-embedders"])
    run_embeddings.main()
    assert capsys.readouterr().out.split() == list(run_embeddings.EMBEDDER_SPECS)


def test_main_rejects_an_empty_selection(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["run_embeddings.py", "--embedders", "ecfp", "--skip", "ecfp"]
    )
    with pytest.raises(ValueError, match="No embedders selected"):
        run_embeddings.main()


class _Recorder:
    calls: list[tuple] = []

    def __init__(self, cfg):
        self.cfg = cfg

    def embed(self, **kwargs):
        self.calls.append(("embed", kwargs))


@pytest.fixture
def recorder(monkeypatch):
    _Recorder.calls = []
    monkeypatch.setattr(run_embeddings, "_EMBEDDER_CACHE", {})
    module = SimpleNamespace(SmilesECFPShardEmbedder=_Recorder, UnimolSDFShardEmbedder=_Recorder)
    monkeypatch.setattr(run_embeddings, "import_module", lambda _name: module)
    return _Recorder


def test_run_embedder_builds_cacheable_embedders_once(recorder):
    cfg = OmegaConf.create({})
    run_embeddings.run_embedder("ecfp", cfg, _args())
    run_embeddings.run_embedder("ecfp", cfg, _args())
    assert recorder.calls == [("embed", {}), ("embed", {})]
    assert isinstance(run_embeddings._EMBEDDER_CACHE["ecfp"], _Recorder)


def test_run_embedder_passes_unimol_options_and_caches(recorder):
    args = _args(unimol_input_path="in.pkl")
    run_embeddings.run_embedder("unimol_sdf", OmegaConf.create({}), args)
    assert recorder.calls == [("embed", {"input_override": "in.pkl"})]
    assert "unimol_sdf" in run_embeddings._EMBEDDER_CACHE


def _stub_compose(monkeypatch, cfg):
    seen = {}

    def fake_compose(config_name, overrides):
        seen.update(config_name=config_name, overrides=overrides)
        return cfg

    monkeypatch.setattr(run_embeddings, "compose", fake_compose)
    return seen


def test_run_embedders_for_dataset_composes_dataset_override(monkeypatch):
    cfg = OmegaConf.create({"embeddings": {}, "splits": {"dataset": {}}})
    seen = _stub_compose(monkeypatch, cfg)
    ran = []
    monkeypatch.setattr(run_embeddings, "run_embedder", lambda name, *_: ran.append(name))
    args = _args(overrides=["x=1"])
    assert run_embeddings.run_embedders_for_dataset("ds", ["ecfp", "rxnfp"], args) == []
    assert seen["config_name"] == "data_processing"
    assert seen["overrides"] == ["data/splits_dataset@splits.dataset=ds", "x=1"]
    assert ran == ["ecfp", "rxnfp"]


def test_run_embedders_for_dataset_points_drfp_at_the_tanimoto_input(monkeypatch):
    cfg = OmegaConf.create(
        {
            "embeddings": {},
            "splits": {"dataset": {"tanimoto_split_input_dataset_parquet_file_path": "t.parquet"}},
        }
    )
    _stub_compose(monkeypatch, cfg)
    monkeypatch.setattr(run_embeddings, "run_embedder", lambda *_: None)
    run_embeddings.run_embedders_for_dataset("ds", ["drfp"], _args())
    assert cfg.embeddings.drfp_embeddings_input_dataset_parquet_file_path == "t.parquet"


def _failing(name, *_):
    if name == "ecfp":
        raise RuntimeError("boom")


def test_run_embedders_for_dataset_collects_failures_when_continuing(monkeypatch):
    _stub_compose(monkeypatch, OmegaConf.create({"embeddings": {}, "splits": {"dataset": {}}}))
    ran = []
    monkeypatch.setattr(
        run_embeddings, "run_embedder", lambda name, *a: (ran.append(name), _failing(name))
    )
    failed = run_embeddings.run_embedders_for_dataset(
        "ds", ["ecfp", "rxnfp"], _args(continue_on_error=True)
    )
    assert failed == ["ds/ecfp"]
    assert ran == ["ecfp", "rxnfp"]


def test_run_embedders_for_dataset_stops_at_the_first_failure_by_default(monkeypatch):
    _stub_compose(monkeypatch, OmegaConf.create({"embeddings": {}, "splits": {"dataset": {}}}))
    monkeypatch.setattr(run_embeddings, "run_embedder", _failing)
    with pytest.raises(RuntimeError, match="boom"):
        run_embeddings.run_embedders_for_dataset("ds", ["ecfp", "rxnfp"], _args())
