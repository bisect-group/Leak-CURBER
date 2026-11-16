import sys

import pytest
from omegaconf import OmegaConf

from tests.helpers.builders import load_script

script = load_script("generate_atom_mapped_reactions")


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    class FakeGenerator:
        def __init__(self, cfg):
            seen["cfg"] = cfg

        def generate(self, input_path, output_path, resume, limit):
            seen["generate"] = (input_path, output_path, resume, limit)

    monkeypatch.setattr(script, "AtomMappedReactionGenerator", FakeGenerator)
    return seen


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["generate_atom_mapped_reactions.py", *argv])
    script.main()


def test_main_fails_when_the_config_file_is_missing(monkeypatch, tmp_path, captured):
    with pytest.raises(FileNotFoundError, match="LocalMapper configuration"):
        _run(monkeypatch, "--input-path", "in.parquet", "--config", str(tmp_path / "none.yaml"))


def test_main_applies_only_given_overrides_to_the_config(monkeypatch, tmp_path, captured):
    config = tmp_path / "emb.yaml"
    OmegaConf.save({"localmapper_batch_size": 8, "localmapper_queue_size": 4}, config)
    _run(
        monkeypatch,
        "--input-path", "in.parquet",
        "--config", str(config),
        "--batch-size", "32",
        "--gpu-ids", "0", "2",
        "--resume",
        "--limit", "5",
    )  # fmt: skip
    embeddings = captured["cfg"].embeddings
    assert embeddings.localmapper_batch_size == 32
    assert embeddings.localmapper_queue_size == 4
    assert list(embeddings.localmapper_gpu_ids) == [0, 2]
    assert "localmapper_model_version" not in embeddings
    input_path, output_path, resume, limit = captured["generate"]
    assert (str(input_path), output_path, resume, limit) == ("in.parquet", None, True, 5)
