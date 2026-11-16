import sys

from src.data.components.generators import esmfold2_structure
from tests.helpers.builders import load_script

script = load_script("generate_esmfold2_structures")


def _run_main(monkeypatch, *argv: str) -> tuple[list, list[str]]:
    seen, calls = [], []

    class FakeGenerator:
        def __init__(self, cfg):
            seen.append(cfg)

        def generate(self):
            calls.append("generate")

        def run_smoke_test(self):
            calls.append("smoke")

    monkeypatch.setattr(esmfold2_structure, "ESMFold2StructureGenerator", FakeGenerator)
    monkeypatch.setattr(sys, "argv", ["generate_esmfold2_structures.py", *argv])
    script.main()
    return seen, calls


def test_main_composes_the_real_config_with_defaults(monkeypatch):
    seen, calls = _run_main(monkeypatch)
    embeddings = seen[0].embeddings
    assert calls == ["generate"]
    assert embeddings.esmfold2_max_sequence_length is None
    assert embeddings.esmfold2_skip_known_structures is False


def test_main_applies_flags_and_overrides(monkeypatch):
    seen, calls = _run_main(
        monkeypatch,
        "--gpu-ids", "2", "3",
        "--max-length", "600",
        "--smoke",
        "--overrides", "embeddings.esmfold2_skip_known_structures=true",
    )
    embeddings = seen[0].embeddings
    assert calls == ["smoke"]
    assert list(embeddings.gpu_ids) == [2, 3]
    assert embeddings.esmfold2_max_sequence_length == 600
    assert embeddings.esmfold2_skip_known_structures is True
