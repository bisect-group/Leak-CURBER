from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from src.data.components.generators import esmfold2_structure as esmfold2
from src.data.components.generators.esmfold2_structure import (
    ESMFold2StructureGenerator,
    ESMFold2WorkerConfig,
    ProteinStructureRecord,
    _deduplicate_by_sequence,
    _fold_records_on_gpu,
    _load_optional_msa,
    _load_protein_records,
    _make_structure_prediction_input,
)


def test_load_protein_records_requires_accession_and_sequence(tmp_path: Path) -> None:
    input_path = tmp_path / "records.pkl"
    with open(input_path, "wb") as handle:
        pickle.dump(
            [
                {"acc_id": "B", "sequence": "CCC"},
                {"acc_id": "A", "sequence": "AAA"},
            ],
            handle,
        )

    records = _load_protein_records(input_path)

    assert records == [
        ProteinStructureRecord(acc_id="A", sequence="AAA", uniprot_ids=("A",)),
        ProteinStructureRecord(acc_id="B", sequence="CCC", uniprot_ids=("B",)),
    ]


def test_load_protein_records_accepts_sequence_with_pdbs_schema(tmp_path: Path) -> None:
    input_path = tmp_path / "records.pkl"
    with open(input_path, "wb") as handle:
        pickle.dump(
            [
                {
                    "sequence": "AAA",
                    "uniprot_id": {"B", "A"},
                    "pdbs": {"2BBB", "1AAA"},
                    "pdb_type": "experimental",
                }
            ],
            handle,
        )

    records = _load_protein_records(input_path)

    assert records == [
        ProteinStructureRecord(
            acc_id="A",
            sequence="AAA",
            uniprot_ids=("A", "B"),
            source_pdbs=("1AAA", "2BBB"),
            pdb_type="experimental",
        )
    ]


def test_load_protein_records_accepts_superset_lists(tmp_path: Path) -> None:
    input_path = tmp_path / "records.pkl"
    record = {
        "sequence": "AAA",
        "uniprot_id": ["A", "A|R1Q"],
        "pdbs": ["1AAA", "2BBB"],
        "pdb_type": "experimental",
    }
    input_path.write_bytes(pickle.dumps([record]))

    assert _load_protein_records(input_path) == [
        ProteinStructureRecord(
            acc_id="A",
            sequence="AAA",
            uniprot_ids=("A", "A|R1Q"),
            source_pdbs=("1AAA", "2BBB"),
            pdb_type="experimental",
        )
    ]


def test_load_protein_records_rejects_missing_keys(tmp_path: Path) -> None:
    input_path = tmp_path / "records.pkl"
    with open(input_path, "wb") as handle:
        pickle.dump([{"acc_id": "A"}], handle)

    with pytest.raises(ValueError, match="acc_id.*sequence"):
        _load_protein_records(input_path)


def test_deduplicate_by_sequence_keeps_first_accession() -> None:
    records = [
        ProteinStructureRecord(acc_id="A", sequence="AAA", uniprot_ids=("A",)),
        ProteinStructureRecord(acc_id="B", sequence="AAA", uniprot_ids=("B",)),
        ProteinStructureRecord(acc_id="C", sequence="CCC", uniprot_ids=("C",)),
    ]

    unique_records, by_sequence = _deduplicate_by_sequence(records)

    assert unique_records == [
        ProteinStructureRecord(acc_id="A", sequence="AAA", uniprot_ids=("A", "B")),
        ProteinStructureRecord(acc_id="C", sequence="CCC", uniprot_ids=("C",)),
    ]
    assert by_sequence["AAA"] == records[:2]


def test_optional_msa_loads_a3m_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    msa_dir = tmp_path / "msa"
    msa_dir.mkdir()
    (msa_dir / "P12345.a3m").write_text(">query\nACD\n")

    class FakeMSA:
        @classmethod
        def from_a3m(cls, path, remove_insertions, max_sequences):
            return {
                "path": str(path),
                "remove_insertions": remove_insertions,
                "max_sequences": max_sequences,
            }

    fake_esm = types.ModuleType("esm")
    fake_utils = types.ModuleType("esm.utils")
    fake_msa = types.ModuleType("esm.utils.msa")
    fake_msa.MSA = FakeMSA
    monkeypatch.setitem(sys.modules, "esm", fake_esm)
    monkeypatch.setitem(sys.modules, "esm.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "esm.utils.msa", fake_msa)

    msa, msa_path = _load_optional_msa(str(msa_dir), "P12345", 1000)

    assert msa == {
        "path": str(msa_dir / "P12345.a3m"),
        "remove_insertions": True,
        "max_sequences": 1000,
    }
    assert msa_path == str(msa_dir / "P12345.a3m")


def test_make_structure_prediction_input_builds_single_protein_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProteinInput:
        def __init__(self, id, sequence, msa=None):
            self.id = id
            self.sequence = sequence
            self.msa = msa

    class FakeStructurePredictionInput:
        def __init__(self, sequences):
            self.sequences = sequences

    fake_esm = types.ModuleType("esm")
    fake_models = types.ModuleType("esm.models")
    fake_esmfold2 = types.ModuleType("esm.models.esmfold2")
    fake_esmfold2.ProteinInput = FakeProteinInput
    fake_esmfold2.StructurePredictionInput = FakeStructurePredictionInput
    monkeypatch.setitem(sys.modules, "esm", fake_esm)
    monkeypatch.setitem(sys.modules, "esm.models", fake_models)
    monkeypatch.setitem(sys.modules, "esm.models.esmfold2", fake_esmfold2)

    structure_input = _make_structure_prediction_input("ACD", msa="msa")

    assert len(structure_input.sequences) == 1
    protein = structure_input.sequences[0]
    assert protein.id == "A"
    assert protein.sequence == "ACD"
    assert protein.msa == "msa"


def test_fold_records_writes_manifest_and_cif(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProteinInput:
        def __init__(self, id, sequence, msa=None):
            self.id = id
            self.sequence = sequence
            self.msa = msa

    class FakeStructurePredictionInput:
        def __init__(self, sequences):
            self.sequences = sequences

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return cls(model_id=model_id, kwargs=kwargs)

        def __init__(self, model_id, kwargs):
            self.model_id = model_id
            self.kwargs = kwargs

        def eval(self):
            return self

    class FakeComplex:
        def to_mmcif(self):
            return "data_fake\n"

        def to_protein_complex(self):
            return self

        def normalize_chain_ids_for_pdb(self):
            return self

        def to_pdb(self, path):
            Path(path).write_text("HEADER FAKE\n")

    class FakeBuilder:
        def fold(self, model, structure_input, **kwargs):
            assert model.model_id == "biohub/ESMFold2"
            assert model.kwargs == {"device": torch.device("cuda:0"), "dtype": torch.bfloat16}
            assert structure_input.sequences[0].id == "A"
            assert structure_input.sequences[0].sequence == "ACD"
            assert kwargs["num_loops"] == 1
            return SimpleNamespace(
                complex=FakeComplex(),
                plddt=torch.tensor([90.0, 80.0]),
                ptm=torch.tensor(0.7),
                iptm=torch.tensor(0.6),
            )

    fake_esm = types.ModuleType("esm")
    fake_models = types.ModuleType("esm.models")
    fake_esmfold2 = types.ModuleType("esm.models.esmfold2")
    fake_esmfold2.ESMFold2InputBuilder = FakeBuilder
    fake_esmfold2.EsmFold2Model = FakeModel
    fake_esmfold2.ProteinInput = FakeProteinInput
    fake_esmfold2.StructurePredictionInput = FakeStructurePredictionInput
    monkeypatch.setitem(sys.modules, "esm", fake_esm)
    monkeypatch.setitem(sys.modules, "esm.models", fake_models)
    monkeypatch.setitem(sys.modules, "esm.models.esmfold2", fake_esmfold2)

    monkeypatch.setattr(torch.cuda, "set_device", lambda _gpu_id: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    worker_cfg = ESMFold2WorkerConfig(
        model_id="biohub/ESMFold2",
        cif_path=str(tmp_path / "cif"),
        pdb_path=str(tmp_path / "pdb"),
        msa_dir=None,
        msa_max_sequences=1000,
        num_loops=1,
        num_sampling_steps=1,
        num_diffusion_samples=1,
        seed=0,
        write_pdb_compat=True,
        torch_dtype="bfloat16",
    )

    manifest_rows, failure_rows = _fold_records_on_gpu(
        0,
        [ProteinStructureRecord(acc_id="P12345", sequence="ACD")],
        worker_cfg,
    )

    assert failure_rows == []
    assert len(manifest_rows) == 1
    row = manifest_rows[0]
    assert row["acc_id"] == "P12345"
    assert row["cif_path"].endswith("ESMFold2-P12345.cif")
    assert row["pdb_path"].endswith("ESMFold2-P12345.pdb")
    assert row["plddt_mean"] == 85.0
    assert row["ptm"] == pytest.approx(0.7)
    assert row["iptm"] == pytest.approx(0.6)
    assert Path(row["cif_path"]).read_text() == "data_fake\n"
    assert Path(row["pdb_path"]).read_text() == "HEADER FAKE\n"


def _make_test_generator(
    tmp_path: Path,
    input_path: Path,
    *,
    skip_known_structures: bool = False,
) -> ESMFold2StructureGenerator:
    generator = ESMFold2StructureGenerator.__new__(ESMFold2StructureGenerator)
    generator.input_path = input_path
    generator.cif_path = tmp_path / "cif"
    generator.pdb_path = tmp_path / "pdb"
    generator.manifest_path = tmp_path / "cif" / "manifest.parquet"
    generator.failures_path = tmp_path / "cif" / "failures.parquet"
    generator.cif_path.mkdir()
    generator.pdb_path.mkdir()
    generator.cfg = SimpleNamespace(
        embeddings=SimpleNamespace(
            esmfold2_model_id="biohub/ESMFold2",
            esmfold2_skip_known_structures=skip_known_structures,
        )
    )
    generator.logger = SimpleNamespace(info=lambda _message: None)
    return generator


def test_generate_folds_once_and_copies_to_all_uniprot_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "records.pkl"
    with open(input_path, "wb") as handle:
        pickle.dump(
            [
                {
                    "sequence": "AAA",
                    "uniprot_id": {"B", "A"},
                    "pdbs": {"A"},
                    "pdb_type": "predicted",
                }
            ],
            handle,
        )
    generator = _make_test_generator(tmp_path, input_path)
    folded_records = []

    def fake_run_workers(records, _worker_cfg):
        folded_records.extend(records)
        assert len(records) == 1
        assert records[0].acc_id == "A"
        (generator.cif_path / "ESMFold2-A.cif").write_text("data_a\n")
        (generator.pdb_path / "ESMFold2-A.pdb").write_text("HEADER A\n")

    monkeypatch.setattr(generator, "_run_workers", fake_run_workers)
    monkeypatch.setattr(
        generator,
        "_worker_config",
        lambda: ESMFold2WorkerConfig(
            model_id="biohub/ESMFold2",
            cif_path=str(generator.cif_path),
            pdb_path=str(generator.pdb_path),
            msa_dir=None,
            msa_max_sequences=1000,
            num_loops=1,
            num_sampling_steps=1,
            num_diffusion_samples=1,
            seed=0,
            write_pdb_compat=True,
            torch_dtype="bfloat16",
        ),
    )

    generator.generate()

    assert len(folded_records) == 1
    assert (generator.cif_path / "ESMFold2-B.cif").read_text() == "data_a\n"
    assert (generator.pdb_path / "ESMFold2-B.pdb").read_text() == "HEADER A\n"
    manifest = pd.read_parquet(generator.manifest_path)
    assert manifest.iloc[-1]["status"] == "copied_from_sequence_canonical"
    assert manifest.iloc[-1]["source_acc_id"] == "A"


def test_generate_existing_canonical_output_creates_missing_uniprot_copies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "records.pkl"
    with open(input_path, "wb") as handle:
        pickle.dump(
            [
                {
                    "sequence": "AAA",
                    "uniprot_id": {"B", "A"},
                    "pdbs": set(),
                    "pdb_type": None,
                }
            ],
            handle,
        )
    generator = _make_test_generator(tmp_path, input_path)
    (generator.cif_path / "ESMFold2-A.cif").write_text("data_a\n")
    (generator.pdb_path / "ESMFold2-A.pdb").write_text("HEADER A\n")
    monkeypatch.setattr(generator, "_run_workers", lambda *_args: pytest.fail("refolded"))

    generator.generate()

    assert (generator.cif_path / "ESMFold2-B.cif").read_text() == "data_a\n"
    assert (generator.pdb_path / "ESMFold2-B.pdb").read_text() == "HEADER A\n"
    statuses = set(pd.read_parquet(generator.manifest_path)["status"])
    assert {"existing_output", "copied_from_sequence_canonical"} <= statuses


def test_generate_can_skip_known_structures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "records.pkl"
    with open(input_path, "wb") as handle:
        pickle.dump(
            [
                {
                    "sequence": "AAA",
                    "uniprot_id": {"A"},
                    "pdbs": {"A"},
                    "pdb_type": "predicted",
                }
            ],
            handle,
        )
    generator = _make_test_generator(tmp_path, input_path, skip_known_structures=True)
    monkeypatch.setattr(generator, "_run_workers", lambda *_args: pytest.fail("folded"))

    generator.generate()

    manifest = pd.read_parquet(generator.manifest_path)
    assert manifest.iloc[-1]["status"] == "skipped_existing_structure"
    assert not (generator.cif_path / "ESMFold2-A.cif").exists()


def test_generate_folds_known_structures_when_skip_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "records.pkl"
    with open(input_path, "wb") as handle:
        pickle.dump(
            [
                {
                    "sequence": "AAA",
                    "uniprot_id": {"A"},
                    "pdbs": {"A"},
                    "pdb_type": "predicted",
                }
            ],
            handle,
        )
    generator = _make_test_generator(tmp_path, input_path, skip_known_structures=False)
    folded_records = []

    def fake_run_workers(records, _worker_cfg):
        folded_records.extend(records)
        (generator.cif_path / "ESMFold2-A.cif").write_text("data_a\n")

    monkeypatch.setattr(generator, "_run_workers", fake_run_workers)
    monkeypatch.setattr(
        generator,
        "_worker_config",
        lambda: ESMFold2WorkerConfig(
            model_id="biohub/ESMFold2",
            cif_path=str(generator.cif_path),
            pdb_path=str(generator.pdb_path),
            msa_dir=None,
            msa_max_sequences=1000,
            num_loops=1,
            num_sampling_steps=1,
            num_diffusion_samples=1,
            seed=0,
            write_pdb_compat=True,
            torch_dtype="bfloat16",
        ),
    )

    generator.generate()

    assert len(folded_records) == 1


def test_raw_ids_that_sanitize_alike_raise(tmp_path: Path) -> None:
    path = tmp_path / "in.pkl"
    items = [
        {"acc_id": "A/1", "sequence": "MK", "uniprot_id": ["A/1"]},
        {"acc_id": "A_1", "sequence": "MR", "uniprot_id": ["A_1"]},
    ]
    path.write_bytes(pickle.dumps(items))
    if esmfold2._sanitize_output_id("A/1") != esmfold2._sanitize_output_id("A_1"):
        pytest.skip("sanitizer keeps these IDs apart")
    with pytest.raises(ValueError, match="same output name"):
        esmfold2._load_protein_records(path)


def test_manifest_rows_are_unique_by_id(tmp_path: Path) -> None:
    path = tmp_path / "manifest.parquet"
    esmfold2._append_rows(path, [{"acc_id": "a", "v": 1}, {"acc_id": "b", "v": 1}])
    esmfold2._append_rows(path, [{"acc_id": "a", "v": 2}])
    out = pd.read_parquet(path).sort_values("acc_id")
    assert out["acc_id"].tolist() == ["a", "b"]
    assert out["v"].tolist() == [2, 1]


def test_rows_survive_a_killed_worker(tmp_path: Path) -> None:
    rows_dir = tmp_path / "rows"
    esmfold2._append_row_line(str(rows_dir), 0, "manifest", {"acc_id": "a", "plddt_mean": 0.9})
    esmfold2._append_row_line(str(rows_dir), 1, "failure", {"acc_id": "b", "error": "oom"})
    with open(rows_dir / "gpu0.jsonl", "a") as handle:
        handle.write('{"kind": "manifest", "row": {"acc')  # cut off by the kill
    manifest, failures = tmp_path / "m.parquet", tmp_path / "f.parquet"
    assert esmfold2._merge_incremental_rows(rows_dir, manifest, failures) == (1, 1)
    assert pd.read_parquet(manifest)["acc_id"].tolist() == ["a"]
    assert pd.read_parquet(failures)["acc_id"].tolist() == ["b"]
    assert esmfold2._merge_incremental_rows(rows_dir, manifest, failures) == (0, 0)


def test_generate_leaves_sequences_over_the_cap_unfolded_and_logs_them(tmp_path: Path) -> None:
    input_path = tmp_path / "records.pkl"
    with open(input_path, "wb") as handle:
        pickle.dump([{"acc_id": "S", "sequence": "AAA"}, {"acc_id": "L", "sequence": "AAAAAA"}], handle)
    generator = _make_test_generator(tmp_path, input_path)
    generator.cfg.embeddings.esmfold2_max_sequence_length = 4
    messages: list[str] = []
    generator.logger = SimpleNamespace(info=messages.append)
    folded: list[str] = []
    generator._run_workers = lambda records, _cfg: folded.extend(r.acc_id for r in records)
    generator._worker_config = lambda: None

    generator.generate()

    assert folded == ["S"]
    assert any("1 sequences longer than 4" in message for message in messages)
    assert not generator.manifest_path.exists() and not generator.failures_path.exists()


def test_split_by_length_balances_the_work() -> None:
    records = [ProteinStructureRecord(acc_id=str(n), sequence="A" * n) for n in (100, 90, 50, 40, 30, 20, 10, 5)]

    parts = esmfold2._split_by_length(records, 2)

    loads = [sum(len(r.sequence) ** 2 for r in part) for part in parts]
    assert sorted(r.acc_id for part in parts for r in part) == sorted(r.acc_id for r in records)
    assert max(loads) / min(loads) < 1.1


def test_fold_records_records_an_out_of_memory_failure_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return cls()

        def eval(self):
            return self

    class FakeComplex:
        def to_mmcif(self):
            return "data_fake\n"

    class FakeBuilder:
        def fold(self, model, structure_input, **kwargs):
            if len(structure_input.sequences[0].sequence) > 3:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return SimpleNamespace(complex=FakeComplex(), plddt=None, ptm=None, iptm=None)

    freed: list[int] = []
    monkeypatch.setattr(esmfold2, "_ensure_transformers_vision_imports", lambda: None)
    monkeypatch.setattr(esmfold2, "_make_structure_prediction_input", lambda seq, msa=None: SimpleNamespace(sequences=[SimpleNamespace(sequence=seq)]))
    for name, attrs in {
        "esm.models.esmfold2": {"ESMFold2InputBuilder": FakeBuilder, "EsmFold2Model": FakeModel},
    }.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _gpu_id: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: freed.append(1))
    worker_cfg = ESMFold2WorkerConfig(
        model_id="m", cif_path=str(tmp_path / "cif"), pdb_path=str(tmp_path / "pdb"),
        msa_dir=None, msa_max_sequences=1, num_loops=1, num_sampling_steps=1,
        num_diffusion_samples=1, seed=0, write_pdb_compat=False, torch_dtype=None,
    )

    manifest_rows, failure_rows = _fold_records_on_gpu(
        0,
        [ProteinStructureRecord(acc_id="BIG", sequence="AAAAA"), ProteinStructureRecord(acc_id="OK", sequence="AAA")],
        worker_cfg,
    )

    assert [row["acc_id"] for row in failure_rows] == ["BIG"]
    assert failure_rows[0]["error_type"] == "OutOfMemoryError"
    assert [row["acc_id"] for row in manifest_rows] == ["OK"]
    assert len(freed) >= 3
