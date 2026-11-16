def test_sequence_seed_is_stable_and_sequence_specific() -> None:
    from src.data.components.generators.esm3_pdb import sequence_seed

    assert sequence_seed("MKV") == sequence_seed("MKV")
    assert sequence_seed("MKV") != sequence_seed("MKA")


def test_only_records_without_a_structure_are_generated() -> None:
    from src.data.components.generators.esm3_pdb import proteins_without_structure

    records = [
        {
            "sequence": "MKV",
            "uniprot_id": ["P1", "P1|R2Q"],
            "pdbs": ["P1", "P1|R2Q"],
            "pdb_type": None,
        },
        {"sequence": "MKA", "uniprot_id": ["P2"], "pdbs": ["1ABC"], "pdb_type": "experimental"},
        {"sequence": "MKL", "uniprot_id": ["P3"], "pdbs": ["P3"], "pdb_type": "predicted"},
    ]

    assert proteins_without_structure(records) == [
        {"acc_id": "P1", "sequence": "MKV"},
        {"acc_id": "P1|R2Q", "sequence": "MKV"},
    ]


def test_out_of_memory_skips_one_sequence_and_continues(tmp_path, caplog) -> None:
    import logging
    from types import SimpleNamespace

    import torch

    from src.data.components.generators.esm3_pdb import ESM3PDBGenerator

    class FakeClient:
        def generate(self, protein, config):
            if protein.sequence == "MKV":
                raise torch.cuda.OutOfMemoryError("fake")
            return SimpleNamespace(to_pdb=lambda path: path.write_text("ATOM"))

    logger = logging.getLogger("esm3_oom_test")
    logger.propagate = True
    generator = ESM3PDBGenerator.__new__(ESM3PDBGenerator)
    generator.logger, generator.ESM3_PDB_PATH = logger, tmp_path
    with caplog.at_level(logging.WARNING, logger="esm3_oom_test"):
        for acc_id, sequence in [("P1", "MKV"), ("P2", "MKA")]:
            generator.generate_structure(
                FakeClient(), {"acc_id": acc_id, "sequence": sequence}, 0, use_cpu=True
            )

    assert not (tmp_path / "ESM3-open-small-P1.pdb").exists()
    assert (tmp_path / "ESM3-open-small-P2.pdb").exists()
    assert "P1 (length 3): CUDA out of memory" in caplog.text
