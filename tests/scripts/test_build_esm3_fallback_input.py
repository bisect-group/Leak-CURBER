import sys
import json
import pickle

import pandas as pd

from src.data.components.embedders.utils import sha256_short
from tests.helpers.builders import load_script

script = load_script("build_esm3_fallback_input")


def _record(sequence: str, *acc_ids: str) -> dict:
    return {
        "sequence": sequence,
        "uniprot_id": list(acc_ids),
        "pdbs": list(acc_ids),
        "pdb_type": None,
    }


def test_fallback_records_are_failures_that_did_not_fold(tmp_path, capsys):
    out_dir = tmp_path / "esmfold2_cif"
    (out_dir / ".esmfold2_rows").mkdir(parents=True)
    records = [
        _record("AAA", "P1"),
        _record("CCC", "P2"),
        _record("DDD", "P3"),
        _record("EEE", "P4"),
        _record("GGG", "P5"),
    ]
    pd.DataFrame({"sequence_hash": [sha256_short("AAA"), sha256_short("CCC")]}).to_parquet(
        out_dir / "failures.parquet"
    )
    lines = [
        {"kind": "failure", "row": {"sequence_hash": sha256_short("DDD")}},
        {"kind": "manifest", "row": {"sequence_hash": sha256_short("GGG")}},
    ]
    (out_dir / ".esmfold2_rows" / "gpu0.jsonl").write_text(
        "\n".join(map(json.dumps, lines)) + "\n{cut"
    )
    pd.DataFrame({"acc_id": ["P1"]}).to_parquet(out_dir / "manifest.parquet")
    (out_dir / "ESMFold2-P2.cif").touch()
    input_pkl, output_pkl = tmp_path / "in.pkl", tmp_path / "out.pkl"
    input_pkl.write_bytes(pickle.dumps(records))

    sys.argv = [
        "x",
        "--esmfold2-dir",
        str(out_dir),
        "--input-pkl",
        str(input_pkl),
        "--output-pkl",
        str(output_pkl),
    ]
    script.main()

    assert pickle.loads(output_pkl.read_bytes()) == [records[2]]
    assert "Wrote 1 records" in capsys.readouterr().out
