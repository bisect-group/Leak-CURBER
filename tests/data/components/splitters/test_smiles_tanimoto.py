from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.components.splitters.smiles_tanimoto import (
    SMILESMaxTanimotoSimilaritySplitter,
    _smiles_to_fp_worker,
)


@pytest.mark.slow
def test_splitter_accepts_input_that_already_has_smiles_key() -> None:
    from src.data.components.splitters.smiles_tanimoto import SMILESMaxTanimotoSimilaritySplitter

    splitter = object.__new__(SMILESMaxTanimotoSimilaritySplitter)
    splitter.logger = MagicMock()
    splitter.PANDARALLEL_PROGRESS_BAR, splitter.PANDARALLEL_NB_WORKERS = False, 2
    frame = pd.DataFrame(
        {"smiles": ["N", "[NH4+]", "CCO"], "smiles_key": ["N", "N", "CCO"], "v": [1, 2, 3]}
    )
    df, unique = splitter._clean_full_dataset_df(frame)
    assert df["smiles_key"].tolist() == ["N", "N", "CCO"]
    assert unique["smiles_key"].tolist() == ["N", "CCO"]


def test_keys_sharing_a_canonical_smiles_keep_their_rows(monkeypatch) -> None:
    from src.data.components.splitters import smiles_tanimoto as st

    monkeypatch.setitem(
        sys.modules,
        "pandarallel",
        SimpleNamespace(pandarallel=SimpleNamespace(initialize=lambda **k: None)),
    )
    monkeypatch.setattr(pd.Series, "parallel_apply", pd.Series.apply, raising=False)
    monkeypatch.setattr(
        st.chem_utils, "molecule_key_series", lambda s, fallback: (pd.Series(["k1", "k2"]), 0)
    )
    splitter = object.__new__(st.SMILESMaxTanimotoSimilaritySplitter)
    splitter.logger = logging.getLogger("t")
    splitter.PANDARALLEL_PROGRESS_BAR, splitter.PANDARALLEL_NB_WORKERS = False, 1
    df = pd.DataFrame({"smiles": ["CCO", "OCC"], "y": [1, 2]})
    full, unique = splitter._clean_full_dataset_df(df)
    assert set(full["smiles_key"]) <= set(unique["smiles_key"])
    assert len(full) == 2


_MORGAN_ON_BITS = {
    "C": [240],
    "CC": [33, 251],
    "CCO": [33, 80, 222, 294, 386, 807],
    "c1ccccc1": [64, 389, 849],
    "CC(=O)O": [33, 389, 508, 650, 807, 893, 1017],
    "CC(=O)Oc1ccccc1C(=O)O": [
        11,
        23,
        33,
        64,
        175,
        356,
        386,
        389,
        423,
        444,
        456,
        592,
        650,
        695,
        705,
        726,
        751,
        807,
        849,
        893,
        909,
        946,
        967,
        1017,
    ],
    "C1CCCCC1": [2, 4, 926],
    "CCN(CC)CC": [33, 80, 251, 294, 537, 881, 932, 957],
    "O=C=O": [385, 650, 674, 1016],
    "N#N": [360, 972],
    "CCCCCCCC": [33, 80, 119, 294, 420, 591, 794, 887],
    "c1ccncc1": [64, 131, 378, 383, 389, 437, 579, 842, 849],
    "CC(C)(C)O": [33, 114, 392, 686, 807, 884],
    "OCC(O)CO": [1, 80, 222, 227, 482, 807, 953, 993, 1014],
    "C[N+](C)(C)C": [33, 166, 414, 628],
    "[Na+].[Cl-]": [189, 842],
    "C=CC=C": [40, 342, 621, 694, 784],
    "CS(=O)(=O)C": [33, 46, 350, 452, 650, 692],
    "FC(F)(F)C": [33, 114, 232, 392, 429, 904],
    "c1ccc2ccccc2c1": [63, 64, 175, 356, 598, 726, 831, 849],
}


def test_fp_worker_returns_the_index_and_the_morgan_on_bits() -> None:
    for idx, smiles in enumerate(_MORGAN_ON_BITS):
        got_idx, fp = _smiles_to_fp_worker((idx, smiles))
        assert got_idx == idx
        assert list(fp.GetOnBits()) == _MORGAN_ON_BITS[smiles]


def test_fp_worker_rejects_an_invalid_smiles() -> None:
    with pytest.raises(Exception) as exc:
        _smiles_to_fp_worker((0, "not_a_smiles"))
    assert type(exc.value).__name__ == "ArgumentError"


@pytest.mark.slow
def test_smiles_to_fps_keeps_input_order() -> None:
    fps = SMILESMaxTanimotoSimilaritySplitter._smiles_to_fps(None, list(_MORGAN_ON_BITS))
    assert [list(fp.GetOnBits()) for fp in fps] == list(_MORGAN_ON_BITS.values())


def test_tiny_and_inorganic_molecules_stay_in_train() -> None:
    import numpy as np

    smiles = ["O", "N", "O=C=O", "[Na+].[Cl-]", "OP(=O)(O)O", "CCO", "c1ccccc1", "CCCCCCCC", "CC(=O)O"]
    splitter = object.__new__(SMILESMaxTanimotoSimilaritySplitter)
    splitter.logger = MagicMock()
    splitter.TEST_FRAC = splitter.VAL_FRAC = 0.4
    splitter.full_dataset_df = pd.DataFrame({"smiles_key": smiles * 3})
    splitter.unique_smiles_df = pd.DataFrame({"smiles_key": smiles, "smiles": smiles})
    _, rows, _ = splitter.dissimilarity_split(
        smiles, np.zeros(len(smiles)), {s: 3 for s in smiles}, similarity_threshold=0.5
    )
    tiny = {"O", "N", "O=C=O", "[Na+].[Cl-]", "OP(=O)(O)O", "CCO"}
    assert not tiny & (set(rows["test"]["smiles_key"]) | set(rows["val"]["smiles_key"]))
    assert tiny <= set(rows["train"]["smiles_key"])
    assert len(rows["test"]) > 0


def test_is_tiny_or_inorganic() -> None:
    from src.utils.chem_utils import is_tiny_or_inorganic

    assert all(map(is_tiny_or_inorganic, ["O", "N", "OO", "O=C=O", "OS(=O)(=O)O", "[Zn+2]", "CCO"]))
    assert not any(map(is_tiny_or_inorganic, ["CCCC", "c1ccccc1", "not_a_smiles"]))
