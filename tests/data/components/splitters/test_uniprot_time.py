from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


def test_time_split_keeps_each_date_in_one_split(tmp_path: Path) -> None:
    from src.data.components.splitters.uniprot_time import UniProtTimeBasedSplitter

    dates = pd.date_range("2020-01-01", periods=10).repeat(3)
    pd.DataFrame({"uniprot_date": dates, "row": range(30)}).to_parquet(tmp_path / "d.parquet")
    splitter = object.__new__(UniProtTimeBasedSplitter)
    splitter.__dict__.update(
        TRAIN_FRAC=0.6,
        VALID_FRAC=0.2,
        TEST_FRAC=0.2,
        DATE_FIELD="date_created",
        DATASET_DATE_COLUMN="uniprot_date",
        MISSING_DATE_HANDLING="train",
        INPUT_DATASET_PATH=tmp_path / "d.parquet",
        OUTPUT_DIR=tmp_path / "out",
        logger=MagicMock(),
    )
    splitter._save_distribution_plots = lambda splits: None
    splitter.save_named_splits = lambda **kwargs: None
    splits = splitter.generate_splits()
    date_sets = {name: set(df["uniprot_date"]) for name, df in splits.items()}
    assert not date_sets["train"] & date_sets["val"]
    assert not date_sets["train"] & date_sets["test"]
    assert not date_sets["val"] & date_sets["test"]
    assert max(date_sets["train"]) < min(date_sets["val"]) < min(date_sets["test"])
    assert sum(len(df) for df in splits.values()) == 30


def test_pairs_shared_across_splits_move_to_the_earliest_split() -> None:
    from src.data.components.splitters.uniprot_time import UniProtTimeBasedSplitter

    splitter = object.__new__(UniProtTimeBasedSplitter)
    splitter.logger = MagicMock()
    df = pd.DataFrame(
        {
            "sequence": ["A", "A", "A", "B", "B", "C"],
            "smiles_key": ["x", "x", "x", "x", "y", "x"],
        }
    )
    ranks = pd.Series([0, 1, 2, 1, 2, 2])
    masks = {name: ranks == i for i, name in enumerate(["train", "val", "test"])}
    splits = splitter._move_shared_groups_to_earliest_split(df, masks)
    assert [len(splits[n]) for n in ("train", "val", "test")] == [3, 1, 2]
    assert set(splits["train"]["sequence"]) == {"A"}


def _move(df: pd.DataFrame, ranks: list[int], group_cols: list[str]) -> dict[str, pd.DataFrame]:
    from src.data.components.splitters.uniprot_time import UniProtTimeBasedSplitter

    splitter = object.__new__(UniProtTimeBasedSplitter)
    splitter.logger = MagicMock()
    splitter.GROUP_COLS = group_cols
    ranks = pd.Series(ranks)
    masks = {name: ranks == i for i, name in enumerate(["train", "val", "test"])}
    return splitter._move_shared_groups_to_earliest_split(df, masks)


def test_group_cols_sequence_moves_every_row_of_a_sequence_to_the_earliest_split() -> None:
    df = pd.DataFrame({"sequence": ["A", "A", "B", "C"], "smiles": ["x", "y", "x", "x"]})
    splits = _move(df, [0, 2, 1, 2], ["sequence"])
    assert [len(splits[n]) for n in ("train", "val", "test")] == [2, 1, 1]
    assert splits["train"]["smiles"].tolist() == ["x", "y"]


def test_group_cols_rxn_key_moves_the_reverse_reaction_too() -> None:
    df = pd.DataFrame({"rxn_key": ["A>>B", "B>>A", "C>>D"]})
    splits = _move(df, [0, 2, 2], ["rxn_key"])
    assert splits["train"]["rxn_key"].tolist() == ["A>>B", "B>>A"]
    assert splits["test"]["rxn_key"].tolist() == ["C>>D"]


def test_group_cols_missing_column_is_an_error() -> None:
    with pytest.raises(ValueError, match="group_cols"):
        _move(pd.DataFrame({"sequence": ["A"]}), [0], ["rxn_key"])
