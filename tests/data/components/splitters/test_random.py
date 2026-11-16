from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.components.splitters.random import RandomSplitter


def test_random_split_does_not_depend_on_row_order() -> None:
    def run(frame: pd.DataFrame) -> dict[str, set]:
        splitter = object.__new__(RandomSplitter)
        splitter.RANDOM_SEED, splitter.TRAIN_FRAC = 42, 0.8
        splitter.VALID_FRAC = splitter.TEST_FRAC = 0.1
        splitter.logger = MagicMock()
        splits = splitter._generate_splits_for_column(frame, "g")
        return {name: set(df["g"]) for name, df in splits.items()}

    frame = pd.DataFrame({"g": [f"g{i % 40}" for i in range(200)]})
    assert run(frame) == run(frame.sample(frac=1.0, random_state=1).reset_index(drop=True))


@pytest.mark.slow
def test_random_split_groups_molecule_twins() -> None:
    splitter = object.__new__(RandomSplitter)
    splitter.logger = MagicMock()
    splitter.RANDOM_SEED, splitter.TRAIN_FRAC = 1, 0.5
    splitter.VALID_FRAC = splitter.TEST_FRAC = 0.25
    twins = [("N", "[NH4+]"), ("CC(=O)O", "CC(=O)[O-].[Na+]"), ("CC=O", "C=CO")]
    singles = ["CC", "CCC", "CCCC", "CCCCC", "CO", "CCO"]
    smiles = [s for pair in twins for s in pair] + singles
    splits = splitter._generate_splits_for_column(pd.DataFrame({"smiles": smiles * 3}), "smiles")
    where = {s: name for name, df in splits.items() for s in df["smiles"]}
    assert all(where[a] == where[b] for a, b in twins)


def test_random_split_gives_test_a_mix_of_group_sizes() -> None:
    sizes = [300, 200, 100, 50] + [10] * 150 + [1] * 3000
    frame = pd.DataFrame({"g": [f"g{i}" for i, n in enumerate(sizes) for _ in range(n)]})
    splitter = object.__new__(RandomSplitter)
    splitter.RANDOM_SEED, splitter.TRAIN_FRAC = 42, 0.9
    splitter.VALID_FRAC = splitter.TEST_FRAC = 0.05
    splitter.logger = MagicMock()
    splits = splitter._generate_splits_for_column(frame, "g")
    for name, frac in [("train", 0.9), ("val", 0.05), ("test", 0.05)]:
        assert abs(len(splits[name]) / len(frame) - frac) < 0.02
    for name in ("val", "test"):
        assert splits[name].groupby("g").size().max() > 1


def test_random_split_groups_a_reaction_with_its_reverse() -> None:
    splitter = object.__new__(RandomSplitter)
    splitter.logger = MagicMock()
    splitter.RANDOM_SEED, splitter.TRAIN_FRAC = 1, 0.5
    splitter.VALID_FRAC = splitter.TEST_FRAC = 0.25
    pairs = [(f"A{i}>>B{i}", f"B{i}>>A{i}") for i in range(10)]
    frame = pd.DataFrame({"rxn_key": [k for pair in pairs for k in pair]})
    splits = splitter._generate_splits_for_column(frame, "rxn_key")
    where = {k: name for name, df in splits.items() for k in df["rxn_key"]}
    assert all(where[a] == where[b] for a, b in pairs)
