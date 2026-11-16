from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.components.splitters.reaction_tanimoto import (
    ReactionDRFPTanimotoSimilaritySplitter,
    _max_drfp_tanimoto,
)
from tests.helpers.builders import (
    SilentLogger,
    make_embeddings,
    make_splitter,
    reaction_split,
    split_keys,
)


def test_suspicious_reactions_are_retained_in_training() -> None:
    splitter = object.__new__(ReactionDRFPTanimotoSimilaritySplitter)
    splitter.KEY_COLUMN = "rxn_smiles"
    splitter.TEST_FRAC = 0.25
    splitter.MATCH_VAL_TO_TEST = True
    splitter.VAL_FRAC = 0.25
    splitter.unique_keys = ["r0", "r1", "r2", "r3"]
    splitter.full_dataset_df = pd.DataFrame({"rxn_smiles": ["r0", "r1", "r2", "r3"]})
    splitter._valid_embedding_mask = np.array([False, True, True, True])
    splitter._max_tanimoto_array = np.array([0.0, 0.01, 0.2, 0.3])
    splitter._max_tanimoto_to_dataset = dict(
        zip(splitter.unique_keys, splitter._max_tanimoto_array)
    )
    splitter.logger = SilentLogger()
    splitter.compute_max_tanimoto_to_dataset = lambda: splitter._max_tanimoto_array
    splitter.get_max_tanimoto_for_keys = lambda keys: {
        key: splitter._max_tanimoto_to_dataset[key] for key in keys
    }

    splits, _ = splitter.dissimilarity_based_split(0.1)

    assert set(splits["train"]["rxn_smiles"]) == {"r0", "r3"}
    assert set(splits["val"]["rxn_smiles"]) == {"r2"}
    assert set(splits["test"]["rxn_smiles"]) == {"r1"}
    assert sum(len(split) for split in splits.values()) == len(splitter.full_dataset_df)


def test_duplicate_reaction_rows_stay_in_one_split() -> None:
    splitter = object.__new__(ReactionDRFPTanimotoSimilaritySplitter)
    splitter.KEY_COLUMN = "rxn_smiles"
    splitter.TEST_FRAC = 0.4
    splitter.MATCH_VAL_TO_TEST = True
    splitter.VAL_FRAC = 0.4
    splitter.unique_keys = ["r0", "r1", "r2"]
    splitter.full_dataset_df = pd.DataFrame({"rxn_smiles": ["r0", "r0", "r1", "r2"]})
    splitter._valid_embedding_mask = np.array([False, True, True])
    splitter._max_tanimoto_array = np.array([0.0, 0.01, 0.2])
    splitter._max_tanimoto_to_dataset = dict(
        zip(splitter.unique_keys, splitter._max_tanimoto_array)
    )
    splitter.logger = SilentLogger()
    splitter.compute_max_tanimoto_to_dataset = lambda: splitter._max_tanimoto_array
    splitter.get_max_tanimoto_for_keys = lambda keys: {
        key: splitter._max_tanimoto_to_dataset[key] for key in keys
    }

    splits, _ = splitter.dissimilarity_based_split(0.1)

    memberships = {
        key: [name for name, frame in splits.items() if key in set(frame["rxn_smiles"])]
        for key in splitter.unique_keys
    }
    assert all(len(names) == 1 for names in memberships.values())
    assert sum(len(split) for split in splits.values()) == len(splitter.full_dataset_df)


def test_cold_and_warm_splits_match_and_zero_fingerprints_go_to_train(
    tmp_path: Path,
) -> None:
    embs = make_embeddings(16, binary=True)
    cls = ReactionDRFPTanimotoSimilaritySplitter
    cold = split_keys(make_splitter(cls, tmp_path, embs), 0.3)
    warm = split_keys(make_splitter(cls, tmp_path, embs), 0.3)
    assert tmp_path.joinpath("cache.tsv").exists()
    assert cold == warm
    assert {"r0", "r1", "r2"} <= cold["train"]
    assert "r3" in cold["test"] | cold["val"]


def test_tied_similarities_give_the_same_split_for_any_row_order(tmp_path: Path) -> None:
    keys = [f"r{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    shuffled = list(rng.permutation(keys))
    assert reaction_split(tmp_path, keys) == reaction_split(tmp_path, shuffled)


def test_store_fingerprint_changes_when_the_index_changes(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / "meta.json").write_text("{}")
    (store / "index.parquet").write_bytes(b"a")
    splitter = object.__new__(ReactionDRFPTanimotoSimilaritySplitter)
    splitter._embedding_source, splitter._embedding_source_path = "store", store
    splitter.unique_keys, splitter.KEY_COLUMN = ["k"], "k"
    before = splitter._similarity_fingerprint()
    (store / "index.parquet").write_bytes(b"ab")  # a new shard rewrites the index
    assert splitter._similarity_fingerprint() != before


def test_drfp_tanimoto_is_float64() -> None:
    binary = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=np.uint8)
    sims = _max_drfp_tanimoto(binary)
    assert sims.dtype == np.float64
    assert sims[0] == pytest.approx(1 / 3)


@pytest.mark.parametrize("max_batch_cells", [2**25, 7 * 2])
def test_drfp_tanimoto_gives_each_row_its_best_other_row_similarity(max_batch_cells: int) -> None:
    binary = np.array(
        [
            [1, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],  # all-zero rows have union 0 with each other
            [0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],  # duplicate of row 1
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=bool,
    )
    result = _max_drfp_tanimoto(binary, max_batch_cells=max_batch_cells)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [2 / 3, 1.0, 0.0, 1 / 3, 1.0, 0.0, 1 / 3])
