from __future__ import annotations

from pathlib import Path

from src.data.components.splitters.embedding_cosine import EmbeddingCosineSimilaritySplitter
from tests.helpers.builders import make_embeddings, make_splitter, split_keys


def test_cold_and_warm_splits_match_and_zero_vectors_go_to_train(
    tmp_path: Path,
) -> None:
    embs = make_embeddings(16, binary=False)
    cls = EmbeddingCosineSimilaritySplitter
    cold = split_keys(make_splitter(cls, tmp_path, embs), 0.3)
    warm = split_keys(make_splitter(cls, tmp_path, embs), 0.3)
    assert cold == warm
    assert {"r0", "r1", "r2"} <= cold["train"]


def test_splitter_logs_the_keys_sent_to_train(tmp_path: Path) -> None:
    splitter = make_splitter(
        EmbeddingCosineSimilaritySplitter, tmp_path, make_embeddings(16, binary=False)
    )
    splitter.dissimilarity_based_split(0.3)
    logged = " ".join(str(c.args[0]) for c in splitter.logger.info.call_args_list)
    assert "First keys sent to train" in logged and "r2" in logged


def test_val_is_sized_from_val_frac_not_test_size(tmp_path: Path) -> None:
    splitter = make_splitter(
        EmbeddingCosineSimilaritySplitter,
        tmp_path,
        make_embeddings(16, binary=False),
        MATCH_VAL_TO_TEST=False,
        TEST_FRAC=0.3,
        VAL_FRAC=0.1,
    )
    splits, _ = splitter.dissimilarity_based_split(0.3)
    assert len(splits["val"]) <= int(0.1 * 14) < len(splits["test"])
