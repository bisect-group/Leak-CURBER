from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.data.components.embedders.rxnfp import RxnFPShardEmbedder


class _FakeEmbedder(RxnFPShardEmbedder):
    """A 4-wide fingerprint; `bad` reactions raise."""

    def __init__(self, max_len: int = 512) -> None:
        self.max_len = max_len
        self.tokenizer = SimpleNamespace(tokenize=lambda rxn: rxn.split("."))
        self.batches: list[list[str]] = []

    def _fingerprints(self, batch):
        self.batches.append(list(batch))
        if any("bad" in rxn for rxn in batch):
            raise ValueError("cannot tokenize")
        return [[float(len(rxn))] * 4 for rxn in batch]


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _embedder(batch_size: int = 2, max_len: int = 512) -> RxnFPShardEmbedder:
    embedder = _FakeEmbedder(max_len)
    embedder.batch_size = batch_size
    embedder.logger = _Logger()
    return embedder


def test_reactions_are_embedded_in_batches_as_float16() -> None:
    embedder = _embedder(batch_size=2)
    arrays, failures = embedder.compute_many(["A>>B", "C>>D", "E>>F"])
    assert failures == []
    assert embedder.batches == [["A>>B", "C>>D"], ["E>>F"]]
    assert list(arrays) == ["A>>B", "C>>D", "E>>F"]
    for fingerprint in arrays.values():
        assert fingerprint.shape == (4,)
        assert fingerprint.dtype == np.float16
    assert arrays["A>>B"].tolist() == [4.0] * 4


def test_a_failing_batch_is_retried_row_by_row() -> None:
    embedder = _embedder(batch_size=3)
    arrays, failures = embedder.compute_many(["A>>B", "bad>>C", "D>>E"])
    assert list(arrays) == ["A>>B", "D>>E"]
    assert failures == [
        {"raw_key": "bad>>C", "canonical_key": "bad>>C", "error": "cannot tokenize"}
    ]
    assert embedder.batches[1:] == [["A>>B"], ["bad>>C"], ["D>>E"]]


def test_reactions_longer_than_the_model_limit_are_counted_in_a_warning() -> None:
    embedder = _embedder(batch_size=5, max_len=4)
    embedder.compute_many(["a.b", "a.b.c.d", "a"])
    # "a.b" is 2 tokens + 2 = 4, which fits; only the 4-token reaction exceeds the limit.
    assert embedder.logger.warnings == ["1 of 3 reactions exceed 4 tokens and were truncated."]


_SHORT = "CC(=O)C(=O)O.C=O>>O=C([O-])C(=O)CCO"
_SHORT_FP = [-1.2533893585205078, -0.7602770924568176, 1.2170357704162598, -1.1466985940933228]
# 515 tokens, so it is truncated at the model limit.
_LONG = (
    "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\COP(=O)([O-])OP(=O)([O-])O[C@H]1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O[C@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]3O[C@@H]3O[C@H](C(=O)[O-])[C@@H](O)[C@H](O)[C@H]3O)[C@H]2O)[C@H](O)[C@H]1O.O=c1ccn([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]2O)c(=O)[nH]1.[H+]"
    ">>CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\COP(=O)([O-])OP(=O)([O-])O[C@H]1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O[C@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]3O)[C@H]2O)[C@H](O)[C@H]1O.O=C([O-])[C@H]1O[C@H](OP(=O)([O-])OP(=O)([O-])OC[C@H]2O[C@@H](n3ccc(=O)[nH]c3=O)[C@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"
)
_LONG_FP = [-1.5066906213760376, -0.03598661720752716, -0.8337748646736145, -2.459606647491455]


@pytest.fixture(scope="module")
def real_embedder() -> RxnFPShardEmbedder:
    pytest.importorskip("rxnfp")
    embedder = object.__new__(RxnFPShardEmbedder)
    embedder.batch_size = 2
    embedder.logger = _Logger()
    from importlib.resources import files

    from rxnfp.tokenization import RegexTokenizer
    from transformers import BertModel

    path = files("rxnfp") / "models/transformers/bert_ft"
    embedder.device = "cpu"
    embedder.model = BertModel.from_pretrained(str(path)).eval()
    embedder.max_len = embedder.model.config.max_position_embeddings
    embedder.vocab = {t: i for i, t in enumerate((path / "vocab.txt").read_text().splitlines())}
    embedder.tokenizer = RegexTokenizer()
    return embedder


@pytest.mark.parametrize(("rxn", "expected"), [(_SHORT, _SHORT_FP), (_LONG, _LONG_FP)])
def test_fingerprints_match_rxnfp_reference(real_embedder, rxn, expected) -> None:
    arrays, failures = real_embedder.compute_many([rxn])
    assert failures == []
    assert arrays[rxn].shape == (256,)
    assert arrays[rxn].dtype == np.float16
    np.testing.assert_allclose(arrays[rxn][:4], expected, atol=1e-3)


def test_padding_does_not_change_fingerprints(real_embedder) -> None:
    together = real_embedder._fingerprints([_SHORT, _LONG])
    alone = [real_embedder._fingerprints([rxn])[0] for rxn in (_SHORT, _LONG)]
    np.testing.assert_allclose(together, alone, atol=1e-4)
