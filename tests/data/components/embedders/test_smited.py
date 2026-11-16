from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.components.embedders import smited
from src.data.components.embedders.smited import SMITEDShardEmbedder


def _embedder() -> SMITEDShardEmbedder:
    embedder = object.__new__(SMITEDShardEmbedder)
    embedder.device = "cpu"
    embedder.max_length = 4
    embedder.tokenized: list[str] = []

    def tokenizer(smiles, **_):
        embedder.tokenized = list(smiles)
        ids = torch.ones(len(smiles), 4, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids.clone()}

    embedder.tokenizer = tokenizer
    return embedder


def test_normalized_smiles_are_tokenized_and_keys_stay_original(monkeypatch) -> None:
    monkeypatch.setattr(smited, "normalize_smiles", lambda s: None if s == "bad" else s.upper())
    embedder = _embedder()
    _, keys, _, failures = embedder._prepare_batch(0, ["c", "bad", ""])
    assert keys == ["c", "bad"]
    assert embedder.tokenized == ["C", "bad"]
    assert [f["raw_key"] for f in failures] == [""]


def test_non_finite_embeddings_become_failures() -> None:
    embedder = _embedder()
    embedder.use_autocast = False
    embedder.model = SimpleNamespace(smi_ted=SimpleNamespace(n_embd=1))
    embedder.encoder = lambda ids, mask: ids.float()
    values = torch.tensor([[1.0, 1.0], [float("nan"), 1.0], [float("inf"), 1.0]])
    embedder.embedding_head = lambda _x: values
    tokenized = {"input_ids": torch.ones(3, 4), "attention_mask": torch.ones(3, 4)}
    arrays, failures = embedder._encode_batch(["a", "b", "c"], tokenized)
    assert list(arrays) == ["a"]
    assert arrays["a"].dtype == np.float16
    assert [f["raw_key"] for f in failures] == ["b", "c"]
    assert all("NaN or inf" in f["error"] for f in failures)


def test_default_autocast_dtype_is_bfloat16() -> None:
    cfg = OmegaConf.load("configs/data/embeddings.yaml")
    dtype = SMITEDShardEmbedder._resolve_autocast_dtype(cfg.smited_embeddings_autocast_dtype)
    assert dtype is torch.bfloat16
