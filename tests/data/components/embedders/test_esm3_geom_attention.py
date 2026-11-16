from __future__ import annotations

import hashlib
import inspect
import re

import pytest
import torch

pytest.importorskip("esm")

from esm.layers.geom_attention import GeometricReasoningOriginalImpl  # noqa: E402
from esm.utils.structure.affine3d import build_affine3d_from_coordinates  # noqa: E402

from src.data.components.embedders import esm3  # noqa: E402

# Captured at import so the GPU-worker patch cannot affect the comparison.
_ORIGINAL_FORWARD = GeometricReasoningOriginalImpl.__dict__["forward"]
_ESM_3_4_1_FORWARD_SHA256 = "c32e4daf7d8d27d8693c886593fe913161804c19bc77b6a2d6fde7713ba543c0"

B, L, C_S = 2, 40, 64


def _inputs(with_sequence_id: bool):
    coords = torch.randn(B, L, 3, 3) * 5
    affine, mask = build_affine3d_from_coordinates(coords)
    mask[1, 30:] = False
    s = torch.randn(B, L, C_S)
    sequence_id = torch.zeros(B, L, dtype=torch.int64) if with_sequence_id else None
    chain_id = torch.zeros(B, L, dtype=torch.int64)
    chain_id[:, L // 2 :] = 1
    return s, affine, mask, sequence_id, chain_id


def test_esm3_geometric_attention_source_is_unchanged():
    source = re.sub(r"\s+", " ", inspect.getsource(_ORIGINAL_FORWARD)).strip()
    assert hashlib.sha256(source.encode()).hexdigest() == _ESM_3_4_1_FORWARD_SHA256, (
        "ESM3 geometric attention changed; re-check _chunked_geometric_attention"
    )


@pytest.mark.parametrize("with_sequence_id", [True, False])
def test_chunked_geometric_attention_matches_original(monkeypatch, with_sequence_id):
    assert _ORIGINAL_FORWARD is not esm3._chunked_geometric_attention
    torch.manual_seed(0)
    module = GeometricReasoningOriginalImpl(c_s=C_S, v_heads=8).eval()
    for param in module.parameters():
        torch.nn.init.normal_(param, std=0.3)
    args = _inputs(with_sequence_id)

    with torch.no_grad():
        expected = _ORIGINAL_FORWARD(module, *args)

        # 5 query rows per chunk -> 8 chunks.
        monkeypatch.setattr(esm3, "_GEOM_ATTN_CHUNK_ELEMENTS", B * 8 * L * 3 * 5)
        softmax_calls = []
        real_softmax = torch.softmax
        monkeypatch.setattr(
            torch, "softmax", lambda *a, **k: softmax_calls.append(1) or real_softmax(*a, **k)
        )
        actual = esm3._chunked_geometric_attention(module, *args)

    assert len(softmax_calls) > 1
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
