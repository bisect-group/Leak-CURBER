from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import torch

from src.data.components.embedders.esmc import (
    _extract_final_hidden_state,
    _embed_sequences,
    _make_failures,
    _make_length_buckets,
    _mean_pool_residues,
    _token_budget,
)


def test_mean_pool_residues_skips_bos_eos_and_padding() -> None:
    # tokens are [BOS, residues..., EOS, padding...]; only residues are pooled.
    hidden = torch.tensor(
        [
            [[100.0, 100.0], [1.0, 2.0], [3.0, 4.0], [100.0, 100.0], [100.0, 100.0]],
            [[100.0, 100.0], [2.0, 0.0], [4.0, 2.0], [6.0, 4.0], [100.0, 100.0]],
        ]
    )

    pooled = _mean_pool_residues(hidden, [2, 3])

    np.testing.assert_allclose(
        pooled.numpy(),
        np.array([[2.0, 3.0], [4.0, 2.0]], dtype=np.float32),
    )


def test_extract_final_hidden_state_accepts_common_output_shapes() -> None:
    final = torch.ones(1, 2, 3)

    assert _extract_final_hidden_state(SimpleNamespace(last_hidden_state=final)) is final
    assert _extract_final_hidden_state(SimpleNamespace(hidden_states=(final - 1, final))) is final
    assert _extract_final_hidden_state({"hidden_states": final}) is final


def test_make_failures_records_each_sequence() -> None:
    failures = _make_failures(["AAA", "CCC"], "boom")

    assert failures == [
        {"raw_key": "AAA", "canonical_key": "AAA", "error": "boom"},
        {"raw_key": "CCC", "canonical_key": "CCC", "error": "boom"},
    ]


def test_worker_target_and_args_pickle_under_spawn() -> None:
    from src.data.components.embedders import esmc

    args = (
        0,
        None,
        None,
        None,
        "id",
        None,
        "bfloat16",
        64,
        None,
        0.8,
    )  # queues are passed by spawn itself
    pickle.dumps((esmc._esmc_gpu_worker, args))
    assert "self" not in esmc._esmc_gpu_worker.__code__.co_varnames


def test_worker_loop_embeds_chunks_with_a_fake_model(monkeypatch) -> None:
    import queue

    from src.data.components.embedders import esmc

    class FakeTokenizer:
        def __call__(self, seqs, return_tensors, padding):
            width = max(map(len, seqs)) + 2
            return {"input_ids": torch.zeros(len(seqs), width, dtype=torch.long)}

    class FakeModel:
        def __call__(self, input_ids, output_hidden_states):
            hidden = torch.ones(*input_ids.shape, 4)
            hidden[:, 0] = 100.0  # BOS must not reach the mean
            return type("Out", (), {"last_hidden_state": hidden})()

    monkeypatch.setattr(
        esmc, "_load_esmc", lambda *a: (FakeTokenizer(), FakeModel(), torch.device("cpu"))
    )
    tasks, results, progress_queue = queue.Queue(), queue.Queue(), queue.Queue()
    tasks.put(["AAA", "CC"])
    tasks.put(None)
    esmc._esmc_gpu_worker(0, tasks, results, progress_queue, "id", None, None, 8, 100, 0.8)
    gpu_id, embeddings, failures = results.get_nowait()
    assert gpu_id == 0 and failures == []
    assert embeddings["AAA"].tolist() == [1.0] * 4 and embeddings["CC"].tolist() == [1.0] * 4


def test_buckets_budget_padded_tokens() -> None:
    seqs = ["A" * 8] * 3 + ["A" * 18] * 2  # padded widths 10 and 20
    # 3x10=30 fits; adding a length-18 sequence makes 4x20=80 > 50.
    assert _make_length_buckets(seqs, 64, 50) == [seqs[:3], seqs[3:]]
    assert _make_length_buckets(seqs, 2, 1000) == [seqs[:2], seqs[2:4], seqs[4:]]


def test_oversized_sequence_runs_alone() -> None:
    seqs = ["A" * 5, "A" * 6, "A" * 100, "A" * 200]
    assert _make_length_buckets(seqs, 64, 50) == [seqs[:2], [seqs[2]], [seqs[3]]]


def test_token_budget_formula() -> None:
    assert _token_budget(10_000, 2.0, 0.8) == 4_000
    assert _token_budget(1, 1e6, 0.8) == 1


def test_oom_splits_batch_down_to_single_sequences() -> None:
    class FakeTokenizer:
        def __call__(self, seqs, return_tensors, padding):
            if len(seqs) > 2 or "BAD" in seqs:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return {"input_ids": torch.zeros(len(seqs), max(map(len, seqs)) + 2, dtype=torch.long)}

    class FakeModel:
        def __call__(self, input_ids, output_hidden_states):
            return SimpleNamespace(last_hidden_state=torch.ones(*input_ids.shape, 4))

    sent = []
    seqs = ["AA", "AB", "AC", "BAD", "AD"]
    embeddings, failures = _embed_sequences(
        0, seqs, FakeTokenizer(), FakeModel(), torch.device("cpu"), SimpleNamespace(put=sent.append), 64, 10_000
    )

    assert sorted(embeddings) == ["AA", "AB", "AC", "AD"]
    assert [f["raw_key"] for f in failures] == ["BAD"]
    assert sum(sent) == len(seqs)
