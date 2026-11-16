from __future__ import annotations

import logging
import pickle
import multiprocessing
import queue
import threading
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rich.logging import RichHandler

pytest.importorskip("MolR.featurizer")

from MolR.featurizer import MolEFeaturizer, build_graph_result, load_feature_encoder  # noqa: E402

from src.data.components.embedders import molr  # noqa: E402
from src.data.components.embedders.molr import MolR2DShardEmbedder, _molr_inference_worker  # noqa: E402

SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "C", "CCN"]
FACTOR = 1536.0


def _records(smiles: list[str] = SMILES) -> list:
    encoder = load_feature_encoder("gcn_1024")
    return [build_graph_result(i, s, encoder) for i, s in enumerate(smiles)]


def _embedder(chunk_size: int = 100) -> MolR2DShardEmbedder:
    embedder = object.__new__(MolR2DShardEmbedder)
    embedder.model_name = "gcn_1024"
    embedder.logger = logging.getLogger("molr-test")
    embedder.gpu_batch_size = 2
    embedder.graph_workers = 1
    embedder.use_gpu = False
    embedder.gpu_ids = [0]
    embedder.compute_chunk_size = chunk_size
    embedder.factor = FACTOR
    return embedder


def test_batched_inference_keys_float16_progress_and_factor() -> None:
    out, progress = queue.Queue(), queue.Queue()
    _molr_inference_worker(None, _records(), 2, FACTOR, out, progress)
    arrays, failures = out.get_nowait()
    assert failures == []
    assert list(arrays) == SMILES
    assert all(a.dtype == np.float16 and np.isfinite(a).all() for a in arrays.values())
    assert progress.qsize() == 3

    model = MolEFeaturizer(gpu=None)
    model.mole.factor = FACTOR
    direct, _ = model.transform(SMILES, batch_size=2)
    np.testing.assert_allclose(
        np.stack([arrays[s] for s in SMILES]).astype(np.float32), direct, rtol=1e-2, atol=1e-2
    )


def test_graph_worker_result_has_no_tensors_and_restores_graph() -> None:
    molr._graph_worker_init("gcn_1024")
    result = molr._graph_worker((0, "CCO"))
    graph = result.graph
    assert all(
        isinstance(a, np.ndarray) for a in (graph.feature, graph.src, graph.dst, graph.node_counts)
    )
    payload = bytes(ForkingPickler.dumps(result))
    assert b"torch" not in payload

    expected = _records(["CCO"])[0].graph
    restored = molr._restore_graph(pickle.loads(payload)).graph
    for name in ("feature", "src", "dst", "node_counts"):
        assert torch.equal(getattr(restored, name), getattr(expected, name))


def _logging_child() -> None:
    molr._graph_worker_init("gcn_1024")
    logging.getLogger("pysmiles.read_smiles").warning("Ignoring invalid E/Z annotations")
    assert molr._graph_worker((0, "C/C=C1/C(=O)NC1")).graph is not None


def test_worker_init_survives_held_rich_console_lock() -> None:
    from src.utils.rich_progress import get_console

    console = get_console()
    root = logging.getLogger()
    handler = RichHandler(console=console)
    root.addHandler(handler)
    child = multiprocessing.get_context("fork").Process(target=_logging_child)
    holding, release = threading.Event(), threading.Event()

    def hold_lock() -> None:
        with console._lock:
            holding.set()
            release.wait()

    holder = threading.Thread(target=hold_lock)
    holder.start()
    holding.wait()
    try:
        child.start()  # The child inherits a console lock held by another thread.
        child.join(60)
        hung = child.is_alive()
        if hung:
            child.kill()
        assert not hung and child.exitcode == 0
    finally:
        release.set()
        holder.join()
        root.removeHandler(handler)


def test_worker_failure_is_reported_per_record() -> None:
    out = queue.Queue()
    _molr_inference_worker(
        None, [SimpleNamespace(smiles="x", graph=None)], 2, FACTOR, out, queue.Queue()
    )
    arrays, failures = out.get_nowait()
    assert arrays == {} and failures[0]["raw_key"] == "x"


# Batch composition changes float32 summation order, so allow one float16 step.
def test_vectors_do_not_depend_on_chunking_or_resume() -> None:
    keys = SMILES + ["CCCC", "OCCO", "CC(C)O"]

    def run(chunks: list[list[str]]) -> dict[str, np.ndarray]:
        embedder = _embedder()
        result: dict[str, np.ndarray] = {}
        for chunk in chunks:
            arrays, failures = embedder.compute_many(chunk)
            assert failures == []
            result.update(arrays)
        return result

    one = run([keys])
    several = run([keys[:3], keys[3:5], keys[5:]])
    resumed = run([keys[4:]])
    for key in keys:
        np.testing.assert_allclose(one[key], several[key], rtol=2e-3, atol=1e-3)
    for key in keys[4:]:
        np.testing.assert_allclose(one[key], resumed[key], rtol=2e-3, atol=1e-3)
