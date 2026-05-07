import io
import os
import pickle
import sys
import zipfile
import zlib
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
BAM_ROOT = CODE_ROOT / "BindingAffinityModule"
for candidate in (REPO_ROOT, CODE_ROOT, BAM_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rdkit import Chem, RDLogger

from emulator_bench.common import ligand_cache_path, normalize_sequence, protein_cache_path


RDLogger.DisableLog("rdApp.*")


def _autocast_context(device: torch.device, dtype=None):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def resolve_amp_dtype(device: torch.device):
    """Pick bf16 on Ampere+ GPUs, fp16 on older CUDA, fp32 on CPU."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, "fp32"
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(index)
    if major >= 8:
        return torch.bfloat16, "bf16-mixed"
    return torch.float16, "fp16-mixed"


_ESMFOLD_CHECKPOINT = "facebook/esmfold_v1"


def load_esmfold_model(device: torch.device, chunk_size: Optional[int] = 128):
    """Load ESMFold v1 via HuggingFace transformers (facebook/esmfold_v1)."""
    from transformers import AutoTokenizer, EsmForProteinFolding

    tokenizer = AutoTokenizer.from_pretrained(_ESMFOLD_CHECKPOINT)
    model = EsmForProteinFolding.from_pretrained(_ESMFOLD_CHECKPOINT, low_cpu_mem_usage=True)
    model.eval()
    if chunk_size is not None and hasattr(model.esm, "set_chunk_size"):
        model.esm.set_chunk_size(int(chunk_size))
    if chunk_size is not None and hasattr(model, "trunk") and hasattr(model.trunk, "set_chunk_size"):
        model.trunk.set_chunk_size(int(chunk_size))
    model = model.to(device)
    return _HFEsmFoldWrapper(model, tokenizer, device)


class _HFEsmFoldWrapper:
    def __init__(self, model, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def infer(self, sequences):
        enc = self.tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device) if "attention_mask" in enc else None
        output = self.model(input_ids, attention_mask=attention_mask)
        s_s = getattr(output, "s_s", None)
        if s_s is None and isinstance(output, dict):
            s_s = output.get("s_s")
        return {"s_s": s_s}


def _structure_module_embedding(model, sequence: str, device: torch.device, autocast_dtype=None) -> np.ndarray:
    """Return the ESMFold structure-module single representation `s_s` for one sequence."""
    with torch.no_grad(), _autocast_context(device, autocast_dtype):
        output = model.infer([sequence])
        states = output["s_s"]
        if states is None:
            raise RuntimeError("ESMFold output did not expose `s_s`; check model/version.")
    rep = states[0, : len(sequence)].detach().cpu().float().numpy()
    return rep


def embed_sequences(
    model,
    sequences: Sequence[str],
    device: torch.device,
    autocast_dtype=None,
    max_seq_len: int = 700,
    long_seq_stride: int = 500,
) -> Dict[str, np.ndarray]:
    """Embed a list of sequences, chunking anything longer than max_seq_len.

    Note: the paper drops sequences longer than 700 residues. Callers typically filter
    upstream, but we still support a sliding-window fallback for robustness.
    """
    embedded: Dict[str, np.ndarray] = {}
    for sequence in sequences:
        sequence = normalize_sequence(sequence)
        if len(sequence) <= max_seq_len:
            embedded[sequence] = _structure_module_embedding(model, sequence, device=device, autocast_dtype=autocast_dtype)
        else:
            embedded[sequence] = _embed_long_sequence(
                model,
                sequence,
                device=device,
                autocast_dtype=autocast_dtype,
                max_window=max_seq_len,
                stride=long_seq_stride,
            )
    return embedded


def _embed_long_sequence(
    model,
    sequence: str,
    device: torch.device,
    autocast_dtype=None,
    max_window: int = 700,
    stride: int = 500,
) -> np.ndarray:
    if stride >= max_window:
        raise ValueError("stride must be smaller than max_window for long-sequence embedding")
    accum = None
    counts = np.zeros((len(sequence), 1), dtype=np.float32)
    start = 0
    while start < len(sequence):
        end = min(start + max_window, len(sequence))
        window_sequence = sequence[start:end]
        window_embedding = _structure_module_embedding(model, window_sequence, device=device, autocast_dtype=autocast_dtype)
        if accum is None:
            accum = np.zeros((len(sequence), window_embedding.shape[-1]), dtype=np.float32)
        accum[start:end] += window_embedding.astype(np.float32, copy=False)
        counts[start:end] += 1.0
        if end >= len(sequence):
            break
        start += stride
    accum /= counts
    return accum


def protein_cache_item(sequence: str, embedding: np.ndarray, protein_dtype: str = "float16") -> Dict[str, np.ndarray]:
    target_dtype = np.float16 if protein_dtype == "float16" else np.float32
    return {
        "embedding": embedding.astype(target_dtype, copy=False),
        "length": np.asarray([embedding.shape[0]], dtype=np.int32),
    }


def build_ligand_graph(smiles: str):
    """Build a pytorch-geometric Data object from SMILES using the repo's featurizer."""
    from models.ligand_graph_features import mol_to_graph_data_obj_simple

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    data = mol_to_graph_data_obj_simple(mol)
    return data


def ligand_cache_item(smiles: str) -> Dict:
    graph = build_ligand_graph(smiles)
    return {
        "x": graph.x.cpu().numpy().astype(np.int16),
        "edge_index": graph.edge_index.cpu().numpy().astype(np.int32),
        "edge_attr": graph.edge_attr.cpu().numpy().astype(np.int16),
        "num_nodes": int(graph.x.shape[0]),
    }


def save_protein_npz(path: Path, item: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(handle, **item)
    tmp_path.replace(path)


def save_ligand_pt(path: Path, item: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def load_protein_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (zipfile.BadZipFile, EOFError, OSError, ValueError, zlib.error) as exc:
        raise RuntimeError(
            f"Corrupted cache file: {path}. Rebuild with `cache_embeddings.py --overwrite`."
        ) from exc


def load_ligand_pt(path: Path) -> Dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


class ProteinEmbeddingStore:
    """LRU-backed reader for cached per-residue protein embeddings."""

    def __init__(self, embeddings_dir: Path, sequences: Optional[Sequence[str]] = None, preload: bool = False, max_items: int = 512):
        self.embeddings_dir = Path(embeddings_dir)
        self.max_items = max(1, int(max_items))
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        if preload and sequences is not None:
            unique_sequences = sorted({normalize_sequence(sequence) for sequence in sequences})
            for sequence in unique_sequences:
                path = protein_cache_path(self.embeddings_dir, sequence)
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached protein embedding: {path}")
                self._cache[sequence] = load_protein_npz(path)

    def get(self, sequence: str) -> Dict[str, np.ndarray]:
        normalized = normalize_sequence(sequence)
        if normalized in self._cache:
            self._cache.move_to_end(normalized)
            return self._cache[normalized]
        path = protein_cache_path(self.embeddings_dir, normalized)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached protein embedding: {path}")
        item = load_protein_npz(path)
        self._cache[normalized] = item
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return item


class LigandGraphStore:
    """Fully-preloaded store for ligand graph data; ligand graphs are tiny."""

    def __init__(
        self,
        embeddings_dir: Path,
        smiles_values: Optional[Sequence[str]] = None,
        preload: bool = True,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self._cache: Dict[str, Dict] = {}
        if preload and smiles_values is not None:
            for smiles in sorted({str(value) for value in smiles_values}):
                path = ligand_cache_path(self.embeddings_dir, smiles)
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached ligand graph: {path}")
                self._cache[smiles] = load_ligand_pt(path)

    def get(self, smiles: str) -> Dict:
        smiles = str(smiles)
        if smiles in self._cache:
            return self._cache[smiles]
        path = ligand_cache_path(self.embeddings_dir, smiles)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached ligand graph: {path}")
        item = load_ligand_pt(path)
        self._cache[smiles] = item
        return item
