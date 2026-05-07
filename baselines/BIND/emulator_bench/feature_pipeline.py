import zipfile
import zlib
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import loading
from data import BondType
from emulator_bench.common import ligand_cache_path, normalize_sequence, protein_cache_path


PROTEIN_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
GRAPH_BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED]


def _autocast_context(device: torch.device, dtype=None):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def resolve_amp_dtype(device: torch.device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, "fp32"
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(index)
    if major >= 8:
        return torch.bfloat16, "bf16-mixed"
    return torch.float16, "fp16-mixed"


def load_esm2_model(device: torch.device, model_name: str = PROTEIN_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.requires_grad_(False)
    model = model.to(device)
    return model, tokenizer


def esm_sequence_limit(model) -> int:
    max_positions = int(getattr(model.config, "max_position_embeddings", 1026))
    return max(1, max_positions - 2)


def esm_sequence_limit_from_pretrained(model_name: str = PROTEIN_MODEL_NAME) -> int:
    config = AutoConfig.from_pretrained(model_name)
    max_positions = int(getattr(config, "max_position_embeddings", 1026))
    return max(1, max_positions - 2)


def build_esm_batches(
    sequences: Sequence[str],
    max_residues: int = 4096,
    max_batch: int = 4,
) -> List[List[str]]:
    ordered = sorted([normalize_sequence(seq) for seq in sequences], key=len, reverse=True)
    batches: List[List[str]] = []
    batch: List[str] = []
    batch_residues = 0

    for sequence in ordered:
        seq_len = len(sequence)
        if batch and (len(batch) >= max_batch or batch_residues + seq_len > max_residues):
            batches.append(batch)
            batch = []
            batch_residues = 0
        batch.append(sequence)
        batch_residues += seq_len

    if batch:
        batches.append(batch)
    return batches


def _extract_layers(hidden_states, layer_indices: Sequence[int], sequence_length: int, batch_index: int) -> List[np.ndarray]:
    extracted = []
    for layer_index in layer_indices:
        extracted.append(hidden_states[layer_index][batch_index, 1 : sequence_length + 1].detach().cpu().numpy())
    return extracted


def _esm_forward(
    model,
    tokenizer,
    sequences: Sequence[str],
    layer_indices: Sequence[int],
    device: torch.device,
    autocast_dtype=None,
) -> Dict[str, Dict[str, np.ndarray]]:
    encoded = tokenizer(list(sequences), padding="longest", truncation=False, return_tensors="pt")
    encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
    with torch.no_grad(), _autocast_context(device, autocast_dtype):
        outputs = model(**encoded, output_hidden_states=True)
    embedded = {}
    for batch_index, sequence in enumerate(sequences):
        normalized = normalize_sequence(sequence)
        seq_len = len(normalized)
        item = {"length": np.asarray([seq_len], dtype=np.int32)}
        for hidden_state, layer_index in zip(
            _extract_layers(outputs.hidden_states, layer_indices=layer_indices, sequence_length=seq_len, batch_index=batch_index),
            layer_indices,
        ):
            item[f"layer_{int(layer_index)}"] = np.asarray(hidden_state, dtype=np.float32)
        embedded[normalized] = item
    return embedded


def embed_long_sequence(
    model,
    tokenizer,
    sequence: str,
    layer_indices: Sequence[int],
    device: torch.device,
    autocast_dtype=None,
    max_window: int = 1024,
    stride: int = 896,
) -> Dict[str, np.ndarray]:
    normalized = normalize_sequence(sequence)
    if len(normalized) <= max_window:
        return _esm_forward(model, tokenizer, [normalized], layer_indices=layer_indices, device=device, autocast_dtype=autocast_dtype)[normalized]
    if stride >= max_window:
        raise ValueError("stride must be smaller than max_window for long-sequence extraction")

    accumulators = None
    counts = np.zeros((len(normalized), 1), dtype=np.float32)
    start = 0
    while start < len(normalized):
        end = min(start + max_window, len(normalized))
        window = normalized[start:end]
        window_item = _esm_forward(model, tokenizer, [window], layer_indices=layer_indices, device=device, autocast_dtype=autocast_dtype)[window]
        layer_keys = protein_layer_keys(window_item)
        if accumulators is None:
            accumulators = [np.zeros((len(normalized), window_item[key].shape[-1]), dtype=np.float32) for key in layer_keys]
        for layer_offset, key in enumerate(layer_keys):
            accumulators[layer_offset][start:end] += window_item[key].astype(np.float32, copy=False)
        counts[start:end] += 1.0
        if end >= len(normalized):
            break
        start += stride

    averaged = {}
    for layer_offset, layer_index in enumerate(layer_indices):
        averaged[f"layer_{layer_index}"] = (accumulators[layer_offset] / counts).astype(np.float32, copy=False)
    averaged["length"] = np.asarray([len(normalized)], dtype=np.int32)
    return averaged


def graph_cache_item(smiles: str) -> Dict[str, np.ndarray]:
    graph = loading.get_data(
        str(smiles).strip(),
        apply_paths=False,
        parse_cis_trans=False,
        unknown_atom_is_dummy=True,
    )
    x, adjacency, edge_features = loading.convert(*graph, bonds=GRAPH_BOND_TYPES)
    x = np.asarray(x, dtype=np.uint8)
    adjacency = np.asarray(adjacency)
    edge_index = np.vstack(np.nonzero(adjacency)).astype(np.int64, copy=False)
    if edge_index.shape[1] == 0:
        raise ValueError(f"Graph has no edges for SMILES: {smiles}")
    edge_features = np.asarray(edge_features, dtype=np.uint8)
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_nodes": np.asarray([x.shape[0]], dtype=np.int32),
    }


def protein_cache_item(
    sequence: str,
    hidden_states: Sequence[np.ndarray],
    layer_indices: Sequence[int],
    protein_dtype: str = "float16",
) -> Dict[str, np.ndarray]:
    normalized = normalize_sequence(sequence)
    target_dtype = np.float16 if protein_dtype == "float16" else np.float32
    item = {"length": np.asarray([len(normalized)], dtype=np.int32)}
    for hidden_state, layer_index in zip(hidden_states, layer_indices):
        item[f"layer_{int(layer_index)}"] = np.asarray(hidden_state, dtype=target_dtype)
    return item


def protein_layer_keys(item: Dict[str, np.ndarray]) -> List[str]:
    return sorted([key for key in item.keys() if key.startswith("layer_")], key=lambda value: int(value.split("_")[1]))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (zipfile.BadZipFile, EOFError, OSError, ValueError, zlib.error) as exc:
        raise RuntimeError(
            f"Corrupted cache file: {path}. Rebuild with `cache_embeddings.py --overwrite`."
        ) from exc


class ProteinEmbeddingStore:
    def __init__(
        self,
        embeddings_dir: Path,
        model_name: str,
        layer_indices: Sequence[int],
        sequences: Optional[Sequence[str]] = None,
        preload: bool = False,
        max_items: int = 1024,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.model_name = model_name
        self.layer_indices = list(layer_indices)
        self.max_items = max(1, int(max_items))
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        if preload and sequences is not None:
            unique_sequences = sorted({normalize_sequence(sequence) for sequence in sequences})
            for sequence in unique_sequences:
                path = protein_cache_path(self.embeddings_dir, sequence, model_name=self.model_name, layer_indices=self.layer_indices)
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached protein embedding: {path}")
                self._cache[sequence] = load_npz(path)

    def get(self, sequence: str) -> Dict[str, np.ndarray]:
        normalized = normalize_sequence(sequence)
        if normalized in self._cache:
            self._cache.move_to_end(normalized)
            return self._cache[normalized]

        path = protein_cache_path(self.embeddings_dir, normalized, model_name=self.model_name, layer_indices=self.layer_indices)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached protein embedding: {path}")
        item = load_npz(path)
        self._cache[normalized] = item
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return item


class LigandGraphStore:
    def __init__(
        self,
        embeddings_dir: Path,
        smiles_values: Optional[Sequence[str]] = None,
        preload: bool = True,
        max_items: int = 4096,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.max_items = max(1, int(max_items))
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        if preload and smiles_values is not None:
            for smiles in sorted({str(value).strip() for value in smiles_values}):
                path = ligand_cache_path(self.embeddings_dir, smiles)
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached ligand graph: {path}")
                self._cache[smiles] = load_npz(path)

    def get(self, smiles: str) -> Dict[str, np.ndarray]:
        smiles = str(smiles).strip()
        if smiles in self._cache:
            self._cache.move_to_end(smiles)
            return self._cache[smiles]
        path = ligand_cache_path(self.embeddings_dir, smiles)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached ligand graph: {path}")
        item = load_npz(path)
        self._cache[smiles] = item
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return item
