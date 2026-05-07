from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from emulator_bench.common import normalize_sequence
from emulator_bench.feature_pipeline import LigandGraphStore, ProteinEmbeddingStore, protein_layer_keys


@dataclass
class BindBatch:
    graphs: Batch
    protein_layers: List[torch.Tensor]
    attention_mask: torch.Tensor
    targets: torch.Tensor
    metadata: List[Dict]

    def to(self, device: torch.device):
        self.graphs = self.graphs.to(device)
        self.attention_mask = self.attention_mask.to(device, non_blocking=True)
        self.targets = self.targets.to(device, non_blocking=True)
        self.protein_layers = [layer.to(device, non_blocking=True) for layer in self.protein_layers]
        return self


class CachedBindDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        protein_store: ProteinEmbeddingStore,
        ligand_store: LigandGraphStore,
        sequence_col: str = "sequence",
        smiles_col: str = "smiles",
        target_col: str = "log10_value",
    ):
        self.frame = frame.reset_index(drop=True)
        self.protein_store = protein_store
        self.ligand_store = ligand_store
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.target_col = target_col

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        smiles = str(row[self.smiles_col]).strip()
        sequence = normalize_sequence(str(row[self.sequence_col]))
        protein = self.protein_store.get(sequence)
        ligand = self.ligand_store.get(smiles)

        metadata = {}
        for key, value in row.items():
            metadata[key] = _normalize_metadata_value(value)
        metadata["sequence"] = sequence
        metadata["smiles"] = smiles

        return {
            "graph": ligand,
            "protein": protein,
            "target": float(row[self.target_col]),
            "metadata": metadata,
        }


def _normalize_metadata_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def collate_bind_samples(samples: List[Dict]) -> BindBatch:
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    graphs = []
    metadata = []
    targets = []
    protein_items = [sample["protein"] for sample in samples]
    layer_keys = protein_layer_keys(protein_items[0])
    lengths = [int(item["length"][0]) for item in protein_items]
    max_len = max(lengths)
    batch_size = len(samples)
    protein_layers = []

    for layer_key in layer_keys:
        feature_dim = int(protein_items[0][layer_key].shape[-1])
        padded = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.from_numpy(protein_items[0][layer_key]).dtype)
        for batch_index, item in enumerate(protein_items):
            current = torch.from_numpy(item[layer_key])
            padded[batch_index, : current.shape[0]] = current
        protein_layers.append(padded)

    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for batch_index, sample in enumerate(samples):
        graph_item = sample["graph"]
        graph = Data(
            x=torch.from_numpy(graph_item["x"]),
            edge_index=torch.from_numpy(graph_item["edge_index"]),
            edge_features=torch.from_numpy(graph_item["edge_features"]),
        )
        graphs.append(graph)
        attention_mask[batch_index, : lengths[batch_index]] = True
        targets.append(float(sample["target"]))
        metadata.append(sample["metadata"])

    return BindBatch(
        graphs=Batch.from_data_list(graphs),
        protein_layers=protein_layers,
        attention_mask=attention_mask,
        targets=torch.tensor(targets, dtype=torch.float32).unsqueeze(1),
        metadata=metadata,
    )
