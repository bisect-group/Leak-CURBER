from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from emulator_bench.common import normalize_sequence
from emulator_bench.feature_pipeline import EmbeddingStore


class CachedStructureFreeDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        protein_store: EmbeddingStore,
        molecule_store: EmbeddingStore,
        sequence_col: str = "sequence",
        smiles_col: str = "smiles",
        target_col: Optional[str] = "log10_value",
    ):
        self.frame = frame.reset_index(drop=True)
        self.protein_store = protein_store
        self.molecule_store = molecule_store
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.target_col = target_col

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        protein = self.protein_store.get(normalize_sequence(str(row[self.sequence_col])))
        molecule = self.molecule_store.get(str(row[self.smiles_col]))
        item = {
            "protein_embedding": torch.from_numpy(protein["embedding"]).float(),
            "molecule_embedding": torch.from_numpy(molecule["embedding"]).float(),
        }
        if self.target_col is not None and self.target_col in self.frame.columns:
            item["labels"] = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return item


def collate_cached(batch):
    output = {
        "protein_embedding": torch.stack([item["protein_embedding"] for item in batch], dim=0),
        "molecule_embedding": torch.stack([item["molecule_embedding"] for item in batch], dim=0),
    }
    if "labels" in batch[0]:
        output["labels"] = torch.stack([item["labels"] for item in batch], dim=0)
    return output
