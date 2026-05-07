import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm.auto import tqdm

from emulator_bench.feature_pipeline import (
    compound_cache_key,
    compound_cache_path,
    load_npz_payload,
    protein_cache_key,
    protein_cache_path,
)


class CompoundCacheStore:
    def __init__(
        self,
        embeddings_dir: Path,
        smiles_values: Optional[Iterable[str]] = None,
        preload: bool = False,
        preload_desc: Optional[str] = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self._cache: Dict[str, Dict] = {}
        if preload and smiles_values is not None:
            unique_smiles = sorted(set(str(value) for value in smiles_values))
            iterator = tqdm(unique_smiles, desc=preload_desc, unit="compound") if preload_desc else unique_smiles
            for smiles in iterator:
                self.get(smiles)

    def get(self, smiles: str) -> Dict:
        key = compound_cache_key(smiles)
        if key not in self._cache:
            self._cache[key] = load_npz_payload(compound_cache_path(self.embeddings_dir, smiles))
        return self._cache[key]


class ProteinCacheStore:
    def __init__(
        self,
        embeddings_dir: Path,
        sequences: Optional[Iterable[str]] = None,
        preload: bool = False,
        preload_desc: Optional[str] = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self._cache: Dict[str, Dict] = {}
        if preload and sequences is not None:
            unique_sequences = sorted(set(str(value) for value in sequences))
            iterator = tqdm(unique_sequences, desc=preload_desc, unit="protein") if preload_desc else unique_sequences
            for sequence in iterator:
                self.get(sequence)

    def get(self, sequence: str) -> Dict:
        key = protein_cache_key(sequence)
        if key not in self._cache:
            self._cache[key] = load_npz_payload(protein_cache_path(self.embeddings_dir, sequence))
        return self._cache[key]


class BACPIDataset(Dataset):
    def __init__(self, frame, compound_store: CompoundCacheStore, protein_store: ProteinCacheStore, smiles_col: str, sequence_col: str, target_col: str):
        self.frame = frame.reset_index(drop=True)

        # Pull hot-path data into plain Python lists so __getitem__ never touches pandas
        self._smiles_list: List[str] = self.frame[smiles_col].astype(str).tolist()
        self._sequence_list: List[str] = self.frame[sequence_col].astype(str).tolist()
        self._targets: List[float] = self.frame[target_col].tolist()

        # Pre-resolve cache references per row; avoids a hash + dict lookup on every __getitem__
        # call. Duplicate SMILES/sequences share the same dict object (no extra memory cost).
        self._compound_items: List[Dict] = [compound_store.get(s) for s in self._smiles_list]
        self._protein_items: List[Dict] = [protein_store.get(s) for s in self._sequence_list]

        self._compound_lengths: List[int] = [int(ci["compound_length"][0]) for ci in self._compound_items]
        self._protein_lengths: List[int] = [int(pi["protein_length"][0]) for pi in self._protein_items]
        self._sample_costs: List[int] = [
            int(cl * cl + cl * pl)
            for cl, pl in zip(self._compound_lengths, self._protein_lengths)
        ]
        self._sort_keys: List[tuple] = [
            (int(cl), int(pl), int(sc))
            for cl, pl, sc in zip(self._compound_lengths, self._protein_lengths, self._sample_costs)
        ]

    def __len__(self):
        return len(self._targets)

    @property
    def sort_keys(self) -> List[tuple]:
        return self._sort_keys

    def __getitem__(self, index):
        ci = self._compound_items[index]
        pi = self._protein_items[index]
        return {
            "compounds": ci["compounds"],
            "adjacencies": ci["adjacencies"],
            "fingerprint": ci["fingerprint"],
            "proteins": pi["proteins"],
            "target": self._targets[index],
        }


def _round_up_to_multiple(length: int, multiple: int):
    if multiple <= 1:
        return int(length)
    return int(((length + multiple - 1) // multiple) * multiple)


def _pad_1d(arrays: List[np.ndarray], pad_to_multiple: int):
    max_length = max(array.shape[0] for array in arrays)
    max_length = _round_up_to_multiple(max_length, pad_to_multiple)
    padded = np.zeros((len(arrays), max_length), dtype=np.int64)
    mask = np.zeros((len(arrays), max_length), dtype=np.float32)
    for index, array in enumerate(arrays):
        length = array.shape[0]
        padded[index, :length] = array.astype(np.int64) + 1
        mask[index, :length] = 1.0
    return padded, mask


def _pad_2d(arrays: List[np.ndarray], pad_to_multiple: int):
    max_length = max(array.shape[0] for array in arrays)
    max_length = _round_up_to_multiple(max_length, pad_to_multiple)
    # bool_ instead of int64: 8x smaller, and the model only checks adj > 0 / adj.bool()
    padded = np.zeros((len(arrays), max_length, max_length), dtype=np.bool_)
    for index, array in enumerate(arrays):
        length = array.shape[0]
        padded[index, :length, :length] = array.astype(bool)
    return padded


def make_bacpi_collate_fn(pad_to_multiple: int):
    def bacpi_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        compounds, compound_mask = _pad_1d([item["compounds"] for item in batch], pad_to_multiple=pad_to_multiple)
        proteins, protein_mask = _pad_1d([item["proteins"] for item in batch], pad_to_multiple=pad_to_multiple)
        adjacencies = _pad_2d([item["adjacencies"] for item in batch], pad_to_multiple=pad_to_multiple)
        fingerprints = np.stack([item["fingerprint"] for item in batch]).astype(np.float32)
        targets = np.asarray([item["target"] for item in batch], dtype=np.float32).reshape(-1, 1)
        return {
            "compounds": torch.as_tensor(compounds, dtype=torch.long),
            "compound_mask": torch.as_tensor(compound_mask, dtype=torch.float32),
            "adjacencies": torch.as_tensor(adjacencies, dtype=torch.bool),
            "fingerprint": torch.as_tensor(fingerprints, dtype=torch.float32),
            "proteins": torch.as_tensor(proteins, dtype=torch.long),
            "protein_mask": torch.as_tensor(protein_mask, dtype=torch.float32),
            "targets": torch.as_tensor(targets, dtype=torch.float32),
        }

    return bacpi_collate_fn


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,
        lengths: List,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_size_multiplier: int = 50,
        seed: int = 0,
    ):
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.bucket_size_multiplier = max(1, int(bucket_size_multiplier))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        total = len(self.lengths)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = sorted(range(len(self.lengths)), key=lambda idx: self.lengths[idx], reverse=True)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            neighborhood = self.batch_size * self.bucket_size_multiplier
            reordered_batches = []
            for start in range(0, len(indices), neighborhood):
                bucket = indices[start : start + neighborhood]
                bucket_batches = []
                for batch_start in range(0, len(bucket), self.batch_size):
                    batch = bucket[batch_start : batch_start + self.batch_size]
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    bucket_batches.append(batch)
                rng.shuffle(bucket_batches)
                reordered_batches.extend(bucket_batches)
            for batch in reordered_batches:
                yield batch
            return
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch


def create_loader(
    dataset: BACPIDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    bucket_size_multiplier: int,
    pad_to_multiple: int,
    seed: int,
):
    loader_kwargs = {
        "dataset": dataset,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": make_bacpi_collate_fn(pad_to_multiple=pad_to_multiple),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    if shuffle:
        batch_sampler = BucketBatchSampler(
            dataset.sort_keys,
            batch_size=batch_size,
            shuffle=True,
            bucket_size_multiplier=bucket_size_multiplier,
            seed=seed,
        )
        loader = DataLoader(batch_sampler=batch_sampler, **loader_kwargs)
        return loader, batch_sampler
    loader = DataLoader(batch_size=batch_size, shuffle=False, **loader_kwargs)
    return loader, None
