import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from emulator_bench.common import ensure_parent, stable_hash


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


DEFAULT_RADIUS = 2
DEFAULT_NGRAM = 3
DEFAULT_FP_RADIUS = 2
DEFAULT_NBITS = 1024


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).strip().upper().split())


def normalize_smiles_text(smiles: str) -> str:
    return str(smiles).strip()


def try_canonicalize_smiles(smiles: str):
    raw = normalize_smiles_text(smiles)
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_smiles(smiles: str) -> str:
    canonical = try_canonicalize_smiles(smiles)
    if canonical is None:
        raise ValueError("Invalid SMILES: %s" % normalize_smiles_text(smiles))
    return canonical


def filter_invalid_smiles_rows(frame: pd.DataFrame, smiles_col: str, source_name: str = ""):
    cache = {}
    mask = []
    invalid_examples = []
    for smiles in frame[smiles_col].astype(str).tolist():
        if smiles not in cache:
            cache[smiles] = try_canonicalize_smiles(smiles) is not None
        is_valid = cache[smiles]
        mask.append(is_valid)
        if not is_valid and len(invalid_examples) < 5:
            invalid_examples.append(smiles)

    filtered = frame.loc[mask].reset_index(drop=True)
    dropped = int(len(frame) - len(filtered))
    stats = {
        "source": str(source_name),
        "rows_in": int(len(frame)),
        "rows_out": int(len(filtered)),
        "rows_dropped": dropped,
        "invalid_examples": invalid_examples,
    }
    return filtered, stats


def compound_cache_key(smiles: str) -> str:
    return stable_hash(normalize_smiles_text(smiles))


def protein_cache_key(sequence: str) -> str:
    return stable_hash(normalize_sequence(sequence))


def compound_cache_path(embeddings_dir: Path, smiles: str) -> Path:
    key = compound_cache_key(smiles)
    return Path(embeddings_dir) / "compounds" / key[:2] / ("%s.npz" % key)


def protein_cache_path(embeddings_dir: Path, sequence: str) -> Path:
    key = protein_cache_key(sequence)
    return Path(embeddings_dir) / "proteins" / key[:2] / ("%s.npz" % key)


def save_npz_atomic(path: Path, payload: Dict) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp_path.replace(path)


class BACPIFeaturizer:
    def __init__(
        self,
        radius: int = DEFAULT_RADIUS,
        ngram: int = DEFAULT_NGRAM,
        fp_radius: int = DEFAULT_FP_RADIUS,
        nbits: int = DEFAULT_NBITS,
        atom_dict: Optional[Dict] = None,
        bond_dict: Optional[Dict] = None,
        fingerprint_dict: Optional[Dict] = None,
        edge_dict: Optional[Dict] = None,
        word_dict: Optional[Dict] = None,
    ):
        self.radius = int(radius)
        self.ngram = int(ngram)
        self.fp_radius = int(fp_radius)
        self.nbits = int(nbits)
        self.atom_dict = dict(atom_dict or {})
        self.bond_dict = dict(bond_dict or {})
        self.fingerprint_dict = dict(fingerprint_dict or {})
        self.edge_dict = dict(edge_dict or {})
        self.word_dict = dict(word_dict or {})

    def _lookup(self, mapping: Dict, key, build: bool) -> int:
        if key in mapping:
            return int(mapping[key])
        if not build:
            raise KeyError("Feature `%s` is not present in the frozen BACPI vocabulary." % (key,))
        index = len(mapping)
        mapping[key] = index
        return index

    def _create_atoms(self, mol, build: bool):
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for atom in mol.GetAromaticAtoms():
            index = atom.GetIdx()
            atoms[index] = (atoms[index], "aromatic")
        encoded = [self._lookup(self.atom_dict, atom, build) for atom in atoms]
        return np.asarray(encoded, dtype=np.int32)

    def _create_ijbonddict(self, mol, build: bool):
        i_jbond_dict = {}
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_id = self._lookup(self.bond_dict, str(bond.GetBondType()), build)
            i_jbond_dict.setdefault(begin_idx, []).append((end_idx, bond_id))
            i_jbond_dict.setdefault(end_idx, []).append((begin_idx, bond_id))

        atom_indices = set(range(mol.GetNumAtoms()))
        isolated = atom_indices - set(i_jbond_dict.keys())
        nan_bond = self._lookup(self.bond_dict, "nan", build)
        for atom_idx in isolated:
            i_jbond_dict.setdefault(atom_idx, []).append((atom_idx, nan_bond))
        return i_jbond_dict

    def _atom_features(self, atoms, i_jbond_dict, build: bool):
        if len(atoms) == 1 or self.radius == 0:
            fingerprints = [self._lookup(self.fingerprint_dict, int(atom), build) for atom in atoms]
            return np.asarray(fingerprints, dtype=np.int32)

        nodes = list(atoms.tolist())
        i_jedge_dict = i_jbond_dict
        for _ in range(self.radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = tuple(sorted((nodes[j], edge) for j, edge in j_edge))
                fingerprint = (nodes[i], neighbors)
                fingerprints.append(self._lookup(self.fingerprint_dict, fingerprint, build))

            nodes = fingerprints
            next_edges = {}
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_sides = tuple(sorted((nodes[i], nodes[j])))
                    edge_key = (both_sides, edge)
                    edge_id = self._lookup(self.edge_dict, edge_key, build)
                    next_edges.setdefault(i, []).append((j, edge_id))
            i_jedge_dict = next_edges
        return np.asarray(nodes, dtype=np.int32)

    def _create_adjacency(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        adjacency = np.asarray(adjacency, dtype=np.uint8)
        adjacency += np.eye(adjacency.shape[0], dtype=np.uint8)
        return adjacency

    def _fingerprint_bits(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.nbits, useChirality=True)
        array = np.zeros((self.nbits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, array)
        return array

    def _split_sequence(self, sequence: str, build: bool):
        normalized = "-" + normalize_sequence(sequence) + "="
        words = []
        for index in range(len(normalized) - self.ngram + 1):
            token = normalized[index : index + self.ngram]
            words.append(self._lookup(self.word_dict, token, build))
        return np.asarray(words, dtype=np.int32)

    def encode_compound(self, smiles: str, build: bool = False) -> Dict:
        canonical = canonicalize_smiles(smiles)
        mol = Chem.AddHs(Chem.MolFromSmiles(canonical))
        atoms = self._create_atoms(mol, build=build)
        i_jbond_dict = self._create_ijbonddict(mol, build=build)
        compound_tokens = self._atom_features(atoms, i_jbond_dict, build=build)
        return {
            "compounds": compound_tokens.astype(np.int32),
            "adjacencies": self._create_adjacency(mol).astype(np.uint8),
            "fingerprint": self._fingerprint_bits(mol).astype(np.uint8),
            "compound_length": np.asarray([int(compound_tokens.shape[0])], dtype=np.int32),
            "canonical_smiles": np.asarray([canonical]),
        }

    def encode_protein(self, sequence: str, build: bool = False) -> Dict:
        normalized = normalize_sequence(sequence)
        tokens = self._split_sequence(normalized, build=build)
        return {
            "proteins": tokens.astype(np.int32),
            "protein_length": np.asarray([int(tokens.shape[0])], dtype=np.int32),
            "normalized_sequence": np.asarray([normalized]),
        }

    def build_from_values(self, smiles_values: Iterable[str], sequence_values: Iterable[str]) -> None:
        smiles_values = list(smiles_values)
        sequence_values = list(sequence_values)
        canonical_smiles = []
        for value in tqdm(smiles_values, desc="Canonicalizing compounds for vocab", unit="compound"):
            canonical = try_canonicalize_smiles(value)
            if canonical is not None:
                canonical_smiles.append(canonical)
        unique_canonical_smiles = sorted(set(canonical_smiles))
        for smiles in tqdm(unique_canonical_smiles, desc="Building compound vocabulary", unit="compound"):
            self.encode_compound(smiles, build=True)
        unique_sequences = sorted({normalize_sequence(value) for value in sequence_values})
        for sequence in tqdm(unique_sequences, desc="Building protein vocabulary", unit="protein"):
            self.encode_protein(sequence, build=True)

    @property
    def n_atom_tokens(self) -> int:
        return len(self.fingerprint_dict)

    @property
    def n_amino_tokens(self) -> int:
        return len(self.word_dict)

    def to_payload(self) -> Dict:
        return {
            "radius": self.radius,
            "ngram": self.ngram,
            "fp_radius": self.fp_radius,
            "nbits": self.nbits,
            "atom_dict": self.atom_dict,
            "bond_dict": self.bond_dict,
            "fingerprint_dict": self.fingerprint_dict,
            "edge_dict": self.edge_dict,
            "word_dict": self.word_dict,
        }

    def save(self, path: Path) -> None:
        ensure_parent(path)
        with open(path, "wb") as handle:
            pickle.dump(self.to_payload(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return cls(**payload)


def load_npz_payload(path: Path) -> Dict:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}
