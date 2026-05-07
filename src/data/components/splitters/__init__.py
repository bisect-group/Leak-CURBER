from __future__ import annotations

from importlib import import_module

__all__ = [
    "EmbeddingCosineSimilaritySplitter",
    "ConformerCosineSimilaritySplitter",
    "ECHierarchicalGroupSplitter",
    "GroupShuffleUniqueColumnSplitter",
    "ProteinStructMaxLDDTSimilaritySplitter",
    "ReactionDRFPTanimotoSimilaritySplitter",
    "ProteinSeqMaxFidentSimilaritySplitter",
    "RandomSplitter",
    "SMILESMaxTanimotoSimilaritySplitter",
    "UniProtTimeBasedSplitter",
]

_LAZY_IMPORTS = {
    "EmbeddingCosineSimilaritySplitter": (
        "src.data.components.splitters.embedding_cosine",
        "EmbeddingCosineSimilaritySplitter",
    ),
    "ConformerCosineSimilaritySplitter": (
        "src.data.components.splitters.embedding_cosine",
        "ConformerCosineSimilaritySplitter",
    ),
    "ECHierarchicalGroupSplitter": (
        "src.data.components.splitters.ec_hierarchical",
        "ECHierarchicalGroupSplitter",
    ),
    "GroupShuffleUniqueColumnSplitter": (
        "src.data.components.splitters.group_shuffle",
        "GroupShuffleUniqueColumnSplitter",
    ),
    "ProteinStructMaxLDDTSimilaritySplitter": (
        "src.data.components.splitters.protein_struct_lddt",
        "ProteinStructMaxLDDTSimilaritySplitter",
    ),
    "ReactionDRFPTanimotoSimilaritySplitter": (
        "src.data.components.splitters.reaction_tanimoto",
        "ReactionDRFPTanimotoSimilaritySplitter",
    ),
    "ProteinSeqMaxFidentSimilaritySplitter": (
        "src.data.components.splitters.protein_seq_fident",
        "ProteinSeqMaxFidentSimilaritySplitter",
    ),
    "RandomSplitter": (
        "src.data.components.splitters.random",
        "RandomSplitter",
    ),
    "SMILESMaxTanimotoSimilaritySplitter": (
        "src.data.components.splitters.smiles_tanimoto",
        "SMILESMaxTanimotoSimilaritySplitter",
    ),
    "UniProtTimeBasedSplitter": (
        "src.data.components.splitters.uniprot_time",
        "UniProtTimeBasedSplitter",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
