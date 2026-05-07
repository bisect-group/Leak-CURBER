from __future__ import annotations
from importlib import import_module

__all__ = [
    "ClampSmilesShardEmbedder",
    "DRFPShardEmbedder",
    "ESMCShardEmbedder",
    "ESM3StructureShardEmbedder",
    "MolR2DShardEmbedder",
    "RxnFPShardEmbedder",
    "SMITEDShardEmbedder",
    "SmilesECFPShardEmbedder",
    "SmilesAtomPairShardEmbedder",
    "UnimolSDFShardEmbedder",
]

_LAZY_IMPORTS = {
    "ClampSmilesShardEmbedder": (
        "src.data.components.embedders.clamp",
        "ClampSmilesShardEmbedder",
    ),
    "DRFPShardEmbedder": (
        "src.data.components.embedders.drfp",
        "DRFPShardEmbedder",
    ),
    "ESMCShardEmbedder": (
        "src.data.components.embedders.esmc",
        "ESMCShardEmbedder",
    ),
    "ESM3StructureShardEmbedder": (
        "src.data.components.embedders.esm3",
        "ESM3StructureShardEmbedder",
    ),
    "MolR2DShardEmbedder": (
        "src.data.components.embedders.molr",
        "MolR2DShardEmbedder",
    ),
    "RxnFPShardEmbedder": (
        "src.data.components.embedders.rxnfp",
        "RxnFPShardEmbedder",
    ),
    "SMITEDShardEmbedder": (
        "src.data.components.embedders.smited",
        "SMITEDShardEmbedder",
    ),
    "SmilesECFPShardEmbedder": (
        "src.data.components.embedders.ecfp",
        "SmilesECFPShardEmbedder",
    ),
    "SmilesAtomPairShardEmbedder": (
        "src.data.components.embedders.atom_pair",
        "SmilesAtomPairShardEmbedder",
    ),
    "UnimolSDFShardEmbedder": (
        "src.data.components.embedders.unimol_sdf",
        "UnimolSDFShardEmbedder",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
