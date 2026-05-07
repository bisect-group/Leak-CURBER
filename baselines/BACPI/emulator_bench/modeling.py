from dataclasses import dataclass
from types import SimpleNamespace

from model import BACPI


@dataclass
class BACPIModelConfig:
    gat_dim: int = 50
    num_head: int = 3
    dropout: float = 0.1
    alpha: float = 0.1
    comp_dim: int = 80
    prot_dim: int = 80
    latent_dim: int = 80
    window: int = 5
    layer_cnn: int = 3
    layer_out: int = 3


def namespace_from_config(config: BACPIModelConfig):
    return SimpleNamespace(
        gat_dim=config.gat_dim,
        num_head=config.num_head,
        dropout=config.dropout,
        alpha=config.alpha,
        comp_dim=config.comp_dim,
        prot_dim=config.prot_dim,
        latent_dim=config.latent_dim,
        window=config.window,
        layer_cnn=config.layer_cnn,
        layer_out=config.layer_out,
    )


def build_model(n_atom_tokens: int, n_amino_tokens: int, config: BACPIModelConfig):
    return BACPI("affinity", int(n_atom_tokens), int(n_amino_tokens), namespace_from_config(config))

