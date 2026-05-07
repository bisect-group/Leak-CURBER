"""
Drop-in replacement for emulator_bench/modeling.py using BACPIv2.

To switch the training script to the optimized model, change:
  from emulator_bench.modeling   import BACPIModelConfig, build_model
to:
  from emulator_bench.modeling_v2 import BACPIModelConfig, build_model

No other changes needed in train_single_target_tvt.py or tune_optuna.py.

Also exported: convert_v1_state_dict — converts an original BACPI checkpoint
to BACPIv2 weight format if you want to resume from an existing run.
"""

from dataclasses import dataclass, asdict
from types import SimpleNamespace

from emulator_bench.model_v2 import BACPIv2, convert_v1_state_dict


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


def build_model(n_atom_tokens: int, n_amino_tokens: int, config: BACPIModelConfig) -> BACPIv2:
    params = SimpleNamespace(**asdict(config))
    return BACPIv2("affinity", int(n_atom_tokens), int(n_amino_tokens), params)


__all__ = ["BACPIModelConfig", "build_model", "convert_v1_state_dict"]
