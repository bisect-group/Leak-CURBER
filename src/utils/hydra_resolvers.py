import os

from omegaconf import OmegaConf


def _hydra_or(key: str, default: str | None = None) -> str:
    """Return `HydraConfig.<key>` inside a Hydra run, else `default` (the current directory)."""
    from hydra.core.hydra_config import HydraConfig

    # compose(return_hydra_config=True) initializes HydraConfig without runtime values.
    value = OmegaConf.select(HydraConfig.get(), key) if HydraConfig.initialized() else None
    if value is not None:
        return value
    return os.getcwd() if default is None else default


# `${hydra:runtime...}` fails under compose() or in scripts and tests. `hydra_or` gives the
# same value in @hydra.main and a fallback outside.
# `src/__init__.py` imports this module, so any `src.*` import registers it.
OmegaConf.register_new_resolver("hydra_or", _hydra_or, replace=True)
