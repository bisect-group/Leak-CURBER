from __future__ import annotations

import logging
import sys
from importlib.util import find_spec
from types import SimpleNamespace

import pytest

from src.data.components import splitters
from tests.helpers.builders import (
    load_script,
)


def test_parse_args_rejects_unknown_flag(monkeypatch) -> None:
    generate_splits = load_script("generate_splits")
    monkeypatch.setattr(sys, "argv", ["generate_splits.py", "--unknown-flag"])
    with pytest.raises(SystemExit):
        generate_splits.parse_args()


def test_splitter_lazy_imports_point_to_existing_modules() -> None:
    for target in splitters._LAZY_IMPORTS.values():
        module_name, class_name = target
        assert find_spec(module_name) is not None
        assert getattr(splitters, class_name, None) is not None


def test_enzyme_retrieval_protein_variant_structure_split_resolves() -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    generate_splits = load_script("generate_splits")
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None, config_dir=str(generate_splits.CONFIG_DIR)
    ):
        cfg = generate_splits._prepare_dataset_split_cfg(
            "enzyme_retrieval",
            compose(
                config_name="data_processing",
                overrides=[
                    "data/splits_dataset@splits.dataset=enzyme_retrieval",
                    "enzyme_retrieval_dataset.include_protein_information=true",
                ],
            ),
        )
    structure_split = OmegaConf.to_container(cfg.splits, resolve=True)["dataset"][
        "structure_split"
    ]

    assert structure_split["input_dataset_parquet_file_path"].endswith(
        "with_protein_information/enzyme_retrieval_dataset_3d.parquet"
    )
    assert structure_split["drop_to_1d_dataset_parquet_file_path"].endswith(
        "enzyme_retrieval_dataset_1d.parquet"
    )


def test_main_skips_unified_datasets(monkeypatch) -> None:
    generate_splits = load_script("generate_splits")
    ran: list[str] = []
    log = logging.getLogger("test_generate_splits")
    records: list[logging.LogRecord] = []
    log.addHandler(
        type("H", (logging.Handler,), {"emit": lambda self, r: records.append(r)})()
    )
    monkeypatch.setattr(
        generate_splits,
        "component_logger",
        lambda **_: SimpleNamespace(get_logger=lambda: log),
    )
    monkeypatch.setattr(
        generate_splits,
        "run_splits_for_dataset",
        lambda dataset_name, **_: ran.append(dataset_name),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_splits.py",
            "--datasets",
            "kinetic_params_unified",
            "enzyme_substrate",
            "reactions_unified",
        ],
    )
    generate_splits.main()
    assert ran == ["enzyme_substrate"]
    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "kinetic_params_unified, reactions_unified" in warnings[0].getMessage()
