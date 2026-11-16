import pytest
import rootutils
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import hydra
from hydra import compose, initialize, initialize_config_dir
from tests.helpers.builders import REPO_ROOT, compose_config


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    # The model needs `input_dim` from the datamodule at run time, so only check its target.
    assert cfg_train.model._target_.endswith("BaselineRegressionLightningModule")
    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.trainer)


def test_data_processing_config_includes_esmfold2_and_esmc_defaults() -> None:
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="data_processing.yaml",
            overrides=[f"paths.root_dir={rootutils.find_root(indicator='.project-root')}"],
        )

    assert cfg.embeddings.esmc_model_id == "biohub/ESMC-6B"
    assert cfg.embeddings.esmc_model_name == "ESMC-6B"
    assert cfg.embeddings.esmfold2_model_id == "biohub/ESMFold2"
    assert cfg.embeddings.esmfold2_input_pkl_path == f"{cfg.paths.intermediate_data_dir}/all_proteins.pkl"
    assert cfg.embeddings.esmfold2_skip_known_structures is False
    assert cfg.embeddings.esmfold2_write_pdb_compat is True


def test_embedders_read_the_supersets() -> None:
    cfg = compose_config("data_processing.yaml", [f"paths.root_dir={REPO_ROOT}"])
    superset = lambda name: f"{cfg.paths.intermediate_data_dir}/{name}.pkl"  # noqa: E731
    expected = {
        "all_proteins": ["esmc_embeddings", "esm3_pdbs", "esm3_embeddings", "esmfold2"],
        "all_canonical_smiles": [
            "smited_embeddings",
            "smiles_ecfp_embeddings",
            "smiles_atom_pair_embeddings",
            "molr_embeddings",
            "unimol_embeddings",
        ],
        "all_rxn_smiles": ["rxnfp_embeddings", "drfp_embeddings"],
    }
    for name, prefixes in expected.items():
        for prefix in prefixes:
            assert cfg.embeddings[f"{prefix}_input_pkl_path"] == superset(name)


def test_train_composes_with_baseline_regression_defaults() -> None:
    """`python src/train.py --cfg job` must compose."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        train_cfg = compose(config_name="train.yaml")
    assert train_cfg.data._target_.endswith("BaselineRegressionDataModule")
    assert train_cfg.model._target_.endswith("BaselineRegressionLightningModule")


@pytest.mark.parametrize(
    "dataset, section",
    [
        ("kinetic_params_km", "kinetic_params_dataset"),
        ("kinetic_params_unified", "kinetic_params_dataset"),
        ("binding_affinity_unified", "binding_affinity_dataset"),
    ],
)
def test_cosine_split_reads_the_1d_parquet(dataset: str, section: str) -> None:
    cfg = compose_config(
        "data_processing.yaml",
        [f"paths.root_dir={REPO_ROOT}", f"data/splits_dataset@splits.dataset={dataset}"],
    )
    key = dataset.removeprefix("kinetic_params_").removeprefix("binding_affinity_")
    expected = cfg[section].dataset_paths[key].parquet_1d
    assert cfg.splits.dataset.cosine_split.input_dataset_parquet_file_path == expected


def test_final_runs_use_the_composed_scheduler() -> None:
    cfg = compose_config("train.yaml", [f"paths.root_dir={REPO_ROOT}"])
    assert cfg.model.scheduler._target_.endswith("CosineAnnealingWarmRestarts")
    OmegaConf.resolve(cfg.model)
    assert cfg.model.scheduler.T_0 == cfg.scheduler.T_0


def test_classification_model_composes_with_scheduler() -> None:
    cfg = compose_config(
        "train.yaml",
        [
            f"paths.root_dir={REPO_ROOT}",
            "model=baseline_classification",
            "scheduler=cosine_warm_restarts",
        ],
    )
    assert cfg.model.scheduler._target_.endswith("CosineAnnealingWarmRestarts")


def test_plain_train_config_composes() -> None:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose("train.yaml")
        assert cfg.optimized_metric is None
        experiment = compose("train.yaml", overrides=["experiment=baseline_regression_torch_mlp"])
        assert experiment.get("logger") is None


def test_hpo_storage_does_not_use_the_hydra_runtime() -> None:
    for name in ("baseline_regression_optuna4", "baseline_classification_optuna4"):
        text = (REPO_ROOT / "configs/hparams_search" / f"{name}.yaml").read_text()
        assert "storage: sqlite:///${paths.log_dir}/" in text


def test_logistic_regression_config_has_no_multi_class() -> None:
    path = (
        REPO_ROOT
        / "configs/pipeline/baseline_classification_sklearn_logistic_regression_optuna100.yaml"
    )
    assert "multi_class" not in path.read_text()


def test_removed_files_stay_removed() -> None:
    for name in ("src/eval.py", "configs/eval.yaml", "pixi.toml", ".dvc/config", ".dvcignore"):
        assert not (REPO_ROOT / name).exists(), name


def test_embeddings_config_has_no_home_paths() -> None:
    assert "/home/" not in (REPO_ROOT / "configs/data/embeddings.yaml").read_text()


def test_environment_declares_runtime_dependencies() -> None:
    text = (REPO_ROOT / "environment.yaml").read_text()
    assert all(name in text for name in ("xgboost", "scipy", "joblib"))


_DATASET_CONFIGS = sorted(
    p.stem for p in (REPO_ROOT / "configs/data/splits_dataset").glob("*.yaml")
)


@pytest.mark.parametrize("dataset", _DATASET_CONFIGS)
def test_data_processing_composes_for_each_splits_dataset(dataset: str) -> None:
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="data_processing",
            overrides=[f"data/splits_dataset@splits.dataset={dataset}"],
        )
    assert cfg.splits.dataset


def test_enzyme_classification_split_config_has_no_ec_split(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    import generate_splits

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(
            config_name="data_processing",
            overrides=["data/splits_dataset@splits.dataset=enzyme_classification"],
        )
    dataset = cfg.splits.dataset
    assert "ec_split" not in dataset
    assert not [key for key in cfg.splits if key.startswith("ec_split")]
    base = f"{cfg.paths.enzyme_classification_dataset_path}/enzyme_classification_dataset"
    assert base.endswith("datasets/enzyme_classification_dataset/enzyme_classification_dataset")
    for block in ("random", "sequence_split"):
        assert dataset[block].input_dataset_parquet_file_path == f"{base}_1d.parquet"
    structure = dataset.structure_split
    assert structure.input_dataset_parquet_file_path == f"{base}_3d.parquet"
    assert structure.drop_to_1d_dataset_parquet_file_path == f"{base}_1d.parquet"
    assert structure.drop_to_1d
    assert "ec" not in generate_splits.DEFAULT_SPLITTER_ORDER


def test_enzyme_retrieval_protein_splits_use_the_1d_and_3d_files() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(
            config_name="data_processing",
            overrides=["data/splits_dataset@splits.dataset=enzyme_retrieval"],
        )
    dataset = cfg.splits.dataset
    base = f"{cfg.paths.enzyme_retrieval_dataset_path}/with_protein_information/enzyme_retrieval_dataset"
    protein = dataset.with_protein_information
    assert protein.sequence_split.input_dataset_parquet_file_path == f"{base}_1d.parquet"
    assert protein.structure_split.input_dataset_parquet_file_path == f"{base}_3d.parquet"
    assert protein.structure_split.drop_to_1d_dataset_parquet_file_path == f"{base}_1d.parquet"
    assert protein.structure_split.drop_to_1d
    assert dataset.collapsed_structure_split.input_with_protein_information_parquet_file_path == f"{base}_1d.parquet"


def test_classification_and_retrieval_share_the_foldseek_search() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        searches = [
            compose(
                config_name="data_processing",
                overrides=[f"data/splits_dataset@splits.dataset={name}"],
            ).splits.dataset
            for name in ("enzyme_classification", "enzyme_retrieval")
        ]
    classification = searches[0].structure_split
    retrieval = searches[1].with_protein_information.structure_split
    assert classification.shared_search_inputs
    assert classification.shared_search_inputs == retrieval.shared_search_inputs
    assert classification.shared_search_dir == retrieval.shared_search_dir


def test_mlp_pipeline_guide_documents_expected_configs() -> None:
    guide = (REPO_ROOT / "docs" / "baselines" / "mlp_pipeline.md").read_text()

    for expected in [
        "configs/hparams_search/baseline_regression_optuna4.yaml",
        "configs/model/baseline_regression.yaml",
        "configs/data/baseline_regression.yaml",
        "configs/trainer/default.yaml",
        "configs/logger/wandb.yaml",
        "scripts/run_baseline_mlp_pipeline.py",
    ]:
        assert expected in guide


def test_experimental_structures_come_from_mmcif_files() -> None:
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="data_processing.yaml",
            overrides=[f"paths.root_dir={rootutils.find_root(indicator='.project-root')}"],
        )

    assert cfg.embeddings.mmcif_path == cfg.splits.mmcif_path == cfg.paths.mmcif_path
    assert cfg.embeddings.uniprot_acc_path == cfg.paths.unirprot_acc_path
    assert "processed_exp_pdb_path" not in cfg.paths
    assert "pdb_path" not in cfg.paths
    for section in ("chem_utils", "embeddings", "splits"):
        assert "processed_exp_pdb_path" not in cfg[section]
    assert "ebi_pdb_url" not in cfg.chem_utils


def test_split_inputs_compose_without_the_dataset_builders(tmp_path) -> None:
    """The public export omits data/builders.yaml; split configs must still resolve."""
    import shutil

    configs = tmp_path / "configs"
    shutil.copytree(REPO_ROOT / "configs", configs, ignore=shutil.ignore_patterns("__pycache__"))
    (configs / "data" / "builders.yaml").unlink(missing_ok=True)
    GlobalHydra.instance().clear()
    with initialize_config_dir(str(configs), version_base="1.3"):
        cfg = compose(
            "data_processing",
            overrides=[
                f"paths.root_dir={REPO_ROOT}",
                "data/splits_dataset@splits.dataset=enzyme_classification",
            ],
        )
    OmegaConf.resolve(cfg)
    assert "chebi" not in cfg
    assert cfg.splits.dataset.sequence_split.input_dataset_parquet_file_path.endswith(
        "enzyme_classification_dataset_1d.parquet"
    )
