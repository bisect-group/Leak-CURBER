import argparse
import logging
from importlib import import_module
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.rich_progress import component_logger  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"


SPLITTER_SPECS: dict[str, tuple[str, str, str]] = {
    "smiles": (
        "src.data.components.splitters.smiles_tanimoto",
        "SMILESMaxTanimotoSimilaritySplitter",
        "Generating substrate SMILES Tanimoto splits",
    ),
    "sequence": (
        "src.data.components.splitters.protein_seq_fident",
        "ProteinSeqMaxFidentSimilaritySplitter",
        "Generating protein sequence splits",
    ),
    "structure": (
        "src.data.components.splitters.protein_struct_lddt",
        "ProteinStructMaxLDDTSimilaritySplitter",
        "Generating protein structure splits",
    ),
    "random": (
        "src.data.components.splitters.random",
        "RandomSplitter",
        "Generating random splits",
    ),
    "cosine": (
        "src.data.components.splitters.embedding_cosine",
        "EmbeddingCosineSimilaritySplitter",
        "Generating embedding cosine splits",
    ),
    "time": (
        "src.data.components.splitters.uniprot_time",
        "UniProtTimeBasedSplitter",
        "Generating UniProt time splits",
    ),
    "reaction_tanimoto": (
        "src.data.components.splitters.reaction_tanimoto",
        "ReactionDRFPTanimotoSimilaritySplitter",
        "Generating reaction DRFP Tanimoto splits",
    ),
}
DEFAULT_SPLITTER_ORDER = (
    "smiles",
    "sequence",
    "structure",
    "random",
    "cosine",
    "time",
    "reaction_tanimoto",
)


def _activate_enzyme_retrieval_split_variant(cfg: DictConfig) -> None:
    dataset_cfg = cfg.splits.dataset
    if not cfg.enzyme_retrieval_dataset.get("include_protein_information", False):
        return
    if not dataset_cfg.get("with_protein_information"):
        return

    variant_cfg = dataset_cfg.with_protein_information
    dataset_path = variant_cfg.dataset_parquet_file_path
    output_dir = variant_cfg.output_dir

    with open_dict(dataset_cfg):
        dataset_cfg.dataset_parquet_file_path = dataset_path
        dataset_cfg.output_dir = output_dir
        dataset_cfg.random.input_dataset_parquet_file_path = dataset_path
        dataset_cfg.random.output_dir = f"{output_dir}/random_splits"
        dataset_cfg.time_split.input_dataset_parquet_file_path = dataset_path
        dataset_cfg.time_split.output_dir = f"{output_dir}/uniprot_time_splits"
        dataset_cfg.sequence_split = variant_cfg.sequence_split
        dataset_cfg.dataset_3d_parquet_file_path = (
            variant_cfg.dataset_3d_parquet_file_path
        )
        dataset_cfg.structure_split = variant_cfg.structure_split
        dataset_cfg.tanimoto_split_input_dataset_parquet_file_path = dataset_path
        dataset_cfg.tanimoto_split_output_dir = (
            f"{output_dir}/reaction_drfp_tanimoto_splits"
        )
        dataset_cfg.unique_tanimoto_similarity_plot_path = (
            f"{dataset_cfg.tanimoto_split_output_dir}/"
            "tanimoto_similarity_distribution_all_thresholds.png"
        )
        dataset_cfg.full_dataset_tanimoto_similarity_plot_path = (
            f"{dataset_cfg.tanimoto_split_output_dir}/"
            "full_dataset_reaction_tanimoto_similarity_distribution.png"
        )


def _prepare_dataset_split_cfg(dataset_name: str, cfg: DictConfig) -> DictConfig:
    if dataset_name == "enzyme_retrieval":
        _activate_enzyme_retrieval_split_variant(cfg)
    return cfg


def _dataset_has_splitter_cfg(cfg: DictConfig, splitter_name: str) -> bool:
    dataset_cfg = cfg.splits.dataset

    if splitter_name == "smiles":
        return bool(dataset_cfg.get("smiles_split"))
    if splitter_name == "sequence":
        return bool(dataset_cfg.get("sequence_split"))
    if splitter_name == "structure":
        return bool(dataset_cfg.get("structure_split"))
    if splitter_name == "random":
        random_cfg = dataset_cfg.get("random")
        if random_cfg is not None and not random_cfg.get("enabled", True):
            return False
        return bool(random_cfg) or bool(dataset_cfg.get("random_output_dir"))
    if splitter_name == "cosine":
        cosine_cfg = dataset_cfg.get("cosine_split")
        if not cosine_cfg:
            return False
        return bool(cosine_cfg.get("enabled", True))
    if splitter_name == "time":
        time_cfg = dataset_cfg.get("time_split")
        if time_cfg is not None and not time_cfg.get("enabled", True):
            return False
        return bool(time_cfg) or bool(dataset_cfg.get("sequence_split"))
    if splitter_name == "reaction_tanimoto":
        return bool(dataset_cfg.get("tanimoto_split_input_dataset_parquet_file_path"))

    raise ValueError(f"Unsupported splitter name: {splitter_name}")


def _run_splitter(splitter_name: str, cfg: DictConfig, log: logging.Logger) -> None:
    module_name, class_name, message = SPLITTER_SPECS[splitter_name]
    log.info(message)
    splitter_cls = getattr(import_module(module_name), class_name)
    splitter = splitter_cls(cfg=cfg)
    splitter.cfg = cfg  # lets the splitter write its config next to the output
    splitter.generate_splits()


def build_splitter_list(args: argparse.Namespace) -> list[str]:
    selected = list(args.splits) if args.splits else list(DEFAULT_SPLITTER_ORDER)
    skipped = set(args.skip_splits or [])
    return [name for name in selected if name not in skipped]


def run_splits_for_dataset(
    dataset_name: str,
    splitter_names: list[str],
    log: logging.Logger,
    overrides: list[str] | None = None,
) -> None:
    cfg = _prepare_dataset_split_cfg(
        dataset_name,
        compose(
            config_name="data_processing",
            overrides=[f"data/splits_dataset@splits.dataset={dataset_name}"]
            + (overrides or []),
        ),
    )
    log.info("Generating splits for dataset: %s", dataset_name)

    for splitter_name in splitter_names:
        if _dataset_has_splitter_cfg(cfg, splitter_name):
            _run_splitter(splitter_name, cfg, log)
        else:
            log.info("Skipping splitter '%s' for %s: no config.", splitter_name, dataset_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all configured dataset splits."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Optional dataset config names from configs/data/splits_dataset. "
            "If omitted, uses splits.datasets from config. *_unified datasets are skipped."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLITTER_SPECS),
        default=None,
        help="Optional subset of split families to run. Defaults to all supported families.",
    )
    parser.add_argument(
        "--skip-splits",
        nargs="+",
        choices=sorted(SPLITTER_SPECS),
        default=None,
        help="Optional split families to skip.",
    )
    parser.add_argument(
        "--list-splits",
        action="store_true",
        help="Print available split family names and exit.",
    )
    parser.add_argument(
        "--similarity-cache-overwrite",
        action="store_true",
        help="Recompute similarity caches instead of reusing existing cache files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_splits:
        for name in DEFAULT_SPLITTER_ORDER:
            print(name)
        return

    splitter_names = build_splitter_list(args)
    if not splitter_names:
        raise ValueError("No splits selected after applying --splits/--skip-splits.")

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        overrides = (
            ["splits.similarity_cache_overwrite=true"]
            if args.similarity_cache_overwrite
            else []
        )
        base_cfg = compose(config_name="data_processing", overrides=overrides)
        log = component_logger(
            log_dir=base_cfg.paths.log_dir, log_file_name="generate_splits.log"
        ).get_logger()
        dataset_names = args.datasets or list(base_cfg.splits.datasets)
        # Unified datasets are never split.
        unified = [name for name in dataset_names if name.endswith("_unified")]
        if unified:
            log.warning("Skipping unified datasets: %s", ", ".join(unified))
            dataset_names = [name for name in dataset_names if name not in unified]

        for dataset_name in dataset_names:
            run_splits_for_dataset(
                dataset_name=dataset_name,
                splitter_names=splitter_names,
                log=log,
                overrides=overrides,
            )


if __name__ == "__main__":
    main()
