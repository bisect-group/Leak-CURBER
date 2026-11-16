import argparse
from importlib import import_module
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "configs"

EMBEDDER_SPECS: dict[str, tuple[str, str]] = {
    "atom_pair": (
        "src.data.components.embedders.atom_pair",
        "SmilesAtomPairShardEmbedder",
    ),
    "drfp": (
        "src.data.components.embedders.drfp",
        "DRFPShardEmbedder",
    ),
    "ecfp": (
        "src.data.components.embedders.ecfp",
        "SmilesECFPShardEmbedder",
    ),
    "esm3": (
        "src.data.components.embedders.esm3",
        "ESM3StructureShardEmbedder",
    ),
    "esmc": (
        "src.data.components.embedders.esmc",
        "ESMCShardEmbedder",
    ),
    "molr": (
        "src.data.components.embedders.molr",
        "MolR2DShardEmbedder",
    ),
    "rxnfp": (
        "src.data.components.embedders.rxnfp",
        "RxnFPShardEmbedder",
    ),
    "smited": (
        "src.data.components.embedders.smited",
        "SMITEDShardEmbedder",
    ),
    "unimol_sdf": (
        "src.data.components.embedders.unimol",
        "UnimolSDFShardEmbedder",
    ),
}


# Embedders whose inputs do not change per dataset are built once and reused, so
# their models load once. DRFP reads dataset-specific inputs.
DATASET_DEPENDENT = {"drfp"}
_EMBEDDER_CACHE: dict[str, object] = {}


def run_embedder(
    embedder_name: str,
    cfg: DictConfig,
    args: argparse.Namespace,
) -> None:
    module_name, class_name = EMBEDDER_SPECS[embedder_name]
    embedder = _EMBEDDER_CACHE.get(embedder_name)
    if embedder is None:
        embedder = getattr(import_module(module_name), class_name)(cfg)
        if embedder_name not in DATASET_DEPENDENT:
            _EMBEDDER_CACHE[embedder_name] = embedder

    if embedder_name == "unimol_sdf":
        embedder.embed(input_override=args.unimol_input_path)
        return

    embedder.embed()


def build_embedder_list(args: argparse.Namespace) -> list[str]:
    selected = list(args.embedders) if args.embedders else list(EMBEDDER_SPECS)
    skipped = set(args.skip or [])
    return [name for name in selected if name not in skipped]


def run_embedders_for_dataset(
    dataset_name: str,
    embedder_names: list[str],
    args: argparse.Namespace,
) -> list[str]:
    overrides = [f"data/splits_dataset@splits.dataset={dataset_name}"]
    overrides.extend(args.overrides or [])
    cfg = compose(
        config_name="data_processing",
        overrides=overrides,
    )
    if "drfp" in embedder_names:
        drfp_input_path = cfg.splits.dataset.get(
            "tanimoto_split_input_dataset_parquet_file_path"
        )
        if drfp_input_path:
            with open_dict(cfg.embeddings):
                cfg.embeddings.drfp_embeddings_input_dataset_parquet_file_path = (
                    drfp_input_path
                )
    print(f"Running embedders for dataset: {dataset_name}")

    failed = []
    for embedder_name in embedder_names:
        print(f"  -> {embedder_name}")
        if args.continue_on_error:
            try:
                run_embedder(embedder_name, cfg, args)
            except Exception as exc:
                print(f"     failed: {exc}")
                failed.append(f"{dataset_name}/{embedder_name}")
        else:
            run_embedder(embedder_name, cfg, args)
    return failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shard-based embedders from the project Hydra config."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Optional dataset config names from configs/data/splits_dataset. "
            "If omitted, uses splits.datasets from config. This mostly matters for "
            "dataset-aware embedders such as UniMol SDF."
        ),
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        choices=sorted(EMBEDDER_SPECS),
        default=None,
        help="Optional subset of embedders to run. Defaults to all.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=sorted(EMBEDDER_SPECS),
        default=None,
        help="Optional embedders to skip.",
    )
    parser.add_argument(
        "--list-embedders",
        action="store_true",
        help="Print available embedder names and exit.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later embedders if one embedder fails.",
    )
    parser.add_argument(
        "--unimol-input-path",
        default=None,
        help="Optional SMILES pickle override passed only to the UniMol SDF embedder.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Extra Hydra overrides passed to compose(), e.g. embeddings.foo=/tmp/input.pkl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_embedders:
        for name in EMBEDDER_SPECS:
            print(name)
        return

    embedder_names = build_embedder_list(args)
    if not embedder_names:
        raise ValueError("No embedders selected after applying --embedders/--skip.")

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        base_cfg = compose(config_name="data_processing")
        dataset_names = args.datasets or list(base_cfg.splits.datasets)

        failed = []
        for dataset_name in dataset_names:
            failed += run_embedders_for_dataset(
                dataset_name=dataset_name,
                embedder_names=embedder_names,
                args=args,
            )
    if failed:
        raise SystemExit(f"Embedders failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
