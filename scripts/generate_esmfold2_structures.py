import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold protein sequences with ESMFold2.")
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="GPU IDs to use. Defaults to embeddings.gpu_ids.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Do not fold sequences longer than this. Overrides embeddings.esmfold2_max_sequence_length.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fold one short test sequence on GPU 0 and exit.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Extra Hydra overrides passed to compose(), e.g. embeddings.esmfold2_skip_known_structures=true.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides or [])
    if args.gpu_ids is not None:
        overrides.append(f"embeddings.gpu_ids=[{','.join(map(str, args.gpu_ids))}]")
    if args.max_length is not None:
        overrides.append(f"embeddings.esmfold2_max_sequence_length={args.max_length}")

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="data_processing", overrides=overrides)

    from src.data.components.generators.esmfold2_structure import ESMFold2StructureGenerator

    generator = ESMFold2StructureGenerator(cfg)
    if args.smoke:
        generator.run_smoke_test()
    else:
        generator.generate()


if __name__ == "__main__":
    main()
