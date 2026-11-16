import argparse
import pickle
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structures with ESM3 from embeddings.esm3_pdbs_input_pkl_path."
    )
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="GPU IDs to use. Defaults to embeddings.gpu_ids.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Extra Hydra overrides passed to compose(), e.g. embeddings.esm3_pdbs_input_pkl_path=data/intermediate/esm3_fallback.pkl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides or [])
    if args.gpu_ids is not None:
        overrides.append(f"embeddings.gpu_ids=[{','.join(map(str, args.gpu_ids))}]")

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="data_processing", overrides=overrides)

    from src.data.components.generators.esm3_pdb import ESM3PDBGenerator

    generator = ESM3PDBGenerator(cfg)
    with open(generator.INPUT_FILE_PATH, "rb") as handle:
        n_records = len(pickle.load(handle))
    generator.logger.info(f"Processing {n_records:,} records from {generator.INPUT_FILE_PATH}.")
    generator.generate()


if __name__ == "__main__":
    main()
