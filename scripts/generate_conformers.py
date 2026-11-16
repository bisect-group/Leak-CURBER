import argparse
import hashlib
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.components.embedders.utils import load_pickle_items
from src.data.components.generators.datamol_conformer import (
    Datamol3DConformerGenerator,
)

CONFIG_DIR = REPO_ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Datamol conformers for the SMILES superset."
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Optional SMILES pickle override. Defaults to the configured conformer input pickle.",
    )
    return parser.parse_args()


def build_smiles_dict(input_path: Path) -> dict[str, str]:
    """Map sha256(smiles_key) -> smiles_key for every unique key in a superset pickle."""
    keys = sorted(set(load_pickle_items(input_path, "smiles_key")))
    return {hashlib.sha256(key.encode()).hexdigest(): key for key in keys}


def main() -> None:
    args = parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="data_processing")
    input_path = Path(args.input_path or cfg.embeddings.datamol_conformer_input_pkl_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input pickle not found at {input_path}")

    smiles_dict = build_smiles_dict(input_path)
    print(f"Generating conformers from {input_path} ({len(smiles_dict):,} unique smiles_key values)")
    Datamol3DConformerGenerator(cfg).generate(smiles_dict=smiles_dict)


if __name__ == "__main__":
    main()
