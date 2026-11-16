import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from omegaconf import OmegaConf

from src.data.components.generators.atom_mapped_reaction import AtomMappedReactionGenerator, AtomMappingConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "embeddings.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate atom-mapped reaction SMILES with LocalMapper")
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=None)
    parser.add_argument("--model-version", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--postprocess-workers", type=int)
    parser.add_argument("--queue-size", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"LocalMapper configuration not found: {args.config}")
    embeddings = OmegaConf.load(args.config)
    overrides = {
        "localmapper_gpu_ids": args.gpu_ids,
        "localmapper_model_version": args.model_version,
        "localmapper_batch_size": args.batch_size,
        "localmapper_preprocess_workers": args.preprocess_workers,
        "localmapper_postprocess_workers": args.postprocess_workers,
        "localmapper_queue_size": args.queue_size,
        "localmapper_checkpoint_interval": args.checkpoint_interval,
    }
    for key, value in overrides.items():
        if value is not None:
            embeddings[key] = value
    generator = AtomMappedReactionGenerator(cfg=OmegaConf.create({"embeddings": embeddings}))
    generator.generate(args.input_path, args.output_path, resume=args.resume, limit=args.limit)


if __name__ == "__main__":
    main()
