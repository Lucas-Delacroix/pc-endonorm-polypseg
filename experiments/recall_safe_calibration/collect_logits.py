from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.recall_safe_calibration.utils import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_TEST_SOURCE,
    collect_predictions,
    load_config,
    write_prediction_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect ESFPNet logits, probabilities and masks for calibration/evaluation."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--images", default=None)
    parser.add_argument("--masks", default=None)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--split", default="all")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    source_name = args.source_name
    if args.role == "test" and source_name is None and not args.images:
        source_name = DEFAULT_TEST_SOURCE

    predictions, metadata = collect_predictions(
        config,
        checkpoint=args.checkpoint,
        role=args.role,
        images_dir=args.images,
        masks_dir=args.masks,
        source_name=source_name,
        split=args.split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_choice=args.device,
    )
    metadata["config_path"] = args.config
    path = write_prediction_cache(Path(args.output), predictions, metadata, overwrite=args.overwrite)
    print(f"Saved {metadata['n_images']} samples to: {path}")


if __name__ == "__main__":
    main()
