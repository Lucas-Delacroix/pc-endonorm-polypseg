from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "raw" / "kvasir-seg"
DEFAULT_OUTPUT = ROOT / "upstream" / "data"
SPLIT_FILE = ROOT / "data" / "splits" / "kvasir_split.json"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Prepared Kvasir-SEG root.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Upstream data output root.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking where possible.")
    parser.add_argument("--force", action="store_true", help="Replace existing output root.")
    return parser.parse_args()


def load_or_create_split(total: int) -> dict[str, list[int]]:
    if SPLIT_FILE.exists():
        with open(SPLIT_FILE) as file:
            split = json.load(file)
        split_total = sum(len(split.get(name, [])) for name in ("train", "val", "test"))
        if split.get("total") == total or split_total == total:
            return {name: split[name] for name in ("train", "val", "test")}

    indices = list(range(total))
    random.seed(SEED)
    random.shuffle(indices)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    return {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }


def ensure_empty(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {path}. Use --force to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy:
        shutil.copy2(source, destination)
    else:
        os.symlink(source.resolve(), destination)


def mirror_split(samples: list[tuple[Path, Path]], indices: list[int], destination: Path, copy: bool) -> None:
    for idx in indices:
        image, mask = samples[idx]
        link_or_copy(image, destination / "images" / image.name, copy)
        link_or_copy(mask, destination / "masks" / mask.name, copy)

def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    images_dir = source / "images"
    masks_dir = source / "masks"

    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg images found in {images_dir}")
    samples = [(image, masks_dir / image.name) for image in images]
    missing = [mask for _, mask in samples if not mask.exists()]
    if missing:
        raise FileNotFoundError(f"Missing masks, first example: {missing[0]}")

    split = load_or_create_split(len(samples))
    ensure_empty(output, args.force)

    for split_name, indices in split.items():
        mirror_split(samples, indices, output / "kvasir" / split_name, args.copy)
    mirror_split(samples, list(range(len(samples))), output / "kvasir" / "all", args.copy)

    for split_name, indices in split.items():
        print(f"{split_name}: {len(indices)}")
    print(f"Upstream data prepared at: {output}")


if __name__ == "__main__":
    main()
