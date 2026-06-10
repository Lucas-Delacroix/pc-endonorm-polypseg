from __future__ import annotations

import hashlib
import json
import random
import re
from bisect import bisect_right
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MASK_THRESHOLD = 127
SPLIT_NAMES = ("train", "val", "test")


def normalized_name(value: str | Path) -> str:
    return re.sub(r"[^a-z0-9]+", "", Path(value).stem.lower())


def stable_int(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16)


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir) / resolved
    return resolved


def match_mask(masks_dir: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = masks_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No mask found for image '{stem}' in {masks_dir}")


def list_image_mask_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[str, str]]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    images = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {images_dir}")

    return [(str(image), str(match_mask(masks_dir, image.stem))) for image in images]


def source_dirs(source: dict, base_dir: str | Path | None = None) -> tuple[Path, Path]:
    root = source.get("root")
    root_dir = resolve_path(root, base_dir) if root else None

    images_dir = source.get("images_dir")
    masks_dir = source.get("masks_dir")
    if images_dir is None and root_dir is not None:
        images_dir = "images"
    if masks_dir is None and root_dir is not None:
        masks_dir = "masks"
    if images_dir is None or masks_dir is None:
        raise ValueError(f"Source '{source.get('name', '<unnamed>')}' needs images_dir and masks_dir or root.")

    return resolve_path(images_dir, root_dir or base_dir), resolve_path(masks_dir, root_dir or base_dir)


def split_key(source: dict, images_dir: Path, masks_dir: Path) -> str:
    return str(source.get("split_group") or f"{images_dir.resolve()}::{masks_dir.resolve()}")


def load_split_selection(split_file: Path, split: str) -> list[int | str]:
    with open(split_file) as file:
        payload = json.load(file)
    if split not in payload:
        raise KeyError(f"Split '{split}' not found in {split_file}")
    selection = payload[split]
    if not isinstance(selection, list):
        raise TypeError(f"Split '{split}' in {split_file} must be a list.")
    return selection


def select_from_split_file(
    pairs: list[tuple[str, str]],
    selection: list[int | str],
) -> list[tuple[str, str]]:
    if not selection:
        return []
    if all(isinstance(item, int) for item in selection):
        return [pairs[item] for item in selection]

    by_key = {}
    for pair in pairs:
        image_path = Path(pair[0])
        by_key[normalized_name(image_path)] = pair
        by_key[image_path.name.lower()] = pair

    selected = []
    missing = []
    for item in selection:
        key = str(item).lower()
        pair = by_key.get(key) or by_key.get(normalized_name(key))
        if pair is None:
            missing.append(str(item))
            continue
        selected.append(pair)

    if missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} split entries were not found. Examples: {examples}")
    return selected


def random_split_indices(total: int, split_config: dict, key: str) -> dict[str, list[int]]:
    seed = int(split_config.get("seed", 42)) + stable_int(key)
    train_ratio = float(split_config.get("train_ratio", 0.9))
    val_ratio = float(split_config.get("val_ratio", 0.1))
    test_ratio = float(split_config.get("test_ratio", max(0.0, 1.0 - train_ratio - val_ratio)))

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Random split ratios must sum to a positive value.")
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum

    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    return {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }


def select_pairs_for_split(
    pairs: list[tuple[str, str]],
    source: dict,
    split_config: dict | None,
    images_dir: Path,
    masks_dir: Path,
) -> tuple[list[tuple[str, str]], str]:
    split = source.get("split")
    if split in (None, "all"):
        return pairs, "all files"
    if split not in SPLIT_NAMES:
        raise ValueError(f"Invalid split '{split}' for source '{source.get('name', '<unnamed>')}'.")

    split_file = source.get("split_file")
    if split_file:
        split_path = resolve_path(split_file)
        selected = select_from_split_file(pairs, load_split_selection(split_path, split))
        return selected, f"predefined split file {split_path}:{split}"

    mode = (split_config or {}).get("mode", "predefined")
    if mode == "random":
        indices = random_split_indices(len(pairs), split_config or {}, split_key(source, images_dir, masks_dir))[split]
        return [pairs[index] for index in indices], f"random split '{split}'"
    if mode == "predefined":
        return pairs, "predefined directory"

    raise ValueError(f"Unsupported split mode '{mode}'.")


class SegmentationSourceDataset(Dataset):
    def __init__(
        self,
        name: str,
        samples: Iterable[tuple[str, str]],
        transform=None,
    ):
        self.dataset_name = name
        self.samples = list(samples)
        self.transform = transform
        if not self.samples:
            raise ValueError(f"Source '{name}' has no samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_rgb(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Invalid image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Invalid mask: {path}")
        return (mask > MASK_THRESHOLD).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        image_path, mask_path = self.samples[idx]
        image = self._read_rgb(image_path)
        mask = self._read_mask(mask_path)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        elif isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask.float(),
            "image_path": image_path,
            "mask_path": mask_path,
            "dataset_name": self.dataset_name,
        }


class MultiSourceSegmentationDataset(Dataset):
    def __init__(self, datasets: list[SegmentationSourceDataset]):
        if not datasets:
            raise ValueError("At least one source dataset is required.")
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    @property
    def source_counts(self) -> dict[str, int]:
        return {dataset.dataset_name: len(dataset) for dataset in self.datasets}

    @property
    def samples(self) -> list[tuple[str, str]]:
        return [sample for dataset in self.datasets for sample in dataset.samples]

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += len(self)
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        previous_size = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][idx - previous_size]


def build_source_dataset(
    source: dict,
    transform,
    split_config: dict | None = None,
    base_dir: str | Path | None = None,
) -> tuple[SegmentationSourceDataset, str]:
    if "name" not in source:
        raise ValueError("Every source must define a name.")
    images_dir, masks_dir = source_dirs(source, base_dir)
    pairs = list_image_mask_pairs(images_dir, masks_dir)
    selected, split_description = select_pairs_for_split(pairs, source, split_config, images_dir, masks_dir)
    dataset = SegmentationSourceDataset(source["name"], selected, transform=transform)
    return dataset, split_description
