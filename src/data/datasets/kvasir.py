import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from data.datasets.base_dataset import BaseDataset
from data.transforms.augmentation import get_geometric_transforms, get_photometric_transforms

MODE_COMPONENTS = {
    "rgb": ("rgb",),
    "rgb_norm": ("rgb", "norm"),
    "rgb_phase": ("rgb", "phase"),
    "rgb_norm_phase": ("rgb", "norm", "phase"),
    "rgb_norm_phase_morph": ("rgb", "norm", "phase", "morph"),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAP_MEAN = (0.5,)
MAP_STD = (0.5,)
COMPONENT_STATS = {
    "rgb": (IMAGENET_MEAN, IMAGENET_STD),
    "norm": (IMAGENET_MEAN, IMAGENET_STD),
    "phase": (MAP_MEAN, MAP_STD),
    "morph": (MAP_MEAN, MAP_STD),
}
COMPONENT_DIRS = {"norm": "images_norm", "phase": "phase", "morph": "morph"}
MASK_THRESHOLD = 127


class KvasirDataset(BaseDataset):
    SPLIT_FILE = "data/splits/kvasir_split.json"
    SPLIT_SEED = 42

    def __init__(self, root, split, transform=None, image_size=352, input_mode="rgb", preprocessed_root=None):
        self.image_size = image_size
        self.input_mode = input_mode
        self.components = MODE_COMPONENTS[input_mode]
        self.preprocessed_root = Path(preprocessed_root) if preprocessed_root else None
        super().__init__(root, split, transform)

        if self.components != ("rgb",):
            self.photometric = get_photometric_transforms()
            extra = {name: "image" for name in self.components if name != "rgb"}
            self.geometric = get_geometric_transforms(image_size, split == "train", extra)
            self.mean, self.std = self._channel_stats()

    def _load_samples(self) -> list:
        root = Path(self.root)
        images_dir = root / "images"
        masks_dir = root / "masks"

        if not root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {root}\n"
                "Run: uv run python -m scripts.download_dataset --dataset kvasir"
            )
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        all_images = sorted(images_dir.glob("*.jpg"))
        if not all_images:
            raise ValueError(f"No .jpg images found in {images_dir}")

        all_masks = [masks_dir / img.name for img in all_images]
        missing_masks = [mask_path for mask_path in all_masks if not mask_path.exists()]

        if missing_masks:
            examples = ", ".join(str(path) for path in missing_masks[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_masks)} masks for Kvasir images. Examples: {examples}"
            )

        split_indices = self._get_or_create_split(len(all_images))
        indices = split_indices[self.split]
        if not indices:
            raise ValueError(
                f"Split '{self.split}' is empty in {self.SPLIT_FILE}. "
                "Remove the split file and run again."
            )

        return [(str(all_images[i]), str(all_masks[i])) for i in indices]

    def _get_or_create_split(self, total: int) -> dict:
        split_path = Path(self.SPLIT_FILE)

        if split_path.exists():
            with open(split_path) as f:
                split_indices = json.load(f)

            split_total = sum(len(split_indices.get(split, [])) for split in ("train", "val", "test"))
            if split_indices.get("total") == total or split_total == total:
                return split_indices

        split_path.parent.mkdir(parents=True, exist_ok=True)

        indices = list(range(total))
        random.seed(self.SPLIT_SEED)
        random.shuffle(indices)

        n_train = int(total * 0.8)
        n_val = int(total * 0.1)

        splits = {
            "dataset": "kvasir",
            "total": total,
            "seed": self.SPLIT_SEED,
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        }

        with open(split_path, "w") as f:
            json.dump(splits, f, indent=2)

        return splits

    def _channel_stats(self):
        means, stds = [], []
        for name in self.components:
            mean, std = COMPONENT_STATS[name]
            means.extend(mean)
            stds.extend(std)
        mean = np.array(means, dtype=np.float32).reshape(1, 1, -1)
        std = np.array(stds, dtype=np.float32).reshape(1, 1, -1)
        return mean, std

    def _read_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        return (mask > MASK_THRESHOLD).astype(np.float32)

    def _read_rgb(self, image_path):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (self.image_size, self.image_size))

    def _read_map(self, stem, name):
        if name == "norm":
            path = self.preprocessed_root / COMPONENT_DIRS[name] / f"{stem}.png"
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            return cv2.resize(image, (self.image_size, self.image_size)).astype(np.float32) / 255.0

        path = self.preprocessed_root / COMPONENT_DIRS[name] / f"{stem}.npy"
        array = np.load(path).astype(np.float32)
        if array.shape != (self.image_size, self.image_size):
            array = cv2.resize(array, (self.image_size, self.image_size))
        return array

    def __getitem__(self, idx: int) -> dict:
        image_path, mask_path = self.samples[idx]
        if self.input_mode == "rgb":
            return self._rgb_item(image_path, mask_path)
        return self._multichannel_item(image_path, mask_path)

    def _rgb_item(self, image_path, mask_path):
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

        return {"image": image, "mask": mask, "image_path": image_path}

    def _multichannel_item(self, image_path, mask_path):
        stem = Path(image_path).stem
        rgb = self._read_rgb(image_path)
        mask = self._read_mask(mask_path)

        if self.split == "train":
            rgb = self.photometric(image=rgb)["image"]

        targets = {"image": rgb.astype(np.float32) / 255.0, "mask": mask}
        for name in self.components:
            if name != "rgb":
                targets[name] = self._read_map(stem, name)

        out = self.geometric(**targets)

        layers = [out["image"]]
        for name in self.components:
            if name == "rgb":
                continue
            layer = out[name]
            layers.append(layer if layer.ndim == 3 else layer[..., None])

        stacked = (np.concatenate(layers, axis=2) - self.mean) / self.std

        return {
            "image": torch.from_numpy(stacked.transpose(2, 0, 1).copy()).float(),
            "mask": torch.from_numpy(out["mask"]).unsqueeze(0).float(),
            "image_path": image_path,
        }
