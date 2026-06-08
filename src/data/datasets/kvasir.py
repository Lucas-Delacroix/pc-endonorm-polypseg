import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from data.datasets.base_dataset import BaseDataset


class KvasirDataset(BaseDataset):
    SPLIT_FILE = "data/splits/kvasir_split.json"
    SPLIT_SEED = 42

    def __init__(self, root: str, split: str, transform=None, image_size: int = 352):
        self.image_size = image_size
        super().__init__(root, split, transform)

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

    def __getitem__(self, idx: int) -> dict:
        image_path, mask_path = self.samples[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = (mask > 127).astype(np.float32)

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
            "mask": mask,
            "image_path": image_path,
        }
