from __future__ import annotations

import torch

from data.datasets.kvasir import KvasirDataset
from data.transforms.robust_style import (
    build_appearance_transforms,
    build_geometric_transforms,
    build_normalize_transforms,
)


class ConsistencyViews:
    def __init__(self, image_size: int, augmentation: dict | None):
        self.geometric = build_geometric_transforms(image_size)
        self.appearance = build_appearance_transforms(augmentation)
        self.normalize = build_normalize_transforms()

    def __call__(self, image, mask) -> dict:
        geometric = self.geometric(image=image, mask=mask)
        shared_image = geometric["image"]
        shared_mask = geometric["mask"]

        weak = self.normalize(image=shared_image)["image"]
        strong = self.normalize(image=self.appearance(image=shared_image)["image"])["image"]
        mask_tensor = torch.from_numpy(shared_mask).unsqueeze(0).float()

        return {"image_weak": weak, "image_strong": strong, "mask": mask_tensor}


class ConsistencyKvasirDataset(KvasirDataset):
    def _rgb_item(self, image_path, mask_path):
        image = self._read_rgb(image_path)
        mask = self._read_mask(mask_path)
        views = self.transform(image=image, mask=mask)
        return {**views, "image_path": image_path}
