import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.datamodule import PolypDataModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw/kvasir-seg",
        help="Path to the Kvasir-SEG root directory.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=352,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/data_artifacts",
        help="Directory where the visual artifacts are saved.",
    )
    return parser.parse_args()


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    overlay_color = np.array([1.0, 0.2, 0.1])
    mask_pixels = mask > 0.5

    overlay[mask_pixels] = overlay[mask_pixels] * 0.45 + overlay_color * 0.55
    return overlay


def save_artifacts(batch: dict, split: str, output_dir: Path):
    images = batch["image"]
    masks = batch["mask"]
    paths = batch["image_path"]
    total = min(4, len(images))

    fig, axes = plt.subplots(total, 3, figsize=(10, total * 3.2))
    fig.suptitle(f"{split}: image, mask, overlay", fontsize=14)

    if total == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(total):
        image = denormalize(images[row])
        mask = masks[row].squeeze().numpy()
        overlay = make_overlay(image, mask)
        image_name = Path(paths[row]).name

        samples = [
            (image, image_name, None),
            (mask, "mask", "gray"),
            (overlay, "overlay", None),
        ]

        for col, (sample, title, cmap) in enumerate(samples):
            axes[row][col].imshow(sample, cmap=cmap)
            axes[row][col].set_title(title)
            axes[row][col].axis("off")

    fig.tight_layout()

    output_path = output_dir / f"{split}_artifacts.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = PolypDataModule(
        dataset_name="kvasir",
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    loaders = {
        "train": dm.train_loader(),
        "val": dm.val_loader(),
        "test": dm.test_loader(),
    }

    for split, loader in loaders.items():
        batch = next(iter(loader))
        save_artifacts(batch, split, output_dir)

    print(f"Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
