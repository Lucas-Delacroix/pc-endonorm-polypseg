from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.transforms.robust_style import (
    FourierAmplitudeRandomization,
    _strong_color_transforms,
    build_randconv_transform,
)
from scripts.train import load_config

DEFAULT_CONFIG = "configs/experiments/K_robust_full.yaml"
DEFAULT_DATA_ROOT = "data/raw/kvasir-seg"
DEFAULT_OUTPUT_DIR = "results/debug_augmentations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the robust style, Fourier and RandConv augmentations.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=352)
    return parser.parse_args()


def read_rgb(path: Path, size: int) -> np.ndarray:
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size))


def read_mask(path: Path, size: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(mask, (size, size))


def build_transforms(augmentation: dict) -> dict:
    strong = augmentation.get("strong_style", {})
    fourier_config = augmentation.get("fourier", {})
    randconv_config = dict(augmentation.get("randconv", {}))
    randconv_config["p"] = 1.0

    color_transforms = _strong_color_transforms(strong) if strong.get("enabled", True) else []
    color = A.Compose(color_transforms) if color_transforms else None

    fourier = FourierAmplitudeRandomization(
        beta=fourier_config.get("beta", 0.05),
        strength=fourier_config.get("strength", 0.15),
        low_freq_only=fourier_config.get("low_freq_only", True),
        p=1.0,
    )
    randconv = build_randconv_transform(randconv_config)

    return {
        "strong": color,
        "fourier": fourier,
        "randconv": randconv,
        "fourier_randconv": A.Compose([fourier, randconv]),
        "full": A.Compose(color_transforms + [fourier, randconv]),
    }


def save_panel(output_path: Path, panels: list[tuple[np.ndarray, str]]) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2))
    for axis, (data, title) in zip(axes, panels):
        axis.imshow(data, cmap="gray" if data.ndim == 2 else None)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    augmentation = config.get("augmentation", {})
    transforms = build_transforms(augmentation)

    images_dir = Path(args.data_root) / "images"
    masks_dir = Path(args.data_root) / "masks"
    image_paths = sorted(images_dir.glob("*.jpg"))[: args.num_samples]

    output_dir = Path(args.output_dir)
    if augmentation.get("randconv", {}).get("enabled", False) and args.output_dir == DEFAULT_OUTPUT_DIR:
        output_dir = output_dir / "randconv"
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        rgb = read_rgb(image_path, args.image_size)
        mask = read_mask(masks_dir / image_path.name, args.image_size)

        strong_image = transforms["strong"](image=rgb)["image"] if transforms["strong"] is not None else rgb

        save_panel(output_dir / f"{image_path.stem}.png", [
            (rgb, "RGB"),
            (strong_image, "strong style"),
            (transforms["fourier"](image=rgb)["image"], "fourier"),
            (transforms["randconv"](image=rgb)["image"], "randconv"),
            (transforms["fourier_randconv"](image=rgb)["image"], "fourier+randconv"),
            (transforms["full"](image=rgb)["image"], "full"),
            (mask, "mask"),
        ])
        print(f"Saved {output_dir / f'{image_path.stem}.png'}")


if __name__ == "__main__":
    main()
