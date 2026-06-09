import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing.clahe import compute_clahe_rgb
from preprocessing.endonorm import compute_endonorm_rgb
from preprocessing.morphology_maps import compute_morph_map
from preprocessing.phase_congruency import compute_phase_congruency_map

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
COMPONENTS = ("endonorm", "clahe", "phase", "morph")
COMPONENT_DIRS = {"endonorm": "images_norm", "clahe": "images_clahe", "phase": "phase", "morph": "morph"}
DEBUG_SAMPLES = 6
PROGRESS_EVERY = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--config", default=None)
    parser.add_argument("--only", nargs="*", default=None, choices=COMPONENTS)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_config(path):
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def read_rgb(path, size):
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size))


def save_rgb(path, image):
    cv2.imwrite(str(path), cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def save_debug(path, rgb, norm, clahe, phase, morph, debug):
    panels = [
        (rgb, "RGB", None),
        (norm, "EndoNorm", None),
        (clahe, "CLAHE", None),
        (debug["illumination_field"], "illumination", "magma"),
        (debug["specular_mask"], "specular", "gray"),
        (phase, "phase", "inferno"),
        (morph, "morph", "viridis"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2))
    for ax, (data, title, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    config = read_config(args.config)
    selected = set(args.only) if args.only else set(COMPONENTS)
    full_run = args.only is None

    images = [p for p in sorted(Path(args.images_dir).glob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if args.limit:
        images = images[: args.limit]

    output = Path(args.output_dir)
    for component in selected:
        (output / COMPONENT_DIRS[component]).mkdir(parents=True, exist_ok=True)
    if full_run:
        (output / "debug").mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(images):
        rgb = read_rgb(image_path, args.image_size)
        stem = image_path.stem
        norm = clahe = phase = morph = debug = None

        if "endonorm" in selected:
            norm, debug = compute_endonorm_rgb(rgb, config.get("endonorm"))
            save_rgb(output / COMPONENT_DIRS["endonorm"] / f"{stem}.png", norm)
        if "clahe" in selected:
            clahe = compute_clahe_rgb(rgb, config.get("clahe"))
            save_rgb(output / COMPONENT_DIRS["clahe"] / f"{stem}.png", clahe)
        if "phase" in selected:
            phase = compute_phase_congruency_map(rgb, config.get("phase_congruency"))
            np.save(output / COMPONENT_DIRS["phase"] / f"{stem}.npy", phase)
        if "morph" in selected:
            morph = compute_morph_map(rgb, config.get("morphology"))
            np.save(output / COMPONENT_DIRS["morph"] / f"{stem}.npy", morph)

        if full_run and index < DEBUG_SAMPLES:
            save_debug(output / "debug" / f"{stem}.png", rgb / 255.0, norm, clahe, phase, morph, debug)

        if (index + 1) % PROGRESS_EVERY == 0 or index + 1 == len(images):
            print(f"{args.dataset_name}: {index + 1}/{len(images)}")


if __name__ == "__main__":
    main()
