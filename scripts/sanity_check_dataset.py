import argparse
from pathlib import Path

import numpy as np

from data.datasets.kvasir import KvasirDataset

EXPECTED_CHANNELS = {
    "rgb": 3,
    "rgb_norm": 6,
    "rgb_phase": 4,
    "rgb_norm_phase": 7,
    "rgb_norm_phase_morph": 8,
}
IMAGE_SIZE = 352


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/kvasir-seg")
    parser.add_argument("--preprocessed-root", default="data/preprocessed/kvasir-seg")
    return parser.parse_args()


def find_ready_sample(mode, data_root, preprocessed_root):
    phase_dir = Path(preprocessed_root) / "phase"
    for split in ("train", "val", "test"):
        dataset = KvasirDataset(
            root=data_root,
            split=split,
            image_size=IMAGE_SIZE,
            input_mode=mode,
            preprocessed_root=preprocessed_root,
        )
        if mode == "rgb":
            return dataset, 0
        for index in range(len(dataset)):
            stem = Path(dataset.samples[index][0]).stem
            if (phase_dir / f"{stem}.npy").exists():
                return dataset, index
    return None, None


def main():
    args = parse_args()
    for mode, expected in EXPECTED_CHANNELS.items():
        dataset, index = find_ready_sample(mode, args.data_root, args.preprocessed_root)
        if dataset is None:
            print(f"  {mode}: skipped (no preprocessed sample available)")
            continue

        item = dataset[index]
        image, mask = item["image"], item["mask"]
        assert image.shape == (expected, IMAGE_SIZE, IMAGE_SIZE), f"{mode}: image {tuple(image.shape)}"
        assert mask.shape == (1, IMAGE_SIZE, IMAGE_SIZE), f"{mode}: mask {tuple(mask.shape)}"
        assert not np.isnan(image.numpy()).any(), f"{mode}: NaN in image"
        assert set(np.unique(mask.numpy())).issubset({0.0, 1.0}), f"{mode}: mask not binary"
        print(f"  {mode}: image {tuple(image.shape)} mask {tuple(mask.shape)} range=[{image.min():.2f},{image.max():.2f}] ok")

    print("Dataset sanity checks passed.")


if __name__ == "__main__":
    main()
