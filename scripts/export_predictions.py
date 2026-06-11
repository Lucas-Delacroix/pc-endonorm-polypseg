from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data.datasets.external import ExternalPolypDataset
from data.datasets.kvasir import KvasirDataset
from data.transforms.augmentation import get_val_transforms
from models import get_model
from scripts.train import load_config
from training.configured_datamodule import checkpoint_dir_for, is_multisource_config

DEFAULT_CONFIG = "configs/models/esfpnet.yaml"
DEFAULT_DATA_ROOT = "data/raw/kvasir-seg"
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT_ROOT = "outputs/predictions"
PROBABILITY_SCALE = 255


def default_checkpoint_path(config: dict) -> Path:
    multi_source = is_multisource_config(config)
    return checkpoint_dir_for(config, multi_source) / "best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=("train", "val", "test"))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preprocessed-root", default=None)
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def original_size(image_path: str) -> tuple[int, int]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Invalid image: {image_path}")
    return image.shape[0], image.shape[1]


def save_prediction(probability: np.ndarray, image_path: str, output_dir: Path) -> None:
    height, width = original_size(image_path)
    resized = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = (resized * PROBABILITY_SCALE).round().astype(np.uint8)
    output_path = output_dir / f"{Path(image_path).stem}.png"
    cv2.imwrite(str(output_path), mask)


@torch.no_grad()
def export(model: torch.nn.Module, dataset: KvasirDataset, device: str, output_dir: Path) -> int:
    model.eval()
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        logits = model(image)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        save_prediction(probability, sample["image_path"], output_dir)
    return len(dataset)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_config = config["model"]
    run_name = config["logging"]["run_name"]
    image_size = config["data"]["image_size"]

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(config)
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run training first (make train)."
        )

    device = resolve_device(args.device)

    model = get_model(
        model_config["name"],
        num_classes=model_config["num_classes"],
        model_type=model_config.get("model_type", "b2"),
        in_channels=model_config.get("in_channels", 3),
        pretrained_path=None,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    dataset_cls = ExternalPolypDataset if args.external else KvasirDataset
    dataset = dataset_cls(
        root=args.data_root,
        split=args.split,
        transform=get_val_transforms(image_size),
        image_size=image_size,
        input_mode=config["data"].get("input_mode", "rgb"),
        preprocessed_root=args.preprocessed_root or config["data"].get("preprocessed_root"),
    )

    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = export(model, dataset, device, output_dir)

    print(f"Exported {exported} predictions to: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
