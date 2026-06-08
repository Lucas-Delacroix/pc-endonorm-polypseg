import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from data.datamodule import PolypDataModule
from models import get_model
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models/esfpnet.yaml",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw/kvasir-seg",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Overrides config value when set",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Runs 3 epochs only to confirm the pipeline works",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_default_path(config_path: Path, default: str) -> Path:
    default_path = (config_path.parent / default).resolve()
    if default_path.suffix:
        return default_path

    for suffix in (".yaml", ".yml"):
        candidate = default_path.with_suffix(suffix)
        if candidate.exists():
            return candidate

    return default_path


def merge_config(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    config = read_yaml(config_path)
    defaults = config.pop("defaults", [])

    merged: dict = {}
    for default in defaults:
        if default == "_self_":
            continue
        if isinstance(default, str):
            default_config = load_config(str(resolve_default_path(config_path, default)))
            merged = merge_config(merged, default_config)
        else:
            raise TypeError(f"Unsupported default config entry: {default!r}")

    return merge_config(merged, config)


def truncate_dataset(dataset, max_samples: int) -> Subset:
    return Subset(dataset, range(min(max_samples, len(dataset))))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Seed: {seed}")

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    if args.smoke_test:
        original_epochs = config["training"].get("epochs", 200)
        config["training"].setdefault("scheduler", {})["t_max"] = original_epochs
        config["training"]["epochs"] = 3
        config["training"]["batch_size"] = min(
            config["training"].get("batch_size", 8),
            2,
        )
        config["data"]["image_size"] = min(
            config["data"].get("image_size", 352),
            128,
        )
        config["data"]["num_workers"] = 0
        config["data"]["pin_memory"] = False
        print("Smoke test mode: running 3 epochs on a small subset only.")

    training_config = config["training"]
    model_config = config["model"]

    training_config["checkpoint_dir"] = str(
        Path("checkpoints") / config["logging"]["run_name"]
    )

    print("\nInitializing DataModule...")
    dm = PolypDataModule(
        dataset_name=config["data"]["dataset"],
        data_root=args.data_root,
        image_size=config["data"]["image_size"],
        batch_size=training_config["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )
    dm.setup()

    if args.smoke_test:
        dm._train_dataset = truncate_dataset(dm._train_dataset, 8)
        dm._val_dataset = truncate_dataset(dm._val_dataset, 4)
        print("Smoke test subset:")
        print(f"  Train:    {len(dm._train_dataset)} images")
        print(f"  Validation: {len(dm._val_dataset)} images")

    print("\nBuilding model...")
    model = get_model(
        model_config["name"],
        num_classes=model_config["num_classes"],
        model_type=model_config.get("model_type", "b2"),
        pretrained_path=model_config.get("pretrained_path"),
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {model_config['name']} — {total_params:.1f}M parameters")

    print("\nStarting training...")
    trainer = Trainer(
        model=model,
        train_loader=dm.train_loader(),
        val_loader=dm.val_loader(),
        config=training_config,
    )

    history = trainer.fit()

    output_dir = Path("outputs") / config["logging"]["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    import csv
    csv_path = output_dir / "history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_dice",
            "val_loss", "val_dice", "val_iou",
            "val_precision", "val_recall", "lr",
        ])
        writer.writeheader()
        for record in history:
            writer.writerow({
                "epoch": record["epoch"],
                "train_loss": round(record["train"]["loss"], 6),
                "train_dice": round(record["train"]["dice"], 6),
                "val_loss": round(record["val"]["loss"], 6),
                "val_dice": round(record["val"]["dice"], 6),
                "val_iou": round(record["val"]["iou"], 6),
                "val_precision": round(record["val"]["precision"], 6),
                "val_recall": round(record["val"]["recall"], 6),
                "lr": record["lr"],
            })

    print(f"\nHistory saved to: {csv_path}")
    print(f"Best checkpoint: checkpoints/{config['logging']['run_name']}/best.pth")


if __name__ == "__main__":
    main()