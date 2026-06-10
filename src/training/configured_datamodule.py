from __future__ import annotations

from pathlib import Path

import yaml

from data.datamodule import PolypDataModule
from data.leakage import write_leakage_report
from data.multisource_datamodule import MultiSourcePolypDataModule


def is_multisource_config(config: dict) -> bool:
    dataset = config.get("dataset")
    return (
        isinstance(dataset, dict)
        and isinstance(dataset.get("train"), dict)
        and "sources" in dataset["train"]
    )


def checkpoint_dir_for(config: dict, multi_source: bool) -> Path:
    run_name = config["logging"]["run_name"]
    paths = config.get("paths", {})
    if multi_source:
        return Path(paths.get("results", "results")) / run_name
    return Path(paths.get("checkpoints", "checkpoints")) / run_name


def save_resolved_config(config: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config_resolved.yaml"
    with open(config_path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    print(f"Resolved config saved to: {config_path}")


def build_datamodule(config: dict, args, multi_source: bool):
    data_config = config["data"]
    training_config = config["training"]
    if multi_source:
        if data_config.get("input_mode", "rgb") != "rgb":
            raise ValueError("MS_baseline_rgb supports RGB input only.")
        if int(config["model"].get("in_channels", 3)) != 3:
            raise ValueError("MS_baseline_rgb must keep model.in_channels=3.")
        return MultiSourcePolypDataModule(
            dataset_config=config["dataset"],
            image_size=data_config["image_size"],
            batch_size=training_config["batch_size"],
            num_workers=data_config["num_workers"],
            pin_memory=data_config["pin_memory"],
            augmentation=config.get("augmentation"),
            base_dir=config["dataset"].get("base_dir"),
        )

    return PolypDataModule(
        dataset_name=data_config["dataset"],
        data_root=args.data_root,
        image_size=data_config["image_size"],
        batch_size=training_config["batch_size"],
        num_workers=data_config["num_workers"],
        pin_memory=data_config["pin_memory"],
        input_mode=data_config.get("input_mode", "rgb"),
        preprocessed_root=data_config.get("preprocessed_root"),
        augmentation=config.get("augmentation"),
    )


def smoke_check_batch(loader) -> None:
    batch = next(iter(loader))
    image_shape = tuple(batch["image"].shape)
    mask_shape = tuple(batch["mask"].shape)
    if len(image_shape) != 4 or image_shape[1] != 3:
        raise AssertionError(f"Expected image batch Bx3xHxW, got {image_shape}")
    if len(mask_shape) != 4 or mask_shape[1] != 1:
        raise AssertionError(f"Expected mask batch Bx1xHxW, got {mask_shape}")
    print(f"Smoke batch image shape: {image_shape}")
    print(f"Smoke batch mask shape:  {mask_shape}")
    if "dataset_name" in batch:
        print(f"Smoke batch dataset_name: {batch['dataset_name']}")


def write_train_leakage_report(config: dict, dm, multi_source: bool) -> None:
    if not multi_source or not hasattr(dm, "_test_datasets"):
        return
    output_path = (
        Path(config.get("paths", {}).get("outputs", "outputs"))
        / "tables"
        / f"{config['logging']['run_name']}_leakage_check.csv"
    )
    write_leakage_report(
        dm._train_dataset,
        dm._test_datasets,
        output_path,
        train_protocol=config["logging"]["run_name"],
    )
