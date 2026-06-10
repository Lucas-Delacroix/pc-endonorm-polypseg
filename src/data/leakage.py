from __future__ import annotations

import csv
from pathlib import Path

from torch.utils.data import Subset

from data.datasets.multisource import normalized_name


def dataset_samples(dataset) -> list[tuple[str, str]]:
    if isinstance(dataset, Subset):
        samples = dataset_samples(dataset.dataset)
        return [samples[index] for index in dataset.indices]
    if hasattr(dataset, "datasets"):
        return [sample for child in dataset.datasets for sample in dataset_samples(child)]
    if hasattr(dataset, "samples"):
        return list(dataset.samples)
    raise TypeError(f"Cannot extract samples from dataset type {type(dataset)!r}")


def image_index(samples: list[tuple[str, str]]) -> dict[str, str]:
    return {normalized_name(image_path): image_path for image_path, _ in samples}


def write_leakage_report(
    train_dataset,
    test_datasets: dict[str, object],
    output_path: str | Path,
    train_protocol: str,
) -> list[dict[str, str | int]]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_samples = dataset_samples(train_dataset)
    train_by_name = image_index(train_samples)
    rows = []

    for test_name, test_dataset in test_datasets.items():
        test_samples = dataset_samples(test_dataset)
        test_by_name = image_index(test_samples)
        overlap = sorted(set(train_by_name) & set(test_by_name))
        if overlap:
            for name in overlap:
                rows.append({
                    "train_protocol": train_protocol,
                    "test_dataset": test_name,
                    "n_train": len(train_samples),
                    "n_test": len(test_samples),
                    "n_overlap": len(overlap),
                    "normalized_file": name,
                    "train_image_path": train_by_name[name],
                    "test_image_path": test_by_name[name],
                    "status": "leak",
                })
        else:
            rows.append({
                "train_protocol": train_protocol,
                "test_dataset": test_name,
                "n_train": len(train_samples),
                "n_test": len(test_samples),
                "n_overlap": 0,
                "normalized_file": "",
                "train_image_path": "",
                "test_image_path": "",
                "status": "ok",
            })

    fieldnames = [
        "train_protocol",
        "test_dataset",
        "n_train",
        "n_test",
        "n_overlap",
        "normalized_file",
        "train_image_path",
        "test_image_path",
        "status",
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    leaked = [row for row in rows if row["status"] == "leak"]
    if leaked:
        print(f"WARNING: leakage check found {len(leaked)} overlapping train/test files.")
    print(f"Leakage report: {output_path}")
    return rows
