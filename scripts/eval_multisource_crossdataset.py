from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch

from data.datasets.multisource import (
    MultiSourceSegmentationDataset,
    build_source_dataset,
    normalized_name,
)
from data.leakage import write_leakage_report
from data.transforms.augmentation import get_val_transforms
from models import get_model
try:
    from scripts.evaluate_predictions import boundary_metrics, metric_values, read_binary_mask
    from scripts.train import load_config
except ModuleNotFoundError:
    from evaluate_predictions import boundary_metrics, metric_values, read_binary_mask
    from train import load_config


REQUIRED_EXPERIMENTS = (
    "A_baseline_rgb",
    "F_pc_endonorm",
    "I_aug_combo",
    "K_robust_full",
    "K3_robust_balanced_tversky",
    "C1_consistency",
    "FD1_dino_distill",
    "MS_baseline_rgb",
    "C2_consistency_multisource",
    "FD2_dino_distill_multisource",
)
REQUIRED_DATASETS = ("Kvasir", "CVC-ColonDB", "ETIS-Larib")
OUTPUT_DATASET_KEYS = {
    "Kvasir": "kvasir",
    "CVC-ColonDB": "cvc_colondb",
    "ETIS-Larib": "etis_larib",
    "CVC-ClinicDB": "cvc_clinicdb",
}
DATASET_LABELS = {
    "kvasir": "Kvasir",
    "kvasir_val": "Kvasir",
    "kvasir_test": "Kvasir",
    "cvc_colondb": "CVC-ColonDB",
    "cvc-colondb": "CVC-ColonDB",
    "etis_larib": "ETIS-Larib",
    "etis-larib": "ETIS-Larib",
    "cvc_clinicdb": "CVC-ClinicDB",
    "cvc_clinicdb_val": "CVC-ClinicDB",
}
EXPERIMENT_ALIASES = {
    "esfpnet": "A_baseline_rgb",
    "ESFPNet": "A_baseline_rgb",
}
METRIC_FIELDS = ("dice", "iou", "precision", "recall", "hd95", "assd", "boundary_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MS_baseline_rgb on each configured cross-dataset test source."
    )
    parser.add_argument("--config", default="configs/experiments/MS_baseline_rgb.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tables-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--no-optional", action="store_true")
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def source_list(section: dict) -> list[dict]:
    if "sources" in section:
        return list(section["sources"])
    if "source" in section:
        return [section["source"]]
    return [section]


def output_dataset_key(name: str) -> str:
    label = dataset_label(name)
    return OUTPUT_DATASET_KEYS.get(label, normalized_name(name))


def dataset_label(name: str) -> str:
    key = name.lower().replace("-", "_")
    return DATASET_LABELS.get(key, name)


def checkpoint_path(config: dict, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    run_name = config["logging"]["run_name"]
    return Path(config.get("paths", {}).get("results", "results")) / run_name / "best.pth"


def load_model(config: dict, checkpoint: Path, device: str) -> torch.nn.Module:
    model_config = config["model"]
    model = get_model(
        model_config["name"],
        num_classes=model_config["num_classes"],
        model_type=model_config.get("model_type", "b2"),
        in_channels=model_config.get("in_channels", 3),
        pretrained_path=None,
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def optional_sources(config: dict) -> list[dict]:
    sources = []
    for source in source_list(config["dataset"]["train"]):
        name = normalized_name(source["name"])
        if "kvasir" in name:
            optional = dict(source)
            optional["name"] = "kvasir_test"
            optional["split"] = "test"
            sources.append(optional)
        if "cvcclinicdb" in name:
            optional = dict(source)
            optional["name"] = "cvc_clinicdb_val"
            optional["split"] = "val"
            sources.append(optional)
    return sources


def build_eval_datasets(config: dict, include_optional: bool) -> dict[str, object]:
    split_config = config["dataset"].get("split", {})
    transform = get_val_transforms(config["data"]["image_size"])
    sources = source_list(config["dataset"]["test"])
    if include_optional:
        existing = {source["name"] for source in sources}
        sources.extend(source for source in optional_sources(config) if source["name"] not in existing)

    datasets = {}
    for source in sources:
        try:
            dataset, split_description = build_source_dataset(
                source,
                transform=transform,
                split_config=split_config,
                base_dir=config["dataset"].get("base_dir"),
            )
        except (FileNotFoundError, ValueError) as exc:
            if source.get("name") in {item["name"] for item in source_list(config["dataset"]["test"])}:
                raise
            print(f"Skipping optional dataset {source.get('name')}: {exc}")
            continue
        datasets[source["name"]] = dataset
        print(f"Eval dataset {source['name']}: {len(dataset)} images ({split_description})")
    return datasets


def build_train_dataset_for_leakage(config: dict) -> MultiSourceSegmentationDataset:
    split_config = config["dataset"].get("split", {})
    datasets = []
    for source in source_list(config["dataset"]["train"]):
        dataset, _ = build_source_dataset(
            source,
            transform=None,
            split_config=split_config,
            base_dir=config["dataset"].get("base_dir"),
        )
        datasets.append(dataset)
    return MultiSourceSegmentationDataset(datasets)


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataset,
    device: str,
    threshold: float,
    max_samples: int | None,
    run_name: str,
    dataset_name: str,
) -> dict[str, float | int | str]:
    totals = {metric: 0.0 for metric in METRIC_FIELDS}
    n_evaluated = 0
    n_missing = 0
    limit = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for index in range(limit):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        logits = model(image)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

        target = read_binary_mask(Path(sample["mask_path"]), threshold=0.5)
        prediction = cv2.resize(
            probability,
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ) > threshold
        metrics = metric_values(prediction, target)
        metrics.update(boundary_metrics(prediction, target))
        for key in totals:
            totals[key] += metrics[key]
        n_evaluated += 1

    if n_evaluated == 0:
        raise ValueError(f"{dataset_name}: no samples evaluated.")

    return {
        "model": "ESFPNet",
        "train_protocol": run_name,
        "test_dataset": dataset_label(dataset_name),
        **{key: totals[key] / n_evaluated for key in METRIC_FIELDS},
        "n_evaluated": n_evaluated,
        "n_missing": n_missing,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "train_protocol",
        "test_dataset",
        "dice",
        "iou",
        "precision",
        "recall",
        "hd95",
        "assd",
        "boundary_f1",
        "n_evaluated",
        "n_missing",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                **{key: f"{float(row[key]):.6f}" for key in METRIC_FIELDS},
            })


def infer_dataset_from_path(path: Path) -> str | None:
    name = path.stem.lower()
    if "colondb" in name:
        return "CVC-ColonDB"
    if "etis" in name:
        return "ETIS-Larib"
    if "kvasir" in name or name == "table2":
        return "Kvasir"
    return None


def read_metric_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    dataset = infer_dataset_from_path(csv_path)
    rows = []
    with open(csv_path) as file:
        for row in csv.DictReader(file):
            protocol = row.get("train_protocol") or row.get("model_key") or row.get("model")
            protocol = EXPERIMENT_ALIASES.get(protocol, protocol)
            rows.append({
                "train_protocol": protocol,
                "test_dataset": row.get("test_dataset") or dataset or "",
                **{metric: row.get(metric, "") for metric in METRIC_FIELDS},
                "n_evaluated": row.get("n_evaluated", ""),
                "n_missing": row.get("n_missing", ""),
            })
    return rows


def write_master_comparison(tables_dir: Path) -> Path:
    candidates = [
        *tables_dir.glob("*crossdataset_summary.csv"),
        *tables_dir.glob("table2*.csv"),
        *Path("resultados").glob("table2*.csv"),
    ]
    by_key = {}
    for path in candidates:
        for row in read_metric_rows(path):
            protocol = row["train_protocol"]
            dataset = row["test_dataset"]
            if protocol in REQUIRED_EXPERIMENTS and dataset:
                by_key[(protocol, dataset)] = row

    rows = []
    for protocol in REQUIRED_EXPERIMENTS:
        for dataset in REQUIRED_DATASETS:
            row = by_key.get((protocol, dataset))
            if row is None:
                row = {
                    "train_protocol": protocol,
                    "test_dataset": dataset,
                    **{metric: "" for metric in METRIC_FIELDS},
                    "n_evaluated": "",
                    "n_missing": "",
                }
            rows.append(row)

    output_path = tables_dir / "master_crossdataset_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["train_protocol", "test_dataset", *METRIC_FIELDS, "n_evaluated", "n_missing"]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config["logging"]["run_name"]
    if int(config["model"].get("in_channels", 3)) != 3:
        raise ValueError("MS_baseline_rgb evaluation expects model.in_channels=3.")

    tables_dir = Path(args.tables_dir or Path(config.get("paths", {}).get("outputs", "outputs")) / "tables")
    checkpoint = checkpoint_path(config, args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")

    datasets = build_eval_datasets(config, include_optional=not args.no_optional)
    train_dataset = build_train_dataset_for_leakage(config)
    leakage_path = tables_dir / f"{run_name}_leakage_check.csv"
    write_leakage_report(train_dataset, datasets, leakage_path, train_protocol=run_name)

    model = load_model(config, checkpoint, device)
    max_samples = 2 if args.smoke_test else None
    rows = []
    for dataset_name, dataset in datasets.items():
        row = evaluate_dataset(
            model,
            dataset,
            device,
            args.threshold,
            max_samples,
            run_name,
            dataset_name,
        )
        rows.append(row)
        per_dataset_path = tables_dir / f"{run_name}_{output_dataset_key(dataset_name)}.csv"
        write_csv([row], per_dataset_path)
        print(f"{dataset_name}: dice={row['dice']:.4f} recall={row['recall']:.4f} -> {per_dataset_path}")

    summary_path = tables_dir / f"{run_name}_crossdataset_summary.csv"
    write_csv(rows, summary_path)
    master_path = write_master_comparison(tables_dir)

    print(f"Summary: {summary_path}")
    print(f"Master comparison: {master_path}")


if __name__ == "__main__":
    main()
