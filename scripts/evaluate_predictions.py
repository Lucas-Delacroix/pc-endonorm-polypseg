from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


MODEL_NAMES = {"esfpnet": "ESFPNet"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", default="outputs/predictions")
    parser.add_argument("--data-root", default="data/raw/kvasir-seg")
    parser.add_argument("--split-file", default="data/splits/kvasir_split.json")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def load_split_indices(split_file: Path, split: str) -> list[int]:
    with open(split_file) as file:
        return json.load(file)[split]


def load_samples(data_root: Path, split_file: Path, split: str) -> list[tuple[Path, Path]]:
    images = sorted((data_root / "images").glob("*.jpg"))
    masks_dir = data_root / "masks"
    return [(images[idx], masks_dir / images[idx].name) for idx in load_split_indices(split_file, split)]


def model_dirs(predictions_root: Path, selected: list[str] | None) -> list[Path]:
    if selected:
        return [predictions_root / name for name in selected]
    return sorted(path for path in predictions_root.iterdir() if path.is_dir())


def build_prediction_index(prediction_dir: Path) -> dict[str, Path]:
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return {
        path.stem: path
        for path in prediction_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    }


def read_binary_mask(path: Path, threshold: float, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Invalid mask: {path}")

    if target_shape is not None and mask.shape != target_shape:
        width = target_shape[1]
        height = target_shape[0]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    mask = mask.astype(np.float32)
    cutoff = threshold if mask.max(initial=0.0) <= 1.0 else threshold * 255.0
    return mask > cutoff


def metric_values(prediction: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> dict[str, float]:
    pred = prediction.astype(bool).ravel()
    gt = target.astype(bool).ravel()

    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, np.logical_not(gt)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(pred), gt).sum(dtype=np.float64)

    return {
        "dice": float((2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)),
        "iou": float((tp + smooth) / (tp + fp + fn + smooth)),
        "precision": float((tp + smooth) / (tp + fp + smooth)),
        "recall": float((tp + smooth) / (tp + fn + smooth)),
    }


def evaluate_model(
    prediction_dir: Path,
    samples: list[tuple[Path, Path]],
    threshold: float,
    allow_missing: bool,
) -> dict[str, float | int | str]:
    predictions = build_prediction_index(prediction_dir)
    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    missing: list[str] = []
    evaluated = 0

    for image_path, mask_path in samples:
        prediction_path = predictions.get(image_path.stem)
        if prediction_path is None:
            missing.append(image_path.name)
            continue

        target = read_binary_mask(mask_path, threshold=0.5)
        prediction = read_binary_mask(prediction_path, threshold=threshold, target_shape=target.shape)
        metrics = metric_values(prediction, target)

        for key in totals:
            totals[key] += metrics[key]
        evaluated += 1

    if missing and not allow_missing:
        raise FileNotFoundError(f"{prediction_dir.name}: {len(missing)} missing predictions")
    if evaluated == 0:
        raise ValueError(f"{prediction_dir.name}: no predictions evaluated")

    return {
        "model": MODEL_NAMES.get(prediction_dir.name, prediction_dir.name),
        "model_key": prediction_dir.name,
        "dice": totals["dice"] / evaluated,
        "iou": totals["iou"] / evaluated,
        "precision": totals["precision"] / evaluated,
        "recall": totals["recall"] / evaluated,
        "n_evaluated": evaluated,
        "n_missing": len(missing),
    }


def write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "model_key",
        "dice",
        "iou",
        "precision",
        "recall",
        "n_evaluated",
        "n_missing",
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "dice": f"{row['dice']:.6f}",
                "iou": f"{row['iou']:.6f}",
                "precision": f"{row['precision']:.6f}",
                "recall": f"{row['recall']:.6f}",
            })


def write_markdown(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Modelo | DICE | IoU | Precisao | Cobertura |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | "
            f"{row['dice']:.4f} | "
            f"{row['iou']:.4f} | "
            f"{row['precision']:.4f} | "
            f"{row['recall']:.4f} |"
        )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    predictions_root = Path(args.predictions_root)
    data_root = Path(args.data_root)
    split_file = Path(args.split_file)
    output_dir = Path(args.output_dir)

    samples = load_samples(data_root, split_file, args.split)
    rows = [
        evaluate_model(path, samples, args.threshold, args.allow_missing)
        for path in model_dirs(predictions_root, args.models)
    ]
    rows.sort(key=lambda row: row["dice"], reverse=True)

    csv_path = output_dir / "table2.csv"
    markdown_path = output_dir / "table2.md"
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path)

    print(f"Evaluated {len(rows)} model(s) on {args.split} split ({len(samples)} images).")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
