from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from experiments.recall_safe_calibration.metrics_calibration import compute_metrics
from experiments.recall_safe_calibration.utils import ensure_new_file, read_json


def final_variants(temperature_path: Path, threshold_path: Path) -> list[dict[str, Any]]:
    temperature_payload = read_json(temperature_path)
    thresholds = read_json(threshold_path)
    temperature = float(temperature_payload["temperature"])
    uncalibrated_threshold = float(thresholds["uncalibrated"]["threshold"])
    calibrated_threshold = float(thresholds["temperature_scaled"]["threshold"])
    return [
        {
            "method": "ESFPNet threshold 0.5 sem calibracao",
            "variant": "baseline_threshold_0_5",
            "temperature": 1.0,
            "threshold": 0.5,
            "selection": "fixed_0.5",
        },
        {
            "method": "ESFPNet threshold recall-safe sem calibracao",
            "variant": "uncalibrated_recall_safe",
            "temperature": 1.0,
            "threshold": uncalibrated_threshold,
            "selection": thresholds["uncalibrated"]["selected_by"],
        },
        {
            "method": "ESFPNet com Temperature Scaling + threshold recall-safe",
            "variant": "temperature_scaled_recall_safe",
            "temperature": temperature,
            "threshold": calibrated_threshold,
            "selection": thresholds["temperature_scaled"]["selected_by"],
        },
    ]

def evaluate_variants(
    cache: dict[str, Any],
    variants: list[dict[str, Any]],
    *,
    ece_bins: int,
    lesion_dilation: int,
    include_boundary: bool,
) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        metrics = compute_metrics(
            cache["logits"],
            cache["targets"],
            threshold=float(variant["threshold"]),
            temperature=float(variant["temperature"]),
            ece_bins=ece_bins,
            lesion_dilation=lesion_dilation,
            include_boundary=include_boundary,
        )
        rows.append({**variant, **metrics})
    return rows

def compare_to_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in rows if row["variant"] == "baseline_threshold_0_5")
    comparisons = {}
    for row in rows:
        if row is baseline:
            continue
        recall_delta = float(row["recall"]) - float(baseline["recall"])
        dice_delta = float(row["dice"]) - float(baseline["dice"])
        precision_delta = float(row["precision"]) - float(baseline["precision"])
        comparisons[row["variant"]] = {
            "delta_recall": recall_delta,
            "delta_dice": dice_delta,
            "delta_precision": precision_delta,
            "requested_success_criterion_met": bool(
                recall_delta >= 0.02 and dice_delta >= -0.015 and precision_delta >= -0.10
            ),
        }
    return comparisons

def write_results_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    ensure_new_file(path, overwrite=overwrite)
    metric_fields = [
        "dice",
        "iou",
        "recall",
        "precision",
        "f1",
        "brier_score",
        "ece_global",
        "ece_foreground",
        "hd95",
        "assd",
        "boundary_f1",
        "n_images",
        "n_pixels",
    ]
    fieldnames = ["method", "variant", "threshold", "temperature", "selection", *metric_fields]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
