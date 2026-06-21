from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.recall_safe_calibration.metrics_calibration import reliability_rows
from experiments.recall_safe_calibration.utils import read_json, read_prediction_cache


def read_sweep(path: Path) -> list[dict[str, str]]:
    with open(path) as file:
        return list(csv.DictReader(file))

def plot_threshold_metric(rows: list[dict[str, str]], metric: str, output_dir: Path) -> Path:
    variants = sorted({row["variant"] for row in rows})
    plt.figure(figsize=(6, 4))
    for variant in variants:
        variant_rows = sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: float(row["threshold"]),
        )
        thresholds = [float(row["threshold"]) for row in variant_rows]
        values = [float(row[metric]) for row in variant_rows]
        plt.plot(thresholds, values, marker="o", label=variant)
    plt.xlabel("Threshold")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = output_dir / f"threshold_vs_{metric}.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return path

def plot_reliability(cache: dict, temperature: float, ece_bins: int, output_dir: Path) -> Path:
    variants = {
        "uncalibrated": 1.0,
        "temperature_scaled": temperature,
    }
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="perfect")
    for variant, temp in variants.items():
        rows = reliability_rows(
            cache["logits"],
            cache["targets"],
            temperature=temp,
            ece_bins=ece_bins,
        )
        xs = []
        ys = []
        for row in rows:
            if row["count"] <= 0 or np.isnan(row["confidence"]):
                continue
            xs.append(row["confidence"])
            ys.append(row["empirical_positive_rate"])
        plt.plot(xs, ys, marker="o", label=variant)
    plt.xlabel("Mean predicted foreground probability")
    plt.ylabel("Empirical foreground frequency")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = output_dir / "reliability_diagram_val.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return path

def plot_all(
    *,
    sweep_path: Path,
    val_cache_path: Path,
    temperature_path: Path,
    output_dir: Path,
    ece_bins: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_sweep(sweep_path)
    paths = [
        plot_threshold_metric(rows, "dice", output_dir),
        plot_threshold_metric(rows, "recall", output_dir),
        plot_threshold_metric(rows, "precision", output_dir),
    ]
    cache = read_prediction_cache(val_cache_path)
    temperature = float(read_json(temperature_path)["temperature"])
    paths.append(plot_reliability(cache, temperature, ece_bins, output_dir))
    return paths
