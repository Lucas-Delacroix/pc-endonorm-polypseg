from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.recall_safe_calibration.metrics_calibration import compute_metrics  # noqa: E402
from experiments.recall_safe_calibration.utils import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_TEST_SOURCE,
    collect_predictions,
    ensure_new_file,
    load_config,
    read_json,
    read_prediction_cache,
    write_json,
    write_prediction_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recall-safe calibration variants on ETIS/test data.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache", default=None, help="Optional .npz cache from collect_logits.py.")
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--threshold", required=True, help="selected_threshold.json from sweep_thresholds.py")
    parser.add_argument("--test-images", default=None)
    parser.add_argument("--test-masks", default=None)
    parser.add_argument("--test-source-name", default=DEFAULT_TEST_SOURCE)
    parser.add_argument("--test-split", default="all")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--lesion-dilation", type=int, default=0)
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--cache-output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_or_collect_test(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if args.cache:
        return read_prediction_cache(Path(args.cache))

    predictions, metadata = collect_predictions(
        config,
        checkpoint=args.checkpoint,
        role="test",
        images_dir=args.test_images,
        masks_dir=args.test_masks,
        source_name=args.test_source_name,
        split=args.test_split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_choice=args.device,
    )
    metadata["config_path"] = args.config
    if args.cache_output:
        write_prediction_cache(Path(args.cache_output), predictions, metadata, overwrite=args.overwrite)
    return {**predictions, "metadata": metadata}


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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cache = load_or_collect_test(args, config)
    variants = final_variants(Path(args.temperature), Path(args.threshold))
    rows = evaluate_variants(
        cache,
        variants,
        ece_bins=args.ece_bins,
        lesion_dilation=args.lesion_dilation,
        include_boundary=not args.no_boundary,
    )
    payload = {
        "test_cache_metadata": cache.get("metadata", {}),
        "temperature_file": args.temperature,
        "threshold_file": args.threshold,
        "metrics_scope": "dataset-level pixel aggregation; boundary metrics are image-level means",
        "rows": rows,
        "comparisons_vs_baseline_threshold_0_5": compare_to_baseline(rows),
    }
    output = Path(args.output)
    write_json(output, payload, overwrite=args.overwrite)
    csv_output = Path(args.csv_output) if args.csv_output else output.with_suffix(".csv")
    write_results_csv(csv_output, rows, overwrite=args.overwrite)
    print(f"ETIS/test JSON saved to: {output}")
    print(f"ETIS/test CSV saved to: {csv_output}")


if __name__ == "__main__":
    main()
