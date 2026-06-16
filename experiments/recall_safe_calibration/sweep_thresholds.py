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
    collect_predictions,
    ensure_new_file,
    load_config,
    parse_thresholds,
    read_json,
    read_prediction_cache,
    write_json,
    write_prediction_cache,
)


DICE_TOLERANCE = 0.01
PRECISION_TOLERANCE = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep validation thresholds and select recall-safe points.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache", default=None, help="Optional .npz cache from collect_logits.py.")
    parser.add_argument("--temperature", default=None)
    parser.add_argument("--val-images", default=None)
    parser.add_argument("--val-masks", default=None)
    parser.add_argument("--val-source-name", default=None)
    parser.add_argument("--val-split", default="all")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--thresholds", nargs="*", default=None)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--lesion-dilation", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected-output", default=None)
    parser.add_argument("--cache-output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_or_collect_validation(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if args.cache:
        return read_prediction_cache(Path(args.cache))

    predictions, metadata = collect_predictions(
        config,
        checkpoint=args.checkpoint,
        role="val",
        images_dir=args.val_images,
        masks_dir=args.val_masks,
        source_name=args.val_source_name,
        split=args.val_split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_choice=args.device,
    )
    metadata["config_path"] = args.config
    if args.cache_output:
        write_prediction_cache(Path(args.cache_output), predictions, metadata, overwrite=args.overwrite)
    return {**predictions, "metadata": metadata}


def sweep_thresholds(
    logits,
    targets,
    *,
    thresholds: list[float],
    variants: dict[str, float],
    ece_bins: int,
    lesion_dilation: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, temperature in variants.items():
        for threshold in thresholds:
            metrics = compute_metrics(
                logits,
                targets,
                threshold=threshold,
                temperature=temperature,
                ece_bins=ece_bins,
                lesion_dilation=lesion_dilation,
                include_boundary=False,
            )
            rows.append({"variant": variant, **metrics})
    return rows


def select_recall_safe_threshold(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    variant_rows = [row for row in rows if row["variant"] == variant]
    baseline = min(variant_rows, key=lambda row: abs(float(row["threshold"]) - 0.5))
    dice_floor = float(baseline["dice"]) - DICE_TOLERANCE
    precision_floor = float(baseline["precision"]) - PRECISION_TOLERANCE

    def best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda row: (
                float(row["recall"]),
                float(row["dice"]),
                float(row["precision"]),
                -abs(float(row["threshold"]) - 0.5),
            ),
        )

    strict_candidates = [
        row for row in variant_rows
        if float(row["dice"]) >= dice_floor and float(row["precision"]) >= precision_floor
    ]
    selected = best(strict_candidates)
    selected_by = "dice_and_precision"
    if selected is None:
        dice_candidates = [row for row in variant_rows if float(row["dice"]) >= dice_floor]
        selected = best(dice_candidates)
        selected_by = "dice_only"
    if selected is None:
        selected = baseline
        selected_by = "fallback_0.5"

    return {
        "variant": variant,
        "threshold": float(selected["threshold"]),
        "selected_by": selected_by,
        "rule": {
            "objective": "maximize validation recall",
            "dice_floor": dice_floor,
            "precision_floor": precision_floor,
            "dice_tolerance": DICE_TOLERANCE,
            "precision_tolerance": PRECISION_TOLERANCE,
            "fallback": "threshold 0.5 if no threshold satisfies the Dice constraint",
        },
        "reference_threshold_0_5": baseline,
        "selected_metrics": selected,
    }


def write_sweep_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    ensure_new_file(path, overwrite=overwrite)
    fieldnames = [
        "variant",
        "threshold",
        "temperature",
        "dice",
        "iou",
        "recall",
        "precision",
        "f1",
        "brier_score",
        "ece_global",
        "ece_foreground",
        "specificity",
        "tp",
        "fp",
        "fn",
        "tn",
        "n_images",
        "n_pixels",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_selected_payload(
    rows: list[dict[str, Any]],
    *,
    thresholds: list[float],
    variants: dict[str, float],
    cache_metadata: dict[str, Any],
) -> dict[str, Any]:
    selections = {
        variant: select_recall_safe_threshold(rows, variant)
        for variant in variants
    }
    return {
        "threshold_grid": thresholds,
        "variants": variants,
        "validation_cache_metadata": cache_metadata,
        "selection_rule": {
            "primary": "max Recall_val",
            "constraint_1": "Dice_val >= Dice_val@0.5 - 0.01",
            "constraint_2": "Precision_val >= Precision_val@0.5 - 0.05, if possible",
            "fallback": "use only Dice constraint; otherwise keep threshold 0.5",
        },
        **selections,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cache = load_or_collect_validation(args, config)
    thresholds = parse_thresholds(args.thresholds)
    variants = {"uncalibrated": 1.0}
    if args.temperature:
        variants["temperature_scaled"] = float(read_json(Path(args.temperature))["temperature"])

    rows = sweep_thresholds(
        cache["logits"],
        cache["targets"],
        thresholds=thresholds,
        variants=variants,
        ece_bins=args.ece_bins,
        lesion_dilation=args.lesion_dilation,
    )
    output = Path(args.output)
    write_sweep_csv(output, rows, overwrite=args.overwrite)
    selected_output = Path(args.selected_output) if args.selected_output else output.with_name("selected_threshold.json")
    payload = build_selected_payload(
        rows,
        thresholds=thresholds,
        variants=variants,
        cache_metadata=cache.get("metadata", {}),
    )
    write_json(selected_output, payload, overwrite=args.overwrite)
    print(f"Threshold sweep saved to: {output}")
    print(f"Selected thresholds saved to: {selected_output}")


if __name__ == "__main__":
    main()
