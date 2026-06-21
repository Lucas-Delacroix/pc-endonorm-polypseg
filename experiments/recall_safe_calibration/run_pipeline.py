from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.recall_safe_calibration.calibrate_temperature import (  # noqa: E402
    fit_temperature,
    temperature_payload,
)
from experiments.recall_safe_calibration.evaluate_calibrated import (  # noqa: E402
    compare_to_baseline,
    evaluate_variants,
    final_variants,
    write_results_csv,
)
from experiments.recall_safe_calibration.plot_results import plot_all  # noqa: E402
from experiments.recall_safe_calibration.sweep_thresholds import (  # noqa: E402
    build_selected_payload,
    sweep_thresholds,
    write_sweep_csv,
)
from experiments.recall_safe_calibration.utils import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_TEST_SOURCE,
    collect_predictions,
    load_config,
    parse_thresholds,
    read_json,
    unique_run_dir,
    write_json,
    write_prediction_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full recall-safe post-training calibration experiment."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--val-images", default=None)
    parser.add_argument("--val-masks", default=None)
    parser.add_argument("--val-source-name", default=None)
    parser.add_argument("--val-split", default="all")
    parser.add_argument("--test-images", default=None)
    parser.add_argument("--test-masks", default=None)
    parser.add_argument("--test-source-name", default=DEFAULT_TEST_SOURCE)
    parser.add_argument("--test-split", default="all")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-device", default="cpu")
    parser.add_argument("--max-temperature-iter", type=int, default=100)
    parser.add_argument("--temperature-lr", type=float, default=0.05)
    parser.add_argument("--thresholds", nargs="*", default=None)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--lesion-dilation", type=int, default=0)
    parser.add_argument("--output-root", default="outputs/recall_safe_calibration")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = args.run_name or config.get("logging", {}).get("run_name", "esfpnet")
    run_dir = unique_run_dir(Path(args.output_root), run_name)
    thresholds = parse_thresholds(args.thresholds)

    write_json(run_dir / "pipeline_config.json", {
        "config": vars(args),
        "thresholds": thresholds,
        "method_variants": [
            "baseline threshold 0.5 without calibration",
            "uncalibrated recall-safe threshold selected on validation",
            "temperature-scaled recall-safe threshold selected on validation",
        ],
        "methodological_guardrails": [
            "ETIS/test cache is not used for temperature fitting",
            "ETIS/test cache is not used for threshold selection",
            "ESFPNet weights are loaded for inference only",
        ],
    })

    print(f"Run directory: {run_dir}")
    print("Collecting validation logits...")
    val_predictions, val_metadata = collect_predictions(
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
    val_metadata["config_path"] = args.config
    val_cache = {**val_predictions, "metadata": val_metadata}
    val_cache_path = write_prediction_cache(run_dir / "val_predictions.npz", val_predictions, val_metadata)

    print("Collecting ETIS/test logits...")
    test_predictions, test_metadata = collect_predictions(
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
    test_metadata["config_path"] = args.config
    test_cache = {**test_predictions, "metadata": test_metadata}
    write_prediction_cache(run_dir / "test_predictions.npz", test_predictions, test_metadata)

    print("Fitting temperature on validation only...")
    fit = fit_temperature(
        val_cache["logits"],
        val_cache["targets"],
        device=args.calibration_device,
        max_iter=args.max_temperature_iter,
        lr=args.temperature_lr,
    )
    temperature_path = write_json(
        run_dir / "temperature.json",
        temperature_payload(
            val_cache,
            fit,
            config_path=args.config,
            checkpoint=args.checkpoint,
            ece_bins=args.ece_bins,
            lesion_dilation=args.lesion_dilation,
            lr=args.temperature_lr,
            max_iter=args.max_temperature_iter,
        ),
    )

    print("Sweeping validation thresholds...")
    variants = {"uncalibrated": 1.0, "temperature_scaled": fit["temperature"]}
    sweep_rows = sweep_thresholds(
        val_cache["logits"],
        val_cache["targets"],
        thresholds=thresholds,
        variants=variants,
        ece_bins=args.ece_bins,
        lesion_dilation=args.lesion_dilation,
    )
    sweep_path = run_dir / "threshold_sweep.csv"
    write_sweep_csv(sweep_path, sweep_rows)
    selected_path = write_json(
        run_dir / "selected_threshold.json",
        build_selected_payload(
            sweep_rows,
            thresholds=thresholds,
            variants=variants,
            cache_metadata=val_cache.get("metadata", {}),
        ),
    )

    print("Evaluating final variants on ETIS/test...")
    rows = evaluate_variants(
        test_cache,
        final_variants(temperature_path, selected_path),
        ece_bins=args.ece_bins,
        lesion_dilation=args.lesion_dilation,
        include_boundary=True,
    )
    results_path = write_json(run_dir / "etis_results.json", {
        "test_cache_metadata": test_cache.get("metadata", {}),
        "temperature_file": str(temperature_path),
        "threshold_file": str(selected_path),
        "metrics_scope": "dataset-level pixel aggregation; boundary metrics are image-level means",
        "rows": rows,
        "comparisons_vs_baseline_threshold_0_5": compare_to_baseline(rows),
    })
    write_results_csv(run_dir / "etis_results.csv", rows)

    print("Plotting validation curves and reliability diagram...")
    plot_all(
        sweep_path=sweep_path,
        val_cache_path=val_cache_path,
        temperature_path=temperature_path,
        output_dir=run_dir / "plots",
        ece_bins=args.ece_bins,
    )

    selected = read_json(selected_path)
    print("Done.")
    print(f"Temperature: {fit['temperature']:.6f}")
    print(f"Uncalibrated recall-safe threshold: {selected['uncalibrated']['threshold']}")
    print(f"Temperature-scaled recall-safe threshold: {selected['temperature_scaled']['threshold']}")
    print(f"Final results: {results_path}")


if __name__ == "__main__":
    main()
