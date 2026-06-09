from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_DATASETS = ("cvc-colondb", "etis-larib")
DEFAULT_EXPERIMENTS = ("A_baseline_rgb", "F_pc_endonorm")
PREPROCESS_CONFIG = "configs/preprocess_pc_endonorm.yaml"
IMAGE_SIZE = 352


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation of trained experiments on external polyp datasets."
    )
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--experiments", nargs="*", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--preprocessed-root", default="data/preprocessed")
    parser.add_argument("--predictions-root", default="outputs/predictions_cross")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--only-preprocess", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, check=True)


def preprocess(dataset: str, raw_root: str, preprocessed_root: str) -> None:
    run([
        "uv", "run", "python", "scripts/generate_pc_endonorm_dataset.py",
        "--dataset-name", dataset,
        "--images-dir", str(Path(raw_root) / dataset / "images"),
        "--output-dir", str(Path(preprocessed_root) / dataset),
        "--image-size", str(IMAGE_SIZE),
        "--config", PREPROCESS_CONFIG,
    ])


def predict(experiment: str, dataset: str, args: argparse.Namespace) -> None:
    run([
        "uv", "run", "python", "-m", "scripts.export_predictions",
        "--config", f"configs/experiments/{experiment}.yaml",
        "--data-root", str(Path(args.raw_root) / dataset),
        "--preprocessed-root", str(Path(args.preprocessed_root) / dataset),
        "--output-root", str(Path(args.predictions_root) / dataset),
        "--external",
        "--device", args.device,
    ])


def evaluate(dataset: str, args: argparse.Namespace) -> None:
    run([
        "uv", "run", "python", "-m", "scripts.evaluate_predictions",
        "--predictions-root", str(Path(args.predictions_root) / dataset),
        "--data-root", str(Path(args.raw_root) / dataset),
        "--external",
        "--output-dir", args.tables_dir,
        "--output-name", f"table2_{dataset}",
    ])


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        if not args.skip_preprocess:
            preprocess(dataset, args.raw_root, args.preprocessed_root)
        if args.only_preprocess:
            continue
        for experiment in args.experiments:
            predict(experiment, dataset, args)
        evaluate(dataset, args)
        print(f"=== {dataset}: {args.tables_dir}/table2_{dataset}.md ===")

    if args.only_preprocess:
        print("\nCross-dataset preprocessing complete.")
    else:
        print("\nCross-dataset evaluation complete.")


if __name__ == "__main__":
    main()
