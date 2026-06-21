from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DATASETS = ("cvc-colondb", "etis-larib")
EXPERIMENTS = (
    "A_baseline_rgb",
    "B2_clahe_replace",
    "C1_consistency",
    "FD1_dino_distill",
    "F_pc_endonorm",
    "G_aug_jitter",
    "I_aug_combo",
    "K3_robust_balanced_tversky",
    "K_robust_full",
)
RAW_ROOT = Path("data/raw")
PREPROCESSED_ROOT = Path("data/preprocessed")
PREDICTIONS_ROOT = Path("outputs/predictions_cross")
TABLES_DIR = Path("outputs/tables")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)


def predict(experiment: str, dataset: str, device: str) -> None:
    run([
        "uv", "run", "python", "-m", "scripts.export_predictions",
        "--config", f"configs/experiments/{experiment}.yaml",
        "--data-root", str(RAW_ROOT / dataset),
        "--preprocessed-root", str(PREPROCESSED_ROOT / dataset),
        "--output-root", str(PREDICTIONS_ROOT / dataset),
        "--external",
        "--device", device,
    ])


def evaluate(dataset: str) -> None:
    run([
        "uv", "run", "python", "-m", "scripts.evaluate_predictions",
        "--predictions-root", str(PREDICTIONS_ROOT / dataset),
        "--data-root", str(RAW_ROOT / dataset),
        "--external",
        "--output-dir", str(TABLES_DIR),
        "--output-name", f"table2_{dataset}",
    ])


def main() -> None:
    args = parse_args()
    for dataset in DATASETS:
        for experiment in args.experiments:
            predict(experiment, dataset, args.device)
        evaluate(dataset)
        table = "table2_" + dataset + ".md"
        print(f"=== {dataset}: {TABLES_DIR / table} ===")
    print("\nCross-dataset evaluation complete.")


if __name__ == "__main__":
    main()
