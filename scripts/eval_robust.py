from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from scripts.train import load_config

CHECKPOINT_TYPES = ("best_raw", "best_ema", "best_swa")
IN_DOMAIN_DATASET = "kvasir-seg"
DEFAULT_CROSS_DATASETS = ("cvc-colondb", "etis-larib")
KVASIR_SPLIT_FILE = "data/splits/kvasir_split.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate raw / EMA / SWA checkpoints of a robust run on in-domain and cross datasets."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--cross-datasets", nargs="*", default=list(DEFAULT_CROSS_DATASETS))
    parser.add_argument("--checkpoint-types", nargs="*", default=list(CHECKPOINT_TYPES))
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--checkpoints-root", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, check=True)


def export(config: str, checkpoint: Path, dataset: str, output_root: Path, external: bool, device: str) -> None:
    command = [
        "uv", "run", "python", "-m", "scripts.export_predictions",
        "--config", config,
        "--checkpoint", str(checkpoint),
        "--data-root", str(Path("data/raw") / dataset),
        "--output-root", str(output_root),
        "--device", device,
    ]
    if external:
        command.append("--external")
    run(command)


def evaluate(dataset: str, predictions_root: Path, tables_dir: Path, output_name: str, external: bool) -> None:
    command = [
        "uv", "run", "python", "-m", "scripts.evaluate_predictions",
        "--predictions-root", str(predictions_root),
        "--data-root", str(Path("data/raw") / dataset),
        "--output-dir", str(tables_dir),
        "--output-name", output_name,
    ]
    if external:
        command += ["--external"]
    else:
        command += ["--split-file", KVASIR_SPLIT_FILE, "--split", "test"]
    run(command)


def read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path) as file:
        return list(csv.DictReader(file))


def write_summary(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["checkpoint_type", "dataset"] + [k for k in rows[0] if k not in ("checkpoint_type", "dataset")]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config["logging"]["run_name"]

    checkpoints_dir = Path(args.checkpoints_root) / run_name
    results_dir = Path(args.results_dir) / run_name
    predictions_dir = results_dir / "predictions"
    tables_dir = results_dir / "tables"

    datasets = [IN_DOMAIN_DATASET, *args.cross_datasets]
    all_rows: list[dict] = []

    for checkpoint_type in args.checkpoint_types:
        checkpoint = checkpoints_dir / f"{checkpoint_type}.pt"
        if not checkpoint.exists():
            print(f"Skipping {checkpoint_type}: {checkpoint} not found.")
            continue

        for dataset in datasets:
            external = dataset != IN_DOMAIN_DATASET
            output_root = predictions_dir / checkpoint_type / dataset
            output_name = f"metrics_{dataset}_{checkpoint_type}"

            export(args.config, checkpoint, dataset, output_root, external, args.device)
            evaluate(dataset, output_root, tables_dir, output_name, external)

            for row in read_rows(tables_dir / f"{output_name}.csv"):
                all_rows.append({"checkpoint_type": checkpoint_type, "dataset": dataset, **row})

    write_summary(all_rows, results_dir / "summary_all.csv")
    for checkpoint_type in args.checkpoint_types:
        type_rows = [row for row in all_rows if row["checkpoint_type"] == checkpoint_type]
        write_summary(type_rows, results_dir / f"{checkpoint_type}_summary.csv")

    print(f"\nRobust evaluation complete. Summary: {results_dir / 'summary_all.csv'}")


if __name__ == "__main__":
    main()
