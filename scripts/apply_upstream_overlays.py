from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ROOT / "upstream" / "commands.yaml"
VENDOR = ROOT / "upstream" / "vendor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Model keys to apply overlays for.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.only) if args.only else None
    with open(COMMANDS) as file:
        commands = yaml.safe_load(file)["commands"]

    for name, entry in commands.items():
        if selected is not None and name not in selected:
            continue
        overlays = entry.get("overlay", [])
        if not overlays:
            continue
        repo_dir = ROOT / entry["cwd"]
        if not repo_dir.exists():
            raise FileNotFoundError(f"Vendor repo not found: {repo_dir}")
        for overlay in overlays:
            source = (repo_dir / overlay["source"]).resolve()
            target = repo_dir / overlay["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"{name}: {source} -> {target}")


if __name__ == "__main__":
    main()
