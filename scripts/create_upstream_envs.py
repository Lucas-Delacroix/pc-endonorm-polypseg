from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "upstream" / "envs"


def env_name(env_file: Path) -> str:
    with open(env_file) as file:
        data = yaml.safe_load(file)
    return data["name"]


def existing_env_names() -> set[str]:
    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    env_paths = json.loads(result.stdout).get("envs", [])
    return {Path(env_path).name for env_path in env_paths}


def run(command: list[str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    existing = existing_env_names()
    for env_file in sorted(ENV_DIR.glob("*.yml")):
        name = env_name(env_file)
        if name in existing:
            run(["conda", "env", "update", "-f", str(env_file), "--prune"])
        else:
            run(["conda", "env", "create", "-f", str(env_file)])


if __name__ == "__main__":
    main()
