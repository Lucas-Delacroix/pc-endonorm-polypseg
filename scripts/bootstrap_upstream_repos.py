from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "upstream" / "repos.yaml"
VENDOR_DIR = ROOT / "upstream" / "vendor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Repo keys to download.")
    parser.add_argument("--force", action="store_true", help="Replace existing vendor dirs.")
    return parser.parse_args()


def load_repos() -> dict:
    with open(MANIFEST) as file:
        return yaml.safe_load(file)["repos"]


def download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "colonoscopy-reproduction"})
    with urlopen(request) as response, open(destination, "wb") as file:
        shutil.copyfileobj(response, file)


def extract_tarball(archive_path: Path, destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="upstream-repo-") as tmp:
        tmp_dir = Path(tmp)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(tmp_dir, filter="data")
        roots = [path for path in tmp_dir.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"Expected one root directory in {archive_path}, found {len(roots)}.")
        shutil.move(str(roots[0]), destination)


def main() -> None:
    args = parse_args()
    repos = load_repos()
    selected = set(args.only or repos.keys())
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    for name, meta in repos.items():
        if name not in selected:
            continue
        destination = VENDOR_DIR / name
        if destination.exists():
            if not args.force:
                print(f"Already exists: {destination}")
                continue
            shutil.rmtree(destination)

        archive_path = VENDOR_DIR / f"{name}.tar.gz"
        url = f"{meta['url']}/archive/{meta['ref']}.tar.gz"
        print(f"Downloading {name}: {url}")
        download(url, archive_path)
        extract_tarball(archive_path, destination)
        archive_path.unlink()
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
