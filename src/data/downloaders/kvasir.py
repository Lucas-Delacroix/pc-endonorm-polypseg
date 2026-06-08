from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

from data.downloaders.base import DatasetDownloader


class KvasirDownloader(DatasetDownloader):
    name = "kvasir"
    dataset_dir_name = "kvasir-seg"
    url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
    archive_name = "kvasir-seg.zip"
    sha256 = "03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7"
    expected_samples = 1000

    def prepare(self) -> None:
        if self.dataset_dir.exists():
            if not self.force:
                raise FileExistsError(
                    f"Dataset directory already exists but is incomplete: {self.dataset_dir}\n"
                    "Remove it manually or run the downloader with --force."
                )
            shutil.rmtree(self.dataset_dir)

        self.raw_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="kvasir-seg-", dir=self.raw_root) as tmp:
            tmp_dir = Path(tmp)
            self._extract_zip(self.archive_path, tmp_dir)
            source_dir = self._find_extracted_dataset_dir(tmp_dir)

            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir / "images", self.dataset_dir / "images")
            shutil.copytree(source_dir / "masks", self.dataset_dir / "masks")

            for json_path in source_dir.glob("*.json"):
                shutil.copy2(json_path, self.dataset_dir / json_path.name)

    def validate(self) -> None:
        images_dir = self.dataset_dir / "images"
        masks_dir = self.dataset_dir / "masks"

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        images = sorted(images_dir.glob("*.jpg"))
        masks = sorted(masks_dir.glob("*.jpg"))

        if not images:
            raise ValueError(f"No .jpg images found in {images_dir}")
        if len(images) != len(masks):
            raise ValueError(
                f"Image/mask count mismatch: {len(images)} images and {len(masks)} masks."
            )
        if len(images) != self.expected_samples:
            raise ValueError(
                f"Expected {self.expected_samples} Kvasir-SEG samples, found {len(images)}."
            )

        missing_masks = [image.name for image in images if not (masks_dir / image.name).exists()]
        if missing_masks:
            examples = ", ".join(missing_masks[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_masks)} masks with matching image filenames. "
                f"Examples: {examples}"
            )

        print(f"Validated {len(images)} Kvasir-SEG samples at {self.dataset_dir}")

    def is_available(self) -> bool:
        images_dir = self.dataset_dir / "images"
        masks_dir = self.dataset_dir / "masks"
        return (
            images_dir.is_dir()
            and masks_dir.is_dir()
            and any(images_dir.glob("*.jpg"))
            and any(masks_dir.glob("*.jpg"))
        )

    def _extract_zip(self, archive_path: Path, output_dir: Path) -> None:
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target_path = output_dir / member.filename
                target_path.resolve().relative_to(output_dir.resolve())
            archive.extractall(output_dir)

    def _find_extracted_dataset_dir(self, root: Path) -> Path:
        for candidate in [root, *root.rglob("*")]:
            if (candidate / "images").is_dir() and (candidate / "masks").is_dir():
                return candidate

        raise FileNotFoundError(
            "Could not find extracted Kvasir-SEG directories. "
            "Expected a folder containing images/ and masks/."
        )
