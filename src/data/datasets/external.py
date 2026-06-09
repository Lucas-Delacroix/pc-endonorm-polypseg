from __future__ import annotations

from pathlib import Path

from data.datasets.kvasir import KvasirDataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class ExternalPolypDataset(KvasirDataset):
    def _load_samples(self) -> list:
        root = Path(self.root)
        images_dir = root / "images"
        masks_dir = root / "masks"

        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No images found in {images_dir}")

        return [(str(image), str(self._match_mask(masks_dir, image.stem))) for image in images]

    def _match_mask(self, masks_dir: Path, stem: str) -> Path:
        for extension in IMAGE_EXTENSIONS:
            candidate = masks_dir / f"{stem}{extension}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No mask found for image '{stem}' in {masks_dir}")
