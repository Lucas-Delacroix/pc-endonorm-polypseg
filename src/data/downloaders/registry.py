from __future__ import annotations

from data.downloaders.base import DatasetDownloader
from data.downloaders.kvasir import KvasirDownloader


DOWNLOADERS: dict[str, type[DatasetDownloader]] = {
    "kvasir": KvasirDownloader,
}


def available_datasets() -> tuple[str, ...]:
    return tuple(sorted(DOWNLOADERS))


def get_downloader(dataset_name: str) -> type[DatasetDownloader]:
    try:
        return DOWNLOADERS[dataset_name]
    except KeyError as exc:
        available = ", ".join(available_datasets())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
        ) from exc
