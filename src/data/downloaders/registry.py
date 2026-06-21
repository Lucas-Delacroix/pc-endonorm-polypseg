from data.downloaders.base import DatasetDownloader
from data.downloaders.kvasir import KvasirDownloader
DOWNLOADERS = {'kvasir': KvasirDownloader}

def available_datasets():
    return tuple(sorted(DOWNLOADERS))

def get_downloader(dataset_name):
    try:
        return DOWNLOADERS[dataset_name]
    except KeyError as exc:
        available = ', '.join(available_datasets())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}") from exc
