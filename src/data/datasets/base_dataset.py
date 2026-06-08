from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):

    def __init__(self, root: str, split: str, transform=None):
        assert split in ("train", "val", "test"), f"Invalid Splt: {split}"
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    @abstractmethod
    def _load_samples(self) -> list:
        ...

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        ...
