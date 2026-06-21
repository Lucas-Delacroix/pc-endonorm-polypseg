from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, root: str, split: str, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)
