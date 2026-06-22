from torch.utils.data import Dataset

class BaseDataset(Dataset):

    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)
