from torch.utils.data import DataLoader

from data.datasets.kvasir import KvasirDataset
from data.transforms.augmentation import get_train_transforms, get_val_transforms


class PolypDataModule:
    DATASETS = {
        "kvasir": KvasirDataset,
    }

    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        image_size: int = 352,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        input_mode: str = "rgb",
        preprocessed_root: str | None = None,
    ):
        assert dataset_name in self.DATASETS, (
            f"Dataset '{dataset_name}' not recognized. "
            f"Options: {list(self.DATASETS.keys())}"
        )

        self.dataset_cls = self.DATASETS[dataset_name]
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.input_mode = input_mode
        self.preprocessed_root = preprocessed_root

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def _build_dataset(self, split, transform):
        return self.dataset_cls(
            root=self.data_root,
            split=split,
            transform=transform,
            image_size=self.image_size,
            input_mode=self.input_mode,
            preprocessed_root=self.preprocessed_root,
        )

    def setup(self):
        self._train_dataset = self._build_dataset("train", get_train_transforms(self.image_size))
        self._val_dataset = self._build_dataset("val", get_val_transforms(self.image_size))
        self._test_dataset = self._build_dataset("test", get_val_transforms(self.image_size))
        self._log_split_info()

    def _log_split_info(self):
        print("Dataset loaded:")
        print(f"  Train:    {len(self._train_dataset)} images")
        print(f"  Validation: {len(self._val_dataset)} images")
        print(f"  Test:     {len(self._test_dataset)} images")

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_loader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
