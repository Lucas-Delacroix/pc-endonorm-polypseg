from torch.utils.data import DataLoader
from data.datasets.kvasir import KvasirDataset
from data.transforms.augmentation import get_train_transforms, get_val_transforms
from data.transforms.robust_style import build_robust_train_transforms

class PolypDataModule:

    def __init__(self, data_root, image_size=352, batch_size=8, num_workers=4, pin_memory=True, input_mode='rgb', preprocessed_root=None, augmentation=None, consistency=None):
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.input_mode = input_mode
        self.preprocessed_root = preprocessed_root
        self.augmentation = augmentation
        self.consistency = consistency
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def _build_dataset(self, split, transform):
        return KvasirDataset(root=self.data_root, split=split, transform=transform, image_size=self.image_size, input_mode=self.input_mode, preprocessed_root=self.preprocessed_root)

    def _train_transform(self):
        if self.augmentation:
            return build_robust_train_transforms(self.image_size, self.augmentation)
        return get_train_transforms(self.image_size)

    def _build_train_dataset(self):
        if self.consistency and self.consistency.get('enabled'):
            from data.datasets.consistency import ConsistencyKvasirDataset, ConsistencyViews
            return ConsistencyKvasirDataset(root=self.data_root, split='train', transform=ConsistencyViews(self.image_size, self.augmentation), image_size=self.image_size, input_mode='rgb', preprocessed_root=self.preprocessed_root)
        return self._build_dataset('train', self._train_transform())

    def setup(self):
        self._train_dataset = self._build_train_dataset()
        self._val_dataset = self._build_dataset('val', get_val_transforms(self.image_size))
        self._test_dataset = self._build_dataset('test', get_val_transforms(self.image_size))
        self._log_split_info()

    def _log_split_info(self):
        print('Dataset loaded:')
        print(f'  Train:    {len(self._train_dataset)} images')
        print(f'  Validation: {len(self._val_dataset)} images')
        print(f'  Test:     {len(self._test_dataset)} images')

    def train_loader(self):
        return DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True)

    def val_loader(self):
        return DataLoader(self._val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_loader(self):
        return DataLoader(self._test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
