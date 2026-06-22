import hashlib
import json
import random
import re
from bisect import bisect_right
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
MASK_THRESHOLD = 127
SPLIT_NAMES = ('train', 'val', 'test')

def normalized_name(value):
    return re.sub('[^a-z0-9]+', '', Path(value).stem.lower())

def stable_int(value):
    return int(hashlib.sha1(value.encode('utf-8')).hexdigest()[:8], 16)

def resolve_path(path, base_dir=None):
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir) / resolved
    return resolved

def match_mask(masks_dir, stem):
    for extension in IMAGE_EXTENSIONS:
        candidate = masks_dir / f'{stem}{extension}'
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No mask found for image '{stem}' in {masks_dir}")

def list_image_mask_pairs(images_dir, masks_dir):
    if not images_dir.is_dir():
        raise FileNotFoundError(f'Images directory not found: {images_dir}')
    if not masks_dir.is_dir():
        raise FileNotFoundError(f'Masks directory not found: {masks_dir}')
    images = sorted((path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS))
    if not images:
        raise ValueError(f'No images found in {images_dir}')
    return [(str(image), str(match_mask(masks_dir, image.stem))) for image in images]

def source_dirs(source, base_dir=None):
    root = source.get('root')
    root_dir = resolve_path(root, base_dir) if root else None
    images_dir = source.get('images_dir')
    masks_dir = source.get('masks_dir')
    if images_dir is None and root_dir is not None:
        images_dir = 'images'
    if masks_dir is None and root_dir is not None:
        masks_dir = 'masks'
    if images_dir is None or masks_dir is None:
        raise ValueError(f"Source '{source.get('name', '<unnamed>')}' needs images_dir and masks_dir or root.")
    return (resolve_path(images_dir, root_dir or base_dir), resolve_path(masks_dir, root_dir or base_dir))

def split_key(source, images_dir, masks_dir):
    return str(source.get('split_group') or f'{images_dir.resolve()}::{masks_dir.resolve()}')

def load_split_selection(split_file, split):
    with open(split_file) as file:
        return json.load(file)[split]

def select_from_split_file(pairs, selection):
    if not selection:
        return []
    if all((isinstance(item, int) for item in selection)):
        return [pairs[item] for item in selection]
    by_key = {}
    for pair in pairs:
        image_path = Path(pair[0])
        by_key[normalized_name(image_path)] = pair
        by_key[image_path.name.lower()] = pair
    selected = []
    missing = []
    for item in selection:
        key = str(item).lower()
        pair = by_key.get(key) or by_key.get(normalized_name(key))
        if pair is None:
            missing.append(str(item))
            continue
        selected.append(pair)
    if missing:
        examples = ', '.join(missing[:5])
        raise FileNotFoundError(f'{len(missing)} split entries were not found. Examples: {examples}')
    return selected

def random_split_indices(total, split_config, key):
    seed = int(split_config.get('seed', 42)) + stable_int(key)
    train_ratio = float(split_config.get('train_ratio', 0.9))
    val_ratio = float(split_config.get('val_ratio', 0.1))
    test_ratio = float(split_config.get('test_ratio', max(0.0, 1.0 - train_ratio - val_ratio)))
    ratio_sum = train_ratio + val_ratio + test_ratio
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    return {'train': indices[:n_train], 'val': indices[n_train:n_train + n_val], 'test': indices[n_train + n_val:]}

def select_pairs_for_split(pairs, source, split_config, images_dir, masks_dir):
    split = source.get('split')
    if split in (None, 'all'):
        return (pairs, 'all files')
    if split not in SPLIT_NAMES:
        raise ValueError(f"Invalid split '{split}' for source '{source.get('name', '<unnamed>')}'.")
    split_file = source.get('split_file')
    if split_file:
        split_path = resolve_path(split_file)
        selected = select_from_split_file(pairs, load_split_selection(split_path, split))
        return (selected, f'predefined split file {split_path}:{split}')
    mode = (split_config or {}).get('mode', 'predefined')
    if mode == 'random':
        indices = random_split_indices(len(pairs), split_config or {}, split_key(source, images_dir, masks_dir))[split]
        return ([pairs[index] for index in indices], f"random split '{split}'")
    if mode == 'predefined':
        return (pairs, 'predefined directory')
    raise ValueError(f"Unsupported split mode '{mode}'.")

class SegmentationSourceDataset(Dataset):

    def __init__(self, name, samples, transform=None):
        self.dataset_name = name
        self.samples = list(samples)
        self.transform = transform
        if not self.samples:
            raise ValueError(f"Source '{name}' has no samples.")

    def __len__(self):
        return len(self.samples)

    def _read_rgb(self, path):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f'Invalid image: {path}')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f'Invalid mask: {path}')
        return (mask > MASK_THRESHOLD).astype(np.float32)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = self._read_rgb(image_path)
        mask = self._read_mask(mask_path)
        metadata = {'image_path': image_path, 'mask_path': mask_path, 'dataset_name': self.dataset_name}
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            if 'image_weak' in augmented:
                return {**augmented, **metadata}
            image = augmented['image']
            mask = augmented['mask']
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        elif isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0).float()
        return {'image': image, 'mask': mask.float(), **metadata}

class MultiSourceSegmentationDataset(Dataset):

    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]

    @property
    def source_counts(self):
        return {dataset.dataset_name: len(dataset) for dataset in self.datasets}

    @property
    def samples(self):
        return [sample for dataset in self.datasets for sample in dataset.samples]

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        previous_size = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][idx - previous_size]

def build_source_dataset(source, transform, split_config=None, base_dir=None):
    images_dir, masks_dir = source_dirs(source, base_dir)
    pairs = list_image_mask_pairs(images_dir, masks_dir)
    selected, split_description = select_pairs_for_split(pairs, source, split_config, images_dir, masks_dir)
    dataset = SegmentationSourceDataset(source['name'], selected, transform=transform)
    return (dataset, split_description)
