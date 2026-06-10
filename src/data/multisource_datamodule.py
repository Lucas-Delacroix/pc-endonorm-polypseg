from __future__ import annotations

from torch.utils.data import DataLoader

from data.datasets.multisource import (
    MultiSourceSegmentationDataset,
    build_source_dataset,
    normalized_name,
)
from data.transforms.augmentation import get_train_transforms, get_val_transforms
from data.transforms.robust_style import build_robust_train_transforms


EXTERNAL_TEST_ONLY = ("etis", "larib", "colondb")


class MultiSourcePolypDataModule:
    def __init__(
        self,
        dataset_config: dict,
        image_size: int = 352,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        augmentation: dict | None = None,
        base_dir: str | None = None,
        consistency: dict | None = None,
    ):
        self.dataset_config = dataset_config
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augmentation = augmentation
        self.base_dir = base_dir
        self.consistency = consistency

        self._train_dataset = None
        self._val_dataset = None
        self._test_datasets = {}
        self._split_descriptions = {}

    def _train_transform(self):
        if self.consistency and self.consistency.get("enabled"):
            from data.datasets.consistency import ConsistencyViews
            return ConsistencyViews(self.image_size, self.augmentation)
        if self.augmentation:
            return build_robust_train_transforms(self.image_size, self.augmentation)
        return get_train_transforms(self.image_size)

    def _sources_from_section(self, section: dict) -> list[dict]:
        if "sources" in section:
            return list(section["sources"])
        if "source" in section:
            return [section["source"]]
        return [section]

    def _validate_sources(self, train_sources: list[dict], test_sources: list[dict]) -> None:
        for source in train_sources:
            name = normalized_name(source.get("name", ""))
            if any(forbidden in name for forbidden in EXTERNAL_TEST_ONLY):
                raise ValueError(
                    f"Source '{source.get('name')}' is test-only and cannot be used for training."
                )

        train_names = {normalized_name(source.get("name", "")) for source in train_sources}
        test_names = {normalized_name(source.get("name", "")) for source in test_sources}
        overlap = train_names & test_names
        if overlap:
            raise ValueError(f"Train/test source names overlap: {sorted(overlap)}")

    def _build_multi_dataset(self, sources: list[dict], transform) -> MultiSourceSegmentationDataset:
        split_config = self.dataset_config.get("split", {})
        datasets = []
        for source in sources:
            dataset, split_description = build_source_dataset(
                source,
                transform=transform,
                split_config=split_config,
                base_dir=self.base_dir,
            )
            datasets.append(dataset)
            self._split_descriptions[source["name"]] = split_description
        return MultiSourceSegmentationDataset(datasets)

    def setup(self):
        train_sources = self._sources_from_section(self.dataset_config["train"])
        val_sources = self._sources_from_section(self.dataset_config["val"])
        test_sources = self._sources_from_section(self.dataset_config["test"])
        self._validate_sources(train_sources, test_sources)

        self._train_dataset = self._build_multi_dataset(train_sources, self._train_transform())
        self._val_dataset = self._build_multi_dataset(val_sources, get_val_transforms(self.image_size))
        self._test_datasets = {
            source["name"]: self._build_multi_dataset([source], get_val_transforms(self.image_size))
            for source in test_sources
        }
        self._log_split_info()

    def _log_counts(self, label: str, dataset: MultiSourceSegmentationDataset) -> None:
        print(f"  {label}: {len(dataset)} images")
        for name, count in dataset.source_counts.items():
            split = self._split_descriptions.get(name, "unknown split")
            print(f"    - {name}: {count} ({split})")

    def _log_split_info(self):
        print("Multi-source dataset loaded:")
        self._log_counts("Train", self._train_dataset)
        self._log_counts("Validation", self._val_dataset)
        print("  Test:")
        for name, dataset in self._test_datasets.items():
            self._log_counts(name, dataset)

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

    def test_loaders(self) -> dict[str, DataLoader]:
        return {
            name: DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            for name, dataset in self._test_datasets.items()
        }

    def test_loader(self) -> DataLoader:
        if len(self._test_datasets) != 1:
            raise ValueError("Use test_loaders() when multiple test datasets are configured.")
        return next(iter(self.test_loaders().values()))
