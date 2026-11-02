"""Factory and normalized records for supported industrial datasets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forgegan.data.mvtec import MVTecDataset, load_mvtec_records
from forgegan.data.visa import VisaDataset, load_visa_records

DATASET_NAMES = ("mvtec", "visa")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Immutable input for the shared dataset factory."""

    name: str
    root: Path
    split: str
    category: str | None = None
    split_file: Path | None = None
    verify_files: bool = True

    def __post_init__(self) -> None:
        if self.name not in DATASET_NAMES:
            raise ValueError(f"Unknown dataset {self.name!r}; supported: {DATASET_NAMES}")
        if self.split not in {"train", "test"}:
            raise ValueError("Official dataset split must be 'train' or 'test'")
        if self.name == "mvtec" and self.split_file is not None:
            raise ValueError("MVTec uses directory-based splits; split_file is not supported")


class NormalizedDataset:
    """Expose a stable sample schema while retaining original dataset semantics."""

    def __init__(self, source: Any, name: str) -> None:
        self.source = source
        self.name = name
        self.records = source.records

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.source[index]
        return {
            "image": sample["image"],
            "mask": sample["mask"],
            "dataset": self.name,
            "category": sample["category"],
            "split": sample["split"],
            "label": sample["label"],
            "is_anomaly": sample["is_anomaly"],
            "defect_type": sample.get("defect_type"),
            "image_path": sample["image_path"],
            "mask_path": sample["mask_path"],
        }


def create_dataset(
    config: DatasetConfig,
    *,
    image_transform: Any = None,
    mask_transform: Any = None,
) -> NormalizedDataset:
    """Create a read-only dataset with a normalized output schema."""
    common = {
        "split": config.split,
        "category": config.category,
        "verify_files": config.verify_files,
        "image_transform": image_transform,
        "mask_transform": mask_transform,
        "mask_mode": "binary",
    }
    if config.name == "visa":
        source = VisaDataset(
            config.root, split_file=config.split_file, **common
        )
    else:
        source = MVTecDataset(config.root, **common)
    return NormalizedDataset(source, config.name)


def validate_source_splits(
    configs: list[DatasetConfig], *, check_files: bool = False
) -> dict[str, Any]:
    """Reject exact source-image paths shared between split assignments.

    This checks source identity by resolved paths, not visual similarity or
    near-duplicate content. It does not certify leakage-free image semantics.
    """
    if not configs:
        raise ValueError("At least one dataset configuration is required")

    membership: dict[str, set[str]] = defaultdict(set)
    datasets: set[str] = set()
    for config in configs:
        if config.name == "visa":
            records = load_visa_records(
                config.root,
                config.split_file,
                category=config.category,
                verify_files=check_files,
            )
        else:
            records = load_mvtec_records(
                config.root, category=config.category, verify_files=check_files
            )
        datasets.add(config.name)
        for record in records:
            if record.split == config.split:
                key = str(record.image_path.resolve())
                membership[key].add(record.split)

    conflicts = sorted(path for path, splits in membership.items() if len(splits) > 1)
    if conflicts:
        raise ValueError(f"Source-image split leakage detected: {conflicts[:5]}")
    return {
        "datasets": sorted(datasets),
        "source_images_checked": len(membership),
        "source_image_path_leakage": False,
        "warning": "Path-only check; visually duplicate or derived files are not detected.",
    }
