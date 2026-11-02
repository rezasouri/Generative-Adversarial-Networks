"""MVTec AD indexing, validation, and sample loading.

The adapter reads the original MVTec AD directory layout without modifying
source data. It does not download or extract the dataset.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)
MVTEC_SPLITS = {"train", "test"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = {".png"}


@dataclass(frozen=True, slots=True)
class MVTecRecord:
    """One MVTec image and its inferred ground-truth metadata."""

    category: str
    split: str
    label: str
    defect_type: str
    image_path: Path
    mask_path: Path | None

    @property
    def is_anomaly(self) -> bool:
        return self.label == "anomaly"

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["image_path"] = str(self.image_path)
        record["mask_path"] = str(self.mask_path) if self.mask_path else None
        record["is_anomaly"] = self.is_anomaly
        return record


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _resolve_mask(category_root: Path, defect_type: str, image_path: Path) -> Path:
    """Resolve the official <stem>_mask.png annotation for an anomaly image."""
    mask_dir = category_root / "ground_truth" / defect_type
    candidates = [
        mask_dir / f"{image_path.stem}_mask{suffix}"
        for suffix in sorted(MASK_SUFFIXES)
    ]
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if not resolved.is_relative_to(category_root.resolve()):
                raise ValueError(f"Ground-truth mask escapes category root: {candidate}")
            return resolved
    expected = candidates[0]
    raise FileNotFoundError(
        f"Missing MVTec ground-truth mask for {image_path}: expected {expected}"
    )


def load_mvtec_records(
    root: str | Path,
    *,
    category: str | None = None,
    verify_files: bool = True,
) -> list[MVTecRecord]:
    """Index the official MVTec AD train/test/ground_truth directory structure.

    Root must contain one directory per MVTec category. Training images are
    read only from train/good; test images are read from test/good and each
    defect-type directory. Anomaly masks use ground_truth/<defect>/<stem>_mask.png.
    """
    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"MVTec AD root does not exist: {dataset_root}")

    if category is not None and category not in MVTEC_CATEGORIES:
        raise ValueError(
            f"Unknown MVTec category {category!r}. "
            f"Expected one of: {', '.join(MVTEC_CATEGORIES)}"
        )

    categories = [category] if category else list(MVTEC_CATEGORIES)
    records: list[MVTecRecord] = []

    for category_name in categories:
        category_root = dataset_root / category_name
        if not category_root.is_dir():
            raise FileNotFoundError(
                f"MVTec category directory is missing: {category_root}"
            )

        train_good = category_root / "train" / "good"
        test_root = category_root / "test"
        ground_truth_root = category_root / "ground_truth"
        if not train_good.is_dir():
            raise FileNotFoundError(f"MVTec training directory is missing: {train_good}")
        if not test_root.is_dir():
            raise FileNotFoundError(f"MVTec test directory is missing: {test_root}")
        if not ground_truth_root.is_dir():
            raise FileNotFoundError(
                f"MVTec ground_truth directory is missing: {ground_truth_root}"
            )

        train_images = _image_files(train_good)
        if not train_images:
            raise ValueError(f"No training images found in {train_good}")
        for image_path in train_images:
            records.append(
                MVTecRecord(
                    category=category_name,
                    split="train",
                    label="normal",
                    defect_type="good",
                    image_path=image_path.resolve(),
                    mask_path=None,
                )
            )

        test_dirs = sorted(path for path in test_root.iterdir() if path.is_dir())
        if not test_dirs or not (test_root / "good").is_dir():
            raise ValueError(
                f"MVTec test split must contain a 'good' directory: {test_root}"
            )

        for defect_dir in test_dirs:
            defect_type = defect_dir.name
            image_paths = _image_files(defect_dir)
            if not image_paths:
                raise ValueError(f"No test images found in {defect_dir}")
            is_anomaly = defect_type != "good"
            for image_path in image_paths:
                mask_path = (
                    _resolve_mask(category_root, defect_type, image_path)
                    if is_anomaly
                    else None
                )
                records.append(
                    MVTecRecord(
                        category=category_name,
                        split="test",
                        label="anomaly" if is_anomaly else "normal",
                        defect_type=defect_type,
                        image_path=image_path.resolve(),
                        mask_path=mask_path,
                    )
                )

        if verify_files:
            for record in records:
                if record.category != category_name:
                    continue
                if not record.image_path.is_file():
                    raise FileNotFoundError(
                        f"MVTec image referenced by directory layout is missing: "
                        f"{record.image_path}"
                    )
                if record.mask_path is not None and not record.mask_path.is_file():
                    raise FileNotFoundError(
                        f"MVTec mask referenced by directory layout is missing: "
                        f"{record.mask_path}"
                    )

    if not records:
        raise ValueError(f"No MVTec images found under {dataset_root}")
    return records


def validate_mvtec_split(records: list[MVTecRecord]) -> dict[str, Any]:
    """Summarize category/split coverage and verify source-image isolation."""
    if not records:
        raise ValueError("Cannot validate an empty MVTec record collection")

    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    image_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_category[record.category][
            f"{record.split}_{record.label}"
        ] += 1
        image_splits[str(record.image_path)].add(record.split)

    leakage = sorted(path for path, splits in image_splits.items() if len(splits) > 1)
    if leakage:
        raise ValueError(f"MVTec source-image leakage detected across splits: {leakage[:5]}")

    categories = {
        name: dict(sorted(counts.items()))
        for name, counts in sorted(by_category.items())
    }
    totals: Counter[str] = Counter()
    for counts in by_category.values():
        totals.update(counts)

    defect_types = sorted(
        {record.defect_type for record in records if record.is_anomaly}
    )
    return {
        "dataset": "MVTec AD",
        "protocol": "official_directory_split",
        "record_count": len(records),
        "category_count": len(categories),
        "categories": categories,
        "totals": dict(sorted(totals.items())),
        "anomaly_types": defect_types,
        "source_image_leakage": False,
    }


Transform = Callable[[Image.Image], Any]


class MVTecDataset:
    """Map-style MVTec AD dataset compatible with PyTorch DataLoader.

    The adapter does not import PyTorch. Supply transforms that return tensors
    when using PyTorch's default collation.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: Literal["train", "test"],
        category: str | None = None,
        image_transform: Transform | None = None,
        mask_transform: Transform | None = None,
        mask_mode: Literal["raw", "binary"] = "raw",
        verify_files: bool = True,
    ) -> None:
        if split not in MVTEC_SPLITS:
            raise ValueError(f"split must be one of {sorted(MVTEC_SPLITS)}, got {split!r}")
        if mask_mode not in {"raw", "binary"}:
            raise ValueError("mask_mode must be 'raw' or 'binary'")

        records = load_mvtec_records(root, category=category, verify_files=verify_files)
        self.records = [record for record in records if record.split == split]
        if not self.records:
            raise ValueError(f"No MVTec records found for split {split!r}")
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mask_mode = mask_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        mask: Any = None
        if record.mask_path is not None:
            with Image.open(record.mask_path) as source:
                mask = source.copy()
            if mask.size != Image.open(record.image_path).size:
                raise ValueError(
                    f"MVTec image/mask dimensions differ for {record.image_path}: "
                    f"{Image.open(record.image_path).size} vs {mask.size}"
                )
            if self.mask_mode == "binary":
                mask = mask.convert("L").point(lambda value: 255 if value > 0 else 0)
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)

        return {
            "image": image,
            "mask": mask,
            "category": record.category,
            "split": record.split,
            "label": record.label,
            "is_anomaly": record.is_anomaly,
            "defect_type": record.defect_type,
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path) if record.mask_path else None,
        }
