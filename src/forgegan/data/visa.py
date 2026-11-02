"""VisA dataset indexing, validation, and sample loading.

The adapter reads the official 1-class split CSV and leaves source data untouched.
It intentionally does not download or extract third-party data.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from PIL import Image

VISA_CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
VISA_SPLITS = {"train", "test"}
VISA_LABELS = {"normal", "anomaly"}
_REQUIRED_COLUMNS = {"object", "split", "label", "image", "mask"}


@dataclass(frozen=True, slots=True)
class VisaRecord:
    """One source image and its official split/annotation metadata."""

    category: str
    split: str
    label: str
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


def _resolve_dataset_path(root: Path, value: str, *, field: str) -> Path | None:
    """Resolve a CSV-relative path and reject absolute paths or traversal."""
    if not value.strip():
        return None

    relative = PurePosixPath(value.strip())
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe {field} path in VisA CSV: {value!r}")

    candidate = (root / Path(*relative.parts)).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"{field} path escapes the VisA root: {value!r}")
    return candidate


def load_visa_records(
    root: str | Path,
    split_file: str | Path | None = None,
    *,
    category: str | None = None,
    verify_files: bool = True,
) -> list[VisaRecord]:
    """Read the official VisA 1-class split CSV into validated records.

    Root points to the extracted VisA directory containing category folders.
    By default, the official split_csv/1cls.csv file is used.
    """
    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"VisA root does not exist: {dataset_root}")

    if category is not None and category not in VISA_CATEGORIES:
        raise ValueError(
            f"Unknown VisA category {category!r}. "
            f"Expected one of: {', '.join(VISA_CATEGORIES)}"
        )

    csv_path = (
        Path(split_file).expanduser().resolve()
        if split_file is not None
        else dataset_root / "split_csv" / "1cls.csv"
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"VisA split CSV not found: {csv_path}")

    records: list[VisaRecord] = []
    seen_images: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing_columns = _REQUIRED_COLUMNS - columns
        if missing_columns:
            raise ValueError(
                "VisA split CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            row_category = (row.get("object") or "").strip()
            row_split = (row.get("split") or "").strip().lower()
            row_label = (row.get("label") or "").strip().lower()
            if category is not None and row_category != category:
                continue

            if row_category not in VISA_CATEGORIES:
                raise ValueError(
                    f"Unknown VisA category {row_category!r} at CSV row {row_number}"
                )
            if row_split not in VISA_SPLITS:
                raise ValueError(f"Unsupported split {row_split!r} at CSV row {row_number}")
            if row_label not in VISA_LABELS:
                raise ValueError(f"Unsupported label {row_label!r} at CSV row {row_number}")
            if row_split == "train" and row_label != "normal":
                raise ValueError(
                    f"Official 1-class training rows must be normal (CSV row {row_number})"
                )

            image_path = _resolve_dataset_path(
                dataset_root, (row.get("image") or "").strip(), field="image"
            )
            if image_path is None:
                raise ValueError(f"Missing image path at CSV row {row_number}")

            mask_path = _resolve_dataset_path(
                dataset_root, (row.get("mask") or "").strip(), field="mask"
            )
            if row_label == "anomaly" and mask_path is None:
                raise ValueError(f"Anomalous image has no mask path at CSV row {row_number}")
            if row_label == "normal" and mask_path is not None:
                raise ValueError(
                    f"Normal image unexpectedly has a mask path at CSV row {row_number}"
                )

            relative_image = image_path.relative_to(dataset_root).as_posix()
            previous_split = seen_images.get(relative_image)
            if previous_split is not None:
                raise ValueError(
                    f"Image appears more than once in the split CSV: {relative_image} "
                    f"({previous_split} and {row_split})"
                )
            seen_images[relative_image] = row_split

            if verify_files:
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image referenced by VisA CSV does not exist: {image_path}"
                    )
                if mask_path is not None and not mask_path.is_file():
                    raise FileNotFoundError(
                        f"Mask referenced by VisA CSV does not exist: {mask_path}"
                    )

            records.append(
                VisaRecord(
                    category=row_category,
                    split=row_split,
                    label=row_label,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )

    if not records:
        scope = f" for category {category!r}" if category else ""
        raise ValueError(f"No VisA records found{scope} in {csv_path}")
    return records


def validate_visa_split(records: list[VisaRecord]) -> dict[str, Any]:
    """Summarize split coverage and enforce source-image isolation."""
    if not records:
        raise ValueError("Cannot validate an empty VisA record collection")

    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    image_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_category[record.category][f"{record.split}_{record.label}"] += 1
        image_splits[str(record.image_path)].add(record.split)

    leakage = sorted(path for path, splits in image_splits.items() if len(splits) > 1)
    if leakage:
        raise ValueError(f"VisA source-image leakage detected across splits: {leakage[:5]}")

    categories = {
        name: dict(sorted(counts.items()))
        for name, counts in sorted(by_category.items())
    }
    totals: Counter[str] = Counter()
    for counts in by_category.values():
        totals.update(counts)

    return {
        "dataset": "VisA",
        "protocol": "official_1cls",
        "record_count": len(records),
        "category_count": len(categories),
        "categories": categories,
        "totals": dict(sorted(totals.items())),
        "source_image_leakage": False,
    }


Transform = Callable[[Image.Image], Any]


class VisaDataset:
    """Lightweight map-style VisA dataset compatible with PyTorch DataLoader.

    Pass transforms that return tensors to use the default PyTorch collation
    path. The core package does not require PyTorch at import time.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: Literal["train", "test"],
        category: str | None = None,
        split_file: str | Path | None = None,
        image_transform: Transform | None = None,
        mask_transform: Transform | None = None,
        mask_mode: Literal["raw", "binary"] = "raw",
        verify_files: bool = True,
    ) -> None:
        if split not in VISA_SPLITS:
            raise ValueError(f"split must be one of {sorted(VISA_SPLITS)}, got {split!r}")
        if mask_mode not in {"raw", "binary"}:
            raise ValueError("mask_mode must be 'raw' or 'binary'")

        records = load_visa_records(
            root, split_file, category=category, verify_files=verify_files
        )
        self.records = [record for record in records if record.split == split]
        if not self.records:
            raise ValueError(f"No VisA records found for split {split!r}")
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
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path) if record.mask_path else None,
        }
