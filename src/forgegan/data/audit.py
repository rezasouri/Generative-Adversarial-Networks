"""Dataset integrity and metadata audit utilities."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def audit_dataset(root: str | Path) -> dict[str, Any]:
    """Audit class-folder image datasets without modifying source files."""
    dataset_root = Path(root).expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    class_dirs = sorted(
        (
            path
            for path in dataset_root.iterdir()
            if path.is_dir() and path.name.startswith("Class ")
        ),
        key=lambda path: path.name,
    )
    if not class_dirs:
        raise ValueError(f"No class directories named 'Class *' found in: {dataset_root}")

    all_hashes: dict[str, list[str]] = defaultdict(list)
    class_reports: list[dict[str, Any]] = []
    total_bytes = 0
    total_images = 0
    total_invalid = 0

    for class_dir in class_dirs:
        image_paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        sizes: list[int] = []
        dimensions: set[tuple[int, int]] = set()
        modes: set[str] = set()
        formats: set[str] = set()
        invalid_files: list[str] = []

        for path in image_paths:
            relative_path = path.relative_to(dataset_root).as_posix()
            file_size = path.stat().st_size
            sizes.append(file_size)
            total_bytes += file_size
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                all_hashes[digest].append(relative_path)
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    dimensions.add(tuple(image.size))
                    modes.add(image.mode)
                    formats.add(image.format or path.suffix.lstrip(".").upper())
            except (OSError, ValueError, Image.DecompressionBombError):
                invalid_files.append(relative_path)

        valid_count = len(image_paths) - len(invalid_files)
        total_images += len(image_paths)
        total_invalid += len(invalid_files)
        class_reports.append(
            {
                "name": class_dir.name,
                "image_count": len(image_paths),
                "valid_count": valid_count,
                "invalid_count": len(invalid_files),
                "size_bytes": {
                    "min": min(sizes) if sizes else None,
                    "median": median(sizes) if sizes else None,
                    "max": max(sizes) if sizes else None,
                    "total": sum(sizes),
                },
                "dimensions": [list(value) for value in sorted(dimensions)],
                "modes": sorted(modes),
                "formats": sorted(formats),
                "invalid_files": invalid_files,
            }
        )

    duplicate_groups = [
        sorted(paths) for paths in all_hashes.values() if len(paths) > 1
    ]
    counts = [item["image_count"] for item in class_reports]
    return {
        "root": str(dataset_root.resolve()),
        "class_count": len(class_reports),
        "image_count": total_images,
        "valid_count": total_images - total_invalid,
        "invalid_count": total_invalid,
        "total_size_bytes": total_bytes,
        "class_imbalance": {
            "min_images_per_class": min(counts),
            "max_images_per_class": max(counts),
            "balanced": len(set(counts)) <= 1,
        },
        "duplicate_content_groups": duplicate_groups,
        "duplicate_content_file_count": sum(len(group) for group in duplicate_groups),
        "classes": class_reports,
    }
