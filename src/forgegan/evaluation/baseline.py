"""Split-isolated image/pixel baseline and train-only synthetic threshold calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from forgegan.data.registry import DatasetConfig, create_dataset
from forgegan.data.transforms import SpatialTransform
from forgegan.evaluation.metrics import binary_metrics


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    image_size: int = 64
    max_train: int = 64
    max_test: int = 256
    max_synthetic: int = 64
    normal_quantile: float = 0.99

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.image_size, self.max_train, self.max_test, self.max_synthetic
            )
        ):
            raise ValueError("image_size and sample limits must be positive integers")
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16")
        if not 0 < self.normal_quantile < 1:
            raise ValueError("normal_quantile must be in (0, 1)")


def _rgb(image: Image.Image, size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _binary_mask(image: Image.Image, size: int) -> np.ndarray:
    resized = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _build_normal_reference(
    dataset: Any, limit: int, size: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    train_samples: list[np.ndarray] = []
    for i in range(min(limit, len(dataset))):
        sample = dataset[i]
        if sample["split"] != "train" or sample["is_anomaly"]:
            raise ValueError("Normal reference must use normal train images only")
        train_samples.append(_rgb(sample["image"], size))
    if not train_samples:
        raise ValueError("Training split has no normal images")
    data = np.stack(train_samples, axis=0)
    mean = data.mean(axis=0)
    # Regularize image-aligned low-variance pixels to avoid dividing by ~zero.
    scale = np.maximum(data.std(axis=0), 0.08)
    return mean, scale, train_samples


def _score(image: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Per-pixel RGB standardized absolute reconstruction/prototype error."""
    return (np.abs(image - mean) / scale).mean(axis=-1)


def _image_score(score: np.ndarray) -> float:
    """Use top 1% of pixel anomaly scores for an image-level score."""
    count = max(1, score.size // 100)
    return float(np.partition(score.ravel(), -count)[-count:].mean())


def _load_synthetic(
    manifest_file: Path,
    config: DatasetConfig,
    train_source_paths: set[Path],
    test_source_paths: set[Path],
    limit: int,
    size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    manifest_file = manifest_file.resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("method") != "procedural_masked_intensity_baseline"
        or manifest.get("not_learned") is not True
        or manifest.get("source_split") != "train"
        or manifest.get("dataset") != config.name
    ):
        raise ValueError("Unsupported or inconsistent synthetic manifest")
    rows = manifest.get("items")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Synthetic manifest has no items")
    if manifest.get("count") != len(rows):
        raise ValueError("Synthetic manifest count differs from items")

    originals: dict[Path, str] = {}
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    root = manifest_file.parent
    root_resolved = root.resolve()
    for row in rows[:limit]:
        if (
            row.get("source_split") != "train"
            or row.get("source_category") != config.category
            or row.get("synthetic_label") != "procedural_anomaly"
        ):
            raise ValueError("Synthetic sample has invalid split, category or label")
        source = Path(row["source_path"]).resolve()
        if source not in train_source_paths or source in test_source_paths:
            raise ValueError("Synthetic source is not an approved train image")
        checksum = originals.setdefault(source, _sha256(source))
        if row.get("source_sha256") != checksum:
            raise ValueError("Synthetic source SHA-256 mismatch")
        paths = []
        for field in ("image", "mask"):
            relative = Path(row[field])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Synthetic paths must remain within the manifest directory")
            path = (root / relative).resolve()
            if not path.is_relative_to(root_resolved) or not path.is_file():
                raise ValueError("Synthetic image or mask escapes or is missing")
            paths.append(path)
        with Image.open(paths[0]) as source_image:
            images.append(_rgb(source_image, size))
        with Image.open(paths[1]) as source_mask:
            masks.append(_binary_mask(source_mask, size))
        if not masks[-1].any():
            raise ValueError("Synthetic anomaly mask is empty")
    return images, masks, {
        "manifest": str(manifest_file),
        "sample_count": len(images),
        "source_count": len(originals),
        "source_sha256_verified": True,
    }


def _calibrate_threshold(
    normal_scores: np.ndarray,
    synthetic_scores: np.ndarray,
    synthetic_masks: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    truth = np.concatenate(
        (np.zeros(normal_scores.size, dtype=bool), synthetic_masks.ravel())
    )
    values = np.concatenate((normal_scores.ravel(), synthetic_scores.ravel()))
    # Threshold tuning uses ONLY train normals and synthetic train-derived masks.
    candidates = np.unique(np.quantile(values, np.linspace(0.01, 0.999, 100)))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        predicted = values >= threshold
        tp = int(np.count_nonzero(truth & predicted))
        fp = int(np.count_nonzero(~truth & predicted))
        fn = int(np.count_nonzero(truth & ~predicted))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = f1
    return best_threshold, {"train_synthetic_pixel_f1": best_f1}


def compare_real_vs_synthetic(
    config: DatasetConfig,
    synthetic_manifest: str | Path,
    *,
    settings: EvaluationSettings | None = None,
) -> dict[str, Any]:
    """Evaluate SAME normal prototype scores with two train-only thresholds.

    Real-only threshold = train-normal upper quantile.
    Synthetic-calibrated threshold = train-normal + train-derived synthetic mask F1.
    Test set is opened only AFTER both thresholds are determined.
    """
    if config.split != "train" or config.category is None:
        raise ValueError("Evaluation needs a single category and official train split")
    settings = settings or EvaluationSettings()
    training = create_dataset(config)
    held_out = create_dataset(
        DatasetConfig(
            name=config.name,
            root=config.root,
            split="test",
            category=config.category,
            split_file=config.split_file,
            verify_files=config.verify_files,
        )
    )
    train_paths = {r.image_path.resolve() for r in training.records}
    test_paths = {r.image_path.resolve() for r in held_out.records}
    if train_paths & test_paths:
        raise ValueError("Exact source path overlaps train and test splits")
    # Catch exact byte-content copies under different paths (within this category).
    train_hashes = {_sha256(path) for path in train_paths}
    if any(_sha256(path) in train_hashes for path in test_paths):
        raise ValueError("Exact source content overlaps train and test splits")

    mean, scale, normal_images = _build_normal_reference(
        training, settings.max_train, settings.image_size
    )
    normal_scores = np.stack([_score(im, mean, scale) for im in normal_images])
    threshold_real = float(np.quantile(normal_scores, settings.normal_quantile))

    synthetic_images, synthetic_masks, provenance = _load_synthetic(
        Path(synthetic_manifest),
        config,
        train_paths,
        test_paths,
        settings.max_synthetic,
        settings.image_size,
    )
    calibrated_scores = np.stack(
        [_score(image, mean, scale) for image in synthetic_images]
    )
    threshold_augmented, calibration = _calibrate_threshold(
        normal_scores, calibrated_scores, np.stack(synthetic_masks)
    )

    # The official test set does not influence score fitting or thresholds.
    image_truth: list[bool] = []
    image_scores: list[float] = []
    pixel_truth: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []
    for index in range(min(len(held_out), settings.max_test)):
        sample = held_out[index]
        image = _rgb(sample["image"], settings.image_size)
        anomaly = _score(image, mean, scale)
        gt_mask = sample["mask"]
        if sample["is_anomaly"] and gt_mask is None:
            raise ValueError("Held-out anomalous test image has no mask")
        truth = (
            _binary_mask(gt_mask, settings.image_size)
            if gt_mask is not None
            else np.zeros((settings.image_size, settings.image_size), dtype=bool)
        )
        if sample["is_anomaly"] and not truth.any():
            raise ValueError("Held-out anomaly has an empty localization mask")
        image_truth.append(bool(sample["is_anomaly"]))
        image_scores.append(_image_score(anomaly))
        pixel_truth.append(truth)
        pixel_scores.append(anomaly)
        samples.append({
            "source_path": sample["image_path"],
            "label": sample["label"],
            "score": image_scores[-1],
        })
    if not samples:
        raise ValueError("Test split is empty")

    image_truth_array = np.asarray(image_truth, dtype=bool)
    image_score_array = np.asarray(image_scores, dtype=np.float64)
    pixel_truth_array = np.stack(pixel_truth)
    pixel_score_array = np.stack(pixel_scores)
    # For image classification, calibrate thresholds on image scores, separately.
    normal_image_scores = np.asarray(
        [_image_score(score) for score in normal_scores], dtype=np.float64
    )
    synthetic_image_scores = np.asarray(
        [_image_score(score) for score in calibrated_scores], dtype=np.float64
    )
    img_threshold_real = float(
        np.quantile(normal_image_scores, settings.normal_quantile)
    )
    img_threshold_augmented, img_calibration = _calibrate_threshold(
        normal_image_scores,
        synthetic_image_scores,
        np.ones(len(synthetic_image_scores), dtype=bool),
    )
    outcomes: dict[str, Any] = {}
    for name, pixel_threshold, image_threshold in (
        ("real_only", threshold_real, img_threshold_real),
        ("synthetic_calibrated", threshold_augmented, img_threshold_augmented),
    ):
        outcomes[name] = {
            "pixel_threshold": pixel_threshold,
            "image_threshold": image_threshold,
            "pixel": binary_metrics(
                pixel_truth_array, pixel_score_array, pixel_threshold
            ),
            "image": binary_metrics(
                image_truth_array, image_score_array, image_threshold
            ),
        }
    return {
        "schema_version": 1,
        "study": "normal_prototype_with_train_only_synthetic_threshold_calibration",
        "dataset": config.name,
        "category": config.category,
        "settings": {
            "image_size": settings.image_size,
            "max_train": settings.max_train,
            "max_test": settings.max_test,
            "max_synthetic": settings.max_synthetic,
            "normal_quantile": settings.normal_quantile,
        },
        "train_normal_count": len(normal_images),
        "test_count": len(samples),
        "test_normal_count": image_truth.count(False),
        "test_anomaly_count": image_truth.count(True),
        "train_test_exact_path_and_hash_overlap": False,
        "synthetic": provenance,
        "calibration": {**calibration, **img_calibration},
        "arms": outcomes,
        "test_samples": samples,
        "interpretation": (
            "Same one-class Gaussian pixel score in both arms; synthetic images "
            "ONLY change train-derived decision thresholds. ROC-AUC/AP are identical "
            "by construction; this does NOT measure learned synthetic augmentation gain."
        ),
    }


def save_comparison(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write an experiment report without overwriting previous results."""
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return destination
