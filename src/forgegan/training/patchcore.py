"""PatchCore-style training-free industrial anomaly evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from forgegan.data.registry import DatasetConfig, create_dataset
from forgegan.evaluation.baseline import _rgb, _sha256
from forgegan.evaluation.metrics import binary_metrics
from forgegan.models.patchcore import (
    ResNetFeatureExtractor,
    greedy_coreset,
    nearest_neighbor_scores,
)


def _image_tensor(paths: list[Path], size: int) -> np.ndarray:
    """Load RGB images as contiguous float32 NHWC arrays in [0, 1]."""
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(_rgb(image, size))
    return np.stack(images).astype(np.float32)


def _extract_embeddings(
    extractor: ResNetFeatureExtractor,
    images: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    import torch

    all_embeddings: list[np.ndarray] = []
    grid: tuple[int, int] | None = None
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[start : start + batch_size])
        batch = batch.permute(0, 3, 1, 2).to(device)
        feature_map = extractor(batch)
        grid = (int(feature_map.shape[-2]), int(feature_map.shape[-1]))
        patches = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
        all_embeddings.append(patches.cpu().numpy().astype(np.float32))
    if grid is None:
        raise ValueError("Cannot extract embeddings from an empty image set")
    return np.concatenate(all_embeddings, axis=0), grid


def _score_images(
    extractor: ResNetFeatureExtractor,
    images: np.ndarray,
    memory: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    maps: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[start : start + batch_size])
        batch = batch.permute(0, 3, 1, 2).to(device)
        feature_map = extractor(batch)
        batch_size_actual, channels, height, width = feature_map.shape
        queries = feature_map.permute(0, 2, 3, 1).reshape(-1, channels).cpu().numpy()
        scores = nearest_neighbor_scores(queries, memory)
        score_map = torch.from_numpy(scores.reshape(batch_size_actual, height, width))
        score_map = functional.interpolate(
            score_map.unsqueeze(1),
            size=images.shape[1:3],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        maps.append(score_map.cpu().numpy())
    anomaly_maps = np.concatenate(maps, axis=0)
    image_scores = anomaly_maps.reshape(len(images), -1).max(axis=1)
    return anomaly_maps, image_scores


def run_patchcore_benchmark(
    config: DatasetConfig,
    output_path: str | Path,
    *,
    image_size: int = 224,
    max_train: int = 128,
    max_test: int = 512,
    memory_size: int = 2048,
    batch_size: int = 16,
    quantile: float = 0.99,
    pretrained: bool = True,
    device: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    """Fit a PatchCore-style memory bank on normal train data and evaluate held-out test."""
    import torch

    if config.split != "train" or config.category is None:
        raise ValueError("Use one category and the official train split")
    if image_size < 64 or image_size % 32 != 0:
        raise ValueError("image_size must be >=64 and divisible by 32")
    if max_train < 3 or max_test < 1 or memory_size < 1 or batch_size < 1:
        raise ValueError("Sample limits, memory size and batch size must be positive")
    if not 0 < quantile < 1 or type(seed) is not int or seed < 0:
        raise ValueError("quantile must be in (0, 1) and seed must be non-negative")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_dataset = create_dataset(config)
    normal_records = [record for record in train_dataset.records if not record.is_anomaly]
    if len(normal_records) < 3:
        raise ValueError("At least three normal train images are required")
    selected = sorted(normal_records, key=lambda record: str(record.image_path))[:max_train]
    calibration_count = max(1, int(round(len(selected) * 0.2)))
    calibration_records = selected[:calibration_count]
    fit_records = selected[calibration_count:]
    if len(fit_records) < 2:
        raise ValueError("Need at least two fit images after calibration holdout")

    all_train_paths = {record.image_path.resolve() for record in normal_records}
    test_dataset = create_dataset(
        DatasetConfig(
            name=config.name,
            root=config.root,
            split="test",
            category=config.category,
            split_file=config.split_file,
            verify_files=config.verify_files,
        )
    )
    test_records = test_dataset.records[:max_test]
    test_paths = {record.image_path.resolve() for record in test_dataset.records}
    if all_train_paths & test_paths:
        raise ValueError("Exact train/test path overlap")
    train_hashes = {
        _sha256(path) for path in all_train_paths
    }
    if any(
        _sha256(path) in train_hashes for path in test_paths
    ):
        raise ValueError("Exact train/test content overlap")
    if not test_records or len({record.is_anomaly for record in test_records}) != 2:
        raise ValueError("Test subset must contain normal and anomalous samples")

    extractor = ResNetFeatureExtractor(pretrained=pretrained, device=device)
    fit_images = _image_tensor(
        [record.image_path for record in fit_records], image_size
    )
    calibration_images = _image_tensor(
        [record.image_path for record in calibration_records], image_size
    )
    test_images = _image_tensor(
        [record.image_path for record in test_records], image_size
    )
    fit_embeddings, grid = _extract_embeddings(
        extractor, fit_images, device=device, batch_size=batch_size
    )
    memory = greedy_coreset(fit_embeddings, max_points=memory_size, seed=seed)

    calibration_maps, calibration_image_scores = _score_images(
        extractor, calibration_images, memory, device=device, batch_size=batch_size
    )
    pixel_threshold = float(np.quantile(calibration_maps, quantile))
    image_threshold = float(np.quantile(calibration_image_scores, quantile))
    test_maps, test_image_scores = _score_images(
        extractor, test_images, memory, device=device, batch_size=batch_size
    )

    masks: list[np.ndarray] = []
    labels: list[bool] = []
    test_manifest: list[dict[str, Any]] = []
    for record in test_records:
        labels.append(bool(record.is_anomaly))
        if record.mask_path is None:
            mask = np.zeros((image_size, image_size), dtype=bool)
        else:
            with Image.open(record.mask_path) as image:
                mask = np.asarray(
                    image.convert("L").resize(
                        (image_size, image_size), Image.Resampling.NEAREST
                    )
                ) > 0
        if record.is_anomaly and not mask.any():
            raise ValueError(f"Anomalous sample has empty mask: {record.image_path}")
        masks.append(mask)
        test_manifest.append({
            "source_path": str(record.image_path.resolve()),
            "label": record.label,
        })

    report = {
        "schema_version": 1,
        "method": "patchcore_resnet18_layer2_layer3",
        "dataset": config.name,
        "category": config.category,
        "backbone": "resnet18",
        "pretrained": pretrained,
        "device": device,
        "settings": {
            "image_size": image_size,
            "normal_fit_count": len(fit_records),
            "normal_calibration_count": len(calibration_records),
            "test_count": len(test_records),
            "fit_patch_count": len(fit_embeddings),
            "memory_bank_count": len(memory),
            "feature_grid": list(grid),
            "batch_size": batch_size,
            "threshold_quantile": quantile,
            "seed": seed,
        },
        "thresholds": {"pixel": pixel_threshold, "image": image_threshold},
        "pixel": binary_metrics(np.stack(masks), test_maps, pixel_threshold),
        "image": binary_metrics(
            np.asarray(labels, dtype=bool), test_image_scores, image_threshold
        ),
        "test_samples": test_manifest,
        "test_image_scores": test_image_scores.tolist(),
        "provenance": {
            "train_test_exact_path_overlap": False,
            "train_test_exact_sha256_overlap": False,
            "threshold_source": "disjoint normal-only official train holdout",
            "memory_source": "normal-only official train fit subset",
        },
        "limitations": [
            "PatchCore-style ResNet18 layer2/layer3 implementation; "
            "not the official Anomalib implementation.",
            "Greedy coreset is bounded to a deterministic candidate subset for runtime.",
            "No perceptual near-duplicate or product-instance leakage audit.",
            "A max-test subset is not a full official benchmark.",
            "ImageNet pretrained weights are required for meaningful pretrained results.",
        ],
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {
        "report_path": str(destination),
        "dataset": config.name,
        "category": config.category,
        "image": report["image"],
        "pixel": report["pixel"],
        "memory_bank_count": len(memory),
    }
