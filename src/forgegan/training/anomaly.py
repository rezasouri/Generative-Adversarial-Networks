"""Matched real-only vs procedural-synthetic denoising anomaly benchmark."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from forgegan.data.registry import DatasetConfig, create_dataset
from forgegan.evaluation.baseline import _image_score, _rgb, _sha256
from forgegan.evaluation.metrics import binary_metrics


def _load_manifest_pairs(
    manifest_path: Path,
    config: DatasetConfig,
    train_paths: set[Path],
    fit_paths: set[Path],
    test_paths: set[Path],
    size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("method") != "procedural_masked_intensity_baseline"
        or manifest.get("not_learned") is not True
        or manifest.get("source_split") != "train"
        or manifest.get("dataset") != config.name
    ):
        raise ValueError("Unsupported synthetic manifest")
    rows = manifest.get("items")
    if not isinstance(rows, list) or not rows or manifest.get("count") != len(rows):
        raise ValueError("Synthetic manifest is empty or inconsistent")
    root = manifest_path.parent.resolve()
    synthetic: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sources: set[Path] = set()
    for row in rows:
        if (
            row.get("source_split") != "train"
            or row.get("source_category") != config.category
            or row.get("synthetic_label") != "procedural_anomaly"
        ):
            raise ValueError("Synthetic record has invalid split/category/label")
        source = Path(row["source_path"]).resolve()
        if source not in train_paths or source in test_paths:
            raise ValueError("Synthetic source is not an approved normal train image")
        if row.get("source_sha256") != _sha256(source):
            raise ValueError("Synthetic source SHA-256 mismatch")
        # Calibration-holdout sources are deliberately excluded from model fitting.
        if source not in fit_paths:
            continue
        sources.add(source)
        resolved: list[Path] = []
        for field in ("image", "mask"):
            relative = Path(row[field])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Synthetic paths must be manifest-relative")
            candidate = (root / relative).resolve()
            if not candidate.is_relative_to(root) or not candidate.is_file():
                raise ValueError("Synthetic file escapes manifest directory or is missing")
            resolved.append(candidate)
        with Image.open(resolved[0]) as image:
            synthetic.append(_rgb(image, size))
        with Image.open(resolved[1]) as mask:
            if not np.asarray(mask.convert("L")).any():
                raise ValueError("Synthetic mask is empty")
        with Image.open(source) as image:
            targets.append(_rgb(image, size))
    return synthetic, targets, {
        "manifest": str(manifest_path),
        "synthetic_count": len(synthetic),
        "source_count": len(sources),
        "source_sha256_verified": True,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _train_one(
    normal: np.ndarray,
    synthetic: np.ndarray,
    synthetic_targets: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    width: int,
    device: str,
    augmented: bool,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    from forgegan.models.autoencoder import ConvAutoencoder

    _seed_everything(seed)
    model = ConvAutoencoder(width=width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    normal_tensor = torch.from_numpy(normal).permute(0, 3, 1, 2).to(device)
    synthetic_tensor = torch.from_numpy(synthetic).permute(0, 3, 1, 2).to(device)
    target_tensor = torch.from_numpy(synthetic_targets).permute(0, 3, 1, 2).to(device)
    steps_per_epoch = max(1, int(np.ceil(len(normal) / batch_size)))
    history: list[dict[str, float]] = []
    rng = torch.Generator(device="cpu").manual_seed(seed + 1000)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for _ in range(steps_per_epoch):
            if augmented:
                half = max(1, batch_size // 2)
                other = batch_size - half
                normal_idx = torch.randint(len(normal), (half,), generator=rng)
                synth_idx = torch.randint(len(synthetic), (other,), generator=rng)
                inputs = torch.cat(
                    (normal_tensor[normal_idx.to(device)], synthetic_tensor[synth_idx.to(device)])
                )
                targets = torch.cat(
                    (normal_tensor[normal_idx.to(device)], target_tensor[synth_idx.to(device)])
                )
            else:
                idx = torch.randint(len(normal), (batch_size,), generator=rng)
                inputs = normal_tensor[idx.to(device)]
                targets = inputs
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(inputs)
            loss = functional.l1_loss(reconstruction, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        history.append({"epoch": epoch + 1, "train_l1": total / steps_per_epoch})
    return model, {
        "epochs": epochs,
        "optimizer_steps": epochs * steps_per_epoch,
        "batch_size": batch_size,
        "history": history,
        "training_mode": (
            "normal_plus_synthetic_denoising"
            if augmented
            else "normal_identity_reconstruction"
        ),
    }


def _score_batch(model: Any, images: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    results: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start : start + batch_size])
            batch = batch.permute(0, 3, 1, 2).to(device)
            reconstruction = model(batch)
            residual = (batch - reconstruction).abs().mean(dim=1)
            results.append(residual.cpu().numpy())
    return np.concatenate(results, axis=0)


def _arm_metrics(
    model: Any,
    normal: np.ndarray,
    test_images: np.ndarray,
    test_masks: np.ndarray,
    image_labels: np.ndarray,
    *,
    device: str,
    batch_size: int,
    quantile: float,
) -> dict[str, Any]:
    train_scores = _score_batch(model, normal, device, batch_size)
    test_scores = _score_batch(model, test_images, device, batch_size)
    pixel_threshold = float(np.quantile(train_scores.ravel(), quantile))
    train_image_scores = np.asarray([_image_score(x) for x in train_scores])
    test_image_scores = np.asarray([_image_score(x) for x in test_scores])
    image_threshold = float(np.quantile(train_image_scores, quantile))
    return {
        "thresholds": {"pixel": pixel_threshold, "image": image_threshold},
        "pixel": binary_metrics(test_masks, test_scores, pixel_threshold),
        "image": binary_metrics(image_labels, test_image_scores, image_threshold),
        "train_normal_pixel_score_quantile": quantile,
        "test_image_scores": test_image_scores.tolist(),
    }


def run_matched_benchmark(
    config: DatasetConfig,
    synthetic_manifest: str | Path,
    output_path: str | Path,
    *,
    seeds: tuple[int, ...] = (11, 22, 33),
    image_size: int = 64,
    max_train: int = 128,
    max_test: int = 512,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    width: int = 16,
    quantile: float = 0.99,
    device: str = "auto",
) -> dict[str, Any]:
    """Train matched AEs and evaluate on untouched official test images."""
    import torch

    if config.split != "train" or config.category is None:
        raise ValueError("Use one category and the official train split")
    if not seeds or len(set(seeds)) != len(seeds) or any(
        type(seed) is not int or seed < 0 for seed in seeds
    ):
        raise ValueError("seeds must be unique non-negative integers")
    if (
        image_size < 16 or image_size % 8 != 0
        or max_train < 3 or max_test < 1 or epochs < 1 or batch_size < 2
    ):
        raise ValueError(
            "image_size must be >=16 and divisible by 8; "
            "train/test limits, epochs and batch size must be valid"
        )
    if learning_rate <= 0 or not 0 < quantile < 1:
        raise ValueError("learning_rate must be positive and quantile in (0, 1)")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    training = create_dataset(config)
    train_records = [r for r in training.records if not r.is_anomaly]
    if len(train_records) < 3:
        raise ValueError("At least three normal train images are required")
    all_train_paths = {r.image_path.resolve() for r in train_records}
    if len(all_train_paths) != len(train_records):
        raise ValueError("Duplicate normal training paths detected")
    selected_records = train_records[:max_train]
    if len(selected_records) < 3:
        raise ValueError("max_train must select at least three normal train images")
    # Deterministic source-level holdout: fit images and threshold-calibration images
    # are disjoint. The official test split remains completely untouched.
    ordered = sorted(selected_records, key=lambda record: str(record.image_path))
    calibration_count = max(1, int(round(len(ordered) * 0.2)))
    calibration_records = ordered[:calibration_count]
    fit_records = ordered[calibration_count:]
    if len(fit_records) < 2:
        raise ValueError("Need at least two fit images after train calibration holdout")
    fit_paths = {r.image_path.resolve() for r in fit_records}
    fit_images: list[np.ndarray] = []
    calibration_images: list[np.ndarray] = []
    for record in fit_records:
        with Image.open(record.image_path) as image:
            fit_images.append(_rgb(image, image_size))
    for record in calibration_records:
        with Image.open(record.image_path) as image:
            calibration_images.append(_rgb(image, image_size))
    normal = np.stack(fit_images).astype(np.float32)
    calibration_normal = np.stack(calibration_images).astype(np.float32)

    held_out = create_dataset(
        DatasetConfig(
            name=config.name, root=config.root, split="test",
            category=config.category, split_file=config.split_file,
            verify_files=config.verify_files,
        )
    )
    test_paths = {r.image_path.resolve() for r in held_out.records}
    if all_train_paths & test_paths:
        raise ValueError("Exact train/test source path overlap")
    train_hashes = {_sha256(p) for p in all_train_paths}
    if any(_sha256(p) in train_hashes for p in test_paths):
        raise ValueError("Exact train/test source content overlap")

    synthetic, clean_targets, provenance = _load_manifest_pairs(
        Path(synthetic_manifest), config, all_train_paths, fit_paths, test_paths, image_size
    )
    if not synthetic:
        raise ValueError(
            "No synthetic samples derive from fit images; generate more variants "
            "or use a larger --max-train selection"
        )
    synthetic_array = np.stack(synthetic).astype(np.float32)
    target_array = np.stack(clean_targets).astype(np.float32)

    test_images: list[np.ndarray] = []
    test_masks: list[np.ndarray] = []
    image_labels: list[bool] = []
    test_manifest: list[dict[str, Any]] = []
    for index in range(min(len(held_out), max_test)):
        sample = held_out[index]
        with Image.open(sample["image_path"]) as image:
            test_images.append(_rgb(image, image_size))
        mask = sample["mask"]
        if sample["is_anomaly"] and mask is None:
            raise ValueError("Anomalous test sample has no ground-truth mask")
        if mask is None:
            binary = np.zeros((image_size, image_size), dtype=bool)
        else:
            with Image.open(sample["mask_path"]) as mask_image:
                binary = np.asarray(
                    mask_image.convert("L").resize(
                        (image_size, image_size), Image.Resampling.NEAREST
                    )
                ) > 0
        if sample["is_anomaly"] and not binary.any():
            raise ValueError("Anomalous test sample has an empty ground-truth mask")
        test_masks.append(binary)
        image_labels.append(bool(sample["is_anomaly"]))
        test_manifest.append({
            "source_path": str(Path(sample["image_path"]).resolve()),
            "label": sample["label"],
        })
    if not test_images or len(set(image_labels)) != 2:
        raise ValueError("Test subset must contain normal and anomalous samples")
    test_array = np.stack(test_images).astype(np.float32)
    mask_array = np.stack(test_masks)
    labels_array = np.asarray(image_labels, dtype=bool)

    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        arms: dict[str, Any] = {}
        for name, augmented in (("real_only", False), ("real_plus_synthetic", True)):
            model, training_info = _train_one(
                normal, synthetic_array, target_array,
                seed=seed, epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, width=width, device=device,
                augmented=augmented,
            )
            metrics = _arm_metrics(
                model, calibration_normal, test_array, mask_array, labels_array,
                device=device, batch_size=batch_size, quantile=quantile,
            )
            arms[name] = {"training": training_info, **metrics}
        seed_results.append({"seed": seed, "arms": arms})

    metric_names = ("roc_auc", "average_precision", "precision", "recall", "f1")
    summary: dict[str, Any] = {}
    for arm in ("real_only", "real_plus_synthetic"):
        summary[arm] = {}
        for level in ("image", "pixel"):
            summary[arm][level] = {}
            for metric in metric_names:
                values = [
                    run["arms"][arm][level][metric]
                    for run in seed_results
                    if run["arms"][arm][level][metric] is not None
                ]
                if not values:
                    summary[arm][level][metric] = None
                    continue
                mean = float(np.mean(values))
                sem = (
                    float(np.std(values, ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0
                )
                summary[arm][level][metric] = {
                    "mean": mean,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "normal_approx_95ci": [
                        float(mean - 1.96 * sem), float(mean + 1.96 * sem)
                    ],
                    "n_seeds": len(values),
                }

    report = {
        "schema_version": 1,
        "study": "matched_conv_autoencoder_real_only_vs_procedural_denoising",
        "dataset": config.name,
        "category": config.category,
        "device": device,
        "settings": {
            "seeds": list(seeds), "image_size": image_size,
            "normal_fit_count": len(normal),
            "normal_threshold_calibration_count": len(calibration_normal),
            "test_count": len(test_array),
            "synthetic_count": len(synthetic_array), "epochs": epochs,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "width": width, "normal_threshold_quantile": quantile,
        },
        "train_test_exact_path_and_hash_overlap": False,
        "synthetic_provenance": provenance,
        "test_samples": test_manifest,
        "seed_results": seed_results,
        "summary": summary,
        "interpretation": (
            "Same convolutional autoencoder, optimizer, update count and seed per arm. "
            "Real-only reconstructs normal inputs; augmented training reconstructs "
            "normal targets from procedural defect inputs. Thresholds use only normal "
            "train reconstruction residuals. Test labels/masks are evaluation-only. "
            "This is a small research baseline, not a state-of-the-art benchmark."
        ),
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
        "seeds": list(seeds),
        "summary": summary,
    }
