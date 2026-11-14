"""Visual and threshold diagnostics for a saved PatchCore benchmark."""

from __future__ import annotations

import base64
import html
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from forgegan.data.registry import DatasetConfig, create_dataset
from forgegan.models.patchcore import ResNetFeatureExtractor, greedy_coreset
from forgegan.training.patchcore import (
    _extract_embeddings,
    _image_tensor,
    _score_images,
)
from forgegan.evaluation.metrics import binary_metrics


def _png_data(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _heatmap(values: np.ndarray) -> Image.Image:
    """Render a score map with a robust per-image color scale."""
    scores = np.asarray(values, dtype=np.float32)
    upper = float(np.quantile(scores, 0.995))
    normalized = np.clip(scores / max(upper, 1e-8), 0, 1)
    # Blue -> cyan -> yellow -> red, implemented without plotting dependencies.
    stops = np.asarray(
        [[25, 45, 180], [0, 210, 220], [255, 230, 40], [210, 20, 30]],
        dtype=np.float32,
    )
    position = normalized * (len(stops) - 1)
    left = np.floor(position).astype(np.int64)
    right = np.minimum(left + 1, len(stops) - 1)
    weight = (position - left)[..., None]
    rgb = stops[left] * (1 - weight) + stops[right] * weight
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _overlay(image: Image.Image, heatmap: Image.Image, alpha: float = 0.48) -> Image.Image:
    return Image.blend(image.convert("RGB"), heatmap.convert("RGB"), alpha)


def _mask_image(mask: np.ndarray) -> Image.Image:
    values = np.asarray(mask, dtype=bool)
    canvas = np.zeros((*values.shape, 3), dtype=np.uint8)
    canvas[values] = (255, 255, 255)
    return Image.fromarray(canvas, mode="RGB")


def _threshold_table(
    labels: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    calibration_maps: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for quantile in (0.95, 0.975, 0.99, 0.995, 0.999):
        threshold = float(np.quantile(calibration_maps, quantile))
        metrics = binary_metrics(masks, scores, threshold)
        rows.append(
            {
                "calibration_quantile": quantile,
                "threshold": threshold,
                "test_metrics_descriptive_only": metrics,
            }
        )
    return rows


def run_patchcore_diagnostics(
    config: DatasetConfig,
    benchmark_report: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Recreate the saved PatchCore run and render test masks, heatmaps and FP ranking.

    Threshold candidates are derived exclusively from normal calibration images.
    Test labels are used only to describe each candidate's held-out metrics.
    """
    import torch

    report_path = Path(benchmark_report)
    if not report_path.is_file():
        raise FileNotFoundError(f"Benchmark report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("method") != "patchcore_resnet18_layer2_layer3":
        raise ValueError("Unsupported benchmark report method")
    if report.get("dataset") != config.name or report.get("category") != config.category:
        raise ValueError("Dataset/category do not match the benchmark report")
    if config.split != "train":
        raise ValueError("Diagnostics require the official train dataset config")
    settings = report["settings"]
    image_size = int(settings["image_size"])
    max_train = int(settings["normal_fit_count"]) + int(
        settings["normal_calibration_count"]
    )
    max_test = int(settings["test_count"])
    memory_size = int(settings["memory_bank_count"])
    seed = int(settings["seed"])
    quantile = float(settings["threshold_quantile"])
    batch_size = int(batch_size or settings["batch_size"])
    pretrained = bool(report["pretrained"])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_dataset = create_dataset(config)
    normal_records = [record for record in train_dataset.records if not record.is_anomaly]
    selected = sorted(normal_records, key=lambda record: str(record.image_path))[:max_train]
    calibration_count = max(1, int(round(len(selected) * 0.2)))
    calibration_records = selected[:calibration_count]
    fit_records = selected[calibration_count:]
    if len(fit_records) != int(settings["normal_fit_count"]):
        raise ValueError("Current train split does not reproduce the benchmark fit count")

    test_config = DatasetConfig(
        name=config.name,
        root=config.root,
        split="test",
        category=config.category,
        split_file=config.split_file,
        verify_files=config.verify_files,
    )
    test_dataset = create_dataset(test_config)
    test_records = test_dataset.records[:max_test]
    expected_paths = [item["source_path"] for item in report["test_samples"]]
    actual_paths = [str(record.image_path.resolve()) for record in test_records]
    if actual_paths != expected_paths:
        raise ValueError(
            "Current test sample ordering differs from the saved benchmark; "
            "refusing to generate misleading diagnostics"
        )

    extractor = ResNetFeatureExtractor(pretrained=pretrained, device=device)
    fit_images = _image_tensor([r.image_path for r in fit_records], image_size)
    calibration_images = _image_tensor(
        [r.image_path for r in calibration_records], image_size
    )
    test_images = _image_tensor([r.image_path for r in test_records], image_size)
    fit_embeddings, _ = _extract_embeddings(
        extractor, fit_images, device=device, batch_size=batch_size
    )
    memory = greedy_coreset(fit_embeddings, max_points=memory_size, seed=seed)
    calibration_maps, _ = _score_images(
        extractor, calibration_images, memory, device=device, batch_size=batch_size
    )
    test_maps, image_scores = _score_images(
        extractor, test_images, memory, device=device, batch_size=batch_size
    )

    stored_threshold = float(report["thresholds"]["pixel"])
    reproduced_threshold = float(np.quantile(calibration_maps, quantile))
    if not np.isclose(stored_threshold, reproduced_threshold, rtol=2e-4, atol=2e-5):
        raise ValueError(
            "Recomputed calibration threshold differs from saved report; "
            "check torchvision weights, package versions and benchmark settings"
        )

    masks = []
    labels = []
    for record in test_records:
        labels.append(bool(record.is_anomaly))
        if record.mask_path is None:
            mask = np.zeros((image_size, image_size), dtype=bool)
        else:
            with Image.open(record.mask_path) as source:
                mask = np.asarray(
                    source.convert("L").resize(
                        (image_size, image_size), Image.Resampling.NEAREST
                    )
                ) > 0
        masks.append(mask)
    mask_array = np.stack(masks)
    label_array = np.asarray(labels, dtype=bool)
    predicted = test_maps >= stored_threshold

    threshold_rows = _threshold_table(
        label_array, mask_array, test_maps, calibration_maps
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)

    cards = []
    per_sample = []
    for index, record in enumerate(test_records):
        truth = mask_array[index]
        prediction = predicted[index]
        tp = int(np.count_nonzero(truth & prediction))
        fp = int(np.count_nonzero(~truth & prediction))
        fn = int(np.count_nonzero(truth & ~prediction))
        tn = int(np.count_nonzero(~truth & ~prediction))
        fp_rate = fp / max(int(np.count_nonzero(~truth)), 1)
        relative = str(record.image_path.relative_to(config.root))
        with Image.open(record.image_path) as source:
            original = source.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
        heat = _heatmap(test_maps[index])
        overlay = _overlay(original, heat)
        truth_image = _mask_image(truth)
        predicted_image = _mask_image(prediction)
        encoded = {
            "original": _png_data(original),
            "ground_truth": _png_data(truth_image),
            "heatmap": _png_data(heat),
            "overlay": _png_data(overlay),
            "prediction": _png_data(predicted_image),
        }
        cards.append(
            {
                "path": relative,
                "label": "anomaly" if label_array[index] else "normal",
                "image_score": float(image_scores[index]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "false_positive_rate": fp_rate,
                "images": encoded,
            }
        )
        per_sample.append({key: value for key, value in cards[-1].items() if key != "images"})

    cards.sort(key=lambda item: (item["fp"], item["image_score"]), reverse=True)
    summary = {
        "schema_version": 1,
        "method": report["method"],
        "dataset": config.name,
        "category": config.category,
        "benchmark_report": str(report_path),
        "diagnostics": {
            "test_count": len(test_records),
            "image_size": image_size,
            "device": device,
            "memory_bank_count": len(memory),
            "saved_pixel_threshold": stored_threshold,
            "recomputed_pixel_threshold": reproduced_threshold,
            "threshold_reproduction_passed": True,
            "threshold_candidates": threshold_rows,
            "threshold_policy": (
                "Candidate thresholds come from normal-only train calibration "
                "maps; test metrics are descriptive and are not used to select a threshold."
            ),
            "per_sample_ranked_by_false_positive_pixels": per_sample,
        },
    }
    summary_path = destination / "diagnostics.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    table_rows = []
    for item in cards:
        title = html.escape(item["path"])
        images = "".join(
            f'<figure><img src="{item["images"][key]}" alt="{label}">'
            f"<figcaption>{label}</figcaption></figure>"
            for key, label in (
                ("original", "Original"),
                ("ground_truth", "Ground truth"),
                ("heatmap", "Anomaly heatmap"),
                ("overlay", "Overlay"),
                ("prediction", "Thresholded prediction"),
            )
        )
        table_rows.append(
            f'<article><h2>{title}</h2><p>Label: {item["label"]} · '
            f'Image score: {item["image_score"]:.4f} · FP pixels: {item["fp"]:,} · '
            f'TP: {item["tp"]:,} · FN: {item["fn"]:,} · '
            f'FP rate: {item["false_positive_rate"]:.3%}</p><div class="images">{images}</div></article>'
        )

    threshold_html = "".join(
        "<tr>"
        f"<td>{row['calibration_quantile']:.3%}</td>"
        f"<td>{row['threshold']:.6f}</td>"
        f"<td>{row['test_metrics_descriptive_only']['precision']:.4f}</td>"
        f"<td>{row['test_metrics_descriptive_only']['recall']:.4f}</td>"
        f"<td>{row['test_metrics_descriptive_only']['f1']:.4f}</td>"
        "</tr>"
        for row in threshold_rows
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ForgeGAN PatchCore diagnostics — {html.escape(config.category or "")}</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;margin:0;background:#10141c;color:#e8edf5}}
main{{max-width:1400px;margin:auto;padding:28px}}
article{{background:#1b2330;border:1px solid #354154;border-radius:12px;padding:18px;margin:22px 0}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
figure{{margin:0}}img{{width:100%;height:auto;border-radius:7px;background:#080b10}}
figcaption{{color:#aab8cc;font-size:13px;margin-top:5px}}
table{{border-collapse:collapse;width:100%;margin:18px 0;background:#1b2330}}
td,th{{padding:10px;border:1px solid #354154;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.note{{background:#202d3d;padding:14px;border-radius:8px}}
</style></head><body><main>
<h1>PatchCore visual diagnostics: {html.escape(config.category or "")}</h1>
<p>Recomputed from the saved benchmark settings and exact test manifest. No model weights were trained.</p>
<div class="note"><b>Threshold policy:</b> every candidate threshold is computed from normal-only train calibration maps.
Test metrics below are descriptive only; test labels are never used to select a threshold.</div>
<h2>Calibration-derived threshold analysis</h2>
<table><thead><tr><th>Calibration quantile</th><th>Threshold</th><th>Test precision</th><th>Test recall</th><th>Test F1</th></tr></thead>
<tbody>{threshold_html}</tbody></table>
<h2>Test samples ranked by false-positive pixel count</h2>
{''.join(table_rows)}
</main></body></html>"""
    html_path = destination / "index.html"
    html_path.write_text(document, encoding="utf-8")
    return {
        "output_dir": str(destination),
        "html_report": str(html_path),
        "json_report": str(summary_path),
        "test_count": len(test_records),
        "threshold_reproduction_passed": True,
        "worst_false_positive_samples": per_sample[:5],
    }
