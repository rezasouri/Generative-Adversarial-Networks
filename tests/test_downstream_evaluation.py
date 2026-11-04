"""Split-safe synthetic threshold calibration and anomaly metric smoke tests."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from forgegan.data.registry import DatasetConfig
from forgegan.evaluation.baseline import (
    EvaluationSettings,
    compare_real_vs_synthetic,
    save_comparison,
)
from forgegan.evaluation.metrics import binary_metrics
from forgegan.synthesis.procedural import DefectSpec, generate_synthetic_dataset


def _write_rgb(path: Path, value: int, *, anomaly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), (value, value + 5, value + 10))
    if anomaly:
        draw = ImageDraw.Draw(image)
        draw.rectangle((15, 13, 28, 29), fill=(220, 25, 20))
    image.save(path)


def _make_data(tmp_path: Path) -> tuple[DatasetConfig, Path]:
    root = tmp_path / "mvtec"
    base = root / "bottle"
    for i, value in enumerate((45, 55, 65, 75)):
        _write_rgb(base / "train/good" / f"{i:03d}.png", value)
    _write_rgb(base / "test/good/normal.png", 90)
    _write_rgb(base / "test/damaged/defect.png", 85, anomaly=True)
    mask_path = base / "ground_truth/damaged/defect_mask.png"
    mask_path.parent.mkdir(parents=True)
    mask = Image.new("L", (48, 48), 0)
    ImageDraw.Draw(mask).rectangle((15, 13, 28, 29), fill=255)
    mask.save(mask_path)
    config = DatasetConfig("mvtec", root, "train", category="bottle")
    manifest = generate_synthetic_dataset(
        config,
        tmp_path / "synthetic",
        spec=DefectSpec(kind="spot", scale=0.2, center_x=0.5, center_y=0.5),
        count=4,
        seed=3,
    )["manifest"]
    return config, Path(manifest)


def test_binary_metrics_exact_ties_and_f1() -> None:
    result = binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.4, 0.8]),
        threshold=0.4,
    )
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["f1"] == pytest.approx(0.8)
    assert result["roc_auc"] == pytest.approx(0.875)
    assert result["average_precision"] == pytest.approx((1 + 2 / 3) / 2)


def test_binary_metrics_undefined_on_one_class() -> None:
    scores = binary_metrics([0, 0], [0.1, 0.7], threshold=0.5)
    assert scores["roc_auc"] is None
    assert scores["average_precision"] is None


@pytest.mark.parametrize("labels,scores", [([0, 2], [1, 2]), ([0], [float("nan")])])
def test_invalid_metrics_rejected(labels: list, scores: list) -> None:
    with pytest.raises(ValueError):
        binary_metrics(labels, scores, threshold=0.5)


def test_downstream_report_and_no_overwrite(tmp_path: Path) -> None:
    config, manifest = _make_data(tmp_path)
    report = compare_real_vs_synthetic(
        config,
        manifest,
        settings=EvaluationSettings(
            image_size=32, max_train=4, max_test=2, max_synthetic=4
        ),
    )
    assert report["test_normal_count"] == 1
    assert report["test_anomaly_count"] == 1
    assert report["synthetic"]["source_sha256_verified"]
    assert set(report["arms"]) == {"real_only", "synthetic_calibrated"}
    assert report["arms"]["real_only"]["image"]["n"] == 2
    assert report["arms"]["real_only"]["pixel"]["n"] == 2 * 32 * 32
    assert (
        report["arms"]["real_only"]["pixel"]["roc_auc"]
        == report["arms"]["synthetic_calibrated"]["pixel"]["roc_auc"]
    )
    output = tmp_path / "reports" / "comparison.json"
    assert save_comparison(report, output) == output
    assert json.loads(output.read_text())["schema_version"] == 1
    with pytest.raises(FileExistsError):
        save_comparison(report, output)


def test_rejects_test_config(tmp_path: Path) -> None:
    config, manifest = _make_data(tmp_path)
    with pytest.raises(ValueError, match="official train split"):
        compare_real_vs_synthetic(
            DatasetConfig("mvtec", config.root, "test", "bottle"), manifest
        )


def test_rejects_tampered_synthetic_source_hash(tmp_path: Path) -> None:
    config, manifest = _make_data(tmp_path)
    data = json.loads(manifest.read_text())
    data["items"][0]["source_sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        compare_real_vs_synthetic(config, manifest)


def test_rejects_manifest_test_source(tmp_path: Path) -> None:
    config, manifest = _make_data(tmp_path)
    data = json.loads(manifest.read_text())
    data["items"][0]["source_split"] = "test"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="split, category or label"):
        compare_real_vs_synthetic(config, manifest)


def test_rejects_train_test_exact_content_overlap(tmp_path: Path) -> None:
    config, manifest = _make_data(tmp_path)
    train = config.root / "bottle/train/good/000.png"
    test = config.root / "bottle/test/good/normal.png"
    test.write_bytes(train.read_bytes())
    with pytest.raises(ValueError, match="Exact source content overlaps"):
        compare_real_vs_synthetic(config, manifest)


def test_rejects_output_path_traversal_in_manifest(tmp_path: Path) -> None:
    config, manifest = _make_data(tmp_path)
    data = json.loads(manifest.read_text())
    data["items"][0]["image"] = "../../evil.png"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="within the manifest"):
        compare_real_vs_synthetic(config, manifest)


def test_invalid_settings_rejected() -> None:
    with pytest.raises(ValueError, match="at least 16"):
        EvaluationSettings(image_size=8)
    with pytest.raises(ValueError, match="positive integers"):
        EvaluationSettings(max_train=0)
