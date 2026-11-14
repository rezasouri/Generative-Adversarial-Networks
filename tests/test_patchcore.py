"""PatchCore feature-memory and split-safe evaluation smoke tests."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from forgegan.data.registry import DatasetConfig
from forgegan.models.patchcore import greedy_coreset, nearest_neighbor_scores


def _dataset(root: Path) -> DatasetConfig:
    base = root / "bottle"
    for index, value in enumerate((35, 55, 75, 95)):
        path = base / "train/good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 48), (value, value + 3, value + 7)).save(path)

    normal = base / "test/good/normal.png"
    normal.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (110, 115, 120)).save(normal)

    anomaly = base / "test/crack/anomaly.png"
    anomaly.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), (110, 115, 120))
    ImageDraw.Draw(image).rectangle((12, 12, 30, 30), fill=(240, 10, 10))
    image.save(anomaly)

    mask = base / "ground_truth/crack/anomaly_mask.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    label = Image.new("L", (48, 48), 0)
    ImageDraw.Draw(label).rectangle((12, 12, 30, 30), fill=255)
    label.save(mask)
    return DatasetConfig("mvtec", root, "train", category="bottle")


def test_greedy_coreset_is_deterministic_and_bounded() -> None:
    values = np.arange(120, dtype=np.float32).reshape(30, 4)
    first = greedy_coreset(values, max_points=7, seed=13)
    second = greedy_coreset(values, max_points=7, seed=13)
    assert first.shape == (7, 4)
    np.testing.assert_array_equal(first, second)


def test_greedy_coreset_validates_inputs() -> None:
    with pytest.raises(ValueError):
        greedy_coreset(np.empty((0, 3)))
    with pytest.raises(ValueError):
        greedy_coreset(np.ones((3, 2)), max_points=0)


def test_nearest_neighbor_scores_exact_and_chunked() -> None:
    memory = np.array([[0, 0], [3, 4]], dtype=np.float32)
    query = np.array([[0, 0], [3, 0], [6, 8]], dtype=np.float32)
    result = nearest_neighbor_scores(
        query, memory, query_batch_size=1, memory_batch_size=1
    )
    np.testing.assert_allclose(result, [0, 3, 5], atol=1e-6)


def test_nearest_neighbor_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError):
        nearest_neighbor_scores(np.ones((2, 3)), np.ones((2, 4)))


def test_patchcore_benchmark_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from forgegan.training.patchcore import run_patchcore_benchmark

    config = _dataset(tmp_path / "mvtec")
    output = tmp_path / "patchcore.json"
    result = run_patchcore_benchmark(
        config,
        output,
        image_size=64,
        max_train=4,
        max_test=2,
        memory_size=24,
        batch_size=2,
        pretrained=False,
        device="cpu",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["method"] == "patchcore_resnet18_layer2_layer3"
    assert report["pretrained"] is False
    assert report["settings"]["normal_fit_count"] == 3
    assert report["settings"]["normal_calibration_count"] == 1
    assert report["settings"]["test_count"] == 2
    assert report["settings"]["memory_bank_count"] <= 24
    assert report["provenance"]["train_test_exact_sha256_overlap"] is False
    assert report["pixel"]["n"] == 2 * 64 * 64
    assert report["image"]["n"] == 2
    assert result["memory_bank_count"] <= 24

    from forgegan.evaluation.patchcore_diagnostics import run_patchcore_diagnostics

    diagnostics_dir = tmp_path / "diagnostics"
    diagnostic_result = run_patchcore_diagnostics(
        config, output, diagnostics_dir, device="cpu", batch_size=2
    )
    assert diagnostic_result["threshold_reproduction_passed"] is True
    assert (diagnostics_dir / "index.html").is_file()
    assert (diagnostics_dir / "diagnostics.json").is_file()
    html_report = (diagnostics_dir / "index.html").read_text(encoding="utf-8")
    assert "Ground truth" in html_report
    assert "Anomaly heatmap" in html_report
    assert "False-positive" in html_report or "false-positive" in html_report


def test_patchcore_rejects_existing_output(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from forgegan.training.patchcore import run_patchcore_benchmark

    config = _dataset(tmp_path / "mvtec")
    output = tmp_path / "exists.json"
    output.write_text("{}")
    with pytest.raises(FileExistsError):
        run_patchcore_benchmark(
            config, output, image_size=64, max_train=4, max_test=2,
            memory_size=8, pretrained=False, device="cpu",
        )
