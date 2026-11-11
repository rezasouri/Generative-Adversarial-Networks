"""PyTorch smoke tests for matched denoising anomaly experiments."""

import json
from pathlib import Path

import pytest

from PIL import Image, ImageDraw

from forgegan.data.registry import DatasetConfig
from forgegan.synthesis.procedural import DefectSpec, generate_synthetic_dataset
from forgegan.training.anomaly import run_matched_benchmark


def _fixture(root: Path) -> tuple[DatasetConfig, Path]:
    base = root / "bottle"
    for index, value in enumerate((40, 55, 70, 85)):
        path = base / "train/good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (value, value + 5, value + 10)).save(path)
    normal = base / "test/good/normal.png"
    normal.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (90, 95, 100)).save(normal)
    anomaly = base / "test/crack/anomaly.png"
    anomaly.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), (90, 95, 100))
    ImageDraw.Draw(image).rectangle((10, 10, 20, 20), fill=(230, 20, 20))
    image.save(anomaly)
    mask = base / "ground_truth/crack/anomaly_mask.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    label = Image.new("L", (32, 32), 0)
    ImageDraw.Draw(label).rectangle((10, 10, 20, 20), fill=255)
    label.save(mask)
    config = DatasetConfig("mvtec", root, "train", category="bottle")
    manifest = generate_synthetic_dataset(
        config,
        root.parent / "synthetic",
        spec=DefectSpec(kind="spot", scale=0.2),
        count=8,
        seed=17,
    )["manifest"]
    return config, Path(manifest)


def test_autoencoder_shape_and_parameter_count() -> None:
    torch = pytest.importorskip("torch")
    from forgegan.models.autoencoder import ConvAutoencoder

    model = ConvAutoencoder(width=4)
    result = model(torch.rand(2, 3, 32, 32))
    assert result.shape == (2, 3, 32, 32)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_matched_benchmark_smoke_and_report(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config, manifest = _fixture(tmp_path / "mvtec")
    output = tmp_path / "report.json"
    result = run_matched_benchmark(
        config,
        manifest,
        output,
        seeds=(7,),
        image_size=32,
        max_train=4,
        max_test=2,
        epochs=1,
        batch_size=2,
        width=4,
        device="cpu",
    )
    report = json.loads(output.read_text())
    assert result["seeds"] == [7]
    assert report["settings"]["normal_fit_count"] == 3
    assert report["settings"]["normal_threshold_calibration_count"] == 1
    assert report["settings"]["test_count"] == 2
    assert report["synthetic_provenance"]["source_sha256_verified"]
    assert report["seed_results"][0]["arms"]["real_only"]["training"]["optimizer_steps"] == 2
    assert report["seed_results"][0]["arms"]["real_plus_synthetic"]["training"]["optimizer_steps"] == 2
    assert report["summary"]["real_only"]["pixel"]["n_seeds"] == 1
    assert report["summary"]["real_plus_synthetic"]["image"]["n_seeds"] == 1


def test_rejects_invalid_image_size(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config, manifest = _fixture(tmp_path / "mvtec")
    with pytest.raises(ValueError, match="divisible by 8"):
        run_matched_benchmark(
            config, manifest, tmp_path / "report.json",
            image_size=30, max_train=4, epochs=1, device="cpu",
        )


def test_report_does_not_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config, manifest = _fixture(tmp_path / "mvtec")
    output = tmp_path / "report.json"
    output.write_text("{}")
    with pytest.raises(FileExistsError):
        run_matched_benchmark(
            config, manifest, output, seeds=(1,), image_size=32,
            max_train=4, max_test=2, epochs=1, batch_size=2,
            width=4, device="cpu",
        )
