from pathlib import Path

import pytest
from PIL import Image

from forgegan.data.mvtec import (
    MVTecDataset,
    load_mvtec_records,
    validate_mvtec_split,
)


def _write_image(path: Path, color: tuple[int, int, int] = (20, 30, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color=color).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (16, 12), color=0)
    mask.putpixel((2, 3), 127)
    mask.save(path)


def _make_mvtec_root(tmp_path: Path) -> Path:
    root = tmp_path / "mvtec_ad"
    category = root / "bottle"
    _write_image(category / "train/good/train-01.png")
    _write_image(category / "test/good/test-good.png")
    _write_image(category / "test/broken_large/test-defect.png", (90, 80, 70))
    _write_mask(category / "ground_truth/broken_large/test-defect_mask.png")
    return root


def test_loads_official_directory_layout_and_counts(tmp_path: Path) -> None:
    root = _make_mvtec_root(tmp_path)

    records = load_mvtec_records(root, category="bottle")
    report = validate_mvtec_split(records)

    assert len(records) == 3
    assert report["protocol"] == "official_directory_split"
    assert report["totals"] == {
        "test_anomaly": 1,
        "test_normal": 1,
        "train_normal": 1,
    }
    assert report["anomaly_types"] == ["broken_large"]
    assert report["source_image_leakage"] is False


def test_dataset_loads_rgb_image_and_binary_mask(tmp_path: Path) -> None:
    root = _make_mvtec_root(tmp_path)
    dataset = MVTecDataset(root, split="test", category="bottle", mask_mode="binary")

    assert len(dataset) == 2
    anomaly_index = next(
        index for index in range(len(dataset)) if dataset[index]["is_anomaly"]
    )
    sample = dataset[anomaly_index]
    assert sample["image"].mode == "RGB"
    assert sample["mask"].mode == "L"
    assert set(sample["mask"].getdata()) == {0, 255}
    assert sample["defect_type"] == "broken_large"


def test_normal_samples_have_no_mask(tmp_path: Path) -> None:
    root = _make_mvtec_root(tmp_path)
    dataset = MVTecDataset(root, split="test", category="bottle")
    normal = next(
        dataset[index]
        for index in range(len(dataset))
        if not dataset[index]["is_anomaly"]
    )

    assert normal["label"] == "normal"
    assert normal["mask"] is None
    assert normal["mask_path"] is None


def test_rejects_unknown_category(tmp_path: Path) -> None:
    root = _make_mvtec_root(tmp_path)
    with pytest.raises(ValueError, match="Unknown MVTec category"):
        load_mvtec_records(root, category="unknown")


def test_rejects_missing_anomaly_mask(tmp_path: Path) -> None:
    root = _make_mvtec_root(tmp_path)
    (root / "bottle/ground_truth/broken_large/test-defect_mask.png").unlink()

    with pytest.raises(FileNotFoundError, match="Missing MVTec ground-truth mask"):
        load_mvtec_records(root, category="bottle")


def test_rejects_missing_required_split_directory(tmp_path: Path) -> None:
    root = _make_mvtec_root(tmp_path)
    (root / "bottle/train/good").rename(root / "bottle/train/good-moved")

    with pytest.raises(FileNotFoundError, match="training directory is missing"):
        load_mvtec_records(root, category="bottle")


def test_rejects_image_mask_dimension_mismatch(tmp_path: Path) -> None:
    root = _make_mvtec_root(tmp_path)
    mask_path = root / "bottle/ground_truth/broken_large/test-defect_mask.png"
    Image.new("L", (8, 8), color=255).save(mask_path)
    dataset = MVTecDataset(root, split="test", category="bottle")
    anomaly_index = next(
        index for index in range(len(dataset)) if dataset[index]["is_anomaly"]
    )

    with pytest.raises(ValueError, match="dimensions differ"):
        dataset[anomaly_index]
