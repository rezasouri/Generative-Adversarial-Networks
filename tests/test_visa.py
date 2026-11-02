import csv
from pathlib import Path

import pytest
from PIL import Image

from forgegan.data.visa import VisaDataset, load_visa_records, validate_visa_split


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color=color).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (16, 12), color=0)
    mask.putpixel((2, 3), 127)
    mask.save(path)


def _make_visa_root(tmp_path: Path) -> Path:
    root = tmp_path / "visa"
    _write_rgb(root / "candle/Data/Images/Normal/train-01.JPG", (20, 30, 40))
    _write_rgb(root / "candle/Data/Images/Normal/test-01.JPG", (50, 60, 70))
    _write_rgb(root / "candle/Data/Images/Anomaly/defect-01.JPG", (80, 90, 100))
    _write_mask(root / "candle/Data/Masks/Anomaly/defect-01.png")

    split_dir = root / "split_csv"
    split_dir.mkdir(parents=True)
    rows = [
        ["candle", "train", "normal", "candle/Data/Images/Normal/train-01.JPG", ""],
        ["candle", "test", "normal", "candle/Data/Images/Normal/test-01.JPG", ""],
        [
            "candle",
            "test",
            "anomaly",
            "candle/Data/Images/Anomaly/defect-01.JPG",
            "candle/Data/Masks/Anomaly/defect-01.png",
        ],
    ]
    with (split_dir / "1cls.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["object", "split", "label", "image", "mask"])
        writer.writerows(rows)
    return root


def test_loads_official_split_and_validates_counts(tmp_path: Path) -> None:
    root = _make_visa_root(tmp_path)

    records = load_visa_records(root)
    report = validate_visa_split(records)

    assert len(records) == 3
    assert report["protocol"] == "official_1cls"
    assert report["totals"] == {
        "test_anomaly": 1,
        "test_normal": 1,
        "train_normal": 1,
    }
    assert report["source_image_leakage"] is False


def test_category_filter_and_dataset_sample_loading(tmp_path: Path) -> None:
    root = _make_visa_root(tmp_path)
    dataset = VisaDataset(root, split="test", category="candle", mask_mode="binary")

    assert len(dataset) == 2
    anomaly = next(dataset[index] for index in range(len(dataset)) if dataset[index]["is_anomaly"])
    assert anomaly["image"].mode == "RGB"
    assert anomaly["mask"].mode == "L"
    assert set(anomaly["mask"].getdata()) == {0, 255}
    assert anomaly["label"] == "anomaly"


def test_rejects_path_traversal(tmp_path: Path) -> None:
    root = _make_visa_root(tmp_path)
    split_file = root / "split_csv/unsafe.csv"
    split_file.write_text(
        "object,split,label,image,mask\n"
        "candle,train,normal,../outside.png,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsafe image path"):
        load_visa_records(root, split_file, verify_files=False)


def test_rejects_anomalies_in_official_training_split(tmp_path: Path) -> None:
    root = _make_visa_root(tmp_path)
    split_file = root / "split_csv/invalid.csv"
    split_file.write_text(
        "object,split,label,image,mask\n"
        "candle,train,anomaly,candle/Data/Images/Anomaly/defect-01.JPG,"
        "candle/Data/Masks/Anomaly/defect-01.png\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training rows must be normal"):
        load_visa_records(root, split_file, verify_files=False)


def test_rejects_duplicate_image_across_splits(tmp_path: Path) -> None:
    root = _make_visa_root(tmp_path)
    split_file = root / "split_csv/leak.csv"
    split_file.write_text(
        "object,split,label,image,mask\n"
        "candle,train,normal,candle/Data/Images/Normal/train-01.JPG,\n"
        "candle,test,normal,candle/Data/Images/Normal/train-01.JPG,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="appears more than once"):
        load_visa_records(root, split_file, verify_files=False)


def test_rejects_missing_anomaly_mask(tmp_path: Path) -> None:
    root = _make_visa_root(tmp_path)
    split_file = root / "split_csv/missing-mask.csv"
    split_file.write_text(
        "object,split,label,image,mask\n"
        "candle,test,anomaly,candle/Data/Images/Anomaly/defect-01.JPG,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no mask path"):
        load_visa_records(root, split_file, verify_files=False)
