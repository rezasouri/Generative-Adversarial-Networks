"""Contract tests for the unified industrial dataset pipeline."""

from pathlib import Path

import pytest
from PIL import Image

from forgegan.data.registry import DatasetConfig, create_dataset, validate_source_splits
from forgegan.data.transforms import SpatialTransform


def _image(path: Path, value: int = 20, *, mask: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L" if mask else "RGB", (12, 10), color=value).save(path)


def _mvtec(tmp_path: Path) -> Path:
    root = tmp_path / "mvtec"
    base = root / "bottle"
    _image(base / "train/good/001.png")
    _image(base / "test/good/002.png", 50)
    _image(base / "test/broken/003.png", 70)
    _image(base / "ground_truth/broken/003_mask.png", 255, mask=True)
    return root


def _visa(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "visa"
    _image(root / "candle/Data/Images/Normal/001.png")
    _image(root / "candle/Data/Images/Normal/002.png")
    _image(root / "candle/Data/Images/Anomaly/003.png")
    _image(root / "candle/Data/Masks/Anomaly/003.png", 255, mask=True)
    csv_path = root / "1cls.csv"
    csv_path.write_text(
        "object,split,label,image,mask\n"
        "candle,train,normal,candle/Data/Images/Normal/001.png,\n"
        "candle,test,normal,candle/Data/Images/Normal/002.png,\n"
        "candle,test,anomaly,candle/Data/Images/Anomaly/003.png,"
        "candle/Data/Masks/Anomaly/003.png\n",
        encoding="utf-8",
    )
    return root, csv_path


@pytest.mark.parametrize("name", ["visa", "mvtec"])
def test_factory_normalizes_sample_schema(tmp_path: Path, name: str) -> None:
    if name == "visa":
        root, csv_path = _visa(tmp_path)
    else:
        root, csv_path = _mvtec(tmp_path), None

    config = DatasetConfig(
        name=name, root=root, split="test", category="candle" if name == "visa" else "bottle",
        split_file=csv_path,
    )
    resize = SpatialTransform(32, 24)
    dataset = create_dataset(
        config, image_transform=resize.image, mask_transform=resize.mask
    )
    assert len(dataset) == 2
    assert all(dataset[i]["image"].size == (24, 32) for i in range(len(dataset)))
    anomaly = next(dataset[i] for i in range(len(dataset)) if dataset[i]["is_anomaly"])
    assert anomaly["mask"].size == (24, 32)
    assert set(anomaly) == {
        "image", "mask", "dataset", "category", "split", "label", "is_anomaly",
        "defect_type", "image_path", "mask_path",
    }
    assert anomaly["dataset"] == name


def test_registry_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        DatasetConfig("unknown", tmp_path, "train")


def test_split_validation_without_leakage(tmp_path: Path) -> None:
    root, csv_path = _visa(tmp_path)
    configs = [
        DatasetConfig("visa", root, "train", category="candle", split_file=csv_path),
        DatasetConfig("visa", root, "test", category="candle", split_file=csv_path),
    ]
    report = validate_source_splits(configs, check_files=True)
    assert report["source_images_checked"] == 3
    assert report["source_image_path_leakage"] is False


def test_transform_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        SpatialTransform(0, 32)


def test_nearest_neighbor_preserves_binary_mask_values() -> None:
    mask = Image.new("L", (2, 2), 0)
    mask.putpixel((0, 0), 255)
    transformed = SpatialTransform(16, 16).mask(mask)
    assert set(transformed.get_flattened_data()) == {0, 255}


def test_dataloader_optional_torch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from forgegan.data.loader import create_dataloader

    root = _mvtec(tmp_path)
    config = DatasetConfig("mvtec", root, "test", category="bottle")
    loader = create_dataloader(config, batch_size=2, image_size=(16, 24), shuffle=False)
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 16, 24)
    assert batch["mask"].shape == (2, 1, 16, 24)
    assert batch["has_mask"].dtype == torch.bool
    assert batch["has_mask"].tolist() == [True, False] or (
        batch["has_mask"].tolist() == [False, True]
    )
    assert batch["is_anomaly"].sum().item() == 1
