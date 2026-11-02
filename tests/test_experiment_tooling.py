from pathlib import Path

import pytest
from PIL import Image

from forgegan.data.registry import DatasetConfig
from forgegan.data.visualize import inspect_dataset
from forgegan.experiments.config import load_experiment_config


def _visa_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "visa"
    for relative, color, mode in [
        ("candle/Data/Images/Normal/normal.png", (20, 30, 40), "RGB"),
        ("candle/Data/Images/Anomaly/defect.png", (80, 90, 100), "RGB"),
        ("candle/Data/Masks/Anomaly/defect.png", 255, "L"),
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, (16, 12), color=color).save(path)
    split_file = root / "split_csv/1cls.csv"
    split_file.parent.mkdir(parents=True)
    split_file.write_text(
        "object,split,label,image,mask\n"
        "candle,train,normal,candle/Data/Images/Normal/normal.png,\n"
        "candle,test,anomaly,candle/Data/Images/Anomaly/defect.png,"
        "candle/Data/Masks/Anomaly/defect.png\n",
        encoding="utf-8",
    )
    return root, split_file


def test_inspector_writes_contact_sheet_and_manifest(tmp_path: Path) -> None:
    root, split_file = _visa_root(tmp_path)
    report = inspect_dataset(
        DatasetConfig("visa", root, "test", "candle", split_file),
        tmp_path / "inspection",
        limit=4,
        columns=2,
    )
    assert report["sample_count"] == 1
    assert Path(report["image"]).is_file()
    assert Path(report["manifest"]).is_file()
    manifest = Path(report["manifest"]).read_text(encoding="utf-8")
    assert '"label": "anomaly"' in manifest


def test_inspector_rejects_invalid_limit(tmp_path: Path) -> None:
    root, split_file = _visa_root(tmp_path)
    with pytest.raises(ValueError, match="limit must be positive"):
        inspect_dataset(
            DatasetConfig("visa", root, "test", "candle", split_file),
            tmp_path / "inspection",
            limit=0,
        )


def test_loads_valid_experiment_yaml(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "version: 1\n"
        "name: baseline-test\n"
        "seed: 42\n"
        "dataset:\n"
        "  name: mvtec\n"
        "  root: datasets/mvtec_ad\n"
        "  split: train\n"
        "model:\n"
        "  name: dcgan\n"
        "training:\n"
        "  epochs: 10\n"
        "  batch_size: 16\n"
        "  learning_rate: 0.0002\n"
        "output_dir: runs/baseline-test\n",
        encoding="utf-8",
    )
    loaded = load_experiment_config(config)
    assert loaded.name == "baseline-test"
    assert loaded.seed == 42
    assert loaded.output_dir == Path("runs/baseline-test")


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("seed: 42", "seed: -1", "non-negative"),
        ("epochs: 10", "epochs: 0", "positive integer"),
        ("batch_size: 16", "batch_size: 0", "positive integer"),
        ("learning_rate: 0.0002", "learning_rate: 0", "positive"),
        ("name: dcgan", "name: ''", "model.name"),
    ],
)
def test_rejects_invalid_experiment_values(
    tmp_path: Path, field: str, replacement: str, message: str
) -> None:
    content = (
        "name: invalid\nseed: 42\n"
        "dataset:\n  name: mvtec\n  root: data\n  split: train\n"
        "model:\n  name: dcgan\n"
        "training:\n  epochs: 10\n  batch_size: 16\n  learning_rate: 0.0002\n"
        "output_dir: runs/test\n"
    )
    if field == "name: dcgan":
        content = content.replace(field, replacement)
    else:
        content = content.replace(field, replacement)
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)


def test_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    path = tmp_path / "unknown.yaml"
    path.write_text(
        "name: x\nseed: 1\ndataset: {}\nmodel: {}\ntraining: {}\n"
        "output_dir: runs/x\nsurprise: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown experiment keys"):
        load_experiment_config(path)
