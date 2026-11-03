"""Category-conditioning contract and training/resume smoke tests."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image  # noqa: E402

from forgegan.experiments.config import load_experiment_config  # noqa: E402
from forgegan.models.conditional import (  # noqa: E402
    ConditionalDiscriminator,
    ConditionalGenerator,
)
from forgegan.training.conditional import train_cgan  # noqa: E402


def _fixture(tmp_path: Path, *, epochs: int = 1, one_category: bool = False) -> Path:
    root = tmp_path / "mvtec"
    categories = ["bottle"] if one_category else ["bottle", "cable"]
    for category in categories:
        base = root / category
        for index in range(2):
            path = base / "train/good" / f"{index:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), (index * 40 + 30, 70, 100)).save(path)
        (base / "test/good").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (10, 20, 30)).save(base / "test/good/000.png")
        (base / "ground_truth").mkdir(exist_ok=True)

    config_path = tmp_path / "conditional.yaml"
    config_path.write_text(
        "version: 1\nname: cgan-smoke\nseed: 7\n"
        "dataset:\n  name: mvtec\n"
        f"  root: {root}\n"
        "  split: train\n"
        f"  category: {'bottle' if one_category else 'null'}\n"
        "model:\n  name: cgan\n  latent_dim: 8\n  features: 4\n"
        "  image_channels: 3\n"
        "training:\n"
        f"  epochs: {epochs}\n"
        "  batch_size: 2\n  learning_rate: 0.0002\n"
        "  max_batches_per_epoch: 1\n"
        f"output_dir: {tmp_path / 'runs' / 'conditional'}\n",
        encoding="utf-8",
    )
    return config_path


def test_conditional_model_forward_and_label_validation() -> None:
    generator = ConditionalGenerator(2, latent_dim=8, features=4)
    discriminator = ConditionalDiscriminator(2, features=4)
    labels = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        images = generator(torch.randn(2, 8, 1, 1), labels)
        logits = discriminator(images, labels)
    assert images.shape == (2, 3, 64, 64)
    assert logits.shape == (2,)

    with pytest.raises(ValueError, match="out-of-range"):
        generator(torch.randn(2, 8, 1, 1), torch.tensor([0, 2]))
    with pytest.raises(ValueError, match="torch.long"):
        discriminator(images, torch.tensor([0.0, 1.0]))


def test_conditional_training_and_resume(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    config_path = _fixture(tmp_path)
    first = train_cgan(load_experiment_config(config_path), device="cpu")
    checkpoint = Path(first["checkpoint"])
    assert first["categories"] == ["bottle", "cable"]
    assert checkpoint.is_file()
    state = torch.load(checkpoint, weights_only=True)
    assert state["epoch"] == 1
    assert state["global_steps"] == 1
    assert state["categories"] == ["bottle", "cable"]
    assert (tmp_path / "runs/conditional/samples/epoch-0001.png").is_file()

    _fixture(tmp_path, epochs=2)
    resumed = train_cgan(load_experiment_config(config_path), resume=checkpoint, device="cpu")
    assert resumed["global_steps"] == 2
    assert (tmp_path / "runs/conditional/samples/epoch-0002.png").is_file()
    assert len((tmp_path / "runs/conditional/losses.jsonl").read_text().splitlines()) == 2


def test_rejects_single_category_training(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path, one_category=True)
    with pytest.raises(ValueError, match="at least two"):
        train_cgan(load_experiment_config(config_path), device="cpu")


def test_rejects_test_split_training(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    config_path.write_text(
        config_path.read_text().replace("  split: train", "  split: test"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official train split"):
        train_cgan(load_experiment_config(config_path), device="cpu")
