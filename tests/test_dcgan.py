"""DCGAN model and single-epoch training/resume smoke tests."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image  # noqa: E402

from forgegan.experiments.config import load_experiment_config  # noqa: E402
from forgegan.models.dcgan import DCGANDiscriminator, DCGANGenerator  # noqa: E402
from forgegan.training.dcgan import train_dcgan  # noqa: E402


def _fixture(tmp_path: Path, *, epochs: int = 1) -> Path:
    root = tmp_path / "mvtec"
    for index in range(2):
        path = root / "bottle" / "train" / "good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (index * 40 + 50, 80, 120)).save(path)

    base = root / "bottle"
    (base / "test" / "good").mkdir(parents=True)
    Image.new("RGB", (64, 64), (10, 20, 30)).save(base / "test/good/000.png")
    (base / "ground_truth").mkdir()

    yaml_path = tmp_path / "smoke.yaml"
    yaml_path.write_text(
        "version: 1\n"
        "name: dcgan-smoke\n"
        "seed: 7\n"
        "dataset:\n"
        "  name: mvtec\n"
        f"  root: {root}\n"
        "  split: train\n"
        "  category: bottle\n"
        "model:\n"
        "  name: dcgan\n"
        "  latent_dim: 8\n"
        "  features: 4\n"
        "  image_channels: 3\n"
        "training:\n"
        f"  epochs: {epochs}\n"
        "  batch_size: 2\n"
        "  learning_rate: 0.0002\n"
        "  max_batches_per_epoch: 1\n"
        f"output_dir: {tmp_path / 'runs' / 'dcgan'}\n",
        encoding="utf-8",
    )
    return yaml_path


def test_dcgan_forward_shapes() -> None:
    generator = DCGANGenerator(latent_dim=8, features=4)
    discriminator = DCGANDiscriminator(features=4)
    with torch.no_grad():
        fake = generator(torch.randn(2, 8, 1, 1))
        logits = discriminator(fake)
    assert fake.shape == (2, 3, 64, 64)
    assert logits.shape == (2,)
    assert fake.min().item() >= -1.0
    assert fake.max().item() <= 1.0


def test_dcgan_train_checkpoint_and_resume(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    config_path = _fixture(tmp_path)
    first = train_dcgan(load_experiment_config(config_path), device="cpu")
    checkpoint = Path(first["checkpoint"])
    assert checkpoint.is_file()
    assert (tmp_path / "runs/dcgan/samples/epoch-0001.png").is_file()
    state = torch.load(checkpoint, weights_only=True)
    assert state["epoch"] == 1
    assert state["global_steps"] == 1

    _fixture(tmp_path, epochs=2)
    second = train_dcgan(
        load_experiment_config(config_path), resume=checkpoint, device="cpu"
    )
    assert second["completed_epochs"] == 2
    assert second["global_steps"] == 2
    assert (tmp_path / "runs/dcgan/samples/epoch-0002.png").is_file()
    assert len((tmp_path / "runs/dcgan/losses.jsonl").read_text().splitlines()) == 2


def test_dcgan_rejects_test_split(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    contents = config_path.read_text().replace("  split: train", "  split: test")
    config_path.write_text(contents)
    with pytest.raises(ValueError, match="official train split"):
        train_dcgan(load_experiment_config(config_path), device="cpu")
