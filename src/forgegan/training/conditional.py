"""Category-conditional GAN training with fixed class labels and resumable epochs."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.utils import save_image

from forgegan.data import DatasetConfig, create_dataloader
from forgegan.experiments.config import ExperimentConfig
from forgegan.models.conditional import ConditionalDiscriminator, ConditionalGenerator
from forgegan.models.dcgan import initialize_dcgan


def _train_settings(config: ExperimentConfig) -> dict[str, Any]:
    model = config.model
    train = config.training
    if model.get("name") != "cgan":
        raise ValueError("Only model.name='cgan' is supported by this trainer")
    latent_dim = model.get("latent_dim", 128)
    features = model.get("features", 64)
    channels = model.get("image_channels", 3)
    if any(type(value) is not int or value <= 0 for value in (latent_dim, features)):
        raise ValueError("model.latent_dim and model.features must be positive integers")
    if type(channels) is not int or channels != 3:
        raise ValueError("Conditional GAN requires image_channels=3 for RGB dataset loading")
    if config.dataset["split"] != "train":
        raise ValueError("Conditional GAN must use the official train split")
    max_batches = train.get("max_batches_per_epoch")
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError("training.max_batches_per_epoch must be a positive integer")
    beta1 = train.get("beta1", 0.5)
    beta2 = train.get("beta2", 0.999)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1
        for value in (beta1, beta2)
    ):
        raise ValueError("Adam beta1/beta2 must be in [0, 1)")
    return {
        "latent_dim": latent_dim,
        "features": features,
        "channels": channels,
        "batch_size": train["batch_size"],
        "learning_rate": train["learning_rate"],
        "beta1": beta1,
        "beta2": beta2,
        "max_batches_per_epoch": max_batches,
    }


def _identity(config: ExperimentConfig, settings: dict[str, Any]) -> dict[str, Any]:
    dataset = dict(config.dataset)
    dataset["root"] = str(Path(dataset["root"]).resolve())
    if dataset.get("split_file"):
        dataset["split_file"] = str(Path(dataset["split_file"]).resolve())
    return {
        "name": config.name,
        "seed": config.seed,
        "dataset": dataset,
        "settings": settings,
        "architecture": "category-cgan-64-v1",
    }


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _save_samples(
    generator: ConditionalGenerator,
    fixed_noise: torch.Tensor,
    fixed_labels: torch.Tensor,
    path: Path,
) -> None:
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        images = generator(fixed_noise, fixed_labels).detach().cpu().add(1).div(2).clamp(0, 1)
    save_image(images, path, nrow=4)
    generator.train(was_training)


def train_cgan(
    config: ExperimentConfig,
    *,
    resume: str | Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Train a category-conditioned GAN on normal train images only.

    Checkpoints are produced at completed epoch boundaries only. Resume restores
    model, optimizer, RNG and sampler state, so interrupted runs may resume at
    the next epoch. The caller must pass the exact same data and config settings.
    """
    settings = _train_settings(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_config = DatasetConfig(
        name=config.dataset["name"],
        root=Path(config.dataset["root"]),
        split=config.dataset["split"],
        category=config.dataset.get("category"),
        split_file=(
            Path(config.dataset["split_file"]) if config.dataset.get("split_file") else None
        ),
        verify_files=config.dataset.get("verify_files", True),
    )
    loader = create_dataloader(
        dataset_config,
        batch_size=settings["batch_size"],
        image_size=(64, 64),
        seed=config.seed,
        num_workers=0,
    )
    if not loader.dataset.records:
        raise ValueError("Training dataset is empty")
    if any(record.is_anomaly for record in loader.dataset.records):
        raise ValueError("Conditional GAN trains on normal-only official training images")

    categories = sorted({record.category for record in loader.dataset.records})
    if len(categories) < 2:
        raise ValueError(
            "Category conditioning needs at least two training categories; "
            "set dataset.category to null and include two or more categories"
        )
    label_to_id = {category: index for index, category in enumerate(categories)}
    identity = _identity(config, settings)
    identity["categories"] = categories

    generator = ConditionalGenerator(
        len(categories), settings["latent_dim"], settings["features"], settings["channels"]
    ).to(selected_device)
    discriminator = ConditionalDiscriminator(
        len(categories), settings["features"], settings["channels"]
    ).to(selected_device)
    generator.apply(initialize_dcgan)
    discriminator.apply(initialize_dcgan)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=settings["learning_rate"],
        betas=(settings["beta1"], settings["beta2"]),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=settings["learning_rate"],
        betas=(settings["beta1"], settings["beta2"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(16, settings["latent_dim"], 1, 1, device=selected_device)
    fixed_labels = torch.arange(16, device=selected_device) % len(categories)
    start_epoch = 0
    global_steps = 0

    if resume is not None:
        # weights_only protects against loading arbitrary Python objects.
        # Only resume checkpoints from sources you trust.
        checkpoint = torch.load(resume, map_location=selected_device, weights_only=True)
        if checkpoint.get("identity") != identity:
            raise ValueError("Checkpoint identity/config does not match this experiment")
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        fixed_noise = checkpoint["fixed_noise"].to(selected_device)
        fixed_labels = checkpoint["fixed_labels"].to(selected_device)
        start_epoch = int(checkpoint["epoch"])
        global_steps = int(checkpoint["global_steps"])
        torch.set_rng_state(checkpoint["rng_state"].cpu())
        if selected_device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        loader.generator.set_state(checkpoint["sampler_rng_state"].cpu())

    if start_epoch >= config.training["epochs"]:
        raise ValueError("Checkpoint already reached the configured total epochs")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    metadata = asdict(config)
    metadata["output_dir"] = str(config.output_dir)
    (output_dir / "experiment.json").write_text(
        json.dumps({"experiment": metadata, "identity": identity}, indent=2) + "\n",
        encoding="utf-8",
    )

    log_path = output_dir / "losses.jsonl"
    if resume is None and log_path.exists():
        raise FileExistsError(
            f"Training log already exists: {log_path}. Use --resume or a new output_dir."
        )

    for epoch in range(start_epoch, config.training["epochs"]):
        generator.train()
        discriminator.train()
        d_total = 0.0
        g_total = 0.0
        steps = 0
        for batch in loader:
            real = batch["image"].to(selected_device).mul(2).sub(1)
            batch_size = real.size(0)
            if not batch_size:
                continue
            labels = torch.tensor(
                [label_to_id[category] for category in batch["category"]],
                dtype=torch.long,
                device=selected_device,
            )
            noise = torch.randn(
                batch_size, settings["latent_dim"], 1, 1, device=selected_device
            )
            fake = generator(noise, labels)

            optimizer_d.zero_grad(set_to_none=True)
            loss_d = criterion(
                discriminator(real, labels), torch.ones(batch_size, device=selected_device)
            ) + criterion(
                discriminator(fake.detach(), labels),
                torch.zeros(batch_size, device=selected_device),
            )
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            for param in discriminator.parameters():
                param.requires_grad_(False)
            loss_g = criterion(
                discriminator(fake, labels), torch.ones(batch_size, device=selected_device)
            )
            loss_g.backward()
            optimizer_g.step()
            for param in discriminator.parameters():
                param.requires_grad_(True)

            d_total += loss_d.item()
            g_total += loss_g.item()
            global_steps += 1
            steps += 1
            if settings["max_batches_per_epoch"] is not None:
                if steps >= settings["max_batches_per_epoch"]:
                    break

        if steps == 0:
            raise RuntimeError("No training batches were produced")

        result = {
            "epoch": epoch + 1,
            "global_steps": global_steps,
            "loss_d": d_total / steps,
            "loss_g": g_total / steps,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
        _save_samples(
            generator, fixed_noise, fixed_labels,
            output_dir / "samples" / f"epoch-{epoch + 1:04d}.png"
        )
        _save_checkpoint(
            output_dir / "checkpoint-last.pt",
            {
                "format_version": 1,
                "identity": identity,
                "epoch": epoch + 1,
                "global_steps": global_steps,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "fixed_noise": fixed_noise.detach().cpu(),
                "fixed_labels": fixed_labels.detach().cpu(),
                "categories": categories,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if selected_device.type == "cuda" else None
                ),
                "sampler_rng_state": loader.generator.get_state(),
            },
        )
        print(json.dumps(result), flush=True)

    return {
        "output_dir": str(output_dir),
        "completed_epochs": config.training["epochs"],
        "global_steps": global_steps,
        "device": str(selected_device),
        "categories": categories,
        "checkpoint": str(output_dir / "checkpoint-last.pt"),
    }
