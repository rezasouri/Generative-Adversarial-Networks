"""Class-conditional 64x64 GAN based on DCGAN, conditioned on object category."""

from __future__ import annotations

import torch
from torch import nn

from forgegan.models.dcgan import DCGANDiscriminator, DCGANGenerator


def _check_labels(labels: torch.Tensor, batch_size: int, classes: int) -> None:
    if labels.dtype != torch.long or labels.ndim != 1 or labels.shape[0] != batch_size:
        raise ValueError("labels must be a 1-D torch.long tensor matching batch size")
    if torch.any(labels < 0) or torch.any(labels >= classes):
        raise ValueError("labels contain an out-of-range category index")


class ConditionalGenerator(nn.Module):
    """Condition noise by concatenating an embedded category vector."""

    def __init__(
        self,
        num_classes: int,
        latent_dim: int = 128,
        features: int = 64,
        channels: int = 3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("Conditional GAN needs at least two categories")
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(num_classes, latent_dim)
        self.generator = DCGANGenerator(latent_dim * 2, features, channels)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 4 or noise.shape[1:] != (self.latent_dim, 1, 1):
            raise ValueError("noise must be [batch, latent_dim, 1, 1]")
        _check_labels(labels, noise.shape[0], self.num_classes)
        condition = self.embedding(labels).unsqueeze(-1).unsqueeze(-1)
        return self.generator(torch.cat((noise, condition), dim=1))


class ConditionalDiscriminator(nn.Module):
    """Evaluate an image paired with a tiled category embedding map."""

    def __init__(
        self,
        num_classes: int,
        features: int = 64,
        channels: int = 3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("Conditional GAN needs at least two categories")
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, 64 * 64)
        self.discriminator = DCGANDiscriminator(features, channels + 1)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[2:] != (64, 64):
            raise ValueError("images must be [batch, channels, 64, 64]")
        _check_labels(labels, images.shape[0], self.num_classes)
        condition = self.embedding(labels).view(-1, 1, 64, 64)
        return self.discriminator(torch.cat((images, condition), dim=1))
