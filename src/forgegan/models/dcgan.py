"""Unconditional 64x64 DCGAN baseline for reproducible image synthesis."""

from __future__ import annotations

import torch
from torch import nn


class DCGANGenerator(nn.Module):
    """Map [B, latent_dim, 1, 1] noise to RGB [-1, 1] 64x64 images."""

    def __init__(self, latent_dim: int = 128, features: int = 64, channels: int = 3) -> None:
        super().__init__()
        if min(latent_dim, features, channels) < 1:
            raise ValueError("latent_dim, features and channels must be positive")
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.ConvTranspose2d(features, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)


class DCGANDiscriminator(nn.Module):
    """Map 64x64 images to one binary classification logit per image."""

    def __init__(self, features: int = 64, channels: int = 3) -> None:
        super().__init__()
        if min(features, channels) < 1:
            raise ValueError("features and channels must be positive")
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images).flatten(start_dim=1).squeeze(1)


def initialize_dcgan(module: nn.Module) -> None:
    """Standard DCGAN convolution and BatchNorm initialization."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
