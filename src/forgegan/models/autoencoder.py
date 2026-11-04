"""Small convolutional denoising autoencoder for reproducible anomaly baselines."""

from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """Compact RGB autoencoder; accepts and returns BCHW tensors in [0, 1]."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        if type(width) is not int or width < 4:
            raise ValueError("width must be an integer >= 4")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.GELU(),
            nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.ConvTranspose2d(width, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images))
