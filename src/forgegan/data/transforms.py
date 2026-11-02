"""Deterministic spatial preprocessing shared by image and anomaly mask.

Transform input/output is PIL Image. Image uses bilinear interpolation;
masks must use nearest-neighbor interpolation to preserve discrete labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True, slots=True)
class SpatialTransform:
    """Resize image and mask to the same (height, width), without randomness."""

    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def image(self, value: Image.Image) -> Image.Image:
        return value.convert("RGB").resize(self.size, resample=Image.Resampling.BILINEAR)

    def mask(self, value: Image.Image) -> Image.Image:
        return value.convert("L").resize(self.size, resample=Image.Resampling.NEAREST)
