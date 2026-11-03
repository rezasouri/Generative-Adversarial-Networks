"""Deterministic procedural defect-mask and pixel-level synthesis baseline.

This is a handcrafted augmentation baseline, not a learned GAN or proof of
realistic defects. No official anomaly/test images are read.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

from forgegan.data.registry import DatasetConfig, create_dataset

DefectKind = Literal["spot", "scratch"]


@dataclass(frozen=True, slots=True)
class DefectSpec:
    """Normalized spatial controls and local intensity blending strength."""

    kind: DefectKind = "spot"
    center_x: float = 0.5
    center_y: float = 0.5
    scale: float = 0.12
    opacity: float = 0.75
    angle_degrees: float = 30.0

    def __post_init__(self) -> None:
        if self.kind not in {"spot", "scratch"}:
            raise ValueError("kind must be 'spot' or 'scratch'")
        for field in ("center_x", "center_y", "scale", "opacity", "angle_degrees"):
            value = getattr(self, field)
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if not 0 <= self.center_x <= 1 or not 0 <= self.center_y <= 1:
            raise ValueError("center_x and center_y must be between 0 and 1")
        if not 0 < self.scale <= 0.5:
            raise ValueError("scale must be in (0, 0.5]")
        if not 0 < self.opacity <= 1:
            raise ValueError("opacity must be in (0, 1]")


def make_mask(size: tuple[int, int], spec: DefectSpec, seed: int) -> Image.Image:
    """Render a binary PIL mask; shape is seeded and spatial controls are explicit."""
    width, height = size
    if min(width, height) < 16:
        raise ValueError("Images must be at least 16x16 for procedural defects")
    rng = np.random.default_rng(seed)
    cx = round(spec.center_x * (width - 1))
    cy = round(spec.center_y * (height - 1))
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    radius = max(2, round(min(width, height) * spec.scale))
    if spec.kind == "spot":
        vertices = []
        for i in range(16):
            theta = (2 * math.pi * i) / 16
            distance = radius * float(rng.uniform(0.72, 1.12))
            vertices.append(
                (
                    round(cx + distance * math.cos(theta)),
                    round(cy + distance * math.sin(theta)),
                )
            )
        draw.polygon(vertices, fill=255)
    else:
        angle = math.radians(spec.angle_degrees)
        half_length = radius * 2
        x1 = round(cx - math.cos(angle) * half_length)
        y1 = round(cy - math.sin(angle) * half_length)
        x2 = round(cx + math.cos(angle) * half_length)
        y2 = round(cy + math.sin(angle) * half_length)
        draw.line((x1, y1, x2, y2), fill=255, width=max(1, radius // 3))
    if not mask.getbbox():
        raise ValueError("Defect mask does not intersect the image")
    return mask


def synthesize_defect(
    image: Image.Image, spec: DefectSpec, *, seed: int
) -> tuple[Image.Image, Image.Image]:
    """Perturb only masked pixels; unmasked pixels remain exactly unchanged."""
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = make_mask(image.size, spec, seed)
    region = np.asarray(mask, dtype=np.uint8) > 0
    rng = np.random.default_rng(seed + 1)
    direction = -1 if seed % 2 else 1
    delta = (55 + rng.integers(0, 60, size=region.shape)) * direction
    pixels = original.astype(np.float32)
    # Apply a high-contrast, textured local color shift rather than global corruption.
    shifted = np.clip(
        pixels + delta[..., None].astype(np.float32) * spec.opacity, 0, 255
    ).astype(np.uint8)
    result = original.copy()
    result[region] = shifted[region]
    return Image.fromarray(result, mode="RGB"), mask


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_synthetic_dataset(
    config: DatasetConfig,
    output_dir: str | Path,
    *,
    spec: DefectSpec,
    count: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate synthetic image+mask pairs with training-source provenance.

    Refuses to touch held-out data and refuses to write inside source root.
    Existing nonempty output directories are rejected to prevent accidental
    overwrite and stale/mixed experiments.
    """
    if config.split != "train":
        raise ValueError("Synthetic defects may only derive from official train samples")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    source_root = Path(config.root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination == source_root or destination.is_relative_to(source_root):
        raise ValueError("output_dir must be outside the original dataset root")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")

    dataset = create_dataset(config)
    if not dataset.records:
        raise ValueError("No normal training records found")
    if any(record.split != "train" or record.is_anomaly for record in dataset.records):
        raise ValueError("Synthetic defects require normal-only training source records")

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "images").mkdir()
    (destination / "masks").mkdir()
    items: list[dict[str, Any]] = []
    for index in range(count):
        record_index = index % len(dataset)
        record = dataset.records[record_index]
        original = dataset[record_index]["image"]
        local_seed = seed + index
        generated, mask = synthesize_defect(original, spec, seed=local_seed)
        item_id = f"synthetic-{index:06d}"
        image_file = destination / "images" / f"{item_id}.png"
        mask_file = destination / "masks" / f"{item_id}.png"
        generated.save(image_file)
        mask.save(mask_file)
        items.append(
            {
                "id": item_id,
                "image": str(image_file.relative_to(destination)),
                "mask": str(mask_file.relative_to(destination)),
                "source_path": str(record.image_path),
                "source_sha256": _digest_file(record.image_path),
                "source_split": record.split,
                "source_category": record.category,
                "synthetic_label": "procedural_anomaly",
                "seed": local_seed,
            }
        )

    manifest = {
        "schema_version": 1,
        "method": "procedural_masked_intensity_baseline",
        "not_learned": True,
        "dataset": config.name,
        "source_split": "train",
        "spec": asdict(spec),
        "count": count,
        "items": items,
    }
    manifest_file = destination / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "count": count,
        "output_dir": str(destination),
        "manifest": str(manifest_file),
        "method": manifest["method"],
    }
