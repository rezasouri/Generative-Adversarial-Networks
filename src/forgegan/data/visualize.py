"""Create deterministic, reviewable contact sheets from dataset samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from forgegan.data.registry import DatasetConfig, create_dataset


def inspect_dataset(
    config: DatasetConfig,
    output_dir: str | Path,
    *,
    limit: int = 32,
    columns: int = 4,
    tile_size: tuple[int, int] = (256, 288),
) -> dict[str, Any]:
    """Save a contact sheet and JSON manifest; source dataset remains untouched."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if columns < 1:
        raise ValueError("columns must be positive")
    width, height = tile_size
    if width < 32 or height < 64:
        raise ValueError("tile dimensions are too small")

    dataset = create_dataset(config)
    count = min(limit, len(dataset))
    if count == 0:
        raise ValueError("Dataset is empty")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = (count + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    manifest: list[dict[str, Any]] = []

    for index in range(count):
        sample = dataset[index]
        x = (index % columns) * width
        y = (index // columns) * height
        image = sample["image"].convert("RGB")
        image.thumbnail((width - 16, height - 76), Image.Resampling.LANCZOS)
        image_x = x + (width - image.width) // 2
        sheet.paste(image, (image_x, y + 8))

        label = (
            f'{sample["category"]} | {sample["label"]}'
            + (f' | {sample["defect_type"]}' if sample["defect_type"] else "")
        )
        draw.text((x + 8, y + height - 60), label[:42], fill="black", font=font)
        draw.text((x + 8, y + height - 43), f'index={index}', fill="black", font=font)

        mask = sample["mask"]
        if mask is not None:
            mask_preview = mask.convert("L").resize(
                (min(72, width // 3), min(48, height // 6)),
                Image.Resampling.NEAREST,
            )
            sheet.paste(mask_preview.convert("RGB"), (x + width - mask_preview.width - 8, y + height - 56))

        manifest.append(
            {
                "index": index,
                "dataset": sample["dataset"],
                "category": sample["category"],
                "split": sample["split"],
                "label": sample["label"],
                "defect_type": sample["defect_type"],
                "image_path": sample["image_path"],
                "mask_path": sample["mask_path"],
            }
        )

    image_path = output / "contact-sheet.png"
    manifest_path = output / "manifest.json"
    sheet.save(image_path, format="PNG", optimize=True)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": config.name,
                "split": config.split,
                "category": config.category,
                "sample_count": count,
                "image": image_path.name,
                "samples": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "sample_count": count,
        "image": str(image_path),
        "manifest": str(manifest_path),
    }
