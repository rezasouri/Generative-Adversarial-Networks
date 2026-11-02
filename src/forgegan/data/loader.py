"""Optional PyTorch DataLoader and safe collation for anomaly datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from forgegan.data.registry import DatasetConfig, create_dataset
from forgegan.data.transforms import SpatialTransform


def collate_anomaly_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate mixed normal/anomaly samples, including missing normal masks.

    Returns images as float32 BCHW in [0,1], binary masks as float32 B1HW,
    and has_mask to distinguish absent annotations from zero-valued masks.
    """
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise ImportError(
            "PyTorch and NumPy are required for batching. Install forgegan[train]."
        ) from error

    if not samples:
        raise ValueError("Cannot collate an empty batch")

    images = []
    masks = []
    has_mask = []
    for sample in samples:
        image = sample["image"]
        if not isinstance(image, Image.Image):
            raise TypeError("DataLoader expects PIL images; do not pre-convert to tensors")
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        image_tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)
        images.append(image_tensor)
        height, width = array.shape[:2]

        raw_mask = sample["mask"]
        if raw_mask is None:
            mask_tensor = torch.zeros((1, height, width), dtype=torch.float32)
            has_mask.append(False)
        else:
            if not isinstance(raw_mask, Image.Image):
                raise TypeError("DataLoader expects PIL masks; do not pre-convert to tensors")
            if raw_mask.size != (width, height):
                raise ValueError(
                    f"Image/mask sizes differ for {sample['image_path']}: "
                    f"{(width, height)} vs {raw_mask.size}"
                )
            mask_array = np.array(raw_mask.convert("L"), dtype=np.uint8, copy=True)
            mask_tensor = torch.from_numpy(mask_array > 0).unsqueeze(0).float()
            has_mask.append(True)
        masks.append(mask_tensor)

    try:
        batched_images = torch.stack(images)
        batched_masks = torch.stack(masks)
    except RuntimeError as error:
        raise ValueError(
            "Images have inconsistent sizes. Use SpatialTransform or batch_size=1."
        ) from error

    return {
        "image": batched_images,
        "mask": batched_masks,
        "has_mask": torch.tensor(has_mask, dtype=torch.bool),
        "is_anomaly": torch.tensor(
            [sample["is_anomaly"] for sample in samples], dtype=torch.bool
        ),
        "dataset": [sample["dataset"] for sample in samples],
        "category": [sample["category"] for sample in samples],
        "split": [sample["split"] for sample in samples],
        "label": [sample["label"] for sample in samples],
        "defect_type": [sample["defect_type"] for sample in samples],
        "image_path": [sample["image_path"] for sample in samples],
        "mask_path": [sample["mask_path"] for sample in samples],
    }


def create_dataloader(
    config: DatasetConfig,
    *,
    batch_size: int = 16,
    image_size: tuple[int, int] = (256, 256),
    seed: int = 42,
    num_workers: int = 0,
    shuffle: bool | None = None,
) -> Any:
    """Return a reproducibly seeded DataLoader with normalized sample schema.

    image_size is (height, width). Default shuffle applies only to train.
    Workers and shuffling must not change source split assignments.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise ImportError("Install forgegan[train] to use a PyTorch DataLoader") from error

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    transform = SpatialTransform(height=image_size[0], width=image_size[1])
    dataset = create_dataset(
        config, image_transform=transform.image, mask_transform=transform.mask
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(config.split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        generator=generator,
        collate_fn=collate_anomaly_samples,
    )
