# Unified industrial data pipeline

The same interface exposes VisA and MVTec AD samples without modifying the original datasets.

## Install

```bash
python -m pip install -e ".[dev]"
# Only required for PyTorch batching:
python -m pip install -e ".[train]"
```

## Registry and normalized schema

```python
from pathlib import Path
from forgegan.data import DatasetConfig, SpatialTransform, create_dataset

config = DatasetConfig(
    name="mvtec", root=Path("datasets/mvtec_ad"), split="train", category="bottle"
)
resize = SpatialTransform(height=256, width=256)
dataset = create_dataset(
    config, image_transform=resize.image, mask_transform=resize.mask
)
sample = dataset[0]
print(sample["dataset"], sample["category"], sample["label"])
```

For VisA, use `name="visa"`, `root=Path("datasets/visa")`, and pass `split_file=Path("datasets/visa/split_csv/1cls.csv")` when the CSV is not at its default location. The factory normalizes the following fields: `image`, `mask`, `dataset`, `category`, `split`, `label`, `is_anomaly`, `defect_type`, `image_path` and `mask_path`. VisA has `defect_type=None` because its manifest does not supply that field.

## Deterministic spatial preprocessing

`SpatialTransform(height, width)` uses bilinear interpolation for RGB images and nearest-neighbor interpolation for masks. These operations do not introduce randomness and preserve binary mask values. Image and mask spatial sizes agree after preprocessing.

## PyTorch DataLoader

```python
from forgegan.data import create_dataloader

loader = create_dataloader(
    config, batch_size=8, image_size=(256, 256), seed=42, num_workers=0
)
batch = next(iter(loader))
print(batch["image"].shape)  # [B, 3, 256, 256], float32, [0, 1]
print(batch["mask"].shape)   # [B, 1, 256, 256], float32, binary
```

The factory imports torch lazily and is available only with the optional `train` extra. It uses a seeded torch generator. Train shuffles by default; test does not. The batch collator replaces absent normal masks with zeros and supplies a boolean `has_mask` to distinguish missing annotations from all-zero masks. It also includes `is_anomaly`, source paths, labels, categories and dataset identifiers.

## Split isolation

```python
from forgegan.data import validate_source_splits

report = validate_source_splits([
    DatasetConfig("mvtec", config.root, "train", category="bottle"),
    DatasetConfig("mvtec", config.root, "test", category="bottle"),
], check_files=True)
print(report)
```

This validator detects *exact source paths* appearing in different split assignments. It cannot detect near-duplicate images, derived samples with new names, or leakage through shared specimens that are stored as different files. For generated data, retain source identifiers and enforce source-group isolation separately before benchmarking.

## Scope and limitations

- The official datasets supply train and test only. No validation split is automatically created.
- MVTec AD and VisA retain their respective data licenses, independent of this repository's MIT license.
- Raw image/mask inputs remain PIL objects in the lightweight factory; the DataLoader converts them to tensors.
- The current registry is explicit and small; it does not automatically discover or install third-party dataset plugins.
- CI uses synthetic miniature directory fixtures. A complete on-disk integration benchmark with the official archives is still pending.
