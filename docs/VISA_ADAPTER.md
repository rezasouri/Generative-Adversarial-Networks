# VisA adapter guide

## Supported source layout

The adapter reads the original VisA directory structure published by Amazon Science:

```text
datasets/visa/
├── candle/
│   ├── Data/
│   │   ├── Images/
│   │   │   ├── Normal/
│   │   │   └── Anomaly/
│   │   └── Masks/
│   │       └── Anomaly/
│   └── ...
├── capsules/
├── ... (12 categories total)
└── split_csv/
    └── 1cls.csv
```

The original dataset archive contains the category folders. The official split CSV is maintained in the upstream SPot-the-Difference repository under split_csv/1cls.csv; obtain it from that repository and place it at the path above, or pass its location explicitly.

Official references:
- Dataset description, raw folder structure, split preparation and citation: https://github.com/amazon-science/spot-diff
- Official 1-class split CSV: https://github.com/amazon-science/spot-diff/blob/main/split_csv/1cls.csv
- Dataset archive and license entry: https://registry.opendata.aws/visa/

## What the adapter guarantees

- Parses the official CSV columns: object, split, label, image and mask.
- Supports all 12 official categories and the official train/test split.
- Enforces normal-only training rows for the official 1-class protocol.
- Requires an annotation mask for every anomalous record and rejects masks on normal records.
- Rejects path traversal and paths that resolve outside the dataset root.
- Detects an image path repeated in the split manifest, preventing source-image leakage.
- Optionally checks that all referenced image and mask files exist.
- Does not copy, rename, download or modify source data.

The loader deliberately fails on malformed or unexpected metadata instead of silently dropping bad rows.

## CLI validation

Install the base package and run:

```bash
python -m pip install -e .
forgegan validate-visa \
  --root datasets/visa \
  --split-file datasets/visa/split_csv/1cls.csv \
  --output reports/visa-split-audit.json
```

To inspect CSV structure and split integrity before the full archive is available:

```bash
forgegan validate-visa \
  --root datasets/visa \
  --split-file datasets/visa/split_csv/1cls.csv \
  --skip-file-check
```

A category-specific check can be run with --category capsules.

## Python API

```python
from forgegan.data import VisaDataset, load_visa_records, validate_visa_split

root = "datasets/visa"
records = load_visa_records(root)
print(validate_visa_split(records))

train_data = VisaDataset(root, split="train", category="capsules")
test_data = VisaDataset(
    root,
    split="test",
    category="capsules",
    mask_mode="binary",
)
sample = test_data[0]

print(sample["image"].size)
print(sample["label"], sample["is_anomaly"])
```

VisaDataset is a map-style dataset and can be passed to torch.utils.data.DataLoader. Provide image_transform and mask_transform that convert PIL images to tensors when using PyTorch's default collation. The base adapter itself does not import PyTorch, keeping dataset inspection available without the optional training extra.

## Mask handling

By default, anomaly masks are returned in their original image mode so class/color IDs are not destroyed. Set mask_mode="binary" when the downstream task needs a binary foreground mask; all nonzero grayscale values are mapped to 255. Normal images have mask=None because the official dataset does not store normal masks.

## Split protocol and leakage policy

The adapter follows the official 1-class train/test split. It does not carve validation images out of test data or invent a random split. A validation protocol will be introduced separately and documented before model selection begins.

For a strict benchmark:
- keep the official test set untouched;
- if a validation set is needed, derive it only from the official normal training pool with a fixed seed and record the selected source paths;
- never place synthetic variants of one source image across different splits;
- record the exact split CSV and its checksum with each experiment.

## Tests

The unit tests create a tiny temporary VisA-shaped fixture. They cover metadata parsing, category filtering, image/mask loading, binary mask conversion, path traversal, missing masks, anomalous training rows and duplicate source paths across splits. CI does not download the full dataset.
