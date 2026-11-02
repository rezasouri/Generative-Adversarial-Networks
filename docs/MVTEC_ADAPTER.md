# MVTec AD adapter guide

## Supported source layout

The adapter expects the original MVTec AD category directory layout:

\`\`\`text
datasets/mvtec_ad/
├── bottle/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   └── broken_large/
│   └── ground_truth/
│       └── broken_large/
│           └── <image_stem>_mask.png
├── cable/
├── capsule/
├── carpet/
├── grid/
├── hazelnut/
├── leather/
├── metal_nut/
├── pill/
├── screw/
├── tile/
├── toothbrush/
├── transistor/
├── wood/
└── zipper/
\`\`\`

The adapter supports all 15 official categories. It reads only the extracted dataset and never downloads, copies, renames or modifies source images.

Official source and access:
- MVTec AD dataset page, download access, category description and license: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- Bergmann et al., “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR 2019.

## Dataset semantics

- Training samples are read only from \`train/good\` and are labeled normal.
- Test samples are read from \`test/good\` and each defect-type directory.
- The directory name \`good\` means normal; every other test directory is treated as an anomaly type.
- Anomaly masks are resolved as \`ground_truth/<defect_type>/<image_stem>_mask.png\`.
- Normal samples have no mask and return \`mask=None\`.
- Original mask values are preserved by default. Use \`mask_mode="binary"\` for a binary 0/255 mask.
- Image and mask dimensions are checked when a sample is loaded.

The adapter fails on missing category/split directories, empty split directories and missing anomaly masks rather than silently omitting samples.

## CLI validation

After obtaining the dataset through the official MVTec download process and extracting it under \`datasets/mvtec_ad\`, run:

\`\`\`bash
python -m pip install -e .
forgegan validate-mvtec --root datasets/mvtec_ad
\`\`\`

To validate a single category and write a JSON report:

\`\`\`bash
forgegan validate-mvtec \\
  --root datasets/mvtec_ad \\
  --category bottle \\
  --output reports/mvtec-bottle-audit.json
\`\`\`

## Python API

\`\`\`python
from forgegan.data import MVTecDataset, load_mvtec_records, validate_mvtec_split

root = "datasets/mvtec_ad"
records = load_mvtec_records(root)
print(validate_mvtec_split(records))

train_data = MVTecDataset(root, split="train", category="bottle")
test_data = MVTecDataset(
    root,
    split="test",
    category="bottle",
    mask_mode="binary",
)

sample = test_data[0]
print(sample["category"], sample["label"], sample["defect_type"])
\`\`\`

\`MVTecDataset\` is a map-style dataset and can be passed to \`torch.utils.data.DataLoader\`. Provide transforms that convert PIL images to tensors when using PyTorch's default collation. The base adapter does not import PyTorch.

## Benchmark and validation policy

- Keep the official test split untouched.
- Do not move anomalous test images into training.
- If a validation set is needed, define and document it separately; do not silently reuse the official test set.
- Keep all derived or resized images outside the original dataset directory.
- Record category selection, split policy, preprocessing, seed and dataset version/checksum with every experiment.

## License

The MVTec AD dataset is distributed under CC BY-NC-SA 4.0 according to its official dataset page. This is separate from ForgeGAN's MIT code license and does not permit commercial use under those published terms. Do not commit the dataset, masks, archives or derived samples to this repository. If intended use may be commercial, obtain clarification or permission from MVTec before proceeding. This note is an engineering summary, not legal advice.

## Tests

CI creates a small synthetic MVTec-shaped directory tree. Tests cover directory indexing, category filtering, image/mask loading, binary masks, normal samples without masks, missing annotations, missing split directories and image/mask dimension mismatch. CI does not download the full dataset.
