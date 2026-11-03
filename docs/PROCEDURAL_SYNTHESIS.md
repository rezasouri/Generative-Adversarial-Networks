# Procedural defect synthesis baseline

This milestone implements a deterministic **handcrafted augmentation baseline**, not a GAN or a learned mask-conditioned generator. It creates paired synthetic anomaly images and binary localization masks from official **normal training images** without reading held-out anomaly/test images.

## Quick start

```bash
python -m pip install -e ".[dev]"
forgegan synthesize-defects \
  --dataset mvtec \
  --root datasets/mvtec_ad \
  --category bottle \
  --output-dir runs/procedural-bottle \
  --kind spot --count 32 --seed 42 \
  --center-x 0.5 --center-y 0.5 --scale 0.12 --opacity 0.75
```

For VisA, replace `--dataset mvtec` with `--dataset visa`, set `--root datasets/visa`, provide `--split-file datasets/visa/split_csv/1cls.csv`, and select a VisA category.

Defect styles:
- `spot`: seeded irregular filled patch around the requested center
- `scratch`: oriented thin line around the requested center; adjust `--angle`

`center-x` and `center-y` are normalized coordinates (0–1). `scale` controls the size relative to the shorter image dimension (0–0.5); `opacity` controls intensity (0–1).

## Outputs and provenance

The command writes into a new, empty output directory:

```text
runs/procedural-bottle/
  images/synthetic-000000.png
  masks/synthetic-000000.png
  ...
  manifest.json
```

Every generated image has a binary 0/255 mask, a source image path, source SHA-256, training split, source category and seed in the manifest. Outside the mask the output pixel values are identical to the source. Original dataset images are never rewritten. To avoid dataset contamination and accidental overwrite, output inside the input dataset root or into a nonempty directory is rejected.

The algorithm is intentionally basic: it shifts intensities within a procedurally drawn binary region, with seeded local texture. It is suitable as a reproducible **toy augmentation / code-path baseline**, not evidence that synthetic industrial defects are physically realistic. Do not label outputs as verified real defects.

## Train/test integrity

Only `dataset.split == "train"` is supported. The generator verifies that the source records are normal and rejects anomalous source labels. It never incorporates official test images or test masks, and generated artifacts must remain in separate folders. For downstream benchmarks, generated samples inherit their source identity: synthetic variants from a training image must never appear in held-out evaluation splits.

The manifest records paths and exact source hashes. This is a provenance mechanism, not an automatic visual near-duplicate detector. The full industrial datasets are not redistributed by this repository. Follow the corresponding VisA or MVTec license, including restrictions on derived data.

## Next steps

The next milestone can integrate the generated image/mask pairs into a controlled real-only vs augmented detector experiment. A *learned* mask-conditioned defect generator is a distinct subsequent project and will require a separate training-eligible defect dataset or a carefully documented self-supervised learning protocol. No measured quality or downstream improvement is claimed here.
