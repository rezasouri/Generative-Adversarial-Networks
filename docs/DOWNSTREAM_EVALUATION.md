# Exploratory downstream evaluation: normal-only vs synthetic threshold calibration

This initial comparison deliberately uses the **same normal-only anomaly-score model in both arms**. The only difference is threshold calibration from synthetic **training-derived** defects. This **is not** a comparison of two separately trained detectors and does **not** demonstrate that adding synthetic images improves learned anomaly-detection representations.

## Quick start

First obtain and extract your dataset through its official distributor. Create synthetic images **only** from normal training images:

```bash
forgegan synthesize-defects \
  --dataset mvtec --root datasets/mvtec_ad \
  --category bottle --output-dir runs/procedural-bottle \
  --kind spot --count 32 --seed 42
```

Then evaluate on the official untouched test split:

```bash
forgegan evaluate-synthetic \
  --dataset mvtec --root datasets/mvtec_ad \
  --category bottle \
  --synthetic-manifest runs/procedural-bottle/manifest.json \
  --output reports/bottle-threshold-comparison.json
```

For VisA, use `--dataset visa`, its root, an official category and the matching `--split-file` if the one-class CSV is not in the default location.

## Method and split boundaries

1. Read *normal train* images only to fit a spatial per-pixel RGB Gaussian prototype. Standardized RGB absolute residuals yield a 64×64 anomaly score map.
2. **Real-only calibration:** take the 99th percentile of normal-train pixel scores as pixel threshold, separately calibrating image threshold on the top-1%-pixel image score.
3. **Synthetic-assisted calibration:** use normal-train residuals plus synthetic defect images and masks whose source paths and SHA-256 match approved normal-train images; choose an F1-maximizing threshold on this **training-only** calibration data.
4. **Only after both thresholds are selected**, compute test image scores and pixel maps with ground truth from the official test split, reporting image-level and pixel-level ROC-AUC, average precision, precision, recall, F1 and confusion counts.

The same score model means image- and pixel-level ROC-AUC/AP **must be identical across the two arms**. Threshold-dependent F1/precision/recall may differ. These findings must be described as *synthetic-aided threshold calibration*, not synthetic-augmented model training.

## Leakage and provenance checks

- Exact train/test path intersection is rejected.
- Exact train/test file SHA-256 overlap is rejected, including differently named copies.
- Synthetic manifest must have the expected schema, method, dataset, normal-train split and matching category.
- Every consumed synthetic image must link to an approved train source whose current SHA-256 matches the saved source hash.
- Synthetic image/mask files are required to stay inside the manifest directory; path traversal and symlink escape are rejected.
- Original test images and masks are never used to choose thresholds.

**Limitations:** No near-duplicate/perceptual leakage check, no subject/group identity separation, and no real-dataset integration benchmark in CI. The normal prototype is position-sensitive and will be unsuitable for many unregistered or variable industrial scenes. Synthetic masks come from a procedural handcrafted generator. Metrics based on an incomplete `--max-test` subset must not be reported as full benchmark results. AUC/AP are undefined and serialized as `null` for single-class test subsets.

## Outputs and next milestone

The JSON report records the dataset/category, sample limits, train/test counts, thresholds, both metric sets, synthetic manifest provenance and the exact test source paths/scores. Reports are create-only, preventing accidental overwrites.

The **next** scientific milestone is a *matched* detector architecture trained under real-only and real+synthetic augmentation conditions with proper validation, multiple seeds, confidence intervals, group-level leakage audits and a fixed official test split. The present baseline intentionally does not claim that milestone.
