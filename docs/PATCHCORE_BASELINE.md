# PatchCore-style industrial anomaly baseline

ForgeGAN includes a small, training-free PatchCore-style baseline for MVTec AD and VisA. It is an independent implementation inspired by PatchCore, **not** a wrapper around the official Anomalib implementation and not a claim of state-of-the-art parity.

## Method

1. Use ImageNet-pretrained ResNet-18 and capture layer2 and layer3 feature maps.
2. Bilinearly align layer3 to layer2 resolution, concatenate the channels and apply local 3x3 average pooling to form patch embeddings.
3. Build a memory bank only from normal images in the official training fit subset.
4. Apply deterministic farthest-first coreset selection to a bounded candidate pool.
5. Score each test patch by its exact L2 distance to the nearest memory-bank vector. Distances are computed in chunks to bound temporary memory.
6. Upsample patch distances to the input image size to obtain a pixel anomaly map. Image score is the maximum pixel score.
7. Select pixel and image thresholds from a disjoint normal-only calibration holdout from the official training split; evaluate once on the official test split.

## Install and run

```bash
python -m pip install -e ".[train,dev]"
forgegan benchmark-patchcore \
  --dataset mvtec \
  --root datasets/mvtec_ad \
  --category bottle \
  --output reports/bottle-patchcore.json \
  --device auto
```

The first pretrained run downloads the torchvision ResNet-18 ImageNet weights. For an offline software smoke test only, use `--no-pretrained`; random backbone weights are **not** suitable for reporting benchmark performance.

Useful controls include `--image-size` (default 224), `--max-train` (128), `--max-test` (512), `--memory-size` (2048), `--batch-size` (16), `--quantile` (0.99), and `--seed` (42). Increase sample limits to cover the full official category before publishing metrics. The output path is create-only to avoid overwriting results.

For VisA, pass `--dataset visa`, its extracted root, category and optional `--split-file` pointing to the official `1cls.csv`.

## Split integrity

A deterministic 20% source-level holdout of selected normal train images is reserved for threshold calibration; remaining selected normal images populate the memory bank. Test images and masks are never used to choose thresholds. The implementation rejects exact train/test resolved-path overlap and byte-identical SHA-256 overlap.

These checks do not detect perceptual near-duplicates, repeated physical product identities, or derived crops. Report dataset category, sample counts, image size, memory size, pretrained status, and whether the full test split was evaluated.

## Metrics

The JSON report contains image-level and pixel-level ROC-AUC, average precision, precision, recall, F1 and confusion counts, plus thresholds, memory-bank size, feature-grid dimensions, test sample identities, and split provenance.

## Known limitations

- Uses ResNet-18 layer2/layer3 features and a simple concatenation/average-pooling embedding; official PatchCore implementations may use different embedding aggregation and reweighting.
- Greedy coreset is run on a deterministic bounded candidate subset for runtime; it is not an exact full-dataset k-center solution.
- Image score is the maximum upsampled pixel distance; no PatchCore neighborhood reweighting is implemented.
- No multi-category aggregate benchmark, multi-seed uncertainty analysis, or comparison against official Anomalib has yet been completed.
- The default CLI test cap is a development convenience, not a publishable full benchmark.
- Pretrained weights require network access on first use and must be reported accurately.


## Visual diagnostics

After a benchmark run, recreate the feature bank and test anomaly maps from the saved report:

```bash
forgegan diagnose-patchcore \
  --dataset mvtec \
  --root datasets/mvtec_ad \
  --category bottle \
  --benchmark-report reports/bottle-patchcore.json \
  --output-dir reports/bottle-patchcore-diagnostics \
  --device auto
```

Open `reports/bottle-patchcore-diagnostics/index.html` in a browser. It shows each test image alongside its ground-truth mask, anomaly heatmap, overlay and thresholded prediction. Samples are ranked by false-positive pixel count. The accompanying `diagnostics.json` contains per-image counts and threshold analysis.

Threshold candidates (95%, 97.5%, 99%, 99.5%, 99.9%) are computed only from the normal-only train calibration maps. Test metrics at those thresholds are descriptive diagnostics; the command does not select a threshold using test labels. The command verifies the saved test manifest and reproduces the saved calibration threshold before rendering. If the test ordering, model weights or effective benchmark settings no longer match, it fails rather than silently producing a mismatched report.

The HTML embeds its images and can be opened locally without a web server. It reruns feature extraction and nearest-neighbor scoring; no detector training occurs.
