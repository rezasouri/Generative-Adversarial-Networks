# Matched learned anomaly baseline: real-only vs synthetic denoising

This experiment is the first learned downstream comparison in ForgeGAN. It compares the same compact convolutional autoencoder trained under two data regimes. It is an exploratory baseline, not a state-of-the-art benchmark.

## Prepare synthetic training pairs

The generator must use normal images from the official training split. Generate enough samples to ensure some variants derive from the fit portion of the train split (the benchmark reserves a deterministic 20% source-level holdout for threshold calibration):

```bash
forgegan synthesize-defects \
  --dataset mvtec --root datasets/mvtec_ad \
  --category bottle --output-dir runs/procedural-bottle \
  --kind spot --count 256 --seed 42
```

The benchmark checks each manifest source path and SHA-256. Synthetic variants derived from the calibration holdout are excluded from model fitting. If no synthetic sample comes from a fit image, increase the synthesis count.

## Run the matched experiment

```bash
forgegan benchmark-anomaly \
  --dataset mvtec --root datasets/mvtec_ad \
  --category bottle \
  --synthetic-manifest runs/procedural-bottle/manifest.json \
  --output reports/bottle-matched-benchmark.json \
  --seeds 11 22 33 \
  --epochs 5 --batch-size 8 --device auto
```

Install the optional training dependencies first: `python -m pip install -e ".[train,dev]"`.

## Experimental design

Both arms use the same architecture, initialization seed, Adam optimizer, learning rate, batch size, number of epochs and optimizer updates per seed.

- **Real-only:** input is a normal official training image; target is the same clean image.
- **Real + synthetic:** each update contains clean normal identity pairs and procedural-defect inputs whose target is the original clean normal source image. This is a denoising objective.

The model predicts a reconstruction. The anomaly map is the mean absolute RGB residual between input and reconstruction. Pixel and image thresholds are selected from a deterministic, source-disjoint 20% normal-only calibration holdout from the official training split. The remaining selected normal training sources fit the model. Synthetic samples sourced from calibration images are not used for fitting.

The official test split is opened only after both models are trained and thresholds are fixed. Test images and masks are used only for final metrics. The benchmark rejects exact train/test path or SHA-256 overlap, validates synthetic source hashes, and requires manifest-relative image/mask paths.

## Metrics and reporting

Per seed and arm, the JSON report includes:
- Pixel-level ROC-AUC, average precision, precision, recall and F1
- Image-level ROC-AUC, average precision, precision, recall and F1
- Train-calibrated pixel and image thresholds
- Training losses, optimizer update counts and test sample identities
- Source manifest provenance and split counts

Across-seed mean, sample standard deviation and a normal-approximation 95% interval are also reported. With only three seeds, these intervals are descriptive and should not be interpreted as precise population confidence intervals. Increase seeds (e.g. 5–10) for a stronger study.

## Important caveats

- The current procedural anomalies are handcrafted intensity shifts; the experiment tests whether denoising these patterns helps the reconstruction baseline, not whether they represent real industrial defect physics.
- The real-only arm is a plain identity reconstruction baseline; it may learn to reconstruct anomalies and is not expected to be competitive with mature methods such as PatchCore, PaDiM or industrial segmentation systems.
- The convolutional autoencoder is intentionally compact and trained from scratch. This is a reproducible engineering baseline, not a competitive benchmark.
- Source checks detect exact paths and byte-identical files, not perceptual near-duplicates or product/subject identity leakage.
- A max-test subset must not be described as the full official benchmark.
- Report the dataset license/terms and category when publishing results. Do not redistribute MVTec AD or VisA data.
- No numerical result or improvement is claimed until the experiment is run on the full official dataset and independently reviewed.

## Next research improvements

1. Add a strong pretrained feature baseline (e.g. PatchCore) under the same split policy.
2. Replace procedural intensity shifts with realistic, documented defect synthesis and human-reviewed masks.
3. Run multiple categories, 5–10 seeds and paired statistical analysis.
4. Add perceptual duplicate and product-instance leakage checks.
5. Publish qualitative image/mask/reconstruction galleries and per-category failure analysis.
