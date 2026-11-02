# Legacy dataset audit

## Scope and method

This audit covers the image files currently stored under `src/Dataset/Class 1` through `Class 5` in the repository's original default branch. It is a metadata and file-integrity audit, not a semantic validation of the class labels or a model-quality assessment.

The GitHub repository API was used to enumerate every class directory and inspect file metadata. Representative images were visually inspected. Duplicate detection compared Git blob identities, which represent identical file contents for these tracked files.

## Findings

| Class directory | Images | Minimum file size | Median file size | Maximum file size |
|---|---:|---:|---:|---:|
| Class 1 | 201 | 391 B | 531 B | 926 B |
| Class 2 | 201 | 566 B | 645 B | 774 B |
| Class 3 | 201 | 554 B | 615 B | 663 B |
| Class 4 | 201 | 614 B | 751 B | 811 B |
| Class 5 | 201 | 564 B | 615 B | 651 B |
| **Total** | **1,005** | **391 B** | **630 B** | **926 B** |

- All 1,005 tracked files use the `.png` extension.
- All 1,005 files have distinct Git blob identities; no byte-identical duplicates were found.
- A representative image from Class 1 has PNG dimensions of 32 × 32 pixels. The sample images inspected visually resemble small binary hand/finger gesture silhouettes. This visual impression is not a verified dataset label or provenance.
- The repository does not provide a dataset citation, collection protocol, subject identity, capture conditions, image license, or reliable class-name mapping. Keep the labels as `Class 1` … `Class 5` until authoritative metadata is found.

## Legacy notebook risks

The Wasserstein notebook contains a label-generation block with strict inequalities at class boundaries. Indices 201, 402, 603 and 804 do not follow the intended 201-sample-per-class mapping. More importantly, it shuffles the label array and image array independently, so even corrected labels would no longer correspond to their images. The shown WGAN critic uses weight clipping rather than gradient penalty, and the training code has hard-coded local dataset paths.

These are reasons not to treat the notebook as a reliable conditional-GAN benchmark. Preserve it for historical reference; do not silently reuse its labels or training pipeline.

## Decision

1. Retain the existing images and notebooks as a legacy educational experiment.
2. Do not use this dataset as ForgeGAN's industrial defect benchmark.
3. Build independent adapters for MVTec AD and VisA after checking their official download instructions, terms and directory structures.
4. Run the new `forgegan audit-dataset` command on every dataset copy before training. Save the JSON report with the experiment metadata.
5. Establish product/category-level train, validation and test splits before any augmentation. Synthetic images must never enter validation or test sets.

## Reproducing the audit

From the repository root:

```bash
python -m pip install -e .
forgegan audit-dataset --root "src/Dataset" --output reports/legacy-dataset-audit.json
```

The command reports class counts, file sizes, image dimensions, modes, formats, unreadable images and byte-identical duplicates. It does not modify the source dataset.
