# ForgeGAN roadmap

This roadmap distinguishes intended work from delivered functionality.

## Milestone 0 — Foundation (current)

- [x] Establish project identity and research objective
- [x] Preserve original notebooks as historical material
- [x] Add installable PyTorch package scaffold and CLI
- [x] Add contribution and experiment-integrity guidance
- [x] Add reusable image dataset integrity audit command
- [x] Audit and document the legacy five-class dataset
- [x] Add CI and baseline tests

## Milestone 1 — Industrial data pipeline

- [x] Verify official MVTec AD access, license/terms and data layout
- [x] Verify official VisA access, license/terms and data layout
- [x] Dataset record abstraction and deterministic metadata loading
- [x] Unified dataset registry and normalized sample schema
- [x] Deterministic image/mask resize transforms
- [x] Optional reproducibly seeded PyTorch DataLoader factory
- [x] MVTec AD adapter using the official train/test/ground-truth layout
- [x] VisA 1-class adapter using the official split CSV
- [x] Product/category-level source-image split leakage checks
- [x] VisA metadata and file validation CLI
- [x] MVTec metadata, annotation and image-mask validation CLI
- [x] Dataset inspection utility and sample visualizations
- [x] Strict versioned YAML experiment configuration
- [x] Optional PyTorch-enabled DataLoader CI job

## Milestone 2 — Generative baselines

- [x] DCGAN 64x64 unconditional baseline with train/resume smoke tests
- [x] Category-conditioned GAN (normal train images, class labels)
- [ ] Mask/defect-conditioned synthesis with authorized defect training data
- [ ] ACGAN baseline
- [ ] WGAN-GP (gradient penalty; no weight clipping)
- [x] DCGAN epoch checkpointing, resume, fixed-noise sample grids
- [ ] Common trainer abstractions for other generative models

## Milestone 3 — Defect synthesis

- [x] Procedural binary defect masks and masked-intensity compositing baseline
- [ ] Learned mask-conditioned defect generator
- [x] Procedural defect type (spot/scratch), position, scale and opacity controls
- [ ] Learned defect-category conditioning
- [ ] Diversity and artifact analysis
- [x] Synthetic image/mask pairs and traceable JSON manifest
- [ ] Dedicated synthetic sample gallery

## Milestone 4 — Downstream evaluation

- [x] Normal-prototype pixel anomaly scoring baseline (non-learned)
- [x] Train-only synthetic-assisted threshold calibration comparison
- [ ] Matched learned real-only vs synthetic-augmented detector training protocol
- [x] Image-level and pixel-level ROC-AUC/AP/precision/recall/F1 reporting
- [x] Multi-seed runs with per-seed metrics and across-seed uncertainty summaries
- [ ] Ablations by defect category and data budget

## Milestone 5 — Modern comparison and presentation

- [ ] Carefully scoped diffusion baseline
- [ ] Gradio demo
- [ ] Benchmark report with limitations
- [ ] Reproducible end-to-end examples
- [ ] Release and model/data cards

## Acceptance criteria

A milestone is complete only when its code, tests, documentation and reproducible evidence are present. A model generating plausible images is not sufficient evidence that synthetic data improves anomaly detection.
