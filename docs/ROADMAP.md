# ForgeGAN roadmap

This roadmap distinguishes intended work from delivered functionality.

## Milestone 0 — Foundation (current)

- [x] Establish project identity and research objective
- [x] Preserve original notebooks as historical material
- [x] Add installable PyTorch package scaffold and CLI
- [x] Add contribution and experiment-integrity guidance
- [x] Add reusable image dataset integrity audit command
- [x] Audit and document the legacy five-class dataset
- [ ] Add CI and baseline tests

## Milestone 1 — Industrial data pipeline

- [ ] Verify official MVTec AD access, license/terms and data layout
- [ ] Verify official VisA access, license/terms and data layout
- [ ] Dataset abstraction and deterministic transforms
- [ ] MVTec AD adapter
- [ ] VisA adapter
- [ ] Product/category-level split validation and leakage checks
- [ ] Dataset inspection utility and sample visualizations

## Milestone 2 — Generative baselines

- [ ] DCGAN baseline
- [ ] Conditional GAN with mask/class conditioning
- [ ] ACGAN baseline
- [ ] WGAN-GP (gradient penalty; no weight clipping)
- [ ] Checkpointing, resume, fixed-noise sample grids

## Milestone 3 — Defect synthesis

- [ ] Defect-mask conditioning and compositing strategy
- [ ] Controls for defect category, position and scale
- [ ] Diversity and artifact analysis
- [ ] Human-reviewable synthetic sample gallery

## Milestone 4 — Downstream evaluation

- [ ] Real-only anomaly-detection baseline
- [ ] Synthetic-augmented training protocol
- [ ] Image-level and pixel-level metrics
- [ ] Multiple seeds and confidence intervals
- [ ] Ablations by defect category and data budget

## Milestone 5 — Modern comparison and presentation

- [ ] Carefully scoped diffusion baseline
- [ ] Gradio demo
- [ ] Benchmark report with limitations
- [ ] Reproducible end-to-end examples
- [ ] Release and model/data cards

## Acceptance criteria

A milestone is complete only when its code, tests, documentation and reproducible evidence are present. A model generating plausible images is not sufficient evidence that synthetic data improves anomaly detection.
