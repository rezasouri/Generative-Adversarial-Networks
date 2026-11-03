# ForgeGAN

**Industrial Defect Synthesis & Generative Anomaly Detection**

ForgeGAN is an open, reproducible research and engineering project for studying whether controlled synthetic defect generation can improve industrial visual anomaly detection.

> Status: industrial dataset pipeline. VisA and MVTec AD read-only adapters and validation commands are available. The original 2023 Keras notebooks are preserved under `src/` as historical experiments; the modern implementation is being built independently with PyTorch.

## Why ForgeGAN?

Industrial inspection datasets are often imbalanced: normal samples are abundant, while defect examples can be scarce and diverse. ForgeGAN explores a measurable workflow:

1. Train generative models on industrial imagery.
2. Generate controlled synthetic defect examples.
3. Evaluate image quality and diversity.
4. Train anomaly-detection baselines with real-only and augmented data.
5. Compare results under reproducible protocols.

Synthetic data is treated as an experimental intervention—not an assumed improvement. Every benchmark must include real-only baselines and disclose the exact data split.

## Planned capabilities

- Modular PyTorch training engine (initial DCGAN baseline implemented)
- DCGAN, conditional GAN, ACGAN and WGAN-GP baselines
- Mask-conditioned defect synthesis
- Dataset adapters for MVTec AD and VisA (dataset access and licensing notes in [DATASET_STRATEGY.md](docs/DATASET_STRATEGY.md))
- Real-only vs. synthetic-augmented anomaly-detection experiments
- Quantitative evaluation, experiment configs and reproducible reports
- Optional diffusion baseline and Gradio demo

## Repository status

The repository is currently in its foundation stage. Planned features are not presented as implemented. See [ROADMAP.md](docs/ROADMAP.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

Requirements: Python 3.10–3.12.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
# Add PyTorch when installing the training components:
python -m pip install -e ".[train]"
forgegan --help
forgegan audit-dataset --root "src/Dataset" --output reports/legacy-dataset-audit.json
forgegan validate-visa --root datasets/visa --split-file datasets/visa/split_csv/1cls.csv
forgegan validate-mvtec --root datasets/mvtec_ad --category bottle
# After obtaining the dataset archive:
forgegan train-dcgan --config configs/experiments/mvtec-bottle-dcgan.yaml
pytest
```

The CLI provides dataset audits and validation. The shared Python dataset registry, spatial transforms and optional PyTorch DataLoader are documented in [DATA_PIPELINE.md](docs/DATA_PIPELINE.md). Dataset inspection and versioned experiment configs are covered in [EXPERIMENT_TOOLING.md](docs/EXPERIMENT_TOOLING.md). The adapters do not download or modify source datasets. Unconditional DCGAN training is available; controlled defect synthesis and downstream evaluation remain planned. See [DCGAN training guide](docs/DCGAN_TRAINING.md).

## Project principles

- Reproducibility: explicit seeds, configs, versioned splits and saved metadata.
- Honest evaluation: no synthetic samples may leak across evaluation splits.
- Modular design: datasets, generators, trainers and metrics remain replaceable.
- Responsible data use: follow each dataset's license and terms.
- Evidence over claims: report negative and neutral results as well as gains.

## Legacy experiments

The original DCGAN, ACGAN and Wasserstein GAN notebooks are retained in `src/` for provenance. They are not yet part of the new training API and may require legacy dependencies and local-path cleanup.

## License

MIT for the project code. Dataset licenses and terms remain separate and must be respected.
