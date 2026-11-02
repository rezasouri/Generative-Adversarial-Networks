# Experiment tooling

## Dataset inspection

Create a deterministic contact sheet and a JSON manifest without changing source data:

```bash
forgegan inspect-dataset \
  --dataset mvtec \
  --root datasets/mvtec_ad \
  --split train \
  --category bottle \
  --limit 32 \
  --output-dir reports/mvtec-bottle-train
```

For VisA, provide the official split CSV:

```bash
forgegan inspect-dataset \
  --dataset visa \
  --root datasets/visa \
  --split-file datasets/visa/split_csv/1cls.csv \
  --split test \
  --category candle \
  --output-dir reports/visa-candle-test
```

The command writes `contact-sheet.png` and `manifest.json`. The manifest records the selected source paths and labels so visual review can be traced back to the dataset. Do not commit private dataset images or generated contact sheets unless redistribution is permitted.

## YAML experiment configuration

A minimal example lives at `configs/experiments/mvtec-bottle-dcgan.yaml`.

```python
from forgegan.experiments.config import load_experiment_config

config = load_experiment_config(
    "configs/experiments/mvtec-bottle-dcgan.yaml"
)
print(config.name, config.seed, config.training)
```

The version-1 loader uses safe YAML parsing, rejects unknown top-level/dataset keys, checks the dataset identifier and split, and validates required training values. Model-specific fields are retained as a mapping for future model plugins. Loading a config does not start training, download data, create checkpoints or claim a benchmark result.

## CI

The standard CI job runs lint and the dependency-light test suite. A separate optional training-extra job installs PyTorch and runs the real DataLoader integration test. This intentionally tests the optional dependency path rather than silently skipping it in every CI run.
