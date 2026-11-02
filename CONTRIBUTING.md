# Contributing

Thanks for helping improve ForgeGAN.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Quality checks

Run before opening a pull request:

```bash
ruff check .
pytest
```

## Experimental requirements

- Keep train, validation and test splits isolated at the source-image/product level.
- Never use test images to tune generation or anomaly-detection thresholds.
- Record random seeds, config files, dataset version and dependency versions.
- Compare every synthetic-data experiment against a real-only baseline.
- Report failed runs and limitations; do not cherry-pick generated samples.
- Do not commit datasets, checkpoints, caches or private data.

## Pull requests

Prefer small, focused changes with tests and updated documentation. Clearly label features that are planned, experimental or validated.
