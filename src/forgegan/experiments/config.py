"""Versioned YAML experiment configuration with strict validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from forgegan.data.registry import DATASET_NAMES


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    seed: int
    dataset: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    output_dir: Path
    version: int = 1


_ALLOWED_TOP_LEVEL = {
    "version", "name", "seed", "dataset", "model", "training", "output_dir"
}


def _require_mapping(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a YAML mapping")
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate config; reject unknown keys to catch typos early."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = _require_mapping(raw, "experiment")
    unknown = set(raw) - _ALLOWED_TOP_LEVEL
    missing = {"name", "seed", "dataset", "model", "training", "output_dir"} - set(raw)
    if unknown:
        raise ValueError(f"Unknown experiment keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing experiment keys: {sorted(missing)}")

    version = raw.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported experiment config version: {version}")
    name = raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    dataset = _require_mapping(raw["dataset"], "dataset")
    dataset_unknown = set(dataset) - {
        "name", "root", "split", "category", "split_file", "verify_files"
    }
    if dataset_unknown:
        raise ValueError(f"Unknown dataset keys: {sorted(dataset_unknown)}")
    if dataset.get("name") not in DATASET_NAMES:
        raise ValueError(f"dataset.name must be one of {DATASET_NAMES}")
    if dataset.get("split") not in {"train", "test"}:
        raise ValueError("dataset.split must be 'train' or 'test'")
    if not isinstance(dataset.get("root"), str) or not dataset["root"]:
        raise ValueError("dataset.root must be a non-empty path string")
    if dataset["name"] == "mvtec" and dataset.get("split_file") is not None:
        raise ValueError("MVTec does not use split_file")

    model = _require_mapping(raw["model"], "model")
    if not isinstance(model.get("name"), str) or not model["name"]:
        raise ValueError("model.name must be a non-empty string")

    training = _require_mapping(raw["training"], "training")
    epochs = training.get("epochs")
    batch_size = training.get("batch_size")
    learning_rate = training.get("learning_rate")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("training.epochs must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("training.batch_size must be a positive integer")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or learning_rate <= 0
    ):
        raise ValueError("training.learning_rate must be positive")

    output_dir = raw["output_dir"]
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a non-empty path string")

    return ExperimentConfig(
        name=name,
        seed=seed,
        dataset=dataset,
        model=model,
        training=training,
        output_dir=Path(output_dir),
        version=version,
    )
