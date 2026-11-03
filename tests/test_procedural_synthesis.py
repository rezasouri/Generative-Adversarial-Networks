"""Procedural defect synthesis contracts and dataset leakage safeguards."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forgegan.data.registry import DatasetConfig
from forgegan.synthesis.procedural import (
    DefectSpec,
    generate_synthetic_dataset,
    make_mask,
    synthesize_defect,
)


def _mvtec_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "mvtec"
    base = root / "bottle"
    for index in (0, 1):
        image_file = base / "train" / "good" / f"{index:03d}.png"
        image_file.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (110 + index, 130, 150)).save(image_file)
    (base / "test" / "good").mkdir(parents=True)
    Image.new("RGB", (64, 64), (30, 40, 50)).save(base / "test/good/test.png")
    (base / "ground_truth").mkdir()
    return root


@pytest.mark.parametrize("kind", ["spot", "scratch"])
def test_mask_and_image_change_only_inside_mask(kind: str) -> None:
    original = Image.new("RGB", (64, 64), (110, 130, 150))
    spec = DefectSpec(kind=kind, scale=0.17, center_x=0.5, center_y=0.5)
    synthesized, mask = synthesize_defect(original, spec, seed=18)
    before = np.asarray(original)
    after = np.asarray(synthesized)
    region = np.asarray(mask) != 0
    assert region.any()
    assert set(np.unique(np.asarray(mask))) <= {0, 255}
    assert np.array_equal(before[~region], after[~region])
    assert np.any(before[region] != after[region])


def test_same_seed_is_reproducible() -> None:
    source = Image.new("RGB", (64, 64), (100, 120, 130))
    spec = DefectSpec(kind="spot", center_x=0.2, scale=0.15)
    a, mask_a = synthesize_defect(source, spec, seed=8)
    b, mask_b = synthesize_defect(source, spec, seed=8)
    assert a.tobytes() == b.tobytes()
    assert mask_a.tobytes() == mask_b.tobytes()


def test_spatial_controls_affect_mask_location() -> None:
    left = make_mask((128, 128), DefectSpec(center_x=0.2, center_y=0.4), 4)
    right = make_mask((128, 128), DefectSpec(center_x=0.8, center_y=0.4), 4)
    assert left.getbbox() != right.getbbox()


@pytest.mark.parametrize(
    "spec",
    [
        {"scale": 0},
        {"scale": 0.9},
        {"opacity": 0},
        {"center_x": -0.1},
        {"center_y": 1.2},
        {"angle_degrees": float("nan")},
        {"kind": "unsupported"},
    ],
)
def test_invalid_controls_rejected(spec: dict) -> None:
    with pytest.raises(ValueError):
        DefectSpec(**spec)


def test_output_manifest_tracks_source_and_masks(tmp_path: Path) -> None:
    root = _mvtec_fixture(tmp_path)
    out = tmp_path / "derived"
    dataset = DatasetConfig("mvtec", root, "train", category="bottle")
    source_bytes = (root / "bottle/train/good/000.png").read_bytes()
    result = generate_synthetic_dataset(
        dataset, out, spec=DefectSpec(kind="scratch"), count=3, seed=12
    )
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert result["count"] == 3
    assert manifest["not_learned"] is True
    assert len(manifest["items"]) == 3
    for row in manifest["items"]:
        assert row["source_split"] == "train"
        assert row["synthetic_label"] == "procedural_anomaly"
        assert (out / row["image"]).is_file()
        assert (out / row["mask"]).is_file()
        original = Path(row["source_path"])
        assert row["source_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert (root / "bottle/train/good/000.png").read_bytes() == source_bytes


def test_rejects_test_source_and_output_inside_dataset(tmp_path: Path) -> None:
    root = _mvtec_fixture(tmp_path)
    test_config = DatasetConfig("mvtec", root, "test", category="bottle")
    train_config = DatasetConfig("mvtec", root, "train", category="bottle")
    with pytest.raises(ValueError, match="official train"):
        generate_synthetic_dataset(
            test_config, tmp_path / "out", spec=DefectSpec(), count=1
        )
    with pytest.raises(ValueError, match="outside"):
        generate_synthetic_dataset(
            train_config, root / "derived", spec=DefectSpec(), count=1
        )


def test_rejects_nonempty_output(tmp_path: Path) -> None:
    root = _mvtec_fixture(tmp_path)
    out = tmp_path / "derived"
    out.mkdir()
    (out / "existing.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="not empty"):
        generate_synthetic_dataset(
            DatasetConfig("mvtec", root, "train", category="bottle"),
            out,
            spec=DefectSpec(),
            count=1,
        )
