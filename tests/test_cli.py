from pathlib import Path

import pytest
from PIL import Image

from forgegan.cli import build_parser
from forgegan.data.audit import audit_dataset


def _write_image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=color).save(path)


def test_audit_reports_class_counts_and_dimensions(tmp_path: Path) -> None:
    _write_image(tmp_path / "Class 1" / "a.png", 0)
    _write_image(tmp_path / "Class 1" / "b.png", 255)
    _write_image(tmp_path / "Class 2" / "c.png", 127)

    report = audit_dataset(tmp_path)

    assert report["class_count"] == 2
    assert report["image_count"] == 3
    assert report["valid_count"] == 3
    assert report["invalid_count"] == 0
    assert report["class_imbalance"]["balanced"] is False
    assert report["classes"][0]["dimensions"] == [[32, 32]]


def test_audit_detects_identical_files(tmp_path: Path) -> None:
    _write_image(tmp_path / "Class 1" / "same-a.png", 42)
    _write_image(tmp_path / "Class 2" / "same-b.png", 42)

    report = audit_dataset(tmp_path)

    assert report["duplicate_content_groups"] == [
        ["Class 1/same-a.png", "Class 2/same-b.png"]
    ]


def test_audit_rejects_missing_class_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No class directories"):
        audit_dataset(tmp_path)


def test_cli_info_parser() -> None:
    args = build_parser().parse_args(["info"])
    assert args.command == "info"


def test_cli_audit_parser() -> None:
    args = build_parser().parse_args(["audit-dataset", "--root", "data/images"])
    assert args.command == "audit-dataset"
    assert args.root == Path("data/images")


def test_cli_version_parser(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--version"])
    assert exc_info.value.code == 0
    assert "forgegan 0.1.0" in capsys.readouterr().out
