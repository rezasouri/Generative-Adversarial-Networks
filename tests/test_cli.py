import pytest

from forgegan.cli import build_parser


def test_cli_info_parser() -> None:
    args = build_parser().parse_args(["info"])
    assert args.command == "info"


def test_cli_version_parser(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--version"])
    assert exc_info.value.code == 0
    assert "forgegan 0.1.0" in capsys.readouterr().out
