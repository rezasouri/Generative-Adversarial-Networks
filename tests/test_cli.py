from forgegan.cli import build_parser


def test_cli_info_parser() -> None:
    args = build_parser().parse_args(["info"])
    assert args.command == "info"


def test_cli_version_parser() -> None:
    args = build_parser().parse_args(["--version"])
    assert args is not None
