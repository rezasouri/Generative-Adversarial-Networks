"""Command-line entry point for ForgeGAN."""

from __future__ import annotations

import argparse

from forgegan import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgegan",
        description="Industrial defect synthesis and generative anomaly detection.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("info", help="Show project and environment information.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "info":
        print(f"ForgeGAN {__version__}")
        print("Status: foundation stage")
        print("Training and dataset commands are planned, not yet available.")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
