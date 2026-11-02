"""Command-line entry point for ForgeGAN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forgegan import __version__
from forgegan.data.audit import audit_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgegan",
        description="Industrial defect synthesis and generative anomaly detection.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("info", help="Show project and environment information.")
    audit_parser = subparsers.add_parser(
        "audit-dataset", help="Audit image counts, dimensions, formats and duplicates."
    )
    audit_parser.add_argument(
        "--root",
        type=Path,
        default=Path("src/Dataset"),
        help="Dataset root containing 'Class *' directories (default: src/Dataset).",
    )
    audit_parser.add_argument("--output", type=Path, help="Optional path for a JSON report.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "info":
        print(f"ForgeGAN {__version__}")
        print("Status: foundation and legacy-dataset audit")
        print("GAN training and industrial dataset adapters are not yet available.")
        return
    if args.command == "audit-dataset":
        report = audit_dataset(args.root)
        rendered = json.dumps(report, indent=2)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
