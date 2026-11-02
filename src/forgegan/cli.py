"""Command-line entry point for ForgeGAN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forgegan import __version__
from forgegan.data.audit import audit_dataset
from forgegan.data.mvtec import load_mvtec_records, validate_mvtec_split
from forgegan.data.visa import load_visa_records, validate_visa_split


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

    visa_parser = subparsers.add_parser(
        "validate-visa", help="Validate the official VisA 1-class split and source files."
    )
    visa_parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/visa"),
        help="Extracted VisA root (default: datasets/visa).",
    )
    visa_parser.add_argument(
        "--split-file",
        type=Path,
        help="Optional path to the official 1cls.csv (default: ROOT/split_csv/1cls.csv).",
    )
    visa_parser.add_argument(
        "--category",
        choices=[
            "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
            "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
        ],
        help="Validate one category instead of all 12.",
    )
    visa_parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Validate CSV structure and split integrity without checking image/mask files.",
    )
    visa_parser.add_argument("--output", type=Path, help="Optional path for a JSON report.")

    mvtec_parser = subparsers.add_parser(
        "validate-mvtec", help="Validate the MVTec AD directory layout and annotations."
    )
    mvtec_parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/mvtec_ad"),
        help="Extracted MVTec AD root (default: datasets/mvtec_ad).",
    )
    mvtec_parser.add_argument(
        "--category",
        choices=[
            "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
            "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
            "zipper",
        ],
        help="Validate one category instead of all 15.",
    )
    mvtec_parser.add_argument("--output", type=Path, help="Optional path for a JSON report.")
    return parser


def _write_report(report: dict, output: Path | None) -> None:
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "info":
        print(f"ForgeGAN {__version__}")
        print("Status: foundation and industrial dataset pipeline")
        print("GAN training and defect synthesis are not yet available.")
        return
    if args.command == "audit-dataset":
        _write_report(audit_dataset(args.root), args.output)
        return
    if args.command == "validate-visa":
        records = load_visa_records(
            args.root,
            args.split_file,
            category=args.category,
            verify_files=not args.skip_file_check,
        )
        report = validate_visa_split(records)
        report["file_checks_performed"] = not args.skip_file_check
        _write_report(report, args.output)
        return
    if args.command == "validate-mvtec":
        records = load_mvtec_records(args.root, category=args.category)
        _write_report(validate_mvtec_split(records), args.output)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
