"""Command-line entry point for ForgeGAN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forgegan import __version__
from forgegan.data.audit import audit_dataset
from forgegan.data.mvtec import load_mvtec_records, validate_mvtec_split
from forgegan.data.registry import DatasetConfig
from forgegan.data.visa import load_visa_records, validate_visa_split
from forgegan.data.visualize import inspect_dataset


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

    inspect_parser = subparsers.add_parser(
        "inspect-dataset", help="Create a contact sheet and sample manifest."
    )
    inspect_parser.add_argument("--dataset", choices=["visa", "mvtec"], required=True)
    inspect_parser.add_argument("--root", type=Path, required=True)
    inspect_parser.add_argument("--split", choices=["train", "test"], default="train")
    inspect_parser.add_argument("--category", help="Optional category filter.")
    inspect_parser.add_argument("--split-file", type=Path, help="VisA 1cls.csv path.")
    inspect_parser.add_argument("--limit", type=int, default=32)
    inspect_parser.add_argument("--columns", type=int, default=4)
    inspect_parser.add_argument("--output-dir", type=Path, default=Path("reports/inspection"))

    train_parser = subparsers.add_parser(
        "train-dcgan", help="Train unconditional 64x64 DCGAN on normal images."
    )
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--resume", type=Path, help="Resume an epoch checkpoint.")
    train_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    conditional_parser = subparsers.add_parser(
        "train-cgan", help="Train category-conditional GAN on normal training images."
    )
    conditional_parser.add_argument("--config", type=Path, required=True)
    conditional_parser.add_argument("--resume", type=Path)
    conditional_parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )

    synthesis_parser = subparsers.add_parser(
        "synthesize-defects",
        help="Create labeled procedural defect/mask pairs from normal train images.",
    )
    synthesis_parser.add_argument("--dataset", choices=["visa", "mvtec"], required=True)
    synthesis_parser.add_argument("--root", type=Path, required=True)
    synthesis_parser.add_argument("--category", required=True)
    synthesis_parser.add_argument("--split-file", type=Path, help="VisA 1cls.csv path.")
    synthesis_parser.add_argument("--output-dir", type=Path, required=True)
    synthesis_parser.add_argument("--kind", choices=["spot", "scratch"], default="spot")
    synthesis_parser.add_argument("--center-x", type=float, default=0.5)
    synthesis_parser.add_argument("--center-y", type=float, default=0.5)
    synthesis_parser.add_argument("--scale", type=float, default=0.12)
    synthesis_parser.add_argument("--opacity", type=float, default=0.75)
    synthesis_parser.add_argument("--angle", type=float, default=30.0)
    synthesis_parser.add_argument("--count", type=int, default=32)
    synthesis_parser.add_argument("--seed", type=int, default=42)

    eval_parser = subparsers.add_parser(
        "evaluate-synthetic",
        help="Compare real-only vs synthetic-calibrated thresholds on held-out test.",
    )
    eval_parser.add_argument("--dataset", choices=["visa", "mvtec"], required=True)
    eval_parser.add_argument("--root", type=Path, required=True)
    eval_parser.add_argument("--category", required=True)
    eval_parser.add_argument("--split-file", type=Path, help="VisA 1cls.csv path")
    eval_parser.add_argument("--synthetic-manifest", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--image-size", type=int, default=64)
    eval_parser.add_argument("--max-train", type=int, default=64)
    eval_parser.add_argument("--max-test", type=int, default=256)
    eval_parser.add_argument("--max-synthetic", type=int, default=64)
    eval_parser.add_argument("--normal-quantile", type=float, default=0.99)

    benchmark_parser = subparsers.add_parser(
        "benchmark-anomaly",
        help="Train matched real-only and synthetic-denoising anomaly detectors.",
    )
    benchmark_parser.add_argument("--dataset", choices=["visa", "mvtec"], required=True)
    benchmark_parser.add_argument("--root", type=Path, required=True)
    benchmark_parser.add_argument("--category", required=True)
    benchmark_parser.add_argument("--split-file", type=Path)
    benchmark_parser.add_argument("--synthetic-manifest", type=Path, required=True)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    benchmark_parser.add_argument("--image-size", type=int, default=64)
    benchmark_parser.add_argument("--max-train", type=int, default=128)
    benchmark_parser.add_argument("--max-test", type=int, default=512)
    benchmark_parser.add_argument("--epochs", type=int, default=5)
    benchmark_parser.add_argument("--batch-size", type=int, default=8)
    benchmark_parser.add_argument("--learning-rate", type=float, default=0.001)
    benchmark_parser.add_argument("--width", type=int, default=16)
    benchmark_parser.add_argument("--quantile", type=float, default=0.99)
    benchmark_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    patchcore_parser = subparsers.add_parser(
        "benchmark-patchcore",
        help="Evaluate a ResNet18 PatchCore-style memory-bank anomaly detector.",
    )
    patchcore_parser.add_argument("--dataset", choices=["visa", "mvtec"], required=True)
    patchcore_parser.add_argument("--root", type=Path, required=True)
    patchcore_parser.add_argument("--category", required=True)
    patchcore_parser.add_argument("--split-file", type=Path)
    patchcore_parser.add_argument("--output", type=Path, required=True)
    patchcore_parser.add_argument("--image-size", type=int, default=224)
    patchcore_parser.add_argument("--max-train", type=int, default=128)
    patchcore_parser.add_argument("--max-test", type=int, default=512)
    patchcore_parser.add_argument("--memory-size", type=int, default=2048)
    patchcore_parser.add_argument("--batch-size", type=int, default=16)
    patchcore_parser.add_argument("--quantile", type=float, default=0.99)
    patchcore_parser.add_argument("--seed", type=int, default=42)
    patchcore_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    patchcore_parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Use random ResNet18 weights for offline smoke tests only.",
    )
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
        print("DCGAN, category cGAN, and procedural defect synthesis are available.")
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
    if args.command == "inspect-dataset":
        config = DatasetConfig(
            name=args.dataset,
            root=args.root,
            split=args.split,
            category=args.category,
            split_file=args.split_file,
        )
        _write_report(
            inspect_dataset(
                config, args.output_dir, limit=args.limit, columns=args.columns
            ),
            None,
        )
        return
    if args.command == "train-dcgan":
        from forgegan.experiments.config import load_experiment_config
        from forgegan.training.dcgan import train_dcgan

        experiment = load_experiment_config(args.config)
        _write_report(
            train_dcgan(experiment, resume=args.resume, device=args.device),
            None,
        )
        return
    if args.command == "train-cgan":
        from forgegan.experiments.config import load_experiment_config
        from forgegan.training.conditional import train_cgan

        experiment = load_experiment_config(args.config)
        _write_report(
            train_cgan(experiment, resume=args.resume, device=args.device),
            None,
        )
        return
    if args.command == "synthesize-defects":
        from forgegan.synthesis.procedural import DefectSpec, generate_synthetic_dataset

        spec = DefectSpec(
            kind=args.kind,
            center_x=args.center_x,
            center_y=args.center_y,
            scale=args.scale,
            opacity=args.opacity,
            angle_degrees=args.angle,
        )
        config = DatasetConfig(
            name=args.dataset,
            root=args.root,
            split="train",
            category=args.category,
            split_file=args.split_file,
        )
        _write_report(
            generate_synthetic_dataset(
                config, args.output_dir, spec=spec, count=args.count, seed=args.seed
            ),
            None,
        )
        return
    if args.command == "evaluate-synthetic":
        from forgegan.evaluation.baseline import (
            EvaluationSettings,
            compare_real_vs_synthetic,
            save_comparison,
        )

        settings = EvaluationSettings(
            image_size=args.image_size,
            max_train=args.max_train,
            max_test=args.max_test,
            max_synthetic=args.max_synthetic,
            normal_quantile=args.normal_quantile,
        )
        config = DatasetConfig(
            name=args.dataset,
            root=args.root,
            split="train",
            category=args.category,
            split_file=args.split_file,
        )
        report = compare_real_vs_synthetic(
            config, args.synthetic_manifest, settings=settings
        )
        save_comparison(report, args.output)
        _write_report(
            {
                "report_path": str(args.output),
                "dataset": report["dataset"],
                "category": report["category"],
                "test_count": report["test_count"],
                "arms": report["arms"],
            },
            None,
        )
        return
    if args.command == "benchmark-anomaly":
        from forgegan.training.anomaly import run_matched_benchmark

        config = DatasetConfig(
            name=args.dataset,
            root=args.root,
            split="train",
            category=args.category,
            split_file=args.split_file,
        )
        result = run_matched_benchmark(
            config,
            args.synthetic_manifest,
            args.output,
            seeds=tuple(args.seeds),
            image_size=args.image_size,
            max_train=args.max_train,
            max_test=args.max_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            width=args.width,
            quantile=args.quantile,
            device=args.device,
        )
        _write_report(result, None)
        return
    if args.command == "benchmark-patchcore":
        from forgegan.training.patchcore import run_patchcore_benchmark

        config = DatasetConfig(
            name=args.dataset,
            root=args.root,
            split="train",
            category=args.category,
            split_file=args.split_file,
        )
        result = run_patchcore_benchmark(
            config,
            args.output,
            image_size=args.image_size,
            max_train=args.max_train,
            max_test=args.max_test,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            quantile=args.quantile,
            pretrained=not args.no_pretrained,
            device=args.device,
            seed=args.seed,
        )
        _write_report(result, None)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
