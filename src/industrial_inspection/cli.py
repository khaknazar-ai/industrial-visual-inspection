"""Dependency-light command line interface for the inspection lifecycle."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import yaml

from industrial_inspection.core.config import load_config, resolve_config_path
from industrial_inspection.core.environment import write_environment_report
from industrial_inspection.core.runtime_resources import runtime_resource_path
from industrial_inspection.data.adapters import build_mvtec_ad_manifest, generate_synthetic_dataset
from industrial_inspection.data.contracts import (
    validate_manifest,
    write_audit_report,
)
from industrial_inspection.data.variants import (
    build_ad2_manifest,
    build_custom_manifest,
    build_loco_manifest,
)
from industrial_inspection.data.visa import build_visa_manifest
from industrial_inspection.export import write_export_metadata
from industrial_inspection.inference.pipeline import InspectionPipeline, decode_image_bytes
from industrial_inspection.models.registry import load_model, model_capabilities


def _json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _model_config(path: Path) -> tuple[str, str, dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    name = data.get("name")
    model_id = data.get("model_id")
    if not name or not model_id:
        raise ValueError(f"Model config must contain name and model_id: {path}")
    return name, model_id, data.get("parameters", {})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="industrial-inspection",
        description="Open-set industrial visual inspection toolkit",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="deployment YAML; explicit CLI value overrides II_CONFIG",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    env = subparsers.add_parser("env", help="inspect the local runtime")
    env.add_argument("action", choices=["inspect"])
    env.add_argument("--output", type=Path, default=Path("reports/environment/environment_report.json"))

    data = subparsers.add_parser("data", help="generate, adapt, or audit data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    generate = data_sub.add_parser("generate-fixture")
    generate.add_argument("--output", type=Path, default=Path("data/fixtures/synthetic"))
    generate.add_argument("--force", action="store_true")
    audit = data_sub.add_parser("audit")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    mvtec = data_sub.add_parser("build-mvtec")
    mvtec.add_argument("--dataset-root", type=Path, required=True)
    mvtec.add_argument("--category", default="bottle")
    mvtec.add_argument("--output", type=Path, default=Path("data/manifests/mvtec_ad_bottle.csv"))
    manifest = data_sub.add_parser("build-manifest")
    manifest.add_argument("--dataset", choices=["mvtec_ad", "mvtec_loco", "mvtec_ad2", "custom", "visa"], required=True)
    manifest.add_argument("--root", type=Path, required=True)
    manifest.add_argument("--category", default="bottle")
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--public-only", action="store_true", help="exclude AD2 private images with unknown labels")
    manifest.add_argument("--split-csv", type=Path, help="official VisA one-class split CSV")

    train = subparsers.add_parser("train", help="fit, calibrate, and evaluate a normal-only model")
    train.add_argument("--manifest", type=Path, default=Path("data/fixtures/synthetic/manifest.csv"))
    train.add_argument("--model-config", type=Path, default=Path("configs/models/classical_diff.yaml"))
    train.add_argument("--experiment", default="EXP-0001")
    train.add_argument("--bootstrap-samples", type=int, default=500)

    infer = subparsers.add_parser("infer", help="inspect one image with a saved model")
    infer.add_argument("--image", type=Path, required=True)
    infer.add_argument("--model-name", default=None)
    infer.add_argument("--model-dir", type=Path, default=None)
    infer.add_argument("--category", default=None)

    subparsers.add_parser("models", help="show available and optional model families")

    policy = subparsers.add_parser("bundle-policy", help="create an unvalidated candidate with explicit new thresholds")
    policy.add_argument("--bundle", type=Path, required=True)
    policy.add_argument("--output", type=Path, required=True)
    policy.add_argument("--pass-threshold", type=float, required=True)
    policy.add_argument("--fail-threshold", type=float, required=True)
    policy.add_argument("--pixel-threshold", type=float)

    benchmark = subparsers.add_parser("benchmark", help="measure warmup-aware latency and memory")
    benchmark.add_argument("--model-name", default="classical_diff")
    benchmark.add_argument("--model-dir", type=Path, required=True)
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument("--runs", type=int, default=100)
    benchmark.add_argument("--warmup", type=int, default=5)
    benchmark.add_argument("--split", choices=["validation", "test"], default="validation")
    benchmark.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="calibrate and evaluate an already-fitted saved model"
    )
    evaluate.add_argument("--model-name", default="classical_diff")
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--experiment", default="EXP-0003")
    evaluate.add_argument("--bootstrap-samples", type=int, default=500)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--threshold-mode", choices=["validation", "model"], default="validation")

    export = subparsers.add_parser("export", help="write export metadata for a reviewed artifact")
    export.add_argument("--model-id", required=True)
    export.add_argument("--source-format", default="native")
    export.add_argument("--target-format", choices=["metadata", "onnx", "openvino"], default="metadata")
    export.add_argument("--output", type=Path, required=True)

    report = subparsers.add_parser("report", help="print a saved experiment summary")
    report.add_argument("--experiment", required=True)

    serve = subparsers.add_parser("serve", help="start the local API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    demo = subparsers.add_parser("demo", help="run the complete synthetic foundation slice")
    demo.add_argument("--force-fixture", action="store_true")
    demo.add_argument("--experiment", default="EXP-0001")
    return parser


def _run_training(
    root: Path,
    config_path: Path,
    manifest: Path,
    model_config: Path,
    experiment: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    model_config_path = root / model_config if not model_config.is_absolute() else model_config
    if not model_config_path.is_file() and model_config == Path("configs/models/classical_diff.yaml"):
        model_config_path = runtime_resource_path(model_config.as_posix())
    name, model_id, parameters = _model_config(model_config_path)
    config = load_config(config_path)
    storage = config.storage_paths(config_path)
    manifest_path = root / manifest if not manifest.is_absolute() else manifest
    from industrial_inspection.evaluation.experiment import run_experiment

    return run_experiment(
        manifest_path=manifest_path,
        model_name=name,
        model_id=model_id,
        model_parameters=parameters,
        model_output=storage.model_root / model_id,
        experiment_output=root / "reports/experiments" / experiment,
        experiment_id=experiment,
        project_root=root,
        seed=config.runtime.seed,
        bootstrap_samples=bootstrap_samples,
        review_costs=config.evaluation.review_costs.policy() if config.evaluation.review_costs else None,
        normal_pass_quantile=config.evaluation.normal_pass_quantile,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = resolve_config_path(args.config)
    root = load_config(config_path).storage_paths().project_root
    if args.command == "env":
        output = args.output if args.output.is_absolute() else root / args.output
        _json(write_environment_report(output, root))
        return 0
    if args.command == "data":
        if args.data_command == "generate-fixture":
            output = args.output if args.output.is_absolute() else root / args.output
            print(generate_synthetic_dataset(output, project_root=root, force=args.force))
            return 0
        if args.data_command == "audit":
            manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
            report = validate_manifest(manifest, root)
            output = args.output or root / "reports/datasets" / f"{report.dataset}_audit.json"
            if not output.is_absolute():
                output = root / output
            write_audit_report(report, output)
            _json(report.as_dict())
            return 0 if report.valid else 2
        if args.data_command == "build-mvtec":
            output = args.output if args.output.is_absolute() else root / args.output
            print(
                build_mvtec_ad_manifest(
                    args.dataset_root,
                    args.category,
                    output,
                    project_root=root,
                )
            )
            return 0
        if args.data_command == "build-manifest":
            output = args.output if args.output.is_absolute() else root / args.output
            dataset_root = args.root if args.root.is_absolute() else root / args.root
            if args.dataset == "mvtec_ad":
                result = build_mvtec_ad_manifest(dataset_root, args.category, output, project_root=root)
            elif args.dataset == "mvtec_loco":
                result = build_loco_manifest(dataset_root, args.category, output, project_root=root)
            elif args.dataset == "mvtec_ad2":
                result = build_ad2_manifest(dataset_root, output, project_root=root, include_private=not args.public_only)
            elif args.dataset == "visa":
                split_csv = args.split_csv or dataset_root / "official-1cls.csv"
                result = build_visa_manifest(dataset_root, split_csv, args.category, output, project_root=root)
            else:
                result = build_custom_manifest(dataset_root, output, project_root=root)
            print(result)
            return 0
    if args.command == "train":
        _json(
            _run_training(
                root,
                config_path,
                args.manifest,
                args.model_config,
                args.experiment,
                args.bootstrap_samples,
            )
        )
        return 0
    if args.command == "infer":
        config = load_config(config_path)
        model_dir = args.model_dir
        if model_dir is not None and not model_dir.is_absolute():
            model_dir = root / model_dir
        pipeline = InspectionPipeline.from_config(
            config,
            model_name=args.model_name,
            model_path=model_dir,
            category=args.category,
        )
        with args.image.open("rb") as handle:
            image_bytes = handle.read(config.runtime.max_image_bytes + 1)
        if len(image_bytes) > config.runtime.max_image_bytes:
            raise ValueError("Image payload is too large")
        image = decode_image_bytes(image_bytes, max_pixels=config.runtime.max_image_pixels)
        _json(pipeline.inspect(image, category=args.category, original_bytes=image_bytes).model_dump(mode="json"))
        return 0
    if args.command == "models":
        _json([capability.as_dict() for capability in model_capabilities()])
        return 0
    if args.command == "bundle-policy":
        from industrial_inspection.models.artifacts import ModelBundle
        from industrial_inspection.scoring.pixels import PixelPolicy
        from industrial_inspection.scoring.policy import DecisionThresholds

        source = args.bundle if args.bundle.is_absolute() else root / args.bundle
        output = args.output if args.output.is_absolute() else root / args.output
        bundle = ModelBundle.load(source)
        pixel = None
        if args.pixel_threshold is not None:
            fields = PixelPolicy.from_dict(bundle.calibration.get("pixel")).as_dict()
            fields["threshold"] = args.pixel_threshold
            pixel = PixelPolicy.from_dict(fields)
        result = bundle.with_policy(output, image=DecisionThresholds(
            args.pass_threshold, args.fail_threshold, source="explicit_policy_override",
        ), pixel=pixel)
        _json({"bundle": str(result.root), "calibration_id": result.calibration_id,
               "status": result.calibration_status, "policy_evaluated": False})
        return 0
    if args.command == "benchmark":
        manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
        model_dir = args.model_dir if args.model_dir.is_absolute() else root / args.model_dir
        config = load_config(config_path)
        output = args.output if args.output.is_absolute() else root / args.output
        from industrial_inspection.evaluation.benchmark import benchmark_saved_model

        result = benchmark_saved_model(model_name=args.model_name, model_path=model_dir, manifest_path=manifest,
                                       project_root=root, output=output, device=config.runtime.device,
                                       warmup=args.warmup, runs=args.runs, split=args.split)
        _json(result)
        return 0
    if args.command == "evaluate":
        manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
        model_path = args.model_dir if args.model_dir.is_absolute() else root / args.model_dir
        config = load_config(config_path)
        model = load_model(args.model_name, model_path, device=config.runtime.device)
        output = args.output if args.output.is_absolute() else root / args.output
        from industrial_inspection.evaluation.saved import evaluate_saved_model

        result = evaluate_saved_model(
            model=model,
            manifest_path=manifest,
            output=output,
            project_root=root,
            model_name=args.model_name,
            model_path=model_path,
            experiment_id=args.experiment,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
            threshold_mode=args.threshold_mode,
            review_costs=config.evaluation.review_costs.policy() if config.evaluation.review_costs else None,
            normal_pass_quantile=config.evaluation.normal_pass_quantile,
        )
        _json(result)
        return 0
    if args.command == "export":
        output = args.output if args.output.is_absolute() else root / args.output
        write_export_metadata(
            output,
            model_id=args.model_id,
            source_format=args.source_format,
            target_format=args.target_format,
        )
        print(output)
        return 0
    if args.command == "report":
        directory = root / "reports/experiments" / args.experiment
        summary = directory / "summary.md"
        metrics = directory / "metrics.json"
        if not summary.is_file() or not metrics.is_file():
            raise FileNotFoundError(f"Experiment artifacts not found: {directory}")
        print(summary.read_text(encoding="utf-8"))
        _json(json.loads(metrics.read_text(encoding="utf-8")))
        return 0
    if args.command == "serve":
        import uvicorn

        from industrial_inspection.api.app import create_app

        uvicorn.run(create_app(config=load_config(config_path)), host=args.host, port=args.port)
        return 0
    if args.command == "demo":
        diagnostics = root / "reports/diagnostics" / f"demo-{uuid.uuid4().hex}"
        fixture = generate_synthetic_dataset(
            root / "data/fixtures/synthetic",
            project_root=root,
            force=args.force_fixture,
        )
        audit_report = validate_manifest(fixture, root)
        write_audit_report(
            audit_report,
            diagnostics / "dataset_audit.json",
        )
        if not audit_report.valid:
            _json(audit_report.as_dict())
            return 2
        write_environment_report(diagnostics / "environment.json", root)
        result = _run_training(
            root,
            config_path,
            fixture,
            Path("configs/models/classical_diff.yaml"),
            args.experiment,
            200,
        )
        _json(result)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
