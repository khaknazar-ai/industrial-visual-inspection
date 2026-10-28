"""Shared configuration arguments for the EfficientAD cloud entry points."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from industrial_inspection.models.efficientad_adapter import (
    EfficientAdConfig,
    load_efficientad_config,
)


def parser_for(description: str, root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=root / "configs/models/efficientad.yaml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--pretrained-root", type=Path)
    parser.add_argument("--imagenet-dir", type=Path)
    parser.add_argument("--model-size")
    parser.add_argument("--model-id")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--image-size", type=int, nargs=2)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"])
    for name in (
        "teacher-out-channels",
        "max-epochs",
        "max-steps",
        "num-workers",
        "devices",
        "seed",
        "train-batch-size",
        "eval-batch-size",
    ):
        parser.add_argument("--" + name, type=int)
    return parser


def resolved_config(args: argparse.Namespace) -> EfficientAdConfig:
    args.project_root = args.project_root.resolve()
    for name in ("manifest", "config", "pretrained_root", "output"):
        value = getattr(args, name, None)
        if value is not None and not value.is_absolute():
            setattr(args, name, args.project_root / value)
    overrides = {field.name: getattr(args, field.name, None) for field in fields(EfficientAdConfig)}
    return load_efficientad_config(args.config, project_root=args.project_root, overrides=overrides)
