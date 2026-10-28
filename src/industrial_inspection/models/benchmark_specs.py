"""Explicit Anomalib benchmark command specs for cloud runs."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AnomalibCommand:
    model: str
    command: tuple[str, ...]
    output_root: Path


def build_anomalib_command(
    model: str,
    *,
    category: str,
    output_root: Path,
    manifest: Path | None = None,
    project_root: Path | None = None,
    backbone_weights: Path | None = None,
    accelerator: str = "gpu",
    devices: int = 1,
    num_workers: int = 2,
) -> AnomalibCommand:
    """Build a reviewable CLI command; caller decides whether to execute it."""

    model_map = {
        "patchcore": "cloud_patchcore_manifest_fit.py",
        "efficientad": "cloud_efficientad_manifest_fit.py",
        "superadd": "cloud_superadd_manifest_fit.py",
    }
    if model not in model_map:
        raise ValueError(f"Unsupported Anomalib benchmark model: {model}")
    if manifest is None:
        raise ValueError("A frozen --manifest is required; automatic MVTec splits are disabled")
    root = (project_root or Path(__file__).resolve().parents[3]).resolve()
    command = (
        sys.executable,
        str(root / "scripts" / model_map[model]),
        "--manifest", str(manifest),
        "--project-root", str(root),
        "--category",
        category,
        "--accelerator",
        accelerator,
        "--devices",
        str(devices),
        "--num-workers",
        str(num_workers),
        "--output",
        str(output_root),
    )
    if model == "superadd" and backbone_weights is None:
        raise ValueError("SuperADD requires explicit local --backbone-weights; automatic downloads are disabled")
    if model in {"patchcore", "superadd"} and backbone_weights is not None:
        command += ("--backbone-weights", str(backbone_weights))
    return AnomalibCommand(model, command, output_root)


def run_anomalib_command(spec: AnomalibCommand, *, dry_run: bool = False) -> int:
    """Print or execute a reviewed command; no shell interpolation is used."""

    print(" ".join(spec.command))
    if dry_run:
        return 0
    completed = subprocess.run(spec.command, check=False)
    return completed.returncode
