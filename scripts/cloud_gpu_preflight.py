"""Validate a cloud GPU runtime before any dataset/model download.

This script is intentionally dependency-light and can run immediately after a
repository upload. It never installs packages and never downloads assets.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from industrial_inspection.core.json_files import write_new_json  # noqa: E402


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "detail": str(exc), "devices": []}
    devices = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 6:
            devices.append(
                {
                    "index": values[0],
                    "name": values[1],
                    "memory_total_mib": values[2],
                    "memory_free_mib": values[3],
                    "driver_version": values[4],
                    "compute_capability": values[5],
                }
            )
    return {
        "available": completed.returncode == 0 and bool(devices),
        "detail": completed.stderr.strip() if completed.returncode else None,
        "devices": devices,
    }


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {"installed": False, "error": str(exc)}
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            device: dict[str, Any] = {
                "index": index,
                "name": properties.name,
                "total_memory_mib": round(properties.total_memory / 2**20, 1),
                "multi_processor_count": properties.multi_processor_count,
            }
            try:
                free, total = torch.cuda.mem_get_info(index)
                device["free_memory_mib"] = round(free / 2**20, 1)
                device["total_memory_mib_runtime"] = round(total / 2**20, 1)
            except RuntimeError as exc:
                device["memory_error"] = str(exc)
            info["devices"].append(device)
    return info


def _paths(root: Path, create_dirs: bool) -> dict[str, Any]:
    dataset_root = Path(os.getenv("II_DATASET_ROOT", str(root / "assets/datasets"))).expanduser()
    model_root = Path(os.getenv("II_MODEL_ROOT", str(root / "assets/models"))).expanduser()
    artifact_root = Path(
        os.getenv("II_ARTIFACT_ROOT", str(root / "assets/artifacts/industrial-visual-inspection"))
    ).expanduser()
    cache_root = Path(os.getenv("II_CACHE_ROOT", str(root / "assets/cache"))).expanduser()
    values = {
        "project_root": root.resolve(),
        "dataset_root": dataset_root.resolve(),
        "model_root": model_root.resolve(),
        "artifact_root": artifact_root.resolve(),
        "cache_root": cache_root.resolve(),
    }
    if create_dirs:
        for name in ("dataset_root", "model_root", "artifact_root", "cache_root"):
            values[name].mkdir(parents=True, exist_ok=True)
    return {
        name: {"path": str(path), "exists": path.exists(), "writable": os.access(path, os.W_OK)}
        for name, path in values.items()
    }


def build_report(root: Path, *, create_dirs: bool = False) -> dict[str, Any]:
    disk = shutil.disk_usage(root.anchor or root)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "platform": {"system": platform.system(), "release": platform.release()},
        "torch": _torch_info(),
        "nvidia_smi": _nvidia_smi(),
        "packages": {
            "anomalib": _version("anomalib"),
            "lightning": _version("lightning"),
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
            "opencv-python": _version("opencv-python"),
            "scikit-learn": _version("scikit-learn"),
        },
        "paths": _paths(root, create_dirs),
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.getenv("II_CLOUD_ROOT", "/workspace/industrial-visual-inspection")))
    parser.add_argument("--output", type=Path, help="Immutable JSON destination; defaults to a fresh timestamped snapshot")
    parser.add_argument("--create-dirs", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-anomalib", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    report = build_report(root, create_dirs=args.create_dirs)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("reports/environment") / f"gpu_preflight-{stamp}-{uuid.uuid4().hex}.json"
    output = output.expanduser()
    output = (output if output.is_absolute() else root / output).resolve()
    report["report_path"] = str(output)
    write_new_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Preflight report: {output}", file=sys.stderr)

    failures: list[str] = []
    torch = report["torch"]
    if args.require_cuda and not torch.get("cuda_available"):
        failures.append("torch.cuda.is_available() is false")
    if args.require_cuda and not report["nvidia_smi"].get("available"):
        failures.append("nvidia-smi did not report a GPU")
    if args.require_anomalib and report["packages"]["anomalib"] == "missing":
        failures.append("anomalib is not installed")
    if failures:
        print("Preflight failed:", "; ".join(failures), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
