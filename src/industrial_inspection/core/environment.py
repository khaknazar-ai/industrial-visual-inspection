"""Environment reconnaissance that never installs or downloads anything."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from industrial_inspection.core.json_files import write_new_json
from industrial_inspection.core.paths import StoragePaths, find_project_root
from industrial_inspection.core.reproducibility import git_revision

PACKAGE_NAMES = (
    "numpy",
    "opencv-python",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "torch",
    "torchvision",
    "anomalib",
    "fastapi",
    "uvicorn",
    "pydantic",
    "PyYAML",
    "pytest",
    "ruff",
    "psutil",
    "Pillow",
    "httpx",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def _torch_details() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {"version": "missing", "error": str(exc)}
    cuda_available = bool(torch.cuda.is_available())
    details: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device_name": None,
    }
    if cuda_available:
        try:
            details["device_name"] = torch.cuda.get_device_name(0)
            details["device_count"] = torch.cuda.device_count()
        except RuntimeError as exc:
            details["device_error"] = str(exc)
    return details


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "detail": str(exc)}
    return {
        "available": result.returncode == 0,
        "detail": result.stdout.strip() or result.stderr.strip(),
    }


def _asset_location(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "top_level_entries": 0}
    try:
        count = sum(1 for _ in path.iterdir()) if path.is_dir() else 1
    except PermissionError:
        count = None
    return {"path": str(path), "exists": True, "top_level_entries": count}


def inspect_environment(project_root: Path | None = None) -> dict[str, Any]:
    """Collect a JSON-serializable snapshot without mutating the environment."""

    root = (project_root or find_project_root()).resolve()
    storage = StoragePaths.from_values(root)
    disk = shutil.disk_usage(root.anchor)
    cache_variables = {
        name: os.getenv(name, "unset")
        for name in (
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "TORCH_HOME",
            "PIP_CACHE_DIR",
            "XDG_CACHE_HOME",
            "TEMP",
            "TMP",
        )
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {"system": platform.system(), "release": platform.release()},
        "torch": _torch_details(),
        "nvidia_smi": _nvidia_smi(),
        "packages": _package_versions(),
        "storage": {
            "project_root": str(root),
            "dataset_root": str(storage.dataset_root),
            "model_root": str(storage.model_root),
            "artifact_root": str(storage.artifact_root),
            "drive_total_bytes": disk.total,
            "drive_free_bytes": disk.free,
            "cache_environment": cache_variables,
        },
        "asset_inventory": {
            "datasets": _asset_location(storage.dataset_root),
            "models": _asset_location(storage.model_root),
            "project_data": _asset_location(root / "data"),
        },
        "git": {"revision": git_revision(root)},
    }


def write_environment_report(output: Path, project_root: Path | None = None) -> dict[str, Any]:
    """Write the environment snapshot and return it."""

    report = inspect_environment(project_root)
    write_new_json(output, report)
    return report
