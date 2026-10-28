"""Recorded seeds, worker initialization and code identity for deep fit runs."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

from industrial_inspection.core.reproducibility import git_revision, seed_everything, sha256_file


def seed_training(seed: int, *, lightning_workers: bool = True) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(seed)
    try:
        import torch
    except (ImportError, OSError):
        if lightning_workers:
            raise
        return {"seed": seed, "python_random": True, "numpy": True, "torch": False,
                "lightning_worker_seeding": False, "note": "Optional Torch backend unavailable"}
    if lightning_workers:
        from lightning.pytorch import seed_everything as seed_lightning
        seed_lightning(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": seed, "python_random": True, "numpy": True, "torch": True,
        "lightning_worker_seeding": lightning_workers,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "python_hash_seed_note": "PYTHONHASHSEED applies to child processes; current interpreter hash order is not claimed",
    }


def training_environment(project_root: Path, *, include_git: bool = True) -> dict:
    """Record source identity; allow an explicit archive/no-VCS workflow."""
    if type(include_git) is not bool:
        raise ValueError("include_git must be an explicit boolean")
    package_root = Path(__file__).resolve().parents[1]
    versions = {}
    for name in ("torch", "torchvision", "anomalib", "lightning", "numpy", "opencv-python", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "git_revision": git_revision(project_root) if include_git else None,
        "git_provenance": {
            "mode": "git_and_source_hashes" if include_git else "source_hashes_only",
            "git_access_attempted": include_git,
            "omission_reason": None if include_git else "explicit source-hashes-only request",
        },
        "source_sha256": {
            f"industrial_inspection/{path.relative_to(package_root).as_posix()}": sha256_file(path)
            for path in sorted(package_root.rglob("*.py"))
        },
        "runner_sha256": {path.name: sha256_file(path) for path in sorted((project_root / "scripts").glob("*.py"))},
        "versions": versions,
    }
