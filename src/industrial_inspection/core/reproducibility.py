"""Reproducibility helpers shared by training and evaluation."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import warnings
from pathlib import Path

import numpy as np


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and Torch when it is installed."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except (ImportError, OSError):
        return
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except (ImportError, OSError, RuntimeError) as exc:
        # Torch is optional for the CPU-only classical path. A broken optional
        # backend must not make deterministic NumPy/OpenCV experiments unusable.
        warnings.warn(
            f"Torch reproducibility settings were not applied: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file hash so large assets are not loaded into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(project_root: Path) -> str | None:
    """Return the current commit, or ``None`` without Git or a committed repository."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        # Source archives and the minimal runtime image do not require Git.
        return None
    return result.stdout.strip() if result.returncode == 0 else None
