"""Read-only resources supplied by the wheel or the active source checkout."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path, PurePosixPath


def source_project_root() -> Path | None:
    """Recognize our source layout; never search for an unrelated checkout."""
    root = Path(__file__).resolve().parents[3]
    if (root / "pyproject.toml").is_file() and (root / "src/industrial_inspection").resolve() == Path(__file__).resolve().parents[1]:
        return root
    return None


def runtime_resource_path(relative: str) -> Path:
    """Return an installed filesystem resource, independent of writable roots."""
    name = PurePosixPath(relative)
    if not name.parts or name.parts[0] not in {"configs", "web"} or name.is_absolute() or ".." in name.parts or "\\" in relative or ":" in relative:
        raise ValueError("Runtime resource name must stay within the resource directory")
    resource = files("industrial_inspection").joinpath("_resources", *name.parts)
    if resource.is_file() or resource.is_dir():
        return Path(os.fspath(resource)).resolve()
    source_root = source_project_root()
    if source_root is not None:
        candidate = source_root.joinpath(*name.parts)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing packaged runtime resource: {relative}")
