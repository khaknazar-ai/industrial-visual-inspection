"""Portable project and storage path resolution."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from industrial_inspection.core.runtime_resources import source_project_root


def default_runtime_root() -> Path:
    """Use the source root or per-user writable storage for an installed wheel."""
    override = os.getenv("II_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    source_root = source_project_root()
    if source_root is not None:
        return source_root
    if sys.platform == "win32":
        base = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    return (base.expanduser() / "industrial-visual-inspection").resolve()


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent containing ``pyproject.toml``.

    ``II_PROJECT_ROOT`` wins when set, which makes services and Docker runs
    independent from the current working directory.
    """

    override = os.getenv("II_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    current = (start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find pyproject.toml from {current}. Set II_PROJECT_ROOT explicitly."
    )


def resolve_from_root(value: str | Path, project_root: Path) -> Path:
    """Resolve a possibly relative path against the project root."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


@dataclass(frozen=True, slots=True)
class StoragePaths:
    """Canonical locations for datasets, model assets, and generated artifacts."""

    project_root: Path
    dataset_root: Path
    model_root: Path
    artifact_root: Path

    @classmethod
    def from_values(
        cls,
        project_root: Path,
        dataset_root: str | Path = "data/external",
        model_root: str | Path = "models",
        artifact_root: str | Path = "reports/runtime",
    ) -> StoragePaths:
        dataset = os.getenv("II_DATASET_ROOT", str(dataset_root))
        models = os.getenv("II_MODEL_ROOT", str(model_root))
        artifacts = os.getenv("II_ARTIFACT_ROOT", str(artifact_root))
        return cls(
            project_root=project_root.resolve(),
            dataset_root=resolve_from_root(dataset, project_root),
            model_root=resolve_from_root(models, project_root),
            artifact_root=resolve_from_root(artifacts, project_root),
        )

    def ensure_runtime_directories(self) -> None:
        """Create only small, writable runtime directories; never dataset/model roots."""

        self.artifact_root.mkdir(parents=True, exist_ok=True)
