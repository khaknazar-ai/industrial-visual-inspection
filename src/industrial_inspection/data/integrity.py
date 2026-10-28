"""Frozen input identities and verification of the exact bytes consumed by readers."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from industrial_inspection.core.reproducibility import sha256_file

_FROZEN_INPUTS: ContextVar = ContextVar("frozen_dataset_inputs", default=None)


class FrozenInputs:
    """A picklable content snapshot, including directory membership and absent metadata."""

    def __init__(self, paths, *, optional_paths=()):
        self._files = {}
        self._directories = {}
        self._absent = []
        for path in paths:
            path = Path(path).resolve()
            if path.is_dir():
                files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
                self._directories[str(path)] = tuple(sorted(item.relative_to(path).as_posix() for item in files))
                for item in files:
                    if str(item) not in self._files:
                        self._files[str(item)] = sha256_file(item)
            else:
                if str(path) not in self._files:
                    self._files[str(path)] = sha256_file(path)
        for path in optional_paths:
            path = Path(path).resolve()
            if path.exists():
                self._files[str(path)] = sha256_file(path)
            else:
                self._absent.append(str(path))

    def file_hash(self, path: Path) -> str:
        try:
            return self._files[str(Path(path).resolve())]
        except KeyError as exc:
            raise ValueError(f"Dataset input changed after split freeze or is undeclared: {path}") from exc

    def optional_file_hash(self, path: Path) -> str | None:
        return self._files.get(str(path.resolve()))

    def annotation_hash(self, path: Path | None):
        if path is None:
            return None
        path = path.resolve()
        if str(path) in self._directories:
            return {name: self.file_hash(path / name) for name in self._directories[str(path)]}
        return self.file_hash(path)

    def verify_file(self, path: Path) -> str:
        expected = self.file_hash(path)
        try:
            actual = sha256_file(path)
        except OSError as exc:
            raise ValueError(f"Dataset input changed after split freeze: {path}") from exc
        if actual != expected:
            raise ValueError(f"Dataset input changed after split freeze: {path}")
        return expected

    def verify_directory(self, path: Path) -> None:
        path = path.resolve()
        expected = self._directories.get(str(path))
        actual = tuple(sorted(item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()))
        if expected is None or not path.is_dir() or actual != expected:
            raise ValueError(f"Dataset directory changed after split freeze: {path}")

    def verify_unchanged(self) -> None:
        for path in self._absent:
            if Path(path).exists():
                raise ValueError(f"Dataset metadata changed after split freeze: {path}")
        for path in self._directories:
            self.verify_directory(Path(path))
        for path in self._files:
            self.verify_file(Path(path))

    @contextmanager
    def scope(self):
        token = _FROZEN_INPUTS.set(self)
        try:
            yield
        finally:
            _FROZEN_INPUTS.reset(token)


def read_verified_bytes(path: Path) -> bytes:
    snapshot = _FROZEN_INPUTS.get()
    expected = snapshot.file_hash(path) if snapshot is not None else None
    try:
        payload = path.read_bytes()
    except OSError as exc:
        if snapshot is not None:
            raise ValueError(f"Dataset input changed after split freeze: {path}") from exc
        raise
    if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Dataset input changed after split freeze: {path}")
    return payload


def verify_frozen_path(path: Path) -> str | None:
    snapshot = _FROZEN_INPUTS.get()
    return snapshot.verify_file(path) if snapshot is not None else None


def verify_frozen_directory(path: Path) -> None:
    snapshot = _FROZEN_INPUTS.get()
    if snapshot is not None:
        snapshot.verify_directory(path)
