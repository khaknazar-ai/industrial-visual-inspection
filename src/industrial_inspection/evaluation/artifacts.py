"""Append-only experiment directory lifecycle."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read_json_object(path: Path) -> dict[str, Any]:
    """Read finite JSON, including rejection of exponent overflow such as 1e999."""
    value = json.loads(path.read_text(encoding="utf-8"))
    try:
        json.dumps(value, allow_nan=False)
    except ValueError as exc:
        raise ValueError(f"Non-finite JSON in {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


@contextmanager
def publish_files(files: Mapping[Path, Path]):
    """Exclusively publish staged files on one volume; roll back only our links.

    Readers still need the run's completion marker: a group of files is not an
    atomic filesystem operation. Keep staging until this context has finished.
    """
    published: list[tuple[Path, Path, bool]] = []
    try:
        for source, target in files.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
                published.append((source, target, False))
            except PermissionError:
                # Windows can deny hardlinks even when both paths are on the
                # same volume. Rename remains atomic and, on Windows, refuses
                # to replace an existing destination.
                if os.name != "nt" or target.exists():
                    raise
                os.rename(source, target)
                published.append((source, target, True))
        yield
    except BaseException:
        for source, target, moved in reversed(published):
            if moved:
                if target.exists() and not source.exists():
                    os.rename(target, source)
            elif target.exists() and source.samefile(target):
                target.unlink()
        raise


@dataclass(slots=True)
class ArtifactReservation:
    root: Path
    spec_fingerprint: str
    run_id: str

    @classmethod
    def reserve_new(cls, root: Path, *, spec_fingerprint: str) -> ArtifactReservation:
        """Atomically claim a fresh directory; never resume or replace an existing run."""
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=False)
        reservation = cls(root, spec_fingerprint, str(uuid.uuid4()))
        reservation._write("in_progress")
        return reservation

    @classmethod
    def reserve(cls, root: Path, *, spec_fingerprint: str) -> ArtifactReservation:
        root = root.resolve()
        marker = root / "run_status.json"
        if marker.is_file():
            data = json.loads(marker.read_text(encoding="utf-8"))
            if data.get("status") == "complete":
                raise FileExistsError(f"Completed experiment cannot be overwritten: {root}")
            if data.get("spec_fingerprint") != spec_fingerprint:
                raise ValueError(f"Incomplete experiment has a different spec: {root}")
            run_id = str(data.get("run_id") or uuid.uuid4())
        else:
            root.mkdir(parents=True, exist_ok=False)
            run_id = str(uuid.uuid4())
        reservation = cls(root, spec_fingerprint, run_id)
        reservation._write("in_progress")
        return reservation

    def _write(self, status: str, **extra: Any) -> None:
        payload = {
            "run_id": self.run_id,
            "status": status,
            "spec_fingerprint": self.spec_fingerprint,
            "updated_at": datetime.now(UTC).isoformat(),
            **extra,
        }
        temporary = self.root / f".run_status.{uuid.uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.root / "run_status.json")

    def complete(self, *, artifacts: list[str] | None = None) -> None:
        self._write("complete", artifacts=sorted(artifacts or []))

    def fail(self, reason: str) -> None:
        self._write("failed", reason=reason)
