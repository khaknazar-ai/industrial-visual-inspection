"""Publish complete finite JSON documents without replacing previous evidence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def write_new_json(output: Path, value) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".json-", dir=output.parent) as directory:
        staged = Path(directory) / "document.json"
        staged.write_text(encoded, encoding="utf-8")
        try:
            os.link(staged, output)
        except PermissionError:
            if os.name != "nt":
                raise
            # NTFS rename is atomic and, unlike POSIX rename, refuses to replace
            # an existing destination. Keep the create-only publication contract
            # when a Windows policy denies hard links.
            os.rename(staged, output)
