"""Unicode-safe image I/O through encoded bytes.

OpenCV's filename helpers are not reliable for non-ASCII Windows paths. Keeping path handling in
Python and passing bytes to OpenCV works the same way on Windows and Linux.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from industrial_inspection.data.integrity import (
    read_verified_bytes,
    verify_frozen_directory,
    verify_frozen_path,
)


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    payload = read_verified_bytes(path)
    if not payload:
        return None
    try:
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
    except cv2.error:
        return None
    verify_frozen_path(path)
    return decoded


def write_image(path: Path, image: np.ndarray, *, params: list[int] | None = None) -> None:
    suffix = path.suffix.lower() or ".png"
    if suffix == ".jpeg":
        suffix = ".jpg"
    ok, encoded = cv2.imencode(suffix, image, params or [])
    if not ok:
        raise OSError(f"Could not encode image for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def read_color_image(path: Path) -> np.ndarray | None:
    # Manifests, masks and HTTP evidence all use encoded pixel coordinates.
    return read_image(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def read_mask(path: Path) -> np.ndarray | None:
    """Read annotation values, or union LOCO channels for generic binary metrics.

    Individual instance/type PNGs remain intact for the official LOCO evaluator.
    Color masks are rejected rather than silently converted into new label values.
    """
    if path.is_dir():
        verify_frozen_directory(path)
        files = sorted(path.glob("*.png"))
        if not files:
            raise ValueError(f"Empty annotation directory: {path}")
        combined = None
        for file in files:
            channel = read_mask(file)
            if channel is None:
                return None
            if combined is None:
                combined = channel > 0
            elif combined.shape != channel.shape:
                raise ValueError(f"Annotation channels have different dimensions: {path}")
            else:
                combined |= channel > 0
        verify_frozen_directory(path)
        return combined.astype(np.uint8) * 255
    mask = read_image(path, cv2.IMREAD_UNCHANGED)
    if mask is not None and (mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer)):
        raise ValueError(f"Mask must be a 2-D integer image: {path}")
    return mask
