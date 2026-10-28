"""Reusable preprocessing, alignment, ROI, and input quality checks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class QualityReport:
    valid: bool
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Deployment thresholds for the input quality gate."""

    min_width: int = 32
    min_height: int = 32
    min_sharpness: float = 5.0
    brightness_range: tuple[float, float] = (3.0, 252.0)

    def check(self, image: np.ndarray) -> QualityReport:
        return check_image_quality(
            image,
            min_width=self.min_width,
            min_height=self.min_height,
            min_sharpness=self.min_sharpness,
            brightness_range=self.brightness_range,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "min_width": self.min_width,
            "min_height": self.min_height,
            "min_sharpness": self.min_sharpness,
            "brightness_range": list(self.brightness_range),
        }


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to ``(width, height)`` using area interpolation for downsampling."""

    width, height = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def normalize_illumination(image: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Match global mean/std to a reference or use a stable neutral target."""

    source = image.astype(np.float32)
    source_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = (
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if reference is not None
        else np.full_like(source_gray, 128.0)
    )
    source_std = float(source_gray.std())
    if source_std < 1e-6:
        return image.copy()
    normalized = (source - source_gray.mean()) * (target_gray.std() / max(source_std, 1e-6))
    normalized += target_gray.mean()
    return np.clip(normalized, 0, 255).astype(np.uint8)


def align_translation(image: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, bool]:
    """Align a frame to a reference with ECC translation, falling back safely."""

    target = resize_image(image, (reference.shape[1], reference.shape[0]))
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    image_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    try:
        cv2.findTransformECC(ref_gray, image_gray, warp, cv2.MOTION_TRANSLATION, criteria)
        aligned = cv2.warpAffine(
            target,
            warp,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned, True
    except cv2.error:
        return target, False


def crop_roi(image: np.ndarray, roi_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a bounded ``x1,y1,x2,y2`` ROI and reject inverted coordinates."""

    x1, y1, x2, y2 = roi_xyxy
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must be a positive xyxy rectangle")
    height, width = image.shape[:2]
    if x2 > width or y2 > height:
        raise ValueError("ROI exceeds image bounds")
    return image[y1:y2, x1:x2]


def check_image_quality(
    image: np.ndarray,
    *,
    min_width: int = 32,
    min_height: int = 32,
    min_sharpness: float = 5.0,
    brightness_range: tuple[float, float] = (3.0, 252.0),
) -> QualityReport:
    """Return explainable quality flags before model inference."""

    if image is None or image.ndim != 3:
        return QualityReport(False, 0, 0, 0.0, 0.0, 0.0, ("invalid_image",))
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons: list[str] = []
    if width < min_width or height < min_height:
        reasons.append("resolution_too_small")
    if brightness < brightness_range[0] or brightness > brightness_range[1]:
        reasons.append("brightness_out_of_range")
    if sharpness < min_sharpness:
        reasons.append("blurred")
    return QualityReport(not reasons, width, height, brightness, contrast, sharpness, tuple(reasons))
