"""One original-coordinate pixel policy for runtime, evaluation and evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from industrial_inspection.models.base import AnomalyPrediction, AnomalyRegion


class InvalidPredictionError(ValueError):
    """A model returned an output that cannot support an inspection decision."""


@dataclass(frozen=True, slots=True)
class PixelPolicy:
    """Version 1 retains >=; version 2 excludes values tied with the threshold."""
    threshold: float | None
    morphology_kernel: int = 0
    min_component_area: int = 1
    version: int = 1

    def __post_init__(self):
        if self.threshold is not None and (isinstance(self.threshold, bool) or not np.isfinite(self.threshold)):
            raise ValueError("Pixel threshold must be finite or null")
        if type(self.version) is not int or self.version not in {1, 2}:
            raise ValueError("Unsupported pixel policy version")
        if type(self.morphology_kernel) is not int or self.morphology_kernel < 0 or (self.morphology_kernel > 0 and self.morphology_kernel % 2 == 0):
            raise ValueError("Morphology kernel must be zero or a positive odd integer")
        if type(self.min_component_area) is not int or self.min_component_area < 1:
            raise ValueError("Minimum component area must be a positive integer")

    @classmethod
    def from_dict(cls, value):
        return cls(**(value or {"threshold": None}))

    def as_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        content = json.dumps(self.as_dict(), sort_keys=True, allow_nan=False, separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()

    def apply(self, prediction: AnomalyPrediction) -> AnomalyPrediction:
        response = prediction.anomaly_map
        if response is not None and (response.ndim != 2 or not np.isfinite(response).all()):
            raise InvalidPredictionError("Pixel policy requires a finite 2D anomaly map")
        mask = None
        regions = []
        if self.threshold is not None:
            if response is None:
                raise InvalidPredictionError("Calibrated pixel policy requires an anomaly map")
            # V2 compares in calibration precision: NumPy scalar promotion must not
            # round the float64 quantile to a lower-precision map boundary.
            positive = (response >= self.threshold if self.version == 1
                        else np.asarray(response, dtype=np.float64) > self.threshold)
            mask = positive.astype(np.uint8) * 255
            if prediction.valid_mask is not None:
                mask[~prediction.valid_mask.astype(bool)] = 0
            if self.morphology_kernel:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morphology_kernel,) * 2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                if prediction.valid_mask is not None:
                    mask[~prediction.valid_mask.astype(bool)] = 0
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            keep = np.zeros(count, dtype=np.uint8)
            for component in range(1, count):
                x, y, width, height, area = map(int, stats[component])
                if area >= self.min_component_area:
                    keep[component] = 255
                    regions.append(AnomalyRegion((x, y, x + width, y + height), area,
                                                 float(response[labels == component].max())))
            mask = keep[labels]
            regions.sort(key=lambda region: region.region_score, reverse=True)
        return AnomalyPrediction(
            prediction.image_score, response, mask, regions,
            {**prediction.metadata, "pixel_policy": self.as_dict(), "pixel_policy_id": self.fingerprint},
            prediction.valid_mask,
        )


def finalize_prediction(model, prediction: AnomalyPrediction, original_shape) -> AnomalyPrediction:
    """Validate geometry and finite output before applying a bound calibration."""
    if prediction.image_score is None or not np.isfinite(prediction.image_score):
        raise InvalidPredictionError("Model returned a non-finite image score")
    for name in ("anomaly_map", "binary_mask", "valid_mask"):
        array = getattr(prediction, name)
        if array is not None and (array.shape != tuple(original_shape) or not np.isfinite(array).all()):
            raise InvalidPredictionError(f"Model {name} must be finite and in original image coordinates")
    policy = getattr(model, "pixel_policy", None)
    if policy is not None:
        prediction = policy.apply(prediction)
    prediction.metadata = {**prediction.metadata, "geometry": "original_coordinates",
                           "score_space": getattr(model, "score_space", "raw_distance")}
    if prediction.valid_mask is not None:
        prediction.metadata["pixel_valid_fraction"] = float(np.count_nonzero(prediction.valid_mask) / prediction.valid_mask.size)
    height, width = original_shape
    for region in prediction.regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height) or not np.isfinite(region.region_score) or region.area < 1:
            raise InvalidPredictionError("Model returned invalid region geometry or score")
    try:
        json.dumps(prediction.metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InvalidPredictionError("Model returned non-finite or non-JSON metadata") from exc
    return prediction


def calibrated_pixel_policy(model, threshold):
    config = getattr(model, "config", None)
    return PixelPolicy(threshold, morphology_kernel=getattr(config, "morphology_kernel", 0),
                       min_component_area=getattr(config, "min_component_area", 1), version=2)
