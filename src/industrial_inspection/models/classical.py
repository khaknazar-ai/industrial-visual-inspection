"""Deterministic alignment-and-difference inspection baseline."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from industrial_inspection.data.image_io import read_color_image, write_image
from industrial_inspection.models.base import (
    AnomalyPrediction,
    AnomalyRegion,
    ImageInput,
    load_color_image,
)
from industrial_inspection.scoring.pixels import finalize_prediction


@dataclass(frozen=True, slots=True)
class ClassicalConfig:
    align: bool = True
    normalize_illumination: bool = True
    blur_kernel: int = 5
    pixel_threshold: float = 0.10
    morphology_kernel: int = 3
    min_component_area: int = 12
    top_k_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer")
        if self.morphology_kernel < 0:
            raise ValueError("morphology_kernel cannot be negative")
        if not 0 < self.pixel_threshold < 1:
            raise ValueError("pixel_threshold must be in (0, 1)")
        if not 0 < self.top_k_fraction <= 1:
            raise ValueError("top_k_fraction must be in (0, 1]")


class ClassicalDifferenceModel:
    """Reference-median baseline with optional ECC translation alignment."""

    def __init__(
        self,
        config: ClassicalConfig | None = None,
        model_id: str = "classical-diff-v1",
    ) -> None:
        self.config = config or ClassicalConfig()
        self.model_id = model_id
        self.reference: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.reference is not None

    def fit(self, images: Sequence[ImageInput]) -> None:
        if not images:
            raise ValueError("At least one normal image is required")
        decoded = [load_color_image(image) for image in images]
        target_height, target_width = decoded[0].shape[:2]
        resized = [
            cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
            if image.shape[:2] != (target_height, target_width)
            else image
            for image in decoded
        ]
        # A pixel-wise median is resistant to isolated capture noise in normal images.
        self.reference = np.median(np.stack(resized).astype(np.float32), axis=0).astype(np.uint8)

    def _align(self, image: np.ndarray) -> tuple[np.ndarray, bool, np.ndarray]:
        assert self.reference is not None
        target_height, target_width = self.reference.shape[:2]
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        if not self.config.align:
            return resized, False, np.eye(2, 3, dtype=np.float32)
        reference_gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        image_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        try:
            cv2.findTransformECC(reference_gray, image_gray, warp, cv2.MOTION_TRANSLATION, criteria)
            aligned = cv2.warpAffine(
                resized,
                warp,
                (target_width, target_height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            return aligned, True, warp
        except cv2.error:
            # Failed registration is evidence, not a reason to drop an inspection.
            return resized, False, np.eye(2, 3, dtype=np.float32)

    def _normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        assert self.reference is not None
        if not self.config.normalize_illumination:
            return image
        source = image.astype(np.float32)
        source_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        source_mean, source_std = float(source_gray.mean()), float(source_gray.std())
        target_mean, target_std = float(reference_gray.mean()), float(reference_gray.std())
        if source_std < 1e-6:
            return image
        normalized = (source - source_mean) * (target_std / source_std) + target_mean
        return np.clip(normalized, 0, 255).astype(np.uint8)

    def _regions(self, mask: np.ndarray, response: np.ndarray) -> list[AnomalyRegion]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions: list[AnomalyRegion] = []
        for component in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[component])
            if area < self.config.min_component_area:
                continue
            component_values = response[labels == component]
            regions.append(
                AnomalyRegion(
                    bbox_xyxy=(x, y, x + width, y + height),
                    area=area,
                    region_score=float(component_values.max(initial=0.0)),
                )
            )
        return sorted(regions, key=lambda region: region.region_score, reverse=True)

    def predict(self, image: ImageInput) -> AnomalyPrediction:
        if self.reference is None:
            raise RuntimeError("Model must be fitted or loaded before prediction")
        started = time.perf_counter()
        decoded = load_color_image(image)
        aligned, alignment_succeeded, warp = self._align(decoded)
        normalized = self._normalize_illumination(aligned)
        difference = cv2.absdiff(normalized, self.reference)
        response = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        response = cv2.GaussianBlur(response, (self.config.blur_kernel, self.config.blur_kernel), 0)
        raw_mask = (response >= self.config.pixel_threshold).astype(np.uint8) * 255
        if self.config.morphology_kernel > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.morphology_kernel, self.config.morphology_kernel),
            )
            raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
            raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        regions = self._regions(raw_mask, response)
        cleaned_mask = np.zeros_like(raw_mask)
        for region in regions:
            x1, y1, x2, y2 = region.bbox_xyxy
            cleaned_mask[y1:y2, x1:x2] = np.maximum(
                cleaned_mask[y1:y2, x1:x2], raw_mask[y1:y2, x1:x2]
            )
        flat = response.reshape(-1)
        top_k = max(1, int(round(flat.size * self.config.top_k_fraction)))
        image_score = float(np.partition(flat, -top_k)[-top_k:].mean())
        # ECC maps input to reference with WARP_INVERSE_MAP; undo it before the resize.
        height, width = response.shape
        support = cv2.warpAffine(np.ones_like(response), warp, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= 1 - 1e-6
        response = cv2.warpAffine(response, warp, (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cleaned_mask = cv2.warpAffine(cleaned_mask, warp, (width, height), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        original_size = (decoded.shape[1], decoded.shape[0])
        response = cv2.resize(response, original_size, interpolation=cv2.INTER_LINEAR)
        cleaned_mask = cv2.resize(cleaned_mask, original_size, interpolation=cv2.INTER_NEAREST)
        support = cv2.resize(support.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        regions = self._regions(cleaned_mask, response)
        return finalize_prediction(self, AnomalyPrediction(
            image_score=image_score,
            anomaly_map=response,
            binary_mask=cleaned_mask,
            regions=regions,
            valid_mask=support,
            metadata={
                "model_family": "classical",
                "alignment_requested": self.config.align,
                "alignment_succeeded": alignment_succeeded,
                "reference_to_resized_input": warp.tolist(),
                "inverse_geometry": "forward ECC translation, bilinear map, nearest mask, zero outside support",
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        ), decoded.shape[:2])

    def predict_batch(self, images: Sequence[ImageInput]) -> list[AnomalyPrediction]:
        return [self.predict(image) for image in images]

    def save(self, path: Path) -> None:
        if self.reference is None:
            raise RuntimeError("Cannot save an unfitted model")
        path.mkdir(parents=True, exist_ok=True)
        write_image(path / "reference.png", self.reference)
        metadata = {"model_id": self.model_id, "config": asdict(self.config), "format_version": 1}
        (path / "model.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> ClassicalDifferenceModel:
        metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported classical model format")
        model = cls(ClassicalConfig(**metadata["config"]), metadata["model_id"])
        model.reference = read_color_image(path / "reference.png")
        if model.reference is None:
            raise ValueError(f"Model reference is missing or corrupt in {path}")
        return model
