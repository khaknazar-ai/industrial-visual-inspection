"""Inference runtime for a trusted Anomalib EfficientAD checkpoint."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from industrial_inspection.models.base import (
    AnomalyPrediction,
    AnomalyRegion,
    ImageInput,
    load_color_image,
)
from industrial_inspection.models.efficientad_adapter import build_efficientad
from industrial_inspection.models.efficientad_training import (
    CHECKPOINT_FORMAT,
    SCORE_SPACE,
    validate_fitted_state,
)
from industrial_inspection.scoring.pixels import InvalidPredictionError, finalize_prediction


class EfficientAdRuntime:
    """Inference-only wrapper for a fitted Anomalib EfficientAD checkpoint."""

    score_space = SCORE_SPACE
    calibration_status = "uncalibrated"
    requires_calibration_bundle = True

    def __init__(self, checkpoint: Path, *, device: str = "auto") -> None:
        self.checkpoint = checkpoint.resolve()
        self._torch: Any
        self._model: Any
        self.device: Any
        self._load(device)

    def _load(self, device: str) -> None:
        try:
            import torch
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "EfficientAD requires a compatible anomalib, lightning, torch, and torchvision installation."
            ) from exc
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"EfficientAD checkpoint not found: {self.checkpoint}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("EfficientAD device must be 'auto', 'cpu', or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for EfficientAD but it is unavailable")
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Unsupported EfficientAD checkpoint; use the reviewed manifest runner")
        if checkpoint.get("score_space") != self.score_space:
            raise ValueError("EfficientAD checkpoint score space mismatch")
        model = build_efficientad(checkpoint["model_spec"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        validate_fitted_state(model)
        self.model_id = checkpoint["model_id"]
        self.model_spec = checkpoint["model_spec"]
        self._torch = torch
        self.device = torch.device(device)
        self._model = model.to(self.device).eval()

    @staticmethod
    def _first_array(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = value.detach().cpu().numpy()
        while array.ndim > 2:
            array = array[0]
        return np.asarray(array)

    @staticmethod
    def _regions(mask: np.ndarray, anomaly_map: np.ndarray) -> list[AnomalyRegion]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions: list[AnomalyRegion] = []
        for component in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[component])
            if area < 12:
                continue
            values = anomaly_map[labels == component]
            regions.append(
                AnomalyRegion(
                    bbox_xyxy=(x, y, x + width, y + height),
                    area=area,
                    region_score=float(values.max()) if values.size else 0.0,
                )
            )
        return sorted(regions, key=lambda region: region.region_score, reverse=True)

    def predict(self, image: ImageInput) -> AnomalyPrediction:
        started = time.perf_counter()
        decoded = load_color_image(image)
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        tensor = (
            self._torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        with self._torch.inference_mode():
            output = self._model(tensor)
        score = self._first_array(output.pred_score)
        anomaly_map = self._first_array(output.anomaly_map)
        if score is None or score.size == 0 or anomaly_map is None:
            raise RuntimeError("EfficientAD returned incomplete prediction output")
        if not np.isfinite(score).all() or not np.isfinite(anomaly_map).all():
            raise InvalidPredictionError("EfficientAD returned non-finite scores/maps")
        anomaly_map = cv2.resize(
            anomaly_map.astype(np.float32, copy=False), (decoded.shape[1], decoded.shape[0])
        )
        raw_mask = self._first_array(getattr(output, "pred_mask", None))
        binary_mask = cv2.resize((raw_mask > 0).astype(np.uint8) * 255, (decoded.shape[1], decoded.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) if raw_mask is not None else None
        return finalize_prediction(self, AnomalyPrediction(
            image_score=float(score.reshape(-1)[0]),
            anomaly_map=anomaly_map,
            binary_mask=binary_mask,
            regions=self._regions(binary_mask, anomaly_map) if binary_mask is not None else [],
            metadata={
                "model_family": "anomalib_efficientad",
                "checkpoint": self.checkpoint.name,
                "device": str(self.device),
                "score_space": self.score_space,
                "calibration_status": self.calibration_status,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        ), decoded.shape[:2])

    def predict_batch(self, images: Sequence[ImageInput]) -> list[AnomalyPrediction]:
        return [self.predict(image) for image in images]

    def fit(self, images: Sequence[ImageInput]) -> None:
        raise RuntimeError("EfficientAD runtime is inference-only; train with the manifest runner")

    def save(self, path: Path) -> None:
        raise RuntimeError("EfficientAD runtime does not overwrite the source checkpoint")
