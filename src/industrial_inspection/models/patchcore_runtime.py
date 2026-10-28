"""Local inference runtime for Anomalib Lightning PatchCore checkpoints."""

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
from industrial_inspection.models.geometry import restore_map_geometry
from industrial_inspection.scoring.pixels import InvalidPredictionError, finalize_prediction
from industrial_inspection.scoring.policy import DecisionThresholds


class PatchCoreRuntime:
    """Inference-only wrapper for a trusted Anomalib Lightning checkpoint."""

    score_space = "anomalib_postprocessed_v1"
    calibration_status = "legacy_exploratory"

    def __init__(self, checkpoint: Path, *, device: str = "auto") -> None:
        self.checkpoint = checkpoint.resolve()
        self._torch: Any
        self._model: Any
        self.device: Any
        self._load(device)
        self.model_id = f"patchcore-{self.checkpoint.stem}"

    def _load(self, device: str) -> None:
        try:
            import torch
            from anomalib import PrecisionType
            from anomalib.models import Patchcore
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "PatchCore requires a compatible anomalib, lightning, torch, and torchvision installation."
            ) from exc
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"PatchCore checkpoint not found: {self.checkpoint}")
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("PatchCore device must be 'auto', 'cpu', or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for PatchCore but is not available")
        self.device = torch.device(device)
        with torch.serialization.safe_globals([PrecisionType]):
            checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(f"Unsupported PatchCore checkpoint format: {self.checkpoint}")
        hyper_parameters = checkpoint.get("hyper_parameters", {})
        pre_processor = hyper_parameters.get("pre_processor", True)
        if checkpoint.get("format") == "industrial_patchcore_inference_v1":
            image_size = hyper_parameters.get("image_size")
            if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
                    or any(type(value) is not int or value < 32 for value in image_size)):
                raise ValueError("PatchCore inference checkpoint requires valid image_size")
            pre_processor = Patchcore.configure_pre_processor(
                image_size=tuple(image_size), center_crop_size=tuple(image_size),
            )
        model = Patchcore(
            backbone=hyper_parameters.get("backbone", "wide_resnet50_2"),
            layers=tuple(hyper_parameters.get("layers", ("layer2", "layer3"))),
            # The checkpoint already contains the pretrained backbone weights.
            pre_trained=False,
            coreset_sampling_ratio=float(hyper_parameters.get("coreset_sampling_ratio", 0.1)),
            num_neighbors=int(hyper_parameters.get("num_neighbors", 9)),
            precision=hyper_parameters.get("precision", "float32"),
            pre_processor=pre_processor,
            post_processor=hyper_parameters.get("post_processor", True),
        )
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if missing:
            raise ValueError(f"PatchCore checkpoint is missing state: {missing[:8]}")
        memory_bank = getattr(getattr(model, "model", None), "memory_bank", None)
        if memory_bank is None or not torch.is_tensor(memory_bank) or memory_bank.numel() == 0:
            raise ValueError("PatchCore checkpoint does not contain a fitted memory bank")
        self._unexpected_state = list(unexpected)
        self._model = model.to(self.device).eval()

    @property
    def default_thresholds(self) -> DecisionThresholds:
        # Anomalib normalizes scores around its calibrated F1 threshold to 0.5.
        return DecisionThresholds(
            pass_threshold=0.5,
            fail_threshold=float(np.nextafter(0.5, 1.0)),
            source="anomalib_post_processor",
            objective="image_f1",
        )

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
            raise RuntimeError("PatchCore returned incomplete prediction output")
        anomaly_map = anomaly_map.astype(np.float32, copy=False)
        if not np.isfinite(score).all() or not np.isfinite(anomaly_map).all():
            raise InvalidPredictionError("PatchCore returned non-finite scores/maps")
        transform = getattr(getattr(self._model, "pre_processor", None), "transform", None)
        anomaly_map, support = restore_map_geometry(anomaly_map, decoded.shape[:2], transform)
        raw_mask = self._first_array(output.pred_mask)
        # Bound bundles replace this exploratory native threshold in finalize_prediction.
        binary_mask = ((anomaly_map >= 0.5) & support).astype(np.uint8) * 255 if raw_mask is not None else None
        return finalize_prediction(self, AnomalyPrediction(
            image_score=float(score.reshape(-1)[0]),
            anomaly_map=anomaly_map,
            binary_mask=binary_mask,
            valid_mask=support,
            regions=self._regions(binary_mask, anomaly_map) if binary_mask is not None else [],
            metadata={
                "model_family": "anomalib_patchcore",
                "checkpoint": self.checkpoint.name,
                "device": str(self.device),
                "unexpected_state_keys": self._unexpected_state,
                "geometry": "original_coordinates_bilinear_map_nearest_mask",
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        ), decoded.shape[:2])

    def predict_batch(self, images: Sequence[ImageInput]) -> list[AnomalyPrediction]:
        return [self.predict(image) for image in images]

    def fit(self, images: Sequence[ImageInput]) -> None:
        raise RuntimeError("PatchCore runtime is inference-only; train with the cloud benchmark")

    def save(self, path: Path) -> None:
        raise RuntimeError("PatchCore runtime does not overwrite the source checkpoint")
