"""End-to-end inspection orchestration with a separate decision policy."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from industrial_inspection.core.config import AppConfig
from industrial_inspection.inference.evidence import EvidenceStore
from industrial_inspection.inference.schemas import (
    EvidenceSchema,
    InspectionResponse,
    RegionSchema,
    TimingSchema,
)
from industrial_inspection.models.artifacts import ModelBundleError
from industrial_inspection.models.base import AnomalyModel, AnomalyPrediction, load_color_image
from industrial_inspection.models.registry import load_bundle_model, load_model
from industrial_inspection.preprocessing import QualityPolicy
from industrial_inspection.scoring.pixels import InvalidPredictionError, finalize_prediction
from industrial_inspection.scoring.policy import DecisionThresholds, InspectionDecision, decide


def _load_thresholds(path: Path) -> DecisionThresholds:
    """Read the decision fields from a saved threshold report."""

    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("pass_threshold", "fail_threshold")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Threshold report is missing fields: {', '.join(missing)}")
    values = {
        name: data[name]
        for name in ("pass_threshold", "fail_threshold", "source", "objective")
        if name in data
    }
    return DecisionThresholds(**values)


def _threshold_path(model_path: Path) -> Path:
    """Return the conventional sidecar path for a model directory or file."""

    if model_path.is_dir():
        return model_path / "thresholds.json"
    return model_path.with_name("thresholds.json")


class InspectionPipeline:
    """A loaded model, calibrated validation thresholds, and evidence store."""

    def __init__(
        self,
        model: AnomalyModel,
        thresholds: DecisionThresholds,
        artifact_root: Path,
        *,
        category: str = "assembly",
        quality_policy: QualityPolicy | None = None,
    ) -> None:
        self.model = model
        self.thresholds = thresholds
        self.category = category
        if getattr(model, "category", category) != category:
            raise ModelBundleError("Pipeline category differs from its model")
        self._model_lock = threading.Lock()
        self.evidence = EvidenceStore(artifact_root)
        self.score_space = getattr(model, "score_space", "raw_distance")
        self.calibration_id = getattr(model, "calibration_id", "runtime-thresholds")
        self.calibration_status = getattr(model, "calibration_status", "legacy_exploratory")
        bound_thresholds = getattr(model, "decision_thresholds", None)
        bound_policy = getattr(model, "bundle", None) is not None or getattr(model, "bound_native_thresholds", False)
        if bound_policy and bound_thresholds is not None and thresholds != bound_thresholds:
            raise ModelBundleError("Threshold override requires a new calibration bundle")
        self.quality_policy = quality_policy or QualityPolicy()

    def validate_category(self, category: str | None) -> str:
        if category is not None and category != self.category:
            raise ValueError(f"Loaded model supports category {self.category!r}, not {category!r}")
        return self.category

    def identity(self):
        bundle = getattr(self.model, "bundle", None)
        pixel = getattr(self.model, "pixel_policy", None)
        runtime = getattr(self.model, "runtime_identity", None)
        policy = {"image": self.thresholds.as_dict(), "pixel": pixel.as_dict() if pixel else None,
                  "quality": self.quality_policy.as_dict()}
        policy_id = hashlib.sha256(json.dumps(policy, sort_keys=True, allow_nan=False).encode()).hexdigest()
        identity = {
            "model_id": self.model.model_id, "category": self.category,
            "model_fingerprint": bundle.model_fingerprint if bundle else None,
            "weights_sha256": bundle.model["weights"]["sha256"] if bundle else None,
            "preprocessing_fingerprint": bundle.model["preprocessing"].get("fingerprint") if bundle else None,
            "runtime_fingerprint": runtime["fingerprint"] if runtime else None,
            "runtime_bound_to_calibration": bool(bundle and "runtime_identity" in bundle.model),
            "calibration_id": self.calibration_id, "calibration_status": self.calibration_status,
            "score_space": self.score_space, "policy_id": policy_id, "policy": policy,
            "identity_scope": "bound_bundle" if bundle else "unbound_legacy",
        }
        if bundle is None and getattr(self.model, "bound_native_thresholds", False):
            identity.update({
                "model_fingerprint": self.model.model_fingerprint,
                "weights_sha256": self.model.checkpoint_sha256,
                "native_thresholds_bound": True,
                "identity_scope": "native_checkpoint",
            })
        return identity

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        model_name: str | None = None,
        model_path: Path | None = None,
        category: str | None = None,
    ) -> InspectionPipeline:
        storage = config.storage_paths()
        evaluation = config.evaluation
        thresholds = None
        if evaluation.pass_threshold is not None:
            thresholds = DecisionThresholds(
                evaluation.pass_threshold, evaluation.fail_threshold,
                source=evaluation.threshold_source or "deployment_config",
                objective=evaluation.threshold_objective or evaluation.optimize_metric,
            )
        return cls.load(
            model_name or config.model.name,
            model_path or storage.model_root / config.model.model_id,
            storage.artifact_root,
            category=category or config.data.category,
            thresholds=thresholds,
            device=config.runtime.device,
            quality_policy=quality_policy_from_config(config),
        )

    @classmethod
    def load(
        cls,
        model_name: str,
        model_path: Path,
        artifact_root: Path,
        *,
        category: str = "assembly",
        thresholds: DecisionThresholds | None = None,
        device: str = "auto",
        quality_policy: QualityPolicy | None = None,
    ) -> InspectionPipeline:
        bundle = None
        if model_path.is_dir() and (model_path / "calibration.json").is_file():
            model, bundle = load_bundle_model(model_name, model_path, category=category, device=device)
        else:
            model = load_model(model_name, model_path, device=device)
        if getattr(model, "requires_calibration_bundle", False) and (
            bundle is None or bundle.calibration_status != "validated"
        ):
            raise ValueError("This model requires a validated calibration bundle for decisions")
        selected_thresholds = thresholds
        if bundle is not None:
            if thresholds is not None and thresholds != model.decision_thresholds:
                raise ModelBundleError("Threshold override requires a new calibration bundle")
            selected_thresholds = model.decision_thresholds
            model_path = bundle.artifact_path
        threshold_path = _threshold_path(model_path)
        if selected_thresholds is None and threshold_path.is_file():
            selected_thresholds = _load_thresholds(threshold_path)
        if selected_thresholds is None and hasattr(model, "default_thresholds"):
            selected_thresholds = model.default_thresholds
        if selected_thresholds is None:
            raise FileNotFoundError(f"Thresholds not found at {threshold_path}")
        return cls(
            model,
            selected_thresholds,
            artifact_root,
            category=category,
            quality_policy=quality_policy,
        )

    def inspect(
        self,
        image: np.ndarray,
        *,
        category: str | None = None,
        original_bytes: bytes | None = None,
    ) -> InspectionResponse:
        total_started = time.perf_counter()
        category = self.validate_category(category)
        decoded = load_color_image(image)
        preprocess_started = time.perf_counter()
        quality = self.quality_policy.check(decoded)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        model_started = time.perf_counter()
        try:
            with self._model_lock:
                prediction = finalize_prediction(self.model, self.model.predict(decoded), decoded.shape[:2])
        except InvalidPredictionError as exc:
            prediction = AnomalyPrediction(None, None, None, metadata={"prediction_error": str(exc)})
        model_ms = (time.perf_counter() - model_started) * 1000
        decision_started = time.perf_counter()
        system_decision = (
            InspectionDecision.REVIEW
            if not quality.valid or prediction.image_score is None
            else decide(prediction.image_score, self.thresholds)
        )
        decision_ms = (time.perf_counter() - decision_started) * 1000
        metadata = dict(prediction.metadata)
        metadata["quality"] = {
            "valid": quality.valid,
            "width": quality.width,
            "height": quality.height,
            "brightness": quality.brightness,
            "contrast": quality.contrast,
            "sharpness": quality.sharpness,
            "reasons": list(quality.reasons),
            "policy": self.quality_policy.as_dict(),
        }
        if quality.reasons:
            metadata["quality_reason"] = quality.reasons[0]
        metadata["review_reasons"] = list(quality.reasons)
        if prediction.image_score is None:
            metadata["review_reasons"].append("invalid_model_output")
        elif quality.valid and system_decision == InspectionDecision.REVIEW:
            metadata["review_reasons"].append("score_in_review_band")
        metadata["identity"] = self.identity()
        metadata["timing_scope"] = {
            "preprocess": "quality check",
            "model": "combined model preprocessing, prediction and pixel policy, including model lock wait",
            "postprocess": "included in model; not measured separately",
            "total": "decode plus pipeline when called through HTTP; pipeline only otherwise",
        }
        prediction.metadata = metadata
        request_id = str(uuid.uuid4())
        evidence_started = time.perf_counter()
        original_hash = hashlib.sha256(original_bytes).hexdigest() if original_bytes else None
        evidence = self.evidence.save(
            request_id=request_id,
            image=decoded,
            prediction=prediction,
            model_id=self.model.model_id,
            category=category or self.category,
            decision=system_decision.value,
            thresholds=self.thresholds,
            original_sha256=original_hash,
        )
        evidence_ms = (time.perf_counter() - evidence_started) * 1000
        total_ms = (time.perf_counter() - total_started) * 1000
        return InspectionResponse(
            request_id=request_id,
            model_id=self.model.model_id,
            category=category or self.category,
            anomaly_score=prediction.image_score,
            decision=system_decision.value,
            thresholds={
                "pass": self.thresholds.pass_threshold,
                "fail": self.thresholds.fail_threshold,
            },
            regions=[RegionSchema(**region.as_dict()) for region in prediction.regions],
            evidence=EvidenceSchema(**evidence.as_dict()),
            timing_ms=TimingSchema(
                preprocess=preprocess_ms,
                model=model_ms,
                decision=decision_ms,
                evidence=evidence_ms,
                total=total_ms,
            ),
            created_at=datetime.now(UTC),
            metadata=prediction.metadata,
        )

    def inspect_batch(
        self,
        images: Sequence[np.ndarray],
        *,
        category: str | None = None,
        original_bytes: Sequence[bytes | None] | None = None,
    ) -> list[InspectionResponse]:
        raw = original_bytes or [None] * len(images)
        self.validate_category(category)
        if len(raw) != len(images):
            raise ValueError("original_bytes must match image count")
        return [
            self.inspect(image, category=category, original_bytes=payload)
            for image, payload in zip(images, raw, strict=True)
        ]


def image_dimensions(payload: bytes, *, max_pixels: int) -> tuple[int, int]:
    if not payload:
        raise ValueError("Image body is empty")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as header:
            width, height = header.size
            if header.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Image must be PNG, JPEG or WebP")
            if width * height > max_pixels:
                raise ValueError(f"Decoded image exceeds {max_pixels} pixels")
            return width, height
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Image is corrupt or uses an unsupported format") from exc


def decode_image_bytes(payload: bytes, *, max_pixels: int) -> np.ndarray:
    width, height = image_dimensions(payload, max_pixels=max_pixels)
    # Keep encoded pixel coordinates; do not silently rotate by EXIF during decoding.
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if decoded is None:
        raise ValueError("Image is corrupt or uses an unsupported format")
    if decoded.shape[:2] != (height, width):
        raise ValueError("Decoded image dimensions differ from the validated header")
    return decoded


def quality_policy_from_config(config: AppConfig) -> QualityPolicy:
    runtime = config.runtime
    return QualityPolicy(
        min_width=runtime.quality_min_width,
        min_height=runtime.quality_min_height,
        min_sharpness=runtime.quality_min_sharpness,
        brightness_range=(runtime.quality_brightness_min, runtime.quality_brightness_max),
    )
