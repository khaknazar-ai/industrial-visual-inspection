"""Common inference adapter for the existing local SuperADD experiment artifact."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.models.artifacts import ModelBundleError
from industrial_inspection.models.base import AnomalyPrediction, load_color_image
from industrial_inspection.models.runtime_identity import digest
from industrial_inspection.scoring.pixels import (
    InvalidPredictionError,
    PixelPolicy,
    finalize_prediction,
)
from industrial_inspection.scoring.policy import DecisionThresholds


def _load_checkpoint(path, device):
    # Keep Anomalib/training imports out of the core service catalogue.
    from industrial_inspection.models.superadd_runtime import load_superadd_checkpoint

    return load_superadd_checkpoint(path, device=device)


class SuperADDRuntime:
    score_space = "anomalib_normalized_anomaly_score"
    calibration_status = "legacy_exploratory"
    bound_native_thresholds = True

    def __init__(self, path: Path, *, device: str = "auto"):
        self.path = path.resolve()
        if self.path.is_dir():
            self.path /= "model.pt"
        root = self.path.parent
        metadata = json.loads((root / "fit_metadata.json").read_text(encoding="utf-8"))
        status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            raise ModelBundleError("SuperADD fit is incomplete")
        policy = metadata.get("training_policy", {})
        if (policy.get("fit_split") != "train" or policy.get("fit_labels") != ["normal"]
                or policy.get("calibration_split") != "validation"
                or policy.get("test_seen_during_fit") is not False):
            raise ModelBundleError("SuperADD requires recorded normal training and validation-only calibration")
        self.checkpoint_sha256 = sha256_file(self.path)
        if self.checkpoint_sha256 != metadata.get("checkpoint", {}).get("sha256"):
            raise ModelBundleError("SuperADD checkpoint differs from its fit receipt")
        self.category = metadata.get("category")
        if not isinstance(self.category, str) or not self.category:
            raise ModelBundleError("SuperADD fit receipt is missing its category")

        saved, self.model, self.device = _load_checkpoint(self.path, device)
        if sha256_file(self.path) != self.checkpoint_sha256:
            raise ModelBundleError("SuperADD checkpoint changed during loading")
        self.config = saved["config"]
        self.model_id = root.name
        post = self.model.post_processor
        image_threshold = float(post.normalized_image_threshold.detach().cpu().item())
        pixel_threshold = float(post.normalized_pixel_threshold.detach().cpu().item())
        if not all(np.isfinite(value) and 0 <= value <= 1 for value in (image_threshold, pixel_threshold)):
            raise ModelBundleError("SuperADD embedded thresholds must be finite normalized values")
        # Reuse the native strict 'score > threshold' boundary. Equality abstains;
        # no new review width is tuned from the already observed test set.
        self.default_thresholds = DecisionThresholds(
            image_threshold, float(np.nextafter(image_threshold, np.inf)),
            source="checkpoint_validation", objective="native_binary_boundary_with_equality_review",
        )
        self.decision_thresholds = self.default_thresholds
        self.pixel_policy = PixelPolicy(pixel_threshold, version=2)
        self.calibration_id = "embedded-" + digest({
            "checkpoint": self.checkpoint_sha256, "image_threshold": image_threshold,
            "pixel_threshold": pixel_threshold, "equality": "review",
        })
        self.model_fingerprint = digest({"checkpoint": self.checkpoint_sha256, "config": self.config})
        self.provenance = {
            "fit_metadata_sha256": sha256_file(root / "fit_metadata.json"),
            "manifest_sha256": metadata.get("manifest_sha256"),
            "split_fingerprint": metadata.get("split_fingerprint"),
        }

    def predict(self, image):
        import torch

        bgr = load_color_image(image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
        tensor = tensor.div(255).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            prediction = self.model(tensor)
        if prediction.pred_score.numel() != 1:
            raise InvalidPredictionError("SuperADD returned multiple image scores")
        score = float(prediction.pred_score.detach().cpu().item())
        response = prediction.anomaly_map.detach().float().cpu().numpy()
        expected = tuple(self.config["image_size"])
        if response.shape == (1, *expected):
            response = response[0]
        elif response.shape == (1, 1, *expected):
            response = response[0, 0]
        else:
            raise InvalidPredictionError(f"Unexpected SuperADD map geometry: {response.shape}")
        if not np.isfinite(score) or not np.isfinite(response).all():
            raise InvalidPredictionError("SuperADD returned non-finite predictions")
        if response.shape != bgr.shape[:2]:
            response = cv2.resize(response, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        result = AnomalyPrediction(score, response, None, metadata={
            "model_family": "superadd", "device": str(self.device),
            "checkpoint_sha256": self.checkpoint_sha256,
            "qualification": "local_exploratory; not official SuperADD reproduction",
            "decision_mapping": "native_binary_boundary; equality_review; quality_policy_may_abstain",
            "map_geometry": "resize_only_model_grid_to_original_linear",
            "source_map_shape": list(expected), "provenance": self.provenance,
        })
        return finalize_prediction(self, result, bgr.shape[:2])

    def predict_batch(self, images):
        return [self.predict(image) for image in images]

    def fit(self, images):
        raise RuntimeError("Use the dedicated SuperADD manifest fit command")

    def save(self, path):
        raise RuntimeError("Reuse the complete SuperADD experiment artifact with its fit receipt")
