"""Fixed-input PatchCore graph with the common geometry and calibrated policy."""

from pathlib import Path

import cv2
import numpy as np

from industrial_inspection.models.artifacts import resolve_device
from industrial_inspection.models.base import AnomalyPrediction, load_color_image
from industrial_inspection.models.geometry import restore_map_geometry
from industrial_inspection.scoring.pixels import InvalidPredictionError, finalize_prediction


def restore_transform(spec):
    """Reconstruct only the deterministic geometry described by the source runtime.

    Color normalization is already in the graph. No class name from metadata is
    imported or evaluated dynamically.
    """
    from torchvision.transforms import CenterCrop, Compose, Resize

    name = spec["class"].rsplit(".", 1)[-1]
    if name == "Compose":
        operations = []
        for item in spec["transforms"]:
            transform = restore_transform(item)
            if transform is not None:
                operations.extend(transform.transforms if isinstance(transform, Compose) else [transform])
        return Compose(operations)
    parameters = spec.get("parameters", {})
    if name == "Resize":
        return Resize(parameters["size"], max_size=parameters.get("max_size"))
    if name == "CenterCrop":
        return CenterCrop(parameters["size"])
    if name in {"identity", "Identity", "Normalize", "ToDtype", "ConvertImageDtype", "ToTensor", "ToImage"}:
        return None
    raise ValueError(f"Unsupported exported geometry: {name}")


class PatchCoreTorchScriptRuntime:
    score_space = "anomalib_postprocessed_v1"
    calibration_status = "legacy_exploratory"
    requires_calibration_bundle = True

    def __init__(self, path: Path, *, device="cpu"):
        import json

        import torch

        self.path = path.resolve()
        self.spec = json.loads((path / "runtime.json").read_text(encoding="utf-8"))
        if self.spec.get("schema_version") != 1 or self.spec.get("format") != "patchcore_torchscript":
            raise ValueError("Unsupported TorchScript runtime descriptor")
        self.shape = tuple(self.spec["input_shape"])
        if len(self.shape) != 4 or self.shape[:2] != (1, 3) or any(type(x) is not int or x <= 0 for x in self.shape):
            raise ValueError("Expected a fixed positive [1, 3, H, W] input shape")
        self.device = torch.device(resolve_device(device))
        if str(self.device) != self.spec["source"]["device"]:
            raise ValueError("Export parity was measured on another device; create a new export for this device")
        self._torch = torch
        self._graph = torch.jit.load(str(path / "model.pt"), map_location=self.device).eval()
        self.transform = restore_transform(self.spec["geometry_transform"])
        self.model_id = self.spec["model_id"]

    def predict(self, image):
        decoded = load_color_image(image)
        if decoded.shape[:2] != self.shape[2:]:
            raise ValueError(f"Export requires image shape {self.shape[2:]}; got {decoded.shape[:2]}")
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div(255).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            score, response = self._graph(tensor)
        if score.numel() != 1 or response.ndim not in (3, 4) or response.shape[0] != 1:
            raise InvalidPredictionError("Export returned invalid tensor shapes")
        if response.ndim == 4 and response.shape[1] != 1:
            raise InvalidPredictionError("Export returned multiple anomaly map channels")
        score = float(score.reshape(-1)[0].item())
        response = response.reshape(response.shape[-2:]).detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(score) or not np.isfinite(response).all():
            raise InvalidPredictionError("Export returned non-finite output")
        response, support = restore_map_geometry(response, decoded.shape[:2], self.transform)
        return finalize_prediction(self, AnomalyPrediction(score, response, None, valid_mask=support,
                                    metadata={"model_family": "patchcore_torchscript", "device": str(self.device)}), decoded.shape[:2])

    def predict_batch(self, images):
        return [self.predict(image) for image in images]

    def fit(self, images):
        raise RuntimeError("Exported models are inference-only")

    def save(self, path):
        raise RuntimeError("Create a new export through the parity validator")
