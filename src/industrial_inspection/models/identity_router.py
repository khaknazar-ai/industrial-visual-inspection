"""Add export identities without changing any existing calibrated identity source."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from dataclasses import asdict
from pathlib import Path

import cv2

from industrial_inspection.models.identity_dispatch import runtime_identity as native_identity
from industrial_inspection.models.runtime_identity import array_identity, digest


def runtime_identity(model):
    from industrial_inspection.models.dinov3_torchscript import DinoV3TorchScriptRuntime

    if not isinstance(model, DinoV3TorchScriptRuntime):
        return native_identity(model)
    from industrial_inspection.core.reproducibility import sha256_file
    from industrial_inspection.data import image_io
    from industrial_inspection.models import base, dino_memory, dinov3_runtime, feature_memory
    from industrial_inspection.models.dinov3_runtime import ENCODER_SHA256, PREPROCESS
    from industrial_inspection.scoring import pixels

    config = asdict(model.config)
    config.pop("encoder_weights")
    config.pop("device")
    config["image_size"] = list(config["image_size"])
    result = {"version": 1, "runtime_class": "industrial_inspection.models.dinov3_torchscript.DinoV3TorchScriptRuntime",
        "decode": {"color": "BGR", "dtype": "uint8", "orientation": "encoded_pixels"},
        "transform": {**PREPROCESS, "image_size": list(model.config.image_size)},
        "score_space": model.score_space, "parameters": config,
        "encoder": {"kind": "torchscript_encoder", "graph_sha256": sha256_file(model.graph_path),
                    "source_asset_sha256": ENCODER_SHA256, "device": "cpu"},
        "fitted_state": {"bank": array_identity(model.memory.memory_bank), "sampling": model.memory.sampling},
        "postprocess": {"distance": "unchanged_native_numpy_cosine", "geometry": "original_coordinates",
                        "image_score": "maximum_patch_distance", "pixel_policy": "external_bound_calibration"},
        "dependencies": {name: importlib.metadata.version(name).split("+")[0] for name in ("torch", "numpy")},
        "implementation": {}}
    result["dependencies"]["opencv"] = cv2.__version__
    for source in (type(model), runtime_identity, base, image_io, dino_memory, dinov3_runtime, feature_memory, pixels):
        file = Path(inspect.getfile(source))
        result["implementation"][file.name] = hashlib.sha256(file.read_text(encoding="utf-8").encode()).hexdigest()
    result["fingerprint"] = digest(result)
    return result
