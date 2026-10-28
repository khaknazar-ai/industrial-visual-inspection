"""Extend identity adapters without changing already-qualified legacy source hashes."""
from __future__ import annotations

import importlib.metadata
import inspect
from dataclasses import asdict
from pathlib import Path

import cv2

from industrial_inspection.models.runtime_identity import (
    array_identity,
    digest,
    state_identity,
)
from industrial_inspection.models.runtime_identity import (
    runtime_identity as legacy_runtime_identity,
)


def runtime_identity(model):
    from industrial_inspection.models.dinov3_runtime import (
        ENCODER_ID,
        ENCODER_REVISION,
        ENCODER_SHA256,
        PREPROCESS,
        DinoV3MemoryModel,
    )

    if not isinstance(model, DinoV3MemoryModel):
        return legacy_runtime_identity(model)
    from industrial_inspection.data import image_io
    from industrial_inspection.models import base, dino_memory, feature_memory
    from industrial_inspection.scoring import pixels

    parameters = asdict(model.config)
    parameters.pop("encoder_weights")
    parameters.pop("device")
    parameters["image_size"] = list(model.config.image_size)
    sources = [type(model), runtime_identity, legacy_runtime_identity, base, image_io,
               dino_memory, feature_memory, pixels, type(model._encoder), type(model._encoder.rope)]
    result = {
        "version": 1, "runtime_class": "industrial_inspection.models.dinov3_runtime.DinoV3MemoryModel",
        "decode": {"color": "BGR", "dtype": "uint8", "orientation": "encoded_pixels"},
        "score_space": model.score_space, "parameters": parameters,
        "transform": {**PREPROCESS, "image_size": list(model.config.image_size)},
        "encoder": {"model": ENCODER_ID, "asset_sha256": ENCODER_SHA256,
                    "revision": ENCODER_REVISION, "state": state_identity(model._encoder),
                    "buffers": {name: array_identity(value) for name, value in model._encoder.named_buffers()}},
        "fitted_state": {"bank": array_identity(model.memory.memory_bank), "sampling": model.memory.sampling},
        "postprocess": {"distance": "1-max_cosine", "map": "opencv_linear_original_coordinates",
                        "image_score": "maximum_patch_distance"},
        "dependencies": {name: importlib.metadata.version(name).split("+")[0]
                         for name in ("torch", "torchvision", "timm", "safetensors", "numpy")},
        "implementation": {},
    }
    import hashlib

    result["dependencies"]["opencv"] = cv2.__version__
    for source in sources:
        path = Path(inspect.getfile(source))
        result["implementation"][path.name] = hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
    result["fingerprint"] = digest(result)
    return result
