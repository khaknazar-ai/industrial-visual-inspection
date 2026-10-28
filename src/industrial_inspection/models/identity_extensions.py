"""Identity for SuperADD while retaining the existing models' identity functions."""

from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path

import cv2

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.models.identity_router import runtime_identity as previous_identity
from industrial_inspection.models.runtime_identity import digest
from industrial_inspection.models.superadd_adapter import SuperADDRuntime


def runtime_identity(model):
    if not isinstance(model, SuperADDRuntime):
        return previous_identity(model)
    from industrial_inspection.data import image_io
    from industrial_inspection.models import base, superadd_runtime
    from industrial_inspection.scoring import pixels

    result = {
        "version": 1, "runtime_class": "industrial_inspection.models.superadd_adapter.SuperADDRuntime",
        "decode": {"color": "BGR", "dtype": "uint8", "orientation": "encoded_pixels"},
        "score_space": model.score_space, "parameters": model.config,
        "checkpoint_sha256": model.checkpoint_sha256, "device": str(model.device),
        "provenance": model.provenance,
        "postprocess": {"image": model.default_thresholds.as_dict(),
                        "pixel": model.pixel_policy.as_dict(), "geometry": "resize_only_original_linear"},
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("torch", "torchvision", "anomalib", "timm", "numpy")},
        "implementation": {},
    }
    result["dependencies"]["opencv"] = cv2.__version__
    for source in (type(model), runtime_identity, superadd_runtime, base, image_io, pixels):
        path = Path(inspect.getfile(source))
        result["implementation"][path.name] = sha256_file(path)
    result["fingerprint"] = digest(result)
    return result
