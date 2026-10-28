"""Content identity of the prediction path actually loaded into memory."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def _class_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return {"enum": _class_name(value), "value": value.value}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    # torch.dtype has a stable name; arbitrary repr(object) is never an identity.
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return str(value)
    raise ValueError(f"Unsupported identity value: {_class_name(value)}")


def array_identity(value: Any) -> dict[str, Any]:
    if value is None:
        raise ValueError("Cannot bind an unfitted model state")
    if hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        import torch

        raw = tensor.reshape(-1).view(torch.uint8).numpy()
        shape, dtype = list(tensor.shape), str(tensor.dtype)
    else:
        raw = np.ascontiguousarray(value)
        shape, dtype = list(raw.shape), str(raw.dtype)
    return {"shape": shape, "dtype": dtype,
            "sha256": hashlib.sha256(memoryview(raw).cast("B")).hexdigest()}


def state_identity(module: Any) -> dict[str, Any]:
    values = {name: array_identity(value) for name, value in sorted(module.state_dict().items())}
    modules = {}
    for name, child in module.named_modules():
        parameters = {}
        for field, value in vars(child).items():
            if field.startswith("_") or field == "training":
                continue
            if value is None or isinstance(value, (bool, str, int, float, tuple, list, dict, Enum)):
                try:
                    parameters[field] = _plain(value)
                except ValueError:
                    continue  # Tensor state is bound separately, not encoded as JSON.
        modules[name] = {"class": _class_name(child), "parameters": parameters}
    # Native postprocessors may contain unused NaN sentinel buffers. Their exact bytes are hashed,
    # never written as non-finite JSON values or used to excuse non-finite prediction output.
    return {"class": _class_name(module), "state_sha256": digest(values), "tensors": values,
            "modules": modules}


def transform_identity(transform: Any) -> dict[str, Any]:
    if transform is None:
        return {"class": "identity"}
    name = type(transform).__name__
    module = type(transform).__module__
    if not module.startswith(("torchvision.transforms.", "torch.nn.")):
        raise ValueError(f"Unsupported prediction transform: {_class_name(transform)}")
    if name == "Compose":
        return {"class": _class_name(transform),
                "transforms": [transform_identity(item) for item in transform.transforms]}
    fields = {
        "Resize": ("size", "interpolation", "max_size", "antialias"),
        "CenterCrop": ("size",), "Normalize": ("mean", "std", "inplace"),
        "ToDtype": ("dtype", "scale"), "ConvertImageDtype": ("dtype",),
        "ToTensor": (), "ToImage": (), "Identity": (),
    }
    if name not in fields:
        raise ValueError(f"Unsupported prediction transform: {_class_name(transform)}")
    return {"class": _class_name(transform),
            "parameters": {field: _plain(getattr(transform, field)) for field in fields[name]}}


def runtime_identity(model: Any) -> dict[str, Any]:
    """Describe native inference before attaching an external calibrated pixel policy.

    Device and training-only paths are excluded. Effective tensor dtypes, normalization buffers,
    installed algorithm versions and source digests are included; changing them needs revalidation.
    """
    from industrial_inspection.data import image_io
    from industrial_inspection.models import base, geometry
    from industrial_inspection.scoring import pixels

    family = type(model).__name__
    sources = [type(model), base, geometry, image_io, pixels, runtime_identity]
    result: dict[str, Any] = {
        "version": 1, "runtime_class": _class_name(model),
        "decode": {"color": "BGR", "dtype": "uint8", "orientation": "encoded_pixels"},
        "score_space": getattr(model, "score_space", "raw_distance"),
        "dependencies": {"numpy": np.__version__, "opencv": cv2.__version__},
    }
    if family in {"ClassicalDifferenceModel", "PatchMemoryModel", "DeepPatchMemoryModel"}:
        config = asdict(model.config)
        for field in ("device", "pretrained", "encoder_weights"):
            config.pop(field, None)
        result["parameters"] = _plain(config)
    if family == "PatchCoreTorchScriptRuntime":
        from industrial_inspection.models.artifacts import artifact_reference

        result["encoder"] = {"kind": "full_torchscript_graph",
                             "artifact": artifact_reference(model.path)["sha256"]}
        result["fitted_state"] = state_identity(model._graph)
        result["transform"] = {"fixed_input_shape": list(model.shape),
                               "geometry": transform_identity(model.transform)}
        result["postprocess"] = {"native_pre_post_in_graph": True, "geometry": "original_coordinates"}
    elif family == "ClassicalDifferenceModel":
        result["encoder"] = {"kind": "median_reference", "reference": array_identity(model.reference)}
        result["transform"] = {"resize": "reference_shape_area", "alignment": "ECC_translation"}
        result["postprocess"] = {"map": "inverse_ECC_original_coordinates", "interpolation": "linear"}
    elif family in {"PatchMemoryModel", "DeepPatchMemoryModel"}:
        result["encoder"] = (state_identity(model._encoder) if family == "DeepPatchMemoryModel"
                             else {"kind": "handcrafted_LAB_gradient_patches"})
        result["fitted_state"] = {name: array_identity(getattr(model, name))
                                  for name in ("memory_bank", "feature_mean", "feature_std")}
        result["transform"] = ({"color": "RGB", "range": [0, 1], "resize": "torch_bilinear",
                                "align_corners": False, "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225]}
                               if family == "DeepPatchMemoryModel" else {"resize": "opencv_area"})
        result["postprocess"] = {"map": "original_coordinates", "interpolation": "cubic"}
    elif family in {"PatchCoreRuntime", "EfficientAdRuntime"}:
        native = model._model
        core = native.model
        pre = getattr(native, "pre_processor", None)
        result["transform"] = transform_identity(getattr(pre, "transform", None))
        result["preprocessor_state"] = state_identity(pre) if pre is not None else None
        result["encoder"] = state_identity(core.feature_extractor if family == "PatchCoreRuntime" else core.teacher)
        result["fitted_state"] = state_identity(core)
        if family == "PatchCoreRuntime":
            if getattr(core, "tiler", None) is not None:
                raise ValueError("Tiled PatchCore needs a separate geometry and identity contract")
            result["parameters"] = {name: _plain(getattr(core, name)) for name in ("backbone", "layers", "num_neighbors")}
        else:
            result["parameters"] = _plain(model.model_spec)
            from industrial_inspection.models import efficientad_adapter

            sources.append(efficientad_adapter)
        post = getattr(native, "post_processor", None)
        result["postprocess"] = {"enabled": post is not None}
        if post is not None:
            result["postprocess"].update(state_identity(post))
            result["postprocess"]["parameters"] = {
                name: _plain(getattr(post, name)) for name in (
                    "enable_thresholding", "enable_normalization", "enable_threshold_matching",
                    "image_sensitivity", "pixel_sensitivity",
                )
            }
    else:
        raise ValueError(f"No verified runtime identity adapter for {family}")
    if family in {"DeepPatchMemoryModel", "PatchCoreRuntime", "EfficientAdRuntime", "PatchCoreTorchScriptRuntime"}:
        names = ["torch", "torchvision"]
        if family in {"PatchCoreRuntime", "EfficientAdRuntime"}:
            names += ["anomalib", "timm"]
        for name in names:
            # Local build tags such as +cpu and +cu128 belong to deployment environment metadata.
            result["dependencies"][name] = importlib.metadata.version(name).split("+")[0]
    result["implementation"] = {}
    for source in sources:
        file = Path(inspect.getfile(source))
        result["implementation"][file.name] = hashlib.sha256(file.read_text(encoding="utf-8").encode()).hexdigest()
    result["fingerprint"] = digest(result)
    return result
