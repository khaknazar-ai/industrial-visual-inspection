"""Compatibility keys for joining measurements from the same inference protocol."""

from __future__ import annotations

import platform

from industrial_inspection.models.identity_router import digest, runtime_identity


def comparison_identity(model, *, split_fingerprint, category, thresholds):
    runtime = getattr(model, "runtime_identity", None) or runtime_identity(model)
    device = str(getattr(model, "device", "cpu"))
    hardware = {"host": platform.node(), "machine": platform.machine(), "processor": platform.processor()}
    if device.startswith("cuda"):
        import torch

        props = torch.cuda.get_device_properties(device)
        hardware["gpu"] = {"name": props.name, "memory_bytes": props.total_memory}
    pixel = getattr(model, "pixel_policy", None)
    policy = {"image": thresholds.as_dict() if thresholds else None,
              "pixel": pixel.as_dict() if pixel else None}
    return {
        "version": 1, "model_id": model.model_id, "category": category,
        "weights_state_sha256": digest({key: runtime[key] for key in ("encoder", "fitted_state") if key in runtime}),
        "weights_scope": "decoded_native_tensors_and_reference",
        "runtime_fingerprint": runtime["fingerprint"], "split_fingerprint": split_fingerprint,
        "policy_fingerprint": digest(policy) if thresholds is not None and pixel is not None else None,
        "device": device, "hardware_fingerprint": digest(hardware),
        "input_scope": "native_predict_with_path_decode",
    }


def compatible_measurements(left, right):
    required = {"version", "model_id", "category", "weights_state_sha256", "weights_scope",
                "runtime_fingerprint", "split_fingerprint", "policy_fingerprint", "device",
                "hardware_fingerprint", "input_scope"}
    return (isinstance(left, dict) and isinstance(right, dict) and left.get("version") == 1
            and all(left.get(key) is not None and left.get(key) == right.get(key) for key in required))
