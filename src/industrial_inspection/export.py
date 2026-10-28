"""Safe export metadata and optional TorchScript/ONNX hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_export_metadata(output: Path, *, model_id: str, source_format: str, target_format: str, **extra: Any) -> None:
    encoded = json.dumps(
            {
                "model_id": model_id,
                "source_format": source_format,
                "target_format": target_format,
                **extra,
                "export_status": "metadata_only_until_runtime_validation",
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def export_torch_onnx(
    module: Any,
    output: Path,
    *,
    sample_input: Any,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    opset_version: int = 17,
) -> Path:
    """Export only a caller-supplied trusted Torch module; never loads pickle input."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("ONNX export requires torch in the export environment") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        sample_input,
        str(output),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamo=True,
    )
    return output


def export_openvino(onnx_path: Path, output_dir: Path) -> Path:
    """Convert an already validated ONNX file when OpenVINO is installed."""

    try:
        from openvino import convert_model, save_model
    except ImportError as exc:
        raise RuntimeError("OpenVINO is optional; install it only in the export environment") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    model = convert_model(str(onnx_path))
    output = output_dir / f"{onnx_path.stem}.xml"
    save_model(model, str(output))
    return output
