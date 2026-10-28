"""Warmup-aware latency, throughput, RAM, and VRAM benchmark helpers."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from industrial_inspection.core.resources import measure_resources
from industrial_inspection.data.manifest_datamodule import load_manifest_splits
from industrial_inspection.evaluation.artifacts import ArtifactReservation, publish_files
from industrial_inspection.evaluation.identity import comparison_identity
from industrial_inspection.evaluation.metrics import latency_summary
from industrial_inspection.evaluation.report import write_json_artifact
from industrial_inspection.models.artifacts import artifact_reference
from industrial_inspection.models.base import AnomalyModel, ImageInput
from industrial_inspection.models.identity_router import digest, runtime_identity
from industrial_inspection.models.registry import load_model


def _rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _cuda_sync(device: str | None) -> None:
    if device is None:
        return
    try:
        import torch
    except (ImportError, OSError):
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _cuda_memory(device: str | None) -> dict[str, int | None]:
    empty = {"allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None}
    if device is None:
        return empty
    try:
        import torch
    except (ImportError, OSError):
        return {"allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None}
    if not torch.cuda.is_available():
        return empty
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def benchmark_model(
    model: AnomalyModel,
    images: Sequence[ImageInput],
    *,
    warmup: int = 5,
    runs: int = 100,
) -> dict[str, Any]:
    if not images:
        raise ValueError("At least one image is required")
    if type(warmup) is not int or warmup < 0 or type(runs) is not int or runs < 1:
        raise ValueError("warmup must be non-negative and runs positive integers")
    device = str(getattr(model, "device", "cpu"))
    cuda_device = device if device.startswith("cuda") else None
    for index in range(warmup):
        model.predict(images[index % len(images)])
    _cuda_sync(cuda_device)
    before_rss = _rss_bytes()
    timings: list[float] = []
    with measure_resources(cuda=cuda_device or False) as resources:
        before_cuda = _cuda_memory(cuda_device)
        for index in range(runs):
            _cuda_sync(cuda_device)
            started = time.perf_counter()
            model.predict(images[index % len(images)])
            _cuda_sync(cuda_device)
            timings.append((time.perf_counter() - started) * 1000)
    after_rss = _rss_bytes()
    after_cuda = _cuda_memory(cuda_device)
    summary = latency_summary(timings)
    summary["throughput_fps"] = 1000.0 / summary["mean_ms"] if summary["mean_ms"] > 0 else None
    return {
        "latency": summary,
        "resources": {**resources, "scope": "measured_runs_after_warmup"},
        "protocol": {"device": device, "warmup": warmup, "runs": runs,
                     "input_scope": "native_predict_with_path_decode" if all(isinstance(image, (str, Path)) for image in images) else "native_predict_predecoded_arrays"},
        "rss_before_bytes": before_rss,
        "rss_after_bytes": after_rss,
        "rss_delta_bytes": after_rss - before_rss if before_rss is not None and after_rss is not None else None,
        "cuda_before": before_cuda,
        "cuda_after": after_cuda,
    }


def benchmark_saved_model(
    *, model_name: str, model_path: Path, manifest_path: Path, project_root: Path,
    output: Path, device: str = "cpu", warmup: int = 5, runs: int = 100,
    split: str = "validation",
) -> dict[str, Any]:
    """Write a new benchmark with an exact evaluation-compatibility key."""
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    if output.suffix.lower() != ".json":
        raise ValueError("Benchmark output must have a .json suffix")
    if split not in {"validation", "test"}:
        raise ValueError("Benchmark split must be validation or test")
    if type(warmup) is not int or warmup < 0 or type(runs) is not int or runs < 1:
        raise ValueError("warmup must be non-negative and runs positive integers")
    splits = load_manifest_splits(manifest_path, project_root=project_root)
    samples = list(getattr(splits, split))
    categories = {sample.category for sample in splits.train + splits.validation + splits.test}
    if not samples or len(categories) != 1:
        raise ValueError("Benchmark requires a non-empty split and one model category")
    spec = {"model_name": model_name, "artifact": artifact_reference(model_path),
            "split_fingerprint": splits.fingerprint, "split": split, "device": device,
            "warmup": warmup, "runs": runs}
    reservation = ArtifactReservation.reserve_new(output.with_suffix(".run"), spec_fingerprint=digest(spec))
    try:
        write_json_artifact(reservation.root / "spec.json", spec)
        model = load_model(model_name, model_path, device=device)
        category = next(iter(categories))
        if getattr(model, "category", category) != category:
            raise ValueError("Benchmark category differs from loaded model")
        identity = comparison_identity(model, split_fingerprint=splits.fingerprint, category=category,
                                       thresholds=getattr(model, "decision_thresholds", None))
        frozen_runtime = runtime_identity(model)
        result = benchmark_model(model, [sample.image_path for sample in samples], warmup=warmup, runs=runs)
        result["schema_version"] = 2
        result["comparison_identity"] = identity
        result["model"] = {"name": model_name, "model_id": model.model_id, "artifact": spec["artifact"]}
        result["protocol"].update({"split": split, "sample_count": len(samples)})
        if artifact_reference(model_path) != spec["artifact"]:
            raise ValueError("Model artifact changed during benchmark")
        if runtime_identity(model) != frozen_runtime:
            raise ValueError("Runtime changed during benchmark")
        if comparison_identity(model, split_fingerprint=splits.fingerprint, category=category,
                               thresholds=getattr(model, "decision_thresholds", None)) != identity:
            raise ValueError("Policy changed during benchmark")
        if load_manifest_splits(manifest_path, project_root=project_root).fingerprint != splits.fingerprint:
            raise ValueError("Data changed during benchmark")
        staged = reservation.root / "report.json"
        write_json_artifact(staged, result)
        with publish_files({staged: output}):
            reservation.complete(artifacts=["spec.json", "report.json", str(output.resolve())])
        return result
    except BaseException as exc:
        reservation.fail(f"{type(exc).__name__}: {exc}")
        raise
