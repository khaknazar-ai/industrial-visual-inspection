"""Publish PatchCore exports after cold-load, whole-pipeline parity checks."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.data.manifest_datamodule import load_manifest_splits
from industrial_inspection.evaluation.artifacts import ArtifactReservation
from industrial_inspection.evaluation.benchmark import benchmark_model
from industrial_inspection.models.artifacts import ModelBundle, artifact_reference
from industrial_inspection.models.base import load_color_image
from industrial_inspection.models.patchcore_torchscript import PatchCoreTorchScriptRuntime
from industrial_inspection.models.registry import load_bundle_model
from industrial_inspection.models.runtime_identity import (
    digest,
    runtime_identity,
    transform_identity,
)
from industrial_inspection.scoring.policy import decide


@dataclass(frozen=True)
class ParityTolerances:
    score_atol: float = 1e-6
    map_atol: float = 1e-6
    rtol: float = 1e-5

    def __post_init__(self):
        if any(isinstance(v, bool) or not np.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Parity tolerances must be finite and non-negative")


def compare_predictions(native, exported, thresholds, tolerance):
    errors = []
    if native.image_score is None or exported.image_score is None or not np.isfinite([native.image_score, exported.image_score]).all():
        return {"passed": False, "errors": ["nonfinite_score"]}
    score_error = abs(float(native.image_score) - float(exported.image_score))
    if not np.isclose(exported.image_score, native.image_score, atol=tolerance.score_atol, rtol=tolerance.rtol):
        errors.append("score_tolerance")
    map_error = None
    for name in ("anomaly_map", "binary_mask", "valid_mask"):
        expected, actual = getattr(native, name), getattr(exported, name)
        if expected is None or actual is None:
            errors.append(f"missing_{name}")
        elif expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
            errors.append(f"invalid_{name}")
        elif name == "anomaly_map":
            map_error = float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
            if not np.allclose(actual, expected, atol=tolerance.map_atol, rtol=tolerance.rtol):
                errors.append("map_tolerance")
        elif not np.array_equal(expected, actual):
            errors.append(f"different_{name}")
    if decide(native.image_score, thresholds) != decide(exported.image_score, thresholds):
        errors.append("different_decision")
    expected_regions = sorted(native.regions, key=lambda r: (r.bbox_xyxy, r.area))
    actual_regions = sorted(exported.regions, key=lambda r: (r.bbox_xyxy, r.area))
    if len(expected_regions) != len(actual_regions) or any(a.bbox_xyxy != b.bbox_xyxy or a.area != b.area or
             not np.isclose(b.region_score, a.region_score, atol=tolerance.map_atol, rtol=tolerance.rtol)
             for a, b in zip(expected_regions, actual_regions, strict=True)):
        errors.append("different_regions")
    return {"passed": not errors, "errors": errors,
            "max_abs_score_error": score_error if np.isfinite(score_error) else None,
            "max_abs_map_error": map_error if map_error is None or np.isfinite(map_error) else None}


def _tensor_wrapper(runtime):
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Lightning's trainer is not part of the graph. This is the same
            # inference order as AnomalibModule.forward.
            self.pre = runtime._model.pre_processor
            self.core = runtime._model.model
            self.post = runtime._model.post_processor

        def forward(self, inputs):
            inputs = self.pre(inputs) if self.pre is not None else inputs
            result = self.core(inputs)
            result = self.post(result) if self.post is not None else result
            return result.pred_score, result.anomaly_map

    return Wrapper().eval()


def _write(path, data):
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def export_patchcore_bundle(*, source: Path, manifest: Path, project_root: Path, category: str,
                            output: Path, device="cpu", runs=30, warmup=5,
                            tolerances=None, near_threshold_margin=1e-3):
    """Trace one shape and validate all validation images, including boundary cases.

    Normal, anomalous and near-boundary probes are required. No test image is
    selected to decide whether an export is publishable.
    """
    tolerances = tolerances or ParityTolerances()
    if type(runs) is not int or runs < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("runs must be positive and warmup non-negative integers")
    if isinstance(near_threshold_margin, bool) or not np.isfinite(near_threshold_margin) or near_threshold_margin < 0:
        raise ValueError("near_threshold_margin must be finite and non-negative")
    if output.exists():
        raise FileExistsError(f"Export output must be new: {output}")
    native, bundle = load_bundle_model("patchcore", source, category=category, device=device)
    if bundle.calibration_status != "validated" or native.pixel_policy.threshold is None:
        raise ValueError("Export requires a validated source bundle with a calibrated pixel policy")
    splits = load_manifest_splits(manifest, project_root=project_root, category=category)
    if bundle.calibration["validation"]["split_fingerprint"] != splits.fingerprint:
        raise ValueError("Export manifest differs from source calibration provenance")
    probes = list(splits.validation)
    if len(probes) < 3 or {s.label for s in probes} != {"normal", "anomaly"}:
        raise ValueError("Export needs at least three validation probes including normal and anomaly")
    shapes = {(s.height, s.width) for s in probes}
    if len(shapes) != 1:
        raise ValueError("A fixed-shape export requires one validation input resolution")
    shape = [1, 3, *next(iter(shapes))]
    spec = {"source_model_fingerprint": bundle.model_fingerprint,
            "source_calibration_id": bundle.calibration_id,
            "source_runtime": native.runtime_identity["fingerprint"],
            "split_fingerprint": splits.fingerprint, "input_shape": shape,
            "tolerances": asdict(tolerances), "near_threshold_margin": near_threshold_margin,
            "device": str(native.device), "runs": runs, "warmup": warmup,
            "probe_ids": [s.sample_id for s in probes],
            "probe_sha256": {s.sample_id: sha256_file(s.image_path) for s in probes}}
    reservation = ArtifactReservation.reserve_new(output, spec_fingerprint=digest(spec))
    try:
        _write(output / "resolved_spec.json", spec)
        splits.write_provenance(output / "manifest_provenance.json")
        import torch

        target = output / "bundle"
        runtime_path = target / "runtime"
        runtime_path.mkdir(parents=True)
        graph_spec = {"schema_version": 1, "format": "patchcore_torchscript",
                      "input_shape": shape, "model_id": native.model_id + "-torchscript",
                      "geometry_transform": transform_identity(getattr(native._model.pre_processor, "transform", None)),
                      "source": spec}
        _write(runtime_path / "runtime.json", graph_spec)
        decoded = cv2.cvtColor(load_color_image(probes[0].image_path), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(decoded).permute(2, 0, 1).contiguous().float().div(255).unsqueeze(0).to(native.device)
        wrapper = _tensor_wrapper(native)
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, tensor, strict=True, check_trace=True)
        traced.save(str(runtime_path / "model.pt"))
        del traced, wrapper
        exported = PatchCoreTorchScriptRuntime(runtime_path, device=str(native.device))
        exported.pixel_policy = native.pixel_policy
        rows = []
        for sample in probes:
            expected_hash = spec["probe_sha256"][sample.sample_id]
            if sha256_file(sample.image_path) != expected_hash:
                raise ValueError("Validation image changed after the export specification was frozen")
            decoded = load_color_image(sample.image_path)
            expected, actual = native.predict(decoded), exported.predict(decoded)
            if sha256_file(sample.image_path) != expected_hash:
                raise ValueError("Validation image changed during parity evaluation")
            parity = compare_predictions(expected, actual, native.decision_thresholds, tolerances)
            distance = min(abs(expected.image_score - boundary) for boundary in
                           (native.decision_thresholds.pass_threshold, native.decision_thresholds.fail_threshold))
            rows.append({"sample_id": sample.sample_id, "sha256": sha256_file(sample.image_path),
                         "label": sample.label, "score": expected.image_score,
                         "distance_to_boundary": distance, **parity})
        near = [row["sample_id"] for row in rows if row["distance_to_boundary"] <= near_threshold_margin]
        passed = all(row["passed"] for row in rows) and bool(near)
        validation = {"status": "passed" if passed else "failed", "scope": "whole_predict_pipeline",
                      "cold_loaded": True, "tolerances": asdict(tolerances), "probes": rows,
                      "near_threshold_probes": near, "near_threshold_margin": near_threshold_margin}
        _write(output / "parity.json", validation)
        if not passed:
            raise ValueError("Export parity failed or no validation probe is near a policy boundary; see parity.json")
        timing = {"native": benchmark_model(native, [s.image_path for s in probes], runs=runs, warmup=warmup),
                  "torchscript": benchmark_model(exported, [s.image_path for s in probes], runs=runs, warmup=warmup),
                  "scope": "decode_preprocess_graph_geometry_pixel_policy; evidence and quality gate excluded"}
        _write(output / "benchmark.json", timing)
        bundle.verify()
        if load_manifest_splits(manifest, project_root=project_root, category=category).fingerprint != splits.fingerprint:
            raise ValueError("Manifest content changed during export")
        if runtime_identity(native) != native.runtime_identity:
            raise ValueError("Source runtime changed during export")
        calibration = {"score_space": native.score_space, "image": bundle.calibration["image"],
                       "pixel": native.pixel_policy.as_dict(), "source": "native_export_parity",
                       "threshold_source": native.decision_thresholds.source,
                       "objective": native.decision_thresholds.objective,
                       "validation": bundle.calibration["validation"],
                       "source_calibration_id": bundle.calibration_id,
                       "export_parity_sha256": sha256_file(output / "parity.json")}
        weights = artifact_reference(runtime_path)
        weights["path"] = "runtime"
        result = ModelBundle.write(target, model_id=exported.model_id, family="patchcore_torchscript",
                                   category=category, weights=weights, preprocessing={"fixed_input_shape": shape},
                                   calibration=calibration, training=bundle.model.get("training"), runtime=exported)
        _write(output / "publication.json", {"model_fingerprint": result.model_fingerprint,
                                            "calibration_id": result.calibration_id})
        reservation.complete(artifacts=["bundle", "parity.json", "benchmark.json", "publication.json",
                                        "resolved_spec.json", "manifest_provenance.json"])
        # Factory checks the completed export receipt and cold-loads the bundle.
        reloaded, _ = load_bundle_model("patchcore_torchscript", target, category=category, device=str(native.device))
        final = compare_predictions(native.predict(probes[0].image_path), reloaded.predict(probes[0].image_path),
                                    native.decision_thresholds, tolerances)
        if not final["passed"]:
            raise ValueError("Published bundle failed the runtime factory round-trip")
        return {"bundle": str(target.resolve()), "parity": validation, "calibration_id": result.calibration_id}
    except BaseException as exc:
        reservation.fail(f"{type(exc).__name__}: {exc}")
        raise
