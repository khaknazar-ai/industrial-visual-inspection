"""Evaluate frozen mask-policy variants on saved real normal-validation maps."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.data.contracts import load_manifest, manifest_sidecar_path
from industrial_inspection.data.integrity import FrozenInputs
from industrial_inspection.evaluation.artifacts import ArtifactReservation, read_json_object
from industrial_inspection.evaluation.predictions import PredictionStore
from industrial_inspection.evaluation.report import write_json_artifact
from industrial_inspection.evaluation.validation_stress import _reference
from industrial_inspection.models.artifacts import ModelBundle, artifact_reference
from industrial_inspection.models.runtime_identity import digest
from industrial_inspection.scoring.pixels import PixelPolicy


def count_policy_masks(prediction, policies):
    """Count the actual postprocessed masks, not thresholded pre-policy maps."""
    valid = prediction.valid_mask
    valid = np.ones(prediction.anomaly_map.shape, dtype=bool) if valid is None else valid.astype(bool)
    count = int(np.count_nonzero(valid))
    if not count:
        raise ValueError("No valid pixels in cached normal prediction")
    rows = []
    for policy in policies:
        result = policy.apply(prediction)
        positives = int(np.count_nonzero(result.binary_mask))
        if np.any(result.binary_mask[~valid]):
            raise ValueError("Policy emitted pixels outside original valid support")
        rows.append({"policy_id": policy.fingerprint, "valid_normal_pixels": count,
                     "false_positive_pixels": positives, "false_positive_regions": len(result.regions),
                     "has_false_positive_mask": int(positives > 0)})
    return rows


def run_mask_ablation(*, model_path: Path, manifest_path: Path, project_root: Path,
                      baseline_index: Path, baseline_predictions: Path, qualification_path: Path,
                      output: Path, legacy_summary_path: Path | None = None):
    cv2.setNumThreads(2)
    bundle = ModelBundle.load(model_path)
    native = PixelPolicy.from_dict(bundle.calibration["pixel"])
    policies = {p.fingerprint: p for p in [native, *[
        replace(native, morphology_kernel=kernel, min_component_area=area)
        for kernel in (0, 3, 5) for area in (1, 12, 36)]]}
    index_hash = sha256_file(baseline_index)
    cache = PredictionStore.open(baseline_index.parent)
    if sha256_file(baseline_index) != index_hash:
        raise ValueError("Prediction index changed while opening the cache")
    paths = [manifest_path, baseline_index, baseline_predictions, qualification_path, Path(__file__),
             Path(PixelPolicy.apply.__code__.co_filename),
             Path(PredictionStore.open.__func__.__code__.co_filename)]
    if legacy_summary_path is not None:
        paths.append(legacy_summary_path)
    arrays = [cache.root / filename for row in cache.rows for filename in row["arrays"].values()]
    paths.extend(arrays)
    frozen = FrozenInputs(paths, optional_paths=[manifest_sidecar_path(manifest_path)])
    spec = {"schema": "normal-validation-mask-ablation-v1", "model_fingerprint": bundle.model_fingerprint,
            "calibration_id": bundle.calibration_id, "artifact": artifact_reference(model_path),
            "runtime_fingerprint": bundle.calibration["runtime_fingerprint"],
            "sources": {str(p): sha256_file(p) for p in paths}, "native_policy": native.as_dict(),
            "variants": [p.as_dict() for p in policies.values()], "cpu_math_threads": 2,
            "thresholds": "unchanged; variations are evaluation only, no selection/recalibration"}
    reservation = ArtifactReservation.reserve_new(output, spec_fingerprint=digest(spec))
    try:
        write_json_artifact(output / "spec.json", spec)
        with frozen.scope():
            if sha256_file(baseline_index) != index_hash:
                raise ValueError("Prediction index changed before protocol freeze")
            samples = [s for s in load_manifest(manifest_path, project_root) if s.split == "validation"]
            _reference(samples, index_path=baseline_index, predictions_path=baseline_predictions,
                       qualification=read_json_object(qualification_path), bundle=bundle,
                       manifest_hash=sha256_file(manifest_path), legacy_summary_path=legacy_summary_path)
            by_id = {sample.sample_id: sample for sample in samples}
            totals = {key: {"images": 0, "valid_normal_pixels": 0, "false_positive_pixels": 0,
                            "false_positive_regions": 0, "images_with_false_positive_masks": 0}
                      for key in policies}
            import json

            with (output / "mask-counts.jsonl").open("x", encoding="utf-8") as journal:
                for index, row in enumerate(cache.rows):
                    prediction = cache[index]
                    sample = by_id[row["sample_id"]]
                    if prediction.anomaly_map.shape != (sample.height, sample.width):
                        raise ValueError("Cached map is not in original image coordinates")
                    for counts in count_policy_masks(prediction, policies.values()):
                        total = totals[counts["policy_id"]]
                        total["images"] += 1
                        for key in ("valid_normal_pixels", "false_positive_pixels", "false_positive_regions"):
                            total[key] += counts[key]
                        total["images_with_false_positive_masks"] += counts["has_false_positive_mask"]
                        journal.write(json.dumps({"sample_id": sample.sample_id, **counts}, allow_nan=False)+"\n")
            for key, total in totals.items():
                total.update(policy=policies[key].as_dict(), is_native=key == native.fingerprint,
                             false_positive_pixel_rate=total["false_positive_pixels"]/total["valid_normal_pixels"])
            frozen.verify_unchanged()
            if artifact_reference(model_path) != spec["artifact"]:
                raise ValueError("Bundle changed during ablation")
            report = {"schema": spec["schema"], "model_id": bundle.model["model_id"],
                      "model_fingerprint": bundle.model_fingerprint, "calibration_id": bundle.calibration_id,
                      "sample_count": len(samples), "runtime_model_loads": 0, "predictions": 0,
                      "policies": totals, "scope": "saved normal validation maps; masks after actual runtime policy",
                      "limitations": ["no defect recall/pixel AP/sPRO claim", "same calibration cohort, not independent",
                                      "no image score/decision changes; no winning policy selected",
                                      "no images or GT masks decoded; no model fit/runtime initialization/predict"]}
            write_json_artifact(output / "report.json", report)
        reservation.complete(artifacts=["spec.json", "mask-counts.jsonl", "report.json"])
        return report
    except BaseException as exc:
        reservation.fail(f"{type(exc).__name__}: {exc}")
        raise
