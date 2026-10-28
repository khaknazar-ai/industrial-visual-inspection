"""Stress real validation inputs without repeating saved baseline inference."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import psutil

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.data.contracts import load_manifest, manifest_sidecar_path
from industrial_inspection.data.integrity import FrozenInputs
from industrial_inspection.evaluation.artifacts import ArtifactReservation, read_json_object
from industrial_inspection.evaluation.metrics import image_level_metrics, review_system_metrics
from industrial_inspection.evaluation.report import write_json_artifact
from industrial_inspection.evaluation.robustness import apply_perturbation, default_perturbations
from industrial_inspection.models.artifacts import artifact_reference
from industrial_inspection.models.base import load_color_image
from industrial_inspection.models.runtime_identity import digest, runtime_identity
from industrial_inspection.scoring.pixels import finalize_prediction
from industrial_inspection.scoring.policy import decide


def _reference(samples, *, index_path, predictions_path, qualification, bundle, manifest_hash,
               legacy_summary_path=None):
    """Check cached scores against raw observations, input bytes and bundle identity."""
    expected = {sample.sample_id: sample for sample in samples}
    validation = bundle.calibration.get("validation", {})
    legacy = None
    if legacy_summary_path is not None:
        legacy = read_json_object(legacy_summary_path)
        bound = bundle.model.get("training", {}).get("evidence_sha256", {})
        if any(bound.get(str(p.resolve())) != sha256_file(p)
               for p in (predictions_path, legacy_summary_path)):
            raise ValueError("Legacy validation sources are not hash-bound to model provenance")
        if (legacy.get("status") != "complete_validation_only"
                or legacy.get("validation_labels") != "normal_only"
                or legacy.get("validation_sample_count") != len(samples)
                or legacy.get("test_prediction_count") != 0
                or legacy.get("manifest_sha256") != manifest_hash
                or legacy.get("runtime_fingerprint") != bundle.calibration["runtime_fingerprint"]
                or legacy.get("checkpoint_sha256") != bundle.model["weights"]["sha256"]
                or any(legacy.get("image_policy", {}).get(key) != value
                       for key, value in bundle.calibration["image"].items())
                or legacy.get("pixel_policy") != bundle.calibration["pixel"]):
            raise ValueError("Legacy validation summary differs from the calibrated bundle")
    if (not expected or len(expected) != len(samples)
            or (legacy is None and set(validation.get("sample_ids", [])) != set(expected))
            or (validation.get("sample_ids") is not None and set(validation["sample_ids"]) != set(expected))
            or validation.get("count") != len(samples)
            or validation.get("manifest_sha256") != manifest_hash
            or validation.get("split") != "validation"
            or (legacy is None and validation.get("normal_only") is not True)
            or any(s.split != "validation" or s.label != "normal" for s in samples)):
        raise ValueError("Require exactly the bundle's normal validation split and manifest")
    required = {"model_fingerprint": bundle.model_fingerprint,
                "runtime_fingerprint": bundle.calibration["runtime_fingerprint"],
                "calibration_id": bundle.calibration_id}
    native_checked = (qualification.get("validation_predictions") == len(samples)
                      and qualification.get("native_cold_cli_api_restart") == "passed")
    if legacy is not None:
        saved = qualification.get("saved_score")
        instances = qualification.get("api_instances", [])
        native_checked = (isinstance(saved, (int, float)) and not isinstance(saved, bool)
                          and math.isfinite(saved) and saved == qualification.get("cold_score")
                          and len(instances) >= 2
                          and all(v.get("ready") is True and v.get("evidence_readback") is True
                                  for v in instances))
    if (qualification.get("status") != "passed" or not native_checked
            or any(qualification.get(key) != value for key, value in required.items())):
        raise ValueError("Baseline qualification does not match the calibrated bundle")
    raw = {}
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in raw:
                raise ValueError("Duplicate raw baseline sample")
            raw[sample_id] = row
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cached = {row["sample_id"]: row for row in rows}
    if len(cached) != len(rows) or set(raw) != set(expected) or set(cached) != set(expected):
        raise ValueError("Cached baseline IDs must exactly cover validation")
    result = {}
    for sample_id, sample in expected.items():
        row, observation = cached[sample_id], raw[sample_id]
        score = float(row["anomaly_score" if legacy is not None else "score"])
        raw_score = observation["score"]
        if (isinstance(raw_score, bool) or not isinstance(raw_score, (float, int))
                or not math.isfinite(score) or not math.isfinite(raw_score)
                or score != raw_score or row["split"] != "validation"
                or row["label"] != "normal"
                or observation["sample_sha256"] != sha256_file(sample.image_path)):
            raise ValueError(f"Cached baseline score/input mismatch: {sample_id}")
        result[sample_id] = {"score": score, "sha256": observation["sample_sha256"],
                             "decision": row["decision"]}
    return result


def _rss_guard(process, limit):
    rss = process.memory_info().rss
    if rss > limit:
        raise MemoryError(f"Sampled process RSS {rss} exceeds configured budget {limit}")
    return rss


def run_validation_stress(*, model_name: str, model_path: Path, manifest_path: Path,
                          project_root: Path, baseline_index: Path, baseline_predictions: Path,
                          qualification_path: Path, output: Path, legacy_summary_path: Path | None = None,
                          max_rss_bytes: int = 2 * 1024**3):
    """CPU normal-only capture stress; no test decoding, calibration or tuning."""
    import cv2

    if type(max_rss_bytes) is not int or max_rss_bytes < 1:
        raise ValueError("RSS budget must be a positive integer")
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    cv2.setNumThreads(2)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(2)
    from industrial_inspection.models.registry import load_model

    conditions = default_perturbations()
    sources = [manifest_path, baseline_index, baseline_predictions, qualification_path]
    if legacy_summary_path is not None:
        sources.append(legacy_summary_path)
    spec = {"schema": "normal-validation-capture-stress-v1", "device": "cpu",
            "model_name": model_name, "artifact": artifact_reference(model_path),
            "sources": {str(p): sha256_file(p) for p in sources},
            "conditions": [{"name": p.name, "severity": p.severity} for p in conditions],
            "code_sha256": {str(p): sha256_file(p) for p in
                            (Path(__file__), Path(apply_perturbation.__code__.co_filename))},
            "baseline": "reuse_saved_scores_no_baseline_predict",
            "resources": {"cpu_math_threads": 2, "max_sampled_rss_bytes": max_rss_bytes,
                          "guard_scope": "after load and each prediction; not an OS peak memory limit"},
            "scope": "model_score_policy_only; no quality gate, pixel metrics or defect sensitivity"}
    reservation = ArtifactReservation.reserve_new(output, spec_fingerprint=digest(spec))
    try:
        write_json_artifact(output / "spec.json", spec)
        samples = [s for s in load_manifest(manifest_path, project_root) if s.split == "validation"]
        inputs = FrozenInputs([*sources, model_path, *(Path(p) for p in spec["code_sha256"]),
                               *(s.image_path for s in samples)],
                              optional_paths=[manifest_sidecar_path(manifest_path)])
        with inputs.scope():
            if any(sha256_file(p) != spec["sources"][str(p)] for p in sources):
                raise ValueError("Reference source changed after protocol freeze")
            model = load_model(model_name, model_path, device="cpu")
            process = psutil.Process()
            rss = [_rss_guard(process, max_rss_bytes)]
            bundle = getattr(model, "bundle", None)
            if bundle is None or any(s.category != model.category for s in samples):
                raise ValueError("Require a bound bundle matching the validation category")
            identity = runtime_identity(model)
            if identity["fingerprint"] != bundle.calibration["runtime_fingerprint"]:
                raise ValueError("Loaded runtime differs from baseline calibration")
            baseline = _reference(samples, index_path=baseline_index,
                                  predictions_path=baseline_predictions,
                                  qualification=read_json_object(qualification_path), bundle=bundle,
                                  manifest_hash=sha256_file(manifest_path), legacy_summary_path=legacy_summary_path)
            policy = model.decision_thresholds
            if any(decide(v["score"], policy).value != v["decision"] for v in baseline.values()):
                raise ValueError("Cached decisions differ from the frozen policy")
            frozen = {"image": policy.as_dict(), "pixel": model.pixel_policy.as_dict(),
                      "runtime_fingerprint": identity["fingerprint"],
                      "calibration_id": bundle.calibration_id,
                      "model_fingerprint": bundle.model_fingerprint}
            write_json_artifact(output / "frozen_policy.json", frozen)
            labels = [0] * len(samples)
            base_scores = [baseline[s.sample_id]["score"] for s in samples]
            summary = {"schema": spec["schema"], "scope": spec["scope"],
                       "category": model.category, "sample_count": len(samples),
                       "baseline_inference_count": 0, "frozen_policy": frozen,
                       "baseline": {"image_level": image_level_metrics(labels, base_scores, policy.fail_threshold),
                                    "review_system": review_system_metrics(labels, base_scores, policy)},
                       "conditions": [], "errors": 0}
            with (output / "predictions.jsonl").open("x", encoding="utf-8") as journal:
                for condition in conditions:
                    records = []
                    for sample in samples:
                        record = {"sample_id": sample.sample_id, "split": "validation", "label": "normal",
                                  "condition": condition.name, "severity": condition.severity,
                                  "source_sha256": baseline[sample.sample_id]["sha256"],
                                  "baseline_score": baseline[sample.sample_id]["score"]}
                        try:
                            start = time.perf_counter()
                            image = load_color_image(sample.image_path)
                            decoded = time.perf_counter()
                            image = apply_perturbation(image, condition)
                            transformed = time.perf_counter()
                            prediction = finalize_prediction(model, model.predict(image), image.shape[:2])
                            stop = time.perf_counter()
                            score = float(prediction.image_score)
                            if not math.isfinite(score):
                                raise ValueError("Non-finite stress score")
                            record.update(score=score, decision=decide(score, policy).value,
                                          score_delta=score - record["baseline_score"],
                                          alignment_succeeded=prediction.metadata.get("alignment_succeeded"),
                                          timing_ms={"decode": (decoded-start)*1000,
                                                     "perturbation": (transformed-decoded)*1000,
                                                     "model_and_finalize": (stop-transformed)*1000})
                        except Exception as exc:
                            record.update(error={"type": type(exc).__name__, "message": str(exc)})
                        journal.write(json.dumps(record, allow_nan=False) + "\n")
                        journal.flush()
                        rss.append(_rss_guard(process, max_rss_bytes))
                        records.append(record)
                    journal.flush()
                    errors = sum("error" in r for r in records)
                    row = {"condition": condition.name, "severity": condition.severity,
                           "attempted": len(records), "errors": errors}
                    if not errors:
                        scores = [r["score"] for r in records]
                        row.update(image_level=image_level_metrics(labels, scores, policy.fail_threshold),
                                   review_system=review_system_metrics(labels, scores, policy),
                                   mean_score_delta=math.fsum(r["score_delta"] for r in records)/len(records),
                                   changed_decisions=sum(r["decision"] != baseline[r["sample_id"]]["decision"]
                                                         for r in records),
                                   alignment_failures=sum(r["alignment_succeeded"] is False for r in records))
                    summary["conditions"].append(row)
                    summary["errors"] += errors
                    print(json.dumps({"condition_complete": condition.name, "severity": condition.severity,
                                      "attempted": len(records), "errors": errors}), flush=True)
            if (runtime_identity(model) != identity or artifact_reference(model_path) != spec["artifact"]
                    or model.decision_thresholds.as_dict() != frozen["image"]
                    or model.pixel_policy.as_dict() != frozen["pixel"]):
                raise ValueError("Model or policy changed during stress measurement")
            inputs.verify_unchanged()
            summary["memory"] = {"max_sampled_process_rss_bytes": max(rss),
                                 "configured_max_sampled_rss_bytes": max_rss_bytes,
                                 "cpu_math_threads": 2,
                                 "scope": "this CPU process, sampled after load and each prediction; not a peak guarantee"}
            summary["completed_predictions"] = len(samples)*len(conditions)-summary["errors"]
            summary["limitations"] = ["normal validation only; no AUROC/AP/sensitivity claim",
                                       "same calibration images; descriptive stress, not independent evaluation",
                                       "unknown item/session groups; no confidence interval or operating-mode selection",
                                       "no test images or GT masks decoded; no data or thresholds changed"]
            write_json_artifact(output / "report.json", summary)
            if summary["errors"]:
                raise ValueError(f"Stress incomplete: {summary['errors']} errors; all attempted rows retained")
        reservation.complete(artifacts=["spec.json", "frozen_policy.json", "predictions.jsonl", "report.json"])
        return summary
    except BaseException as exc:
        reservation.fail(f"{type(exc).__name__}: {exc}")
        raise
