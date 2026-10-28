"""Evaluate an already-fitted model without refitting or touching test data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from industrial_inspection.core.environment import inspect_environment
from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.core.training_protocol import training_environment
from industrial_inspection.data.contracts import (
    InspectionSample,
    load_manifest,
    require_known_labels,
    validate_manifest,
    write_audit_report,
)
from industrial_inspection.data.manifest_datamodule import load_manifest_splits
from industrial_inspection.evaluation.artifacts import ArtifactReservation
from industrial_inspection.evaluation.identity import comparison_identity
from industrial_inspection.evaluation.metrics import latency_summary
from industrial_inspection.evaluation.predictions import (
    calibrate_pixels as _pixel_threshold,
)
from industrial_inspection.evaluation.predictions import (
    predict_samples as _predict,
)
from industrial_inspection.evaluation.report import build_metrics_report, write_evaluation_documents
from industrial_inspection.evaluation.writers import (
    save_failure_gallery,
    write_predictions,
    write_threshold_curve,
)
from industrial_inspection.models.artifacts import artifact_reference
from industrial_inspection.models.base import AnomalyModel, AnomalyPrediction
from industrial_inspection.models.identity_router import runtime_identity
from industrial_inspection.models.registry import _load_native_model
from industrial_inspection.scoring.costs import ReviewCosts
from industrial_inspection.scoring.pixels import calibrated_pixel_policy
from industrial_inspection.scoring.policy import (
    DecisionThresholds,
    select_thresholds,
    validate_normal_pass_quantile,
)


def _split(samples: list[InspectionSample], name: str) -> list[InspectionSample]:
    return [sample for sample in samples if sample.split == name]




def _calibrate_thresholds(
    model: AnomalyModel,
    samples: list[InspectionSample],
    predictions: list[AnomalyPrediction],
    *,
    mode: str,
    normal_pass_quantile: float = 0.95,
) -> tuple[DecisionThresholds, list[dict[str, float]]]:
    labels = [int(sample.label == "anomaly") for sample in samples]
    scores = [prediction.image_score for prediction in predictions]
    if mode == "model":
        defaults = getattr(model, "decision_thresholds", None)
        if defaults is None:
            defaults = getattr(model, "default_thresholds", None)
        if defaults is None:
            raise ValueError("Model threshold mode is unavailable for this runtime")
        return defaults, []
    if mode != "validation":
        raise ValueError("threshold mode must be 'validation' or 'model'")
    return select_thresholds(labels, scores, normal_pass_quantile=normal_pass_quantile)








def evaluate_saved_model(
    *,
    model: AnomalyModel,
    manifest_path: Path,
    output: Path,
    project_root: Path,
    model_name: str,
    model_path: Path,
    experiment_id: str,
    seed: int = 42,
    bootstrap_samples: int = 500,
    review_costs: ReviewCosts | None = None,
    threshold_mode: str = "validation",
    normal_pass_quantile: float = 0.95,
) -> dict[str, Any]:
    """Calibrate on validation and evaluate a fitted model once on the test split."""

    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    validate_normal_pass_quantile(normal_pass_quantile)
    splits = load_manifest_splits(manifest_path, project_root=project_root)
    categories = {sample.category for sample in splits.train + splits.validation + splits.test}
    if len(categories) != 1 or (getattr(model, "category", None) is not None and categories != {model.category}):
        raise ValueError("Evaluation category differs from the single loaded model")
    if getattr(model, "bundle", None) is not None and threshold_mode != "model":
        raise ValueError("A bound bundle must use threshold_mode='model'; recalibration requires a new bundle")
    if getattr(model, "bundle", None) is not None and model.bundle.calibration["validation"]["split_fingerprint"] != splits.fingerprint:
        raise ValueError("Evaluation manifest differs from the bound calibration split fingerprint")
    identity = runtime_identity(model)
    native_path = model.bundle.artifact_path if getattr(model, "bundle", None) else model_path
    cold = _load_native_model(model_name, native_path, device=str(getattr(model, "device", "cpu")))
    if runtime_identity(cold) != identity:
        raise ValueError("Evaluation runtime differs from the supplied saved model")
    del cold
    model.runtime_identity = identity
    spec = {
        "experiment_id": experiment_id, "model_name": model_name,
        "weights": artifact_reference(native_path), "split_fingerprint": splits.fingerprint,
        "runtime_identity": identity,
        "seed": seed, "bootstrap_samples": bootstrap_samples,
        "review_costs": review_costs.as_dict() if review_costs else None, "threshold_mode": threshold_mode,
        "environment": training_environment(project_root),
        "normal_pass_quantile": normal_pass_quantile if threshold_mode == "validation" else None,
        "pixel_calibration": {"pixel_policy_version": 2, "threshold_operator": ">", "quantile": .995}
        if threshold_mode == "validation" else None,
    }
    encoded = json.dumps(spec, sort_keys=True, allow_nan=False)
    reservation = ArtifactReservation.reserve_new(output, spec_fingerprint=hashlib.sha256(encoded.encode()).hexdigest())
    try:
        (output / "resolved_spec.json").write_text(encoded + "\n", encoding="utf-8")
        splits.write_provenance(output / "manifest_provenance.json")
        with splits.inputs.scope():
            result = _evaluate_saved_model(
                model=model, manifest_path=manifest_path, output=output, project_root=project_root,
                model_name=model_name, model_path=model_path, experiment_id=experiment_id, seed=seed,
                bootstrap_samples=bootstrap_samples, threshold_mode=threshold_mode,
                split_fingerprint=splits.fingerprint, review_costs=review_costs,
                normal_pass_quantile=normal_pass_quantile,
            )
        splits.verify_unchanged()
        reservation.complete(artifacts=["metrics.json", "predictions.csv", "validation_predictions.csv",
                                        "frozen_policy.json", "manifest_provenance.json", "resolved_spec.json"])
        return result
    except BaseException as exc:
        reservation.fail(f"{type(exc).__name__}: {exc}")
        raise


def _evaluate_saved_model(
    *, model: AnomalyModel, manifest_path: Path, output: Path, project_root: Path,
    model_name: str, model_path: Path, experiment_id: str, seed: int,
    bootstrap_samples: int, threshold_mode: str, split_fingerprint: str, review_costs: ReviewCosts | None,
    normal_pass_quantile: float,
) -> dict[str, Any]:

    audit = validate_manifest(manifest_path, project_root)
    if not audit.valid:
        raise ValueError(f"Manifest audit failed: {audit.errors}")
    samples = load_manifest(manifest_path, project_root)
    require_known_labels(samples)
    validation = _split(samples, "validation")
    test = _split(samples, "test")
    if not validation or not test:
        raise ValueError("validation and test splits must both be non-empty")

    output.mkdir(parents=True, exist_ok=True)
    write_audit_report(audit, output / "dataset_audit.json")
    if getattr(model, "bundle", None) is not None and threshold_mode != "model":
        raise ValueError("A bound bundle must use threshold_mode='model'; recalibration requires a new bundle")
    validation_predictions, validation_latencies = _predict(model, validation, output=output / "validation_predictions")
    thresholds, threshold_curve = _calibrate_thresholds(
        model, validation, validation_predictions, mode=threshold_mode, normal_pass_quantile=normal_pass_quantile
    )
    if threshold_mode == "model":
        policy = getattr(model, "pixel_policy", None)
        if policy is None:
            raise ValueError("Model threshold mode requires a bound pixel policy")
        pixel_threshold = policy.threshold
    else:
        pixel_threshold = _pixel_threshold(validation, validation_predictions, output=output / "pixel_calibration.json")
        model.pixel_policy = calibrated_pixel_policy(model, pixel_threshold)
    (output / "frozen_policy.json").write_text(
        json.dumps({"image": thresholds.as_dict(), "pixel": model.pixel_policy.as_dict(),
                    "runtime_fingerprint": model.runtime_identity["fingerprint"],
                    "calibration_id": getattr(model, "calibration_id", None)}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_predictions(output / "validation_predictions.csv", validation, validation_predictions, thresholds)
    test_predictions, test_latencies = _predict(model, test, output=output / "test_predictions")
    report = build_metrics_report(test, test_predictions, thresholds, pixel_threshold,
                                  bootstrap_samples=bootstrap_samples, seed=seed, costs=review_costs)
    metrics: dict[str, Any] = {
        "experiment_id": experiment_id,
        "comparison_identity": comparison_identity(model, split_fingerprint=split_fingerprint,
                                                   category=samples[0].category, thresholds=thresholds),
        "model": {"name": model_name, "model_id": getattr(model, "model_id", model_path.stem)},
        "dataset": {"name": samples[0].dataset, "categories": sorted({sample.category for sample in test})},
        **report,
    }
    timings = {
        "validation_inference": latency_summary(validation_latencies),
        "test_inference": latency_summary(test_latencies),
        "fit_ms": None,
        "fit_note": "Model was fitted externally; this evaluator does not refit it.",
    }
    environment = inspect_environment(project_root)
    config_snapshot = {
        "experiment_id": experiment_id,
        "dataset": samples[0].dataset,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "model": {
            "name": model_name,
            "model_id": getattr(model, "model_id", model_path.stem),
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path) if model_path.is_file() else None,
        },
        "seed": seed,
        "threshold_policy": thresholds.as_dict(),
        "normal_pass_quantile": normal_pass_quantile if threshold_mode == "validation" else None,
        "review_costs": review_costs.as_dict() if review_costs else None,
        "threshold_mode": threshold_mode,
        "pixel_threshold": pixel_threshold,
        "split_counts": {
            "train": len(_split(samples, "train")),
            "validation": len(validation),
            "test": len(test),
        },
    }
    (output / "thresholds.json").write_text(json.dumps({**thresholds.as_dict(), "runtime_fingerprint": model.runtime_identity["fingerprint"], "pixel_threshold": pixel_threshold, "pixel": model.pixel_policy.as_dict()}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_predictions(output / "predictions.csv", test, test_predictions, thresholds)
    write_threshold_curve(output / "threshold_curve.csv", threshold_curve)
    gallery = save_failure_gallery(output / "failure_cases", test, test_predictions, thresholds)
    write_evaluation_documents(output, metrics=metrics, timings=timings, config=config_snapshot,
                               environment=environment, gallery=gallery, fitted=False)
    return {"metrics": metrics, "timings": timings, "thresholds": thresholds.as_dict()}
