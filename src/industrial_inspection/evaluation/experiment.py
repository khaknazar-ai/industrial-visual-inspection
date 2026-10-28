"""Reproducible train/calibrate/test experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from industrial_inspection.core.environment import inspect_environment
from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.core.training_protocol import seed_training, training_environment
from industrial_inspection.data.manifest_datamodule import ManifestSplits, load_manifest_splits
from industrial_inspection.data.observations import SampleObservations, observe_model_inputs
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
from industrial_inspection.models.registry import create_model
from industrial_inspection.scoring.costs import ReviewCosts
from industrial_inspection.scoring.pixels import calibrated_pixel_policy
from industrial_inspection.scoring.policy import select_thresholds, validate_normal_pass_quantile

_EXPERIMENT_ID = re.compile(r"^EXP-\d{4}$")


def run_experiment(
    *,
    manifest_path: Path,
    model_name: str,
    model_id: str,
    model_parameters: dict[str, Any],
    model_output: Path,
    experiment_output: Path,
    experiment_id: str,
    project_root: Path,
    seed: int = 42,
    bootstrap_samples: int = 500,
    review_costs: ReviewCosts | None = None,
    normal_pass_quantile: float = 0.95,
) -> dict[str, Any]:
    """Fit on normal train, calibrate on validation, and evaluate once on test."""

    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    validate_normal_pass_quantile(normal_pass_quantile)

    if not _EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError("experiment_id must have the form EXP-0001")
    experiment_output, model_output = experiment_output.resolve(), model_output.resolve()
    if experiment_output.exists() or model_output.exists():
        raise FileExistsError("Experiment and model output must both be new directories")
    if experiment_output.is_relative_to(model_output) or model_output.is_relative_to(experiment_output):
        raise ValueError("Model and experiment output directories must not overlap")
    splits = load_manifest_splits(manifest_path, project_root=project_root)
    if len({sample.category for sample in splits.train + splits.validation + splits.test}) != 1:
        raise ValueError("Fit one category per experiment")
    model_parameters = dict(model_parameters)
    if model_name in {"deep_patch_memory", "dino_memory"} and model_parameters.get("pretrained", True):
        encoder = model_parameters.get("encoder_weights")
        if encoder is None or not Path(encoder).is_file():
            raise FileNotFoundError("Frozen encoder fit requires local encoder_weights before reserving a run")
    if model_name in {"patch_memory", "deep_patch_memory", "dino_memory"}:
        if model_parameters.get("seed", seed) != seed:
            raise ValueError("Model seed must match experiment seed")
        model_parameters["seed"] = seed
    reproducibility = seed_training(seed, lightning_workers=False)
    spec = {
        "experiment_id": experiment_id, "model_name": model_name, "model_id": model_id,
        "parameters": model_parameters, "split_fingerprint": splits.fingerprint,
        "seed": seed, "bootstrap_samples": bootstrap_samples,
        "review_costs": review_costs.as_dict() if review_costs else None,
        "reproducibility": reproducibility, "environment": training_environment(project_root),
        "encoder_sha256": sha256_file(Path(model_parameters["encoder_weights"])) if model_name in {"deep_patch_memory", "dino_memory"} and model_parameters.get("pretrained", True) else None,
        "calibration": {"split": "validation", "normal_pass_quantile": normal_pass_quantile,
                        "normal_fail_quantile": 0.995, "normal_pixel_quantile": 0.995,
                        "pixel_policy_version": 2, "pixel_threshold_operator": ">"},
    }
    spec_json = json.dumps(spec, sort_keys=True, allow_nan=False)
    fingerprint = hashlib.sha256(spec_json.encode()).hexdigest()
    reservation = ArtifactReservation.reserve_new(experiment_output, spec_fingerprint=fingerprint)
    model_reservation = None
    try:
        model_reservation = ArtifactReservation.reserve_new(model_output, spec_fingerprint=fingerprint)
        (experiment_output / "resolved_spec.json").write_text(spec_json + "\n", encoding="utf-8")
        splits.write_provenance(experiment_output / "manifest_provenance.json")
        observations = SampleObservations(splits, experiment_output / "observations")
        with splits.inputs.scope():
            result = _run_experiment(
                manifest_path=manifest_path, model_name=model_name, model_id=model_id,
                model_parameters=model_parameters, model_output=model_output,
                experiment_output=experiment_output, experiment_id=experiment_id,
                project_root=project_root, seed=seed, bootstrap_samples=bootstrap_samples,
                splits=splits, observations=observations, review_costs=review_costs,
                expected_encoder_hash=spec["encoder_sha256"],
                normal_pass_quantile=normal_pass_quantile,
            )
        splits.verify_unchanged()
        model_reservation.complete(artifacts=["model.json", "thresholds.json", "training_manifest.sha256"])
        reservation.complete(artifacts=["metrics.json", "observed_samples.json", "manifest_provenance.json",
                                        "resolved_spec.json", "predictions.csv", "failure_cases"])
        return result
    except BaseException as exc:
        reservation.fail(f"{type(exc).__name__}: {exc}")
        if model_reservation is not None:
            model_reservation.fail(f"{type(exc).__name__}: {exc}")
        raise


def _run_experiment(
    *, manifest_path: Path, model_name: str, model_id: str, model_parameters: dict[str, Any],
    model_output: Path, experiment_output: Path, experiment_id: str, project_root: Path,
    seed: int, bootstrap_samples: int, splits: ManifestSplits, observations: SampleObservations,
    expected_encoder_hash: str | None, review_costs: ReviewCosts | None, normal_pass_quantile: float,
) -> dict[str, Any]:
    train, validation, test = list(splits.train), list(splits.validation), list(splits.test)
    samples = train + validation + test
    model = create_model(model_name, model_id, model_parameters)
    if expected_encoder_hash is not None and getattr(model, "encoder_sha256", None) != expected_encoder_hash:
        raise ValueError("Encoder weights changed after the experiment spec was frozen")
    fit_started = time.perf_counter()
    with observe_model_inputs(observations, stage="fit_input"):
        model.fit([sample.image_path for sample in train])
    fit_ms = (time.perf_counter() - fit_started) * 1000
    with observe_model_inputs(observations, stage="calibration_input"):
        validation_predictions, validation_latencies = _predict(model, validation, output=experiment_output / "validation_predictions")
    observed = observations.write_report(experiment_output / "observed_samples.json")
    if set(observed["events"]["fit_input:train"]["sample_sha256"]) != set(splits.ids("train")):
        raise ValueError("Fit did not read every declared training sample")
    validation_labels = [int(sample.label == "anomaly") for sample in validation]
    validation_scores = [prediction.image_score for prediction in validation_predictions]
    thresholds, threshold_curve = select_thresholds(validation_labels, validation_scores, normal_pass_quantile=normal_pass_quantile)

    pixel_threshold = _pixel_threshold(validation, validation_predictions, output=experiment_output / "pixel_calibration.json")
    model.pixel_policy = calibrated_pixel_policy(model, pixel_threshold)
    frozen_policy = {"image": thresholds.as_dict(), "pixel": model.pixel_policy.as_dict(),
                     "split_fingerprint": splits.fingerprint, "calibration_split": "validation"}
    (experiment_output / "frozen_policy.json").write_text(json.dumps(frozen_policy, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_predictions(experiment_output / "validation_predictions.csv", validation, validation_predictions, thresholds)
    test_predictions, test_latencies = _predict(model, test, output=experiment_output / "test_predictions")
    report = build_metrics_report(test, test_predictions, thresholds, pixel_threshold,
                                  bootstrap_samples=bootstrap_samples, seed=seed, costs=review_costs)
    metrics = {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "threshold_source": thresholds.source,
        "comparison_identity": comparison_identity(model, split_fingerprint=splits.fingerprint,
                                                   category=samples[0].category, thresholds=thresholds),
        **report,
    }
    timings = {
        "fit_ms": fit_ms,
        "validation_inference": latency_summary(validation_latencies),
        "test_inference": latency_summary(test_latencies),
    }
    config_snapshot = {
        "experiment_id": experiment_id,
        "dataset": samples[0].dataset,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "model": {"name": model_name, "model_id": model_id, "parameters": model_parameters},
        "seed": seed,
        "threshold_policy": thresholds.as_dict(),
        "normal_pass_quantile": normal_pass_quantile,
        "review_costs": review_costs.as_dict() if review_costs else None,
        "pixel_threshold": pixel_threshold,
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
    }

    model.save(model_output)
    (model_output / "thresholds.json").write_text(
        json.dumps({**thresholds.as_dict(), "pixel": model.pixel_policy.as_dict()}, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (model_output / "training_manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    write_predictions(experiment_output / "predictions.csv", test, test_predictions, thresholds)
    write_threshold_curve(experiment_output / "threshold_curve.csv", threshold_curve)
    gallery = save_failure_gallery(
        experiment_output / "failure_cases", test, test_predictions, thresholds
    )
    write_evaluation_documents(experiment_output, metrics=metrics, timings=timings,
                               config=config_snapshot, environment=inspect_environment(project_root),
                               gallery=gallery, fitted=True)
    return {"metrics": metrics, "timings": timings, "thresholds": thresholds.as_dict()}
