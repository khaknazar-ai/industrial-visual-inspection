"""Controlled perturbation benchmark for lighting and capture robustness."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from industrial_inspection.models.base import AnomalyModel

PERTURBATIONS = {"brightness", "contrast", "blur", "jpeg", "resize", "rotation", "translation", "color_shift"}


@dataclass(frozen=True, slots=True)
class Perturbation:
    name: str
    severity: float

    def __post_init__(self):
        if self.name not in PERTURBATIONS:
            raise ValueError(f"Unknown perturbation: {self.name}")
        if isinstance(self.severity, bool) or not math.isfinite(self.severity) or not -1 <= self.severity <= 1:
            raise ValueError("Perturbation severity must be finite and in [-1, 1]")


def default_perturbations() -> list[Perturbation]:
    return [Perturbation(name, severity) for name, severities in (
        ("brightness", (-1., 1.)), ("contrast", (-1., 1.)), ("blur", (.5, 1.)),
        ("jpeg", (.5, 1.)), ("resize", (.5, 1.)), ("rotation", (1.,)),
        ("translation", (1.,)), ("color_shift", (1.,)),
    ) for severity in severities]


def apply_perturbation(image: np.ndarray, perturbation: Perturbation) -> np.ndarray:
    """Apply a labelled capture perturbation, not a synthetic defect."""

    severity = float(perturbation.severity)
    if severity == 0:
        return image.copy()
    if perturbation.name == "brightness":
        return np.clip(image.astype(np.float32) + 45.0 * severity, 0, 255).astype(np.uint8)
    if perturbation.name == "contrast":
        return np.clip((image.astype(np.float32) - 127.5) * (1 + 0.7 * severity) + 127.5, 0, 255).astype(np.uint8)
    if perturbation.name == "blur":
        kernel = max(3, int(round(3 + abs(severity) * 8)) | 1)
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    if perturbation.name == "jpeg":
        quality = int(np.clip(100 - 55 * abs(severity), 5, 100))
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise ValueError("JPEG perturbation encoding failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("JPEG perturbation decoding failed")
        return decoded
    if perturbation.name == "resize":
        scale = float(np.clip(1.0 - 0.55 * abs(severity), 0.2, 1.0))
        height, width = image.shape[:2]
        reduced = cv2.resize(image, (max(2, round(width * scale)), max(2, round(height * scale))))
        return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
    if perturbation.name == "rotation":
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 5 * severity, 1.0)
        return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    if perturbation.name == "translation":
        matrix = np.float32([[1, 0, round(5 * severity)], [0, 1, round(5 * severity)]])
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)
    if perturbation.name == "color_shift":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + round(12 * severity)) % 180
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    raise ValueError(f"Unknown perturbation: {perturbation.name}")


def evaluate_perturbations(
    model: AnomalyModel,
    images: Sequence[np.ndarray],
    labels: Sequence[int],
    perturbations: Sequence[Perturbation],
) -> list[dict[str, Any]]:
    if not images or len(images) != len(labels):
        raise ValueError("images and labels must have equal non-zero length")
    rows: list[dict[str, Any]] = []
    for perturbation in perturbations:
        scores = [model.predict(apply_perturbation(image, perturbation)).image_score for image in images]
        if not np.isfinite(scores).all():
            raise ValueError("Model returned non-finite scores")
        rows.append(
            {
                "perturbation": perturbation.name,
                "severity": perturbation.severity,
                "scores": scores,
                "labels": list(labels),
                "mean_score": float(np.mean(scores)),
            }
        )
    return rows


def run_robustness(
    *, model_name: str, model_path: Path, manifest_path: Path, project_root: Path,
    output: Path, device: str = "cpu", split: str = "test", max_images: int = 0,
    threshold: float | None = None, perturbations: Sequence[Perturbation] | None = None,
    review_costs=None,
) -> dict[str, Any]:
    """Measure image decisions under capture shifts with an unchanged saved policy."""
    from industrial_inspection.core.reproducibility import sha256_file
    from industrial_inspection.data.manifest_datamodule import load_manifest_splits
    from industrial_inspection.evaluation.artifacts import ArtifactReservation, publish_files
    from industrial_inspection.evaluation.metrics import image_level_metrics, review_system_metrics
    from industrial_inspection.evaluation.report import write_json_artifact
    from industrial_inspection.models.artifacts import artifact_reference
    from industrial_inspection.models.base import load_color_image
    from industrial_inspection.models.identity_router import digest, runtime_identity
    from industrial_inspection.models.registry import load_model
    from industrial_inspection.scoring.pixels import finalize_prediction
    from industrial_inspection.scoring.policy import decide

    output = output.resolve()
    csv_path, run_root = output.with_suffix(".csv"), output.with_suffix(".run")
    if output.suffix.lower() != ".json":
        raise ValueError("Robustness output must have a .json suffix")
    for path in (output, csv_path, run_root):
        if path.exists():
            raise FileExistsError(f"Robustness output already exists: {path}")
    if split not in {"validation", "test"} or type(max_images) is not int or max_images < 0:
        raise ValueError("Use validation/test and a non-negative integer max_images")
    conditions = list(default_perturbations() if perturbations is None else perturbations)
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Provide unique, non-empty perturbations")
    splits = load_manifest_splits(manifest_path, project_root=project_root)
    categories = {sample.category for sample in splits.train + splits.validation + splits.test}
    if len(categories) != 1:
        raise ValueError("Robustness requires one model category")
    category = next(iter(categories))
    samples = list(getattr(splits, split))
    available = len(samples)
    if max_images:
        samples = samples[:max_images]
    hashes = {s.sample_id: sha256_file(s.image_path) for s in samples}
    spec = {"model_name": model_name, "artifact": artifact_reference(model_path),
            "split_fingerprint": splits.fingerprint, "split": split, "category": category,
            "sample_sha256": hashes, "device": device, "threshold_assertion": threshold,
            "perturbations": [asdict(p) for p in conditions],
            "review_costs": review_costs.as_dict() if review_costs else None,
            "selection": {"method": "sample_id_sorted_first_n", "max_images": max_images,
                          "available": available, "selected": len(samples)}}
    reservation = ArtifactReservation.reserve_new(run_root, spec_fingerprint=digest(spec))
    try:
        write_json_artifact(run_root / "spec.json", spec)
        model = load_model(model_name, model_path, device=device)
        if getattr(model, "category", category) != category:
            raise ValueError("Robustness category differs from loaded model")
        policy = getattr(model, "decision_thresholds", None)
        if policy is None:
            raise ValueError("Robustness requires saved validation thresholds; build a calibrated bundle first")
        if threshold is not None and threshold != policy.fail_threshold:
            raise ValueError("Threshold override requires a new calibration bundle")
        identity = runtime_identity(model)
        pixel = getattr(model, "pixel_policy", None)
        frozen = {"image": policy.as_dict(), "pixel": pixel.as_dict() if pixel else None,
                  "runtime_fingerprint": identity["fingerprint"],
                  "calibration_id": getattr(model, "calibration_id", None)}
        write_json_artifact(run_root / "frozen_policy.json", frozen)
        labels = [int(s.label == "anomaly") for s in samples]
        rows = []
        with (run_root / "predictions.jsonl").open("x", encoding="utf-8") as journal:
            for condition in [None, *conditions]:
                scores = []
                for sample in samples:
                    if sha256_file(sample.image_path) != hashes[sample.sample_id]:
                        raise ValueError("Robustness input changed after split freeze")
                    image = load_color_image(sample.image_path)
                    if condition is not None:
                        image = apply_perturbation(image, condition)
                    prediction = finalize_prediction(model, model.predict(image), image.shape[:2])
                    score = float(prediction.image_score)
                    if sha256_file(sample.image_path) != hashes[sample.sample_id]:
                        raise ValueError("Robustness input changed during prediction")
                    scores.append(score)
                    journal.write(json.dumps({"perturbation": condition.name if condition else "baseline",
                                              "severity": condition.severity if condition else 0.,
                                              "sample_id": sample.sample_id, "label": sample.label,
                                              "sha256": hashes[sample.sample_id], "score": score,
                                              "decision": decide(score, policy).value}, allow_nan=False) + "\n")
                image_metrics = image_level_metrics(labels, scores, policy.fail_threshold)
                row = {"perturbation": condition.name if condition else "baseline",
                       "severity": condition.severity if condition else 0.,
                       "mean_score": math.fsum(score / len(scores) for score in scores),
                       "image_level": image_metrics,
                       "review_system": review_system_metrics(labels, scores, policy, costs=review_costs)}
                row["delta_from_baseline"] = {
                    key: image_metrics[key] - rows[0]["image_level"][key]
                    if rows and image_metrics[key] is not None and rows[0]["image_level"][key] is not None else None
                    for key in ("roc_auc", "average_precision", "f1", "false_positive_rate", "false_negative_rate")}
                rows.append(row)
        if runtime_identity(model) != identity or artifact_reference(model_path) != spec["artifact"]:
            raise ValueError("Model changed during robustness measurement")
        current_pixel = getattr(model, "pixel_policy", None)
        if model.decision_thresholds.as_dict() != frozen["image"] or (current_pixel.as_dict() if current_pixel else None) != frozen["pixel"]:
            raise ValueError("Policy changed during robustness measurement")
        if load_manifest_splits(manifest_path, project_root=project_root).fingerprint != splits.fingerprint:
            raise ValueError("Manifest changed during robustness measurement")
        report = {"schema_version": 2, "model": {"name": model_name, "model_id": model.model_id,
                  "artifact": spec["artifact"], "runtime_identity": identity},
                  "manifest": {"sha256": splits.manifest_sha256, "fingerprint": splits.fingerprint},
                  "split": split, "category": category, "sample_count": len(samples),
                  "selection": spec["selection"], "frozen_policy": frozen,
                  "threshold": policy.fail_threshold, "baseline": rows[0]["image_level"],
                  "baseline_review_system": rows[0]["review_system"], "perturbations": rows[1:],
                  "scope": "image decisions on predecoded, perturbed arrays; no pixel robustness measurement",
                  "note": "Capture shifts retain original image labels. This is a paired stress test, not new defects or a deployment acceptance gate."}
        staged_json, staged_csv = run_root / "report.json", run_root / "report.csv"
        write_json_artifact(staged_json, report)
        with staged_csv.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["perturbation", "severity", "mean_score", "roc_auc", "average_precision", "f1", "review_rate", "expected_cost_per_image"])
            for row in rows:
                writer.writerow([row["perturbation"], row["severity"], row["mean_score"],
                                 *[row["image_level"][key] for key in ("roc_auc", "average_precision", "f1")],
                                 row["review_system"]["review_rate"], row["review_system"]["expected_cost_per_image"]])
        with publish_files({staged_json: output, staged_csv: csv_path}):
            reservation.complete(artifacts=["spec.json", "frozen_policy.json", "predictions.jsonl", "report.json", "report.csv"])
        return report
    except BaseException as exc:
        reservation.fail(f"{type(exc).__name__}: {exc}")
        raise
