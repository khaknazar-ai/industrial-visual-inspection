"""Package PatchCore only from matched fit, evaluation and runtime evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.evaluation.artifacts import (
    ArtifactReservation,
    publish_files,
    read_json_object,
)
from industrial_inspection.evaluation.metrics import image_level_metrics, review_system_metrics
from industrial_inspection.evaluation.report import write_json_artifact
from industrial_inspection.models.artifacts import ModelBundle, artifact_reference
from industrial_inspection.models.registry import _load_bundle_runtime, load_model
from industrial_inspection.models.runtime_identity import digest, runtime_identity
from industrial_inspection.scoring.costs import ReviewCosts
from industrial_inspection.scoring.policy import DecisionThresholds, decide


def _completed_spec(directory: Path) -> dict:
    status = read_json_object(directory / "run_status.json")
    spec = read_json_object(directory / "resolved_spec.json")
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if status.get("status") != "complete" or status.get("spec_fingerprint") != fingerprint:
        raise ValueError(f"Requires a completed run with matching spec: {directory}")
    return spec


def _provenance_rows(provenance):
    rows = {}
    for split in ("train", "validation", "test"):
        data = provenance["splits"][split]
        ids, hashes = data["sample_ids"], data["image_sha256"]
        if not ids or len(ids) != data["count"] or len(set(ids)) != len(ids):
            raise ValueError("Invalid sample counts in provenance")
        rows[split] = dict(zip(ids, hashes, strict=True))
    if sum(map(len, rows.values())) != len(set().union(*rows.values())):
        raise ValueError("Provenance reuses sample IDs across splits")
    return rows


def _validate_fit(fit, provenance, fit_spec, observed, checkpoint_hash, category):
    split_fingerprint = provenance["manifest"]["fingerprint"]
    if fit.get("format_version") != 2 or fit.get("checkpoint", {}).get("sha256") != checkpoint_hash:
        raise ValueError("Fit evidence must bind observed training to these checkpoint weights; revalidate legacy artifacts")
    fingerprint = hashlib.sha256(json.dumps(fit_spec, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if fit.get("spec_fingerprint") != fingerprint or fit.get("model") != {"name": "patchcore", **fit_spec["config"]}:
        raise ValueError("Fit metadata differs from the executed model spec")
    if fit_spec["config"]["no_pre_trained"]:
        if fit.get("encoder", {}).get("source") != "untrained_fixture":
            raise ValueError("Untrained fit must be identified as a fixture")
    elif fit.get("encoder", {}).get("sha256") != fit_spec.get("encoder_sha256"):
        raise ValueError("Encoder identity differs from fit spec")
    if fit.get("category") != category or fit_spec.get("category") != category:
        raise ValueError("Fit category differs from requested category")
    if fit.get("split_fingerprint") != split_fingerprint or fit_spec.get("split_fingerprint") != split_fingerprint:
        raise ValueError("Fit split identity differs from evaluation provenance")
    if fit.get("observed_samples") != observed or observed.get("split_fingerprint") != split_fingerprint:
        raise ValueError("Fit observations differ from the recorded split")
    policy = fit.get("training_policy", {})
    if policy.get("fit_split") != "train" or policy.get("fit_labels") != ["normal"] or policy.get("test_seen_during_fit") is not False or observed.get("test_seen_during_fit") is not False:
        raise ValueError("Fit evidence does not confirm normal-only training without test consumption")
    expected = _provenance_rows(provenance)
    if fit.get("counts") != {key: len(value) for key, value in expected.items()}:
        raise ValueError("Fit sample counts differ from provenance")
    events = observed.get("events", {})
    for name, record in events.items():
        event, _, split = name.partition(":")
        allowed = {"read": {"train", "validation"}, "training_batch": {"train"}, "validation_batch": {"validation"}}
        if split not in allowed.get(event, set()):
            raise ValueError("Fit observations contain forbidden sample consumption")
        seen = record["sample_sha256"]
        if any(expected[split].get(key) != value for key, value in seen.items()):
            raise ValueError("Observed training image hash differs from provenance")
        if record.get("unique_count") != len(seen) or record.get("event_count", 0) < len(seen):
            raise ValueError("Invalid observed training counts")
    for event, split in (("training_batch", "train"), ("validation_batch", "validation")):
        if events.get(f"{event}:{split}", {}).get("sample_sha256") != expected[split]:
            raise ValueError("Fit observations do not cover the frozen training/validation samples")
    return expected


def _prediction_rows(path, expected, category, split, policy):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected) or {r["sample_id"] for r in rows} != set(expected):
        raise ValueError(f"Prediction IDs differ from provenance: {path}")
    for row in rows:
        score = float(row["anomaly_score"])
        if row["category"] != category or row["split"] != split or row["label"] not in {"normal", "anomaly"}:
            raise ValueError("Prediction category/split/label differs from protocol")
        if not math.isfinite(score) or row["decision"] != decide(score, policy).value:
            raise ValueError("Prediction decision differs from frozen policy")
    return rows


def _number(value):
    return "NA" if value is None else f"{value:.4f}"


def model_card(*, model_id, category, fit, metrics, timings, frozen, provenance, comparisons, robustness):
    image, pixel, review = metrics["image_level"], metrics.get("pixel_level") or {}, metrics["review_system"]
    counts = {name: data["count"] for name, data in provenance["splits"].items()}
    latency = timings["test_inference"]
    lines = [f"# PatchCore {category}: {model_id}", "",
             "Status: evaluated candidate with verified fit/evaluation provenance and cold-loaded weights.", "",
             f"Dataset: {metrics['dataset']['name']}; category: {category}; evaluation: {metrics['experiment_id']}.",
             f"Split fingerprint: `{provenance['manifest']['fingerprint']}`.",
             f"Samples: train {counts['train']}, validation {counts['validation']}, test {counts['test']}.",
             "Observed normal-only training and validation normalization; no test reads in fit observations.",
             f"Backbone: {fit['model']['backbone']}; layers: {fit['model']['layers']}.",
             f"Encoder source: {fit['encoder'].get('source', 'local pretrained weights')}.", "",
             "## Results", "",
             f"Image ROC-AUC/AP: {_number(image.get('roc_auc'))} / {_number(image.get('average_precision'))}.",
             f"Image F1: {_number(image.get('f1'))}; confusion matrix: `{image['confusion_matrix']}`.",
             f"Pixel ROC-AUC/AP: {_number(pixel.get('roc_auc'))} / {_number(pixel.get('average_precision'))}.",
             f"Pixel Dice/IoU: {_number(pixel.get('dice'))} / {_number(pixel.get('iou'))}.",
             f"Automation/review: {_number(review.get('automation_rate'))} / {_number(review.get('review_rate'))}.",
             f"Test inference p50/p95: {_number(latency.get('p50_ms'))} / {_number(latency.get('p95_ms'))} ms; device: {metrics['comparison_identity']['device']}.",
             "These timings belong to the evaluation; no separate warm benchmark or resource peak is inferred.", "",
             "## Calibration", "", f"Image policy: `{json.dumps(frozen['image'], allow_nan=False)}`.",
             f"Pixel policy: `{json.dumps(frozen['pixel'], allow_nan=False)}`.",
             f"Runtime fingerprint: `{frozen['runtime_fingerprint']}`.", "",
             "## Limits", "", "NA means undefined or unmeasured, not zero.",
             "Evidence covers this dataset, category and split only; it does not establish performance on production or other categories.",
             "The package includes scalar evaluation reports and prediction tables; original images, dense maps and galleries remain in the source experiment."]
    if fit["encoder"].get("source") == "untrained_fixture":
        lines.append("The encoder was randomly initialized for a fixture. These numbers are not pretrained-model quality evidence.")
    if comparisons:
        lines.append("A comparison table is included as a reference only; its rows are not independently validated by this packaging run.")
    if robustness:
        lines.append("Included robustness evidence matches the runtime and image/pixel policy; its own scope and selected samples apply.")
    else:
        lines.append("No matching robustness report was supplied.")
    return "\n\n".join(lines) + "\n"


def build_patchcore_bundle(
    *, checkpoint: Path, fit_metadata: Path, evaluation: Path, manifest_provenance: Path,
    output: Path, model_card_path: Path, model_id: str, category: str,
    device: str = "cpu", comparison: Path | None = None, robustness: Path | None = None,
) -> dict:
    output, model_card_path = output.resolve(), model_card_path.resolve()
    if output.exists() or model_card_path.exists():
        raise FileExistsError("Bundle output and model card must be new paths")
    if model_card_path.is_relative_to(output) and model_card_path != output / "MODEL_CARD.md":
        raise ValueError("Inside a bundle the model card must be MODEL_CARD.md")
    if output.is_relative_to(model_card_path):
        raise ValueError("Model card path cannot contain the bundle output")
    checkpoint, fit_metadata, evaluation = checkpoint.resolve(), fit_metadata.resolve(), evaluation.resolve()
    fit_root = fit_metadata.parent
    inputs = {"fit/fit_metadata.json": fit_metadata, "fit/manifest_provenance.json": fit_root / "manifest_provenance.json",
              "fit/resolved_spec.json": fit_root / "resolved_spec.json", "fit/run_status.json": fit_root / "run_status.json",
              "fit/observed_samples.json": fit_root / "observed_samples.json", "manifest_provenance.json": manifest_provenance}
    for name in ("metrics.json", "timings.json", "thresholds.json", "frozen_policy.json", "resolved_spec.json",
                 "run_status.json", "manifest_provenance.json", "predictions.csv", "validation_predictions.csv", "environment.json", "config.yaml"):
        inputs[f"evaluation/{name}"] = evaluation / name
    if comparison:
        inputs["comparison.md"] = comparison
    if robustness:
        inputs["robustness.json"] = robustness
        inputs["robustness_run_status.json"] = robustness.with_suffix(".run") / "run_status.json"
    hashes = {key: sha256_file(path) for key, path in inputs.items()}
    checkpoint_hash = sha256_file(checkpoint)
    fit, provenance = read_json_object(fit_metadata), read_json_object(manifest_provenance)
    fit_spec, eval_spec = _completed_spec(fit_root), _completed_spec(evaluation)
    observed = read_json_object(fit_root / "observed_samples.json")
    for path in (fit_root / "manifest_provenance.json", evaluation / "manifest_provenance.json"):
        other = read_json_object(path)
        if other["manifest"]["fingerprint"] != provenance["manifest"]["fingerprint"] or other["splits"] != provenance["splits"]:
            raise ValueError("Fit/evaluation provenance differs from supplied provenance")
    expected = _validate_fit(fit, provenance, fit_spec, observed, checkpoint_hash, category)
    if eval_spec.get("weights", {}).get("sha256") != checkpoint_hash or eval_spec.get("split_fingerprint") != provenance["manifest"]["fingerprint"]:
        raise ValueError("Evaluation used different checkpoint weights or split identity")
    metrics, timings = read_json_object(evaluation / "metrics.json"), read_json_object(evaluation / "timings.json")
    frozen, thresholds = read_json_object(evaluation / "frozen_policy.json"), read_json_object(evaluation / "thresholds.json")
    if not frozen.get("runtime_fingerprint") or thresholds.get("runtime_fingerprint") != frozen["runtime_fingerprint"] or eval_spec.get("runtime_identity", {}).get("fingerprint") != frozen["runtime_fingerprint"]:
        raise ValueError("Evaluation runtime identity differs from the frozen policy")
    policy = DecisionThresholds(**frozen["image"])
    if policy.as_dict() != {key: thresholds.get(key) for key in policy.as_dict()} or frozen["pixel"] != thresholds.get("pixel"):
        raise ValueError("Threshold report differs from frozen evaluation policy")
    if metrics["dataset"]["categories"] != [category] or metrics["model"]["name"] != "patchcore":
        raise ValueError("Evaluation category or model family differs from bundle")
    if metrics["comparison_identity"]["runtime_fingerprint"] != frozen["runtime_fingerprint"]:
        raise ValueError("Metrics runtime differs from frozen policy")
    _prediction_rows(evaluation / "validation_predictions.csv", expected["validation"], category, "validation", policy)
    rows = _prediction_rows(evaluation / "predictions.csv", expected["test"], category, "test", policy)
    labels, scores = [int(r["label"] == "anomaly") for r in rows], [float(r["anomaly_score"]) for r in rows]
    costs = ReviewCosts(**eval_spec["review_costs"]) if eval_spec.get("review_costs") else None
    if metrics["image_level"] != image_level_metrics(labels, scores, policy.fail_threshold) or metrics["review_system"] != review_system_metrics(labels, scores, policy, costs=costs):
        raise ValueError("Image/review metrics differ from saved predictions")
    if robustness:
        stress = read_json_object(robustness)
        stress_policy = stress.get("frozen_policy", {})
        if read_json_object(inputs["robustness_run_status.json"]).get("status") != "complete" or any(stress_policy.get(key) != frozen[key] for key in ("image", "pixel", "runtime_fingerprint")):
            raise ValueError("Robustness report is incomplete or belongs to another runtime/policy")
    spec = {"model_id": model_id, "category": category, "device": device,
            "checkpoint_sha256": checkpoint_hash, "input_sha256": hashes}
    reservation = ArtifactReservation.reserve_new(output, spec_fingerprint=digest(spec))
    try:
        write_json_artifact(output / "resolved_spec.json", spec)
        weights = output / "weights" / "model.ckpt"
        weights.parent.mkdir()
        with checkpoint.open("rb") as source, weights.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        if sha256_file(weights) != checkpoint_hash:
            raise ValueError("Checkpoint changed during copy")
        runtime = load_model("patchcore", weights, device=device)
        if runtime_identity(runtime) != eval_spec["runtime_identity"]:
            raise ValueError("Cold runtime differs from evaluated runtime; re-evaluate under a new ID")
        for name, source in inputs.items():
            target = output / "provenance" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            if sha256_file(target) != hashes[name] or sha256_file(source) != hashes[name]:
                raise ValueError("Input evidence changed during packaging")
        card = output / "MODEL_CARD.md"
        with card.open("x", encoding="utf-8") as handle:
            handle.write(model_card(model_id=model_id, category=category, fit=fit, metrics=metrics, timings=timings,
                                    frozen=frozen, provenance=provenance, comparisons=comparison, robustness=robustness))
        bundle = ModelBundle.write(
            output, model_id=model_id, family="anomalib_patchcore", category=category,
            weights=artifact_reference(weights, relative_to=output), preprocessing={"pipeline": "checkpoint_preprocessor"},
            training={"experiment": fit["experiment"], "manifest": provenance["manifest"],
                      "split_counts": fit["counts"], "training_policy": fit["training_policy"], "evidence_sha256": hashes},
            parameters=fit["model"], software={key: str(value) for key, value in fit["runtime"].items() if value is not None},
            calibration={"score_space": runtime.score_space, "runtime_fingerprint": frozen["runtime_fingerprint"],
                         "source": policy.source, "objective": policy.objective, "image": policy.as_dict(), "pixel": frozen["pixel"],
                         "validation": {"split": "validation", "count": len(expected["validation"]),
                                        "split_fingerprint": provenance["manifest"]["fingerprint"]}},
            runtime=runtime, status="validated")
        loaded = _load_bundle_runtime("patchcore", bundle, device=device)
        if loaded.runtime_identity != eval_spec["runtime_identity"]:
            raise ValueError("Bundle runtime differs from evaluation")
        verification = {"model_id": model_id, "model_fingerprint": bundle.model_fingerprint,
                        "calibration_id": bundle.calibration_id, "weights_sha256": checkpoint_hash,
                        "runtime": {"status": "cold_bundle_loaded_and_identity_matched", "device": str(runtime.device)},
                        "scope": "bundle integrity and evaluated policy; no production quality acceptance"}
        write_json_artifact(output / "bundle_verification.json", verification)
        if sha256_file(checkpoint) != checkpoint_hash or any(sha256_file(path) != hashes[name] for name, path in inputs.items()):
            raise ValueError("Source evidence changed before publication")
        links = {} if model_card_path == card else {card: model_card_path}
        with publish_files(links):
            reservation.complete(artifacts=[p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file() and p.name != "run_status.json"])
        return {"bundle": str(output), "model_card": str(model_card_path), **verification}
    except BaseException as exc:
        reservation.fail(f"{type(exc).__name__}: {exc}")
        raise
