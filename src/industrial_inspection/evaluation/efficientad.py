"""Evaluate a completed EfficientAD fit and publish a portable calibration bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.data.contracts import load_manifest
from industrial_inspection.data.manifest_datamodule import load_manifest_splits
from industrial_inspection.evaluation.artifacts import ArtifactReservation, read_json_object
from industrial_inspection.evaluation.benchmark import benchmark_model
from industrial_inspection.evaluation.identity import comparison_identity
from industrial_inspection.evaluation.report import write_json_artifact
from industrial_inspection.evaluation.saved import evaluate_saved_model
from industrial_inspection.inference.pipeline import InspectionPipeline
from industrial_inspection.models.artifacts import ModelBundle
from industrial_inspection.models.efficientad_adapter import EfficientAdConfig
from industrial_inspection.models.efficientad_runtime import EfficientAdRuntime
from industrial_inspection.models.registry import _load_bundle_runtime
from industrial_inspection.scoring.policy import DecisionThresholds


def _read(path: Path) -> dict[str, Any]:
    return read_json_object(path)


def _write(path: Path, payload: dict[str, Any]) -> None:
    write_json_artifact(path, payload)


def _validate_observations(fit, observed, splits, spec):
    if fit.get("observed_samples") != observed or observed.get("split_fingerprint") != splits.fingerprint:
        raise ValueError("Fit observations differ from the recorded split")
    policy = fit.get("training_policy", {})
    if (policy.get("fit_split") != "train" or policy.get("fit_labels") != ["normal"]
            or policy.get("map_normalization_split") != "validation"
            or policy.get("test_seen_during_fit") is not False
            or observed.get("test_seen_during_fit") is not False):
        raise ValueError("Fit evidence must confirm normal-only training without test consumption")
    assets = fit.get("assets", {})
    if (not spec.get("teacher_sha256") or fit.get("loaded_teacher_sha256") != spec["teacher_sha256"]
            or assets.get("teacher_sha256") != spec["teacher_sha256"]
            or not spec.get("auxiliary_fingerprint")
            or assets.get("auxiliary_dataset", {}).get("fingerprint") != spec["auxiliary_fingerprint"]):
        raise ValueError("Teacher or auxiliary dataset identity differs from fit spec")
    expected = {name: {s.sample_id: sha256_file(s.image_path) for s in getattr(splits, name)}
                for name in ("train", "validation")}
    events = observed.get("events", {})
    allowed = {"read:train", "read:validation", "training_batch:train"}
    for name, record in events.items():
        if name not in allowed:
            raise ValueError("Fit observations contain forbidden sample consumption")
        seen = record.get("sample_sha256", {})
        if any(expected[name.split(":")[1]].get(key) != value for key, value in seen.items()):
            raise ValueError("Observed image hash differs from fitted split")
        count = record.get("event_count")
        if (type(count) is not int or count < len(seen)
                or record.get("unique_count") != len(seen)):
            raise ValueError("Invalid observed sample counts")
    # A step budget may stop before a full epoch; require actual optimizer input,
    # not a claim that every training image was consumed. Normalization reads all validation.
    if not events.get("training_batch:train", {}).get("sample_sha256"):
        raise ValueError("No optimizer training samples were observed")
    if events.get("read:validation", {}).get("sample_sha256") != expected["validation"]:
        raise ValueError("Fit observations do not cover validation normalization")
    if events["training_batch:train"]["event_count"] != fit["training"]["actual_steps"]:
        raise ValueError("Observed optimizer inputs differ from completed batch-size-one steps")


def finish_efficientad(
    *,
    fit_dir: Path,
    manifest: Path,
    project_root: Path,
    output: Path,
    category: str = "bottle",
    device: str = "cpu",
    runs: int = 100,
    warmup: int = 5,
    bootstrap_samples: int = 500,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Evaluation directory already exists: {output}")
    if any(type(value) is not int for value in (runs, warmup, bootstrap_samples)) or runs < 1 or warmup < 0 or bootstrap_samples < 1:
        raise ValueError("runs/bootstrap_samples must be positive; warmup must be non-negative")
    fit_dir, output = fit_dir.resolve(), output.resolve()
    evidence_names = ("fit_metadata.json", "run_status.json", "manifest_provenance.json",
                      "resolved_spec.json", "observed_samples.json")
    evidence_hashes = {name: sha256_file(fit_dir / name) for name in evidence_names}
    fit = _read(fit_dir / "fit_metadata.json")
    status = _read(fit_dir / "run_status.json")
    provenance = _read(fit_dir / "manifest_provenance.json")
    resolved_spec = _read(fit_dir / "resolved_spec.json")
    if status.get("status") != "complete" or fit.get("format_version") != 2:
        raise ValueError("A completed fit from the reviewed EfficientAD runner is required")
    fingerprint = hashlib.sha256(json.dumps(resolved_spec, sort_keys=True).encode()).hexdigest()
    if fingerprint != fit.get("spec_fingerprint") or fingerprint != status.get("spec_fingerprint"):
        raise ValueError("Fit spec fingerprint mismatch")
    if resolved_spec.get("config") != fit.get("resolved_config"):
        raise ValueError("Fit configuration differs from resolved spec")
    checkpoint = fit_dir / "model.ckpt"
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != fit["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint checksum differs from fit metadata")
    if fit.get("category") != category or resolved_spec.get("category") != category:
        raise ValueError("Fit category mismatch")
    samples = load_manifest(manifest, project_root)
    if {sample.category for sample in samples} != {category}:
        raise ValueError("Evaluation manifest must contain only the fitted category")
    splits = load_manifest_splits(manifest, project_root=project_root, category=category)
    if splits.manifest_sha256 != fit.get("manifest_sha256"):
        raise ValueError("Manifest checksum differs from fit")
    if splits.fingerprint != fit.get(
        "split_fingerprint"
    ) or splits.fingerprint != resolved_spec.get("split_fingerprint"):
        raise ValueError("Dataset split fingerprint differs from fit")
    if provenance.get("manifest", {}).get("fingerprint") != splits.fingerprint:
        raise ValueError("Manifest provenance differs from fitted split")
    if provenance.get("splits") != splits.provenance()["splits"]:
        raise ValueError("Manifest provenance sample IDs/hashes differ from fitted split")
    counts = {name: len(getattr(splits, name)) for name in ("train", "validation", "test")}
    if fit.get("counts") != counts:
        raise ValueError("Fit sample counts differ from manifest")
    config = EfficientAdConfig(**fit["resolved_config"])
    if fit["training"].get("actual_steps") != config.training_steps(len(splits.train)):
        raise ValueError("Fit did not complete its configured training budget")
    _validate_observations(fit, _read(fit_dir / "observed_samples.json"), splits, resolved_spec)
    spec = {
        "fit_metadata_sha256": sha256_file(fit_dir / "fit_metadata.json"),
        "checkpoint_sha256": checkpoint_sha256,
        "split_fingerprint": splits.fingerprint,
        "manifest_sha256": splits.manifest_sha256,
        "device": device,
        "runs": runs,
        "warmup": warmup,
        "bootstrap_samples": bootstrap_samples,
        "threshold_mode": "validation",
        "seed": fit["resolved_config"]["seed"],
        "fit_evidence_sha256": evidence_hashes,
    }
    reservation = ArtifactReservation.reserve_new(
        output,
        spec_fingerprint=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
    )
    try:
        _write(output / "evaluation_spec.json", spec)
        bundle_root = output / "bundle"
        weights = bundle_root / "weights/model.ckpt"
        weights.parent.mkdir(parents=True)
        shutil.copyfile(checkpoint, weights)
        if sha256_file(weights) != checkpoint_sha256:
            raise ValueError("Checkpoint changed during packaging")
        model = EfficientAdRuntime(weights, device=device)
        if model.model_id != fit["resolved_config"]["model_id"] or model.score_space != fit.get(
            "score_space"
        ):
            raise ValueError("Checkpoint identity differs from fit metadata")
        if model.model_spec != config.model_spec():
            raise ValueError("Checkpoint preprocessing/model spec differs from fit")
        evaluation = evaluate_saved_model(
            model=model,
            manifest_path=manifest,
            output=output / "evaluation",
            project_root=project_root,
            model_name="efficientad",
            model_path=weights,
            experiment_id=fit["experiment"],
            seed=spec["seed"],
            bootstrap_samples=bootstrap_samples,
            threshold_mode="validation",
        )
        thresholds = _read(output / "evaluation/thresholds.json")
        model.decision_thresholds = DecisionThresholds(**{
            key: thresholds[key] for key in ("pass_threshold", "fail_threshold", "source", "objective")
        })
        measured_identity = comparison_identity(model, split_fingerprint=splits.fingerprint,
                                               category=category, thresholds=model.decision_thresholds)
        if measured_identity != evaluation["metrics"]["comparison_identity"]:
            raise ValueError("Benchmark policy/runtime differs from evaluation")
        uses_cuda = str(model.device).startswith("cuda")
        benchmark = benchmark_model(
            model, [sample.image_path for sample in splits.validation], runs=runs, warmup=warmup
        )
        if comparison_identity(model, split_fingerprint=splits.fingerprint, category=category,
                               thresholds=model.decision_thresholds) != measured_identity:
            raise ValueError("Benchmark identity changed during measurement")
        benchmark["comparison_identity"] = measured_identity
        if not uses_cuda:
            for key in ("cuda_before", "cuda_after"):
                benchmark[key] = {name: None for name in benchmark[key]}
        benchmark["model"] = {
            "name": "efficientad",
            "model_id": model.model_id,
            "sha256": checkpoint_sha256,
            "size_bytes": weights.stat().st_size,
        }
        benchmark["protocol"].update({
            "split": "validation",
            "device": str(model.device),
            "warmup": warmup,
            "runs": runs,
            "rss_note": "RSS snapshots before/after; sampled peak is in resources.ram_peak_sampled_bytes",
        })
        _write(output / "benchmark.json", benchmark)
        thresholds = _read(output / "evaluation/thresholds.json")
        bundle = ModelBundle.write(
            bundle_root,
            model_id=model.model_id,
            family="anomalib_efficientad",
            category=category,
            weights={"path": "weights/model.ckpt", "sha256": checkpoint_sha256},
            preprocessing={
                "pipeline": "efficientad_resize_v1",
                "color": "rgb",
                "image_size": model.model_spec["image_size"],
                "antialias": True,
            },
            parameters=model.model_spec,
            software=fit["runtime"],
            training={
                "experiment": fit["experiment"],
                "split_fingerprint": splits.fingerprint,
                "manifest_sha256": splits.manifest_sha256,
                "counts": fit["counts"],
                "budget": fit["training"],
            },
            calibration={
                "score_space": model.score_space,
                "runtime_fingerprint": thresholds["runtime_fingerprint"],
                "threshold_source": thresholds["source"],
                "objective": thresholds["objective"],
                "image": {key: thresholds[key] for key in ("pass_threshold", "fail_threshold")},
                "pixel": thresholds["pixel"],
                "validation": {
                    "split": "validation",
                    "count": len(splits.validation),
                    "sample_ids": list(splits.ids("validation")),
                    "manifest_sha256": splits.manifest_sha256,
                    "split_fingerprint": splits.fingerprint,
                },
            },
            status="validated",
            publication_receipt={"path": "../run_status.json", "run_id": reservation.run_id,
                                 "spec_fingerprint": reservation.spec_fingerprint},
        )
        probe = splits.validation[0]
        expected = model.predict(probe.image_path)
        del model
        cold = _load_bundle_runtime("efficientad", bundle, device=device)
        pipeline = InspectionPipeline(cold, cold.decision_thresholds, output / "smoke_evidence",
                                      category=category)
        actual = pipeline.model.predict(probe.image_path)
        if not np.isclose(actual.image_score, expected.image_score, rtol=1e-6, atol=1e-7):
            raise ValueError("Bundle score changed after reload")
        if not np.allclose(actual.anomaly_map, expected.anomaly_map, rtol=1e-6, atol=1e-7):
            raise ValueError("Bundle anomaly map changed after reload")
        pipeline.inspect(probe.image_path)
        verification = {
            "status": "loaded_and_predicted",
            "model_id": bundle.model_id,
            "model_fingerprint": bundle.model_fingerprint,
            "calibration_id": bundle.calibration_id,
            "checkpoint_sha256": checkpoint_sha256,
            "probe_sample_id": probe.sample_id,
            "device": device,
        }
        _write(output / "bundle_verification.json", verification)
        for name, expected_hash in evidence_hashes.items():
            target = output / ("fit_run_status.json" if name == "run_status.json" else name)
            shutil.copyfile(fit_dir / name, target)
            if sha256_file(target) != expected_hash or sha256_file(fit_dir / name) != expected_hash:
                raise ValueError("Fit evidence changed during evaluation")
        if (sha256_file(checkpoint) != checkpoint_sha256
                or load_manifest_splits(manifest, project_root=project_root, category=category).fingerprint != splits.fingerprint):
            raise ValueError("Fit checkpoint or dataset changed during evaluation")
        card = (
            f"# EfficientAD {category}: {fit['experiment']}\n\n"
            f"Checkpoint SHA-256: `{checkpoint_sha256}`. Split: `{splits.fingerprint}`.\n\n"
            f"Calibration uses {len(splits.validation)} validation images; test has {len(splits.test)} images. "
            f"Policy: `{thresholds['source']}`; score space: `{bundle.score_space}`.\n\n"
            f"Image metrics: `{json.dumps(evaluation['metrics']['image_level'])}`.\n\n"
            f"Pixel metrics: `{json.dumps(evaluation['metrics'].get('pixel_level'))}`.\n\n"
            f"Inference p50/p95: {benchmark['latency']['p50_ms']:.3f}/{benchmark['latency']['p95_ms']:.3f} ms "
            f"on {device}; {runs} runs after {warmup} warmups. RSS values are snapshots, not a peak.\n\n"
            "Scope: this checkpoint and this manifest only. No claim for other categories or production data. "
            "Inspect evaluation/failure_cases before deployment. Training resources are recorded only if present "
            "in fit_metadata.json; absent measurements are not inferred.\n"
        )
        (output / "MODEL_CARD.md").write_text(card, encoding="utf-8")
        reservation.complete(
            artifacts=[
                "bundle",
                "evaluation",
                "benchmark.json",
                "MODEL_CARD.md",
                "bundle_verification.json",
                "evaluation_spec.json",
            ]
        )
        return {
            "output": str(output),
            "bundle": str(bundle_root),
            "experiment": fit["experiment"],
            "test_sample_count": len(splits.test),
            "verification": verification,
        }
    except BaseException as exc:
        reservation.fail(f"{type(exc).__name__}: {exc}")
        raise
