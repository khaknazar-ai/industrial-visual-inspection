"""Read-only release acceptance from existing experiment/bundle artifacts.

A release index only selects existing evidence: ``milestone``, ``experiments``
({evaluation, bundle, protocol}), and ``branches`` ({id, status, evidence}).
Protocol/evidence references are {path, sha256}; evaluation points to an existing
resolved_spec.json, bundle to an existing schema-2 directory. Asset formats are
unchanged. Missing independent protocol evidence is a failure, never inferred
from a successful fit, bundle load, test count or filename.

This checks declared evidence consistency, not the truth of physical capture or
an external review. It never imports a model, decodes an image or runs a process.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

FAMILIES = ("classical_diff", "patchcore", "efficientad", "dino_memory")
CATEGORIES = ("bottle", "screw", "capsule", "hazelnut", "carpet")
_COMMON = tuple(f"P{i:02}" for i in range(7))
REQUIRED_BRANCHES = {
    "LOCAL_READY": (*_COMMON, "P16"),
    "PORTFOLIO_COMPLETE": (*_COMMON, "P07", "P09", "P10", "P11", "P12", "P13", "P14", "P16", "P17"),
    "BLUEPRINT_COMPLETE": (*_COMMON, "P07", "P08", "P09", "P10", "P11", "P12", "P13", "P14", "P15a", "P15b", "P16", "P17"),
}
_BUNDLE_FAMILIES = {
    "classical_diff": {"classical_cv"},
    "patchcore": {"anomalib_patchcore"},
    "efficientad": {"anomalib_efficientad"},
    "dino_memory": {"dino_memory", "dinov3_memory", "dinov3_feature_memory"},
    "deep_patch_memory": {"frozen_cnn_memory_bank", "deep_patch_memory"},
}
_REPORTS = ("metrics.json", "timings.json", "thresholds.json", "frozen_policy.json",
            "manifest_provenance.json", "run_status.json", "environment.json")
_HEX = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


def _require(condition: Any, code: str, detail: str) -> None:
    if not condition:
        raise EvidenceError(code, detail)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), "missing_artifact", str(path))
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "invalid_json", f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path: Path) -> dict:
    _require(path.is_file() and not path.is_symlink(), "missing_artifact", str(path))
    _require(path.stat().st_size <= 16 * 1024**2, "invalid_json", f"JSON exceeds 16 MiB: {path}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    json.dumps(value, allow_nan=False)
    _require(isinstance(value, dict), "invalid_json", f"Expected object: {path}")
    return value


def _path(base: Path, value: Any) -> Path:
    _require(isinstance(value, str) and value and not any(c in value for c in "\0\r\n"),
             "invalid_path", "Evidence path must be nonempty local text")
    _require(not value.startswith(("\\\\", "//")) and "://" not in value,
             "invalid_path", "Remote paths/URLs are not offline evidence")
    path = Path(value)
    path = path if path.is_absolute() else base / path
    _require(not any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction())
                     for p in (path, *path.parents)), "invalid_path", f"Linked evidence: {path}")
    return path.resolve()


def _reference(base: Path, value: Any) -> Path:
    _require(isinstance(value, dict), "missing_reference", "Expected {path, sha256} evidence reference")
    path = _path(base, value.get("path"))
    expected = value.get("sha256")
    _require(isinstance(expected, str) and _HEX.fullmatch(expected), "missing_hash", str(path))
    _require(_sha(path) == expected, "hash_mismatch", str(path))
    return path


def _weights(base: Path, reference: dict) -> str:
    path = _path(base, reference.get("path"))
    if reference.get("kind", "file") == "file":
        actual = _sha(path)
    else:
        _require(reference.get("kind") == "directory" and path.is_dir(), "missing_artifact", str(path))
        files = reference.get("files")
        _require(isinstance(files, dict) and files, "missing_hash", f"Directory file hashes: {path}")
        actual_files = {}
        for item in path.rglob("*"):
            _require(not item.is_symlink() and not (hasattr(item, "is_junction") and item.is_junction()),
                     "invalid_path", f"Linked weights: {item}")
            if item.is_file():
                actual_files[item.relative_to(path).as_posix()] = _sha(item)
        _require(actual_files == files, "hash_mismatch", f"Weights directory members/bytes: {path}")
        actual = _digest(actual_files)
    _require(actual == reference.get("sha256"), "hash_mismatch", f"Weights digest: {path}")
    return actual


def _fingerprint(value: dict, field="fingerprint", *, prefix="", code="fingerprint_mismatch") -> str:
    actual = prefix + _digest({k: v for k, v in value.items() if k != field})
    _require(value.get(field) == actual, code, field)
    return actual


def _bundle(root: Path, family: str, category: str) -> tuple[dict, dict]:
    model, calibration = _read(root / "model.json"), _read(root / "calibration.json")
    _require(model.get("schema_version") == calibration.get("schema_version") == 2,
             "bundle_schema", "Release needs a schema-2 runtime-bound bundle")
    _require(model.get("status") == "validated", "bundle_unvalidated", str(root))
    _require(model.get("family") in _BUNDLE_FAMILIES[family] and model.get("category") == category,
             "identity_mismatch", "Bundle family/category differs from selected experiment")
    fingerprint = _fingerprint(model)
    _require(calibration.get("model_fingerprint") == fingerprint
             and calibration.get("model_id") == model.get("model_id"),
             "identity_mismatch", "Calibration model identity")
    weights = model.get("weights", {})
    actual_weights = _weights(root, weights)
    _require(calibration.get("weights_sha256") == actual_weights, "identity_mismatch", "Calibration weights")
    preprocessing = model.get("preprocessing", {})
    preprocess_hash = _fingerprint(preprocessing)
    _require(calibration.get("preprocessing_fingerprint") == preprocess_hash,
             "identity_mismatch", "Calibration preprocessing")
    runtime = model.get("runtime_identity", {})
    _require(all(runtime.get(k) for k in ("runtime_class", "decode", "transform", "encoder", "postprocess", "dependencies", "implementation")),
             "bundle_schema", "Runtime contract is incomplete")
    runtime_hash = _fingerprint(runtime)
    _require(runtime.get("version") == 1 and calibration.get("runtime_fingerprint") == runtime_hash,
             "identity_mismatch", "Calibration runtime")
    _require(runtime.get("decode") == preprocessing.get("decode")
             and runtime.get("transform") == preprocessing.get("transform"),
             "identity_mismatch", "Actual preprocessing differs from runtime")
    _require(model.get("score_space") == calibration.get("score_space") == runtime.get("score_space"),
             "identity_mismatch", "Score space")
    pixel = calibration.get("pixel")
    _require(isinstance(pixel, dict) and type(pixel.get("threshold")) in (int, float)
             and math.isfinite(pixel["threshold"]) and type(pixel.get("version")) is int
             and pixel["version"] in {1, 2}, "policy_mismatch", "Explicit calibrated finite pixel policy")
    kernel, minimum = pixel.get("morphology_kernel"), pixel.get("min_component_area")
    _require(type(kernel) is int and kernel >= 0 and (kernel == 0 or kernel % 2 == 1)
             and type(minimum) is int and minimum >= 1, "policy_mismatch", "Invalid pixel morphology/components")
    publication = model.get("publication_receipt")
    if publication is not None:
        _require(isinstance(publication, dict) and publication.get("path") == "../run_status.json",
                 "publication_mismatch", "Unsupported bundle publication receipt")
        parent = _read(root.parent / "run_status.json")
        _require(parent.get("status") == "complete", "run_incomplete", "Bundle parent publication is incomplete")
        _require(all(parent.get(k) == publication.get(k) and parent.get(k) for k in ("run_id", "spec_fingerprint"))
                 and root.name in parent.get("artifacts", []), "publication_mismatch", "Bundle parent receipt identity")
    runtime_class = runtime.get("runtime_class", "").lower()
    if family == "dino_memory":
        _require("dino" in runtime_class and "deep_patch_memory" not in runtime_class,
                 "family_substitution", "DINO requires its encoder runtime; DPM is not DINO")
    _fingerprint(calibration, "calibration_id", prefix="cal-")
    commit = _read(root / "bundle_commit.json")
    _require(commit.get("model") == _digest(model) and commit.get("calibration") == _digest(calibration),
             "fingerprint_mismatch", "Bundle publication record")
    marker = root / "run_status.json"
    if marker.exists():
        _require(_read(marker).get("status") == "complete", "run_incomplete", str(marker))
    return model, calibration


def _protocol(base: Path, reference: Any, manifest: dict, category: str) -> dict:
    protocol = _read(_reference(base, reference))
    classification = protocol.get("class")
    _require(classification in {"independent", "synthetic", "legacy-exploratory"},
             "protocol_unclassified", "Protocol class must be explicit")
    _require(classification == "independent", "protocol_not_independent", str(classification))
    _require(protocol.get("test_previously_used") is False and protocol.get("frozen_before_test") is True
             and protocol.get("selection_split") == "validation",
             "protocol_not_independent", "Test reuse/freeze/selection discipline not established")
    _require(protocol.get("split_fingerprint") == manifest.get("fingerprint")
             and protocol.get("manifest_sha256") == manifest.get("sha256"),
             "identity_mismatch", "Independent protocol dataset identity")
    if category == "bottle":
        _require(protocol.get("grouping") == ["physical_item_id", "session_id"],
                 "independence_groups_missing", "Independent bottle needs physical item and session groups")
    return protocol


def _dataset(base: Path, provenance: dict, category: str, protocol: dict) -> dict[str, dict]:
    manifest = provenance.get("manifest", {})
    path = _reference(base, manifest)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(rows and len(rows) <= 1_000_000, "invalid_manifest", str(path))
    by_id = {row.get("sample_id"): row for row in rows}
    _require(len(by_id) == len(rows) and all(by_id), "invalid_manifest", "Missing/duplicate sample IDs")
    splits = provenance.get("splits", {})
    seen_ids, seen_hashes = set(), set()
    expected_ids = set()
    for split in ("train", "validation", "test"):
        data = splits.get(split, {})
        ids, hashes = data.get("sample_ids", []), data.get("image_sha256", [])
        _require(isinstance(ids, list) and isinstance(hashes, list) and ids and type(data.get("count")) is int
                 and len(ids) == len(set(ids)) == len(hashes) == data["count"],
                 "invalid_manifest", f"{split}: provenance counts")
        _require(not seen_ids.intersection(ids) and not seen_hashes.intersection(hashes)
                 and len(set(hashes)) == len(hashes), "split_leakage", split)
        _require(all(isinstance(h, str) and _HEX.fullmatch(h) for h in hashes),
                 "missing_hash", f"{split}: input hashes")
        for sample_id in ids:
            row = by_id.get(sample_id, {})
            _require(row.get("split") == split and row.get("category") == category,
                     "identity_mismatch", f"Manifest split/category: {sample_id}")
            _require(row.get("label") in {"normal", "anomaly"}
                     and (split != "train" or row.get("label") == "normal"),
                     "invalid_manifest", f"Unknown label or nonnormal train: {sample_id}")
        seen_ids.update(ids)
        seen_hashes.update(hashes)
        expected_ids.update(ids)
    _require(expected_ids == set(by_id), "identity_mismatch", "CSV/provenance sample coverage")
    grouping = protocol.get("grouping", [])
    _require(isinstance(grouping, list), "invalid_manifest", "Protocol grouping must be a list")
    for field in grouping:
        used = set()
        for split in ("train", "validation", "test"):
            groups = {by_id[i].get(field) for i in splits[split]["sample_ids"]}
            _require(all(groups), "independence_groups_missing", field)
            _require(not used.intersection(groups), "group_leakage", field)
            used.update(groups)
    return by_id


def _predictions(path: Path, split: str, samples: dict, provenance: dict, category: str, image: dict):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row.get("sample_id") for row in rows]
    _require(len(ids) == len(set(ids)) and set(ids) == set(provenance["splits"][split]["sample_ids"]),
             "prediction_coverage", f"{split}: prediction IDs")
    for row in rows:
        score = float(row.get("anomaly_score", "nan"))
        decision = "pass" if score < image["pass_threshold"] else "review" if score < image["fail_threshold"] else "fail"
        _require(math.isfinite(score) and row.get("decision") == decision,
                 "policy_mismatch", f"{split}: score/decision for {row.get('sample_id')}")
        _require(row.get("split") == split and row.get("category") == category
                 and row.get("label") == samples[row["sample_id"]]["label"],
                 "identity_mismatch", f"{split}: prediction label/category")
    if split == "test":
        _require({r["label"] for r in rows} == {"normal", "anomaly"},
                 "evaluation_coverage", "Independent test must measure normal and anomaly cases")


def _experiment(base: Path, entry: dict) -> tuple[str, str]:
    spec_path = _path(base, entry.get("evaluation"))
    spec = _read(spec_path)
    root = spec_path.parent
    reports = {name: _read(root / name) for name in _REPORTS}
    for filename in ("config.yaml", "predictions.csv", "validation_predictions.csv"):
        _require((root / filename).is_file(), "missing_artifact", str(root / filename))
    status = reports["run_status.json"]
    _require(status.get("status") == "complete", "run_incomplete", str(root))
    _require(isinstance(status.get("run_id"), str) and status["run_id"], "run_incomplete", "Missing run identity")
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()
    _require(status.get("spec_fingerprint") == spec_hash, "fingerprint_mismatch", "Executed evaluation spec")
    metrics, provenance = reports["metrics.json"], reports["manifest_provenance.json"]
    family = spec.get("model_name")
    _require(family in _BUNDLE_FAMILIES, "unsupported_family", str(family))
    categories = metrics.get("dataset", {}).get("categories", [])
    _require(isinstance(categories, list) and len(categories) == 1, "invalid_manifest", "One category per experiment")
    category = categories[0]
    _require(metrics.get("dataset", {}).get("name") == "mvtec_ad", "dataset_mismatch", "Matrix requires MVTec AD")
    _require(metrics.get("experiment_id") == spec.get("experiment_id") and metrics.get("model", {}).get("name") == family,
             "identity_mismatch", "Metrics experiment/model")
    for data in (spec, metrics, metrics.get("dataset", {})):
        _require(data.get("not_independent") is not True and data.get("test_previously_used") is not True,
                 "protocol_not_independent", "Evaluation declares reused test")
        if "protocol_class" in data:
            _require(data["protocol_class"] == "independent", "protocol_not_independent", str(data["protocol_class"]))
    manifest = provenance.get("manifest", {})
    protocol = _protocol(base, entry.get("protocol"), manifest, category)
    samples = _dataset(root, provenance, category, protocol)
    model, calibration = _bundle(_path(base, entry.get("bundle")), family, category)
    runtime = model["runtime_identity"]
    identity = metrics.get("comparison_identity", {})
    runtime_hash = runtime["fingerprint"]
    _require(spec.get("runtime_identity") == runtime and identity.get("runtime_fingerprint") == runtime_hash,
             "identity_mismatch", "Evaluation/bundle runtime")
    split = manifest.get("fingerprint")
    _require(isinstance(split, str) and _HEX.fullmatch(split)
             and spec.get("split_fingerprint") == identity.get("split_fingerprint")
             == calibration.get("validation", {}).get("split_fingerprint") == split,
             "identity_mismatch", "Evaluation/runtime/calibration split")
    _require(spec.get("weights", {}).get("sha256") == model["weights"]["sha256"],
             "identity_mismatch", "Evaluation used different weights")
    _require(identity.get("model_id") == metrics["model"].get("model_id") == model.get("model_id")
             and identity.get("category") == category, "identity_mismatch", "Comparison model/category")
    image = calibration.get("image", {})
    _require(all(type(image.get(k)) in (int, float) and math.isfinite(image[k]) for k in ("pass_threshold", "fail_threshold"))
             and image["pass_threshold"] < image["fail_threshold"], "policy_mismatch", "Image thresholds")
    policy_image = {**image, "source": image.get("source", calibration.get("threshold_source", calibration.get("source"))),
                    "objective": image.get("objective", calibration.get("objective"))}
    frozen, thresholds = reports["frozen_policy.json"], reports["thresholds.json"]
    _require(frozen.get("runtime_fingerprint") == thresholds.get("runtime_fingerprint") == runtime_hash
             and frozen.get("image") == policy_image and frozen.get("pixel") == calibration.get("pixel")
             and all(thresholds.get(k) == v for k, v in policy_image.items())
             and thresholds.get("pixel") == calibration.get("pixel"), "policy_mismatch", "Frozen/calibrated policy")
    _require(identity.get("policy_fingerprint") == _digest({"image": policy_image, "pixel": calibration.get("pixel")}),
             "identity_mismatch", "Comparison policy fingerprint")
    state = {k: runtime[k] for k in ("encoder", "fitted_state") if k in runtime}
    _require(state and identity.get("weights_state_sha256") == _digest(state),
             "identity_mismatch", "Comparison decoded state identity")
    validation = calibration.get("validation", {})
    _require(validation.get("split") == "validation" and type(validation.get("count")) is int
             and validation["count"] == provenance["splits"]["validation"]["count"]
             and validation.get("policy_evaluated") is not False, "identity_mismatch", "Calibration validation coverage")
    if "sample_ids" in validation:
        _require(isinstance(validation["sample_ids"], list)
                 and len(validation["sample_ids"]) == len(set(validation["sample_ids"])) == validation["count"]
                 and set(validation["sample_ids"]) == set(provenance["splits"]["validation"]["sample_ids"]),
                 "identity_mismatch", "Calibration validation IDs")
    if "manifest_sha256" in validation:
        _require(validation["manifest_sha256"] == manifest["sha256"],
                 "identity_mismatch", "Calibration manifest hash")
    for split_name, filename in (("validation", "validation_predictions.csv"), ("test", "predictions.csv")):
        _predictions(root / filename, split_name, samples, provenance, category, image)
    _require(all(isinstance(metrics.get(k), dict) and metrics[k] for k in
                 ("image_level", "pixel_level", "review_system", "confidence_intervals_95")),
             "missing_metrics", "Image/pixel/review/CI reports")
    timing = reports["timings.json"].get("test_inference", {})
    _require(all(type(timing.get(k)) in (int, float) and math.isfinite(timing[k]) and timing[k] >= 0
                 for k in ("p50_ms", "p95_ms")), "missing_metrics", "Measured test inference p50/p95")
    return family, category


def validate_release(manifest_path: Path, *, milestone: str) -> dict:
    """Return deterministic concrete failures; never infer the requested milestone."""
    if milestone not in REQUIRED_BRANCHES:
        raise ValueError(f"Unknown milestone: {milestone}")
    issues, accepted = [], []
    manifest_path = manifest_path.resolve()

    def fail(code, location, message):
        issues.append({"code": code, "location": location, "message": message})

    try:
        manifest = _read(manifest_path)
        _require(manifest.get("milestone") == milestone, "milestone_mismatch", "Index must explicitly select the requested milestone")
    except (EvidenceError, OSError, ValueError, TypeError) as exc:
        fail(getattr(exc, "code", "invalid_json"), str(manifest_path), str(exc))
        manifest = {}
    base = manifest_path.parent
    experiments = manifest.get("experiments", [])
    if not isinstance(experiments, list):
        fail("invalid_manifest", "experiments", "Expected list of existing evaluation references")
        experiments = []
    for i, entry in enumerate(experiments):
        try:
            _require(isinstance(entry, dict), "invalid_manifest", "Experiment entry must be an object")
            cell = _experiment(base, entry)
            _require(cell not in accepted, "duplicate_cell", f"Selected more than one run for {cell}")
            accepted.append(cell)
        except (EvidenceError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            fail(getattr(exc, "code", "invalid_manifest"), f"experiments[{i}]", str(exc))
    required = set() if milestone == "LOCAL_READY" else {(family, category) for family in FAMILIES for category in CATEGORIES}
    for family, category in sorted(required - set(accepted)):
        fail("missing_cell", f"{family}/{category}", "No complete independent compatible experiment")
    branches = manifest.get("branches", [])
    if not isinstance(branches, list):
        fail("invalid_manifest", "branches", "Expected explicit branch acceptance list")
        branches = []
    seen = set()
    for i, branch in enumerate(branches):
        location = f"branches[{i}]"
        try:
            _require(isinstance(branch, dict), "invalid_manifest", "Branch must be an object")
            name = branch.get("id")
            _require(name in REQUIRED_BRANCHES["BLUEPRINT_COMPLETE"] and name not in seen,
                     "invalid_branch", f"Unknown/duplicate branch: {name}")
            seen.add(name)
            _require(branch.get("status") == "complete", "branch_incomplete", str(name))
            evidence = branch.get("evidence")
            _require(isinstance(evidence, list) and evidence, "missing_reference", f"Branch evidence: {name}")
            for reference in evidence:
                path = _reference(base, reference)
                _require(path.suffix.lower() == ".json", "branch_scope_missing", "Branch acceptance needs a JSON receipt")
                receipt = _read(path)
                _require(receipt.get("status") in {"complete", "passed"}, "branch_incomplete", str(path))
                _require(receipt.get("stage", receipt.get("task_id")) == name,
                         "branch_scope_mismatch", f"Receipt does not accept {name}: {path}")
        except (EvidenceError, OSError, ValueError, TypeError, KeyError) as exc:
            fail(getattr(exc, "code", "invalid_manifest"), location, str(exc))
    for name in REQUIRED_BRANCHES[milestone]:
        if name not in seen:
            fail("missing_branch", name, "Mandatory branch acceptance is absent")
    return {"status": "passed" if not issues else "failed", "milestone": milestone,
            "scope": "offline_declared_evidence_consistency; not physical-provenance certification",
            "manifest": str(manifest_path), "required_cells": len(required),
            "accepted_cells": [{"family": f, "category": c} for f, c in sorted(accepted)],
            "required_branches": list(REQUIRED_BRANCHES[milestone]), "issues": issues}
