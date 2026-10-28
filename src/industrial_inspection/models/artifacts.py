"""Portable model and calibration bundle metadata.

The bundle is deliberately small: large weights stay at their canonical location and are
referenced by a relative path plus a digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from industrial_inspection.scoring.pixels import PixelPolicy

SCHEMA_VERSION = 2
SCORE_SPACES = {
    "raw_distance", "raw_score", "normalized_score", "probability",
    "anomalib_postprocessed_v1", "efficientad_quantiles_v1",
}


class ModelBundleError(ValueError):
    """Raised when a model or calibration bundle is not self-consistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value {value} in {path}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
        json.dumps(value, allow_nan=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ModelBundleError(f"Could not read JSON metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ModelBundleError(f"Metadata must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _publish_pair(root: Path, model: dict[str, Any], calibration: dict[str, Any]) -> None:
    """Publish metadata with an exclusive reservation and a commit record written last.

    Existing weights in root are not moved. Readers require the commit record for runtime-bound
    bundles, so they cannot observe a half-published pair. Failed writes remove only our own files.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".bundle-write.lock"
    published: list[Path] = []
    with lock.open("x", encoding="utf-8") as reservation:
        try:
            with tempfile.TemporaryDirectory(prefix=".bundle-", dir=root) as directory:
                staging = Path(directory)
                payloads = {"model.json": model, "calibration.json": calibration,
                            "bundle_commit.json": {"version": 1, "model": _json_digest(model),
                                                   "calibration": _json_digest(calibration)}}
                for name, value in payloads.items():
                    _write_json(staging / name, value)
                for name in payloads:
                    target = root / name
                    # Prefer exclusive hard-link publication. If Windows denies hard
                    # links, its same-volume rename still atomically refuses an existing
                    # destination and preserves the no-overwrite contract.
                    try:
                        os.link(staging / name, target)
                    except PermissionError:
                        if os.name != "nt":
                            raise
                        os.rename(staging / name, target)
                    published.append(target)
        except BaseException:
            for path in reversed(published):
                path.unlink()
            raise
        finally:
            reservation.close()
            lock.unlink()


def _required_text(data: dict[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelBundleError(f"{label} requires a non-empty {key}")
    return value


def _required_digest(data: dict[str, Any], key: str, label: str) -> str:
    value = _required_text(data, key, label)
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ModelBundleError(f"{label} requires a SHA-256 {key}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ModelBundleError(f"{label} must be a finite number")
    return float(value)


def artifact_reference(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    """Hash a checkpoint or all files in a directory-based model, without copying it."""
    path = path.resolve()
    reference_path = str(path.relative_to(relative_to.resolve())) if relative_to else str(path)
    if path.is_file():
        return {"path": reference_path, "kind": "file", "sha256": sha256_file(path)}
    if not path.is_dir():
        raise ModelBundleError(f"Referenced weights are missing: {path}")
    files = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise ModelBundleError(f"Model artifacts cannot contain links: {item}")
        if item.is_file():
            files[item.relative_to(path).as_posix()] = sha256_file(item)
    if not files:
        raise ModelBundleError(f"Model artifact directory is empty: {path}")
    return {"path": reference_path, "kind": "directory", "files": files, "sha256": _json_digest(files)}


def calibration_digest(calibration: dict[str, Any]) -> str:
    content = {key: value for key, value in calibration.items() if key != "calibration_id"}
    digest = _json_digest(content)
    return "cal-" + (digest[:16] if calibration.get("schema_version") == 1 else digest)


def _validate_thresholds(calibration: dict[str, Any]) -> None:
    image = calibration.get("image")
    if not isinstance(image, dict):
        raise ModelBundleError("Calibration requires an image threshold object")
    try:
        pass_threshold = _finite_number(image["pass_threshold"], "pass_threshold")
        fail_threshold = _finite_number(image["fail_threshold"], "fail_threshold")
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelBundleError("Calibration requires both image thresholds") from exc
    if not pass_threshold < fail_threshold:
        raise ModelBundleError("Calibration pass_threshold must be lower than fail_threshold")
    if _required_text(calibration, "score_space", "Calibration") not in SCORE_SPACES:
        raise ModelBundleError("Unsupported calibration score space")
    pixel = calibration.get("pixel")
    if pixel is not None:
        if not isinstance(pixel, dict) or "threshold" not in pixel:
            raise ModelBundleError("Pixel calibration requires a threshold or explicit null")
        if pixel["threshold"] is not None:
            _finite_number(pixel["threshold"], "pixel threshold")
        try:
            PixelPolicy.from_dict(pixel)
        except (ValueError, TypeError) as exc:
            raise ModelBundleError(f"Invalid pixel policy: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """A verified model metadata pair rooted at one directory."""

    root: Path
    model: dict[str, Any]
    calibration: dict[str, Any]

    @property
    def model_id(self) -> str:
        return _required_text(self.model, "model_id", "Model metadata")

    @property
    def category(self) -> str:
        return _required_text(self.model, "category", "Model metadata")

    @property
    def family(self) -> str:
        return _required_text(self.model, "family", "Model metadata")

    @property
    def score_space(self) -> str:
        return _required_text(self.calibration, "score_space", "Calibration")

    @property
    def model_fingerprint(self) -> str:
        return _required_text(self.model, "fingerprint", "Model metadata")

    @property
    def calibration_id(self) -> str:
        return _required_text(self.calibration, "calibration_id", "Calibration")

    @property
    def artifact_path(self) -> Path:
        path = Path(self.model["weights"]["path"])
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @property
    def calibration_status(self) -> str:
        return "legacy_exploratory" if self.model["schema_version"] == 1 else self.model["status"]

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        expected_category: str | None = None,
        expected_model_id: str | None = None,
        expected_weights_sha256: str | None = None,
        allow_legacy: bool = False,
    ) -> ModelBundle:
        root = root.resolve()
        marker = root / "run_status.json"
        if marker.exists() and _read_json(marker).get("status") != "complete":
            raise ModelBundleError("Bundle belongs to an incomplete or failed build")
        model = _read_json(root / "model.json")
        calibration = _read_json(root / "calibration.json")
        allowed_versions = {1, SCHEMA_VERSION} if allow_legacy else {SCHEMA_VERSION}
        if model.get("schema_version") not in allowed_versions:
            if model.get("schema_version") == 1:
                raise ModelBundleError(
                    "Schema 1 bundle requires explicit allow_legacy=True for exploratory reading; "
                    "create a schema 2 bundle with validation provenance for deployment"
                )
            raise ModelBundleError("Unsupported model bundle schema_version")
        if calibration.get("schema_version") != model.get("schema_version"):
            raise ModelBundleError("Unsupported calibration schema_version")
        bundle = cls(root, model, calibration)
        bundle.verify(
            expected_category=expected_category,
            expected_model_id=expected_model_id,
            expected_weights_sha256=expected_weights_sha256,
        )
        receipt = model.get("publication_receipt")
        if receipt is not None:
            completed = _read_json(root.parent / "run_status.json")
            if (completed.get("status") != "complete"
                    or any(completed.get(key) != receipt[key] for key in ("run_id", "spec_fingerprint"))
                    or root.name not in completed.get("artifacts", [])):
                raise ModelBundleError("Bundle publication receipt is incomplete, failed or mismatched")
        commit_path = root / "bundle_commit.json"
        if "runtime_identity" in model or model.get("publication_version") == 1 or commit_path.exists():
            commit = _read_json(commit_path)
            if commit != {"version": 1, "model": _json_digest(model), "calibration": _json_digest(calibration)}:
                raise ModelBundleError("Bundle publication record does not match its metadata")
        return bundle

    @classmethod
    def write(
        cls,
        root: Path,
        *,
        model_id: str,
        family: str,
        category: str,
        weights: dict[str, Any],
        preprocessing: dict[str, Any],
        calibration: dict[str, Any],
        training: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        software: dict[str, str] | None = None,
        status: str = "validated",
        runtime: Any | None = None,
        publication_receipt: dict[str, str] | None = None,
    ) -> ModelBundle:
        if not model_id or not family or not category:
            raise ModelBundleError("model_id, family, and category are required")
        if (root / "model.json").exists() or (root / "calibration.json").exists():
            raise FileExistsError(f"Refusing to replace existing bundle metadata: {root}")
        runtime_contract = None
        if runtime is not None or status == "validated":
            from industrial_inspection.models.identity_router import runtime_identity
            from industrial_inspection.models.registry import _BUNDLE_FAMILIES, _load_native_model

            if runtime is None:
                name = next((name for name, families in _BUNDLE_FAMILIES.items() if family in families), None)
                if name is None:
                    raise ModelBundleError(f"No runtime identity adapter for family {family!r}")
                reference = Path(weights["path"])
                artifact = reference if reference.is_absolute() else root / reference
                runtime = _load_native_model(name, artifact, device="cpu")
            runtime_contract = runtime_identity(runtime)
        transform = ({"decode": runtime_contract["decode"], "transform": runtime_contract["transform"]}
                     if runtime_contract else dict(preprocessing))
        transform_fingerprint = _json_digest({k: v for k, v in transform.items() if k != "fingerprint"})
        transform.setdefault("fingerprint", transform_fingerprint)
        model = {
            "schema_version": SCHEMA_VERSION,
            "publication_version": 1,
            "model_id": model_id,
            "family": family,
            "category": category,
            "weights": dict(weights),
            "preprocessing": transform,
            "score_space": calibration.get("score_space"),
            "training": dict(training or {}),
            "parameters": dict(parameters or {}),
            "software": dict(software or {}),
            "status": status,
        }
        if runtime_contract is not None:
            model["runtime_identity"] = runtime_contract
            model["declared_preprocessing"] = dict(preprocessing)
        if publication_receipt is not None:
            model["publication_receipt"] = dict(publication_receipt)
        model["fingerprint"] = _json_digest(model)
        calibration_data = dict(calibration)
        calibration_data.setdefault("schema_version", SCHEMA_VERSION)
        calibration_data.setdefault("model_id", model_id)
        calibration_data.setdefault("model_fingerprint", model["fingerprint"])
        calibration_data.setdefault("weights_sha256", weights.get("sha256"))
        calibration_data.setdefault("preprocessing_fingerprint", transform_fingerprint)
        if runtime_contract is not None:
            calibration_data.setdefault("runtime_fingerprint", runtime_contract["fingerprint"])
            calibration_data.setdefault("pixel", PixelPolicy(None).as_dict())
        calibration_data.setdefault("calibration_id", calibration_digest(calibration_data))
        bundle = cls(root.resolve(), model, calibration_data)
        bundle.verify()
        _publish_pair(root, model, calibration_data)
        return bundle

    def verify(
        self,
        *,
        expected_category: str | None = None,
        expected_model_id: str | None = None,
        expected_weights_sha256: str | None = None,
    ) -> None:
        _required_text(self.model, "model_id", "Model metadata")
        _required_text(self.model, "category", "Model metadata")
        _required_text(self.model, "family", "Model metadata")
        receipt = self.model.get("publication_receipt")
        if receipt is not None:
            if not isinstance(receipt, dict) or set(receipt) != {"path", "run_id", "spec_fingerprint"}:
                raise ModelBundleError("Invalid bundle publication receipt")
            if receipt["path"] != "../run_status.json":
                raise ModelBundleError("Bundle publication receipt must identify its parent run")
            _required_text(receipt, "run_id", "Publication receipt")
            _required_digest(receipt, "spec_fingerprint", "Publication receipt")
        version = self.model.get("schema_version")
        if version not in {1, SCHEMA_VERSION} or self.calibration.get("schema_version") != version:
            raise ModelBundleError("Unsupported or mismatched bundle schema_version")
        _validate_thresholds(self.calibration)
        if expected_category is not None and self.category != expected_category:
            raise ModelBundleError(
                f"Model category mismatch: bundle={self.category!r}, requested={expected_category!r}"
            )
        if expected_model_id is not None and self.model_id != expected_model_id:
            raise ModelBundleError(
                f"Model ID mismatch: bundle={self.model_id!r}, requested={expected_model_id!r}"
            )
        expected_fingerprint = _json_digest(
            {key: value for key, value in self.model.items() if key != "fingerprint"}
        )
        if self.model_fingerprint != expected_fingerprint:
            raise ModelBundleError("Model metadata fingerprint does not match its contents")
        if self.calibration.get("model_id") != self.model_id:
            raise ModelBundleError("Calibration model_id does not match model.json")
        if self.calibration.get("model_fingerprint") != self.model_fingerprint:
            raise ModelBundleError("Calibration is bound to a different model fingerprint")
        weights = self.model.get("weights")
        if not isinstance(weights, dict):
            raise ModelBundleError("Model metadata requires a weights reference")
        _required_text(weights, "path", "Weights")
        weight_hash = _required_digest(weights, "sha256", "Weights")
        candidate = self.artifact_path
        kind = weights.get("kind", "file")
        if kind not in {"file", "directory"}:
            raise ModelBundleError("Unsupported model artifact kind")
        if not (candidate.is_file() if kind == "file" else candidate.is_dir()):
            raise ModelBundleError(f"Referenced weights are missing: {candidate}")
        actual = artifact_reference(candidate)
        if actual["sha256"] != weight_hash or (kind == "directory" and actual["files"] != weights.get("files")):
            raise ModelBundleError(f"Weights checksum mismatch: {candidate}")
        if expected_weights_sha256 is not None and weight_hash != expected_weights_sha256:
            raise ModelBundleError("Requested weights checksum does not match the bundle")
        if self.calibration.get("weights_sha256") != weight_hash:
            raise ModelBundleError("Calibration weights checksum does not match the model")
        if version == SCHEMA_VERSION:
            if self.model.get("status") not in {"validated", "candidate", "exploratory"}:
                raise ModelBundleError("Unsupported model validation status")
            if self.model.get("score_space") != self.score_space:
                raise ModelBundleError("Calibration score space does not match the model")
            preprocessing = self.model.get("preprocessing")
            if not isinstance(preprocessing, dict) or not any(k != "fingerprint" for k in preprocessing):
                raise ModelBundleError("Model requires a preprocessing specification")
            transform_hash = _json_digest({k: v for k, v in preprocessing.items() if k != "fingerprint"})
            if preprocessing.get("fingerprint") != transform_hash:
                raise ModelBundleError("Preprocessing fingerprint does not match its contents")
            if self.calibration.get("preprocessing_fingerprint") != transform_hash:
                raise ModelBundleError("Calibration preprocessing fingerprint mismatch")
            identity = self.model.get("runtime_identity")
            if self.model["status"] == "validated" and identity is None:
                raise ModelBundleError("Validated bundle requires actual runtime identity; migrate as candidate first")
            if identity is not None:
                if not isinstance(identity, dict) or identity.get("version") != 1:
                    raise ModelBundleError("Unsupported runtime identity")
                for field in ("runtime_class", "decode", "transform", "encoder", "postprocess", "dependencies", "implementation"):
                    if not identity.get(field):
                        raise ModelBundleError(f"Runtime identity requires {field}")
                actual_hash = _json_digest({k: v for k, v in identity.items() if k != "fingerprint"})
                if identity.get("fingerprint") != actual_hash:
                    raise ModelBundleError("Runtime identity fingerprint mismatch")
                if self.calibration.get("runtime_fingerprint") != actual_hash:
                    raise ModelBundleError("Calibration runtime fingerprint mismatch")
                if identity.get("score_space") != self.score_space:
                    raise ModelBundleError("Runtime identity score space mismatch")
                if preprocessing.get("decode") != identity["decode"] or preprocessing.get("transform") != identity["transform"]:
                    raise ModelBundleError("Preprocessing differs from actual runtime identity")
                if "pixel" not in self.calibration:
                    raise ModelBundleError("Runtime-bound calibration requires explicit pixel policy")
            validation = self.calibration.get("validation")
            if not isinstance(validation, dict) or validation.get("split") != "validation":
                raise ModelBundleError("Calibration requires validation split provenance")
            if self.model["status"] == "validated" and validation.get("policy_evaluated") is False:
                raise ModelBundleError("Changed policy has not been validated")
            _required_digest(validation, "split_fingerprint", "Validation")
            count = validation.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ModelBundleError("Validation count must be a positive integer")
            if "sample_ids" in validation:
                ids = validation["sample_ids"]
                if not isinstance(ids, list) or any(not isinstance(v, str) or not v for v in ids):
                    raise ModelBundleError("Validation sample_ids must be non-empty strings")
                if len(ids) != count or len(set(ids)) != count:
                    raise ModelBundleError("Validation sample_ids do not match count or contain duplicates")
        if self.calibration_id != calibration_digest(self.calibration):
            raise ModelBundleError("Calibration ID does not match its contents")

    def with_policy(
        self, output: Path, *, image: Any, pixel: PixelPolicy | None = None,
    ) -> ModelBundle:
        """Create a candidate policy without transferring validation claims to new thresholds."""
        self.verify()
        if self.family == "patchcore_torchscript":
            raise ModelBundleError("Change the native source policy and repeat export parity before publishing an exported policy")
        if output.exists():
            raise FileExistsError(f"Policy changes require a new directory: {output}")
        calibration = {key: value for key, value in self.calibration.items() if key not in {
            "schema_version", "calibration_id", "model_fingerprint", "preprocessing_fingerprint",
            "runtime_fingerprint", "weights_sha256",
        }}
        calibration["image"] = {"pass_threshold": image.pass_threshold, "fail_threshold": image.fail_threshold}
        calibration["pixel"] = (pixel or PixelPolicy.from_dict(calibration.get("pixel"))).as_dict()
        calibration["source"] = "explicit_policy_override"
        calibration["threshold_source"] = "explicit_policy_override"
        calibration["objective"] = image.objective
        calibration["previous_calibration_id"] = self.calibration_id
        calibration["validation"] = {**calibration["validation"], "policy_evaluated": False}
        weights = {**self.model["weights"],
                   "path": Path(os.path.relpath(self.artifact_path, output.resolve())).as_posix()}
        runtime = None
        if "runtime_identity" in self.model:
            from industrial_inspection.models.registry import _BUNDLE_FAMILIES, load_bundle_model

            name = next(name for name, families in _BUNDLE_FAMILIES.items() if self.family in families)
            runtime, _ = load_bundle_model(name, self.root, device="cpu")
        return self.write(
            output, model_id=self.model_id, family=self.family, category=self.category,
            weights=weights, preprocessing=self.model["preprocessing"], calibration=calibration,
            training=self.model.get("training"), parameters=self.model.get("parameters"),
            software=self.model.get("software"), status="candidate", runtime=runtime,
        )


def resolve_device(requested: str, *, torch_module: Any | None = None) -> str:
    """Resolve a requested device without silently upgrading an explicit CPU request."""

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return "cpu"
    if torch_module is None:
        try:
            import torch as torch_module
        except (ImportError, OSError):
            torch_module = None
    cuda_available = bool(torch_module is not None and torch_module.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available in this environment")
    return "cuda" if cuda_available else "cpu"


def load_bundle(root: Path, **kwargs: Any) -> ModelBundle:
    """Compatibility function for callers that prefer a function-style loader."""

    return ModelBundle.load(root, **kwargs)
