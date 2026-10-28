"""Explicit, non-destructive upgrade of historical deployment metadata."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from industrial_inspection.models.artifacts import ModelBundle, ModelBundleError, sha256_file


def migrate_legacy_bundle(source: Path, output: Path, provenance_path: Path) -> ModelBundle:
    """Bind an existing calibration to recorded validation IDs without promoting its status."""
    legacy = ModelBundle.load(source, allow_legacy=True)
    if legacy.model["schema_version"] != 1:
        raise ModelBundleError("Migration expects a historical schema 1 bundle")
    if output.exists():
        raise FileExistsError(f"Migration requires a new output directory: {output}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest = provenance.get("manifest", {})
    expected_manifest = legacy.model.get("training", {}).get("manifest", {})
    if not expected_manifest or any(
        not manifest.get(key) or manifest[key] != expected_manifest.get(key)
        for key in ("fingerprint", "sha256")
    ):
        raise ModelBundleError("Validation provenance does not match the historical training manifest")
    validation = provenance.get("splits", {}).get("validation", {})
    ids, hashes = validation.get("sample_ids", []), validation.get("image_sha256", [])
    count = validation.get("count")
    if (
        not isinstance(count, int) or isinstance(count, bool) or count < 1
        or not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids)
        or len(ids) != count or len(set(ids)) != count
        or not isinstance(hashes, list) or len(hashes) != count
        or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)
        or count != legacy.calibration.get("validation", {}).get("count")
    ):
        raise ModelBundleError("Historical validation IDs, hashes or count are incomplete")
    calibration = {
        key: value for key, value in legacy.calibration.items()
        if key not in {"schema_version", "calibration_id", "model_fingerprint", "preprocessing_fingerprint"}
    }
    calibration["validation"] = {
        "split": "validation", "count": count, "sample_ids": ids, "image_sha256": hashes,
        "manifest_sha256": manifest["sha256"], "split_fingerprint": manifest["fingerprint"],
    }
    weights = dict(legacy.model["weights"])
    weights["path"] = Path(os.path.relpath(legacy.artifact_path, output.resolve())).as_posix()
    training = dict(legacy.model.get("training", {}))
    training["migration"] = {
        "source_model_sha256": sha256_file(source / "model.json"),
        "source_calibration_sha256": sha256_file(source / "calibration.json"),
        "provenance_sha256": sha256_file(provenance_path),
        "note": "Metadata migration only; no new fit, calibration or independent evaluation.",
    }
    return ModelBundle.write(
        output, model_id=legacy.model_id, family=legacy.family, category=legacy.category,
        weights=weights, preprocessing=legacy.model["preprocessing"], calibration=calibration,
        training=training, parameters=legacy.model.get("parameters"),
        software=legacy.model.get("software"), status="candidate",
    )
