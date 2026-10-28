"""Manifest-first split bridge for optional Anomalib training.

This module does not instantiate Anomalib or download data. It freezes the exact samples that a
training adapter is allowed to consume and writes their identities into the run provenance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.data.contracts import (
    InspectionSample,
    load_manifest,
    manifest_sidecar_path,
    require_known_labels,
    validate_manifest,
)
from industrial_inspection.data.integrity import FrozenInputs, read_verified_bytes


@dataclass(frozen=True, slots=True)
class ManifestSplits:
    manifest_path: Path
    manifest_sha256: str
    train: tuple[InspectionSample, ...]
    validation: tuple[InspectionSample, ...]
    test: tuple[InspectionSample, ...]
    fingerprint: str
    inputs: FrozenInputs = field(repr=False)
    split_protocol: dict[str, object] = field(default_factory=lambda: {"name": "declared", "version": 1})

    def ids(self, split: str) -> tuple[str, ...]:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        return tuple(sample.sample_id for sample in getattr(self, split))

    def verify_unchanged(self) -> None:
        """Reject drift; never replace the identity of a partly computed run."""
        self.inputs.verify_unchanged()
        if _split_fingerprint(list(self.train + self.validation + self.test), self.manifest_path.parent,
                              self.split_protocol, inputs=self.inputs) != self.fingerprint:
            raise ValueError("Dataset fingerprint changed after split freeze")

    def provenance(self) -> dict[str, object]:
        return {
            "manifest": {
                "path": str(self.manifest_path),
                "sha256": self.manifest_sha256,
                "sidecar_sha256": self.inputs.optional_file_hash(manifest_sidecar_path(self.manifest_path)),
                "fingerprint": self.fingerprint,
                "fingerprint_schema_version": 2,
                "split_protocol": self.split_protocol,
            },
            "splits": {name: {
                "count": len(samples), "sample_ids": list(self.ids(name)),
                "image_sha256": [self.inputs.file_hash(sample.image_path) for sample in samples],
                "mask_sha256": [self.inputs.annotation_hash(sample.mask_path) for sample in samples],
                "annotation_config_sha256": [self.inputs.annotation_hash(sample.annotation_config_path) for sample in samples],
            } for name, samples in (("train", self.train), ("validation", self.validation), ("test", self.test))},
            "training_policy": {
                "evidence_scope": "declared split assignment; observed_samples.json records actual consumption",
                "fit_split": "train",
                "fit_labels": ["normal"],
                "calibration_split": "validation",
                "evaluation_split": "test",
                "test_seen_during_fit": False,
            },
        }

    def write_provenance(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.provenance(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _sample_hash(sample: InspectionSample) -> str:
    if not sample.image_path.is_file():
        raise ValueError(f"Manifest sample is missing: {sample.image_path}")
    return sha256_file(sample.image_path)


def _annotation_hash(path: Path | None) -> object:
    if path is None:
        return None
    if path.is_dir():
        return {item.relative_to(path).as_posix(): sha256_file(item) for item in sorted(path.rglob("*")) if item.is_file()}
    return sha256_file(path)


def _split_fingerprint(
    samples: list[InspectionSample], root: Path, split_protocol: dict[str, object] | None = None,
    *, inputs: FrozenInputs | None = None,
) -> str:
    """Content identity v2: independent of machine paths, sensitive to all annotations."""
    rows = [
        {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "label": sample.label,
            "category": sample.category,
            "dataset": sample.dataset,
            "sha256": inputs.file_hash(sample.image_path) if inputs else _sample_hash(sample),
            "mask_sha256": inputs.annotation_hash(sample.mask_path) if inputs else _annotation_hash(sample.mask_path),
            "annotation_config_sha256": inputs.annotation_hash(sample.annotation_config_path) if inputs else _annotation_hash(sample.annotation_config_path),
            "defect_type": sample.defect_type,
            "source_group": sample.source_group,
            "session_id": sample.session_id,
            "physical_item_id": sample.physical_item_id,
            "domain": sample.domain,
            "lighting": sample.lighting,
            "size": [sample.width, sample.height],
        }
        for sample in sorted(samples, key=lambda item: item.sample_id)
    ]
    content = {"fingerprint_schema_version": 2, "split_protocol": split_protocol or {"name": "declared", "version": 1}, "samples": rows}
    payload = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def load_manifest_splits(
    manifest_path: Path,
    *,
    project_root: Path,
    category: str | None = None,
    root_aliases: dict[str, Path] | None = None,
) -> ManifestSplits:
    """Load and freeze the only split assignment allowed for a training run."""

    sidecar = manifest_sidecar_path(manifest_path)
    metadata_inputs = FrozenInputs([manifest_path], optional_paths=[sidecar])
    with metadata_inputs.scope():
        samples = load_manifest(manifest_path, project_root, root_aliases=root_aliases)
        metadata = json.loads(read_verified_bytes(sidecar).decode("utf-8")) if sidecar.is_file() else {}
    # Snapshot every manifest input before auditing any decoded bytes. Category selection
    # changes the fit/evaluation identity, but the manifest audit still visits all rows.
    inputs = FrozenInputs([manifest_path, *(path for sample in samples for path in
                          (sample.image_path, sample.mask_path, sample.annotation_config_path) if path is not None)],
                          optional_paths=[sidecar])
    metadata_inputs.verify_unchanged()
    with inputs.scope():
        audit = validate_manifest(manifest_path, project_root, root_aliases=root_aliases)
    if not audit.valid:
        raise ValueError(f"Manifest audit failed: {audit.errors}")
    if category is not None:
        samples = [sample for sample in samples if sample.category == category]
    if not samples:
        raise ValueError("Manifest contains no samples for the requested category")
    require_known_labels(samples)
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    test = [sample for sample in samples if sample.split == "test"]
    if not train or not validation or not test:
        raise ValueError("Manifest must contain non-empty train, validation, and test splits")
    if any(sample.label != "normal" for sample in train):
        raise ValueError("Manifest train split contains an anomaly")
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest sample IDs must be unique")
    protocol = metadata.get("split_protocol", {"name": "declared", "version": 1})
    splits = ManifestSplits(
        manifest_path=manifest_path.resolve(),
        manifest_sha256=inputs.file_hash(manifest_path),
        train=tuple(sorted(train, key=lambda item: item.sample_id)),
        validation=tuple(sorted(validation, key=lambda item: item.sample_id)),
        test=tuple(sorted(test, key=lambda item: item.sample_id)),
        fingerprint=_split_fingerprint(samples, project_root, protocol, inputs=inputs),
        inputs=inputs,
        split_protocol=protocol,
    )
    splits.verify_unchanged()
    return splits


class ManifestDataModule:
    """Small dependency-free adapter consumed by optional training integrations."""

    def __init__(self, splits: ManifestSplits) -> None:
        self.splits = splits

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        *,
        project_root: Path,
        category: str | None = None,
    ) -> ManifestDataModule:
        return cls(load_manifest_splits(manifest_path, project_root=project_root, category=category))

    def samples(self, split: str) -> tuple[InspectionSample, ...]:
        return getattr(self.splits, split)

    def train_images(self) -> tuple[Path, ...]:
        return tuple(sample.image_path for sample in self.splits.train)

    def validation_images(self) -> tuple[Path, ...]:
        return tuple(sample.image_path for sample in self.splits.validation)

    def test_images(self) -> tuple[Path, ...]:
        return tuple(sample.image_path for sample in self.splits.test)
