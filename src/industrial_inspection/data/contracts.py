"""Canonical dataset manifest and integrity audit."""

from __future__ import annotations

import csv
import io
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np

from industrial_inspection.core.paths import find_project_root
from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.data.image_io import read_color_image, read_mask
from industrial_inspection.data.integrity import read_verified_bytes

Split = Literal["train", "validation", "test"]
Label = Literal["normal", "anomaly", "unknown"]
LEGACY_MANIFEST_COLUMNS = (
    "sample_id",
    "dataset",
    "category",
    "split",
    "label",
    "defect_type",
    "image_path",
    "mask_path",
    "source_group",
    "width",
    "height",
    "sha256_optional",
)
MANIFEST_COLUMNS = LEGACY_MANIFEST_COLUMNS + (
    "session_id", "physical_item_id", "domain", "lighting", "annotation_config_path",
)
MANIFEST_SCHEMA_VERSION = 2


def manifest_sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


@dataclass(frozen=True, slots=True)
class InspectionSample:
    """One image and its optional pixel-level ground truth."""

    sample_id: str
    dataset: str
    category: str
    image_path: Path
    split: Split
    label: Label
    defect_type: str | None
    mask_path: Path | None
    source_group: str | None
    width: int
    height: int
    sha256_optional: str | None = None
    session_id: str | None = None
    physical_item_id: str | None = None
    domain: str | None = None
    lighting: str | None = None
    annotation_config_path: Path | None = None

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {self.split}")
        if self.label not in {"normal", "anomaly", "unknown"}:
            raise ValueError(f"Invalid label: {self.label}")
        if self.label == "unknown" and (self.split != "test" or self.mask_path is not None):
            raise ValueError("Unknown labels are allowed only in unannotated test data")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image width and height must be positive")
        if self.label == "normal" and self.mask_path is not None:
            raise ValueError("Normal samples cannot have anomaly masks")

    def to_row(self, root: Path, root_aliases: dict[str, Path] | None = None) -> dict[str, str | int]:
        def portable(path: Path | None) -> str:
            if path is None:
                return ""
            absolute = path.resolve()
            for alias, location in (root_aliases or {}).items():
                if absolute.is_relative_to(location.resolve()):
                    return f"@{alias}/{absolute.relative_to(location.resolve()).as_posix()}"
            try:
                return absolute.relative_to(root.resolve()).as_posix()
            except ValueError:
                return absolute.as_posix()

        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "category": self.category,
            "split": self.split,
            "label": self.label,
            "defect_type": self.defect_type or "",
            "image_path": portable(self.image_path),
            "mask_path": portable(self.mask_path),
            "source_group": self.source_group or "",
            "width": self.width,
            "height": self.height,
            "sha256_optional": self.sha256_optional or "",
            "session_id": self.session_id or "",
            "physical_item_id": self.physical_item_id or "",
            "domain": self.domain or "",
            "lighting": self.lighting or "",
            "annotation_config_path": portable(self.annotation_config_path),
        }

    @classmethod
    def from_row(cls, row: dict[str, str], root: Path, root_aliases: dict[str, Path] | None = None) -> InspectionSample:
        def resolve(value: str) -> Path | None:
            if not value:
                return None
            if value.startswith("@"):
                alias, separator, relative = value[1:].partition("/")
                if not separator or alias not in (root_aliases or {}):
                    raise ValueError(f"Unresolved manifest root alias: {alias}")
                parts = PurePosixPath(relative)
                if parts.is_absolute() or ".." in parts.parts or "\\" in relative or ":" in relative:
                    raise ValueError(f"Invalid alias path: {value}")
                location = root_aliases[alias].resolve()
                resolved = location.joinpath(*parts.parts).resolve()
                if not resolved.is_relative_to(location):
                    raise ValueError(f"Alias path escapes its root: {value}")
                return resolved
            path = Path(value)
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        image_path = resolve(row["image_path"])
        if image_path is None:
            raise ValueError("image_path is required")
        return cls(
            sample_id=row["sample_id"],
            dataset=row["dataset"],
            category=row["category"],
            image_path=image_path,
            split=row["split"],  # type: ignore[arg-type]
            label=row["label"],  # type: ignore[arg-type]
            defect_type=row.get("defect_type") or None,
            mask_path=resolve(row.get("mask_path", "")),
            source_group=row.get("source_group") or None,
            width=int(row["width"]),
            height=int(row["height"]),
            sha256_optional=row.get("sha256_optional") or None,
            session_id=row.get("session_id") or None,
            physical_item_id=row.get("physical_item_id") or None,
            domain=row.get("domain") or None,
            lighting=row.get("lighting") or None,
            annotation_config_path=resolve(row.get("annotation_config_path", "")),
        )


@dataclass(slots=True)
class AuditReport:
    dataset: str
    manifest_path: str
    valid: bool
    sample_count: int
    counts: dict[str, int] = field(default_factory=dict)
    resolutions: dict[str, float | int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_manifest(
    samples: list[InspectionSample], path: Path, root: Path | None = None, *,
    root_aliases: dict[str, Path] | None = None, split_protocol: dict[str, Any] | None = None,
) -> None:
    """Write a deterministic, portable CSV manifest."""

    project_root = (root or find_project_root()).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    aliases = root_aliases or {}
    if any(not name or not name.replace("_", "").replace("-", "").isalnum() for name in aliases):
        raise ValueError("Root aliases must contain only letters, digits, underscore and hyphen")
    if len({sample.sample_id for sample in ordered}) != len(ordered):
        raise ValueError("sample_id values must be unique")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sample.to_row(project_root, aliases) for sample in ordered)
    sidecar = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "path_mode": "aliases_or_project_relative",
        "root_aliases": {
            alias: location.resolve().relative_to(project_root).as_posix()
            if location.resolve().is_relative_to(project_root) else location.resolve().as_posix()
            for alias, location in aliases.items()
        },
        "split_protocol": split_protocol or {"name": "declared", "version": 1},
        "sample_count": len(ordered),
        "columns": list(MANIFEST_COLUMNS),
        "manifest_sha256": sha256_file(path),
    }
    manifest_sidecar_path(path).write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path, root: Path | None = None, *, root_aliases: dict[str, Path] | None = None) -> list[InspectionSample]:
    """Load a canonical CSV manifest and reject unexpected schemas."""

    sidecar_path = manifest_sidecar_path(path)
    sidecar: dict[str, Any] = {}
    if sidecar_path.is_file():
        sidecar = json.loads(read_verified_bytes(sidecar_path).decode("utf-8"))
        if sidecar.get("schema_version") not in {1, MANIFEST_SCHEMA_VERSION}:
            raise ValueError(f"Unsupported manifest sidecar schema: {sidecar_path}")
        expected_hash = sidecar.get("manifest_sha256")
        if expected_hash and expected_hash != sha256_file(path):
            raise ValueError(f"Manifest sidecar hash does not match: {path}")
    project_root = (root or find_project_root(path.parent)).resolve()
    aliases = {alias: (project_root / value).resolve() for alias, value in sidecar.get("root_aliases", {}).items()}
    for alias in aliases:
        override = os.environ.get(f"II_MANIFEST_ROOT_{alias.upper()}")
        if override:
            aliases[alias] = Path(override).resolve()
    aliases.update(root_aliases or {})
    with io.StringIO(read_verified_bytes(path).decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle)
        required = MANIFEST_COLUMNS if sidecar.get("schema_version") == 2 else LEGACY_MANIFEST_COLUMNS
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        samples = [InspectionSample.from_row(row, project_root, aliases) for row in reader]
    if sidecar and sidecar.get("sample_count") != len(samples):
        raise ValueError("Manifest sidecar sample_count does not match CSV")
    return samples


def require_known_labels(samples: list[InspectionSample]) -> None:
    unknown = [sample.sample_id for sample in samples if sample.label == "unknown"]
    if unknown:
        raise ValueError(f"Unknown test labels cannot be used for calibration or metrics; select a public manifest: {unknown[:5]}")


def validate_manifest(path: Path, root: Path | None = None, *, root_aliases: dict[str, Path] | None = None) -> AuditReport:
    """Audit files, masks, hashes, and split leakage for a manifest."""

    samples = load_manifest(path, root, root_aliases=root_aliases)
    errors: list[str] = []
    warnings: list[str] = []
    hashes: dict[str, list[InspectionSample]] = defaultdict(list)
    group_splits: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []

    ids = [sample.sample_id for sample in samples]
    if not samples:
        errors.append("manifest is empty")
    for duplicate_id, count in Counter(ids).items():
        if count > 1:
            errors.append(f"duplicate sample_id: {duplicate_id}")

    for sample in samples:
        counts[f"split:{sample.split}"] += 1
        counts[f"label:{sample.label}"] += 1
        counts[f"{sample.split}:{sample.label}"] += 1
        counts[f"category:{sample.category}"] += 1
        widths.append(sample.width)
        heights.append(sample.height)

        if sample.split == "train" and sample.label != "normal":
            errors.append(f"{sample.sample_id}: anomaly is not allowed in normal-only train")
        for kind in ("source_group", "session_id", "physical_item_id"):
            value = getattr(sample, kind)
            if value:
                group_splits[f"{kind}:{sample.dataset}:{value}"].add(sample.split)
        if not sample.image_path.is_file():
            errors.append(f"{sample.sample_id}: image does not exist: {sample.image_path}")
            continue
        image = read_color_image(sample.image_path)
        if image is None:
            errors.append(f"{sample.sample_id}: image is corrupt or unsupported")
            continue
        actual_height, actual_width = image.shape[:2]
        if (actual_width, actual_height) != (sample.width, sample.height):
            errors.append(
                f"{sample.sample_id}: manifest size {sample.width}x{sample.height} "
                f"!= image size {actual_width}x{actual_height}"
            )

        digest = sha256_file(sample.image_path)
        if sample.sha256_optional and sample.sha256_optional != digest:
            errors.append(f"{sample.sample_id}: declared SHA-256 does not match image content")
        hashes[digest].append(sample)
        if sample.mask_path is not None:
            if not sample.mask_path.exists():
                errors.append(f"{sample.sample_id}: mask does not exist: {sample.mask_path}")
            else:
                try:
                    mask = read_mask(sample.mask_path)
                except (ValueError, OSError) as exc:
                    errors.append(f"{sample.sample_id}: invalid mask: {exc}")
                    mask = None
                if mask is None:
                    errors.append(f"{sample.sample_id}: mask is corrupt or unsupported")
                elif mask.shape != image.shape[:2]:
                    errors.append(f"{sample.sample_id}: mask/image dimensions differ")
                elif not (mask > 0).any():
                    warnings.append(f"{sample.sample_id}: anomaly mask is empty")
                elif sample.dataset == "mvtec_ad2" and not set(np.unique(mask)).issubset({0, 255}):
                    errors.append(f"{sample.sample_id}: AD2 mask must contain only 0/255")
        elif sample.label == "anomaly":
            message = f"{sample.sample_id}: anomaly has no pixel mask"
            if sample.dataset in {"mvtec_ad", "mvtec_loco", "mvtec_ad2", "visa"}:
                errors.append(message)
            else:
                warnings.append(message)
        if sample.annotation_config_path is not None:
            try:
                config = json.loads(read_verified_bytes(sample.annotation_config_path).decode("utf-8"))
                if not isinstance(config, list) or not config:
                    raise ValueError("expected a non-empty defects list")
                values = set()
                for entry in config:
                    value = entry["pixel_value"]
                    if type(value) is not int or not 1 <= value <= 255 or value in values:
                        raise ValueError("defect pixel values must be unique integers in 1..255")
                    values.add(value)
                    threshold = entry["saturation_threshold"]
                    relative = entry["relative_saturation"]
                    if not entry["defect_name"] or type(relative) is not bool or not np.isfinite(threshold) or threshold <= 0 or (relative and threshold > 1):
                        raise ValueError("invalid saturation threshold or defect name")
                if sample.mask_path is not None and sample.mask_path.is_dir():
                    for channel_path in sample.mask_path.glob("*.png"):
                        channel = read_mask(channel_path)
                        if channel is None:
                            continue
                        foreground = set(np.unique(channel)) - {0}
                        if len(foreground) != 1 or not foreground.issubset(values):
                            raise ValueError(f"LOCO channel must encode one declared defect type: {channel_path.name}")
            except (ValueError, OSError, KeyError, TypeError) as exc:
                errors.append(f"{sample.sample_id}: invalid annotation config: {exc}")

    duplicates: list[dict[str, Any]] = []
    for digest, group in hashes.items():
        if len(group) <= 1:
            continue
        splits = sorted({item.split for item in group})
        record = {"sha256": digest, "sample_ids": [item.sample_id for item in group], "splits": splits}
        duplicates.append(record)
        if len(splits) > 1:
            errors.append(f"content leakage across splits: {record['sample_ids']}")
        else:
            warnings.append(f"duplicate content within {splits[0]}: {record['sample_ids']}")

    for group, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(f"{group.split(':')[0]} leakage for {group}: {sorted(splits)}")

    ungrouped = sum(not sample.session_id or not sample.physical_item_id for sample in samples)
    if ungrouped:
        warnings.append(f"{ungrouped} samples lack session/item identity; physical-group independence is unverified")
    if counts.get("label:unknown", 0):
        warnings.append("Private test labels are unknown; supervised metrics require a public-only manifest")

    resolutions: dict[str, float | int] = {}
    if samples:
        areas = [w * h for w, h in zip(widths, heights, strict=True)]
        resolutions = {
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "mean_pixels": round(sum(areas) / len(areas), 2),
        }
    dataset = samples[0].dataset if samples else "unknown"
    return AuditReport(
        dataset=dataset,
        manifest_path=str(path.resolve()),
        valid=not errors,
        sample_count=len(samples),
        counts=dict(sorted(counts.items())),
        resolutions=resolutions,
        errors=errors,
        warnings=warnings,
        duplicates=duplicates,
    )


def write_audit_report(report: AuditReport, output: Path) -> None:
    from industrial_inspection.core.json_files import write_new_json

    write_new_json(output, report.as_dict())
