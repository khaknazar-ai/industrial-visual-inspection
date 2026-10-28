"""Index official LOCO/AD2 trees and declared custom acquisition groups."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from industrial_inspection.core.paths import find_project_root
from industrial_inspection.data.contracts import InspectionSample, write_manifest
from industrial_inspection.data.image_io import read_color_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def _images(root: Path):
    return (path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _sample(
    image_path: Path, *, dataset: str, category: str, split: str, label: str,
    defect_type: str | None, mask_path: Path | None, source_group: str | None,
    sample_id: str, session_id: str | None = None, physical_item_id: str | None = None,
    domain: str | None = None, lighting: str | None = None, annotation_config_path: Path | None = None,
) -> InspectionSample:
    image = read_color_image(image_path)
    height, width = image.shape[:2] if image is not None else (1, 1)
    return InspectionSample(
        sample_id=sample_id, dataset=dataset, category=category, image_path=image_path,
        split=split, label=label, defect_type=defect_type, mask_path=mask_path,  # type: ignore[arg-type]
        source_group=source_group, width=width, height=height, session_id=session_id,
        physical_item_id=physical_item_id, domain=domain, lighting=lighting,
        annotation_config_path=annotation_config_path,
    )


def _stable_id(dataset: str, image_path: Path, root: Path) -> str:
    relative = image_path.relative_to(root).as_posix()
    return f"{dataset}-{hashlib.sha256(relative.encode()).hexdigest()[:24]}"


def _write(samples: list[InspectionSample], output: Path, project_root: Path | None, dataset_root: Path, protocol: str) -> Path:
    if not samples:
        raise ValueError("No images were found")
    write_manifest(
        samples, output, project_root or find_project_root(),
        root_aliases={"dataset": dataset_root}, split_protocol={"name": protocol, "version": 1},
    )
    return output


def build_loco_manifest(
    dataset_root: Path, category: str, output: Path, *, project_root: Path | None = None,
) -> Path:
    """Keep official splits and per-image instance/type mask directories intact."""
    category_root = dataset_root / category
    if not category_root.is_dir():
        raise FileNotFoundError(category_root)
    samples: list[InspectionSample] = []
    for split in ("train", "validation", "test"):
        for image_path in _images(category_root / split):
            relative = image_path.relative_to(category_root / split)
            if len(relative.parts) != 2 or relative.parts[0] not in {"good", "logical_anomalies", "structural_anomalies"}:
                raise ValueError(f"Unexpected LOCO image layout: {image_path}")
            defect = relative.parts[0]
            normal = defect == "good"
            samples.append(_sample(
                image_path, dataset="mvtec_loco", category=category, split=split,
                label="normal" if normal else "anomaly", defect_type=None if normal else defect,
                mask_path=None if normal else category_root / "ground_truth" / defect / image_path.stem,
                source_group=None, sample_id=_stable_id("mvtec-loco", image_path, dataset_root),
                annotation_config_path=category_root / "defects_config.json",
            ))
    return _write(samples, output, project_root, dataset_root, "mvtec_loco_official")


def build_ad2_manifest(
    dataset_root: Path, output: Path, *, project_root: Path | None = None, include_private: bool = True,
) -> Path:
    """Follow MVTec's public loader; private images have no known class label.

    Lighting is not inferred from filenames. The private_mixed domain denotes the
    mixed-condition split; per-image lighting remains unknown without metadata.
    """
    samples: list[InspectionSample] = []
    split_names = {"train", "validation", "test_public", "test_private", "test_private_mixed"}
    for image_path in _images(dataset_root):
        parts = image_path.relative_to(dataset_root).parts
        if "ground_truth" in parts:
            continue
        if len(parts) < 3 or parts[1] not in split_names:
            raise ValueError(f"AD2 image is outside an official split directory: {image_path}")
        category, official_split = parts[:2]
        private = official_split in {"test_private", "test_private_mixed"}
        if private and not include_private:
            continue
        mask = None
        if private:
            if len(parts) != 3:
                raise ValueError(f"Unexpected private AD2 layout: {image_path}")
            label, defect = "unknown", None
        else:
            if len(parts) != 4 or parts[2] not in {"good", "bad"}:
                raise ValueError(f"Unknown AD2 label directory: {image_path}")
            label = "normal" if parts[2] == "good" else "anomaly"
            defect = None if label == "normal" else "unspecified"
            if label == "anomaly":
                mask = dataset_root / category / official_split / "ground_truth" / "bad" / f"{image_path.stem}_mask.png"
        samples.append(_sample(
            image_path, dataset="mvtec_ad2", category=category,
            split=official_split if official_split in {"train", "validation"} else "test",
            label=label, defect_type=defect, mask_path=mask, source_group=None,
            sample_id=_stable_id("mvtec-ad2", image_path, dataset_root), domain=official_split,
            lighting=None,
        ))
    protocol = "mvtec_ad2_official_all" if include_private else "mvtec_ad2_official_public"
    return _write(samples, output, project_root, dataset_root, protocol)


def build_custom_manifest(
    dataset_root: Path, output: Path, *, project_root: Path | None = None,
    dataset_name: str = "custom_industrial", category: str = "assembly",
) -> Path:
    """Mirror nested masks; groups require metadata or session/item folders."""
    samples: list[InspectionSample] = []
    metadata: dict[str, dict[str, str]] = {}
    metadata_path = dataset_root / "metadata.csv"
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("path") or row.get("image_path") or "").replace("\\", "/")
                if not key or key in metadata:
                    raise ValueError(f"Missing or duplicated metadata path: {key!r}")
                metadata[key] = row
    used_metadata = set()
    for split in ("train", "validation", "test"):
        for label in ("normal", "anomaly"):
            image_root = dataset_root / "images" / split / label
            for image_path in _images(image_root):
                relative = image_path.relative_to(dataset_root).as_posix()
                nested = image_path.relative_to(image_root)
                row = metadata.get(relative, {})
                if row:
                    used_metadata.add(relative)
                session_id = row.get("session_id") or (nested.parts[0] if len(nested.parts) == 3 else None)
                item_id = row.get("physical_item_id") or row.get("item_id") or (nested.parts[1] if len(nested.parts) == 3 else None)
                mask = dataset_root / "masks" / split / label / nested.with_suffix(".png") if label == "anomaly" else None
                samples.append(_sample(
                    image_path, dataset=row.get("dataset") or dataset_name,
                    category=row.get("category") or category, split=split, label=label,
                    defect_type=row.get("anomaly_type") or ("unknown" if label == "anomaly" else None),
                    mask_path=mask if mask is not None and mask.is_file() else None,
                    source_group=row.get("source_group") or (f"{session_id}:{item_id}" if session_id and item_id else None),
                    session_id=session_id, physical_item_id=item_id,
                    domain=row.get("domain") or None, lighting=row.get("lighting") or None,
                    sample_id=row.get("sample_id") or _stable_id(dataset_name, image_path, dataset_root),
                ))
    if metadata.keys() - used_metadata:
        raise ValueError(f"Metadata references unknown images: {sorted(metadata.keys() - used_metadata)[:5]}")
    return _write(samples, output, project_root, dataset_root, "custom_declared_groups")
