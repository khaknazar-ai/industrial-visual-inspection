"""Per-image EDA with bounded image memory and explicit acquisition limitations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from industrial_inspection.data.contracts import load_manifest, validate_manifest
from industrial_inspection.data.image_io import read_color_image, read_mask


def describe_manifest(manifest: Path, root: Path) -> dict:
    samples = load_manifest(manifest, root)
    audit = validate_manifest(manifest, root)
    rows = []
    for sample in samples:
        if not sample.image_path.is_file():
            continue
        image = read_color_image(sample.image_path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        row = {
            "sample_id": sample.sample_id, "split": sample.split, "label": sample.label,
            "width": gray.shape[1], "height": gray.shape[0],
            "brightness_mean": float(gray.mean()), "contrast_std": float(gray.std()),
            "mask_area_fraction": None, "mask_bbox_xywh": None,
        }
        if sample.mask_path is not None and sample.mask_path.exists():
            try:
                mask = read_mask(sample.mask_path)
                if mask is not None and mask.shape == gray.shape:
                    binary = (mask > 0).astype(np.uint8)
                    row["mask_area_fraction"] = float(binary.mean())
                    row["mask_bbox_xywh"] = list(cv2.boundingRect(binary))
            except (ValueError, OSError):
                pass  # The audit retains the exact annotation error.
        rows.append(row)
    return {
        "sample_count": len(samples),
        "split_counts": dict(Counter(sample.split for sample in samples)),
        "label_counts": dict(Counter(sample.label for sample in samples)),
        "category_counts": dict(Counter(sample.category for sample in samples)),
        "defect_type_counts": dict(Counter(sample.defect_type or sample.label for sample in samples)),
        "domain_counts": dict(Counter(sample.domain or "unknown" for sample in samples)),
        "lighting_counts": dict(Counter(sample.lighting or "unknown" for sample in samples)),
        "group_counts": {
            field: {split: len({getattr(sample, field) for sample in samples if sample.split == split and getattr(sample, field)})
                    for split in ("train", "validation", "test")}
            for field in ("source_group", "session_id", "physical_item_id")
        },
        "width": {"min": min((row["width"] for row in rows), default=None), "max": max((row["width"] for row in rows), default=None)},
        "height": {"min": min((row["height"] for row in rows), default=None), "max": max((row["height"] for row in rows), default=None)},
        "images": rows, "audit": audit.as_dict(),
        "limitations": audit.warnings,
    }
