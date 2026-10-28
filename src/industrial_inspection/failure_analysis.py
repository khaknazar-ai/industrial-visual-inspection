"""Failure taxonomy and deterministic case mining."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any


class FailureTaxonomy(StrEnum):
    ILLUMINATION_SHIFT = "illumination_shift"
    REFLECTION = "reflection"
    MISALIGNMENT = "object_misalignment"
    BLUR = "blur"
    TINY_DEFECT = "tiny_defect"
    EDGE_DEFECT = "edge_defect"
    BACKGROUND_CHANGE = "background_change"
    NORMAL_VARIATION = "normal_variation_false_positive"
    LOGICAL = "logical_anomaly"
    MASK_BOUNDARY = "mask_boundary_error"
    UNKNOWN = "unknown"


def classify_failure(metadata: dict[str, Any], *, area: int | None = None) -> FailureTaxonomy:
    if metadata.get("alignment_succeeded") is False:
        return FailureTaxonomy.MISALIGNMENT
    if metadata.get("quality_reason") == "blurred":
        return FailureTaxonomy.BLUR
    if area is not None and area < 32:
        return FailureTaxonomy.TINY_DEFECT
    return FailureTaxonomy.UNKNOWN


def mine_borderline_cases(
    rows: Iterable[dict[str, Any]], pass_threshold: float, fail_threshold: float, limit: int = 25
) -> list[dict[str, Any]]:
    """Return the cases closest to either review boundary."""

    candidates = [
        row
        for row in rows
        if pass_threshold <= float(row.get("anomaly_score", 0)) < fail_threshold
    ]
    candidates.sort(
        key=lambda row: min(
            abs(float(row["anomaly_score"]) - pass_threshold),
            abs(float(row["anomaly_score"]) - fail_threshold),
        )
    )
    return candidates[:limit]
