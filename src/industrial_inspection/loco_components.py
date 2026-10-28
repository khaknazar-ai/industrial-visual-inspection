"""Small OpenCV-only component proposals for the LOCO splicing demo.

This is a transparent heuristic, not a qualified semantic detector. It uses
image pixels only and never reads LOCO ground-truth masks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from industrial_inspection.logical import (
    ComponentInstance,
    ComponentRule,
    ComponentSelector,
    LogicalCheck,
    RelativePositionRule,
    RuleBasedAssemblyChecker,
)


@dataclass(frozen=True, slots=True)
class SplicingConnectorProfile:
    """Fixed, inspectable thresholds for a first CPU demonstration."""

    warm_hue_ranges: tuple[tuple[int, int], ...] = ((0, 15),)
    min_saturation: int = 100
    min_value: int = 60
    endpoint_min_area: int = 300
    cable_min_area: int = 100
    cable_row_fraction: float = 0.42
    cable_merge_gap_fraction: float = 0.12
    max_cable_endpoint_gap_fraction: float = 0.11
    min_cable_corridor_coverage: float = 0.68
    max_cable_vertical_offset_fraction: float = 0.23
    min_cable_continuity_saturation: int = 40
    min_cable_continuity_value: int = 20
    cable_column_occupancy_fraction: float = 0.10
    max_cable_internal_gap_fraction: float = 0.06
    endpoint_zone_fraction: float = 0.30
    cable_zone_fraction: tuple[float, float] = (0.28, 0.72)
    vertical_zone_fraction: tuple[float, float] = (0.18, 0.90)

    def __post_init__(self) -> None:
        if not self.warm_hue_ranges or any(
            type(low) is not int or type(high) is not int or not 0 <= low <= high <= 179
            for low, high in self.warm_hue_ranges
        ):
            raise ValueError("warm_hue_ranges must contain valid OpenCV hue intervals")
        for name in (
            "min_saturation", "min_value",
            "min_cable_continuity_saturation", "min_cable_continuity_value",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError(f"{name} must be an integer from 0 to 255")
        for name in ("endpoint_min_area", "cable_min_area"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.cable_row_fraction < 1:
            raise ValueError("cable_row_fraction must be between 0 and 1")
        if not 0 <= self.cable_merge_gap_fraction < 0.25:
            raise ValueError("cable_merge_gap_fraction must be in [0, 0.25)")
        if not 0 < self.max_cable_endpoint_gap_fraction < 0.25:
            raise ValueError("max_cable_endpoint_gap_fraction must be in (0, 0.25)")
        if not 0 < self.min_cable_corridor_coverage <= 1:
            raise ValueError("min_cable_corridor_coverage must be in (0, 1]")
        if not 0 < self.max_cable_vertical_offset_fraction < 0.5:
            raise ValueError("max_cable_vertical_offset_fraction must be in (0, 0.5)")
        if not 0 < self.cable_column_occupancy_fraction < 1:
            raise ValueError("cable_column_occupancy_fraction must be in (0, 1)")
        if not 0 < self.max_cable_internal_gap_fraction < 0.25:
            raise ValueError("max_cable_internal_gap_fraction must be in (0, 0.25)")
        if not 0.25 <= self.endpoint_zone_fraction < 0.5:
            raise ValueError("endpoint_zone_fraction must be in [0.25, 0.5)")
        if (len(self.cable_zone_fraction) != 2
                or not 0 < self.cable_zone_fraction[0] < self.cable_zone_fraction[1] < 1):
            raise ValueError("cable_zone_fraction must be an increasing pair inside (0, 1)")
        if (len(self.vertical_zone_fraction) != 2
                or not 0 < self.vertical_zone_fraction[0] < self.vertical_zone_fraction[1] < 1):
            raise ValueError("vertical_zone_fraction must be an increasing pair inside (0, 1)")


@dataclass(frozen=True, slots=True)
class SplicingConnectorPrediction:
    status: str
    issues: tuple[str, ...]
    components: tuple[ComponentInstance, ...]
    check: LogicalCheck
    additional_checks: tuple[dict[str, object], ...] = ()
    appearance_review: dict[str, object] | None = None

    def as_dict(self) -> dict:
        logical_check = self.check.as_dict()
        logical_check["additional_checks"] = list(self.additional_checks)
        return {
            "category": "splicing_connectors",
            "status": self.status,
            "issues": list(self.issues),
            "components": [component.as_dict() for component in self.components],
            "logical_check": logical_check,
            "appearance_review": self.appearance_review,
            "extractor": "opencv_hsv_endpoint_and_row_band_proposals_v1",
            "logical_rule_spec": "splicing_connector_count_zone_order_cable_continuity_v5",
            "calibration_status": "unqualified_demo_heuristic",
            "ground_truth_used_as_input": False,
        }


def _mask_component(
    mask: np.ndarray,
    *,
    class_name: str,
    component_id: str,
    x_offset: int = 0,
    y_offset: int = 0,
    min_area: int,
) -> ComponentInstance | None:
    ys, xs = np.nonzero(mask)
    area = int(xs.size)
    if area < min_area:
        return None
    x1 = int(xs.min()) + x_offset
    y1 = int(ys.min()) + y_offset
    x2 = int(xs.max()) + 1 + x_offset
    y2 = int(ys.max()) + 1 + y_offset
    centroid = (float(xs.mean() + x_offset), float(ys.mean() + y_offset))
    return ComponentInstance(
        component_id=component_id,
        class_name=class_name,
        bbox_xyxy=(x1, y1, x2, y2),
        area=area,
        centroid=centroid,
        id_source="provided",
    )


def _warm_mask(hsv: np.ndarray, profile: SplicingConnectorProfile) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for hue_low, hue_high in profile.warm_hue_ranges:
        band = cv2.inRange(
            hsv,
            (hue_low, profile.min_saturation, profile.min_value),
            (hue_high, 255, 255),
        )
        mask = cv2.bitwise_or(mask, band)
    return mask


def _keep_large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area:
            kept[labels == index] = 255
    return kept


def _cable_components(
    hsv: np.ndarray, profile: SplicingConnectorProfile,
) -> tuple[list[ComponentInstance], np.ndarray]:
    height, width = hsv.shape[:2]
    zone_x1 = int(round(width * profile.cable_zone_fraction[0]))
    zone_x2 = int(round(width * profile.cable_zone_fraction[1]))
    zone_y1 = int(round(height * profile.vertical_zone_fraction[0]))
    zone_y2 = int(round(height * profile.vertical_zone_fraction[1]))
    saturated = cv2.inRange(
        hsv,
        (0, profile.min_saturation, profile.min_value),
        (179, 255, 255),
    )
    central = np.zeros_like(saturated)
    central[zone_y1:zone_y2, zone_x1:zone_x2] = saturated[zone_y1:zone_y2, zone_x1:zone_x2]
    row_counts = np.count_nonzero(central[zone_y1:zone_y2, zone_x1:zone_x2], axis=1)
    row_threshold = max(1, int(round((zone_x2 - zone_x1) * profile.cable_row_fraction)))
    active_rows = row_counts >= row_threshold

    raw_runs = []
    start = None
    for index, active in enumerate(active_rows):
        if active and start is None:
            start = index
        elif not active and start is not None:
            raw_runs.append((start, index))
            start = None
    if start is not None:
        raw_runs.append((start, len(active_rows)))

    merge_gap = int(round(height * profile.cable_merge_gap_fraction))
    runs: list[list[int]] = []
    for run_start, run_end in raw_runs:
        if runs and run_start - runs[-1][1] <= merge_gap:
            runs[-1][1] = run_end
        else:
            runs.append([run_start, run_end])

    proposals = []
    line_mask = np.zeros_like(central)
    for run_start, run_end in runs:
        y1 = zone_y1 + run_start
        y2 = zone_y1 + run_end
        if int(row_counts[run_start:run_end].sum()) < profile.cable_min_area:
            continue
        line_mask[y1:y2, zone_x1:zone_x2] = central[y1:y2, zone_x1:zone_x2]
        component = _mask_component(
            central[y1:y2, zone_x1:zone_x2],
            class_name="cable",
            component_id=f"cable:{len(proposals) + 1}",
            x_offset=zone_x1,
            y_offset=y1,
            min_area=profile.cable_min_area,
        )
        if component is not None:
            proposals.append(component)
    return proposals, line_mask


def extract_splicing_connector_components(
    image_bgr: np.ndarray,
    profile: SplicingConnectorProfile = SplicingConnectorProfile(),
) -> tuple[ComponentInstance, ...]:
    """Propose left/right warm connector regions and central cable bands on CPU.

    Thresholds are fixed defaults for a demonstration and should be treated as
    unqualified until checked on a frozen validation protocol. Returned boxes
    and centroids stay in the input image's original coordinate system.
    """
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8
            or not image_bgr.shape[0] or not image_bgr.shape[1]):
        raise ValueError("Expected a nonempty uint8 BGR image")
    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    warm = _warm_mask(hsv, profile)
    cables, cable_mask = _cable_components(hsv, profile)

    # Exclude the detected central line from warm connector pixels. This keeps
    # red/orange cables from joining the endpoint regions in the fixed demo.
    warm = cv2.bitwise_and(warm, cv2.bitwise_not(cable_mask))
    warm = _keep_large_components(warm, profile.endpoint_min_area)
    zone_y1 = int(round(height * profile.vertical_zone_fraction[0]))
    zone_y2 = int(round(height * profile.vertical_zone_fraction[1]))
    endpoint_width = int(round(width * profile.endpoint_zone_fraction))
    left_end = min(width, endpoint_width)
    right_start = max(0, width - endpoint_width)
    left_component = _mask_component(
        warm[zone_y1:zone_y2, :left_end],
        class_name="left_connector",
        component_id="left_connector:1",
        y_offset=zone_y1,
        min_area=profile.endpoint_min_area,
    )
    right_component = _mask_component(
        warm[zone_y1:zone_y2, right_start:],
        class_name="right_connector",
        component_id="right_connector:1",
        x_offset=right_start,
        y_offset=zone_y1,
        min_area=profile.endpoint_min_area,
    )

    components = [item for item in (left_component, right_component) if item is not None]
    components.extend(cables)
    return tuple(components)


def _checker_for_shape(
    image_shape: tuple[int, int], profile: SplicingConnectorProfile,
) -> RuleBasedAssemblyChecker:
    height, width = image_shape
    left_zone = (0, 0, int(width * profile.endpoint_zone_fraction), height)
    right_zone = (int(width * (1 - profile.endpoint_zone_fraction)), 0, width, height)
    cable_zone = (
        int(width * profile.cable_zone_fraction[0]), 0,
        int(width * profile.cable_zone_fraction[1]), height,
    )
    rules = (
        ComponentRule("left_connector", 1, zone_xyxy=left_zone, min_area=profile.endpoint_min_area),
        ComponentRule("right_connector", 1, zone_xyxy=right_zone, min_area=profile.endpoint_min_area),
        ComponentRule("cable", 1, zone_xyxy=cable_zone, min_area=profile.cable_min_area),
    )
    relations = (
        RelativePositionRule(
            "left_connector_before_right",
            ComponentSelector("left_connector"),
            ComponentSelector("right_connector"),
            "left_of",
        ),
    )
    return RuleBasedAssemblyChecker(rules, relations)


def _cable_endpoint_gap_checks(
    image_bgr: np.ndarray,
    components: tuple[ComponentInstance, ...],
    profile: SplicingConnectorProfile,
) -> tuple[dict[str, object], ...]:
    by_class: dict[str, list[ComponentInstance]] = {}
    for component in components:
        by_class.setdefault(component.class_name, []).append(component)
    if any(len(by_class.get(name, ())) != 1
           for name in ("left_connector", "right_connector", "cable")):
        return ()

    left = by_class["left_connector"][0]
    right = by_class["right_connector"][0]
    cable = by_class["cable"][0]
    image_height, image_width = image_bgr.shape[:2]
    limit_px = image_width * profile.max_cable_endpoint_gap_fraction
    gaps = (
        ("cable_gap_to_left_connector", max(0, cable.bbox_xyxy[0] - left.bbox_xyxy[2])),
        ("cable_gap_to_right_connector", max(0, right.bbox_xyxy[0] - cable.bbox_xyxy[2])),
    )
    checks = [
        {
            "name": name,
            "observed_gap_px": float(gap),
            "maximum_gap_px": float(limit_px),
            "passed": gap <= limit_px,
            "failure_reason": "gap_exceeds_maximum",
        }
        for name, gap in gaps
    ]
    corridor_width = right.bbox_xyxy[0] - left.bbox_xyxy[2]
    coverage = (
        (cable.bbox_xyxy[2] - cable.bbox_xyxy[0]) / corridor_width
        if corridor_width > 0 else 0.0
    )
    checks.append({
        "name": "cable_corridor_coverage",
        "observed_fraction": float(coverage),
        "minimum_fraction": profile.min_cable_corridor_coverage,
        "passed": coverage >= profile.min_cable_corridor_coverage,
        "failure_reason": "coverage_below_minimum",
    })
    endpoint_y_low = min(left.centroid[1], right.centroid[1])
    endpoint_y_high = max(left.centroid[1], right.centroid[1])
    cable_y = cable.centroid[1]
    outside_fraction = max(0.0, endpoint_y_low - cable_y, cable_y - endpoint_y_high) / image_height
    checks.append({
        "name": "cable_vertical_alignment",
        "observed_outside_fraction": float(outside_fraction),
        "maximum_outside_fraction": profile.max_cable_vertical_offset_fraction,
        "passed": outside_fraction <= profile.max_cable_vertical_offset_fraction,
        "failure_reason": "vertical_offset_exceeds_maximum",
    })
    cable_x1, cable_y1, cable_x2, cable_y2 = cable.bbox_xyxy
    cable_hsv = cv2.cvtColor(
        image_bgr[cable_y1:cable_y2, cable_x1:cable_x2],
        cv2.COLOR_BGR2HSV,
    )
    continuity_mask = cv2.inRange(
        cable_hsv,
        (0, profile.min_cable_continuity_saturation, profile.min_cable_continuity_value),
        (179, 255, 255),
    )
    column_counts = np.count_nonzero(continuity_mask, axis=0)
    min_column_pixels = max(
        1,
        int(round((cable_y2 - cable_y1) * profile.cable_column_occupancy_fraction)),
    )
    active_columns = column_counts >= min_column_pixels
    longest_gap = current_gap = 0
    for active in active_columns:
        current_gap = 0 if active else current_gap + 1
        longest_gap = max(longest_gap, current_gap)
    max_internal_gap_px = image_width * profile.max_cable_internal_gap_fraction
    checks.append({
        "name": "cable_internal_continuity",
        "observed_gap_px": float(longest_gap),
        "maximum_gap_px": float(max_internal_gap_px),
        "column_occupancy_threshold_px": min_column_pixels,
        "passed": longest_gap <= max_internal_gap_px,
        "failure_reason": "internal_gap_exceeds_maximum",
    })
    return tuple(checks)


def check_splicing_connectors_image(
    image_bgr: np.ndarray,
    profile: SplicingConnectorProfile = SplicingConnectorProfile(),
    normal_appearance_profile: dict[str, object] | None = None,
) -> SplicingConnectorPrediction:
    """Run the demo extractor and existing typed logical checker on one image."""
    components = extract_splicing_connector_components(image_bgr, profile)
    checker = _checker_for_shape(image_bgr.shape[:2], profile)
    result = checker.check_components(components, image_shape=image_bgr.shape[:2])
    additional_checks = _cable_endpoint_gap_checks(
        image_bgr, components, profile,
    )
    all_reasons = list(result.reasons)
    all_reasons.extend(
        f"{item['name']}:{item['failure_reason']}"
        for item in additional_checks if not item["passed"]
    )
    if additional_checks:
        denominator = len(result.class_checks) + len(result.relations) + len(additional_checks)
        result = replace(
            result,
            passed=not all_reasons,
            score=min(1.0, len(all_reasons) / denominator),
            reasons=tuple(all_reasons),
        )

    appearance_review = None
    if normal_appearance_profile is not None:
        from industrial_inspection.loco_appearance import extract_features, score_profile

        feature_vector = extract_features(image_bgr, components, profile)
        if feature_vector is None:
            appearance_review = {
                "available": False,
                "status": "unavailable",
                "reason": "appearance profile requires exactly one component of each class",
                "ground_truth_used_as_input": False,
            }
        else:
            appearance_review = score_profile(feature_vector, normal_appearance_profile)

    issues = []
    for item in result.class_checks:
        if item.actual_count < item.expected_count:
            issues.append("missing")
        if item.actual_count > item.expected_count:
            issues.append("extra")
        if item.outside_zone_ids:
            issues.append("misplaced")
    if any(not relation.passed for relation in result.relations):
        issues.append("misplaced")
    if any(not item["passed"] for item in additional_checks):
        issues.append("misplaced")
    if result.passed and appearance_review and appearance_review.get("status") == "review":
        issues.append("appearance_outlier")
        status = "review"
    elif result.passed:
        status = "normal"
    elif issues:
        status = "+".join(dict.fromkeys(issues))
    else:
        status = "review"
    return SplicingConnectorPrediction(
        status,
        tuple(dict.fromkeys(issues)),
        components,
        result,
        additional_checks,
        appearance_review,
    )
