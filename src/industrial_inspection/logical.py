"""Class-aware count, zone and centroid-relation checks in original coordinates.

Masks are component-class segmentations from a caller, not anomaly ground truth.
This module does not provide an image detector. Mask-generated component IDs are
local geometry-order IDs, not tracked physical identities. A real extractor can
instead supply its own class-scoped IDs through ``check_components``.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Literal

import cv2
import numpy as np


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _rectangle(value: Any) -> tuple[int, int, int, int]:
    if (not isinstance(value, (tuple, list)) or len(value) != 4
            or any(type(item) is not int for item in value)
            or not 0 <= value[0] < value[2] or not 0 <= value[1] < value[3]):
        raise ValueError("Expected a nonempty non-negative integer rectangle")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class ComponentRule:
    name: str
    expected_count: int
    zone_xyxy: tuple[int, int, int, int] | None = None
    min_area: int = 10

    def __post_init__(self):
        if not _text(self.name) or type(self.expected_count) is not int or self.expected_count < 0:
            raise ValueError("A component rule needs a class name and a non-negative integer count")
        if type(self.min_area) is not int or self.min_area < 1:
            raise ValueError("min_area must be a positive integer")
        if self.zone_xyxy is not None:
            object.__setattr__(self, "zone_xyxy", _rectangle(self.zone_xyxy))


@dataclass(frozen=True, slots=True)
class ComponentInstance:
    component_id: str
    class_name: str
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    centroid: tuple[float, float]
    id_source: Literal["provided", "mask_geometry_order"] = "provided"

    def __post_init__(self):
        if not _text(self.component_id) or not _text(self.class_name):
            raise ValueError("Component IDs and class names must be nonempty text")
        box = _rectangle(self.bbox_xyxy)
        object.__setattr__(self, "bbox_xyxy", box)
        if (type(self.area) is not int or not 0 < self.area <= (box[2] - box[0]) * (box[3] - box[1])):
            raise ValueError("Component area must be positive and fit its bounding box")
        if (not isinstance(self.centroid, (tuple, list)) or len(self.centroid) != 2
                or any(type(v) not in (int, float) or not isfinite(v) for v in self.centroid)
                or not box[0] <= self.centroid[0] < box[2] or not box[1] <= self.centroid[1] < box[3]):
            raise ValueError("Component centroid must be finite and inside its original bounding box")
        object.__setattr__(self, "centroid", tuple(float(v) for v in self.centroid))
        if self.id_source not in {"provided", "mask_geometry_order"}:
            raise ValueError("Unknown component ID source")

    def as_dict(self) -> dict[str, Any]:
        return {"component_id": self.component_id, "class_name": self.class_name,
                "bbox_xyxy": list(self.bbox_xyxy), "area": self.area,
                "centroid": list(self.centroid), "id_source": self.id_source}


@dataclass(frozen=True, slots=True)
class ComponentSelector:
    class_name: str
    component_id: str | None = None

    def __post_init__(self):
        if not _text(self.class_name) or (self.component_id is not None and not _text(self.component_id)):
            raise ValueError("Selectors require a class and an optional nonempty component ID")


@dataclass(frozen=True, slots=True)
class RelativePositionRule:
    name: str
    subject: ComponentSelector
    reference: ComponentSelector
    relation: Literal["left_of", "right_of", "above", "below"]
    min_gap_px: float = 0.0

    def __post_init__(self):
        if not _text(self.name) or not all(isinstance(v, ComponentSelector) for v in (self.subject, self.reference)):
            raise ValueError("A relation requires a name and typed component selectors")
        if self.relation not in {"left_of", "right_of", "above", "below"}:
            raise ValueError("Unsupported relative-position relation")
        if type(self.min_gap_px) not in (int, float) or not isfinite(self.min_gap_px) or self.min_gap_px < 0:
            raise ValueError("min_gap_px must be finite and non-negative")
        if self.subject == self.reference and self.subject.component_id is not None:
            raise ValueError("A relation cannot explicitly compare a component with itself")


@dataclass(frozen=True, slots=True)
class ClassCheck:
    class_name: str
    expected_count: int
    actual_count: int
    min_area: int
    component_ids: tuple[str, ...]
    zone_xyxy: tuple[int, int, int, int] | None
    outside_zone_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RelationCheck:
    rule: RelativePositionRule
    passed: bool
    issues: tuple[str, ...]
    subject_candidates: tuple[str, ...]
    reference_candidates: tuple[str, ...]
    observed_gap_px: float | None


@dataclass(frozen=True, slots=True)
class LogicalCheck:
    passed: bool
    score: float
    reasons: tuple[str, ...]
    components: tuple[dict[str, Any], ...]
    class_checks: tuple[ClassCheck, ...] = ()
    relations: tuple[RelationCheck, ...] = ()
    image_shape: tuple[int, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Separate logical evidence; the bounded heuristic is not a probability."""
        return {"passed": self.passed, "score": self.score,
                "score_space": "bounded_rule_violation_score", "calibration_status": "uncalibrated",
                "geometry": "original_coordinates", "image_shape": list(self.image_shape) if self.image_shape else None,
                "reasons": list(self.reasons), "components": deepcopy(list(self.components)),
                "unruled_classes": sorted({c["class_name"] for c in self.components}
                                           - {c.class_name for c in self.class_checks}),
                "class_checks": [asdict(item) for item in self.class_checks],
                "relations": [asdict(item) for item in self.relations]}


class RuleBasedAssemblyChecker:
    """Check a frozen component specification; never choose a matching silently."""

    def __init__(self, rules: tuple[ComponentRule, ...],
                 relations: tuple[RelativePositionRule, ...] = ()) -> None:
        self.rules = tuple(rules)
        self.relations = tuple(relations)
        if not self.rules or not all(isinstance(r, ComponentRule) for r in self.rules):
            raise ValueError("At least one component rule is required")
        names = [r.name for r in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("Component class rules must be unique")
        if not all(isinstance(r, RelativePositionRule) for r in self.relations):
            raise ValueError("Relations must be typed relative-position rules")
        if len({r.name for r in self.relations}) != len(self.relations):
            raise ValueError("Relation names must be unique")
        if any(s.class_name not in names for r in self.relations for s in (r.subject, r.reference)):
            raise ValueError("Relation selectors must refer to declared component classes")

    def check(self, mask: np.ndarray | Mapping[str, np.ndarray]) -> LogicalCheck:
        """Check named component masks; a single array supports one component class."""
        names = {rule.name for rule in self.rules}
        if isinstance(mask, np.ndarray):
            if len(names) != 1:
                raise ValueError("Multiple component classes require a mapping of class masks")
            masks = {self.rules[0].name: mask}
        elif isinstance(mask, Mapping):
            masks = dict(mask)
        else:
            raise ValueError("Expected an array or a mapping of component class masks")
        if not masks:
            raise ValueError("At least one component mask is required")
        if missing := names.difference(masks):
            raise ValueError(f"Missing component class masks: {sorted(missing)}")
        if not all(_text(name) for name in masks):
            raise ValueError("Mask class names must be nonempty text")
        components: list[ComponentInstance] = []
        shape = None
        for name in sorted(masks):
            class_mask = masks[name]
            if (not isinstance(class_mask, np.ndarray) or class_mask.ndim != 2 or not class_mask.size
                    or class_mask.dtype.kind not in "buif" or not np.isfinite(class_mask).all()
                    or (class_mask < 0).any()):
                raise ValueError("Component masks must be nonempty finite non-negative numeric 2D arrays")
            if shape is not None and class_mask.shape != shape:
                raise ValueError("Component masks must share the same image geometry")
            shape = class_mask.shape
            count, _, stats, centroids = cv2.connectedComponentsWithStats(
                (class_mask > 0).astype(np.uint8), connectivity=8
            )
            found = []
            for index in range(1, count):
                x, y, width, height, area = (int(value) for value in stats[index])
                found.append(((x, y, x + width, y + height), area, tuple(centroids[index].tolist())))
            found.sort(key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2]))
            for index, (box, area, centroid) in enumerate(found, 1):
                components.append(ComponentInstance(f"{name}:{index}", name, box, area, centroid, "mask_geometry_order"))
        assert shape is not None
        return self.check_components(tuple(components), image_shape=shape)

    def check_components(self, components: tuple[ComponentInstance, ...], *,
                         image_shape: tuple[int, int]) -> LogicalCheck:
        """Consume typed detections in original geometry; IDs are scoped by class.

        Class-only selectors require exactly one eligible component. Multiple
        matches are an explicit failure, not an arbitrary index/nearest choice.
        Named relations require a strictly positive gap beyond min_gap_px.
        """
        if (not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2
                or any(type(v) is not int or v <= 0 for v in image_shape)):
            raise ValueError("image_shape must be positive original (height, width)")
        image_shape = tuple(image_shape)
        components = tuple(components)
        if not all(isinstance(item, ComponentInstance) for item in components):
            raise ValueError("Expected typed component instances")
        keys = [(item.class_name, item.component_id) for item in components]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate component ID within a class")
        if any(item.bbox_xyxy[2] > image_shape[1] or item.bbox_xyxy[3] > image_shape[0] for item in components):
            raise ValueError("Component bounding box exceeds original image geometry")
        components = tuple(sorted(components, key=lambda c: (c.class_name, c.bbox_xyxy[1], c.bbox_xyxy[0], c.component_id)))
        reasons: list[str] = []
        class_checks = []
        eligible = {}
        for rule in self.rules:
            matching = [item for item in components if item.class_name == rule.name and item.area >= rule.min_area]
            eligible[rule.name] = matching
            if len(matching) != rule.expected_count:
                reasons.append(f"{rule.name}:expected_{rule.expected_count}_got_{len(matching)}")
            outside = []
            if rule.zone_xyxy:
                x1, y1, x2, y2 = rule.zone_xyxy
                outside = [
                    item for item in matching if not (x1 <= item.centroid[0] <= x2 and y1 <= item.centroid[1] <= y2)
                ]
                if outside:
                    reasons.append(f"{rule.name}:component_outside_expected_zone")
            class_checks.append(ClassCheck(rule.name, rule.expected_count, len(matching), rule.min_area,
                tuple(item.component_id for item in matching), rule.zone_xyxy, tuple(item.component_id for item in outside)))
        relation_checks = []
        for rule in self.relations:
            candidates = []
            issues = []
            for role, selector in (("subject", rule.subject), ("reference", rule.reference)):
                selected = [item for item in eligible[selector.class_name]
                            if selector.component_id is None or item.component_id == selector.component_id]
                candidates.append(selected)
                if not selected:
                    issues.append(f"missing_{role}")
                elif len(selected) > 1:
                    issues.append(f"ambiguous_{role}")
            gap = None
            if not issues:
                subject, reference = candidates[0][0], candidates[1][0]
                if (subject.class_name, subject.component_id) == (reference.class_name, reference.component_id):
                    issues.append("same_component")
                else:
                    sx, sy = subject.centroid
                    rx, ry = reference.centroid
                    gap = {"left_of": rx - sx, "right_of": sx - rx, "above": ry - sy, "below": sy - ry}[rule.relation]
                    if gap <= rule.min_gap_px:
                        issues.append("relation_violated")
            reasons.extend(f"{rule.name}:{issue}" for issue in issues)
            relation_checks.append(RelationCheck(rule, not issues, tuple(issues),
                tuple(item.component_id for item in candidates[0]), tuple(item.component_id for item in candidates[1]), gap))
        score = min(1.0, len(reasons) / (len(self.rules) + len(self.relations)))
        return LogicalCheck(not reasons, score, tuple(reasons), tuple(c.as_dict() for c in components),
                            tuple(class_checks), tuple(relation_checks), image_shape)
