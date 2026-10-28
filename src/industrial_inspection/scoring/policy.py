"""Validation-only threshold selection and abstaining decision policy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from industrial_inspection.scoring.validation import binary_inputs, finite_scores, quantile


class InspectionDecision(StrEnum):
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class DecisionThresholds:
    pass_threshold: float
    fail_threshold: float
    source: str = "validation"
    objective: str = "f1"

    def __post_init__(self) -> None:
        if isinstance(self.pass_threshold, bool) or isinstance(self.fail_threshold, bool):
            raise ValueError("Thresholds must be numbers, not booleans")
        if not np.isfinite(self.pass_threshold) or not np.isfinite(self.fail_threshold):
            raise ValueError("Thresholds must be finite")
        if self.pass_threshold >= self.fail_threshold:
            raise ValueError("pass_threshold must be lower than fail_threshold")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide(score: float, thresholds: DecisionThresholds) -> InspectionDecision:
    if not np.isfinite(score):
        return InspectionDecision.REVIEW
    if score < thresholds.pass_threshold:
        return InspectionDecision.PASS
    if score < thresholds.fail_threshold:
        return InspectionDecision.REVIEW
    return InspectionDecision.FAIL


def _binary_counts(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    return tp, fp, tn, fn


def validate_normal_pass_quantile(value: float) -> None:
    if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value < 0.995:
        raise ValueError("normal_pass_quantile must be finite and in [0, 0.995)")


def select_thresholds(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    normal_pass_quantile: float = 0.95,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> tuple[DecisionThresholds, list[dict[str, float]]]:
    """Select fail by F1/cost tie-break and pass from normal validation scores only."""

    y_true, y_score = binary_inputs(labels, scores)
    validate_normal_pass_quantile(normal_pass_quantile)
    for cost in (false_negative_cost, false_positive_cost):
        if isinstance(cost, bool) or not np.isfinite(cost) or cost < 0:
            raise ValueError("Misclassification costs must be finite and non-negative")
    normal_scores = y_score[y_true == 0]
    if len(normal_scores) == 0:
        raise ValueError("At least one normal validation sample is required")
    if not np.any(y_true):
        return normal_only_thresholds(normal_scores, pass_quantile=normal_pass_quantile), []

    unique = np.unique(y_score)
    midpoints = unique[:-1] / 2 + unique[1:] / 2
    midpoints = np.where(midpoints <= unique[:-1], unique[1:], midpoints)
    candidates = np.concatenate(([unique[0]], midpoints))
    with np.errstate(over="ignore"):
        above_max = np.nextafter(unique[-1], np.inf)
    if np.isfinite(above_max):
        candidates = np.append(candidates, above_max)

    curve: list[dict[str, float]] = []
    best_key: tuple[float, float, float] | None = None
    fail_threshold = float(candidates[0])
    for threshold in candidates:
        tp, fp, tn, fn = _binary_counts(y_true, y_score, float(threshold))
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        expected_cost = false_negative_cost * (fn / len(y_true)) + false_positive_cost * (fp / len(y_true))
        specificity = tn / (tn + fp) if tn + fp else 0.0
        curve.append(
            {
                "threshold": float(threshold),
                "f1": float(f1),
                "expected_cost": float(expected_cost),
                "specificity": float(specificity),
            }
        )
        key = (f1, -expected_cost, specificity)
        if best_key is None or key > best_key:
            best_key = key
            fail_threshold = float(threshold)

    pass_threshold = quantile(normal_scores, normal_pass_quantile)
    if pass_threshold >= fail_threshold:
        normal_center = quantile(normal_scores, 0.5)
        pass_threshold = normal_center / 2.0 + fail_threshold / 2.0
    if pass_threshold >= fail_threshold:
        with np.errstate(over="ignore"):
            pass_threshold = float(np.nextafter(fail_threshold, -np.inf))
    if not np.isfinite(pass_threshold):
        raise ValueError("No finite lower review boundary is representable for these scores")
    return DecisionThresholds(pass_threshold, fail_threshold), curve


def normal_only_thresholds(scores, *, pass_quantile: float = 0.95) -> DecisionThresholds:
    if isinstance(pass_quantile, bool) or not np.isfinite(pass_quantile) or not 0 <= pass_quantile < 0.995:
        raise ValueError("pass_quantile must be in [0, 0.995)")
    values = finite_scores(scores)
    passed, failed = quantile(values, pass_quantile), quantile(values, 0.995)
    if failed <= passed:
        with np.errstate(over="ignore"):
            failed = float(np.nextafter(passed, np.inf))
    if not np.isfinite(failed):
        raise ValueError("No finite upper review boundary is representable for these scores")
    return DecisionThresholds(passed, failed, source="validation_normal_quantiles", objective="normal_quantile")


@dataclass(slots=True)
class RobustMinMaxNormalizer:
    """Quantile-based score normalizer fitted on validation data."""

    lower: float | None = None
    upper: float | None = None

    def fit(self, scores: Iterable[float], lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("Expected 0 <= lower_quantile < upper_quantile <= 1")
        values = finite_scores(scores)
        self.lower = quantile(values, lower_quantile)
        self.upper = quantile(values, upper_quantile)

    def transform(self, scores: Iterable[float]) -> np.ndarray:
        if self.lower is None or self.upper is None:
            raise RuntimeError("Normalizer must be fitted")
        values = finite_scores(scores)
        if self.upper == self.lower:
            return (values > self.lower).astype(np.float64)
        values = np.clip(values, self.lower, self.upper)
        if self.lower < 0 < self.upper:
            scale = max(abs(self.lower), abs(self.upper))
            return (values / scale - self.lower / scale) / (self.upper / scale - self.lower / scale)
        return (values - self.lower) / (self.upper - self.lower)
