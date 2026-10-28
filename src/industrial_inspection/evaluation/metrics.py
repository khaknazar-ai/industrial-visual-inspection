"""Image, pixel, review, bootstrap, and latency metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from industrial_inspection.evaluation.pixel_metrics import PixelMetricAccumulator
from industrial_inspection.scoring.costs import ReviewCosts
from industrial_inspection.scoring.policy import DecisionThresholds, InspectionDecision, decide
from industrial_inspection.scoring.validation import binary_inputs as _binary_inputs


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def image_level_metrics(
    labels: Iterable[int], scores: Iterable[float], threshold: float
) -> dict[str, Any]:
    y_true, y_score = _binary_inputs(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    y_pred = (y_score >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    unique = np.unique(y_true)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(unique) == 2 else None,
        "average_precision": float(average_precision_score(y_true, y_score)) if 1 in unique else None,
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "false_negative_rate": _safe_divide(fn, fn + tp),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": float(threshold),
        "sample_count": int(len(y_true)),
    }


def pixel_level_metrics(
    masks: Iterable[np.ndarray], anomaly_maps: Iterable[np.ndarray], pixel_threshold: float,
    **budget: int,
) -> dict[str, Any]:
    with PixelMetricAccumulator(pixel_threshold, **budget) as accumulator:
        for mask, anomaly_map in zip(masks, anomaly_maps, strict=True):
            accumulator.update(mask, anomaly_map)
        return accumulator.finalize()


def review_system_metrics(
    labels: Iterable[int], scores: Iterable[float], thresholds: DecisionThresholds,
    *, costs: ReviewCosts | None = None,
) -> dict[str, Any]:
    y_true, y_score = _binary_inputs(labels, scores, allow_nonfinite=True)
    decisions = [decide(float(score), thresholds) for score in y_score]
    auto = np.array([decision != InspectionDecision.REVIEW for decision in decisions])
    predicted = np.array([decision == InspectionDecision.FAIL for decision in decisions], dtype=np.int8)
    correct_auto = int(np.sum((predicted == y_true) & auto))
    auto_count = int(auto.sum())
    positives_auto = (y_true == 1) & auto
    negatives_auto = (y_true == 0) & auto
    result = {
        "automation_rate": _safe_divide(auto_count, len(y_true)),
        "review_rate": _safe_divide(decisions.count(InspectionDecision.REVIEW), len(y_true)),
        "fail_rate": _safe_divide(decisions.count(InspectionDecision.FAIL), len(y_true)),
        "accuracy_auto_decided": _safe_divide(correct_auto, auto_count),
        "false_negative_rate_after_abstention": _safe_divide(
            int(np.sum((predicted == 0) & positives_auto)), int(positives_auto.sum())
        ),
        "false_positive_rate_after_abstention": _safe_divide(
            int(np.sum((predicted == 1) & negatives_auto)), int(negatives_auto.sum())
        ),
        "auto_decided_count": auto_count,
        "review_count": decisions.count(InspectionDecision.REVIEW),
        "sample_count": int(len(y_true)),
        "normal_count": int(np.sum(y_true == 0)),
        "anomaly_count": int(np.sum(y_true == 1)),
        "nonfinite_review_count": int(np.sum(~np.isfinite(y_score))),
        "auto_true_positive_count": int(np.sum((predicted == 1) & positives_auto)),
        "auto_true_negative_count": int(np.sum((predicted == 0) & negatives_auto)),
        "auto_false_negative_count": int(np.sum((predicted == 0) & positives_auto)),
        "auto_false_positive_count": int(np.sum((predicted == 1) & negatives_auto)),
        "review_anomaly_count": int(np.sum((y_true == 1) & ~auto)),
        "review_normal_count": int(np.sum((y_true == 0) & ~auto)),
    }
    result["cost_model"] = costs.as_dict() if costs is not None else None
    result["cost_scope"] = "Automatic mistakes and review fees only; reviewer outcomes are unmeasured."
    result["expected_cost_per_image"] = None
    if costs is not None:
        # Fractions avoid overflowing a sum of per-image costs on large datasets.
        result["expected_cost_per_image"] = (
            costs.false_negative * (result["auto_false_negative_count"] / len(y_true))
            + costs.false_positive * (result["auto_false_positive_count"] / len(y_true))
            + costs.review * (result["review_count"] / len(y_true))
        )
    return result


def coverage_risk_curve(
    labels: Iterable[int],
    scores: Iterable[float],
    center: float,
    *,
    points: int = 11,
    costs: ReviewCosts | None = None,
) -> list[dict[str, float | None]]:
    """Measure automation coverage and auto-decision risk as the review band widens."""

    y_true, y_score = _binary_inputs(labels, scores)
    if type(points) is not int or points < 2:
        raise ValueError("points must be an integer of at least 2")
    if not np.isfinite(center):
        raise ValueError("center must be finite")
    half_span = max(float(y_score.max()) / 2 - float(y_score.min()) / 2, 5e-7)
    largest = np.finfo(np.float64).max
    curve: list[dict[str, float | None]] = []
    for width in np.linspace(0.0, half_span, points):
        half_width = max(float(width), 1e-9)
        lower = max(float(center) - half_width, -largest)
        upper = min(float(center) + half_width, largest)
        if lower == upper:
            with np.errstate(over="ignore"):
                lower = max(float(np.nextafter(center, -np.inf)), -largest)
                upper = min(float(np.nextafter(center, np.inf)), largest)
        thresholds = DecisionThresholds(lower, upper, source="coverage_curve")
        review = review_system_metrics(y_true, y_score, thresholds, costs=costs)
        curve.append(
            {
                "band_half_width": half_width,
                "pass_threshold": thresholds.pass_threshold,
                "fail_threshold": thresholds.fail_threshold,
                "coverage": float(review["automation_rate"]),
                "review_rate": float(review["review_rate"]),
                "expected_cost_per_image": review["expected_cost_per_image"],
                "auto_decision_risk": (
                    float(1.0 - review["accuracy_auto_decided"])
                    if review["accuracy_auto_decided"] is not None
                    else None
                ),
                "false_negative_rate_after_abstention": review["false_negative_rate_after_abstention"],
                "false_positive_rate_after_abstention": review["false_positive_rate_after_abstention"],
            }
        )
    return curve


def bootstrap_confidence_interval(
    labels: Iterable[int],
    scores: Iterable[float],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    samples: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int | None]:
    y_true, y_score = _binary_inputs(labels, scores)
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and in (0, 1)")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        value = float(metric(y_true[indices], y_score[indices]))
        if not np.isfinite(value):
            raise ValueError("bootstrap metric returned a non-finite value")
        values.append(value)
    if not values:
        return {"lower": None, "upper": None, "valid_resamples": 0}
    alpha = (1.0 - confidence) / 2.0
    return {
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
        "valid_resamples": len(values),
    }


def latency_summary(milliseconds: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(milliseconds), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("At least one finite, non-negative latency value is required")
    return {
        "runs": int(len(values)),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }
