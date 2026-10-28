"""Shared numeric validation for calibration and evaluation."""

from __future__ import annotations

import numpy as np


def finite_scores(scores) -> np.ndarray:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty one-dimensional finite sequence")
    return values


def binary_inputs(labels, scores, *, allow_nonfinite: bool = False):
    labels_array = np.asarray(list(labels))
    scores_array = np.asarray(list(scores), dtype=np.float64)
    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if len(labels_array) != len(scores_array) or not len(labels_array):
        raise ValueError("labels and scores must have the same non-zero length")
    if not np.isin(labels_array, [0, 1]).all():
        raise ValueError("labels must contain only binary values 0 and 1")
    if not allow_nonfinite and not np.isfinite(scores_array).all():
        raise ValueError("scores must be finite")
    return labels_array.astype(np.uint8), scores_array


def quantile(scores, q: float) -> float:
    """Linear order-statistic interpolation without overflow across opposite signs."""
    if isinstance(q, bool) or not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("quantile must be finite and in [0, 1]")
    values = np.sort(finite_scores(scores))
    position = (len(values) - 1) * q
    left, right = float(values[int(np.floor(position))]), float(values[int(np.ceil(position))])
    fraction = position - np.floor(position)
    if left < 0 < right:
        return float(left * (1 - fraction) + right * fraction)
    return float(left + (right - left) * fraction)
