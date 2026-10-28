"""Calibration diagnostics for anomaly scores (not arbitrary confidence labels)."""

from __future__ import annotations

import numpy as np

from industrial_inspection.scoring.validation import binary_inputs, quantile


def expected_calibration_error(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Compute ECE when scores have been converted to calibrated probabilities."""

    rows = reliability_bins(labels, probabilities, bins=bins)
    count = sum(row["count"] for row in rows)
    return sum(row["count"] / count * abs(row["mean_probability"] - row["empirical_rate"])
               for row in rows if row["count"])


def reliability_bins(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    *,
    bins: int = 10,
) -> list[dict[str, float | int | None]]:
    y_true, values = binary_inputs(labels, probabilities)
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    output: list[dict[str, float | int | None]] = []
    for index in range(bins):
        selected = (values >= edges[index]) & (
            values <= edges[index + 1] if index == bins - 1 else values < edges[index + 1]
        )
        output.append(
            {
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "count": int(selected.sum()),
                "mean_probability": float(values[selected].mean()) if selected.any() else None,
                "empirical_rate": float(y_true[selected].mean()) if selected.any() else None,
            }
        )
    return output


def validation_quantile_calibration(scores: list[float] | np.ndarray) -> dict[str, float]:
    """Return auditable robust quantiles for an uncalibrated anomaly score."""

    return {
        "q50": quantile(scores, 0.50), "q90": quantile(scores, 0.90),
        "q95": quantile(scores, 0.95), "q99": quantile(scores, 0.99),
    }
