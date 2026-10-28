"""Score fusion for independent classical, anomaly, and known-defect branches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ScoreComponent:
    name: str
    score: float
    weight: float = 1.0


def fuse_scores(
    components: list[ScoreComponent],
    *,
    mode: str = "weighted_mean",
) -> tuple[float, dict[str, float]]:
    """Fuse normalized branch scores and return contribution metadata."""

    if not components:
        raise ValueError("At least one score component is required")
    if any(not np.isfinite(item.score) or not 0 <= item.score <= 1 for item in components):
        raise ValueError("All branch scores must be finite and normalized to [0, 1]")
    if any(not np.isfinite(item.weight) or item.weight < 0 for item in components):
        raise ValueError("Component weights must be finite and non-negative")
    if len({item.name for item in components}) != len(components):
        raise ValueError("Component names must be unique")
    if mode == "max":
        score = max(item.score for item in components)
    elif mode == "weighted_mean":
        scale = max(item.weight for item in components)
        if scale <= 0:
            raise ValueError("At least one component weight must be positive")
        total_weight = sum(item.weight / scale for item in components)
        score = sum(item.score * (item.weight / scale) for item in components) / total_weight
    else:
        raise ValueError(f"Unknown score fusion mode: {mode}")
    contributions = {item.name: float(item.score * item.weight) for item in components}
    return float(score), contributions
