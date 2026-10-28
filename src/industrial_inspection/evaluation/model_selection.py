"""Configurable model selection under quality and deployment constraints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum, isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class SelectionWeights:
    quality: float = 0.40
    false_negative: float = 0.20
    latency: float = 0.15
    memory: float = 0.10
    model_size: float = 0.05
    robustness: float = 0.10

    def __post_init__(self) -> None:
        values = (self.quality, self.false_negative, self.latency, self.memory, self.model_size, self.robustness)
        if any(not isfinite(value) or value < 0 for value in values) or max(values) <= 0:
            raise ValueError("Selection weights must be finite, non-negative and not all zero")

    def as_dict(self) -> dict[str, float]:
        return {
            "quality": self.quality,
            "false_negative": self.false_negative,
            "latency": self.latency,
            "memory": self.memory,
            "model_size": self.model_size,
            "robustness": self.robustness,
        }

    def normalized(self) -> SelectionWeights:
        scale = max(self.as_dict().values())
        scaled = {name: value / scale for name, value in self.as_dict().items()}
        total = fsum(scaled.values())
        return replace(self, **{name: value / total for name, value in scaled.items()})


def rank_models(
    candidates: list[dict[str, Any]],
    *,
    weights: SelectionWeights | None = None,
) -> list[dict[str, Any]]:
    """Rank candidates whose deployment metrics are already normalized to [0, 1].

    ``quality`` and ``robustness`` are higher-is-better. The remaining dimensions
    are lower-is-better and must be supplied as already inverted scores.
    """

    if not candidates:
        return []
    selected_weights = (weights or SelectionWeights()).normalized()
    dimensions = ("quality", "false_negative", "latency", "memory", "model_size", "robustness")
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        missing = [dimension for dimension in dimensions if dimension not in candidate]
        if missing:
            raise ValueError(f"Candidate is missing normalized dimensions: {missing}")
        values = {dimension: float(candidate[dimension]) for dimension in dimensions}
        if any(not isfinite(value) or value < 0 or value > 1 for value in values.values()):
            raise ValueError("Normalized selection dimensions must be finite and in [0, 1]")
        score = sum(getattr(selected_weights, dimension) * values[dimension] for dimension in dimensions)
        ranked.append(
            {
                **candidate,
                "selection_score": float(score),
                "selection_weights": selected_weights.as_dict(),
            }
        )
    return sorted(ranked, key=lambda item: item["selection_score"], reverse=True)
