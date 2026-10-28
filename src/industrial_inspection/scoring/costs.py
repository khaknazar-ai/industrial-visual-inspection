"""Explicit costs for automatic mistakes and sending an image to a reviewer."""

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class ReviewCosts:
    false_negative: float
    false_positive: float
    review: float
    unit: str = "relative_cost"

    def __post_init__(self):
        for value in (self.false_negative, self.false_positive, self.review):
            if isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError("Decision costs must be finite, non-negative numbers")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("A cost unit is required")

    def as_dict(self):
        return asdict(self)
