"""Interface for a separate known-defect supervised branch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class KnownDefectDetection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray | None = None


class KnownDefectModel(Protocol):
    model_id: str

    def fit(self, images: Sequence[np.ndarray], labels: Sequence[Any]) -> None: ...

    def predict(self, image: np.ndarray) -> list[KnownDefectDetection]: ...


def separate_known_defect_metrics() -> dict[str, str]:
    """Documentation marker preventing metric mixing with unknown anomalies."""

    return {
        "image_task": "known-defect classification/detection",
        "recommended_metrics": "mAP, per-class precision/recall, mask AP when labels exist",
        "mix_with_unknown_anomaly_metrics": "never",
    }
