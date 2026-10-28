"""Common anomaly model interface and canonical prediction types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from industrial_inspection.data.image_io import read_color_image
from industrial_inspection.data.observations import observed_input_scope

ImageInput = Path | str | np.ndarray


@dataclass(frozen=True, slots=True)
class AnomalyRegion:
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    region_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "area": self.area,
            "region_score": round(self.region_score, 6),
        }


@dataclass(slots=True)
class AnomalyPrediction:
    image_score: float | None
    anomaly_map: np.ndarray | None
    binary_mask: np.ndarray | None
    regions: list[AnomalyRegion] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    valid_mask: np.ndarray | None = None


@runtime_checkable
class AnomalyModel(Protocol):
    model_id: str

    def fit(self, images: Sequence[ImageInput]) -> None: ...

    def predict(self, image: ImageInput) -> AnomalyPrediction: ...

    def predict_batch(self, images: Sequence[ImageInput]) -> list[AnomalyPrediction]: ...

    def save(self, path: Path) -> None: ...


def load_color_image(image: ImageInput) -> np.ndarray:
    """Decode a path or validate an already decoded BGR image."""

    scope = observed_input_scope()
    if scope and not isinstance(image, np.ndarray):
        scope[0].verify(image)
    decoded = image if isinstance(image, np.ndarray) else read_color_image(Path(image))
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError("Expected a readable three-channel image")
    if decoded.dtype != np.uint8:
        decoded = np.clip(decoded, 0, 255).astype(np.uint8)
    if scope and not isinstance(image, np.ndarray):
        scope[0].record(image, scope[1])
    return decoded
