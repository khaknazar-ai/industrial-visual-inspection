"""Frozen-feature memory bank used by a DINO/DINOv3 cloud adapter."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from industrial_inspection.models.feature_memory import FeatureReservoir


@dataclass(frozen=True, slots=True)
class DinoMemoryConfig:
    coreset_ratio: float = 0.1
    max_memory_vectors: int = 100_000
    distance_chunk_size: int = 4096
    memory_chunk_size: int = 4096
    fit_chunk_size: int = 2048
    dtype: str = "float16"
    seed: int = 42

    def __post_init__(self):
        if not np.isfinite(self.coreset_ratio) or not 0 < self.coreset_ratio <= 1:
            raise ValueError("coreset_ratio must be finite and in (0, 1]")
        if any(type(value) is not int or value < 1 for value in (
            self.max_memory_vectors, self.distance_chunk_size, self.memory_chunk_size, self.fit_chunk_size,
        )):
            raise ValueError("Memory and chunk sizes must be positive integers")
        if self.dtype not in {"float16", "float32"}:
            raise ValueError("dtype must be float16 or float32")


def _normalized(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Features must be finite [patches, dimensions]")
    norm = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
    return (values / np.maximum(norm, 1e-8)).astype(np.float32)


class FrozenFeatureMemory:
    """Backbone-agnostic memory bank; the caller supplies a frozen extractor.

    The extractor must return ``[num_patches, feature_dim]`` NumPy-compatible
    arrays. This makes DINOv3 weights and transforms explicit cloud inputs.
    """

    def __init__(self, config: DinoMemoryConfig | None = None, model_id: str = "dino-memory-v1") -> None:
        self.config = config or DinoMemoryConfig()
        self.model_id = model_id
        self.memory_bank: np.ndarray | None = None
        self.sampling = "unfitted"

    def fit_features(self, feature_batches: Iterable[np.ndarray]) -> None:
        reservoir = FeatureReservoir(self.config.max_memory_vectors, self.config.seed)
        for batch in feature_batches:
            features = np.asarray(batch)
            if features.ndim != 2 or not features.shape[1]:
                raise ValueError("Features must be finite [patches, dimensions]")
            for start in range(0, len(features), self.config.fit_chunk_size):
                reservoir.add(_normalized(features[start : start + self.config.fit_chunk_size]))
        if not reservoir.count:
            raise ValueError("At least one feature vector is required")
        count = min(
            self.config.max_memory_vectors,
            max(1, int(round(reservoir.count * self.config.coreset_ratio))),
        )
        dtype = np.float16 if self.config.dtype == "float16" else np.float32
        self.memory_bank = reservoir.sample(count).astype(dtype)
        self.sampling = "priority_reservoir_v1"

    def fit_images(self, images: Iterable[Any], extractor: Callable[[Any], np.ndarray]) -> None:
        self.fit_features(extractor(image) for image in images)

    def score_features(self, features: np.ndarray) -> np.ndarray:
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is not fitted")
        queries = np.asarray(features)
        if queries.ndim != 2 or queries.shape[1] != self.memory_bank.shape[1]:
            raise ValueError("Query and memory feature dimensions must match")
        scores: list[np.ndarray] = []
        for start in range(0, len(queries), self.config.distance_chunk_size):
            chunk = _normalized(queries[start : start + self.config.distance_chunk_size])
            best = np.full(len(chunk), -np.inf, dtype=np.float32)
            for offset in range(0, len(self.memory_bank), self.config.memory_chunk_size):
                # Stored half precision vectors are no longer exactly unit length.
                memory = _normalized(self.memory_bank[offset : offset + self.config.memory_chunk_size])
                best = np.maximum(best, (chunk @ memory.T).max(axis=1))
            scores.append(1.0 - np.clip(best, -1.0, 1.0))
        return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)

    def save(self, path: Path) -> None:
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is not fitted")
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "memory_bank.npz", memory_bank=self.memory_bank)
        (path / "model.json").write_text(
            json.dumps({"model_id": self.model_id, "config": asdict(self.config), "format_version": 2,
                        "sampling": self.sampling}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> FrozenFeatureMemory:
        metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") not in {1, 2}:
            raise ValueError("Unsupported feature memory format version")
        model = cls(DinoMemoryConfig(**metadata["config"]), metadata["model_id"])
        model.sampling = metadata.get("sampling", "legacy_full_concatenation")
        with np.load(path / "memory_bank.npz", allow_pickle=False) as archive:
            model.memory_bank = archive["memory_bank"]
        if (model.memory_bank.ndim != 2 or not all(model.memory_bank.shape)
                or not np.isfinite(model.memory_bank).all()):
            raise ValueError("Memory bank must be a nonempty finite matrix")
        return model
