"""Bounded feature sampling and streaming moments for normal-only bank fitting."""

from __future__ import annotations

import numpy as np


class FeatureReservoir:
    """Keep the smallest seeded random priorities, independent of batch boundaries.

    A caller supplies bounded chunks. Live storage is O(capacity + chunk rows),
    unlike retaining every training feature before selecting the final bank.
    """

    def __init__(self, capacity: int, seed: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Reservoir capacity must be a positive integer")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self._values: np.ndarray | None = None
        self._priorities = np.empty(0, dtype=np.float64)

    def add(self, features: np.ndarray) -> None:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError("Features must be finite [patches, dimensions]")
        if self._values is not None and values.shape[1] != self._values.shape[1]:
            raise ValueError("Feature dimensions changed during fit")
        if not len(values):
            return
        priorities = self.rng.random(len(values))
        self.count += len(values)
        if self._values is not None:
            values = np.concatenate((self._values, values))
            priorities = np.concatenate((self._priorities, priorities))
        count = min(self.capacity, len(values))
        selected = np.argpartition(priorities, count - 1)[:count]
        self._values = values[selected].copy()
        self._priorities = priorities[selected]

    def sample(self, count: int) -> np.ndarray:
        if self._values is None or not 1 <= count <= len(self._values):
            raise ValueError("Requested reservoir sample is unavailable")
        order = np.argsort(self._priorities, kind="stable")[:count]
        return self._values[order].copy()


class FeatureMoments:
    """Merge per-chunk means and centered sums in float64 (sample variance)."""

    def __init__(self):
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, features: np.ndarray) -> None:
        if not len(features):
            return
        chunk = np.asarray(features, dtype=np.float64)
        mean = chunk.mean(axis=0, keepdims=True)
        m2 = chunk.var(axis=0, keepdims=True) * len(chunk)
        if self.mean is None:
            self.mean, self.m2 = mean, m2
        else:
            delta = mean - self.mean
            total = self.count + len(chunk)
            self.m2 += m2 + delta * delta * (self.count * len(chunk) / total)
            self.mean += delta * (len(chunk) / total)
        self.count += len(chunk)

    def statistics(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None:
            raise ValueError("At least one feature vector is required")
        variance = self.m2 / max(1, self.count - 1)
        mean = self.mean.astype(np.float32)
        std = np.maximum(np.sqrt(variance), 1e-6).astype(np.float32)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Feature statistics must be finite")
        return mean, std
