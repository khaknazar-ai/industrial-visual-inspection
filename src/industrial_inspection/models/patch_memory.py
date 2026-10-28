"""Educational normal-only patch memory-bank anomaly detector."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from industrial_inspection.models.base import AnomalyPrediction, ImageInput, load_color_image
from industrial_inspection.scoring.pixels import finalize_prediction


@dataclass(frozen=True, slots=True)
class PatchMemoryConfig:
    image_size: tuple[int, int] = (256, 256)
    patch_size: int = 16
    stride: int = 8
    coreset_ratio: float = 0.25
    max_memory_vectors: int = 10000
    distance_chunk_size: int = 1024
    seed: int = 42

    def __post_init__(self) -> None:
        if self.patch_size < 4 or self.stride < 1:
            raise ValueError("patch_size and stride are too small")
        if not 0 < self.coreset_ratio <= 1:
            raise ValueError("coreset_ratio must be in (0, 1]")


class PatchMemoryModel:
    """Nearest-neighbour detector using handcrafted dense patch descriptors.

    This intentionally small implementation demonstrates PatchCore's memory-bank
    mechanics without pretending to reproduce a pretrained deep backbone.
    """

    def __init__(
        self,
        config: PatchMemoryConfig | None = None,
        model_id: str = "educational-patch-memory-v1",
    ) -> None:
        self.config = config or PatchMemoryConfig()
        self.model_id = model_id
        self.memory_bank: np.ndarray | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def _features(self, image: ImageInput) -> tuple[np.ndarray, tuple[int, int]]:
        decoded = load_color_image(image)
        width, height = self.config.image_size
        resized = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        patch = self.config.patch_size
        descriptors: list[np.ndarray] = []
        rows = list(range(0, height - patch + 1, self.config.stride))
        cols = list(range(0, width - patch + 1, self.config.stride))
        for y in rows:
            for x in cols:
                lab_patch = lab[y : y + patch, x : x + patch]
                grad_patch = gradient[y : y + patch, x : x + patch]
                descriptors.append(
                    np.concatenate(
                        [
                            lab_patch.mean(axis=(0, 1)),
                            lab_patch.std(axis=(0, 1)),
                            np.array([grad_patch.mean(), grad_patch.std()], dtype=np.float32),
                        ]
                    )
                )
        return np.stack(descriptors).astype(np.float32), (len(rows), len(cols))

    def fit(self, images: Sequence[ImageInput]) -> None:
        if not images:
            raise ValueError("At least one normal image is required")
        raw = np.concatenate([self._features(image)[0] for image in images], axis=0)
        self.feature_mean = raw.mean(axis=0, keepdims=True)
        self.feature_std = np.maximum(raw.std(axis=0, keepdims=True), 1e-6)
        normalized = (raw - self.feature_mean) / self.feature_std
        target = min(
            self.config.max_memory_vectors,
            max(1, int(round(len(normalized) * self.config.coreset_ratio))),
        )
        rng = np.random.default_rng(self.config.seed)
        indices = np.sort(rng.choice(len(normalized), size=target, replace=False))
        self.memory_bank = normalized[indices].astype(np.float32)

    def _nearest_distances(self, queries: np.ndarray) -> np.ndarray:
        assert self.memory_bank is not None
        chunks: list[np.ndarray] = []
        memory_norm = np.sum(self.memory_bank * self.memory_bank, axis=1)[None, :]
        for start in range(0, len(queries), self.config.distance_chunk_size):
            chunk = queries[start : start + self.config.distance_chunk_size]
            squared = (
                np.sum(chunk * chunk, axis=1)[:, None]
                + memory_norm
                - 2.0 * chunk @ self.memory_bank.T
            )
            chunks.append(np.sqrt(np.maximum(squared.min(axis=1), 0.0)))
        return np.concatenate(chunks)

    def predict(self, image: ImageInput) -> AnomalyPrediction:
        if self.memory_bank is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Model must be fitted or loaded before prediction")
        started = time.perf_counter()
        decoded = load_color_image(image)
        raw, grid = self._features(decoded)
        queries = (raw - self.feature_mean) / self.feature_std
        distances = self._nearest_distances(queries)
        grid_map = distances.reshape(grid)
        height, width = decoded.shape[:2]
        anomaly_map = cv2.resize(grid_map, (width, height), interpolation=cv2.INTER_CUBIC)
        top_k = max(1, len(distances) // 100)
        score = float(np.partition(distances, -top_k)[-top_k:].mean())
        return finalize_prediction(self, AnomalyPrediction(
            image_score=score,
            anomaly_map=anomaly_map.astype(np.float32),
            binary_mask=None,
            metadata={
                "model_family": "educational_patch_memory",
                "memory_vectors": len(self.memory_bank),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        ), decoded.shape[:2])

    def predict_batch(self, images: Sequence[ImageInput]) -> list[AnomalyPrediction]:
        return [self.predict(image) for image in images]

    def save(self, path: Path) -> None:
        if self.memory_bank is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Cannot save an unfitted model")
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "memory_bank.npz",
            memory_bank=self.memory_bank,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )
        (path / "model.json").write_text(
            json.dumps(
                {"model_id": self.model_id, "config": asdict(self.config), "format_version": 1},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> PatchMemoryModel:
        metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
        config = dict(metadata["config"])
        config["image_size"] = tuple(config["image_size"])
        model = cls(PatchMemoryConfig(**config), metadata["model_id"])
        state = np.load(path / "memory_bank.npz", allow_pickle=False)
        model.memory_bank = state["memory_bank"].astype(np.float32)
        model.feature_mean = state["feature_mean"].astype(np.float32)
        model.feature_std = state["feature_std"].astype(np.float32)
        return model
