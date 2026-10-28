"""Small frozen-CNN memory bank used for the educational P08 comparison."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.models.base import AnomalyPrediction, ImageInput, load_color_image
from industrial_inspection.models.feature_memory import FeatureMoments, FeatureReservoir
from industrial_inspection.scoring.pixels import finalize_prediction


@dataclass(frozen=True, slots=True)
class DeepPatchMemoryConfig:
    image_size: tuple[int, int] = (256, 256)
    backbone: str = "resnet18"
    layers: tuple[str, ...] = ("layer2", "layer3")
    coreset_ratio: float = 0.1
    max_memory_vectors: int = 10000
    query_chunk_size: int = 2048
    memory_chunk_size: int = 4096
    fit_chunk_size: int = 2048
    dtype: str = "float32"
    seed: int = 42
    device: str = "auto"
    pretrained: bool = True
    encoder_weights: str | None = None

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or min(self.image_size) < 32:
            raise ValueError("image_size must contain two dimensions of at least 32")
        if not self.layers:
            raise ValueError("at least one encoder layer is required")
        if not 0 < self.coreset_ratio <= 1:
            raise ValueError("coreset_ratio must be in (0, 1]")
        if any(type(value) is not int or value < 1 for value in (
            self.max_memory_vectors, self.query_chunk_size, self.memory_chunk_size, self.fit_chunk_size,
        )):
            raise ValueError("memory and chunk sizes must be positive integers")
        if self.dtype not in {"float32", "float16"}:
            raise ValueError("dtype must be one of: float32, float16")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torchvision import models
    except (ImportError, OSError) as exc:
        raise RuntimeError("DeepPatchMemoryModel requires torch and torchvision") from exc
    return torch, models


class DeepPatchMemoryModel:
    """Frozen CNN features, random coreset, and chunked nearest-neighbour scoring."""

    def __init__(
        self,
        config: DeepPatchMemoryConfig | None = None,
        model_id: str = "deep-patch-memory-v1",
    ) -> None:
        self.config = config or DeepPatchMemoryConfig()
        self.model_id = model_id
        if self.config.pretrained and (
            self.config.encoder_weights is None or not Path(self.config.encoder_weights).is_file()
        ):
            raise FileNotFoundError("Pretrained Deep Patch Memory requires local encoder_weights; implicit downloads are disabled")
        torch, models = _require_torch()
        self._torch = torch
        self.device = self._resolve_device(torch)
        self._encoder = self._build_encoder(models)
        if self.config.dtype == "float16" and self.device.type == "cuda":
            self._encoder = self._encoder.half()
        self._activations: dict[str, Any] = {}
        self._hooks = []
        for name in self.config.layers:
            layer = getattr(self._encoder, name, None)
            if layer is None:
                raise ValueError(f"Unsupported backbone layer: {name}")
            self._hooks.append(layer.register_forward_hook(self._capture(name)))
        self._encoder.to(self.device).eval()
        for parameter in self._encoder.parameters():
            parameter.requires_grad_(False)
        self.memory_bank: Any | None = None
        self.feature_mean: Any | None = None
        self.feature_std: Any | None = None

    def _resolve_device(self, torch: Any) -> Any:
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for DeepPatchMemoryModel but it is unavailable")
        name = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else self.config.device
        return torch.device(name if name != "auto" else "cpu")

    def _build_encoder(self, models: Any) -> Any:
        constructors = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "wide_resnet50_2": (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.DEFAULT),
        }
        if self.config.backbone not in constructors:
            raise ValueError(f"Unsupported backbone: {self.config.backbone}")
        constructor, _ = constructors[self.config.backbone]
        encoder = constructor(weights=None)
        self.encoder_sha256 = None
        if self.config.pretrained:
            path = Path(self.config.encoder_weights)
            before = sha256_file(path)
            state = self._torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or any(not self._torch.is_tensor(value) or not self._torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Encoder must be a finite tensor state_dict")
            encoder.load_state_dict(state, strict=True)
            if sha256_file(path) != before:
                raise ValueError("Encoder weights changed while loading")
            self.encoder_sha256 = before
        return encoder

    def _preprocess(self, image: ImageInput) -> tuple[Any, tuple[int, int]]:
        torch = self._torch
        decoded = load_color_image(image)
        original_height, original_width = decoded.shape[:2]
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div(255.0)
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=self.config.image_size, mode="bilinear", align_corners=False
        )
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).to(self.device)
        if self.config.dtype == "float16" and self.device.type == "cuda":
            tensor = tensor.half()
        return tensor, (original_height, original_width)

    def _capture(self, name: str) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self._activations[name] = output

        return hook

    def _extract(self, image: ImageInput) -> tuple[Any, tuple[int, int], tuple[int, int]]:
        tensor, original_size = self._preprocess(image)
        self._activations.clear()
        with self._torch.inference_mode():
            self._encoder(tensor)
        outputs = [self._activations[name] for name in self.config.layers]
        target_size = outputs[0].shape[-2:]
        resized = [outputs[0]]
        resized.extend(
            self._torch.nn.functional.interpolate(
                output, size=target_size, mode="bilinear", align_corners=False
            )
            for output in outputs[1:]
        )
        features = self._torch.cat(resized, dim=1)
        features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        return features, original_size, (int(target_size[0]), int(target_size[1]))

    def fit(self, images: Sequence[ImageInput]) -> None:
        if not images:
            raise ValueError("At least one normal image is required")
        reservoir = FeatureReservoir(self.config.max_memory_vectors, self.config.seed)
        moments = FeatureMoments()
        for image in images:
            features = self._extract(image)[0].detach().float().cpu().numpy()
            for start in range(0, len(features), self.config.fit_chunk_size):
                chunk = features[start : start + self.config.fit_chunk_size]
                reservoir.add(chunk)
                moments.update(chunk)
        mean, std = moments.statistics()
        target = min(
            self.config.max_memory_vectors,
            max(1, int(round(reservoir.count * self.config.coreset_ratio))),
        )
        self.feature_mean = self._torch.from_numpy(mean).to(self.device)
        self.feature_std = self._torch.from_numpy(std).to(self.device)
        bank = (reservoir.sample(target) - mean) / std
        if not np.isfinite(bank).all():
            raise ValueError("Normalized memory bank must be finite")
        # Keep normalization and NN state float32 even when the encoder uses half.
        self.memory_bank = self._torch.from_numpy(bank).to(self.device)
        self.fit_summary = {"sampling": "priority_reservoir_v1", "statistics": "streaming_float64_sample_variance",
                            "feature_count": reservoir.count, "bank_vectors": target,
                            "capacity": self.config.max_memory_vectors, "chunk_rows": self.config.fit_chunk_size}

    def _nearest_distances(self, queries: Any) -> Any:
        if self.memory_bank is None:
            raise RuntimeError("Model must be fitted or loaded before prediction")
        distances: list[Any] = []
        memory = self.memory_bank
        if queries.ndim != 2 or memory.ndim != 2 or queries.shape[1] != memory.shape[1] or not len(memory):
            raise ValueError("Query and memory feature dimensions must match")
        for start in range(0, len(queries), self.config.query_chunk_size):
            query = queries[start : start + self.config.query_chunk_size].float()
            best = self._torch.full((len(query),), float("inf"), device=self.device)
            for memory_start in range(0, len(memory), self.config.memory_chunk_size):
                memory_chunk = memory[memory_start : memory_start + self.config.memory_chunk_size].float()
                # Direct L2 avoids cancellation of large, nearly equal squared norms.
                distance = self._torch.cdist(
                    query, memory_chunk, p=2, compute_mode="donot_use_mm_for_euclid_dist"
                )
                best = self._torch.minimum(best, distance.min(dim=1).values)
            distances.append(best)
        return self._torch.cat(distances) if distances else self._torch.empty(0, device=self.device)

    def predict(self, image: ImageInput) -> AnomalyPrediction:
        if self.memory_bank is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Model must be fitted or loaded before prediction")
        started = time.perf_counter()
        features, original_size, grid_shape = self._extract(image)
        normalized = (features.float() - self.feature_mean.float()) / self.feature_std.float()
        distances = self._nearest_distances(normalized)
        grid = distances.reshape(*grid_shape)
        anomaly_map = grid.detach().cpu().numpy().astype(np.float32)
        anomaly_map = cv2.resize(anomaly_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)
        top_k = max(1, len(distances) // 100)
        score = float(self._torch.topk(distances, top_k).values.mean().item())
        return finalize_prediction(self, AnomalyPrediction(
            image_score=score,
            anomaly_map=anomaly_map,
            binary_mask=None,
            metadata={
                "model_family": "deep_patch_memory",
                "backbone": self.config.backbone,
                "dtype": self.config.dtype,
                "memory_vectors": len(self.memory_bank),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        ), original_size)

    def predict_batch(self, images: Sequence[ImageInput]) -> list[AnomalyPrediction]:
        return [self.predict(image) for image in images]

    def save(self, path: Path) -> None:
        if self.memory_bank is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Cannot save an unfitted model")
        path.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                "encoder": self._encoder.state_dict(),
                "memory_bank": self.memory_bank.detach().cpu(),
                "feature_mean": self.feature_mean.detach().cpu(),
                "feature_std": self.feature_std.detach().cpu(),
            },
            path / "model.pt",
        )
        (path / "model.json").write_text(
            json.dumps(
                {"model_id": self.model_id, "config": asdict(self.config), "format_version": 2,
                 "fit_protocol": getattr(self, "fit_summary", {"sampling": "legacy_native"})},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, *, device: str = "auto") -> DeepPatchMemoryModel:
        metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") not in {1, 2}:
            raise ValueError("Unsupported Deep Patch Memory format version")
        config = dict(metadata["config"])
        config["image_size"] = tuple(config["image_size"])
        config["layers"] = tuple(config["layers"])
        config["pretrained"] = False
        config["encoder_weights"] = None
        config["device"] = device
        model = cls(DeepPatchMemoryConfig(**config), metadata["model_id"])
        state = model._torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model._encoder.load_state_dict(state["encoder"])
        model._encoder.to(model.device).eval()
        model.memory_bank = state["memory_bank"].to(device=model.device, dtype=model._torch.float32)
        model.feature_mean = state["feature_mean"].to(device=model.device, dtype=model._torch.float32)
        model.feature_std = state["feature_std"].to(device=model.device, dtype=model._torch.float32)
        bank, mean, std = model.memory_bank, model.feature_mean, model.feature_std
        if (bank.ndim != 2 or not bank.shape[0] or not bank.shape[1]
                or mean.shape != (1, bank.shape[1]) or std.shape != mean.shape
                or not all(model._torch.isfinite(value).all() for value in (bank, mean, std))
                or not (std > 0).all()):
            raise ValueError("Invalid saved memory bank or normalization state")
        model.fit_summary = metadata.get("fit_protocol", {"sampling": "legacy_full_concatenation"})
        return model
