"""CPU TorchScript encoder with the unchanged native DINO image/bank/pixel path."""
from __future__ import annotations

import json
from pathlib import Path

from industrial_inspection.models.dino_memory import FrozenFeatureMemory
from industrial_inspection.models.dinov3_runtime import (
    ENCODER_SHA256,
    PREPROCESS,
    DinoV3Config,
    DinoV3MemoryModel,
)


class _GraphEncoder:
    num_prefix_tokens = 5

    def __init__(self, graph):
        self.graph = graph

    def forward_features(self, tensor):
        return self.graph(tensor)


class DinoV3TorchScriptRuntime(DinoV3MemoryModel):
    """Inference-only export; pixel policy remains bound by ModelBundle."""

    def __init__(self, path: Path, *, device="cpu"):
        if device not in {"cpu", "auto"}:
            raise ValueError("This DINO export is qualified for CPU only")
        spec = json.loads((path / "runtime.json").read_text(encoding="utf-8"))
        if (spec.get("format") != "dinov3_torchscript_v1" or spec.get("preprocessing") != PREPROCESS
                or spec.get("encoder_sha256") != ENCODER_SHA256):
            raise ValueError("Unsupported DINO TorchScript descriptor")
        config = dict(spec["config"], encoder_weights="embedded_graph", device="cpu")
        config["image_size"] = tuple(config["image_size"])
        self.config = DinoV3Config(**config)
        if self.config.image_size != (256, 256):
            raise ValueError("DINO export requires fixed 256x256 encoder input")
        self.memory = FrozenFeatureMemory.load(path / "bank")
        if (self.memory.config != self.config.memory_config() or self.memory.memory_bank.shape[1] != 384
                or len(self.memory.memory_bank) > self.config.max_memory_vectors
                or str(self.memory.memory_bank.dtype) != self.config.dtype):
            raise ValueError("Export memory bank differs from declared configuration")
        import torch

        self._torch = torch
        self.device = torch.device("cpu")
        self.path = path.resolve()
        self.graph_path = self.path / "encoder.pt"
        self._graph = torch.jit.load(str(self.graph_path), map_location="cpu").eval()
        self._encoder = _GraphEncoder(self._graph)
        self.model_id = spec["model_id"]
        self.fit_summary = spec["source_fit"]

    @classmethod
    def load(cls, path: Path, *, device="cpu"):
        return cls(path, device=device)

    def fit(self, images):
        raise RuntimeError("DINO TorchScript is inference-only; fit the native model")

    def save(self, path):
        raise RuntimeError("Use the explicit DINO export builder")
