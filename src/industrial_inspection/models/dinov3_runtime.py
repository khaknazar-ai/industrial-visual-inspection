"""Explicit local timm DINOv3 Small encoder and bounded normal-feature memory.

This is the timm conversion (zero QKV biases removed, float32 RoPE periods),
not a numerical parity claim for Meta's reference implementation.
"""
from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.models.base import AnomalyPrediction, ImageInput, load_color_image
from industrial_inspection.models.dino_memory import DinoMemoryConfig, FrozenFeatureMemory
from industrial_inspection.scoring.pixels import finalize_prediction

ENCODER_ID = "vit_small_patch16_dinov3.lvd1689m"
ENCODER_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
ENCODER_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
PREPROCESS = {
    "color": "RGB", "range": [0, 1], "resize": "torch_bicubic_full_frame",
    "align_corners": False, "antialias": True, "crop": None,
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "features": "last_block_post_norm_patch_tokens", "prefix_tokens_excluded": 5,
    "feature_normalization": "L2", "encoder_dtype": "float32",
}


@dataclass(frozen=True, slots=True)
class DinoV3Config:
    encoder_weights: str
    image_size: tuple[int, int] = (256, 256)
    coreset_ratio: float = 0.1
    max_memory_vectors: int = 4096
    distance_chunk_size: int = 256
    memory_chunk_size: int = 1024
    fit_chunk_size: int = 256
    dtype: str = "float16"
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self):
        if (len(self.image_size) != 2 or any(type(n) is not int or not 32 <= n <= 512 or n % 16
                                           for n in self.image_size)):
            raise ValueError("DINO image_size must be two multiples of 16 between 32 and 512")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("Unsupported DINO device")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("DINO seed must be a nonnegative integer")
        self.memory_config()
        if self.max_memory_vectors > 100_000 or max(self.distance_chunk_size, self.memory_chunk_size,
                                                   self.fit_chunk_size) > 4096:
            raise ValueError("DINO memory/chunk capacity exceeds bounded runtime limits")

    def memory_config(self):
        return DinoMemoryConfig(**{key: getattr(self, key) for key in DinoMemoryConfig.__dataclass_fields__})


def patch_features(tokens, grid, prefix_tokens):
    """Require the exact CLS + four registers + row-major spatial grid contract."""
    if prefix_tokens != 5 or tuple(tokens.shape) != (1, 5 + grid[0] * grid[1], 384):
        raise ValueError("DINO token/prefix/grid contract differs from the pinned Small encoder")
    patches = tokens[0, prefix_tokens:]
    values = patches.detach().float().cpu().numpy().copy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite DINO patch features")
    return values


def _prepare_native_directory(path):
    """Accept only a fresh target or the common evaluator's empty reservation."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        return
    if set(p.name for p in path.iterdir()) != {"run_status.json"}:
        raise FileExistsError("DINO native output already contains artifacts")
    marker = json.loads((path / "run_status.json").read_text(encoding="utf-8"))
    if marker.get("status") != "in_progress":
        raise FileExistsError("DINO native output is not an active empty reservation")


class DinoV3MemoryModel:
    score_space = "raw_distance"
    encoder_sha256 = ENCODER_SHA256

    def __init__(self, config: DinoV3Config, model_id: str = "dinov3-small-memory-v1"):
        self.config, self.model_id = config, model_id
        self.encoder_path = Path(config.encoder_weights).resolve()
        if not self.encoder_path.is_file():
            raise FileNotFoundError("DINOv3 requires explicit local weights; downloads are disabled")
        if sha256_file(self.encoder_path) != ENCODER_SHA256:
            raise ValueError("DINOv3 encoder checksum differs from the pinned licensed asset")
        import timm
        import torch
        from safetensors.torch import load_file

        from industrial_inspection.models.artifacts import resolve_device

        self._torch = torch
        self.device = torch.device(resolve_device(config.device))
        # pretrained=False forbids network retrieval. No inference is allowed until
        # the entire pinned state has loaded strictly; missing keys are fatal.
        self._encoder = timm.create_model(ENCODER_ID, pretrained=False, num_classes=0)
        state = load_file(str(self.encoder_path), device="cpu")
        if any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError("Non-finite DINO encoder state")
        self._encoder.load_state_dict(state, strict=True)
        if sha256_file(self.encoder_path) != ENCODER_SHA256:
            raise ValueError("DINO encoder changed while loading")
        self._encoder.float().to(self.device).eval().requires_grad_(False)
        if self._encoder.num_prefix_tokens != 5 or tuple(self._encoder.patch_embed.patch_size) != (16, 16):
            raise ValueError("DINO architecture differs from the pinned patch/prefix contract")
        self.memory = FrozenFeatureMemory(config.memory_config(), model_id)
        self.fit_summary = None

    def _extract(self, image: ImageInput):
        torch = self._torch
        decoded = load_color_image(image)
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255)
        # Full-frame resize is explicit: non-square images are not silently cropped.
        tensor = torch.nn.functional.interpolate(tensor, size=self.config.image_size,
            mode="bicubic", align_corners=False, antialias=True)
        mean = torch.tensor(PREPROCESS["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(PREPROCESS["std"]).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).to(self.device)
        grid = tuple(n // 16 for n in self.config.image_size)
        with torch.inference_mode():
            tokens = self._encoder.forward_features(tensor)
        return patch_features(tokens, grid, self._encoder.num_prefix_tokens), decoded.shape[:2]

    def fit(self, images: Sequence[ImageInput]):
        count = 0

        def batches():
            nonlocal count
            for image in images:
                features, _ = self._extract(image)
                count += 1
                yield features

        self.memory.fit_features(batches())
        self.fit_summary = {"images": count, "patches_per_image": int(np.prod(self.config.image_size) // 256),
                            "sampling": self.memory.sampling, "encoder_updated": False,
                            "memory_vectors": len(self.memory.memory_bank)}

    def predict(self, image: ImageInput):
        if self.memory.memory_bank is None:
            raise RuntimeError("DINO normal bank is not fitted")
        features, shape = self._extract(image)
        scores = self.memory.score_features(features)
        grid = tuple(n // 16 for n in self.config.image_size)
        response = cv2.resize(scores.reshape(grid), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return finalize_prediction(self, AnomalyPrediction(float(scores.max()), response, None,
            metadata={"encoder": ENCODER_ID, "encoder_sha256": ENCODER_SHA256,
                      "patch_grid": list(grid), "prefix_tokens_excluded": 5,
                      "memory_vectors": len(self.memory.memory_bank), "score_reduction": "maximum_patch_distance"},
            valid_mask=np.ones(shape, dtype=bool)), shape)

    def predict_batch(self, images: Sequence[ImageInput]):
        return [self.predict(image) for image in images]

    def save(self, path: Path):
        if self.memory.memory_bank is None:
            raise RuntimeError("Cannot save an unfitted DINO bank")
        if sha256_file(self.encoder_path) != ENCODER_SHA256:
            raise ValueError("Encoder source changed before native save")
        for name in ("LICENSE-DINOv3.md", "origin.json"):
            if not (self.encoder_path.parent / name).is_file():
                raise FileNotFoundError(f"DINO native save requires asset notice/provenance: {name}")
        _prepare_native_directory(path)
        # Persist the computed bank first so a later asset-write failure cannot lose it.
        self.memory.save(path / "bank")
        encoder_target = path / "encoder.safetensors"
        try:
            os.link(self.encoder_path, encoder_target)
        except PermissionError:
            # Windows may deny hardlinks without changing file access settings.
            # Preserve exact asset bytes and verify the pinned checksum below.
            shutil.copyfile(self.encoder_path, encoder_target)
        if sha256_file(path / "encoder.safetensors") != ENCODER_SHA256:
            raise ValueError("DINO encoder artifact checksum mismatch")
        for name in ("LICENSE-DINOv3.md", "origin.json"):
            source = self.encoder_path.parent / name
            if not source.is_file():
                raise FileNotFoundError(f"DINO native save requires asset notice/provenance: {source}")
            shutil.copyfile(source, path / name)
        config = asdict(self.config)
        config["encoder_weights"] = "encoder.safetensors"
        metadata = {"format_version": 1, "model_id": self.model_id, "config": config,
                    "encoder_id": ENCODER_ID, "encoder_sha256": ENCODER_SHA256,
                    "encoder_revision": ENCODER_REVISION, "preprocessing": PREPROCESS,
                    "fit_summary": self.fit_summary}
        (path / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, device="cpu"):
        metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if (metadata.get("format_version") != 1 or metadata.get("encoder_id") != ENCODER_ID
                or metadata.get("encoder_sha256") != ENCODER_SHA256
                or metadata.get("encoder_revision") != ENCODER_REVISION
                or metadata.get("preprocessing") != PREPROCESS
                or metadata["config"].get("encoder_weights") != "encoder.safetensors"):
            raise ValueError("Unsupported or changed DINO native identity")
        config = dict(metadata["config"], encoder_weights=str(path / "encoder.safetensors"), device=device)
        config["image_size"] = tuple(config["image_size"])
        memory = FrozenFeatureMemory.load(path / "bank")
        expected = DinoV3Config(**config)
        if (memory.config != expected.memory_config() or memory.model_id != metadata["model_id"]
                or memory.memory_bank.shape[1] != 384
                or len(memory.memory_bank) > expected.max_memory_vectors
                or str(memory.memory_bank.dtype) != expected.dtype):
            raise ValueError("DINO native bank/config mismatch")
        model = cls(expected, metadata["model_id"])
        model.memory, model.fit_summary = memory, metadata["fit_summary"]
        return model
