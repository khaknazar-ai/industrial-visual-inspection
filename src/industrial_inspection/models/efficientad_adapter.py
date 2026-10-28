"""EfficientAD configuration and validation of local training assets."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.models.benchmark_specs import AnomalibCommand, build_anomalib_command


def normalize_model_size(value: str) -> str:
    normalized = {"s": "small", "m": "medium"}.get(value.lower(), value.lower())
    if normalized not in {"small", "medium"}:
        raise ValueError("EfficientAD model_size must be one of: s, m, small, medium")
    return normalized


@dataclass(frozen=True, slots=True)
class EfficientAdConfig:
    imagenet_dir: Path
    model_id: str = "efficientad-bottle-v1"
    model_size: str = "small"
    teacher_out_channels: int = 384
    lr: float = 1e-4
    weight_decay: float = 1e-5
    image_size: tuple[int, int] = (256, 256)
    train_batch_size: int = 1
    eval_batch_size: int = 1
    num_workers: int = 0
    max_epochs: int = 20
    max_steps: int = -1
    accelerator: str = "gpu"
    devices: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_size", normalize_model_size(self.model_size))
        object.__setattr__(self, "imagenet_dir", Path(self.imagenet_dir).resolve())
        object.__setattr__(self, "image_size", tuple(self.image_size))
        if len(self.image_size) != 2 or any(type(v) is not int or v < 256 for v in self.image_size):
            raise ValueError("image_size must contain two integer dimensions of at least 256")
        if not self.model_id.strip() or self.teacher_out_channels < 1:
            raise ValueError("model_id and positive teacher_out_channels are required")
        if (
            not math.isfinite(self.lr)
            or self.lr <= 0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("lr must be positive and weight_decay non-negative, both finite")
        if self.train_batch_size != 1:
            raise ValueError("EfficientAD requires train_batch_size=1")
        if self.eval_batch_size < 1 or self.num_workers < 0:
            raise ValueError("eval_batch_size must be positive and num_workers non-negative")
        if self.max_epochs != -1 and self.max_epochs < 1:
            raise ValueError("max_epochs must be -1 or positive")
        if self.max_steps != -1 and self.max_steps < 2:
            raise ValueError("max_steps must be -1 or at least 2 (Anomalib scheduler)")
        if self.max_epochs == self.max_steps == -1:
            raise ValueError("set a finite max_epochs or max_steps")
        if self.accelerator not in {"cpu", "gpu"} or self.devices != 1:
            raise ValueError("This runner supports cpu/gpu with devices=1")

    def training_steps(self, train_samples: int) -> int:
        budgets = [
            value for value in (self.max_steps, self.max_epochs * train_samples) if value > 0
        ]
        if train_samples < 1 or min(budgets) < 2:
            raise ValueError("EfficientAD requires at least two training steps for its scheduler")
        return min(budgets)

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "imagenet_dir": str(self.imagenet_dir),
            "image_size": list(self.image_size),
        }

    def model_spec(self) -> dict[str, Any]:
        return {
            "model_size": self.model_size,
            "teacher_out_channels": self.teacher_out_channels,
            "image_size": list(self.image_size),
            "padding": False,
            "pad_maps": True,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }


def load_efficientad_config(
    path: Path, *, project_root: Path, overrides: dict[str, Any]
) -> EfficientAdConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("name") != "efficientad":
        raise ValueError("Expected an efficientad YAML configuration")
    values = dict(document.get("parameters", {}))
    values["model_id"] = document["model_id"]
    values.update({key: value for key, value in overrides.items() if value is not None})
    unknown = set(values) - {field.name for field in fields(EfficientAdConfig)}
    if unknown:
        raise ValueError(f"Unknown EfficientAD parameters: {sorted(unknown)}")
    if not values.get("imagenet_dir"):
        raise ValueError("Set imagenet_dir in YAML or supply --imagenet-dir")
    image_dir = Path(values["imagenet_dir"])
    values["imagenet_dir"] = image_dir if image_dir.is_absolute() else project_root / image_dir
    return EfficientAdConfig(**values)


def teacher_weight_path(model_size: str, *, pretrained_root: Path | None = None) -> Path:
    if pretrained_root is None:
        module = importlib.import_module("anomalib.utils.path")
        pretrained_root = Path(module.get_pretrained_weights_dir())
    return (
        pretrained_root.resolve()
        / "efficientad_pretrained_weights"
        / f"pretrained_teacher_{normalize_model_size(model_size)}.pth"
    )


def build_efficientad(spec: dict[str, Any], *, model_class: Any = None, **kwargs: Any) -> Any:
    from anomalib.models import EfficientAd

    cls = model_class or EfficientAd
    return cls(
        model_size=normalize_model_size(spec["model_size"]),
        teacher_out_channels=spec["teacher_out_channels"],
        padding=spec["padding"],
        pad_maps=spec["pad_maps"],
        lr=spec["lr"],
        weight_decay=spec["weight_decay"],
        pre_processor=EfficientAd.configure_pre_processor(tuple(spec["image_size"])),
        post_processor=False,
        evaluator=False,
        visualizer=False,
        **kwargs,
    )


def imagefolder_inventory(root: Path) -> dict[str, Any]:
    from PIL import Image
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(root)
    rows = []
    for filename, class_index in dataset.samples:
        path = Path(filename)
        with Image.open(path) as image:
            image.convert("RGB").load()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "class_index": class_index,
                "sha256": sha256_file(path),
            }
        )
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {
        "classes": dataset.classes,
        "sample_count": len(rows),
        "fingerprint": fingerprint,
        "samples": rows,
    }


def load_local_teacher(model: Any, path: Path, *, expected_sha256: str | None = None) -> str:
    import torch

    # Hash and deserialize the same open file, including when a caller supplies a custom root.
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("Teacher weights changed after preflight")
        handle.seek(0)
        state = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Teacher checkpoint must contain a tensor state_dict")
    if any(
        not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all()
        for tensor in state.values()
    ):
        raise ValueError("Teacher state contains non-tensor or non-finite values")
    model.model.teacher.load_state_dict(state, strict=True)
    return digest


def preflight_efficientad_assets(
    *,
    imagenet_dir: Path,
    model_size: str,
    pretrained_root: Path | None = None,
    teacher_out_channels: int = 384,
    image_size: tuple[int, int] = (256, 256),
) -> dict[str, Any]:
    config = EfficientAdConfig(
        imagenet_dir=imagenet_dir,
        model_size=model_size,
        teacher_out_channels=teacher_out_channels,
        image_size=image_size,
    )
    teacher = teacher_weight_path(config.model_size, pretrained_root=pretrained_root)
    if not teacher.is_file() or not config.imagenet_dir.is_dir():
        raise FileNotFoundError(
            f"EfficientAD preflight failed: teacher={teacher}, imagenet_dir={config.imagenet_dir}"
        )
    model = build_efficientad(config.model_spec())
    digest = load_local_teacher(model, teacher)
    inventory = imagefolder_inventory(config.imagenet_dir)
    return {
        "ready": True,
        "model_size": config.model_size,
        "imagenet_dir": str(config.imagenet_dir),
        "teacher_path": str(teacher),
        "teacher_sha256": digest,
        "auxiliary_dataset": inventory,
        "downloads_performed": False,
    }


def efficientad_command(*args: Any, **kwargs: Any) -> AnomalibCommand:
    """Build the legacy reviewed command without executing it."""
    return build_anomalib_command("efficientad", *args, **kwargs)
