"""Local-weight SuperADD fit and image-inference path.

This runner uses Anomalib's SuperADD implementation with the project's pinned
DINOv3 Small weights. It never asks timm or Hugging Face to download weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.core.resources import measure_resources
from industrial_inspection.core.training_protocol import seed_training, training_environment
from industrial_inspection.data.anomalib_manifest import ManifestAnomalibDataModule
from industrial_inspection.data.observations import observation_callback
from industrial_inspection.evaluation.artifacts import ArtifactReservation
from industrial_inspection.models.dinov3_runtime import ENCODER_ID, ENCODER_SHA256

SUPERADD_FORMAT = "industrial_superadd_inference_v1"


@dataclass(frozen=True, slots=True)
class SuperADDFitConfig:
    encoder_weights: str
    backbone: str = ENCODER_ID
    image_size: tuple[int, int] = (256, 256)
    patch_size: int = 256
    patch_overlap: int = 32
    layers: tuple[int, ...] = (3, 5, 8, 10)
    max_database_size: int = 4096
    subsampling_iterations: int = 100
    gaussian_blur_sigma: float = 4.0
    score_quantile: float = 0.001
    batch_size: int = 1
    num_workers: int = 0
    accelerator: str = "gpu"
    devices: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.backbone != ENCODER_ID:
            raise ValueError(f"This runner supports only the pinned encoder {ENCODER_ID}")
        if (len(self.image_size) != 2 or any(type(size) is not int or size < self.patch_size or size % 16
                                             for size in self.image_size)):
            raise ValueError("image_size must contain two patch-aligned dimensions >= patch_size")
        if (self.patch_size % 16 or self.patch_overlap <= 0 or self.patch_overlap % 16
                or self.patch_size <= 2 * self.patch_overlap):
            raise ValueError("patch_size and patch_overlap must be valid multiples of the DINO patch size 16")
        if not self.layers or any(layer < 0 for layer in self.layers):
            raise ValueError("At least one nonnegative DINO layer is required")
        if self.max_database_size < 1 or self.subsampling_iterations < 1:
            raise ValueError("Memory-bank limits must be positive")
        if self.batch_size < 1 or self.num_workers < 0 or self.devices != 1:
            raise ValueError("SuperADD supports a positive batch size, nonnegative workers, and one device")
        if self.accelerator not in {"cpu", "gpu"}:
            raise ValueError("accelerator must be cpu or gpu")
        if self.gaussian_blur_sigma < 0 or not 0 < self.score_quantile <= 1:
            raise ValueError("Invalid smoothing or image-score quantile")


def _build_model(config: SuperADDFitConfig, encoder_weights: Path | None):
    """Build the installed Anomalib model while forcing local, strict weights."""

    try:
        import timm
        import torch
        from anomalib.models import SuperADD
        from anomalib.models.image.super_add import torch_model as superadd_torch_model
        from safetensors.torch import load_file
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "SuperADD requires the deep dependencies (Anomalib >=2.6, timm, torch, safetensors)"
        ) from exc

    class LocalDinoV3Backbone(torch.nn.Module):
        def __init__(self, backbone: str, layers: list[int]) -> None:
            super().__init__()
            self.dino = timm.create_model(backbone, pretrained=False, num_classes=0).eval()
            self.layers = layers
            self.model_patch_size = int(self.dino.patch_embed.patch_size[0])
            if encoder_weights is not None:
                state = load_file(str(encoder_weights), device="cpu")
                if any(not torch.isfinite(tensor).all() for tensor in state.values()):
                    raise ValueError("DINOv3 weights contain non-finite tensors")
                self.dino.load_state_dict(state, strict=True)
            self.dino.requires_grad_(False)

        def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
            with torch.inference_mode():
                return self.dino.forward_intermediates(
                    images,
                    indices=self.layers,
                    norm=False,
                    output_fmt="NLC",
                    intermediates_only=True,
                )

    # Anomalib's upstream wrapper currently hard-codes pretrained=True. Replace
    # only its backbone constructor during model creation; the model itself and
    # all inference operations remain the installed Anomalib implementation.
    original_backbone = superadd_torch_model.DinoV3Backbone
    superadd_torch_model.DinoV3Backbone = LocalDinoV3Backbone
    try:
        model = SuperADD(
            backbone=config.backbone,
            patch_size=config.patch_size,
            patch_overlap=config.patch_overlap,
            layers=list(config.layers),
            gaussian_blur_sigma=config.gaussian_blur_sigma,
            score_quantile=config.score_quantile,
            pre_processor=SuperADD.configure_pre_processor(image_size=config.image_size),
            visualizer=False,
        )
    finally:
        superadd_torch_model.DinoV3Backbone = original_backbone
    model.model.max_database_size = config.max_database_size
    model.model.subsampling_iterations = config.subsampling_iterations
    return model


def fit_superadd(
    *,
    manifest: Path,
    project_root: Path,
    category: str,
    output: Path,
    config: SuperADDFitConfig,
) -> dict[str, Any]:
    """Build a normal-only feature bank from the frozen manifest train split."""

    if output.exists():
        raise FileExistsError(f"Experiment directory already exists: {output}")
    weights = Path(config.encoder_weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Provide local DINOv3 weights: {weights}")
    if sha256_file(weights) != ENCODER_SHA256:
        raise ValueError("DINOv3 weights differ from the project's pinned licensed encoder")

    try:
        import torch
        from anomalib.engine import Engine
    except (ImportError, OSError) as exc:
        raise RuntimeError("SuperADD fitting requires the deep CPU/GPU environment") from exc
    if config.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA is unavailable")

    root = project_root.resolve()
    data = ManifestAnomalibDataModule(
        manifest,
        project_root=root,
        category=category,
        train_batch_size=config.batch_size,
        eval_batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    splits = data.manifest.splits
    if any(sample.label != "normal" for sample in splits.validation):
        raise ValueError("SuperADD validation calibration requires normal-only validation images")
    spec = {
        "format": SUPERADD_FORMAT,
        "config": asdict(config),
        "category": category,
        "manifest_sha256": splits.manifest_sha256,
        "split_fingerprint": splits.fingerprint,
        "encoder_sha256": sha256_file(weights),
        "environment": training_environment(root),
    }
    fingerprint = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    reservation = ArtifactReservation.reserve_new(output, spec_fingerprint=fingerprint)
    try:
        splits.write_provenance(output / "manifest_provenance.json")
        reproducibility = seed_training(config.seed)
        model = _build_model(config, weights)
        if tuple(model.model.layers) != config.layers:
            raise ValueError("Loaded SuperADD does not expose the requested DINO layers")
        (output / "resolved_spec.json").write_text(
            json.dumps(spec, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        observations = data.observe_fit(output / "observations")
        engine = Engine(
            accelerator=config.accelerator,
            devices=config.devices,
            default_root_dir=str(output),
            max_epochs=1,
            logger=False,
            callbacks=[observation_callback(observations)],
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_progress_bar=False,
            barebones=True,
        )
        with measure_resources(cuda=config.accelerator == "gpu") as resources:
            engine.fit(model=model, datamodule=data)
        if engine.trainer.interrupted:
            raise RuntimeError("SuperADD feature-bank build was interrupted")
        observed = observations.write_report(output / "observed_samples.json")
        seen = observed["events"].get("training_batch:train", {}).get("sample_sha256", {})
        if set(seen) != set(splits.ids("train")):
            raise RuntimeError("SuperADD did not consume every frozen normal training sample")
        if model.model.memory_bank.numel() == 0:
            raise RuntimeError("SuperADD completed without producing a memory bank")

        checkpoint = output / "model.pt"
        torch.save(
            {
                "format": SUPERADD_FORMAT,
                "config": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in asdict(config).items()
                    if key != "encoder_weights"
                },
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            },
            checkpoint,
        )
        metadata = {
            "format_version": 1,
            "category": category,
            "manifest_sha256": splits.manifest_sha256,
            "split_fingerprint": splits.fingerprint,
            "spec_fingerprint": fingerprint,
            "reproducibility": reproducibility,
            "encoder": {"id": config.backbone, "path": str(weights), "sha256": sha256_file(weights)},
            "model": {"name": "anomalib_superadd", **asdict(config), "encoder_weights": str(weights)},
            "observed_samples": observed,
            "resources": {"scope": "normal_memory_fit_and_validation_calibration", **resources},
            "runtime": spec["environment"]["versions"],
            "counts": {name: len(getattr(splits, name)) for name in ("train", "validation", "test")},
            "training_policy": {
                "fit_split": "train",
                "fit_labels": ["normal"],
                "calibration_split": "validation",
                "evaluation_split": "test",
                "test_seen_during_fit": observed["test_seen_during_fit"],
            },
            "qualification": "technical_demo_only; no test evaluation performed",
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        }
        (output / "fit_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        reservation.complete(
            artifacts=[
                "model.pt",
                "manifest_provenance.json",
                "resolved_spec.json",
                "fit_metadata.json",
                "observed_samples.json",
            ]
        )
        return metadata
    except BaseException as exc:
        reservation.fail(f"{type(exc).__name__}: {exc}")
        raise


def predict_superadd_image(
    *, checkpoint: Path, image: Path, output: Path, device: str = "auto"
) -> dict[str, Any]:
    """Run a saved SuperADD bank on one image and save an overlay plus JSON."""

    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if not checkpoint.is_file() or not image.is_file():
        raise FileNotFoundError("Both a SuperADD model.pt and an input image are required")
    try:
        import cv2
        import numpy as np
        import torch
        from industrial_inspection.models.base import load_color_image
        from industrial_inspection.visualization import render_overlay
    except (ImportError, OSError) as exc:
        raise RuntimeError("SuperADD inference requires the deep CPU/GPU environment") from exc

    saved, model, target = load_superadd_checkpoint(checkpoint, device=device)

    bgr = load_color_image(image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255).unsqueeze(0)
    with torch.inference_mode():
        prediction = model(tensor.to(target))
    score = float(prediction.pred_score.reshape(-1)[0].detach().cpu())
    label = bool(prediction.pred_label.reshape(-1)[0].detach().cpu())
    anomaly_map = prediction.anomaly_map[0].detach().float().cpu().numpy()

    output.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(output / "overlay.png"), render_overlay(bgr, anomaly_map)):
        raise OSError("Could not save SuperADD overlay")
    np.save(output / "anomaly_map.npy", anomaly_map)
    report = {
        "format_version": 1,
        "model": "anomalib_superadd",
        "image": str(image.resolve()),
        "image_sha256": sha256_file(image),
        "checkpoint_sha256": sha256_file(checkpoint),
        "status": "anomaly" if label else "normal",
        "anomaly_detected": label,
        "score": score,
        "score_space": "anomalib_normalized_anomaly_score",
        "qualification": "technical_demo_only; not an independent quality evaluation",
        "artifacts": {"overlay": "overlay.png", "anomaly_map": "anomaly_map.npy"},
    }
    (output / "prediction.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def load_superadd_checkpoint(checkpoint: Path, *, device: str = "auto"):
    """Load the safe tensor-only SuperADD artifact for inference or evaluation."""

    try:
        import torch
        from industrial_inspection.models.artifacts import resolve_device
    except (ImportError, OSError) as exc:
        raise RuntimeError("SuperADD inference requires the deep CPU/GPU environment") from exc
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format") != SUPERADD_FORMAT:
        raise ValueError("Unsupported SuperADD checkpoint format")
    values = dict(saved["config"])
    values["image_size"] = tuple(values["image_size"])
    values["layers"] = tuple(values["layers"])
    config = SuperADDFitConfig(encoder_weights="", **values)
    model = _build_model(config, encoder_weights=None)
    model.load_state_dict(saved["state_dict"], strict=True)
    target = torch.device(resolve_device(device))
    model.to(target).eval()
    return saved, model, target


__all__ = [
    "SUPERADD_FORMAT",
    "SuperADDFitConfig",
    "fit_superadd",
    "load_superadd_checkpoint",
    "predict_superadd_image",
]
