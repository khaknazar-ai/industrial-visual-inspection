"""Offline manifest-driven EfficientAD training and versioned inference checkpoints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.core.resources import measure_resources
from industrial_inspection.core.training_protocol import seed_training, training_environment
from industrial_inspection.data.observations import observation_callback
from industrial_inspection.evaluation.artifacts import ArtifactReservation
from industrial_inspection.models.efficientad_adapter import (
    EfficientAdConfig,
    build_efficientad,
    imagefolder_inventory,
    load_local_teacher,
    preflight_efficientad_assets,
)

CHECKPOINT_FORMAT = "industrial_inspection.efficientad.v1"
SCORE_SPACE = "efficientad_quantiles_v1"


def validate_fitted_state(model: Any) -> None:
    import torch

    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("EfficientAD state contains non-finite tensors")
    stats = model.model.mean_std
    quantiles = model.model.quantiles
    if not torch.all(stats["std"] > 0):
        raise ValueError("EfficientAD teacher statistics are not fitted")
    for branch in ("st", "ae"):
        if not quantiles[f"qb_{branch}"] > quantiles[f"qa_{branch}"]:
            raise ValueError("EfficientAD validation quantiles are not fitted or are degenerate")


def save_inference_checkpoint(model: Any, path: Path, config: EfficientAdConfig) -> None:
    import torch

    validate_fitted_state(model)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model_id": config.model_id,
            "model_spec": config.model_spec(),
            "score_space": SCORE_SPACE,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        path,
    )


def _local_model(config: EfficientAdConfig, assets: dict[str, Any]) -> Any:
    from anomalib.models import EfficientAd
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.transforms.v2 import CenterCrop, Compose, RandomGrayscale, Resize, ToTensor

    class LocalEfficientAd(EfficientAd):
        def prepare_pretrained_model(self) -> None:
            self.loaded_teacher_sha256 = load_local_teacher(
                self,
                Path(assets["teacher_path"]),
                expected_sha256=assets["teacher_sha256"],
            )

        def prepare_imagenette_data(self, image_size: Any) -> None:
            current = imagefolder_inventory(config.imagenet_dir)
            if current["fingerprint"] != assets["auxiliary_dataset"]["fingerprint"]:
                raise ValueError("Auxiliary dataset changed after preflight")
            transform = Compose(
                [
                    Resize((image_size[0] * 2, image_size[1] * 2)),
                    RandomGrayscale(p=0.3),
                    CenterCrop(tuple(image_size)),
                    ToTensor(),
                ]
            )
            dataset = ImageFolder(config.imagenet_dir, transform=transform)
            self.imagenet_loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=self.device.type == "cuda",
            )
            self.imagenet_iterator = iter(self.imagenet_loader)

    return build_efficientad(
        config.model_spec(), model_class=LocalEfficientAd, imagenet_dir=config.imagenet_dir
    )


def fit_efficientad(
    *,
    config: EfficientAdConfig,
    manifest: Path,
    project_root: Path,
    category: str,
    output: Path,
    pretrained_root: Path | None = None,
    include_git_provenance: bool = True,
) -> dict[str, Any]:
    # Refuse every existing directory, including failed/incomplete runs. Resume is not implemented.
    if output.exists():
        raise FileExistsError(f"Experiment directory already exists: {output}")
    import torch
    from lightning.pytorch import Trainer

    from industrial_inspection.data.anomalib_manifest import ManifestAnomalibDataModule

    if config.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA is unavailable")
    assets = preflight_efficientad_assets(
        imagenet_dir=config.imagenet_dir,
        model_size=config.model_size,
        pretrained_root=pretrained_root,
        teacher_out_channels=config.teacher_out_channels,
        image_size=config.image_size,
    )
    data = ManifestAnomalibDataModule(
        manifest,
        project_root=project_root,
        category=category,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    splits = data.manifest.splits
    if not any(sample.label == "normal" for sample in splits.validation):
        raise ValueError("EfficientAD needs normal validation images for map normalization")
    expected_steps = config.training_steps(len(splits.train))
    spec = {
        "config": config.as_dict(),
        "category": category,
        "split_fingerprint": splits.fingerprint,
        "teacher_sha256": assets["teacher_sha256"],
        "auxiliary_fingerprint": assets["auxiliary_dataset"]["fingerprint"],
        "environment": training_environment(project_root, include_git=include_git_provenance),
    }
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    reservation = ArtifactReservation.reserve_new(output, spec_fingerprint=fingerprint)
    try:
        splits.write_provenance(output / "manifest_provenance.json")
        (output / "resolved_spec.json").write_text(
            json.dumps(spec, indent=2) + "\n", encoding="utf-8"
        )
        reproducibility = seed_training(config.seed)
        observations = data.observe_fit(output / "observations")
        model = _local_model(config, assets)
        trainer = Trainer(
            accelerator=config.accelerator,
            devices=config.devices,
            max_epochs=config.max_epochs,
            max_steps=config.max_steps,
            default_root_dir=str(output),
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            callbacks=[observation_callback(observations)],
        )
        with measure_resources(cuda=config.accelerator == "gpu") as resources:
            trainer.fit(model=model, datamodule=data)
            if trainer.interrupted or trainer.global_step != expected_steps:
                raise RuntimeError("Training stopped before the configured budget was completed")
            model.to("cuda" if config.accelerator == "gpu" else "cpu")
            # Only validation normals set map quantiles. Decision thresholds are selected separately.
            model.eval()
            with torch.no_grad():
                model.model.quantiles.update(model.map_norm_quantiles(data.val_dataloader()))
        checkpoint = output / "model.ckpt"
        save_inference_checkpoint(model, checkpoint, config)
        observed = observations.write_report(output / "observed_samples.json")
        metadata = {
            "format_version": 2,
            "experiment": output.name,
            "category": category,
            "resolved_config": config.as_dict(),
            "spec_fingerprint": fingerprint,
            "reproducibility": reproducibility,
            "observed_samples": observed,
            "manifest": str(data.manifest_path),
            "manifest_sha256": splits.manifest_sha256,
            "split_fingerprint": splits.fingerprint,
            "counts": {
                "train": len(splits.train),
                "validation": len(splits.validation),
                "test": len(splits.test),
            },
            "training_policy": {
                "fit_split": "train",
                "fit_labels": ["normal"],
                "map_normalization_split": "validation",
                "evaluation_split": "test",
                "test_seen_during_fit": observed["test_seen_during_fit"],
            },
            "resources": {"scope": "fit_and_map_normalization", **resources},
            "calibration_status": "uncalibrated",
            "score_space": SCORE_SPACE,
            "assets": assets,
            "loaded_teacher_sha256": model.loaded_teacher_sha256,
            "training": {
                "max_epochs": config.max_epochs,
                "max_steps": config.max_steps,
                "actual_steps": trainer.global_step,
            },
            "runtime": {
                name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "anomalib", "lightning")
            },
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        }
        (output / "fit_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        reservation.complete(
            artifacts=[
                "model.ckpt",
                "fit_metadata.json",
                "manifest_provenance.json",
                "resolved_spec.json",
                "observed_samples.json",
            ]
        )
        return metadata
    except BaseException as exc:
        reservation.fail(f"{type(exc).__name__}: {exc}")
        raise
