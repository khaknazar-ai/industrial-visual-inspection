"""PatchCore fit with frozen splits, local encoder weights and observed sample IDs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.core.resources import measure_resources
from industrial_inspection.core.training_protocol import seed_training, training_environment
from industrial_inspection.data.observations import observation_callback
from industrial_inspection.evaluation.artifacts import ArtifactReservation


@dataclass(frozen=True)
class PatchCoreFitConfig:
    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    image_size: tuple[int, int] = (256, 256)
    coreset_sampling_ratio: float = 0.1
    batch_size: int = 1
    num_workers: int = 0
    accelerator: str = "gpu"
    devices: int = 1
    seed: int = 42
    no_pre_trained: bool = False

    def __post_init__(self):
        if self.accelerator not in {"cpu", "gpu"} or self.devices != 1:
            raise ValueError("PatchCore runner supports one CPU/GPU device per experiment")
        if not 0 < self.coreset_sampling_ratio <= 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("Invalid PatchCore coreset/batch/worker configuration")
        if not self.layers or len(self.image_size) != 2 or min(self.image_size) < 32:
            raise ValueError("PatchCore needs selected layers and image dimensions >= 32")


def _load_encoder(model, weights: Path) -> dict:
    import torch

    state = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    encoder = model.model.feature_extractor.feature_extractor
    required = encoder.state_dict()
    if not isinstance(state, dict) or not required.keys() <= state.keys():
        raise ValueError("Local encoder checkpoint does not contain all requested feature layers")
    selected = {key: state[key] for key in required}
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in selected.values()):
        raise ValueError("Encoder weights contain invalid tensors")
    encoder.load_state_dict(selected, strict=True)
    return {"path": str(weights.resolve()), "sha256": sha256_file(weights),
            "loaded_keys": sorted(selected), "unused_keys": sorted(state.keys() - required.keys())}


def fit_patchcore(
    *, manifest: Path, project_root: Path, category: str, output: Path,
    config: PatchCoreFitConfig, backbone_weights: Path | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Experiment directory already exists: {output}")
    if not config.no_pre_trained and (backbone_weights is None or not backbone_weights.is_file()):
        raise FileNotFoundError("Provide --backbone-weights pointing to a local encoder state_dict; automatic downloads are disabled")
    if config.no_pre_trained and backbone_weights is not None:
        raise ValueError("Choose local pretrained weights or explicit untrained fixture mode")
    import torch
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    from industrial_inspection.data.anomalib_manifest import ManifestAnomalibDataModule

    if config.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA is unavailable")
    data = ManifestAnomalibDataModule(
        manifest, project_root=project_root, category=category,
        train_batch_size=config.batch_size, eval_batch_size=config.batch_size,
        num_workers=config.num_workers, seed=config.seed,
    )
    splits = data.manifest.splits
    spec = {"config": asdict(config), "category": category, "split_fingerprint": splits.fingerprint,
            "encoder_sha256": sha256_file(backbone_weights) if backbone_weights else None,
            "environment": training_environment(project_root)}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()
    reservation = ArtifactReservation.reserve_new(output, spec_fingerprint=fingerprint)
    try:
        splits.write_provenance(output / "manifest_provenance.json")
        reproducibility = seed_training(config.seed)
        model = Patchcore(
            backbone=config.backbone, layers=config.layers, pre_trained=False,
            coreset_sampling_ratio=config.coreset_sampling_ratio,
            pre_processor=Patchcore.configure_pre_processor(image_size=config.image_size, center_crop_size=config.image_size),
            visualizer=False,
        )
        if tuple(model.model.feature_extractor.layers) != config.layers:
            raise ValueError("Encoder did not expose all requested feature layers")
        encoder = _load_encoder(model, backbone_weights) if backbone_weights else {"source": "untrained_fixture"}
        if backbone_weights and encoder["sha256"] != spec["encoder_sha256"]:
            raise ValueError("Encoder weights changed during preflight")
        (output / "resolved_spec.json").write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        observations = data.observe_fit(output / "observations")
        engine = Engine(
            accelerator=config.accelerator, devices=config.devices, default_root_dir=str(output),
            max_epochs=1, logger=False, callbacks=[observation_callback(observations)], num_sanity_val_steps=0,
        )
        with measure_resources(cuda=config.accelerator == "gpu") as resources:
            engine.fit(model=model, datamodule=data)
        if engine.trainer.interrupted:
            raise RuntimeError("PatchCore fit was interrupted")
        observed = observations.write_report(output / "observed_samples.json")
        seen = observed["events"].get("training_batch:train", {}).get("sample_sha256", {})
        if set(seen) != set(splits.ids("train")):
            raise RuntimeError("PatchCore fit did not consume every frozen training sample")
        checkpoint = output / "model.ckpt"
        # Lightning hparams can pickle processor modules. Persist inference state
        # and plain constructor values so the runtime can keep weights_only=True.
        torch.save({
            "format": "industrial_patchcore_inference_v1",
            "hyper_parameters": {
                "backbone": config.backbone, "layers": list(config.layers),
                "coreset_sampling_ratio": config.coreset_sampling_ratio,
                "num_neighbors": model.model.num_neighbors, "precision": "float32",
                "image_size": list(config.image_size),
            },
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }, checkpoint)
        metadata = {
            "format_version": 2, "experiment": output.name, "category": category,
            "manifest_sha256": splits.manifest_sha256, "split_fingerprint": splits.fingerprint,
            "spec_fingerprint": fingerprint, "reproducibility": reproducibility,
            "preprocessing": repr(model.pre_processor),
            "observed_samples": observed, "encoder": encoder, "model": {"name": "patchcore", **asdict(config)},
            "resources": {"scope": "fit_and_validation_normalization", **resources},
            "runtime": spec["environment"]["versions"],
            "counts": {name: len(getattr(splits, name)) for name in ("train", "validation", "test")},
            "training_policy": {"fit_split": "train", "fit_labels": ["normal"], "calibration_split": "validation",
                                "evaluation_split": "test", "test_seen_during_fit": observed["test_seen_during_fit"]},
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        }
        (output / "fit_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        reservation.complete(artifacts=["model.ckpt", "manifest_provenance.json", "resolved_spec.json", "fit_metadata.json", "observed_samples.json"])
        return metadata
    except BaseException as exc:
        reservation.fail(f"{type(exc).__name__}: {exc}")
        raise
