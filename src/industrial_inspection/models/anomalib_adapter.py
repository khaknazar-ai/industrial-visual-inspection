"""Optional Anomalib 2.x adapter boundary for cloud execution.

Imports are lazy so the CPU foundation slice remains installable without
Lightning/Anomalib. Training uses a frozen manifest and local encoder weights.
The old automatic MVTecAD split path is disabled because it can reuse test data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from industrial_inspection.data.manifest_datamodule import ManifestDataModule
from industrial_inspection.models.base import AnomalyPrediction, ImageInput


class AnomalibUnavailableError(RuntimeError):
    """Raised when an optional deep dependency is not installed or compatible."""


def _require_anomalib() -> tuple[Any, Any, Any, Any]:
    try:
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except (ImportError, OSError) as exc:
        raise AnomalibUnavailableError(
            "Install anomalib in the cloud GPU environment using requirements-gpu.txt"
        ) from exc
    try:
        import torch
    except ImportError as exc:
        raise AnomalibUnavailableError("Anomalib requires PyTorch") from exc
    return MVTecAD, Engine, Patchcore, torch


@dataclass(frozen=True, slots=True)
class AnomalibPatchCoreConfig:
    root: Path
    category: str = "bottle"
    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    coreset_sampling_ratio: float = 0.1
    num_workers: int = 0
    batch_size: int = 1
    accelerator: str = "gpu"
    devices: int = 1
    output_root: Path = Path("artifacts/anomalib")


class AnomalibPatchCoreAdapter:
    """Thin, explicit adapter; no fallback to a different model is allowed."""

    def __init__(self, config: AnomalibPatchCoreConfig, model_id: str = "patchcore-cloud-v1") -> None:
        self.config = config
        self.model_id = model_id
        self.engine: Any | None = None
        self.model: Any | None = None
        self.datamodule: Any | None = None

    def fit(self, images: Sequence[ImageInput]) -> None:
        raise ValueError(
            "PatchCore requires a frozen manifest; call fit_manifest() "
            "or use scripts/cloud_patchcore_manifest_fit.py"
        )

    def prepare_manifest(self, manifest_path: Path, *, project_root: Path) -> ManifestDataModule:
        """Freeze canonical splits before an optional Anomalib integration consumes them."""

        datamodule = ManifestDataModule.from_manifest(
            manifest_path,
            project_root=project_root,
            category=self.config.category,
        )
        self.datamodule = datamodule
        return datamodule

    def fit_mvtec(self) -> Path:
        raise ValueError("Automatic MVTec split training is disabled; call fit_manifest with audited splits and local encoder weights")

    def fit_manifest(self, manifest: Path, *, project_root: Path, backbone_weights: Path, seed: int = 42) -> Path:
        from industrial_inspection.models.patchcore_training import (
            PatchCoreFitConfig,
            fit_patchcore,
        )

        output = self.config.output_root
        fit_patchcore(
            manifest=manifest, project_root=project_root, category=self.config.category,
            output=output, backbone_weights=backbone_weights,
            config=PatchCoreFitConfig(
                backbone=self.config.backbone, layers=self.config.layers,
                coreset_sampling_ratio=self.config.coreset_sampling_ratio,
                num_workers=self.config.num_workers, batch_size=self.config.batch_size,
                accelerator=self.config.accelerator, devices=self.config.devices, seed=seed,
            ),
        )
        return output

    def predict(self, image: ImageInput) -> AnomalyPrediction:
        raise RuntimeError(
            "Use predict_datamodule() for Anomalib datasets; raw-image inference should use "
            "the version-specific TorchInferencer/export path."
        )

    def predict_batch(self, images: Sequence[ImageInput]) -> list[AnomalyPrediction]:
        raise RuntimeError("Use predict_datamodule() with Anomalib's version-specific datamodule")

    def predict_datamodule(self, *, ckpt_path: str | Path | None = None) -> list[Any]:
        if self.engine is None or self.model is None or self.datamodule is None:
            raise RuntimeError("Call fit_mvtec() or load_metadata() before prediction")
        return self.engine.predict(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=str(ckpt_path) if ckpt_path else None,
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter.json").write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "adapter": "anomalib_patchcore",
                    "config": {**asdict(self.config), "root": str(self.config.root), "output_root": str(self.config.output_root)},
                    "format_version": 1,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_metadata(cls, path: Path) -> AnomalibPatchCoreAdapter:
        metadata = json.loads((path / "adapter.json").read_text(encoding="utf-8"))
        values = metadata["config"]
        values["root"] = Path(values["root"])
        values["output_root"] = Path(values["output_root"])
        values["layers"] = tuple(values["layers"])
        return cls(AnomalibPatchCoreConfig(**values), metadata["model_id"])
