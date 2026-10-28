"""Anomalib DataModule backed by the project's frozen manifest.

The regular Anomalib data modules are convenient for demos, but their split
helpers may sample from train or test data.  This adapter creates each subset
directly from :class:`ManifestDataModule` so a fit cannot silently change the
manifest assignment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from industrial_inspection.data.manifest_datamodule import ManifestDataModule
from industrial_inspection.data.observations import SampleObservations

try:
    from anomalib.data.datamodules.base.image import AnomalibDataModule as _AnomalibDataModule
    from anomalib.data.datasets.base import AnomalibDataset as _AnomalibDataset
except (ImportError, OSError):  # pragma: no cover - optional deep dependency
    _AnomalibDataModule = object
    _AnomalibDataset = object


def _require_anomalib() -> tuple[Any, Any, Any, Any]:
    try:
        import pandas as pd
        from anomalib.data.datasets.base import AnomalibDataset
        from anomalib.data.utils import LabelName, Split
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "ManifestAnomalibDataModule requires the optional anomalib dependencies"
        ) from exc
    return pd, AnomalibDataset, LabelName, Split


class _ManifestDataset(_AnomalibDataset):
    """Module-level class so Windows spawn workers can deserialize the dataset."""

    def __init__(self, samples: tuple[Any, ...], augmentations: Any = None, observations: SampleObservations | None = None) -> None:
        pd, _, LabelName, Split = _require_anomalib()
        super().__init__(augmentations=augmentations)
        rows = []
        for sample in samples:
            if sample.label not in {"normal", "anomaly"}:
                raise ValueError("Training dataset requires known labels")
            rows.append({
                "image_path": str(sample.image_path), "label": sample.label,
                "label_index": LabelName.NORMAL if sample.label == "normal" else LabelName.ABNORMAL,
                "mask_path": "", "split": {"train": Split.TRAIN, "validation": Split.VAL, "test": Split.TEST}[sample.split],
            })
        frame = pd.DataFrame(rows)
        frame.attrs["task"] = "classification"
        self.samples = frame
        self.category = samples[0].category if samples else None
        self.observations = observations

    def __getitem__(self, index):
        if self.observations is not None:
            self.observations.verify(self.samples.iloc[index].image_path)
        item = super().__getitem__(index)
        if self.observations is not None:
            self.observations.record(item.image_path, "read")
        return item


class ManifestAnomalibDataModule(_AnomalibDataModule):
    """An Anomalib-compatible DataModule with immutable train/val/test rows."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        project_root: Path,
        category: str,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        num_workers: int = 0,
        seed: int | None = 0,
    ) -> None:
        if _AnomalibDataModule is object:
            raise RuntimeError(
                "ManifestAnomalibDataModule requires the optional anomalib dependencies"
            )
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode="none",
            test_split_mode="none",
            seed=seed,
        )
        self.manifest_path = manifest_path.resolve()
        self.project_root = project_root.resolve()
        self.manifest = ManifestDataModule.from_manifest(
            self.manifest_path,
            project_root=self.project_root,
            category=category,
        )
        self.category = category
        self.observations: SampleObservations | None = None

    def observe_fit(self, directory: Path) -> SampleObservations:
        if self._is_setup:
            raise RuntimeError("Enable fit observations before datamodule setup")
        self.observations = SampleObservations(self.manifest.splits, directory)
        return self.observations

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = _ManifestDataset(self.manifest.splits.train, observations=self.observations)
        self.val_data = _ManifestDataset(self.manifest.splits.validation, observations=self.observations)
        self.test_data = _ManifestDataset(self.manifest.splits.test, observations=self.observations)
        self.train_data.category = self.category
        self.val_data.category = self.category
        self.test_data.category = self.category

    def setup(self, stage: str | None = None) -> None:
        if not self._is_setup:
            self._setup(stage)
            self._is_setup = True
        self._update_augmentations()


__all__ = ["ManifestAnomalibDataModule"]
