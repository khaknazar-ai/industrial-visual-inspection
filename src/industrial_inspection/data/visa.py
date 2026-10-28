"""VisA one-class protocol with an independent train-only validation holdout."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path, PurePosixPath

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.data.contracts import write_manifest
from industrial_inspection.data.variants import _sample, _stable_id


def build_visa_manifest(
    dataset_root: Path, split_csv: Path, category: str, output: Path, *,
    project_root: Path, validation_fraction: float = 0.2, seed: int = 42,
) -> Path:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    with split_csv.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["object"] == category]
    if not rows:
        raise ValueError(f"No VisA samples for {category}")
    train = [row for row in rows if row["split"] == "train"]
    if len(train) < 2 or any(row["label"] != "normal" for row in train):
        raise ValueError("The one-class training split must contain at least two normal images")
    train.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['image']}".encode()).hexdigest())
    count = max(1, min(len(train) - 1, round(len(train) * validation_fraction)))
    validation = {row["image"] for row in train[:count]}

    def resolve(value: str) -> Path:
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or "\\" in value or ":" in value:
            raise ValueError(f"Unsafe VisA path: {value}")
        path = dataset_root.joinpath(*relative.parts).resolve()
        if not path.is_relative_to(dataset_root.resolve()):
            raise ValueError(f"VisA path escapes dataset root: {value}")
        return path

    samples = []
    for row in rows:
        if row["split"] not in {"train", "test"} or row["label"] not in {"normal", "anomaly"}:
            raise ValueError(f"Unexpected one-class split row: {row}")
        path = resolve(row["image"])
        samples.append(_sample(
            path, dataset="visa", category=category,
            split="validation" if row["image"] in validation else row["split"],
            label=row["label"], defect_type="unspecified" if row["label"] == "anomaly" else None,
            mask_path=resolve(row["mask"]) if row["mask"] else None,
            source_group=None, sample_id=_stable_id("visa", path, dataset_root), domain="official_1cls",
        ))
    write_manifest(
        samples, output, project_root, root_aliases={"dataset": dataset_root},
        split_protocol={"name": "visa_official_1cls_train_holdout", "version": 1, "seed": seed,
                        "validation_fraction": validation_fraction, "official_split_sha256": sha256_file(split_csv)},
    )
    return output
