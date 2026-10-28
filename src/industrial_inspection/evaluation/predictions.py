"""Shared, disk-backed predictions and validation-only pixel calibration."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.data.image_io import read_mask
from industrial_inspection.data.integrity import verify_frozen_path
from industrial_inspection.evaluation.pixel_metrics import PixelMetricAccumulator
from industrial_inspection.models.base import AnomalyPrediction, AnomalyRegion


class PredictionStore(Sequence):
    """Keep scalar metadata in memory and load array files on demand."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self.rows = []
        (root / "index.jsonl").touch(exist_ok=False)

    @classmethod
    def open(cls, root: Path):
        """Open an existing prediction cache for read-only policy evaluation."""
        result = cls.__new__(cls)
        result._read_only = True
        result.root = root.resolve()
        with (result.root / "index.jsonl").open(encoding="utf-8") as handle:
            result.rows = [json.loads(line) for line in handle]
        json.dumps(result.rows, allow_nan=False)
        ids = [row["sample_id"] for row in result.rows]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Prediction cache requires unique, non-empty sample IDs")
        for row in result.rows:
            for name, filename in row["arrays"].items():
                path = result.root / filename
                if (name not in {"anomaly_map", "binary_mask", "valid_mask"}
                        or Path(filename).name != filename or path.suffix != ".npy"
                        or path.resolve().parent != result.root):
                    raise ValueError("Prediction array must stay inside its cache directory")
        return result

    def append(self, sample, prediction):
        if getattr(self, "_read_only", False):
            raise PermissionError("Existing prediction cache is read-only")
        digest = verify_frozen_path(sample.image_path) or sha256_file(sample.image_path)
        row = {"sample_id": sample.sample_id, "sample_sha256": digest,
               "score": float(prediction.image_score), "metadata": prediction.metadata,
               "regions": [region.as_dict() for region in prediction.regions], "arrays": {}}
        for name in ("anomaly_map", "binary_mask", "valid_mask"):
            array = getattr(prediction, name)
            if array is not None:
                if array.shape != (sample.height, sample.width) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid {name} in prediction for {sample.sample_id}")
                filename = f"{len(self.rows):06d}-{name}.npy"
                with (self.root / filename).open("xb") as handle:
                    np.save(handle, array, allow_pickle=False)
                row["arrays"][name] = filename
        encoded = json.dumps(row, allow_nan=False, sort_keys=True)
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        row = self.rows[index]
        arrays = {name: np.load(self.root / filename, mmap_mode="r", allow_pickle=False)
                  for name, filename in row["arrays"].items()}
        regions = [AnomalyRegion(tuple(region["bbox_xyxy"]), region["area"], region["region_score"])
                   for region in row["regions"]]
        return AnomalyPrediction(row["score"], arrays.get("anomaly_map"), arrays.get("binary_mask"),
                                 regions, dict(row["metadata"]), arrays.get("valid_mask"))


def predict_samples(model, samples, *, output: Path):
    predictions = PredictionStore(output)
    latencies = []
    for sample in samples:
        verify_frozen_path(sample.image_path)
        started = time.perf_counter()
        prediction = model.predict(sample.image_path)
        verify_frozen_path(sample.image_path)
        latencies.append((time.perf_counter() - started) * 1000)
        predictions.append(sample, prediction)
    return predictions, latencies


def calibrate_pixels(samples, predictions, *, output: Path | None = None):
    ids = []
    with PixelMetricAccumulator(0.0) as accumulator:
        for sample, prediction in zip(samples, predictions, strict=True):
            if sample.split != "validation":
                raise ValueError("Pixel calibration may only use validation")
            if sample.label == "normal" and prediction.anomaly_map is not None:
                accumulator.update(np.zeros(prediction.anomaly_map.shape, dtype=np.uint8),
                                   prediction.anomaly_map, valid_mask=prediction.valid_mask)
                ids.append(sample.sample_id)
        calibration = accumulator.quantile(0.995) if accumulator.pixel_count else {
            "threshold": None, "method": "unavailable", "pixel_count": 0,
        }
        positives = accumulator.count_above(calibration["threshold"]) if accumulator.pixel_count else 0
        calibration.update({"pixel_policy_version": 2, "threshold_operator": ">",
                            "normal_positive_count": positives,
                            "normal_positive_rate": positives / accumulator.pixel_count if accumulator.pixel_count else None,
                            "positive_rate_scope": "valid normal pixels before morphology and component filtering"})
    calibration.update({"split": "validation", "label": "normal", "sample_ids": ids,
                        "limitation": "Normal-only quantiles do not measure defect sensitivity"})
    if output is not None:
        output.write_text(json.dumps(calibration, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return calibration["threshold"]


def evaluate_pixels(samples, predictions, threshold):
    if threshold is None:
        return None
    missing = Counter()
    with PixelMetricAccumulator(threshold) as accumulator:
        for sample, prediction in zip(samples, predictions, strict=True):
            if prediction.anomaly_map is None:
                missing["missing_anomaly_map"] += 1
                continue
            if sample.mask_path is not None:
                mask = read_mask(sample.mask_path)
                if mask is None:
                    raise ValueError(f"Ground truth became unreadable: {sample.mask_path}")
            elif sample.label == "normal":
                mask = np.zeros((sample.height, sample.width), dtype=np.uint8)
            else:
                missing["missing_anomaly_mask"] += 1
                continue
            accumulator.update(mask, prediction.anomaly_map, binary_mask=prediction.binary_mask,
                               valid_mask=prediction.valid_mask)
        result = accumulator.finalize() if accumulator.pixel_count else {"pixel_count": 0, "image_count": 0}
    result["coverage"] = {"requested_images": len(samples), "evaluated_images": result["image_count"],
                          "skipped": dict(missing)}
    return result
