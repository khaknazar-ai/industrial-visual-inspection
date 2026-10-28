"""Pixel counts are exact; large ranking metrics use a declared bounded histogram."""

from __future__ import annotations

import tempfile
from contextlib import ExitStack
from typing import Any

import numpy as np

from industrial_inspection.scoring.validation import quantile as finite_quantile

_RECORD = np.dtype([("label", "u1"), ("score", "<f8")])


class PixelMetricAccumulator:
    """Spool one image at a time, then rank in bounded chunks.

    Small inputs use sklearn's exact ranking. Larger inputs group scores into
    equal-width bins: AUROC treats each bin as tied, and AP uses grouped precision.
    Threshold confusion counts always use the original scores, independent of bins.
    Memory is bounded by one input image, the configured exact limit or one chunk,
    and two bin counters. Temporary disk consumption is nine bytes per pixel.
    """

    def __init__(
        self,
        threshold: float,
        *,
        max_exact_pixels: int = 1_000_000,
        bins: int = 65_536,
        chunk_pixels: int = 262_144,
        max_spool_bytes: int = 16 * 1024**3,
    ) -> None:
        if not np.isfinite(threshold):
            raise ValueError("pixel threshold must be finite")
        for name, value, minimum in (
            ("max_exact_pixels", max_exact_pixels, 0), ("bins", bins, 2),
            ("chunk_pixels", chunk_pixels, 1), ("max_spool_bytes", max_spool_bytes, 9),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.threshold = float(threshold)
        self.max_exact_pixels = max_exact_pixels
        self.bins = bins
        self.chunk_pixels = chunk_pixels
        self.max_spool_bytes = max_spool_bytes
        self.pixel_count = self.image_count = self.positive = self.predicted = self.tp = 0
        self.excluded_pixels = 0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self._resources = ExitStack()
        self._file = self._resources.enter_context(tempfile.TemporaryFile())  # noqa: SIM115 -- owned by this context manager

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._resources.close()

    def update(self, mask: np.ndarray, anomaly_map: np.ndarray, *, binary_mask=None, valid_mask=None) -> None:
        mask = np.asarray(mask)
        scores = np.asarray(anomaly_map, dtype=np.float64)
        if mask.ndim != 2 or scores.shape != mask.shape or not mask.size:
            raise ValueError("Each mask and anomaly map must have the same non-empty 2D shape")
        if not np.isfinite(mask).all() or (mask < 0).any() or not np.isfinite(scores).all():
            raise ValueError("Masks must be finite/non-negative and anomaly maps finite")
        if valid_mask is not None and not np.isin(np.asarray(valid_mask), [0, 1]).all():
            raise ValueError("Valid support must contain only binary values")
        selected = np.ones(mask.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if selected.shape != mask.shape:
            raise ValueError("Valid support must match mask shape")
        if binary_mask is not None and np.asarray(binary_mask).shape != mask.shape:
            raise ValueError("Predicted mask must match ground-truth shape")
        if binary_mask is not None and not np.isin(np.asarray(binary_mask), [0, 1, 255]).all():
            raise ValueError("Predicted mask must contain only finite binary values 0/1/255")
        pixels = int(np.count_nonzero(selected))
        self.excluded_pixels += mask.size - pixels
        self.image_count += 1
        if not pixels:
            return
        if (self.pixel_count + pixels) * _RECORD.itemsize > self.max_spool_bytes:
            raise ValueError("Pixel metric spool exceeds its declared disk budget")
        ground_truth = mask[selected] > 0
        scores = scores[selected]
        predicted = scores >= self.threshold if binary_mask is None else np.asarray(binary_mask)[selected] > 0
        self.tp += int(np.count_nonzero(predicted & ground_truth))
        self.predicted += int(np.count_nonzero(predicted))
        self.positive += int(np.count_nonzero(ground_truth))
        self.minimum = min(self.minimum, float(scores.min()))
        self.maximum = max(self.maximum, float(scores.max()))
        self.pixel_count += pixels
        self._file.seek(0, 2)
        for offset in range(0, pixels, self.chunk_pixels):
            part = slice(offset, offset + self.chunk_pixels)
            records = np.empty(len(scores[part]), dtype=_RECORD)
            records["label"] = ground_truth[part]
            records["score"] = scores[part]
            self._file.write(records.tobytes())

    def _histogram(self):
        positive = np.zeros(self.bins, dtype=np.int64)
        negative = np.zeros(self.bins, dtype=np.int64)
        self._file.seek(0)
        # Scaling first avoids overflow for finite values close to float64 limits.
        scale = max(abs(self.minimum), abs(self.maximum), 1.0)
        low, high = self.minimum / scale, self.maximum / scale
        while raw := self._file.read(self.chunk_pixels * _RECORD.itemsize):
            records = np.frombuffer(raw, dtype=_RECORD)
            if high == low:
                indices = np.zeros(len(records), dtype=np.int64)
            else:
                positions = (records["score"] / scale - low) / (high - low)
                indices = np.minimum((positions * self.bins).astype(np.int64), self.bins - 1)
            labels = records["label"].astype(bool)
            positive += np.bincount(indices[labels], minlength=self.bins)
            negative += np.bincount(indices[~labels], minlength=self.bins)
        return positive.astype(np.float64), negative.astype(np.float64)

    def quantile(self, quantile: float) -> dict[str, Any]:
        if not self.pixel_count or not np.isfinite(quantile) or not 0 <= quantile <= 1:
            raise ValueError("A quantile in [0, 1] and observed pixels are required")
        if self.pixel_count <= self.max_exact_pixels:
            self._file.seek(0)
            values = np.frombuffer(self._file.read(), dtype=_RECORD)["score"]
            value = finite_quantile(values, quantile)
            return {"quantile": quantile, "threshold": value, "method": "exact",
                    "pixel_count": self.pixel_count, "interval": [value, value]}
        positive, negative = self._histogram()
        counts = positive + negative
        # Pick the upper interpolation order statistic's bin, then its upper edge.
        # This is conservative for normal-only false-positive calibration.
        rank = int(np.ceil(quantile * (self.pixel_count - 1))) + 1
        cumulative = np.cumsum(counts)
        index = int(np.searchsorted(cumulative, rank))
        lower_rank = int(np.floor(quantile * (self.pixel_count - 1))) + 1
        lower_index = int(np.searchsorted(cumulative, lower_rank))
        scale = max(abs(self.minimum), abs(self.maximum), 1.0)
        low, high = self.minimum / scale, self.maximum / scale
        lower = (low + (high - low) * lower_index / self.bins) * scale
        upper = min(self.maximum, (low + (high - low) * (index + 1) / self.bins) * scale)
        return {"quantile": quantile, "threshold": float(upper), "method": "histogram_upper_edge",
                "pixel_count": self.pixel_count, "bins": self.bins, "interval": [float(lower), float(upper)]}

    def count_above(self, threshold: float) -> int:
        """Measure the strict calibration tail exactly, including histogram/constant ties."""
        if not np.isfinite(threshold):
            raise ValueError("pixel threshold must be finite")
        count = 0
        self._file.seek(0)
        while raw := self._file.read(self.chunk_pixels * _RECORD.itemsize):
            scores = np.frombuffer(raw, dtype=_RECORD)["score"]
            count += int(np.count_nonzero(scores > threshold))
        return count

    def finalize(self) -> dict[str, Any]:
        if not self.pixel_count:
            raise ValueError("At least one mask/anomaly map pair is required")
        negatives = self.pixel_count - self.positive
        auc = ap = auc_error = None
        method = "exact"
        if self.pixel_count <= self.max_exact_pixels:
            from sklearn.metrics import average_precision_score, roc_auc_score

            self._file.seek(0)
            records = np.frombuffer(self._file.read(), dtype=_RECORD)
            if self.positive and negatives:
                auc = float(roc_auc_score(records["label"], records["score"]))
                auc_error = 0.0
            if self.positive:
                ap = float(average_precision_score(records["label"], records["score"]))
        else:
            method = "equal_width_histogram"
            positive, negative = self._histogram()
            if self.positive and negatives:
                below = np.cumsum(negative) - negative
                auc = float(np.sum(positive * (below + 0.5 * negative)) / (self.positive * negatives))
                auc_error = float(0.5 * np.sum(positive * negative) / (self.positive * negatives))
            if self.positive:
                tp = np.cumsum(positive[::-1])
                total = tp + np.cumsum(negative[::-1])
                precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
                ap = float(np.sum(positive[::-1] * precision) / self.positive)
        union = self.predicted + self.positive - self.tp
        denominator = self.predicted + self.positive
        fp, fn = self.predicted - self.tp, self.positive - self.tp
        return {
            "roc_auc": auc, "average_precision": ap,
            "dice": 2 * self.tp / denominator if denominator else None,
            "iou": self.tp / union if union else None,
            "pixel_threshold": self.threshold, "pixel_count": self.pixel_count,
            "image_count": self.image_count,
            "excluded_pixel_count": self.excluded_pixels,
            "confusion_matrix": {"tn": negatives - fp, "fp": fp, "fn": fn, "tp": self.tp},
            "ranking": {
                "method": method, "max_exact_pixels": self.max_exact_pixels,
                "bins": self.bins if method != "exact" else None,
                "score_min": self.minimum, "score_max": self.maximum,
                "auroc_absolute_error_bound": auc_error,
                "average_precision_approximate": method != "exact",
                "chunk_pixels": self.chunk_pixels,
                "spool_bytes": self.pixel_count * _RECORD.itemsize,
            },
        }
