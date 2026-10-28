"""Small reusable visualization primitives for evidence and reports."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def normalize_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(anomaly_map.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - low) * 255 / (high - low), 0, 255).astype(np.uint8)


def render_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(normalize_heatmap(anomaly_map), cv2.COLORMAP_TURBO)


def render_overlay(image: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    heatmap = render_heatmap(anomaly_map)
    resized = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(resized, 1.0 - alpha, heatmap, alpha, 0)


def save_curve_csv(path: Path, rows: list[dict[str, float]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["x", "y"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
