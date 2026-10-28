"""Shared immutable prediction tables and bounded failure galleries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from industrial_inspection.data.contracts import InspectionSample
from industrial_inspection.data.image_io import read_color_image, read_mask, write_image
from industrial_inspection.models.base import AnomalyPrediction
from industrial_inspection.scoring.policy import DecisionThresholds, decide


def write_predictions(
    output: Path,
    samples: list[InspectionSample],
    predictions: list[AnomalyPrediction],
    thresholds: DecisionThresholds,
) -> None:
    fields = (
        "sample_id",
        "category",
        "split",
        "label",
        "defect_type",
        "anomaly_score",
        "decision",
        "correct_binary",
        "region_count",
    )
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for sample, prediction in zip(samples, predictions, strict=True):
            decision = decide(prediction.image_score, thresholds)
            binary = decision.value == "fail"
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "split": sample.split,
                    "label": sample.label,
                    "defect_type": sample.defect_type or "",
                    "anomaly_score": f"{prediction.image_score:.17g}",
                    "decision": decision.value,
                    "correct_binary": int(binary == (sample.label == "anomaly")),
                    "region_count": len(prediction.regions),
                }
            )


def write_threshold_curve(output: Path, curve: list[dict[str, float]]) -> None:
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys() if curve else
                                ("threshold", "f1", "expected_cost", "specificity"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curve)


def save_failure_gallery(
    output: Path,
    samples: list[InspectionSample],
    predictions: list[AnomalyPrediction],
    thresholds: DecisionThresholds,
) -> dict[str, list[str]]:
    """Save operator-friendly galleries and per-example metadata."""

    output.mkdir(parents=True, exist_ok=False)
    groups: dict[str, list[int]] = {
        "false_positives": [],
        "false_negatives": [],
        "borderline_review": [],
        "highest_confidence_errors": [],
        "lowest_confidence_correct": [],
        "binary_false_positives": [],
        "binary_false_negatives": [],
    }
    scores = []
    for index, (sample, prediction) in enumerate(zip(samples, predictions, strict=True)):
        scores.append(prediction.image_score)
        decision = decide(prediction.image_score, thresholds).value
        binary_correct = (decision == "fail") == (sample.label == "anomaly")
        if sample.label == "normal" and decision == "fail":
            groups["false_positives"].append(index)
        if sample.label == "anomaly" and decision == "pass":
            groups["false_negatives"].append(index)
        if decision == "review":
            groups["borderline_review"].append(index)
        if decision != "review" and not binary_correct:
            groups["highest_confidence_errors"].append(index)
        if decision != "review" and binary_correct:
            groups["lowest_confidence_correct"].append(index)
        if not binary_correct:
            groups["binary_false_negatives" if sample.label == "anomaly" else "binary_false_positives"].append(index)

    def distance_from_fail_threshold(index: int) -> float:
        return abs(float(scores[index]) - thresholds.fail_threshold)

    index: dict[str, list[str]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    for group_name, entries in groups.items():
        group_dir = output / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        if group_name == "false_positives":
            entries.sort(key=lambda item: scores[item], reverse=True)
        elif group_name == "false_negatives":
            entries.sort(key=lambda item: scores[item])
        elif group_name == "borderline_review":
            entries.sort(key=distance_from_fail_threshold)
        elif group_name == "highest_confidence_errors":
            entries.sort(key=distance_from_fail_threshold, reverse=True)
        else:
            entries.sort(key=distance_from_fail_threshold)
        names: list[str] = []
        records: list[dict[str, Any]] = []
        for sample_index in entries[:5]:
            sample, prediction = samples[sample_index], predictions[sample_index]
            image = read_color_image(sample.image_path)
            if image is None:
                continue
            original = image.copy()
            if prediction.anomaly_map is not None:
                anomaly_map = cv2.resize(
                    prediction.anomaly_map,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                low, high = float(anomaly_map.min()), float(anomaly_map.max())
                normalized = np.zeros_like(anomaly_map, dtype=np.uint8)
                if high > low:
                    normalized = np.clip(
                        (anomaly_map - low) * 255 / (high - low), 0, 255
                    ).astype(np.uint8)
                heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
                image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
                if prediction.valid_mask is not None:
                    image[~prediction.valid_mask.astype(bool)] = original[~prediction.valid_mask.astype(bool)]
            stem = f"sample-{sample_index:06d}"
            overlay_name = f"{stem}.png"
            input_name = f"{stem}_input.png"
            overlay_path = group_dir / overlay_name
            write_image(overlay_path, image)
            write_image(group_dir / input_name, original)
            gt_name = None
            if sample.mask_path is not None and sample.mask_path.exists():
                gt_name = f"{stem}_gt.png"
                mask = read_mask(sample.mask_path)
                if mask is not None:
                    write_image(group_dir / gt_name, mask)
            predicted_name = None
            if prediction.binary_mask is not None:
                predicted_name = f"{stem}_predicted_mask.png"
                write_image(group_dir / predicted_name, prediction.binary_mask)
            relative_overlay = f"{group_name}/{overlay_name}"
            names.append(relative_overlay)
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "defect_type": sample.defect_type,
                    "score": float(prediction.image_score),
                    "decision": decide(prediction.image_score, thresholds).value,
                    "thresholds": thresholds.as_dict(),
                    "input": f"{group_name}/{input_name}",
                    "overlay": relative_overlay,
                    "ground_truth_mask": f"{group_name}/{gt_name}" if gt_name else None,
                    "predicted_mask": f"{group_name}/{predicted_name}" if predicted_name else None,
                    "source_image": str(sample.image_path),
                }
            )
        index[group_name] = names
        details[group_name] = records
    (output / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "details.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return index

