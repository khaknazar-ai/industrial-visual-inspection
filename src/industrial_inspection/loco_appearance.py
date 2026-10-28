"""Normal-only appearance profile for the LOCO splicing-connectors demo."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from industrial_inspection.logical import ComponentInstance
from industrial_inspection.loco_components import SplicingConnectorProfile

_CLASSES = ("left_connector", "right_connector", "cable")
_HUE_BINS = 9
_FEATURE_FLOOR = 0.025


def feature_names() -> tuple[str, ...]:
    names = []
    for class_name in _CLASSES:
        names.extend((
            f"{class_name}.bbox_width_fraction",
            f"{class_name}.bbox_height_fraction",
            f"{class_name}.bbox_area_fraction",
            f"{class_name}.mask_fill_fraction",
            f"{class_name}.center_x_fraction",
            f"{class_name}.center_y_fraction",
        ))
        names.extend(f"{class_name}.hue_bin_{index}" for index in range(_HUE_BINS))
        names.extend((f"{class_name}.saturation_median", f"{class_name}.value_median"))
    return tuple(names)


def extract_features(
    image_bgr: np.ndarray,
    components: tuple[ComponentInstance, ...],
    profile: SplicingConnectorProfile = SplicingConnectorProfile(),
) -> np.ndarray | None:
    """Return a fixed, image-only feature vector when all three parts are found."""
    height, width = image_bgr.shape[:2]
    by_class: dict[str, list[ComponentInstance]] = {}
    for component in components:
        by_class.setdefault(component.class_name, []).append(component)
    if any(len(by_class.get(name, ())) != 1 for name in _CLASSES):
        return None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    features: list[float] = []
    for class_name in _CLASSES:
        component = by_class[class_name][0]
        x1, y1, x2, y2 = component.bbox_xyxy
        box_width, box_height = x2 - x1, y2 - y1
        box_area = box_width * box_height
        features.extend((
            box_width / width,
            box_height / height,
            box_area / (width * height),
            component.area / box_area,
            component.centroid[0] / width,
            component.centroid[1] / height,
        ))

        crop = hsv[y1:y2, x1:x2]
        if class_name.endswith("connector"):
            color_mask = cv2.inRange(
                crop,
                (0, profile.min_saturation, profile.min_value),
                (15, 255, 255),
            )
        else:
            color_mask = cv2.inRange(
                crop,
                (0, profile.min_saturation, profile.min_value),
                (179, 255, 255),
            )
        pixels = crop[color_mask > 0]
        if pixels.size == 0:
            pixels = crop.reshape(-1, 3)
        hue = pixels[:, 0].astype(np.float32)
        saturation = pixels[:, 1].astype(np.float32) / 255.0
        value = pixels[:, 2].astype(np.float32) / 255.0
        histogram, _ = np.histogram(
            hue,
            bins=_HUE_BINS,
            range=(0, 180),
            weights=saturation + 1e-3,
        )
        histogram = histogram.astype(np.float64)
        histogram /= max(float(histogram.sum()), 1e-12)
        features.extend(histogram.tolist())
        features.extend((float(np.median(saturation)), float(np.median(value))))
    return np.asarray(features, dtype=np.float64)


def fit_profile(
    train_vectors: list[np.ndarray],
    validation_vectors: list[np.ndarray],
    *,
    source: Mapping[str, object],
) -> dict[str, object]:
    if not train_vectors or not validation_vectors:
        raise ValueError("Normal train and validation feature vectors are both required")
    train = np.vstack(train_vectors)
    validation = np.vstack(validation_vectors)
    if train.shape[1] != len(feature_names()) or validation.shape[1] != train.shape[1]:
        raise ValueError("Feature vector shape does not match this profile version")
    median = np.median(train, axis=0)
    mad = np.median(np.abs(train - median), axis=0)
    scale = np.maximum(1.4826 * mad, _FEATURE_FLOOR)
    validation_scores = np.max(np.abs((validation - median) / scale), axis=1)
    threshold = float(validation_scores.max())
    return {
        "schema": "loco-splicing-normal-appearance-v1",
        "category": "splicing_connectors",
        "calibration_status": "normal_only_train_calibrated_at_validation_max",
        "source": dict(source),
        "protocol": {
            "fit_split": "train/good",
            "threshold_split": "validation/good",
            "ground_truth_masks_read": False,
            "anomaly_labels_used": False,
            "test_used": False,
            "score": "maximum robust absolute deviation across normalized geometry and HSV features",
            "decision": "review only; this profile never emits FAIL",
        },
        "training_samples": int(train.shape[0]),
        "validation_samples": int(validation.shape[0]),
        "feature_names": list(feature_names()),
        "median": median.tolist(),
        "scale": scale.tolist(),
        "review_threshold": threshold,
        "validation_score_max": float(validation_scores.max()),
        "validation_score_q95": float(np.quantile(validation_scores, 0.95)),
    }


def score_profile(vector: np.ndarray, profile: Mapping[str, object]) -> dict[str, object]:
    if profile.get("schema") != "loco-splicing-normal-appearance-v1":
        raise ValueError("Unsupported LOCO normal appearance profile")
    names = profile.get("feature_names")
    if names != list(feature_names()):
        raise ValueError("LOCO normal appearance feature schema does not match")
    median = np.asarray(profile.get("median"), dtype=np.float64)
    scale = np.asarray(profile.get("scale"), dtype=np.float64)
    threshold = float(profile.get("review_threshold"))
    if (median.shape != vector.shape or scale.shape != vector.shape
            or not np.isfinite(median).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0) or not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("LOCO normal appearance profile contains invalid calibration values")
    deviations = np.abs((vector - median) / scale)
    top_indices = np.argsort(deviations)[-5:][::-1]
    score = float(deviations.max())
    return {
        "available": True,
        "status": "review" if score > threshold else "normal",
        "score": score,
        "score_type": "robust_max_deviation_not_probability",
        "threshold": threshold,
        "calibration_status": profile.get("calibration_status"),
        "top_deviations": [
            {"feature": feature_names()[int(index)], "score": float(deviations[index])}
            for index in top_indices
        ],
        "ground_truth_used_as_input": False,
    }
