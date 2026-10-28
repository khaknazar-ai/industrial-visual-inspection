"""Undo deterministic resize/center-crop preprocessing, retaining unobserved pixels."""

from __future__ import annotations

import cv2
import numpy as np


def _size(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def restore_map_geometry(anomaly_map, original_shape, transform):
    """Return a bilinear original-size map and its valid support mask.

    Only supported deterministic transforms are accepted. A crop's discarded
    pixels are marked invalid; zero padding is never interpreted as normal GT.
    """
    shape = tuple(original_shape)
    history = []
    operations = getattr(transform, "transforms", [transform] if transform is not None else [])
    for operation in operations:
        name = type(operation).__name__
        if name in {"Normalize", "ToDtype", "ToTensor", "ToImage", "Identity"}:
            continue
        if name == "Compose":
            raise ValueError("Nested preprocessing must be flattened before geometry restoration")
        before = shape
        if name == "Resize":
            size = operation.size
            size = [size] if isinstance(size, int) else list(size)
            if len(size) == 1:
                short, long = sorted(shape)
                new_short, new_long = size[0], int(size[0] * long / short)
                maximum = getattr(operation, "max_size", None)
                if maximum is not None and new_long > maximum:
                    new_short, new_long = int(maximum * new_short / new_long), maximum
                shape = (new_short, new_long) if shape[0] <= shape[1] else (new_long, new_short)
            else:
                shape = tuple(size)
            history.append(("resize", before, shape, None))
        elif name == "CenterCrop":
            shape = _size(operation.size)
            if len(shape) == 1:
                shape = shape * 2
            padded = (max(before[0], shape[0]), max(before[1], shape[1]))
            pad_top, pad_left = max(0, (shape[0] - before[0]) // 2), max(0, (shape[1] - before[1]) // 2)
            top = int(round((padded[0] - shape[0]) / 2)) - pad_top
            left = int(round((padded[1] - shape[1]) / 2)) - pad_left
            history.append(("crop", before, shape, (top, left)))
        else:
            raise ValueError(f"Unsupported preprocessing geometry: {name}")
    response = cv2.resize(np.asarray(anomaly_map, dtype=np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    support = np.ones(shape, dtype=np.float32)
    for name, before, after, offset in reversed(history):
        if name == "resize":
            response = cv2.resize(response, (before[1], before[0]), interpolation=cv2.INTER_LINEAR)
            support = cv2.resize(support, (before[1], before[0]), interpolation=cv2.INTER_LINEAR)
        else:
            top, left = offset
            canvas = np.zeros(before, dtype=np.float32)
            valid = np.zeros(before, dtype=np.float32)
            y0, x0 = max(top, 0), max(left, 0)
            y1, x1 = min(top + after[0], before[0]), min(left + after[1], before[1])
            source = (slice(y0 - top, y1 - top), slice(x0 - left, x1 - left))
            canvas[y0:y1, x0:x1] = response[source]
            valid[y0:y1, x0:x1] = support[source]
            response, support = canvas, valid
    return response, support >= 1 - 1e-6
