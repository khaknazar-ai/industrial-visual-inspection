"""Synthetic and MVTec AD adapters producing the canonical manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from industrial_inspection.core.paths import find_project_root
from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.data.contracts import InspectionSample, write_manifest
from industrial_inspection.data.image_io import read_color_image, write_image


def _base_assembly(size: int, rng: np.random.Generator) -> np.ndarray:
    def point(x: int, y: int) -> tuple[int, int]:
        scale = size / 256.0
        return round(x * scale), round(y * scale)

    scale = size / 256.0
    y, x = np.mgrid[:size, :size]
    illumination = 152 + 10 * (x / size) + 5 * (y / size)
    noise = rng.normal(0, 1.2, (size, size))
    gray = np.clip(illumination + noise, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image, point(38, 48), point(218, 208), (118, 126, 130), -1)
    cv2.rectangle(image, point(46, 56), point(210, 200), (172, 178, 181), -1)
    for center in (point(72, 80), point(184, 80), point(72, 176), point(184, 176)):
        cv2.circle(image, center, max(3, round(11 * scale)), (72, 76, 78), -1)
        cv2.circle(image, center, max(1, round(4 * scale)), (190, 195, 198), -1)
    cv2.rectangle(image, point(104, 98), point(152, 158), (85, 112, 150), -1)
    cv2.line(image, point(111, 110), point(145, 110), (200, 215, 224), max(1, round(2 * scale)))
    return image


def _inject_anomaly(image: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    result = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    scale = image.shape[1] / 256.0

    def point(x: int, y: int) -> tuple[int, int]:
        return round(x * scale), round(y * scale)

    if kind == "scratch":
        cv2.line(result, point(65, 124), point(190, 145), (28, 28, 34), max(2, round(6 * scale)))
        cv2.line(mask, point(65, 124), point(190, 145), 255, max(3, round(8 * scale)))
    elif kind == "missing_screw":
        cv2.circle(result, point(184, 176), max(4, round(13 * scale)), (172, 178, 181), -1)
        cv2.circle(mask, point(184, 176), max(5, round(14 * scale)), 255, -1)
    elif kind == "contamination":
        axes = (max(4, round(18 * scale)), max(3, round(11 * scale)))
        mask_axes = (max(5, round(20 * scale)), max(4, round(13 * scale)))
        cv2.ellipse(result, point(126, 74), axes, 20, 0, 360, (45, 69, 39), -1)
        cv2.ellipse(mask, point(126, 74), mask_axes, 20, 0, 360, 255, -1)
    else:
        raise ValueError(f"Unknown synthetic anomaly kind: {kind}")
    return result, mask


def generate_synthetic_dataset(
    output_root: Path,
    *,
    project_root: Path | None = None,
    image_size: int = 256,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """Generate a deterministic normal-only train and labelled validation/test fixture."""

    project = (project_root or find_project_root()).resolve()
    manifest_path = output_root / "manifest.csv"
    if manifest_path.exists() and not force:
        return manifest_path

    rng = np.random.default_rng(seed)
    samples: list[InspectionSample] = []
    plan = {
        "train": (8, 0),
        "validation": (4, 4),
        "test": (6, 6),
    }
    anomalies = ("scratch", "missing_screw", "contamination")
    sample_index = 0
    for split, (normal_count, anomaly_count) in plan.items():
        for label, count in (("normal", normal_count), ("anomaly", anomaly_count)):
            for index in range(count):
                local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                image = _base_assembly(image_size, local_rng)
                # Small capture variation is intentional but kept below defect scale.
                dx, dy = (int(local_rng.integers(-2, 3)), int(local_rng.integers(-2, 3)))
                transform = np.float32([[1, 0, dx], [0, 1, dy]])
                image = cv2.warpAffine(image, transform, (image_size, image_size), borderMode=cv2.BORDER_REFLECT)
                defect_type: str | None = None
                mask_path: Path | None = None
                if label == "anomaly":
                    defect_type = anomalies[index % len(anomalies)]
                    image, mask = _inject_anomaly(image, defect_type)
                    mask_dir = output_root / "masks" / split / defect_type
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = mask_dir / f"{sample_index:04d}.png"
                    write_image(mask_path, mask)

                image_dir = output_root / "images" / split / label
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f"{sample_index:04d}.png"
                write_image(image_path, image)
                samples.append(
                    InspectionSample(
                        sample_id=f"synthetic-{sample_index:04d}",
                        dataset="synthetic_industrial",
                        category="assembly",
                        image_path=image_path,
                        split=split,  # type: ignore[arg-type]
                        label=label,  # type: ignore[arg-type]
                        defect_type=defect_type,
                        mask_path=mask_path,
                        source_group=f"capture-{split}-{sample_index:04d}",
                        width=image_size,
                        height=image_size,
                        sha256_optional=sha256_file(image_path),
                    )
                )
                sample_index += 1
    write_manifest(samples, manifest_path, project)
    return manifest_path


def build_mvtec_ad_manifest(
    dataset_root: Path,
    category: str,
    output: Path,
    *,
    project_root: Path | None = None,
    validation_fraction: float = 0.2,
) -> Path:
    """Adapt one MVTec AD category or all categories without copying images."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    project = (project_root or find_project_root()).resolve()
    if category == "all":
        categories = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    else:
        categories = [category]
    if not categories:
        raise ValueError(f"No MVTec categories found under {dataset_root}")

    samples: list[InspectionSample] = []
    for category_name in categories:
        category_root = dataset_root / category_name
        if not category_root.is_dir():
            raise FileNotFoundError(f"MVTec category directory not found: {category_root}")
        for image_path in sorted(category_root.glob("train/good/*")):
            image = read_color_image(image_path)
            token = int(hashlib.sha256(image_path.name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            split = "validation" if token < validation_fraction else "train"
            height, width = image.shape[:2] if image is not None else (1, 1)
            samples.append(
                InspectionSample(
                    sample_id=f"mvtec-{category_name}-train-{image_path.stem}",
                    dataset="mvtec_ad",
                    category=category_name,
                    image_path=image_path,
                    split=split,
                    label="normal",
                    defect_type=None,
                    mask_path=None,
                    source_group=None,
                    width=width,
                    height=height,
                )
            )
        test_root = category_root / "test"
        if not test_root.is_dir():
            raise FileNotFoundError(f"MVTec test directory not found: {test_root}")
        for defect_dir in sorted(test_root.iterdir()):
            if not defect_dir.is_dir():
                continue
            is_normal = defect_dir.name == "good"
            for image_path in sorted(defect_dir.iterdir()):
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                image = read_color_image(image_path)
                height, width = image.shape[:2] if image is not None else (1, 1)
                mask = None
                if not is_normal:
                    candidate = category_root / "ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png"
                    mask = candidate if candidate.is_file() else None
                samples.append(
                    InspectionSample(
                        sample_id=f"mvtec-{category_name}-test-{defect_dir.name}-{image_path.stem}",
                        dataset="mvtec_ad",
                        category=category_name,
                        image_path=image_path,
                        split="test",
                        label="normal" if is_normal else "anomaly",
                        defect_type=None if is_normal else defect_dir.name,
                        mask_path=mask,
                        source_group=None,
                        width=width,
                        height=height,
                    )
                )
    if not samples:
        raise ValueError(f"No readable MVTec images found under {dataset_root}")
    write_manifest(
        samples, output, project, root_aliases={"dataset": dataset_root},
        split_protocol={"name": "mvtec_ad_train_filename_hash_holdout", "version": 1,
                        "validation_fraction": validation_fraction},
    )
    return output
