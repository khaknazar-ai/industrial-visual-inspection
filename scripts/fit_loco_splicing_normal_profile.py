"""Fit a small normal-only appearance review profile for LOCO splicing connectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from industrial_inspection.data.image_io import read_color_image
from industrial_inspection.loco_appearance import extract_features, fit_profile
from industrial_inspection.loco_components import (
    SplicingConnectorProfile,
    extract_splicing_connector_components,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vectors(root: Path, split: str):
    vectors = []
    skipped = []
    for image_path in sorted((root / split / "good").glob("*.png")):
        image = read_color_image(image_path)
        if image is None:
            raise OSError(f"Could not decode image: {image_path}")
        components = extract_splicing_connector_components(image)
        vector = extract_features(image, components, SplicingConnectorProfile())
        if vector is None:
            skipped.append(image_path.name)
        else:
            vectors.append(vector)
    return vectors, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category-root",
        type=Path,
        required=True,
        help="Path to the splicing_connectors directory containing train/ and validation/",
    )
    parser.add_argument("--output", type=Path, required=True, help="New JSON profile path")
    args = parser.parse_args()

    category_root = args.category_root.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        parser.error(f"Refusing to overwrite existing profile: {output_path}")
    if output_path.is_relative_to(category_root):
        parser.error("Profile must be written outside the dataset directory")
    source_path = category_root.parent / "SOURCE.json"
    if not source_path.is_file():
        parser.error(f"Missing dataset source record: {source_path}")
    source_record = json.loads(source_path.read_text(encoding="utf-8"))

    train_vectors, train_skipped = _vectors(category_root, "train")
    validation_vectors, validation_skipped = _vectors(category_root, "validation")
    source = {
        "repository": source_record.get("source_repo"),
        "revision": source_record.get("source_revision"),
        "license": source_record.get("category_license"),
        "official_archive_equivalence_proven": source_record.get(
            "official_archive_equivalence_proven",
        ),
        "source_record_sha256": _sha256(source_path),
        "extractor_sha256": _sha256(
            Path(__file__).resolve().parents[1] / "src/industrial_inspection/loco_components.py",
        ),
        "feature_code_sha256": _sha256(
            Path(__file__).resolve().parents[1] / "src/industrial_inspection/loco_appearance.py",
        ),
    }
    profile = fit_profile(train_vectors, validation_vectors, source=source)
    profile["train_images_seen"] = len(train_vectors) + len(train_skipped)
    profile["train_images_skipped"] = train_skipped
    profile["validation_images_seen"] = len(validation_vectors) + len(validation_skipped)
    profile["validation_images_skipped"] = validation_skipped
    profile["dataset_license_notice"] = (
        "Derived calibration profile; retain the source dataset license and provenance."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"train_used={len(train_vectors)} skipped={len(train_skipped)}")
    print(f"validation_used={len(validation_vectors)} skipped={len(validation_skipped)}")
    print(f"review_threshold={profile['review_threshold']:.4f}")
    print(f"validation_score_max={profile['validation_score_max']:.4f}")
    print(f"profile={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
