"""Run the bounded LOCO splicing-connectors image-to-rule demo on one image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from industrial_inspection.data.image_io import read_color_image, write_image
from industrial_inspection.loco_components import check_splicing_connectors_image


_COLORS = {
    "left_connector": (104, 120, 79),
    "right_connector": (104, 120, 79),
    "cable": (148, 111, 66),
}
_LABELS = {
    "left_connector": "left connector",
    "right_connector": "right connector",
    "cable": "cable",
}


def _overlay(image, prediction):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    scale = max(0.45, min(width, height) / 1700)
    thickness = max(2, round(scale * 2.2))

    for component in prediction.components:
        x1, y1, x2, y2 = component.bbox_xyxy
        color = _COLORS.get(component.class_name, (112, 112, 112))
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color, thickness)
        label = _LABELS.get(component.class_name, component.class_name.replace("_", " "))
        label_scale = max(0.62, scale * 0.72)
        text_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, max(1, thickness - 1),
        )
        label_y = max(text_size[1] + 6, y1)
        cv2.rectangle(
            canvas,
            (x1, label_y - text_size[1] - 5),
            (x1 + text_size[0] + 8, label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 4, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.55,
            (250, 250, 247),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    band_height = max(38, round(height * 0.05))
    band_color = (72, 94, 79) if prediction.status == "normal" else (64, 91, 119)
    cv2.rectangle(canvas, (0, 0), (width, band_height), band_color, cv2.FILLED)
    issue_text = ", ".join(prediction.issues) if prediction.issues else "no rule violations"
    header = f"{prediction.status.upper()}  /  {issue_text}"
    cv2.putText(
        canvas,
        header,
        (max(10, round(width * 0.012)), round(band_height * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.68, scale * 0.76),
        (250, 250, 247),
        max(1, thickness),
        cv2.LINE_AA,
    )
    return canvas


def _new_output_path(path: Path, image_path: Path, other_output: Path | None) -> Path:
    resolved = path.resolve()
    if resolved == image_path.resolve():
        raise ValueError("Output path must not overwrite the input image")
    if other_output is not None and resolved == other_output.resolve():
        raise ValueError("JSON and overlay outputs must use different paths")
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--json-output", type=Path, help="New JSON report path")
    parser.add_argument("--overlay-output", type=Path, help="New annotated image path")
    parser.add_argument("--normal-profile", type=Path, help="Optional normal-only appearance profile")
    args = parser.parse_args()

    image_path = args.image.resolve(strict=True)
    image = read_color_image(image_path)
    if image is None:
        parser.error(f"Could not decode image: {image_path}")

    normal_profile = None
    normal_profile_info = None
    if args.normal_profile is not None:
        profile_path = args.normal_profile.resolve(strict=True)
        normal_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        normal_profile_info = {
            "path": str(profile_path),
            "sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        }
    prediction = check_splicing_connectors_image(
        image,
        normal_appearance_profile=normal_profile,
    )
    report = {
        "schema": "p13-splicing-connectors-image-demo-v1",
        "input": {
            "path": str(image_path),
            "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
        "normal_appearance_profile": normal_profile_info,
        "prediction": prediction.as_dict(),
        "ground_truth_masks_read": False,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"

    json_path = None
    overlay_path = None
    if args.json_output is not None:
        json_path = _new_output_path(args.json_output, image_path, args.overlay_output)
    if args.overlay_output is not None:
        overlay_path = _new_output_path(args.overlay_output, image_path, json_path)

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(rendered, encoding="utf-8")
    if overlay_path is not None:
        write_image(overlay_path, _overlay(image, prediction))
    print(rendered, end="")
    if json_path is not None:
        print(f"json_output={json_path}")
    if overlay_path is not None:
        print(f"overlay_output={overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
