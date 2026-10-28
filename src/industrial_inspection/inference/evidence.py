"""Immutable evidence artifacts tied to model, input hash, and thresholds."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from industrial_inspection.data.image_io import write_image
from industrial_inspection.models.base import AnomalyPrediction
from industrial_inspection.scoring.policy import DecisionThresholds


@dataclass(frozen=True, slots=True)
class EvidencePaths:
    report_id: str
    original_path: str
    heatmap_path: str | None
    mask_path: str | None
    overlay_path: str | None
    metadata_path: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    @staticmethod
    def _normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
        if not np.isfinite(anomaly_map).all():
            raise ValueError("Evidence cannot publish a non-finite anomaly map")
        finite = anomaly_map.astype(np.float32)
        low, high = float(finite.min()), float(finite.max())
        if high <= low:
            return np.zeros(finite.shape, dtype=np.uint8)
        return np.clip((finite - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)

    def save(
        self,
        *,
        request_id: str,
        image: np.ndarray,
        prediction: AnomalyPrediction,
        model_id: str,
        category: str,
        decision: str,
        thresholds: DecisionThresholds,
        original_sha256: str | None = None,
    ) -> EvidencePaths:
        """Create a new evidence directory; an existing request ID is never overwritten."""

        now = datetime.now(UTC)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", request_id):
            raise ValueError("Invalid evidence request ID")
        directory = self.root / now.date().isoformat() / request_id
        temporary = directory.with_name(f".{request_id}.tmp")
        if directory.exists():
            raise FileExistsError(f"Evidence already exists: {request_id}")
        if not temporary.resolve().is_relative_to(self.root):
            raise ValueError("Evidence temporary path escapes its root")
        created = False
        try:
            temporary.mkdir(parents=True, exist_ok=False)
            created = True
            original_path = temporary / "original.png"
            write_image(original_path, image)
            saved_original_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
            upload_hash = original_sha256
            original_sha256 = original_sha256 or saved_original_hash

            heatmap_path: Path | None = None
            overlay_path: Path | None = None
            mask_path: Path | None = None
            if prediction.anomaly_map is not None:
                normalized = self._normalize_map(prediction.anomaly_map)
                heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
                heatmap_path = temporary / "heatmap.png"
                overlay_path = temporary / "overlay.png"
                resized = cv2.resize(
                    image,
                    (heatmap.shape[1], heatmap.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                overlay = cv2.addWeighted(resized, 0.58, heatmap, 0.42, 0)
                if prediction.valid_mask is not None:
                    overlay[~prediction.valid_mask.astype(bool)] = image[~prediction.valid_mask.astype(bool)]
                write_image(heatmap_path, heatmap)
                write_image(overlay_path, overlay)
            if prediction.binary_mask is not None:
                mask_path = temporary / "mask.png"
                write_image(mask_path, prediction.binary_mask)
            if prediction.valid_mask is not None:
                write_image(temporary / "valid_support.png", prediction.valid_mask.astype(np.uint8) * 255)

            metadata_path = temporary / "report.json"
            metadata = {
                "request_id": request_id,
                "created_at": now.isoformat(),
                "model_id": model_id,
                "category": category,
                "decision": decision,
                "anomaly_score": prediction.image_score,
                "thresholds": thresholds.as_dict(),
                "original_sha256": original_sha256,
                "upload_sha256": upload_hash,
                "saved_original_sha256": saved_original_hash,
                "original_hash_source": "upload_bytes" if upload_hash else "saved_original_png",
                "decode_orientation": "encoded_pixel_order",
                "identity": prediction.metadata.get("identity"),
                "valid_support": "valid_support.png" if prediction.valid_mask is not None else None,
                "regions": [region.as_dict() for region in prediction.regions],
                "model_metadata": prediction.metadata,
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(directory)
        except BaseException:
            if created:
                shutil.rmtree(temporary, ignore_errors=True)
            raise
        return EvidencePaths(
            report_id=request_id,
            original_path=self._relative(directory / "original.png"),
            heatmap_path=self._relative(directory / "heatmap.png") if heatmap_path else None,
            mask_path=self._relative(directory / "mask.png") if mask_path else None,
            overlay_path=self._relative(directory / "overlay.png") if overlay_path else None,
            metadata_path=self._relative(directory / "report.json"),
        )

    def read_report(self, request_id: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", request_id):
            raise ValueError("Invalid evidence request ID")
        matches = list(self.root.glob(f"*/{request_id}/report.json"))
        if len(matches) != 1:
            return None
        def reject_constant(value):
            raise ValueError(f"Invalid non-finite value in persisted report: {value}")
        data = json.loads(matches[0].read_text(encoding="utf-8"), parse_constant=reject_constant)
        # Older reports store prediction metadata but not the response's asset URLs.
        # Reconstruct links from the actual saved files without changing report bytes.
        directory = matches[0].parent
        data["evidence"] = {
            "report_id": request_id,
            **{key: self._relative(directory / filename) if (directory / filename).is_file() else None
               for key, filename in (("original_path", "original.png"), ("heatmap_path", "heatmap.png"),
                                     ("overlay_path", "overlay.png"), ("mask_path", "mask.png"),
                                     ("metadata_path", "report.json"))},
        }
        data.setdefault("metadata", data.get("model_metadata", {}))
        return data
