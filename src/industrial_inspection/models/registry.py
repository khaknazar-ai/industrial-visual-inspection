"""Model registry, capability checks, and one explicit runtime loader."""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from industrial_inspection.models.artifacts import ModelBundle, ModelBundleError, resolve_device
from industrial_inspection.models.base import AnomalyModel
from industrial_inspection.models.classical import ClassicalConfig, ClassicalDifferenceModel
from industrial_inspection.models.deep_patch_memory import (
    DeepPatchMemoryConfig,
    DeepPatchMemoryModel,
)
from industrial_inspection.models.dinov3_runtime import DinoV3Config, DinoV3MemoryModel
from industrial_inspection.models.dinov3_torchscript import DinoV3TorchScriptRuntime
from industrial_inspection.models.efficientad_runtime import EfficientAdRuntime
from industrial_inspection.models.patch_memory import PatchMemoryConfig, PatchMemoryModel
from industrial_inspection.models.patchcore_runtime import PatchCoreRuntime
from industrial_inspection.models.patchcore_torchscript import PatchCoreTorchScriptRuntime
from industrial_inspection.models.superadd_adapter import SuperADDRuntime
from industrial_inspection.scoring.pixels import PixelPolicy
from industrial_inspection.scoring.policy import DecisionThresholds

_BUNDLE_FAMILIES = {
    "dino_torchscript": {"dinov3_torchscript"},
    "dino_memory": {"dinov3_timm_memory"},
    "classical_diff": {"classical_cv"},
    "patch_memory": {"educational_memory_bank"},
    "deep_patch_memory": {"frozen_cnn_memory_bank", "deep_patch_memory"},
    "patchcore": {"anomalib_patchcore", "deep_memory_bank"},
    "patchcore_torchscript": {"patchcore_torchscript"},
    "efficientad": {"anomalib_efficientad", "efficientad", "teacher_student"},
}


class ModelUnavailableError(RuntimeError):
    """Raised when a requested model cannot run in the current environment."""


@dataclass(frozen=True, slots=True)
class ModelCapability:
    name: str
    family: str
    implemented: bool
    dependency_ready: bool
    assets_ready: bool
    train_supported: bool
    predict_supported: bool
    training: str
    detail: str
    loaded: bool = False
    validated: bool = False

    @property
    def available(self) -> bool:
        return (
            self.implemented
            and self.dependency_ready
            and self.assets_ready
            and self.predict_supported
        )

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["available"] = self.available
        return values


@lru_cache(maxsize=8)
def _dependencies_ready(*packages: str) -> bool:
    try:
        for package in packages:
            importlib.import_module(package)
    except (ImportError, OSError, RuntimeError):
        return False
    return True


def runtime_model_name(model: AnomalyModel) -> str | None:
    for name, cls in (
        ("dino_torchscript", DinoV3TorchScriptRuntime),
        ("dino_memory", DinoV3MemoryModel),
        ("classical_diff", ClassicalDifferenceModel), ("patch_memory", PatchMemoryModel),
        ("deep_patch_memory", DeepPatchMemoryModel), ("patchcore", PatchCoreRuntime),
        ("efficientad", EfficientAdRuntime),
        ("patchcore_torchscript", PatchCoreTorchScriptRuntime),
        ("superadd", SuperADDRuntime),
    ):
        if isinstance(model, cls):
            return name
    return None


def model_capabilities(
    *,
    loaded_name: str | None = None,
    loaded_validated: bool = False,
    probe_optional: bool = False,
) -> list[ModelCapability]:
    # Catalogue/startup must not load unrelated optional algorithm binaries.
    # A loaded native backend has already proved its imports through its loader.
    anomalib_available = (loaded_name in {"patchcore", "efficientad", "superadd"}
                          or (probe_optional and _dependencies_ready("anomalib.models")))
    return [
        ModelCapability(
            "classical_diff",
            "classical_cv",
            True,
            True,
            True,
            True,
            True,
            "normal-only reference fit",
            "Deterministic median reference, ECC alignment, morphology, and components.",
            loaded=loaded_name == "classical_diff",
            validated=loaded_name == "classical_diff" and loaded_validated,
        ),
        ModelCapability(
            "patch_memory",
            "educational_memory_bank",
            True,
            True,
            True,
            True,
            True,
            "normal-only memory bank",
            "Handcrafted patch descriptors; demonstrates coreset and nearest-neighbour scoring.",
            loaded=loaded_name == "patch_memory",
            validated=loaded_name == "patch_memory" and loaded_validated,
        ),
        ModelCapability(
            "deep_patch_memory",
            "frozen_cnn_memory_bank",
            True,
            _dependencies_ready("torch", "torchvision"),
            loaded_name == "deep_patch_memory",
            True,
            True,
            "frozen CNN features + normal memory bank",
            "Educational frozen-backbone comparison; separate from full Anomalib PatchCore.",
            loaded=loaded_name == "deep_patch_memory",
            validated=loaded_name == "deep_patch_memory" and loaded_validated,
        ),
        ModelCapability(
            "patchcore",
            "deep_memory_bank",
            True,
            anomalib_available,
            loaded_name == "patchcore",
            True,
            True,
            "normal-only deep features via the canonical manifest runner",
            "Requires a compatible anomalib installation and a separate benchmark run.",
            loaded=loaded_name == "patchcore",
            validated=loaded_name == "patchcore" and loaded_validated,
        ),
        ModelCapability(
            "efficientad",
            "teacher_student",
            True,
            anomalib_available,
            loaded_name == "efficientad",
            True,
            True,
            "teacher/student training",
            "Requires a compatible anomalib installation and pretrained teacher assets.",
            loaded=loaded_name == "efficientad",
            validated=loaded_name == "efficientad" and loaded_validated,
        ),
        ModelCapability(
            "patchcore_torchscript", "patchcore_torchscript", True,
            _dependencies_ready("torch", "torchvision"), loaded_name == "patchcore_torchscript",
            False, True, "inference-only fixed-shape export",
            "Requires a bundle published after whole-pipeline parity validation.",
            loaded=loaded_name == "patchcore_torchscript",
            validated=loaded_name == "patchcore_torchscript" and loaded_validated,
        ),
        ModelCapability(
            "dino_memory",
            "foundation_features",
            True,
            _dependencies_ready("torch", "torchvision", "timm", "safetensors"),
            loaded_name == "dino_memory",
            True,
            True,
            "frozen encoder + normal memory",
            "Requires explicitly licensed/configured DINOv3 weights; no implicit download.",
            loaded=loaded_name == "dino_memory",
            validated=loaded_name == "dino_memory" and loaded_validated,
        ),
        ModelCapability(
            "dino_torchscript", "dinov3_torchscript", True, _dependencies_ready("torch"),
            loaded_name == "dino_torchscript", False, True, "inference-only CPU export",
            "Requires an explicit exported graph, bank and bound demo parity receipt.",
            loaded=loaded_name == "dino_torchscript",
            validated=loaded_name == "dino_torchscript" and loaded_validated,
        ),
        ModelCapability(
            "superadd",
            "foundation_memory_bank",
            True,
            anomalib_available,
            loaded_name == "superadd",
            False,
            True,
            "dedicated normal-memory fit CLI",
            "Local exploratory checkpoint supported by CLI/API with embedded validation thresholds. Official SuperADD reproduction is unqualified.",
            loaded=loaded_name == "superadd",
            validated=loaded_name == "superadd" and loaded_validated,
        ),
    ]


def create_model(name: str, model_id: str, parameters: dict[str, Any]) -> AnomalyModel:
    if name == "dino_memory":
        values = dict(parameters)
        if "image_size" in values:
            values["image_size"] = tuple(values["image_size"])
        return DinoV3MemoryModel(DinoV3Config(**values), model_id)
    if name == "classical_diff":
        return ClassicalDifferenceModel(ClassicalConfig(**parameters), model_id)
    if name == "patch_memory":
        values = dict(parameters)
        if "image_size" in values:
            values["image_size"] = tuple(values["image_size"])
        return PatchMemoryModel(PatchMemoryConfig(**values), model_id)
    if name == "deep_patch_memory":
        values = dict(parameters)
        if "image_size" in values:
            values["image_size"] = tuple(values["image_size"])
        if "layers" in values:
            values["layers"] = tuple(values["layers"])
        return DeepPatchMemoryModel(DeepPatchMemoryConfig(**values), model_id)
    capability = next((item for item in model_capabilities() if item.name == name), None)
    if capability is None:
        raise KeyError(f"Unknown model: {name}")
    raise ModelUnavailableError(f"{name} is not enabled: {capability.detail}")


def load_model(name: str, path: Path, *, device: str = "auto") -> AnomalyModel:
    """Resolve a strict bundle or an explicitly uncalibrated native artifact."""

    marker = path / "run_status.json" if path.is_dir() else path.parent / "run_status.json"
    if marker.is_file():
        from industrial_inspection.evaluation.artifacts import read_json_object

        if read_json_object(marker).get("status") != "complete":
            raise ModelBundleError("Model belongs to an incomplete or failed build")
    if path.is_dir() and (path / "calibration.json").is_file():
        model, _ = load_bundle_model(name, path, device=device)
        return model
    model = _load_native_model(name, path, device=device)
    from industrial_inspection.models.identity_extensions import runtime_identity

    model.runtime_name = name
    model.runtime_identity = runtime_identity(model)
    threshold_path = path / "thresholds.json" if path.is_dir() else path.with_name("thresholds.json")
    if name == "superadd":
        if threshold_path.exists():
            raise ModelBundleError("SuperADD uses checkpoint-bound thresholds; a sidecar override is not supported")
        return model
    if threshold_path.is_file():
        thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
        if "pixel" in thresholds:
            model.pixel_policy = PixelPolicy.from_dict(thresholds["pixel"])
        if "pass_threshold" in thresholds and "fail_threshold" in thresholds:
            model.decision_thresholds = DecisionThresholds(
                thresholds["pass_threshold"], thresholds["fail_threshold"],
                source=thresholds.get("source", "native_sidecar"),
                objective=thresholds.get("objective", "unspecified"),
            )
    return model


def _load_native_model(name: str, path: Path, *, device: str) -> AnomalyModel:

    if name == "superadd":
        return SuperADDRuntime(path, device=device)
    if name == "dino_torchscript":
        return DinoV3TorchScriptRuntime(path, device=device)
    if name == "dino_memory":
        return DinoV3MemoryModel.load(path, device=device)
    if name == "patchcore_torchscript":
        return PatchCoreTorchScriptRuntime(path, device=device)
    if name == "classical_diff":
        return ClassicalDifferenceModel.load(path)
    if name == "patch_memory":
        return PatchMemoryModel.load(path)
    if name == "deep_patch_memory":
        return DeepPatchMemoryModel.load(path, device=resolve_device(device))
    if name == "patchcore":
        resolved_device = resolve_device(device)
        if path.is_file():
            checkpoint = path
        elif path.is_dir():
            candidates = sorted(path.rglob("*.ckpt"))
            if not candidates:
                raise FileNotFoundError(f"No .ckpt file found under PatchCore model directory: {path}")
            if len(candidates) != 1:
                raise ValueError(
                    "Multiple PatchCore checkpoints found; pass the checkpoint file directly: "
                    f"{candidates}"
                )
            checkpoint = candidates[0]
        else:
            raise FileNotFoundError(f"Model path does not exist: {path}")
        return PatchCoreRuntime(checkpoint, device=resolved_device)
    if name == "efficientad":
        resolved_device = resolve_device(device)
        if path.is_file():
            checkpoint = path
        elif path.is_dir():
            candidates = sorted(path.rglob("*.ckpt"))
            if not candidates:
                raise FileNotFoundError(f"No .ckpt file found under EfficientAD model directory: {path}")
            if len(candidates) != 1:
                raise ValueError(
                    "Multiple EfficientAD checkpoints found; pass the checkpoint file directly: "
                    f"{candidates}"
                )
            checkpoint = candidates[0]
        else:
            raise FileNotFoundError(f"Model path does not exist: {path}")
        return EfficientAdRuntime(checkpoint, device=resolved_device)
    raise ModelUnavailableError(f"Loading {name!r} is not implemented in the local baseline runtime")


def load_bundle_model(
    name: str,
    bundle_path: Path,
    *,
    category: str | None = None,
    device: str = "auto",
    allow_legacy: bool = False,
) -> tuple[AnomalyModel, ModelBundle]:
    """Load a model only after a strict bundle has passed its identity checks."""

    bundle = ModelBundle.load(bundle_path, expected_category=category, allow_legacy=allow_legacy)
    if bundle.family not in _BUNDLE_FAMILIES.get(name, set()):
        raise ModelBundleError(f"Bundle family {bundle.family!r} does not match requested model {name!r}")
    if name == "patchcore_torchscript":
        from industrial_inspection.core.reproducibility import sha256_file

        receipt = bundle.root.parent
        status = json.loads((receipt / "run_status.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            raise ModelBundleError("Export publication is incomplete")
        if sha256_file(receipt / "parity.json") != bundle.calibration.get("export_parity_sha256"):
            raise ModelBundleError("Export parity report differs from its calibration")
        parity = json.loads((receipt / "parity.json").read_text(encoding="utf-8"))
        if parity.get("status") != "passed":
            raise ModelBundleError("Export parity did not pass")
    return _load_bundle_runtime(name, bundle, device=device), bundle


def _load_bundle_runtime(name: str, bundle: ModelBundle, *, device: str) -> AnomalyModel:
    """Cold verify a metadata pair before its owning builder commits publication."""
    bundle.verify()
    if bundle.family not in _BUNDLE_FAMILIES.get(name, set()):
        raise ModelBundleError(f"Bundle family {bundle.family!r} does not match requested model {name!r}")
    expected_kind = "directory" if name in {"classical_diff", "patch_memory", "deep_patch_memory", "patchcore_torchscript", "dino_memory", "dino_torchscript"} else "file"
    if bundle.model["weights"].get("kind", "file") != expected_kind:
        raise ModelBundleError(f"{name} requires a {expected_kind} artifact in the bundle")
    model = _load_native_model(name, bundle.artifact_path, device=device)
    from industrial_inspection.models.identity_extensions import runtime_identity

    actual_identity = runtime_identity(model)
    if "runtime_identity" in bundle.model and actual_identity != bundle.model["runtime_identity"]:
        raise ModelBundleError("Loaded runtime identity differs from calibration; revalidation is required")
    # Detect artifact replacement between checksum verification and cold loading.
    bundle.verify()
    if getattr(model, "score_space", "raw_distance") != bundle.score_space:
        raise ModelBundleError("Bundle score space does not match the runtime")
    model.model_id = bundle.model_id
    model.runtime_name = name
    model.runtime_identity = actual_identity
    model.bundle = bundle
    model.category = bundle.category
    model.model_fingerprint = bundle.model_fingerprint
    model.score_space = bundle.score_space
    model.calibration_id = bundle.calibration_id
    model.calibration_status = bundle.calibration_status
    model.pixel_policy = PixelPolicy.from_dict(bundle.calibration.get("pixel"))
    image = bundle.calibration["image"]
    model.decision_thresholds = DecisionThresholds(
        pass_threshold=image["pass_threshold"],
        fail_threshold=image["fail_threshold"],
        source=bundle.calibration.get("threshold_source") or bundle.calibration.get("source", "bundle"),
        objective=bundle.calibration.get("objective", "unspecified"),
    )
    return model
