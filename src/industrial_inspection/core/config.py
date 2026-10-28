"""Strict YAML configuration with environment overrides."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from industrial_inspection.core.paths import StoragePaths, default_runtime_root, find_project_root
from industrial_inspection.core.runtime_resources import runtime_resource_path
from industrial_inspection.scoring.costs import ReviewCosts

_ENV_TOKEN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    name: str = "industrial-visual-inspection"
    root: str = "."


class StorageConfig(StrictModel):
    dataset_root: str = "data/external"
    model_root: str = "models"
    artifact_root: str = "reports/runtime"


class DataConfig(StrictModel):
    dataset: str = "synthetic_industrial"
    category: str = "assembly"
    image_size: tuple[int, int] = (256, 256)


class ModelConfig(StrictModel):
    name: str = "classical_diff"
    model_id: str = "classical-synthetic-v1"
    parameters: dict[str, Any] = Field(default_factory=dict)


class ReviewCostConfig(StrictModel):
    false_negative: float = Field(ge=0, allow_inf_nan=False, strict=True)
    false_positive: float = Field(ge=0, allow_inf_nan=False, strict=True)
    review: float = Field(ge=0, allow_inf_nan=False, strict=True)
    unit: str = Field(default="relative_cost", min_length=1)

    def policy(self) -> ReviewCosts:
        return ReviewCosts(**self.model_dump())


class EvaluationConfig(StrictModel):
    pass_threshold: float | None = Field(default=None, allow_inf_nan=False)
    fail_threshold: float | None = Field(default=None, allow_inf_nan=False)
    optimize_metric: str = "f1"
    threshold_source: str | None = None
    threshold_objective: str | None = None
    normal_pass_quantile: float = Field(default=0.95, ge=0, lt=0.995, allow_inf_nan=False)
    bootstrap_samples: int = Field(default=500, ge=1, strict=True)
    review_costs: ReviewCostConfig | None = None

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> EvaluationConfig:
        if (self.pass_threshold is None) != (self.fail_threshold is None):
            raise ValueError("pass_threshold and fail_threshold must be supplied together")
        if (
            self.pass_threshold is not None
            and self.fail_threshold is not None
            and self.pass_threshold >= self.fail_threshold
        ):
            raise ValueError("pass_threshold must be lower than fail_threshold")
        return self


class RuntimeConfig(StrictModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_workers: int = 0
    seed: int = 42
    max_image_bytes: int = Field(default=12 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(default=4096 * 4096, ge=1)
    max_batch_size: int = Field(default=8, ge=1)
    max_batch_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    max_batch_pixels: int = Field(default=32 * 1024 * 1024, ge=1)
    max_batch_decoded_bytes: int = Field(default=96 * 1024 * 1024, ge=1)
    max_concurrent_inference: int = Field(default=1, ge=1)
    quality_min_width: int = Field(default=32, ge=1)
    quality_min_height: int = Field(default=32, ge=1)
    quality_min_sharpness: float = Field(default=5.0, ge=0.0)
    quality_brightness_min: float = Field(default=3.0, ge=0.0, le=255.0)
    quality_brightness_max: float = Field(default=252.0, ge=0.0, le=255.0)

    @model_validator(mode="after")
    def quality_range_is_ordered(self) -> RuntimeConfig:
        if self.quality_brightness_min >= self.quality_brightness_max:
            raise ValueError("quality_brightness_min must be lower than quality_brightness_max")
        return self


class AppConfig(StrictModel):
    _source_path: Path | None = PrivateAttr(default=None)
    _packaged_default: bool = PrivateAttr(default=False)
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def storage_paths(self, config_path: Path | None = None) -> StoragePaths:
        """Resolve configured storage without depending on the caller's CWD."""

        config_path = config_path or self._source_path
        if os.getenv("II_PROJECT_ROOT"):
            project_root = find_project_root()
        else:
            root = Path(self.project.root).expanduser()
            base = default_runtime_root()
            if config_path is not None and not self._packaged_default:
                try:
                    base = find_project_root(config_path.parent)
                except FileNotFoundError:
                    base = config_path.parent.resolve()
            project_root = root.resolve() if root.is_absolute() else base / root
        return StoragePaths.from_values(
            project_root.resolve(),
            self.storage.dataset_root,
            self.storage.model_root,
            self.storage.artifact_root,
        )


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"Environment variable {name} is required by configuration")
            return os.environ[name]

        return _ENV_TOKEN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _apply_runtime_environment(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply documented scalar overrides after YAML and environment expansion."""

    overrides: dict[str, tuple[tuple[str, ...], Any]] = {
        "II_PROJECT_ROOT": (("project", "root"), str),
        "II_CATEGORY": (("data", "category"), str),
        "II_MODEL_NAME": (("model", "name"), str),
        "II_MODEL_ID": (("model", "model_id"), str),
        "II_DEVICE": (("runtime", "device"), str),
        "II_MAX_IMAGE_BYTES": (("runtime", "max_image_bytes"), int),
        "II_MAX_IMAGE_PIXELS": (("runtime", "max_image_pixels"), int),
        "II_MAX_CONCURRENT_INFERENCE": (("runtime", "max_concurrent_inference"), int),
    }
    result = dict(raw)
    for variable, (path, converter) in overrides.items():
        value = os.getenv(variable)
        if value is None:
            continue
        target = result
        for key in path[:-1]:
            nested = target.setdefault(key, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Cannot apply {variable}: {'.'.join(path[:-1])} is not an object")
            target = nested
        target[path[-1]] = converter(value)
    return result


def resolve_config_path(path: str | Path | None = None) -> Path:
    """An explicit path wins over II_CONFIG, then local or packaged defaults."""
    value = path or os.getenv("II_CONFIG")
    if value is None:
        local = default_runtime_root() / "configs/deployment/local.yaml"
        return local if local.is_file() else runtime_resource_path("configs/deployment/local.yaml")
    config_path = Path(value).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() if config_path.is_file() else default_runtime_root()) / config_path
    return config_path.resolve()


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate an application YAML file."""

    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    expanded = _expand_environment(raw)
    if not isinstance(expanded, dict):
        raise ValueError("Configuration root must be an object")
    config = AppConfig.model_validate(_apply_runtime_environment(expanded))
    config._source_path = config_path
    config._packaged_default = config_path == runtime_resource_path("configs/deployment/local.yaml")
    return config
