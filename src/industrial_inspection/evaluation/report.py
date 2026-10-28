"""Shared category reports with explicit aggregation and bootstrap scope."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from industrial_inspection.evaluation.metrics import (
    coverage_risk_curve,
    image_level_metrics,
    review_system_metrics,
)
from industrial_inspection.evaluation.predictions import evaluate_pixels
from industrial_inspection.scoring.costs import ReviewCosts
from industrial_inspection.scoring.policy import DecisionThresholds

IMAGE_MEASURES = ("roc_auc", "average_precision", "precision", "recall", "f1", "specificity",
                  "false_positive_rate", "false_negative_rate")
REVIEW_MEASURES = ("automation_rate", "review_rate", "fail_rate", "accuracy_auto_decided",
                   "false_negative_rate_after_abstention", "false_positive_rate_after_abstention",
                   "expected_cost_per_image")
PIXEL_MEASURES = ("roc_auc", "average_precision", "dice", "iou")


def write_json_artifact(output: Path, value) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def write_markdown_summary(output: Path, title: str, paragraphs: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write("# " + title + "\n\n" + "\n\n".join(paragraphs) + "\n")


def write_evaluation_documents(output, *, metrics, timings, config, environment, gallery, fitted):
    """Publish the same report documents for new fits and saved-model evaluations."""
    for name, value in (("metrics", metrics), ("timings", timings), ("environment", environment)):
        write_json_artifact(output / f"{name}.json", value)
    with (output / "config.yaml").open("x", encoding="utf-8") as handle:
        handle.write(yaml.safe_dump(config, sort_keys=False))
    counts = config["split_counts"]
    mode = config.get("threshold_mode", "validation")
    fit_note = f"Fitted on {counts['train']} normal training images." if fitted else "Evaluated an externally fitted model without refitting."
    calibration_note = ("Reused the model's frozen image/pixel policy."
                        if mode == "model" else "Selected image/pixel policy on validation only.")
    image, review = metrics["image_level"], metrics["review_system"]
    latency = timings["test_inference"]
    write_markdown_summary(output / "summary.md", f"{metrics['experiment_id']}: {config['model']['model_id']}", [
        f"{fit_note} {calibration_note} Validation: {counts['validation']} images; final test: {counts['test']} images.",
        f"- Dataset: {config['dataset']}\n- Categories: {', '.join(metrics['per_category'])}\n"
        f"- Image ROC-AUC/AP: {image['roc_auc']} / {image['average_precision']}\n"
        f"- F1 at frozen fail threshold: {image['f1']}\n- Review rate: {review['review_rate']}\n"
        f"- Expected decision cost per image: {review['expected_cost_per_image']}\n"
        f"- Test prediction p50/p95: {latency['p50_ms']:.3f} / {latency['p95_ms']:.3f} ms\n"
        f"- Failure gallery entries: {sum(len(items) for items in gallery.values())}",
        "Results cover only the reported dataset, categories and frozen policy. Synthetic fixtures establish pipeline integrity only. "
        "Conditional image confidence intervals do not measure training/calibration uncertainty or unseen categories. "
        "Warm GPU/RAM/VRAM measurements are separate benchmark artifacts.",
    ])


class _Subset(Sequence):
    """Select disk-backed predictions without loading their maps into a list."""

    def __init__(self, values, indices):
        self.values, self.indices = values, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.values[self.indices[index]]


def _bootstrap_units(samples):
    # A session or physical item may connect several observations. Union both
    # identifiers so correlated images never become separate bootstrap draws.
    parents = list(range(len(samples)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    annotated = 0
    for i, sample in enumerate(samples):
        annotated += bool(sample.physical_item_id or sample.session_id)
        for field in ("physical_item_id", "session_id"):
            value = getattr(sample, field)
            if value:
                key = (field, value)
                if key in seen:
                    parents[root(i)] = root(seen[key])
                seen[key] = i
    groups = {}
    for i in range(len(samples)):
        groups.setdefault(root(i), []).append(i)
    return [np.asarray(indices) for indices in groups.values()], {
        "unit": "connected_item_session" if annotated else "image",
        "unit_count": len(groups), "images_without_group": len(samples) - annotated,
        "limitation": ("Unknown grouping: independence of ungrouped images is assumed."
                       if annotated < len(samples) else None),
    }


def _interval(values, requested):
    finite = np.asarray(values)[np.isfinite(values)]
    return {"lower": float(np.quantile(finite, .025)) if len(finite) else None,
            "upper": float(np.quantile(finite, .975)) if len(finite) else None,
            "valid_resamples": len(finite), "requested_resamples": requested}


def _rank_bootstrap(labels, scores, units, count, rng):
    draws = {name: np.full(count, np.nan) for name in ("roc_auc", "average_precision")}
    if len(units) < 2:
        return draws
    for i in range(count):
        indices = np.concatenate([units[j] for j in rng.integers(len(units), size=len(units))])
        y, s = labels[indices], scores[indices]
        if len(np.unique(y)) == 2:
            draws["roc_auc"][i] = roc_auc_score(y, s)
        if np.any(y):
            draws["average_precision"][i] = average_precision_score(y, s)
    return draws


def build_metrics_report(samples, predictions, thresholds, pixel_threshold, *,
                         bootstrap_samples=500, seed=42, costs: ReviewCosts | None = None):
    """Use frozen policies; never select a threshold or pool category score scales.

    Multiple categories require explicit policies for every category. A single
    runtime's normal evaluator still accepts only its own category.
    """
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if not samples or len(samples) != len(predictions):
        raise ValueError("Samples and predictions must have matching non-zero lengths")
    if len({s.dataset for s in samples}) != 1 or {s.split for s in samples} != {"test"}:
        raise ValueError("A metrics report must contain one dataset's test split")
    if any(s.label not in {"normal", "anomaly"} for s in samples):
        raise ValueError("Metrics require known ground truth labels")
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs in metrics report")
    categories = sorted({s.category for s in samples})
    grouping = {}
    for sample in samples:
        for field in ("physical_item_id", "session_id"):
            value = getattr(sample, field)
            if value:
                key = (field, value)
                if key in grouping and grouping[key] != sample.category:
                    raise ValueError("Shared groups across categories require a joint cluster bootstrap")
                grouping[key] = sample.category
    if isinstance(thresholds, DecisionThresholds):
        if len(categories) != 1:
            raise ValueError("Multiple categories require category-specific policies")
        thresholds = {categories[0]: thresholds}
        pixel_threshold = {categories[0]: pixel_threshold}
    if not isinstance(pixel_threshold, Mapping) or set(thresholds) != set(categories) or set(pixel_threshold) != set(categories):
        raise ValueError("Policies must cover exactly the reported categories")

    rngs = np.random.SeedSequence(seed).spawn(len(categories))
    per_category, draws = {}, {}
    for category, child_seed in zip(categories, rngs, strict=True):
        indices = [i for i, sample in enumerate(samples) if sample.category == category]
        selected = [samples[i] for i in indices]
        subset = _Subset(predictions, indices)
        labels = np.asarray([int(s.label == "anomaly") for s in selected])
        scores = np.asarray([p.image_score for p in subset], dtype=np.float64)
        policy = thresholds[category]
        image = image_level_metrics(labels, scores, policy.fail_threshold)
        units, bootstrap_protocol = _bootstrap_units(selected)
        draws[category] = _rank_bootstrap(labels, scores, units, bootstrap_samples,
                                         np.random.default_rng(child_seed))
        per_category[category] = {
            "sample_count": len(selected), "threshold_policy": policy.as_dict(),
            "image_level": image,
            "pixel_level": evaluate_pixels(selected, subset, pixel_threshold[category]),
            "review_system": review_system_metrics(labels, scores, policy, costs=costs),
            "coverage_risk_curve": coverage_risk_curve(labels, scores, policy.fail_threshold, costs=costs),
            "confidence_intervals_95": {name: _interval(values, bootstrap_samples)
                                        for name, values in draws[category].items()},
            "bootstrap_protocol": bootstrap_protocol,
        }

    macro = {"method": "equal_weight_over_defined_categories", "category_count": len(categories),
             "coverage": {}}
    for section, names in (("image_level", IMAGE_MEASURES), ("review_system", REVIEW_MEASURES),
                           ("pixel_level", PIXEL_MEASURES)):
        macro[section], macro["coverage"][section] = {}, {}
        for name in names:
            included = [c for c in categories if (per_category[c][section] or {}).get(name) is not None]
            macro[section][name] = (float(sum(per_category[c][section][name] / len(included) for c in included))
                                     if included else None)
            macro["coverage"][section][name] = {"included": included,
                                                 "excluded": [c for c in categories if c not in included]}
    macro["confidence_intervals_95"] = {}
    for name in ("roc_auc", "average_precision"):
        included = macro["coverage"]["image_level"][name]["included"]
        # Keep the estimand fixed: discard a replicate if any included category
        # is undefined, rather than silently changing the categories averaged.
        values = np.mean([draws[c][name] for c in included], axis=0) if included else []
        macro["confidence_intervals_95"][name] = _interval(values, bootstrap_samples)
    result = {
        "evaluation_schema_version": 2, "per_category": per_category, "macro": macro,
        "cost_model": costs.as_dict() if costs else None,
        "uncertainty": {"method": "percentile_bootstrap_within_fixed_categories", "seed": seed,
                        "confidence": .95, "resamples": bootstrap_samples,
                        "scope": "image_roc_auc_and_average_precision_only",
                        "limitation": "Conditional on fitted weights and frozen policies; excludes training/calibration uncertainty and unseen categories."},
    }
    for name in ("image_level", "pixel_level", "review_system", "coverage_risk_curve", "confidence_intervals_95"):
        result[name] = per_category[categories[0]][name] if len(categories) == 1 else None
    return result
