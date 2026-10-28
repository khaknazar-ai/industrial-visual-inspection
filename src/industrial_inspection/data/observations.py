"""Disk journals of samples actually read by training and normalization loaders."""

from __future__ import annotations

import json
import os
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from industrial_inspection.core.reproducibility import sha256_file
from industrial_inspection.data.manifest_datamodule import ManifestSplits

_INPUT_SCOPE: ContextVar = ContextVar("observed_model_input", default=None)
_EVENT_SPLITS = {
    "read": None, "training_batch": "train", "validation_batch": "validation",
    "fit_input": "train", "calibration_input": "validation",
}


@contextmanager
def observe_model_inputs(observations, *, stage: str):
    """Observe synchronous model file reads without treating them as optimizer batches."""
    if stage not in {"fit_input", "calibration_input"}:
        raise ValueError("Unknown synchronous observation stage")
    token = _INPUT_SCOPE.set((observations, stage))
    try:
        with observations.inputs.scope():
            yield
    finally:
        _INPUT_SCOPE.reset(token)


def observed_input_scope():
    return _INPUT_SCOPE.get()


class SampleObservations:
    """One append-only journal per worker process, with verified content identities.

    Read events include prefetched samples. They are not claimed to be optimizer
    steps; the separate batch events record what Lightning actually received.
    """

    def __init__(self, splits: ManifestSplits, directory: Path) -> None:
        self.directory = directory
        self.fingerprint = splits.fingerprint
        self.inputs = splits.inputs
        self.expected = {
            str(sample.image_path.resolve()): {"sample_id": sample.sample_id, "split": sample.split,
                                             "sha256": splits.inputs.file_hash(sample.image_path)}
            for sample in splits.train + splits.validation + splits.test
        }

    def verify(self, image_path: str | Path) -> dict:
        path = Path(image_path).resolve()
        expected = self.expected.get(str(path))
        if expected is None or expected["split"] == "test":
            raise ValueError(f"Training attempted to read a test or undeclared image: {path}")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Training image changed after split freeze: {path}")
        return expected

    def record(self, image_path: str | Path, event: str) -> None:
        expected = self.verify(image_path)
        if event not in _EVENT_SPLITS:
            raise ValueError(f"Unknown observation event: {event}")
        required_split = _EVENT_SPLITS[event]
        if required_split and expected["split"] != required_split:
            raise ValueError(f"{event} consumed a sample from {expected['split']}")
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / f"worker-{os.getpid()}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**expected, "event": event}, sort_keys=True) + "\n")

    def write_report(self, output: Path) -> dict:
        expected_by_id = {row["sample_id"]: row for row in self.expected.values()}
        seen: dict[str, dict[str, str]] = {}
        counts: Counter = Counter()
        for journal in sorted(self.directory.glob("worker-*.jsonl")):
            with journal.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    expected = expected_by_id.get(record["sample_id"])
                    if expected is None or record["sha256"] != expected["sha256"] or record["split"] != expected["split"] or record["split"] == "test":
                        raise ValueError("Observation journal contains an undeclared, changed or test sample")
                    event = record["event"]
                    if event not in _EVENT_SPLITS or _EVENT_SPLITS[event] not in {None, record["split"]}:
                        raise ValueError("Observation journal contains an invalid event/split")
                    key = f"{record['event']}:{record['split']}"
                    seen.setdefault(key, {})[record["sample_id"]] = record["sha256"]
                    counts[key] += 1
        if not counts["training_batch:train"] and not counts["fit_input:train"]:
            raise ValueError("No training batches or synchronous fit inputs were observed")
        report = {
            "schema_version": 1, "split_fingerprint": self.fingerprint,
            "events": {key: {"event_count": counts[key], "unique_count": len(rows), "sample_sha256": dict(sorted(rows.items()))}
                       for key, rows in sorted(seen.items())},
            "test_seen_during_fit": False,
            "read_scope": "dataset reads, including worker prefetch and model normalization",
            "batch_scope": "Lightning training/validation batch start callbacks",
            "synchronous_scope": "fit_input/calibration_input: successfully decoded files in model calls",
        }
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report


def observation_callback(observations: SampleObservations):
    from lightning.pytorch import Callback

    class ObservedBatches(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            for path in batch.image_path:
                observations.record(path, "training_batch")

        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            for path in batch.image_path:
                observations.record(path, "validation_batch")

    return ObservedBatches()
