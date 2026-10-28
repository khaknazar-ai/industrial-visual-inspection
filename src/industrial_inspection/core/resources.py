"""Measure one workload; RSS peaks are sampled, CUDA peaks use allocator counters."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psutil


@contextmanager
def measure_resources(
    *, cuda: bool | str = False, interval_seconds: float = 0.05
) -> Iterator[dict[str, Any]]:
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("RSS sampling interval must be positive")
    process = psutil.Process()
    result: dict[str, Any] = {
        "ram_before_bytes": process.memory_info().rss,
        "ram_sampling_interval_seconds": interval_seconds,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    result["ram_peak_sampled_bytes"] = result["ram_before_bytes"]
    torch = None
    cuda_device = cuda if isinstance(cuda, str) else None
    if cuda:
        import torch

        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(interval_seconds):
            result["ram_peak_sampled_bytes"] = max(
                result["ram_peak_sampled_bytes"], process.memory_info().rss
            )

    sampler = threading.Thread(target=sample, name="workload-rss", daemon=True)
    started = time.perf_counter()
    sampler.start()
    try:
        yield result
    finally:
        try:
            if torch is not None:
                torch.cuda.synchronize(cuda_device)
                result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(cuda_device)
                result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(cuda_device)
        finally:
            result["duration_seconds"] = time.perf_counter() - started
            stop.set()
            sampler.join()
            result["ram_after_bytes"] = process.memory_info().rss
            result["ram_peak_sampled_bytes"] = max(
                result["ram_peak_sampled_bytes"], result["ram_after_bytes"]
            )
