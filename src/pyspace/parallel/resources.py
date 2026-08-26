"""Opt-in resource snapshots; importing PySpace performs no hardware probe."""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class ResourceSnapshot:
    resident_bytes: int
    virtual_bytes: int
    process_cpu_percent: float


def current_process_resources() -> ResourceSnapshot:
    """Measure the current process when explicitly called."""
    process = psutil.Process()
    memory = process.memory_info()
    return ResourceSnapshot(
        resident_bytes=int(memory.rss),
        virtual_bytes=int(memory.vms),
        process_cpu_percent=float(process.cpu_percent(interval=None)),
    )


__all__ = ["ResourceSnapshot", "current_process_resources"]
