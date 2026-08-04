from __future__ import annotations

from types import SimpleNamespace

from pyspace.parallel import current_process_resources
from pyspace.parallel import resources as resource_module


def test_process_resources_are_measured_only_when_requested(monkeypatch) -> None:
    calls: list[str] = []

    class FakeProcess:
        def memory_info(self) -> SimpleNamespace:
            calls.append("memory")
            return SimpleNamespace(rss=10, vms=20)

        def cpu_percent(self, *, interval: None) -> float:
            calls.append("cpu")
            return 3.5

    monkeypatch.setattr(resource_module.psutil, "Process", FakeProcess)

    assert calls == []
    snapshot = current_process_resources()
    assert snapshot == resource_module.ResourceSnapshot(10, 20, 3.5)
    assert calls == ["memory", "cpu"]
