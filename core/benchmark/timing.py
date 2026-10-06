"""Instrumentação de tempo por estágio e contagens de trabalho (engenharia: não altera nenhum resultado científico)."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Iterator

ARM_TIME_KEY = {"mlp_original": "mlp_arm_time", "c45": "c45_time", "trepan_original": "original_time", "reloaded_core": "reloaded_core_time",
                "reloaded_owl_full": "reloaded_owl_time", "reloaded_owl_shuffled": "reloaded_shuffled_time"}


class StageTimer:
    """Acumula segundos por estágio (``with timer.stage('x'):``)."""

    def __init__(self) -> None:
        self._t: Dict[str, float] = defaultdict(float)
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._t[name] += time.perf_counter() - t0

    def add(self, name: str, seconds: float) -> None:
        self._t[name] += float(seconds)

    def report(self) -> Dict[str, Any]:
        out = {k: float(v) for k, v in self._t.items()}
        out["total_time"] = time.perf_counter() - self._start
        return out
