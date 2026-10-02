"""
Logs de performance do pipeline BIURI.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class PerformanceLogger:
    """Regista tempos, trials, cache e timeouts por etapa."""

    def __init__(self, dataset_name: str = "unknown", preset_key: str = "scientific"):
        self.dataset_name = dataset_name
        self.preset_key = preset_key
        self._stages: List[Dict[str, Any]] = []
        self._t0 = time.perf_counter()
        self.used_cache = False
        self.stopped_by_timeout = False
        self.stopped_by_cancel = False
        self.stopped_by_early_fidelity = False

    def start_stage(self, name: str, **meta) -> float:
        return time.perf_counter()

    def end_stage(
        self,
        name: str,
        t_start: float,
        *,
        n_features: Optional[int] = None,
        n_trials: Optional[int] = None,
        best_score: Optional[float] = None,
        used_cache: bool = False,
        timeout: bool = False,
        early_stop: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        elapsed = time.perf_counter() - t_start
        row = {
            "stage": name,
            "elapsed_seconds": round(elapsed, 3),
            "n_features": n_features,
            "n_trials": n_trials,
            "best_score": best_score,
            "used_cache": used_cache,
            "timeout": timeout,
            "early_stop": early_stop,
        }
        if extra:
            row.update(extra)
        self._stages.append(row)
        if used_cache:
            self.used_cache = True
        if timeout:
            self.stopped_by_timeout = True
        if early_stop:
            self.stopped_by_early_fidelity = True

        print(f"\n[PERFORMANCE] {name}")
        print(f"  tempo: {elapsed:.2f}s")
        if n_features is not None:
            print(f"  features: {n_features}")
        if n_trials is not None:
            print(f"  trials: {n_trials}")
        if best_score is not None:
            print(f"  melhor score: {best_score:.4f}")
        if used_cache:
            print("  cache: sim")
        if timeout:
            print("  parou por: timeout")
        if early_stop:
            print("  parou por: fidelidade antecipada")

    def mark_cancelled(self) -> None:
        self.stopped_by_cancel = True

    def summary(self) -> Dict[str, Any]:
        total = time.perf_counter() - self._t0
        return {
            "dataset_name": self.dataset_name,
            "preset": self.preset_key,
            "total_seconds": round(total, 3),
            "used_cache": self.used_cache,
            "stopped_by_timeout": self.stopped_by_timeout,
            "stopped_by_cancel": self.stopped_by_cancel,
            "stopped_by_early_fidelity": self.stopped_by_early_fidelity,
            "stages": self._stages,
            "timestamp": datetime.now().isoformat(),
        }

    def save(self) -> Path:
        out = Path(__file__).resolve().parent.parent / "results" / "performance_logs"
        out.mkdir(parents=True, exist_ok=True)
        safe = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in self.dataset_name
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out / f"performance_{self.preset_key}_{safe}_{ts}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.summary(), fh, indent=2, ensure_ascii=False)
        print(f"[INFO] Log de performance guardado em: {path}")
        return path
