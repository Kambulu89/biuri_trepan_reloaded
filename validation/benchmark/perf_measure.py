"""Medição de desempenho ISOLADA e SEQUENCIAL (baseline vs otimizado): wall-clock, tempo de CPU e pico de RAM por execução.

Não é um resultado científico. Cada execução corre num processo próprio, SEM concorrência com outras, com o mesmo hardware, mesmas variáveis
de threads, mesmo dataset e seed. ``os.wait4`` devolve o rusage EXATO do processo filho (utime/stime/ru_maxrss).

    python -m validation.benchmark.perf_measure --dataset iris --seed 42 --baseline-root /path/wt_base --optimized-root . \
        --order baseline optimized optimized baseline --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def run_once(root: Path, dataset: str, seed: int, out: Path) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "validation.benchmark.run", "--unit", dataset, str(seed), "--out", str(out)]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=str(root), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _pid, status, ru = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - t0
    meta_path = out / "raw" / dataset / f"seed{seed}" / "RUN_META.txt"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    timing = meta.get("timing", {})
    return {"root": str(root), "returncode": os.waitstatus_to_exitcode(status), "wall_s": wall, "cpu_user_s": ru.ru_utime, "cpu_sys_s": ru.ru_stime,
            "cpu_total_s": ru.ru_utime + ru.ru_stime, "peak_rss_mb": ru.ru_maxrss / 1024.0, "tuning_s": timing.get("tuning_time"),
            "unit_total_s": timing.get("total_time"), "tuning_share": (timing["tuning_time"] / timing["total_time"]) if timing.get("total_time") else None}


def environment() -> Dict[str, Any]:
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    return {"cpu_model": cpu, "cpu_count": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0)), "platform": platform.platform(),
            "python": sys.version.split()[0], "threads_env": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "loadavg_before": os.getloadavg()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline-root", required=True); ap.add_argument("--optimized-root", required=True)
    ap.add_argument("--order", nargs="+", default=["baseline", "optimized"], choices=["baseline", "optimized"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    roots = {"baseline": Path(a.baseline_root), "optimized": Path(a.optimized_root)}
    out = Path(a.out)
    record: Dict[str, Any] = {"dataset": a.dataset, "seed": a.seed, "environment": environment(), "runs": []}
    for k, label in enumerate(a.order):
        run = run_once(roots[label], a.dataset, a.seed, out / f"{label}_{k}")
        run.update(label=label, order_index=k, loadavg_after=os.getloadavg())
        record["runs"].append(run)
        print(json.dumps({x: run[x] for x in ("label", "wall_s", "cpu_total_s", "peak_rss_mb", "tuning_share", "returncode")}), flush=True)
        (out / f"PERF_{a.dataset}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
