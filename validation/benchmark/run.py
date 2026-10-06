"""Executor do benchmark (uma unidade = dataset × master seed, em processo próprio; retomável e sem sobrescrever).

    python -m validation.benchmark.run --smoke  --out results/benchmark_protocol_smoke     # 1 seed (42) × 4 datasets reais
    python -m validation.benchmark.run --main   --out results/benchmark_main               # 5 master seeds × 4 datasets reais
    python -m validation.benchmark.run --controlled --out results/benchmark_controlled     # sintéticos (validação controlada)
    python -m validation.benchmark.run --unit digits 42 --out DIR                          # uma unidade
"""
from __future__ import annotations

import os

# Reprodutibilidade numérica entre máquinas (engenharia, não altera o desenho científico): fixa o kernel do OpenBLAS, um único
# thread de BLAS e desativa as rotas SIMD AVX-512 do NumPy, que dependem do CPU do anfitrião e mudam o último bit dos pesos do MLP.
for _k, _v in (("OPENBLAS_CORETYPE", "Haswell"), ("OPENBLAS_NUM_THREADS", "1"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
               ("NPY_DISABLE_CPU_FEATURES", "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_KNL AVX512_KNM")):
    os.environ.setdefault(_k, _v)

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from validation.benchmark import datasets as ds_mod
from validation.benchmark import manifest as mf
from validation.benchmark.raw_store import unit_complete, write_unit


def benchmark_config(seed: int):
    """Configuração ÚNICA para todos os datasets: sem hiperparâmetros por dataset (a estrutura vem do tuning congelado)."""
    from core.benchmark.runner import BenchmarkConfig
    return BenchmarkConfig(seeds=(int(seed),), scheme=mf.EVALUATION["scheme"], test_size=mf.EVALUATION["test_size"], extra_ablations=False,
                           different_oracle_experiment=False, structure_tuning=True, n_boot=10000)


def run_unit(dataset_id: str, seed: int, out: Path, manifest_path: Optional[Path] = None) -> Path:
    from core.benchmark.runner import BenchmarkRunner
    manifest_path = manifest_path or mf.latest_manifest_path()
    verification = mf.verify_manifest(manifest_path, dataset_ids=[dataset_id])
    if not verification["ok"]:
        raise SystemExit(f"Manifesto inválido: {verification['problems']}")
    if seed not in mf.MASTER_SEEDS:
        raise SystemExit(f"Seed {seed} não pertence às master seeds congeladas {mf.MASTER_SEEDS}.")
    spec = ds_mod.REGISTRY[dataset_id]
    ds, _target, _classes = spec.load()
    cfg = benchmark_config(seed)
    arms = [a for a in __import__("core.benchmark.runner", fromlist=["x"]).default_arms(cfg) if a.group in mf.MAIN_ARM_GROUPS]
    t0 = time.perf_counter()
    result = BenchmarkRunner(cfg, arms).run(ds, str(spec.ontology_path) if spec.ontology_path else None)
    sha = verification["manifest_sha256"]
    t_ser = time.perf_counter()
    write_unit(result, out, manifest_sha256=sha, labels=list(range(len(_classes))))
    serialization = time.perf_counter() - t_ser
    split_report = result.semantic_report[0]
    timing = dict(split_report.get("timing", {}), serialization_time=serialization)
    timing["total_time"] = time.perf_counter() - t0
    meta = {"dataset": dataset_id, "seed": seed, "seconds": timing["total_time"], "frozen_code_drift": verification["frozen_code_drift"],
            "run_code_commit": verification["current_code_commit"], "code_dirty": bool(mf._git("status", "--porcelain", "--untracked-files=no")),
            "manifest_used": manifest_path.name, "tuning_execution_count": split_report.get("tuning_execution_count"),
            "tuning_engineering_profile": (split_report.get("structural_protocol") or {}).get("engineering_profile"),
            "status": "completed", "timing": timing,
            "numeric_environment": {k: os.environ.get(k) for k in ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NPY_DISABLE_CPU_FEATURES")},
            "work": split_report.get("work", {})}
    (out / "raw" / dataset_id / f"seed{seed}" / "RUN_META.txt").write_text(json.dumps(meta), encoding="utf-8")
    return out


def _write_status(out: Path, dataset: str, seed: int, status: str, **extra) -> None:
    d = out / "status"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{dataset}_seed{seed}.json").write_text(json.dumps({"dataset": dataset, "seed": seed, "status": status, **extra}), encoding="utf-8")


def _spawn(units, out: Path, workers: int, max_wall_time_per_unit: Optional[float] = None) -> int:
    """Supervisor. ``max_wall_time_per_unit`` (segundos) é um limite OPERACIONAL global, igual para todos os datasets: uma unidade que o
    ultrapasse é terminada e marcada ``resource_limit_exceeded``. A configuração científica NUNCA é alterada para a fazer terminar."""
    logs = out / "logs"; logs.mkdir(parents=True, exist_ok=True)
    pending, running, failed = [u for u in units if not unit_complete(out, *u)], [], 0
    while pending or running:
        while pending and len(running) < workers:
            ds, seed = pending.pop(0)
            log = open(logs / f"{ds}_seed{seed}.log", "w")
            p = subprocess.Popen([sys.executable, "-m", "validation.benchmark.run", "--unit", ds, str(seed), "--out", str(out)],
                                 stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
            running.append((p, ds, seed, log, time.time()))
            print(f"[run] iniciada {ds} seed={seed}", flush=True)
        for item in list(running):
            p, ds, seed, log, t0 = item
            if p.poll() is None and max_wall_time_per_unit is not None and time.time() - t0 > max_wall_time_per_unit:
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                log.close(); running.remove(item); failed += 1
                _write_status(out, ds, seed, "resource_limit_exceeded", limit_s=max_wall_time_per_unit, elapsed_s=time.time() - t0,
                              note="unidade terminada pelo limite operacional global; a configuração científica não foi alterada")
                print(f"[run] resource_limit_exceeded {ds} seed={seed} (> {max_wall_time_per_unit:.0f}s)", flush=True)
            elif p.poll() is not None:
                log.close(); running.remove(item)
                failed += int(p.returncode != 0)
                _write_status(out, ds, seed, "completed" if p.returncode == 0 else "failed", elapsed_s=time.time() - t0, returncode=p.returncode)
                print(f"[run] {'OK' if p.returncode == 0 else 'FALHOU'} {ds} seed={seed} ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(2)
    return failed


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true"); g.add_argument("--main", action="store_true")
    g.add_argument("--controlled", action="store_true"); g.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"))
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-wall-time-per-unit", type=float, default=None,
                    help="limite operacional global (s) por unidade dataset×seed; omissão = sem limite. Excedê-lo marca resource_limit_exceeded")
    a = ap.parse_args()
    out = Path(a.out)
    if a.unit:
        run_unit(a.unit[0], int(a.unit[1]), out)
        return
    if a.smoke:
        units = [(d, mf.MASTER_SEEDS[0]) for d in ds_mod.MAIN_DATASETS]
    elif a.main:
        units = [(d, s) for s in mf.MASTER_SEEDS for d in ds_mod.MAIN_DATASETS]
    else:
        units = [(d, s) for s in mf.MASTER_SEEDS for d in ds_mod.CONTROLLED_DATASETS]
    sys.exit(1 if _spawn(units, out, a.workers, a.max_wall_time_per_unit) else 0)


if __name__ == "__main__":
    main()
