"""Condutor da campanha M29: lança UMA unidade por processo (sem duplicar unidades completas), mede o CPU REAL de cada processo (``os.wait4``: utilizador+sistema, distinto da soma
de tempos de parede), mantém o livro de CPU acumulado e PARA antes de ultrapassar o teto (``--limit-hours``). Para na primeira falha (código ≠ 0, p.ex. comparabilidade em execução).
Retoma: salta unidades com ``UNIT_SHA256.json``. As estimativas por unidade (``--estimates``) só servem à guarda do orçamento.
    python -m validation.benchmark.m29_drive --out DIR --reuse-checkpoint-from DIR --units iris:42,wine:42 [--parallel 3] [--limit-hours 2.5] [--prior-cpu-hours X] [--dry-run]"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EST = {"iris": 47.0, "wine": 180.0, "breast_cancer": 390.0, "digits": 1550.0}          # s de CPU por unidade (M29 + G29), do Doc. 29 §10 acrescido do custo medido-estimado de G29


def unit_done(out: Path, ds: str, seed: str) -> bool:
    return (out / "raw" / ds / f"seed{seed}" / "UNIT_SHA256.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse-checkpoint-from", required=True)
    ap.add_argument("--sc-dir", default=None)
    ap.add_argument("--units", required=True)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--limit-hours", type=float, default=2.5)
    ap.add_argument("--prior-cpu-hours", type=float, default=0.0, help="CPU já gasto pela campanha (p.ex. dry-run medido)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-kinds", default=None)
    ap.add_argument("--estimates", default=None, help="JSON {dataset: segundos de CPU por unidade}")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    est = {**DEFAULT_EST, **(json.loads(a.estimates) if a.estimates else {})}
    ledger = out / "cpu_ledger.jsonl"
    units = [tuple(u.split(":")) for u in a.units.split(",")]
    lock = threading.Lock()
    state = {"cpu": a.prior_cpu_hours * 3600.0, "reserved": 0.0, "stop": False, "failed": []}
    limit = a.limit_hours * 3600.0

    def run(u):
        ds, sd = u
        if not a.dry_run and unit_done(out, ds, sd):
            return
        e = est[ds] * (0.3 if a.dry_run else 1.0)
        with lock:
            if state["stop"]:
                return
            if state["cpu"] + state["reserved"] + e > limit:
                state["stop"] = True
                print(f"[guarda de orçamento] {ds}:{sd} não lançada: CPU {state['cpu'] / 3600:.3f} h + reservado {state['reserved'] / 3600:.3f} h + estimativa {e / 3600:.3f} h > {a.limit_hours} h", flush=True)
                return
            state["reserved"] += e
        cmd = [sys.executable, "-m", "validation.benchmark.run_controls_m29", "--unit", ds, sd, "--out", str(out)] + (["--dry-run"] if a.dry_run else ["--reuse-checkpoint-from", a.reuse_checkpoint_from])
        if a.sc_dir and not a.dry_run:
            cmd += ["--sc-dir", a.sc_dir]
        if a.only_kinds and not a.dry_run:
            cmd += ["--only-kinds", a.only_kinds]
        log = open(out / f"{ds}_seed{sd}.log", "w")
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        _, status, ru = os.wait4(p.pid, 0)
        rc = os.waitstatus_to_exitcode(status)
        cpu = ru.ru_utime + ru.ru_stime
        with lock:
            state["reserved"] -= e
            state["cpu"] += cpu
            rec = {"unit": f"{ds}:{sd}", "rc": rc, "wall_s": round(time.time() - t0, 1), "cpu_user_s": round(ru.ru_utime, 1), "cpu_sys_s": round(ru.ru_stime, 1), "cpu_s": round(cpu, 1),
                   "cpu_over_wall": round(cpu / max(1e-9, time.time() - t0), 2), "estimate_s": e, "cumulative_cpu_h": round(state["cpu"] / 3600.0, 4), "dry_run": a.dry_run}
            with open(ledger, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if rc != 0:
                state["stop"] = True
                state["failed"].append(rec["unit"])

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, a.parallel)) as ex:
        list(ex.map(run, units))
    print(json.dumps({"cumulative_cpu_h": round(state["cpu"] / 3600.0, 4), "limit_h": a.limit_hours, "failed": state["failed"], "stopped_early": state["stop"]}), flush=True)
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
