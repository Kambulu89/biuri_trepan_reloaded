"""Relatório RETROATIVO e só de leitura do contrato conjunto (core/hierarchy_contract.py) sobre brutos existentes. Não altera resultados nem baselines.
Para cada unidade compara um par (Reloaded+OWL, Original equivalente) com o C4.5 canónico nas MESMAS instâncias e contra o MESMO MLP:
accuracy (rótulos reais) e fidelidade ao MLP; precision_macro e macro_f1 são complementares. Réplicas: média por unidade (como no Doc. 26).
    python -m validation.benchmark.hierarchy_joint_report --dir RAW_DIR --pairs "RELOADED_ARM:ORIGINAL_ARM,..." [--sc-replicas] --out FILE
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from core.hierarchy_contract import evaluate_joint
from validation.benchmark.diag_b2_analysis import load

MAIN = ("accuracy_real_labels", "fidelity_to_oracle")
COMPL = ("precision_macro_real_labels", "macro_f1_real_labels")


def report(units: dict, pairs: list) -> dict:
    out = {"n_units": len(units), "definition": "U1 A_R>A_O; U2 A_O>A_C; U3 A_R>A_C; U4 F_R>F_O; U5 F_R>F_C (estrito; mesmas instâncias; mesmo MLP)", "pairs": {}}
    for rel, ori in pairs:
        modes, status, conds = Counter(), Counter(), Counter()
        deltas = {k: [] for k in ("acc_R-O", "acc_O-C", "acc_R-C", "fid_R-O", "fid_R-C")}
        compl = {m: {"R": [], "O": [], "C": []} for m in COMPL}
        n = 0
        for u, arms in units.items():
            if rel not in arms or ori not in arms or "c45" not in arms:
                continue
            R, O, C = arms[rel], arms[ori], arms["c45"]
            v = evaluate_joint(R.get(MAIN[0]), O.get(MAIN[0]), C.get(MAIN[0]), R.get(MAIN[1]), O.get(MAIN[1]), C.get(MAIN[1]))
            n += 1
            status[v["status"]] += 1
            modes[v["mode"]] += 1
            for k, val in v.get("conditions", {}).items():
                conds[k] += int(val)
            for k, val in v.get("deltas", {}).items():
                deltas[k].append(val)
            for m in COMPL:
                for tag, row in (("R", R), ("O", O), ("C", C)):
                    if row.get(m) is not None:
                        compl[m][tag].append(row[m])
        se = lambda x: float(np.std(x, ddof=1) / len(x) ** 0.5) if len(x) > 1 else None
        out["pairs"][f"{rel} vs {ori}"] = {"n": n, "status_counts": dict(status), "mode_counts": dict(modes), "condition_true_counts": dict(conds),
                                           "mean_deltas": {k: {"mean": float(np.mean(v)), "se": se(v)} for k, v in deltas.items() if v},
                                           "complementary_means": {m: {t: float(np.mean(v)) for t, v in d.items() if v} for m, d in compl.items()}}
    return out


def replica_mean_units(root: Path, cx: Path | None, arms_map: dict) -> dict:
    """Une réplicas: ``arms_map`` = {rótulo: (família, ...)} resolvido por sampler_c_analysis.replicas; devolve linhas sintéticas por unidade com a média das réplicas."""
    from validation.benchmark.sampler_c_analysis import replicas
    sc = {f"{k[0]}:{k[1]}": v for k, v in load(root).items()}
    cxd = {f"{k[0]}:{k[1]}": v for k, v in load(cx).items()} if cx else {}
    metrics = MAIN + COMPL
    units = {}
    for u, arms in sc.items():
        row = {"c45": arms.get("c45", {})}
        for fam in "ABCDEF":
            row[fam] = {}
            for m in metrics:
                v = replicas(arms, cxd.get(u), fam, m)
                if v:
                    row[fam][m] = float(np.mean(list(v.values())))
        units[u] = row
    return units


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cx")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--sc-replicas", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",")]
    units = replica_mean_units(Path(a.dir), Path(a.cx) if a.cx else None, {}) if a.sc_replicas else {f"{k[0]}:{k[1]}": v for k, v in load(Path(a.dir)).items()}
    r = report(units, pairs)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
