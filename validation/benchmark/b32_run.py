"""B3.2 (Doc. 38): calibração (SÓ Original e métricas de RECURSOS), congelamento, execução E1/E2/E3/memória. Só validation/ e sintéticos.
Calibração: (i) `max_features_per_node` do Original (taxa de recuperação mais próxima de 0,5 em [0,2; 0,8]; se nenhum, o mais próximo de 0,5, sinalizado); (ii) `beam_width` das
variantes genéricas H1 (candidatos avaliados ≈ braço C) e H2 (CPU de ajuste ≈ braço C) usando SÓ recursos (nunca recuperação, fidelidade ou accuracy)."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from validation.benchmark import b31_identifiability as B31
from validation.benchmark.b31_identifiability import ProblemSpec
from validation.benchmark.b32_comparison import (CAL32, CODE32, EVAL32, FROZEN32, MEM32, STAB32, code32, fit_arm, make_instance_mlp, reference_models)
from validation.benchmark.b32_comparison import make_instance as make_instance_cf

E_ARMS = ("A_ORIG", "B_WIDE", "C_OWL", "D_PERM", "E_RAND", "F1_CONV", "F2_SCREEN", "G_NEUTRALIZED", "H1_WIDE_BEAM_CAND", "H2_WIDE_BEAM_CPU", "N_IRRELEVANT", "N_INCOMPLETE", "N_WRONG_SPLIT")
E2_EXTRA = ("C_HIERARCHY",)
STAB_ARMS = ("A_ORIG", "B_WIDE", "C_OWL", "D_PERM", "E_RAND", "G_NEUTRALIZED")
STAB_CONFIGS = [dict(n_groups=6, group_size=4, rule_m=2), dict(n_groups=8, group_size=3, rule_m=2), dict(n_groups=8, group_size=5, rule_m=3), dict(n_groups=6, group_size=6, rule_m=3)]
MF_GRID = (16, 20, 24, 28, 30)
BW_GRID = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)

CRITERIA32 = {
    "version": 1,
    "unit_of_analysis": "problema sintético (P1', P2'); sementes de avaliação 200–239; comparações emparelhadas por semente; ICs bootstrap (4000) das diferenças médias",
    "resource_equivalence_contract": {
        "same": "dados, MLP (oráculo), instâncias de treino/teste, semente, min_sample, max_queries, max_nodes/depth, max_n, min_samples_leaf",
        "comparison_1_equal_candidates": "C, D, E, F1, F2, G (e negativos) fazem o MESMO nº de pesquisas por unidade (≤ 3) sobre unidades de igual tamanho; H1 iguala os candidatos AVALIADOS (`eval_calls`) de C pelo `beam_width` calibrado (só recursos). Reporta-se a razão efetiva de `eval_calls`; NÃO se afirma igualdade das restantes dimensões",
        "comparison_2_equal_cpu_queries": "H2 iguala o CPU de AJUSTE (`process_time`) de C (calibrado só em CPU); consultas e amostras ao oráculo reportadas à parte (diferem só pela forma da árvore); compilação da ontologia (reasoner Java) reportada em separado",
        "not_equalised": "memória (medida em subconjunto), complexidade da árvore e candidatos GERADOS não são igualados; são reportados",
    },
    "ontological_advantage": {
        "O1_relation_specific": "recovery(C) − max(recovery(D), recovery(E), recovery(F1), recovery(F2), recovery(G)) ≥ 0.20",
        "O2_vs_generic_equal_candidates": "fidelity(C) − fidelity(H1) ≥ +0.005 com limite inferior do IC95 > 0, OU recovery(C) − recovery(H1) ≥ 0.10",
        "O3_vs_generic_equal_cpu": "idem contra H2",
        "O4_vs_wide": "pelo menos um de: (a) fidelity(C) − fidelity(B) ≥ +0.005 com IC95 inferior > 0; (b) cpu_fit(C)/cpu_fit(B) ≤ 0.8 e recovery(C) ≥ recovery(B) − 0.05; (c) nós internos(C) ≤ 0.8·nós internos(B) com fidelity(C) ≥ fidelity(B) − 0.005; (d) DP da fidelidade(C) ≤ 0.8·DP(B) com fidelity(C) ≥ fidelity(B) − 0.005",
        "verdicts": {"ONTOLOGICAL_ADVANTAGE_OVER_GENERIC": "O1 e O2 e O3 e O4", "RELATION_SPECIFIC_ONLY": "O1 mas não todos de O2–O4", "NO_ONTOLOGICAL_ADVANTAGE": "O1 falha"},
        "note": "uma vantagem sobre A (Original limitado) isolada NÃO conta se WIDE iguala a recuperação com menos recursos (pedido B3.2 §5)",
    },
    "hierarchy_exploratory": "por problema: Acc(C) > Acc(A) > Acc(C4.5) e Fid(C) > Fid(A), Fid(C) > Fid(C4.5); reportado como exploratório (C = Original + orientação, não o Reloaded)",
    "stability_E3": "recuperação de A, B, C, D, E, G em 4 configurações genéricas distintas (n_groups, group_size, rule_m) com mf = round(0.875·p), 20 sementes novas (300–319), oráculo de forma fechada",
    "memory": "tracemalloc em 10 sementes (200–209) para A, B, C, H2",
}


def calibrate_mf(base: ProblemSpec) -> Dict[str, Any]:
    rates = {}
    with tempfile.TemporaryDirectory() as td:
        for mf in MF_GRID:
            sp = ProblemSpec(**{**vars_of(base), "max_features_per_node": mf})
            rates[mf] = float(np.mean([fit_arm(sp, make_instance_mlp(sp, s), "A_ORIG", s, Path(td))["recovered"] for s in CAL32]))
    ok = [(abs(r - 0.5), mf) for mf, r in rates.items() if 0.2 <= r <= 0.8]
    if ok:
        return {"rates": rates, "chosen": min(ok)[1], "within_target_band": True}
    best = min((abs(r - 0.5), mf) for mf, r in rates.items())[1]
    return {"rates": rates, "chosen": best, "within_target_band": False, "note": "nenhum mf com recuperação em [0,2; 0,8]; escolhido o mais próximo de 0,5"}


def vars_of(sp: ProblemSpec) -> Dict[str, Any]:
    from dataclasses import asdict
    return asdict(sp)


def calibrate_bw(spec: ProblemSpec, seeds=CAL32[:15]) -> Dict[str, Any]:
    """Usa SÓ recursos (eval_calls, cpu_fit) em sementes de calibração; nunca recuperação/fidelidade/accuracy."""
    out: Dict[str, Any] = {"seeds": [seeds[0], seeds[-1]]}
    with tempfile.TemporaryDirectory() as td:
        insts = {s: make_instance_mlp(spec, s) for s in seeds}
        c = [fit_arm(spec, insts[s], "C_OWL", s, Path(td)) for s in seeds]
        ref = {"eval_calls": float(np.mean([r["eval_calls"] for r in c])), "cpu_fit_s": float(np.mean([r["cpu_fit_s"] for r in c]))}
        out["reference_C"] = ref
        table = {}
        for bw in BW_GRID:
            rr = [fit_arm(spec, insts[s], "H1_WIDE_BEAM_CAND", s, Path(td), bw_h1=bw) for s in seeds]
            table[bw] = {"eval_calls": float(np.mean([r["eval_calls"] for r in rr])), "cpu_fit_s": float(np.mean([r["cpu_fit_s"] for r in rr]))}
        out["table"] = table
        out["bw_h1"] = min(BW_GRID, key=lambda b: abs(np.log(table[b]["eval_calls"] / ref["eval_calls"])))
        out["bw_h2"] = min(BW_GRID, key=lambda b: abs(np.log(table[b]["cpu_fit_s"] / ref["cpu_fit_s"])))
    return out


def freeze() -> Dict[str, Any]:
    fz1 = json.loads(B31.FROZEN_V2.read_text())
    base = {"P1prime_within_group": ProblemSpec("P1prime_within_group", rule="within_group"), "P2prime_cross_group": ProblemSpec("P2prime_cross_group", rule="cross_group")}
    probs, cal = {}, {}
    for k, sp in base.items():
        m = calibrate_mf(sp)
        spec = ProblemSpec(**{**vars_of(sp), "max_features_per_node": m["chosen"]})
        b = calibrate_bw(spec)
        probs[k] = vars_of(spec)
        cal[k] = {"mf": m, "beam": b}
    doc = {"frozen": True, "purpose": "congelado ANTES de avaliar qualquer braço nas sementes de avaliação", "problems": probs, "calibration": cal,
           "arms": {"E1_E2": list(E_ARMS), "E2_extra": list(E2_EXTRA), "stability": list(STAB_ARMS)}, "stability_configs": STAB_CONFIGS,
           "bw_h1": {k: cal[k]["beam"]["bw_h1"] for k in cal}, "bw_h2": {k: cal[k]["beam"]["bw_h2"] for k in cal},
           "seeds": {"calibration": [CAL32[0], CAL32[-1]], "evaluation": [EVAL32[0], EVAL32[-1]], "stability": [STAB32[0], STAB32[-1]], "memory": [MEM32[0], MEM32[-1]]},
           "criteria": CRITERIA32, "code_sha256": code32(), "b31_v2_frozen_sha256": __import__("hashlib").sha256(B31.FROZEN_V2.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
           "oracle": "MLP sklearn (StandardScaler + MLPClassifier(32,), max_iter 800) treinado nos rótulos reais da regra (n_train=300); accuracy do MLP ≈ 0,80 (oráculo imperfeito, deliberado)",
           "calibration_used_only": "Original (recuperação) e recursos (eval_calls, CPU) — nunca desempenho dos braços semânticos"}
    FROZEN32.parent.mkdir(parents=True, exist_ok=True)
    FROZEN32.write_text(json.dumps(doc, indent=1, default=str))
    return doc


def _load() -> Dict[str, Any]:
    fz = json.loads(FROZEN32.read_text())
    drift = {f: 1 for f, h in fz["code_sha256"].items() if code32()[f] != h}
    if drift:
        raise SystemExit(f"código alterado desde o congelamento: {sorted(drift)}")
    return fz


def run_e(problem: str, out: Path, seeds=EVAL32, memory: bool = False) -> None:
    fz = _load()
    spec = ProblemSpec(**fz["problems"][problem])
    arms = list(E_ARMS) + (list(E2_EXTRA) if problem.startswith("P2") else [])
    if memory:
        arms = ["A_ORIG", "B_WIDE", "C_OWL", "H2_WIDE_BEAM_CPU"]
    rows: List[Dict[str, Any]] = []
    refs: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        for s in seeds:
            inst = make_instance_mlp(spec, s)
            refs.append(reference_models(spec, inst, s))
            for a in arms:
                rows.append(fit_arm(spec, inst, a, s, Path(td), bw_h1=fz["bw_h1"][problem], bw_h2=fz["bw_h2"][problem], measure_memory=memory))
    Path(out).write_text(json.dumps({"problem": problem, "rows": rows, "refs": refs, "frozen": FROZEN32.name}, indent=1))


def run_stability(out: Path, seeds=STAB32) -> None:
    fz = _load()
    res = []
    with tempfile.TemporaryDirectory() as td:
        for ci, cfg in enumerate(fz["stability_configs"]):
            p = cfg["n_groups"] * cfg["group_size"]
            spec = ProblemSpec(f"stab{ci}", n_groups=cfg["n_groups"], group_size=cfg["group_size"], rule="within_group", rule_m=cfg["rule_m"], max_features_per_node=int(round(0.875 * p)),
                               max_n=cfg["group_size"])
            for s in seeds:
                inst = make_instance_cf(spec, s)
                for a in fz["arms"]["stability"]:
                    r = fit_arm(spec, inst, a, s, Path(td))
                    r["config"] = ci
                    res.append(r)
    Path(out).write_text(json.dumps({"configs": fz["stability_configs"], "rows": res}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    e = sub.add_parser("run")
    e.add_argument("--problem", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--memory", action="store_true")
    e.add_argument("--n-seeds", type=int, default=0)
    s = sub.add_parser("stability")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "freeze":
        d = freeze()
        print(json.dumps({k: {"mf": v["mf"], "beam": {x: v["beam"][x] for x in ("reference_C", "bw_h1", "bw_h2")}} for k, v in d["calibration"].items()}, indent=1, default=str))
    elif a.cmd == "run":
        seeds = (MEM32 if a.memory else EVAL32)
        run_e(a.problem, Path(a.out), seeds[:a.n_seeds] if a.n_seeds else seeds, a.memory)
    else:
        run_stability(Path(a.out))


if __name__ == "__main__":
    main()
