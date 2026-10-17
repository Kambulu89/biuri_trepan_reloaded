"""B3.3 (Doc. 41): calibração técnica ÚNICA (SÓ Original e recursos), congelamento, execução. Só validation/ e sintéticos."""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from validation.benchmark import b33_interactions as B

BW_GRID = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
UNIT_BUDGET = 3

CRITERIA33 = {
    "version": 1,
    "unit_of_inference": "família de problemas (I1–I6); 30 sementes de avaliação 400–429 por família; comparações emparelhadas por semente; IC95 bootstrap (4000) das diferenças médias",
    "families_roles": {"I1_AND_PAIR": "controlo com sinal univariado (não conta para o veredicto global)", "I2_XOR_NO_SIGNAL": "sem sinal univariado (limite indutivo esperado)",
                       "I3..I6": "interação com sinal univariado fraco (q=0,65): sem ruído, ruído de rótulo, hierarquia entre conceitos irmãos, espaço maior com orçamento mais apertado"},
    "expressiveness": "a XOR de um par NÃO é representável por UM teste m-of-n; é representável por uma árvore de 2 níveis (max_depth=3, max_nodes=7 chegam). «rule_recovery» = o par de atributos no mesmo teste OU no mesmo caminho raiz→folha (`recovered_path`)",
    "difficulty_definition": "só pelo Original (taxa de recuperação com `max_features_per_node` da grelha da família, mais próxima de 0,5 em [0,2; 0,8], senão a mais próxima) em sementes de calibração 3000–3019; a força do sinal univariado vem do gerador (quantil q), nunca de variantes ontológicas",
    "arms": {"A": "A_ORIG (Original canónico, orçamento limitado)", "B": "B_WIDE (Original com mf=p)", "C": "C_OWL (orientação por ontologia verdadeira da família)", "D": "D_PERM", "E": "E_RAND",
             "F": "F_CONV (engenharia convencional: agrupamento por correlação)", "G": "G_UNI_SCREEN (rastreio univariado)", "H": "H_PAIR_SCREEN (rastreio genérico de interações por pares; mesmo nº e tamanho de unidades)",
             "I": "I_NEUTRALIZED (ontologia verdadeira com as relações do par neutralizadas)", "J1": "WIDE com beam calibrado a candidatos avaliados de C (só recursos)", "J2": "WIDE com beam calibrado ao CPU de C (só recursos)",
             "N": "N_IRRELEVANT, N_INCOMPLETE (negativos)"},
    "resource_contract": "comparação 1 (igual nº de candidatos avaliados): C vs J1; comparação 2 (igual CPU): C vs J2; unidades de C–I com o mesmo nº/tamanho (≤3 pesquisas); reportam-se SEPARADAMENTE candidatos disponíveis (literais listados, gerados), examinados (eval_calls), consultas e amostras ao oráculo, CPU de ajuste, CPU de compilação/rastreio, custo total, memória (sementes 500–509), nós, profundidade, condições m-of-n. Não se alarga o espaço de candidatos para a ontologia; não se declara igualdade quando só uma dimensão é igual",
    "joint_gain": {"level": "Nível 1 (C vs A equivalente: mesmo oráculo, dados, partições, orçamento de ajuste, semente)",
                   "OBSERVED_JOINT_GAIN": "Δaccuracy(C−A) > 0 e Δfidelity(C−A) > 0 (médias)", "SUPPORTED_JOINT_GAIN": "ambos com limite inferior do IC95 > 0",
                   "TRADEOFF": "um > 0 e o outro < 0", "NO_JOINT_GAIN": "caso contrário"},
    "attribution": {"ATTRIBUTED": "SUPPORTED_JOINT_GAIN e, contra CADA controlo D, E, I (ontologia) e B, F, G, H, J1, J2 (genéricos), Δaccuracy>0 e Δfidelity>0 com IC95 inferior > 0",
                    "NOT_ATTRIBUTED": "caso contrário (não significa que os controlos refutem a ontologia: significa ausência de ganho direto atribuível)"},
    "recovery_advantage": "R1: recovery(C) − max(recovery de D, E, F, G, H, I) ≥ 0,20; R2: recovery(C) − recovery(B) ≥ 0,10 e recovery(C) − max(recovery(J1), recovery(J2)) ≥ 0,10",
    "cheaper_equivalent": "para cada controlo X: «SIMPLER_EQUIVALENT» se |Δaccuracy|≤0,005, |Δfidelity|≤0,005 face a C e custo total(X) ≤ 0,8·custo total(C) (ou X≥C em recovery com custo menor); reporta-se sempre",
    "family_verdicts": {"ONTOLOGICAL_ADVANTAGE": "R1 e R2 e ATTRIBUTED", "RELATION_SPECIFIC_ONLY": "R1 mas não (R2 e ATTRIBUTED)", "NO_ONTOLOGICAL_ADVANTAGE": "R1 falha",
                        "UNINFORMATIVE_NO_ARM_RECOVERS": "recovery de TODOS os braços < 0,10 (limite indutivo; não conta a favor nem contra)"},
    "global_verdict": {"B33_ADVANTAGE_ESTABLISHED": "≥ 2 das famílias I3–I6 com ONTOLOGICAL_ADVANTAGE", "B33_PARTIAL": "exatamente 1", "B33_NO_ONTOLOGICAL_ADVANTAGE": "0"},
    "hierarchy_exploratory": "por família: Acc(C) > Acc(A) > Acc(C4.5) e Fid(C) > Fid(A) e Fid(C) > Fid(C4.5); exploratório (C é Original + orientação, não o Reloaded integrado)",
    "no_claim": "nenhuma superioridade se declara sem evidência; DEV/confirmatórios não usados; sementes de avaliação não repetidas nem escolhidas",
}


def calibrate_family(key: str) -> Dict[str, Any]:
    fam = B.FAMILIES[key]
    rates: Dict[int, float] = {}
    with tempfile.TemporaryDirectory() as td:
        insts = {}
        for mf in fam.mf_grid:
            sp = B.spec_of(fam, mf)
            rr = []
            for s in B.CAL33:
                if s not in insts:
                    insts[s] = B.make_instance33(fam, B.spec_of(fam, fam.mf_grid[0]), s)
                rr.append(B.fit_arm33(fam, sp, insts[s], "A_ORIG", s, Path(td))["recovered_path"])
            rates[mf] = float(np.mean(rr))
        ok = [(abs(r - 0.5), mf) for mf, r in rates.items() if 0.2 <= r <= 0.8]
        chosen = min(ok)[1] if ok else min((abs(r - 0.5), mf) for mf, r in rates.items())[1]
        mfc = {"rates": rates, "chosen": chosen, "within_target_band": bool(ok)}
        spec = B.spec_of(fam, chosen)
        seeds = B.CAL33[:12]
        c = [B.fit_arm33(fam, spec, insts[s], "C_OWL", s, Path(td)) for s in seeds]
        ref = {"eval_calls": float(np.mean([r["eval_calls"] for r in c])), "cpu_fit_s": float(np.mean([r["cpu_fit_s"] for r in c]))}
        table = {}
        for bw in BW_GRID:
            rr = [B.fit_arm33(fam, spec, insts[s], "J1_WIDE_BEAM_CAND", s, Path(td), bw_j1=bw) for s in seeds]
            table[bw] = {"eval_calls": float(np.mean([r["eval_calls"] for r in rr])), "cpu_fit_s": float(np.mean([r["cpu_fit_s"] for r in rr]))}
        beam = {"reference_C": ref, "table": table, "bw_j1": min(BW_GRID, key=lambda b: abs(np.log(table[b]["eval_calls"] / ref["eval_calls"]))),
                "bw_j2": min(BW_GRID, key=lambda b: abs(np.log(table[b]["cpu_fit_s"] / ref["cpu_fit_s"])))}
    return {"family": asdict(fam), "mf": mfc, "beam": beam, "spec": asdict(spec)}


def freeze(parts: List[Path]) -> Dict[str, Any]:
    cal = {}
    for p in parts:
        d = json.loads(Path(p).read_text())
        cal[d["family"]["key"]] = d
    doc = {"frozen": True, "purpose": "congelado ANTES de avaliar qualquer braço nas sementes de avaliação 400–429", "families": cal, "arms": list(B.ARMS33), "unit_budget": UNIT_BUDGET,
           "seeds": {"calibration": [B.CAL33[0], B.CAL33[-1]], "evaluation": [B.EVAL33[0], B.EVAL33[-1]], "diagnostic_memory": [B.DIAG33[0], B.DIAG33[-1]]},
           "criteria": CRITERIA33, "code_sha256": B.code33(),
           "oracle": "MLP sklearn (StandardScaler + MLPClassifier(32,), max_iter 800) treinado nos rótulos (com ruído) da família; accuracy do MLP reportada por semente",
           "calibration_used_only": "Original (recuperação por caminho) e recursos (eval_calls, CPU) — nunca o desempenho dos braços semânticos"}
    B.FROZEN33.write_text(json.dumps(doc, indent=1, default=str))
    return doc


def _load() -> Dict[str, Any]:
    fz = json.loads(B.FROZEN33.read_text())
    cur = B.code33()
    drift = sorted(f for f, h in fz["code_sha256"].items() if cur.get(f) != h)
    if drift:
        raise SystemExit(f"código alterado desde o congelamento: {drift}")
    return fz


def run_family(key: str, out: Path, seeds=B.EVAL33, memory: bool = False) -> None:
    fz = _load()
    d = fz["families"][key]
    fam = B.FAMILIES[key]
    spec = B.spec_of(fam, d["mf"]["chosen"])
    arms = ["A_ORIG", "B_WIDE", "C_OWL", "J2_WIDE_BEAM_CPU"] if memory else list(B.ARMS33)
    rows, refs = [], []
    with tempfile.TemporaryDirectory() as td:
        for s in seeds:
            inst = B.make_instance33(fam, spec, s)
            refs.append(dict(B.reference33(inst, s), family=key))
            for a in arms:
                rows.append(B.fit_arm33(fam, spec, inst, a, s, Path(td), bw_j1=d["beam"]["bw_j1"], bw_j2=d["beam"]["bw_j2"], measure_memory=memory))
    Path(out).write_text(json.dumps({"family": key, "rows": rows, "refs": refs, "frozen": B.FROZEN33.name}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate"); c.add_argument("--family", required=True); c.add_argument("--out", required=True)
    f = sub.add_parser("freeze"); f.add_argument("parts", nargs="+")
    r = sub.add_parser("run"); r.add_argument("--family", required=True); r.add_argument("--out", required=True); r.add_argument("--memory", action="store_true")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        Path(a.out).write_text(json.dumps(calibrate_family(a.family), indent=1, default=str))
    elif a.cmd == "freeze":
        d = freeze([Path(p) for p in a.parts])
        print(json.dumps({k: {"mf": v["mf"], "bw": (v["beam"]["bw_j1"], v["beam"]["bw_j2"])} for k, v in d["families"].items()}, indent=1, default=str))
    else:
        run_family(a.family, Path(a.out), B.DIAG33 if a.memory else B.EVAL33, a.memory)


if __name__ == "__main__":
    main()
