"""Relatório RETROATIVO e só de leitura da hierarquia em DOIS NÍVEIS (core/hierarchy_contract.py; Doc. 28) sobre brutos existentes. Não altera resultados, baselines
nem seleção; nenhuma unidade é excluída (as percentagens são sobre TODAS as unidades; unidades com Original ≤ C4.5 aparecem como HIERARCHY_INFEASIBLE).

Nível 1 — ganho ontológico DIRETO: Reloaded+OWL vs Original EQUIVALENTE (+ controlos disponíveis: permutada, grupos aleatórios, convencional).
Nível 2 — superioridade GLOBAL: cadeia completa U1–U5 + sinalizadores (ganho direto, superioridade de accuracy/fidelidade, compromissos, inviabilidade pelos baselines).
Evidência (OBSERVED/SUPPORTED/…) SEPARADA do estado lógico; unidade de inferência = dataset; DEV já observado ⇒ nunca confirmatório.
As métricas são ``accuracy_real_labels`` (principal) e ``fidelity_to_oracle`` (mesmo MLP); ``precision_macro_real_labels`` e ``macro_f1_real_labels`` só complementares.
    python -m validation.benchmark.hierarchy_joint_report --spec sampler_c|coherent|historical --dir DIR [--cx DIR] --out FILE
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from core import hierarchy_contract as H
from validation.benchmark.diag_b2_analysis import load
from validation.benchmark.sampler_c_analysis import replicas as _replica_rows_value

ACC, FID = "accuracy_real_labels", "fidelity_to_oracle"
COMPL = ("precision_macro_real_labels", "macro_f1_real_labels")


def _fam_rows(arms, cx_arms, fam):
    """Linhas (bruto completo) das réplicas da família A–F de uma unidade (mesma lógica de sampler_c_analysis.replicas)."""
    rows = []
    for r in range(3):
        if fam == "A":
            row = arms.get("trepan_original") if r == 0 else arms.get(f"sc_A_r{r}")
        elif fam == "E":
            row = arms.get("reloaded_core") if r == 0 else arms.get(f"sc_E_r{r}")
        elif fam == "C":
            row = arms.get("reloaded_owl_full") if r == 0 else (arms.get(f"sc_C_r{r}") or (cx_arms or {}).get(f"cx_h_on_n_r{r}"))
        else:
            row = arms.get(f"sc_{fam}_r{r}")
        if row is not None:
            rows.append(row)
    return rows


SPECS = {
    # Nível 1: D vs B (OWL + amostrador C vs Original + C: MESMO amostrador). Controlos sob C não foram executados ⇒ NOT_ASSESSABLE.
    "sampler_c": {"R": ("fam", "D"), "Oeq": ("fam", "B"), "C45": ("arm", "c45"), "controls": {},
                  "title": "Reloaded OWL + amostrador C (D) vs Original + amostrador C (B)"},
    # histórico (amostrador marginal): C vs A; controlo permutado inexistente nesta campanha
    "sampler_c_historical": {"R": ("fam", "C"), "Oeq": ("fam", "A"), "C45": ("arm", "c45"), "controls": {}, "title": "Reloaded OWL histórico (C) vs Original histórico (A)"},
    # Doc. 22: amostragem coerente (mesma representação enriquecida: Original equivalente = trepan_original_enr)
    "coherent": {"R": ("arm", "coh_full"), "Oeq": ("arm", "trepan_original_enr"), "C45": ("arm", "c45"),
                 "controls": {"permuted_ontology": ("arm", "coh_full_shuf"), "conventional_features": ("arm", "conv_coh")},
                 "title": "Reloaded coerente + OWL (coh_full) vs Original na mesma representação (trepan_original_enr)"},
    "coherent_vs_original": {"R": ("arm", "coh_full"), "Oeq": ("arm", "trepan_original"), "C45": ("arm", "c45"),
                             "controls": {"permuted_ontology": ("arm", "coh_full_shuf"), "conventional_features": ("arm", "conv_coh")},
                             "title": "coh_full vs Original canónico (espaço original)"},
    "historical": {"R": ("arm", "reloaded_owl_full"), "Oeq": ("arm", "trepan_original"), "C45": ("arm", "c45"), "controls": {"permuted_ontology": ("arm", "reloaded_owl_shuffled")},
                   "title": "Reloaded OWL histórico vs Original (controlo: ontologia permutada)"},
}


def _get(arms, cx_arms, ref):
    kind, name = ref
    if kind == "fam":
        return _fam_rows(arms, cx_arms, name)
    return [arms[name]] if name in arms else []


def _mean(rows, key):
    v = [r.get(key) for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def report(units: dict, cx: dict, spec_name: str) -> dict:
    spec = SPECS[spec_name]
    out = {"spec": spec_name, "title": spec["title"], "n_units": len(units), "exclusions": "nenhuma", "metrics": {"principal": ACC, "fidelity": FID, "complementary_only": list(COMPL)},
           "inference_status": "exploratório: DEV já observado, não confirmatório"}
    # --- comparabilidade (mesmas instâncias, mesmo MLP) em TODAS as linhas usadas
    problems, u_vals = [], {}
    for u, arms in units.items():
        ca = cx.get(u)
        R, O, C = _get(arms, ca, spec["R"]), _get(arms, ca, spec["Oeq"]), _get(arms, ca, spec["C45"])
        ctrl = {k: _get(arms, ca, v) for k, v in spec["controls"].items()}
        if not (R and O and C):
            u_vals[u] = None
            continue
        rows = {f"R{i}": r for i, r in enumerate(R)} | {f"O{i}": r for i, r in enumerate(O)} | {"C45": C[0]} | {f"{k}{i}": r for k, rs in ctrl.items() for i, r in enumerate(rs)}
        cmp_ = H.check_comparability(rows)
        if not cmp_["comparable"]:
            problems.append({"unit": u, "problems": cmp_["problems"]})
        u_vals[u] = {"R": (_mean(R, ACC), _mean(R, FID)), "O": (_mean(O, ACC), _mean(O, FID)), "C": (_mean(C, ACC), _mean(C, FID)),
                     "ctrl": {k: ((_mean(rs, ACC), _mean(rs, FID)) if rs else None) for k, rs in ctrl.items()}, "n_rep": (len(R), len(O)),
                     "compl": {m: {"R": _mean(R, m), "O": _mean(O, m), "C45": _mean(C, m)} for m in COMPL}}
    out["comparability"] = {"checked_units": sum(v is not None for v in u_vals.values()), "problems": problems, "all_comparable": not problems}
    if problems:
        out["verdict"] = "INSUFFICIENT_EVIDENCE: instâncias/oráculo não comparáveis"
        return out
    # --- Nível 1
    l1_state, l1_noninf, l1_strict, attr = Counter(), 0, 0, Counter()
    l2_state, l2_flags, l2_mode = Counter(), Counter(), Counter()
    by_ds_o, by_ds_c, by_ds_ctrl = defaultdict(lambda: {"acc": [], "fid": []}), defaultdict(lambda: {"acc": [], "fid": []}), defaultdict(lambda: defaultdict(lambda: {"acc": [], "fid": []}))
    per_unit = {}
    n = 0
    for u, v in u_vals.items():
        if v is None:
            l2_state[H.INSUFFICIENT_EVIDENCE] += 1
            l1_state[H.INSUFFICIENT_EVIDENCE] += 1
            continue
        n += 1
        (aR, fR), (aO, fO), (aC, fC) = v["R"], v["O"], v["C"]
        d1 = H.evaluate_direct_gain(aR, aO, fR, fO)
        l1_state[d1["state"]] += 1
        l1_noninf += int(bool(d1["non_inferior"]))
        l1_strict += int(bool(d1["strict_joint_gain"]))
        ctrl_c = {k: ((aR - c[0], fR - c[1]) if c and None not in c else None) for k, c in v["ctrl"].items()}
        at = H.attribute_to_ontology(d1, ctrl_c)
        attr[at["attribution"]] += 1
        g2 = H.evaluate_global(aR, aO, aC, fR, fO, fC)
        l2_state[g2["state"]] += 1
        l2_mode[g2["mode"]] += 1
        for k, b in g2["flags"].items():
            if isinstance(b, bool):
                l2_flags[k] += int(b)
            else:
                l2_flags[f"{k}={b}"] += 1
        for k, b in g2["conditions"].items():
            l2_flags[f"cond:{k}"] += int(b)
        ds = u.split(":")[0]
        by_ds_o[ds]["acc"].append(aR - aO); by_ds_o[ds]["fid"].append(fR - fO)
        by_ds_c[ds]["acc"].append(aR - aC); by_ds_c[ds]["fid"].append(fR - fC)
        for k, d in ctrl_c.items():
            if d is not None:
                by_ds_ctrl[k][ds]["acc"].append(d[0]); by_ds_ctrl[k][ds]["fid"].append(d[1])
        per_unit[u] = {"level1": d1["state"], "attribution": at["attribution"], "level2": g2["state"], "deltas": g2["deltas"]}
    pct = lambda c: {k: {"n": v, "pct_of_all_units": round(100.0 * v / max(1, len(u_vals)), 1)} for k, v in c.items()}
    out["level1_direct_ontological_gain"] = {"definition": "A_R vs A_Oeq e F_R vs F_Oeq (mesma estratégia de amostragem/oráculo/partições/orçamento)", "state_counts": pct(l1_state),
                                             "non_inferior_eligible": {"n": l1_noninf, "of": len(u_vals)}, "strict_joint_gain": {"n": l1_strict, "of": len(u_vals)},
                                             "attribution_to_ontology_counts": dict(attr),
                                             "controls_available": sorted(spec["controls"]), "controls_missing": [c for c in H.REQUIRED_CONTROLS if c not in spec["controls"]],
                                             "evidence_vs_original_equivalent": H.classify_evidence(by_ds_o, confirmatory=False),
                                             "evidence_vs_controls": {k: H.classify_evidence(d, confirmatory=False) for k, d in by_ds_ctrl.items()}}
    out["level2_global_hierarchy"] = {"definition": "U1 A_R>A_O; U2 A_O>A_C; U3 A_R>A_C; U4 F_R>F_O; U5 F_R>F_C (estrito)", "state_counts": pct(l2_state), "mode_counts": dict(l2_mode),
                                      "flags_and_conditions_true_counts": dict(l2_flags), "evidence_vs_c45": H.classify_evidence(by_ds_c, confirmatory=False),
                                      "note": "HIERARCHY_INFEASIBLE = Original ≤ C4.5 em accuracy NESSA unidade com os baselines fixados; não exclui a unidade; não é impossibilidade universal."}
    out["questions"] = {
        "Q1_joint_gain_over_equivalent_original": out["level1_direct_ontological_gain"]["evidence_vs_original_equivalent"].get("evidence"),
        "Q2_caused_by_ontological_knowledge": "NOT_ASSESSABLE" if out["level1_direct_ontological_gain"]["controls_missing"] else dict(attr),
        "Q3_global_hierarchy_units": {"JOINT_HIERARCHY_SATISFIED": l2_state.get(H.JOINT_HIERARCHY_SATISFIED, 0), "of_all_units": len(u_vals)}}
    out["complementary_metric_means_over_units"] = {m: {t: float(np.mean([v["compl"][m][t] for v in u_vals.values() if v and v["compl"][m][t] is not None])) for t in ("R", "O", "C45")} for m in COMPL}
    out["per_unit"] = per_unit
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=sorted(SPECS))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cx")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    units = {f"{k[0]}:{k[1]}": v for k, v in load(Path(a.dir)).items()}
    cx = {f"{k[0]}:{k[1]}": v for k, v in load(Path(a.cx)).items()} if a.cx else {}
    r = report(units, cx, a.spec)
    Path(a.out).write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps({k: v for k, v in r.items() if k != "per_unit"}, indent=1, default=str)[:6000])
