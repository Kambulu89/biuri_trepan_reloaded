"""Relatório da matriz de controlos M29 (Doc. 29): métricas obrigatórias por braço, Nível 1 (ganho direto + atribuição por controlos), Nível 2 (cadeia global), evidência por
dataset e mecanismos ontológicos. Só leitura dos brutos; NENHUMA unidade excluída; controlo em falta ⇒ ``NOT_ASSESSABLE``.
    python -m validation.benchmark.m29_report --sc DIR_DOC26 --m29 DIR_CONTROLOS --out FILE.json
Mapa dos braços: A29 = ``sc_B`` (Original+C) · B29 = ``sc_D`` (Reloaded OWL real+C) · C29 = ``m29_P`` (permutada) · D29 = ``m29_D`` (grupos aleatórios) · E29 = ``m29_E`` (convencional) ·
F29 = ``sc_F`` (Reloaded sem OWL+C, controlo de equivalência com A29). C4.5 = ``c45``. Réplicas/draws: média por unidade (nunca são unidades independentes)."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from core import hierarchy_contract as H
from validation.benchmark.hierarchy_joint_report import load as load_full

METRICS = {"accuracy": "accuracy_real_labels", "fidelity": "fidelity_to_oracle", "precision_macro": "precision_macro_real_labels", "macro_f1": "macro_f1_real_labels",
           "node_count": "node_count", "leaf_count": "leaf_count", "tree_depth": "depth", "m_of_n_split_count": "m_of_n_count", "semantic_split_count": "semantic_split_count",
           "oracle_queries": "oracle_queries", "oracle_samples": "total_oracle_samples", "training_time": "tree_training_time"}
ARM_SOURCE = {"A29": ("sc", "sc_B_r"), "B29": ("sc", "sc_D_r"), "F29": ("sc", "sc_F_r"), "C29": ("m29", "m29_P_k"), "D29": ("m29", "m29_D_k"), "E29": ("m29", "m29_E_k")}
CONTROL_OF = {"C29": "permuted_ontology", "D29": "random_groups", "E29": "conventional_features"}


def arm_records(sc_unit, m_unit, arm):
    src, prefix = ARM_SOURCE[arm]
    unit = sc_unit if src == "sc" else (m_unit or {})
    return [unit[k] for k in sorted(unit) if k.startswith(prefix) and re.fullmatch(prefix + r"\d+", k)]


def mean_metric(recs, col):
    v = [r["row"].get(col) for r in recs if r["row"].get(col) is not None]
    return float(np.mean(v)) if v else None


def control_is_full(recs):
    return bool(recs) and all((r["row"].get("control_variant") in (None, "full")) for r in recs)


def build(sc, m29):
    res = {"arms": {}, "units": sorted(f"{k[0]}:{k[1]}" for k in sc), "n_units": len(sc), "exclusions": "nenhuma", "metrics": list(METRICS)}
    per_unit, comp_problems = {}, []
    for (ds, seed), sc_unit in sc.items():
        u = f"{ds}:{seed}"
        m_unit = m29.get((ds, seed))
        c45 = sc_unit.get("c45")
        recs = {a: arm_records(sc_unit, m_unit, a) for a in ARM_SOURCE}
        # comparabilidade ESTRITA entre todos os braços presentes e o C4.5 (A29/B29/F29 vêm de outra execução: configuração/manifest são verificados e só o config_hash cruzado é isento)
        allrecs = {f"{a}:{i}": r for a, rs in recs.items() for i, r in enumerate(rs)}
        if c45 is not None:
            allrecs["c45"] = c45
        cross = bool(m_unit)
        cmp_ = H.check_comparability(allrecs, relaxed=("experimental_configuration",) if cross else ())
        if not cmp_["comparable"]:
            comp_problems.append({"unit": u, "problems": cmp_["problems"]})
        vals = {a: {m: mean_metric(rs, col) for m, col in METRICS.items()} if rs else None for a, rs in recs.items()}
        per_unit[u] = {"vals": vals, "c45": {m: (c45["row"].get(col) if c45 else None) for m, col in METRICS.items() if m in ("accuracy", "fidelity", "precision_macro", "macro_f1")},
                       "control_variants": {a: sorted({r["row"].get("control_variant") or "full" for r in rs}) for a, rs in recs.items() if a in CONTROL_OF and rs},
                       "n_draws": {a: len(rs) for a, rs in recs.items()}}
    res["comparability"] = {"level": "strict", "all_comparable": not comp_problems, "problems": comp_problems}
    # --- Nível 1 e 2 por unidade
    l1, l2, attr = Counter(), Counter(), Counter()
    diffs = {"vs_original_eq": defaultdict(lambda: {"acc": [], "fid": []}), "vs_c45": defaultdict(lambda: {"acc": [], "fid": []})}
    ctl_diffs = {c: defaultdict(lambda: {"acc": [], "fid": []}) for c in CONTROL_OF.values()}
    eq_check = Counter()
    unit_states = {}
    for u, d in per_unit.items():
        v, c = d["vals"], d["c45"]
        if not v["A29"] or not v["B29"] or c.get("accuracy") is None:
            l1[H.INSUFFICIENT_EVIDENCE] += 1
            l2[H.INSUFFICIENT_EVIDENCE] += 1
            continue
        a, b = v["A29"], v["B29"]
        d1 = H.evaluate_direct_gain(b["accuracy"], a["accuracy"], b["fidelity"], a["fidelity"])
        controls = {}
        for arm, name in CONTROL_OF.items():
            cv = v.get(arm)
            if cv and (arm == "E29" or control_is_full(arm_records(sc[tuple(_k(u))], m29.get(tuple(_k(u))), arm))):
                controls[name] = (b["accuracy"] - cv["accuracy"], b["fidelity"] - cv["fidelity"])
            elif cv:
                controls[name + "_structure_only_fallback"] = (b["accuracy"] - cv["accuracy"], b["fidelity"] - cv["fidelity"])   # controlo parcial: NÃO satisfaz o controlo exigido
        at = H.attribute_to_ontology(d1, controls)
        g = H.evaluate_global(b["accuracy"], a["accuracy"], c["accuracy"], b["fidelity"], a["fidelity"], c["fidelity"])
        l1[d1["state"]] += 1
        attr[at["attribution"]] += 1
        l2[g["state"]] += 1
        unit_states[u] = {"direct": d1["state"], "attribution": at["attribution"], "global": g["state"], "flags": g["flags"], "missing": at["missing_required_controls"]}
        ds = u.split(":")[0]
        diffs["vs_original_eq"][ds]["acc"].append(b["accuracy"] - a["accuracy"]); diffs["vs_original_eq"][ds]["fid"].append(b["fidelity"] - a["fidelity"])
        diffs["vs_c45"][ds]["acc"].append(b["accuracy"] - c["accuracy"]); diffs["vs_c45"][ds]["fid"].append(b["fidelity"] - c["fidelity"])
        for name, (da, df) in controls.items():
            if name in ctl_diffs:
                ctl_diffs[name][ds]["acc"].append(da); ctl_diffs[name][ds]["fid"].append(df)
        f = v.get("F29")
        if f:
            eq_check["identical" if all(f[m] == a[m] for m in ("accuracy", "fidelity", "node_count", "oracle_queries")) else "different"] += 1
    pct = lambda cn: {k: {"n": n, "pct_of_all_units": round(100.0 * n / max(1, len(per_unit)), 1)} for k, n in cn.items()}
    res["equivalence_F29_vs_A29"] = dict(eq_check)
    res["level1"] = {"direct_gain_state_counts": pct(l1), "attribution_counts": dict(attr),
                     "evidence_vs_original_equivalent": H.classify_evidence(diffs["vs_original_eq"], confirmatory=False),
                     "evidence_vs_controls": {c: H.classify_evidence(dd, confirmatory=False) for c, dd in ctl_diffs.items() if dd},
                     "controls_without_data": [c for c, dd in ctl_diffs.items() if not dd]}
    res["level2"] = {"global_state_counts": pct(l2), "evidence_vs_c45": H.classify_evidence(diffs["vs_c45"], confirmatory=False)}
    res["per_unit"] = {u: {**per_unit[u], "states": unit_states.get(u)} for u in per_unit}
    # --- tabela de métricas por braço (média das unidades; sem arredondar)
    for arm in ARM_SOURCE:
        rows = [per_unit[u]["vals"][arm] for u in per_unit if per_unit[u]["vals"].get(arm)]
        res["arms"][arm] = {"n_units": len(rows), **{m: (float(np.mean([r[m] for r in rows if r[m] is not None])) if rows and any(r[m] is not None for r in rows) else None) for m in METRICS}}
    res["semantic_gain"] = {"accuracy": (res["arms"]["B29"]["accuracy"] - res["arms"]["A29"]["accuracy"]) if res["arms"]["B29"]["accuracy"] is not None and res["arms"]["A29"]["accuracy"] is not None else None,
                            "fidelity": (res["arms"]["B29"]["fidelity"] - res["arms"]["A29"]["fidelity"]) if res["arms"]["B29"]["fidelity"] is not None and res["arms"]["A29"]["fidelity"] is not None else None}
    res["evidence_level_overall"] = res["level1"]["evidence_vs_original_equivalent"].get("evidence")
    return res


def _k(u):
    ds, s = u.split(":")
    return (ds, int(s))


def mechanisms(sc, m29):
    """Perguntas 1–10 do utilizador, o que o bruto permite responder HOJE (campos agregados por árvore). Auditoria por split (m29_*/audits) acrescenta o detalhe."""
    out = defaultdict(lambda: defaultdict(list))
    for (ds, seed), su in sc.items():
        for arm in ARM_SOURCE:
            for r in arm_records(su, m29.get((ds, seed)), arm):
                row = r["row"]
                for k in ("ontology_influenced_splits", "semantic_decision_changed_count", "semantic_rule_count", "semantic_features_used", "semantic_split_count", "m_of_n_count", "internal_nodes"):
                    if row.get(k) is not None:
                        out[arm][k].append(float(row[k]))
    return {a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", required=True)
    ap.add_argument("--m29")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sc = load_full(Path(a.sc))
    m29 = load_full(Path(a.m29)) if a.m29 and (Path(a.m29) / "raw").exists() else {}
    r = build(sc, m29)
    r["mechanism_aggregates"] = mechanisms(sc, m29)
    Path(a.out).write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps({k: r[k] for k in ("n_units", "comparability", "equivalence_F29_vs_A29", "semantic_gain", "evidence_level_overall")}, indent=1, default=str))
    print(json.dumps(r["level1"]["attribution_counts"]), json.dumps({k: v["n"] for k, v in r["level1"]["direct_gain_state_counts"].items()}), json.dumps({k: v["n"] for k, v in r["level2"]["global_state_counts"].items()}))
