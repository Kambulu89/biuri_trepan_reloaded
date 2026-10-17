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
ARM_SOURCE = {"A29": ("sc", "sc_B_r"), "B29": ("sc", "sc_D_r"), "F29": ("sc", "sc_F_r"), "C29": ("m29", "m29_P_k"), "D29": ("m29", "m29_D_k"), "E29": ("m29", "m29_E_k"),
              "G29": ("m29", "m29_G_k")}
CONTROL_OF = {"C29": "permuted_ontology", "D29": "random_groups", "E29": "conventional_features", "G29": "budget_matched_original"}
REQUIRED = tuple(H.REQUIRED_CONTROLS) + ("budget_matched_original",)
E_CONFOUND = {"max_relative_dimension_gap": 0.25, "queries_ratio_range": (0.8, 1.25)}      # limiares PRÉ-DECLARADOS (Doc. 29/30) para marcar E29 como confundido


def arm_records(sc_unit, m_unit, arm):
    src, prefix = ARM_SOURCE[arm]
    unit = sc_unit if src == "sc" else (m_unit or {})
    return [unit[k] for k in sorted(unit) if k.startswith(prefix) and re.fullmatch(prefix + r"\d+", k)]


def mean_metric(recs, col):
    v = [r["row"].get(col) for r in recs if r["row"].get(col) is not None]
    return float(np.mean(v)) if v else None


def control_is_full(recs):
    return bool(recs) and all((r["row"].get("control_variant") in (None, "full")) for r in recs)


def confounding_e29(b_recs, e_recs) -> dict:
    """E29 está CONFUNDIDO face a B29 se a dimensão derivada ou o consumo de consultas diferirem substancialmente (limiares pré-declarados). Nunca se completam colunas artificiais."""
    # dimensão derivada EFETIVA de cada braço: ``n_derived_effective`` (injetada a partir da auditoria do braço, que conhece o contexto realmente usado); o campo
    # ``semantic_features_generated`` da linha bruta descreve o contexto da OWL real e NÃO o do controlo — por isso só serve de recurso para B29.
    db = mean_metric(b_recs, "n_derived_effective") if mean_metric(b_recs, "n_derived_effective") is not None else mean_metric(b_recs, "semantic_features_generated")
    de = mean_metric(e_recs, "n_derived_effective") if mean_metric(e_recs, "n_derived_effective") is not None else mean_metric(e_recs, "semantic_features_generated")
    qb, qe = mean_metric(b_recs, "oracle_queries"), mean_metric(e_recs, "oracle_queries")
    gap = abs((de or 0) - (db or 0)) / max(1.0, (db or 1.0))
    ratio = (qe / qb) if qb else None
    reasons = []
    if db is None or de is None or gap > E_CONFOUND["max_relative_dimension_gap"]:
        reasons.append(f"dimensão derivada {de} vs {db}")
    return {"confounded": bool(reasons), "reasons": reasons, "derived_B29": db, "derived_E29": de, "queries_B29": qb, "queries_E29": qe, "queries_ratio": ratio}


def budget_confounding(b_recs, c_recs, *, b_wins: bool) -> dict:
    """Confusão de ORÇAMENTO assimétrica (limiares pré-declarados 0,8–1,25 sobre ρ = consultas do controlo / consultas de B29): uma vitória de B29 sobre um controlo que gastou MENOS
    (ρ < 0,8) pode dever-se ao orçamento; uma derrota de B29 perante um controlo que gastou MAIS (ρ > 1,25) também. Fora disso o contraste é informativo."""
    qb, qc = mean_metric(b_recs, "oracle_queries"), mean_metric(c_recs, "oracle_queries")
    if not qb or qc is None:
        return {"confounded": True, "ratio": None, "reasons": ["consultas não verificáveis"]}
    rho = qc / qb
    lo, hi = E_CONFOUND["queries_ratio_range"]
    if b_wins and rho < lo:
        return {"confounded": True, "ratio": rho, "reasons": [f"B29 supera o controlo, mas este gastou só {rho:.2f}× as consultas de B29"]}
    if (not b_wins) and rho > hi:
        return {"confounded": True, "ratio": rho, "reasons": [f"B29 não supera o controlo, que gastou {rho:.2f}× as consultas de B29"]}
    return {"confounded": False, "ratio": rho, "reasons": []}


def build(sc, m29):
    res = {"arms": {}, "units": sorted(f"{k[0]}:{k[1]}" for k in sc), "n_units": len(sc), "exclusions": "nenhuma", "metrics": list(METRICS)}
    per_unit, comp_problems = {}, []
    for (ds, seed), sc_unit in sc.items():
        u = f"{ds}:{seed}"
        m_unit = m29.get((ds, seed))
        c45 = sc_unit.get("c45")
        recs = {a: arm_records(sc_unit, m_unit, a) for a in ARM_SOURCE}
        # comparabilidade ESTRITA entre todos os braços presentes e o C4.5 (A29/B29/F29 vêm de outra execução: configuração/manifest são verificados e só o config_hash cruzado é isento)
        main_recs = {f"{a}:{i}": r for a, rs in recs.items() if a != "G29" for i, r in enumerate(rs)}
        if c45 is not None:
            main_recs["c45"] = c45
        cross = bool(m_unit)
        cmp_ = H.check_comparability(main_recs, relaxed=("experimental_configuration",) if cross else ())
        if not cmp_["comparable"]:
            comp_problems.append({"unit": u, "problems": cmp_["problems"]})
        for i, r in enumerate(recs.get("G29", [])):                       # G29: orçamento diferente POR DESENHO (query_budget isento e documentado); restantes propriedades estritas
            cg = H.check_comparability({"A29:0": recs["A29"][0], f"G29:{i}": r, **({"c45": c45} if c45 is not None else {})} if recs.get("A29") else {f"G29:{i}": r},
                                       relaxed=("experimental_configuration", "query_budget"))
            if not cg["comparable"]:
                comp_problems.append({"unit": u, "arm": f"G29:{i}", "problems": cg["problems"]})
        vals = {a: {m: mean_metric(rs, col) for m, col in METRICS.items()} if rs else None for a, rs in recs.items()}
        per_draw = {a: [{"accuracy": r["row"].get("accuracy_real_labels"), "fidelity": r["row"].get("fidelity_to_oracle"), "oracle_queries": r["row"].get("oracle_queries"),
                        "node_count": r["row"].get("node_count"), "variant": r["row"].get("control_variant")} for r in rs] for a, rs in recs.items() if rs}
        per_unit[u] = {"per_draw": per_draw, "vals": vals, "c45": {m: (c45["row"].get(col) if c45 else None) for m, col in METRICS.items() if m in ("accuracy", "fidelity", "precision_macro", "macro_f1")},
                       "control_variants": {a: sorted({r["row"].get("control_variant") or "full" for r in rs}) for a, rs in recs.items() if a in CONTROL_OF and rs},
                       "n_draws": {a: len(rs) for a, rs in recs.items()}}
    res["comparability"] = {"level": "strict", "all_comparable": not comp_problems, "problems": comp_problems}
    # --- Nível 1 e 2 por unidade
    l1, l2, attr, attr_partial = Counter(), Counter(), Counter(), Counter()
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
        controls, conf = {}, {}
        b_recs = arm_records(sc[tuple(_k(u))], m29.get(tuple(_k(u))), "B29")
        for arm, name in CONTROL_OF.items():
            cv = v.get(arm)
            if not cv:
                continue
            recs_arm = arm_records(sc[tuple(_k(u))], m29.get(tuple(_k(u))), arm)
            full = arm in ("E29", "G29") or control_is_full(recs_arm)
            da, df = b["accuracy"] - cv["accuracy"], b["fidelity"] - cv["fidelity"]
            bud = budget_confounding(b_recs, recs_arm, b_wins=(da > 0 and df > 0))
            reasons = list(bud["reasons"])
            if arm == "E29":
                c_e = confounding_e29(b_recs, recs_arm)
                reasons += [r for r in c_e["reasons"] if r not in reasons and r.startswith("dimensão")]
            conf.setdefault(u, {})[arm] = {"confounded": bool(reasons), "reasons": reasons, "queries_ratio_control_over_B29": bud["ratio"]}
            if full and not reasons:
                controls[name] = (da, df)
            else:
                controls[name + ("_confounded" if reasons else "_structure_only_fallback")] = (da, df)
        at = H.attribute_to_ontology(d1, controls, required=REQUIRED)
        at_partial = H.attribute_to_ontology(d1, controls, required=("permuted_ontology", "random_groups", "budget_matched_original"))
        g = H.evaluate_global(b["accuracy"], a["accuracy"], c["accuracy"], b["fidelity"], a["fidelity"], c["fidelity"])
        l1[d1["state"]] += 1
        attr[at["attribution"]] += 1
        attr_partial[at_partial["attribution"]] += 1
        l2[g["state"]] += 1
        unit_states[u] = {"direct": d1["state"], "attribution": at["attribution"], "attribution_excluding_E29": at_partial["attribution"], "control_confounding": conf.get(u), "global": g["state"], "flags": g["flags"], "missing": at["missing_required_controls"]}
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
    res["level1"] = {"direct_gain_state_counts": pct(l1), "attribution_counts": dict(attr), "attribution_counts_excluding_E29": dict(attr_partial),
                     "attribution_note": "atribuição completa exige C29, D29, E29 (não confundido) e G29; sem E29 comparável a atribuição é NOT_ASSESSABLE. 'excluding_E29' é um estado SECUNDÁRIO (não conclusivo).",
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
    res["contrast_tests"] = contrast_tests(per_unit)
    return res


def contrast_tests(per_unit) -> dict:
    """Família primária pré-registada (+G29 autorizado depois): B29−A29, B29−C29, B29−D29, B29−E29, B29−G29. Cada contraste é uma interseção-união sobre (accuracy, fidelidade): p_IUT = max(p_acc, p_fid).
    Dois níveis de inferência reportados: (i) DATASET (média das unidades por dataset; t de uma amostra, gl = nº de datasets − 1; é a unidade de generalização) e (ii) UNIDADE (t pareado
    sobre unidades; descritivo, as unidades do mesmo dataset NÃO são independentes). Holm sobre a família no nível dataset. Exploratório (DEV)."""
    from scipy.stats import t as T
    out, pv = {}, []
    for arm, name in (("A29", "B29-A29"), ("C29", "B29-C29"), ("D29", "B29-D29"), ("E29", "B29-E29"), ("G29", "B29-G29")):
        by_ds = {m: defaultdict(list) for m in ("accuracy", "fidelity")}
        units = {m: [] for m in ("accuracy", "fidelity")}
        for u, d in per_unit.items():
            v = d["vals"]
            if not v.get("B29") or not v.get(arm):
                continue
            for m in ("accuracy", "fidelity"):
                diff = v["B29"][m] - v[arm][m]
                by_ds[m][u.split(":")[0]].append(diff)
                units[m].append(diff)
        if not units["accuracy"]:
            out[name] = {"available": False}
            continue
        def tt(x):
            x = np.asarray(x, dtype=float)
            if len(x) < 2 or x.std(ddof=1) == 0:
                return {"mean": float(x.mean()), "se": None, "t": None, "p_two_sided": None, "n": len(x)}
            se = x.std(ddof=1) / np.sqrt(len(x))
            tv = x.mean() / se
            return {"mean": float(x.mean()), "se": float(se), "t": float(tv), "p_two_sided": float(2 * T.sf(abs(tv), len(x) - 1)), "n": len(x)}
        dsm = {m: [float(np.mean(v)) for v in by_ds[m].values()] for m in by_ds}
        res = {"available": True, "dataset_level": {m: tt(dsm[m]) for m in dsm}, "unit_level": {m: tt(units[m]) for m in units},
               "per_dataset_mean": {m: {d: float(np.mean(v)) for d, v in by_ds[m].items()} for m in by_ds}}
        pa, pf = res["dataset_level"]["accuracy"]["p_two_sided"], res["dataset_level"]["fidelity"]["p_two_sided"]
        both_pos = res["dataset_level"]["accuracy"]["mean"] > 0 and res["dataset_level"]["fidelity"]["mean"] > 0
        res["p_IUT_dataset"] = (max(pa, pf) if (pa is not None and pf is not None and both_pos) else (1.0 if pa is not None else None))
        out[name] = res
        pv.append((name, res["p_IUT_dataset"]))
    valid = [(n, p) for n, p in pv if p is not None]
    order = sorted(valid, key=lambda x: x[1])
    run = 0.0
    for rank, (n, p) in enumerate(order):
        run = max(run, min(1.0, p * (len(order) - rank)))
        out[n]["p_IUT_holm_family"] = run
    out["note"] = "exploratório (DEV, 4 datasets): p-valores descritivos, nunca confirmatórios; SUPPORTED_JOINT_GAIN exige avaliação confirmatória com ≥12 datasets"
    return out


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
    ap.add_argument("--m29", nargs="*", default=[], help="um ou mais diretórios de resultados M29 (os últimos prevalecem em braços com o mesmo nome)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sc = load_full(Path(a.sc))
    m29 = {}
    for d in a.m29:
        if (Path(d) / "raw").exists():
            for key, arms in load_full(Path(d)).items():
                for arm_name, rec_ in arms.items():
                    af = Path(d) / "audits" / f"{key[0]}_seed{key[1]}__{arm_name}.json"
                    if af.exists():
                        meta = json.loads(af.read_text()).get("control") or {}
                        if "n_derived" in meta:
                            rec_["row"]["n_derived_effective"] = meta["n_derived"]
                m29.setdefault(key, {}).update(arms)
    r = build(sc, m29)
    r["mechanism_aggregates"] = mechanisms(sc, m29)
    Path(a.out).write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps({k: r[k] for k in ("n_units", "comparability", "equivalence_F29_vs_A29", "semantic_gain", "evidence_level_overall")}, indent=1, default=str))
    print(json.dumps(r["level1"]["attribution_counts"]), json.dumps({k: v["n"] for k, v in r["level1"]["direct_gain_state_counts"].items()}), json.dumps({k: v["n"] for k, v in r["level2"]["global_state_counts"].items()}))
