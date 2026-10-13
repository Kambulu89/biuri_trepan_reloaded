"""Análise DEV de B1 segundo o pré-registo (Doc. 17 §0). Só leitura dos resultados brutos; sem seleção por teste; nível de evidência observed_only.

    python -m validation.benchmark.b1_analysis --b1 DIR --ref DIR --out DIR_OUT
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List

LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)
REF_ARMS = ("mlp_original", "c45", "trepan_original", "reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled")


def _tag(cal: str, lam: float) -> str:
    return "b1_tiebreak" if cal == "tiebreak" else f"b1_{cal}_l{lam:g}".replace(".", "p")


def variants() -> List[Dict[str, Any]]:
    out = [{"id": "b1_tiebreak", "cal": "tiebreak", "lam": None}]
    for cal in ("bounded", "calibrated"):
        out += [{"id": _tag(cal, l), "cal": cal, "lam": l} for l in LAMBDAS]
    return out


def load_units(root: Path) -> Dict[tuple, Dict[str, dict]]:
    units: Dict[tuple, Dict[str, dict]] = {}
    for ud in sorted((root / "raw").glob("*/seed*")):
        ds, sd = ud.parent.name, int(ud.name[4:])
        arms = {}
        for f in ud.glob("*.json"):
            if f.name in ("split.json", "UNIT_SHA256.json"):
                continue
            arms[f.stem] = json.loads(f.read_text())
        units[(ds, sd)] = arms
    return units


def _g(arm, key):
    return arm["row"].get(key)


def integrity(b1: dict, ref: dict) -> Dict[str, Any]:
    """B1 desligado (braços A–F) ≡ referência; b1_legacy_logged ≡ E."""
    res = {"units_compared": 0, "arms_identical": 0, "arms_total": 0, "differences": [], "legacy_logged_equals_E": 0, "legacy_logged_total": 0}
    for key, arms in b1.items():
        if key not in ref:
            continue
        res["units_compared"] += 1
        for a in REF_ARMS:
            if a not in arms or a not in ref[key]:
                continue
            res["arms_total"] += 1
            pa, pr = arms[a]["predictions"], ref[key][a]["predictions"]
            same = all(pa[k] == pr[k] for k in ("test_row_index", "y_true", "oracle_predictions", "surrogate_predictions"))
            for m in ("fidelity_to_oracle", "accuracy_real_labels", "node_count", "leaf_count", "depth"):
                same = same and (_g(arms[a], m) == _g(ref[key][a], m))
            res["arms_identical"] += int(same)
            if not same:
                res["differences"].append([list(key), a])
        if "b1_legacy_logged" in arms and "reloaded_owl_full" in arms:
            res["legacy_logged_total"] += 1
            res["legacy_logged_equals_E"] += int(arms["b1_legacy_logged"]["predictions"]["surrogate_predictions"] == arms["reloaded_owl_full"]["predictions"]["surrogate_predictions"])
    return res


def per_dataset(units, arm: str, metric: str) -> Dict[str, float]:
    by: Dict[str, List[float]] = {}
    for (ds, _), arms in units.items():
        if arm in arms and _g(arms[arm], metric) is not None:
            by.setdefault(ds, []).append(float(_g(arms[arm], metric)))
    return {ds: st.mean(v) for ds, v in by.items()}


def paired(units, a: str, b: str, metric: str) -> Dict[str, Dict[str, Any]]:
    """Diferença a−b por dataset: média, nº de vitórias/empates/derrotas estritos (δ=0) sobre as unidades."""
    by: Dict[str, List[float]] = {}
    for (ds, _), arms in units.items():
        if a in arms and b in arms and _g(arms[a], metric) is not None and _g(arms[b], metric) is not None:
            by.setdefault(ds, []).append(float(_g(arms[a], metric)) - float(_g(arms[b], metric)))
    return {ds: {"mean": st.mean(v), "wins": sum(x > 0 for x in v), "ties": sum(x == 0 for x in v), "losses": sum(x < 0 for x in v), "n": len(v)}
            for ds, v in by.items()}


def evaluate(units) -> Dict[str, Any]:
    out: Dict[str, Any] = {"variants": {}}
    for v in variants():
        vid, ctl = v["id"], v["id"] + "_shuf"
        if not any(vid in a for a in units.values()):
            continue
        d_O_f, d_O_a = paired(units, vid, "trepan_original", "fidelity_to_oracle"), paired(units, vid, "trepan_original", "accuracy_real_labels")
        d_E_f, d_E_a = paired(units, vid, "reloaded_owl_full", "fidelity_to_oracle"), paired(units, vid, "reloaded_owl_full", "accuracy_real_labels")
        d_C_f, d_C_a = paired(units, vid, ctl, "fidelity_to_oracle"), paired(units, vid, ctl, "accuracy_real_labels")
        nodes_v, nodes_E = per_dataset(units, vid, "node_count"), per_dataset(units, "reloaded_owl_full", "node_count")
        samp_v, samp_E, samp_O = (per_dataset(units, vid, "total_oracle_samples"), per_dataset(units, "reloaded_owl_full", "total_oracle_samples"),
                                  per_dataset(units, "trepan_original", "oracle_queries"))
        dss = sorted(d_O_f)
        c1 = sum(d_O_f[d]["mean"] >= 0 for d in dss)
        c2 = sum(d_E_f[d]["mean"] > 0 and d_E_a[d]["mean"] >= 0 for d in dss)
        c3 = sum(d_C_f[d]["mean"] > 0 and d_C_a[d]["mean"] >= 0 for d in dss)
        c4 = sum(nodes_v[d] <= 1.10 * nodes_E[d] for d in dss)
        c5 = sum((samp_v.get(d) or 0) <= (samp_E.get(d) or 0) for d in dss)
        need = 3 if len(dss) >= 4 else len(dss)
        crit = {"C1_fidelity_non_inferior_vs_Original": [c1, len(dss)], "C2_improves_historical_E": [c2, len(dss)],
                "C3_beats_own_negative_control": [c3, len(dss)], "C4_complexity_le_1.10xE": [c4, len(dss)], "C5_oracle_cost_le_E": [c5, len(dss)]}
        passed = c1 >= need and c2 >= need and c3 >= need and c4 >= need and c5 >= need
        diag = {m: per_dataset(units, vid, m) for m in ("b1_switch_rate", "b1_regret_median", "b1_ratio_m_over_s_median")}
        out["variants"][vid] = {"cal": v["cal"], "lam": v["lam"], "criteria": crit, "carry_to_B2": bool(passed),
                                "delta_fid_vs_Original": d_O_f, "delta_acc_vs_Original": d_O_a, "delta_fid_vs_E": d_E_f, "delta_acc_vs_E": d_E_a,
                                "delta_fid_vs_control": d_C_f, "delta_acc_vs_control": d_C_a, "nodes": nodes_v, "total_oracle_samples": samp_v,
                                "original_oracle_samples": samp_O, "diagnostics": diag,
                                "F_ge_O_datasets": c1, "F_gt_O_datasets": sum(d_O_f[d]["mean"] > 0 for d in dss),
                                "Acc_gt_O_datasets": sum(d_O_a[d]["mean"] > 0 for d in dss)}
    out["reference_E"] = {"delta_fid_vs_Original": paired(units, "reloaded_owl_full", "trepan_original", "fidelity_to_oracle"),
                          "delta_acc_vs_Original": paired(units, "reloaded_owl_full", "trepan_original", "accuracy_real_labels"),
                          "delta_fid_E_vs_control": paired(units, "reloaded_owl_full", "reloaded_owl_shuffled", "fidelity_to_oracle"),
                          "delta_acc_E_vs_control": paired(units, "reloaded_owl_full", "reloaded_owl_shuffled", "accuracy_real_labels"),
                          "diagnostics_legacy": {m: per_dataset(units, "b1_legacy_logged", m) for m in ("b1_switch_rate", "b1_regret_median", "b1_ratio_m_over_s_median")}}
    return out


def to_markdown(ev: Dict[str, Any], integ: Dict[str, Any]) -> str:
    f = lambda x: "—" if x is None else f"{x:+.4f}"
    L = ["# B1 — resultados DEV (análise segundo o pré-registo; evidência observed_only)\n",
         f"**Integridade:** B1 desligado ≡ referência em {integ['arms_identical']}/{integ['arms_total']} braços ({integ['units_compared']} unidades); "
         f"`b1_legacy_logged ≡ E`: {integ['legacy_logged_equals_E']}/{integ['legacy_logged_total']}. Diferenças: {integ['differences'][:5]}\n",
         "## Referência E (OWL histórico) vs Original e vs controlo permutado (médias por dataset de ΔFidelity / ΔAccuracy)\n",
         "| dataset | ΔF(E−O) | ΔAcc(E−O) | ΔF(E−F) | ΔAcc(E−F) |", "|---|---|---|---|---|"]
    r = ev["reference_E"]
    for d in sorted(r["delta_fid_vs_Original"]):
        L.append(f"| {d} | {f(r['delta_fid_vs_Original'][d]['mean'])} | {f(r['delta_acc_vs_Original'][d]['mean'])} | {f(r['delta_fid_E_vs_control'][d]['mean'])} | {f(r['delta_acc_E_vs_control'][d]['mean'])} |")
    L += ["\n## Variantes B1: critérios pré-registados (n datasets que cumprem / total) e decisão\n",
          "| variante | C1 F≥O | C2 >E | C3 >controlo | C4 nós | C5 custo | F>O (datasets) | Acc>O | → B2 |", "|---|---|---|---|---|---|---|---|---|"]
    for vid, v in ev["variants"].items():
        c = v["criteria"]
        L.append(f"| {vid} | {c['C1_fidelity_non_inferior_vs_Original'][0]}/{c['C1_fidelity_non_inferior_vs_Original'][1]} | {c['C2_improves_historical_E'][0]} | "
                 f"{c['C3_beats_own_negative_control'][0]} | {c['C4_complexity_le_1.10xE'][0]} | {c['C5_oracle_cost_le_E'][0]} | {v['F_gt_O_datasets']} | {v['Acc_gt_O_datasets']} | "
                 f"{'**SIM**' if v['carry_to_B2'] else 'não'} |")
    L += ["\n## Detalhe por variante e dataset (ΔF/ΔAcc vs Original; ΔF/ΔAcc vs E; ΔF/ΔAcc vs controlo; nós; amostras ao oráculo; switch; regret; m/s)\n"]
    for vid, v in ev["variants"].items():
        L += [f"### {vid}", "| dataset | ΔF−O | ΔAcc−O | ΔF−E | ΔAcc−E | ΔF−ctl | ΔAcc−ctl | nós | amostras (O) | switch | regret med | m/s |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in sorted(v["delta_fid_vs_Original"]):
            dg = v["diagnostics"]
            L.append(f"| {d} | {f(v['delta_fid_vs_Original'][d]['mean'])} | {f(v['delta_acc_vs_Original'][d]['mean'])} | {f(v['delta_fid_vs_E'][d]['mean'])} | "
                     f"{f(v['delta_acc_vs_E'][d]['mean'])} | {f(v['delta_fid_vs_control'][d]['mean'])} | {f(v['delta_acc_vs_control'][d]['mean'])} | "
                     f"{v['nodes'][d]:.1f} | {v['total_oracle_samples'].get(d, 0):.0f} ({v['original_oracle_samples'].get(d, 0):.0f}) | "
                     f"{dg['b1_switch_rate'].get(d, float('nan')):.2f} | {dg['b1_regret_median'].get(d, float('nan')):.3f} | {dg['b1_ratio_m_over_s_median'].get(d, float('nan')):.2f} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b1, ref = load_units(Path(a.b1)), load_units(Path(a.ref))
    ev, integ = evaluate(b1), integrity(b1, ref)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "b1_analysis.json").write_text(json.dumps({"integrity": integ, "evaluation": ev}, indent=1, default=str))
    (out / "B1_RESULTS.md").write_text(to_markdown(ev, integ), encoding="utf-8")
    print(to_markdown(ev, integ)[:3000])


if __name__ == "__main__":
    main()
