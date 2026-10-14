"""Análise do diagnóstico DEV para o desenho de B2 (exploratória; observed_only; sem seleção por teste).
    python -m validation.benchmark.diag_b2_analysis --dir DIR [--out FILE.json]
Decomposição: Δ(arm − trepan_original) em fidelidade/accuracy, emparelhado por (dataset, semente); estatísticas de aceitação por nó e tempos por componente.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("trepan_original", "diag_original_enr", "reloaded_core", "reloaded_owl_features", "reloaded_semantic_score", "diag_gate_off_active_off",
        "diag_gate_off", "diag_active_off", "reloaded_owl_full", "reloaded_owl_shuffled")
TIMES = ("query_time_", "synthetic_generation_time_", "oracle_prediction_time_", "split_search_time_", "m_of_n_search_time_", "local_model_time_", "training_time_")


def load(root: Path):
    units = {}
    for ud in sorted((root / "raw").glob("*/seed*")):
        key = (ud.parent.name, int(ud.name[4:]))
        arms = {f.stem: json.loads(f.read_text())["row"] for f in ud.glob("*.json") if f.name not in ("split.json", "UNIT_SHA256.json")}
        units[key] = arms
    return units


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.dir)
    units = load(root)
    dss = sorted({k[0] for k in units})
    res = {"units": sorted(f"{k[0]}:{k[1]}" for k in units), "decomposition": {}, "gate": {}, "timing": {}}
    for arm in ARMS:
        per_ds = {}
        for ds in dss:
            d = {"dF": [], "dAcc": [], "nodes": [], "samples_ratio": [], "n": 0}
            for (d_, s), arms in units.items():
                if d_ != ds or arm not in arms or "trepan_original" not in arms:
                    continue
                o, r = arms["trepan_original"], arms[arm]
                d["dF"].append(r["fidelity_to_oracle"] - o["fidelity_to_oracle"])
                d["dAcc"].append(r["accuracy_real_labels"] - o["accuracy_real_labels"])
                d["nodes"].append(r["node_count"] - o["node_count"])
                if o.get("oracle_queries"):
                    d["samples_ratio"].append((r.get("oracle_queries") or 0) / o["oracle_queries"])
                d["n"] += 1
            if d["n"]:
                per_ds[ds] = {k: (mean(v) if isinstance(v, list) else v) for k, v in d.items()}
        if per_ds:
            res["decomposition"][arm] = {"per_dataset": per_ds, "macro_dF": mean([v["dF"] for v in per_ds.values()]),
                                         "macro_dAcc": mean([v["dAcc"] for v in per_ds.values()]), "macro_dNodes": mean([v["nodes"] for v in per_ds.values()]),
                                         "macro_samples_ratio": mean([v["samples_ratio"] for v in per_ds.values()])}
    # gate (auditorias de E)
    rows = []
    for f in sorted((root / "audits").glob("*__reloaded_owl_full.json")):
        rec = json.loads(f.read_text())
        for r in rec["rows"]:
            rows.append((rec["unit"], r))
    att = [r for _, r in rows if r.get("error_focused_intervention_attempted")]
    acc = [r for r in att if r.get("error_focused_intervention_accepted")]
    rej = [r for r in att if not r.get("error_focused_intervention_accepted")]
    g = {"nodes": len(rows), "proposed_changed": sum(bool(r.get("proposed_decision_changed")) for _, r in rows), "attempted": len(att), "accepted": len(acc),
         "accepted_within_tolerance": sum(bool(r.get("error_focused_accepted_within_tolerance")) for r in acc),
         "reasons": dict(Counter(r.get("error_focused_intervention_reason") for r in att)),
         "real_gate_evaluated": sum(bool(r.get("real_training_gate_evaluated")) for r in att),
         "local_gain_accepted_mean": mean([r.get("local_fidelity_gain") for r in acc]), "local_gain_rejected_mean": mean([r.get("local_fidelity_gain") for r in rej]),
         "real_gain_accepted_mean": mean([r.get("real_fidelity_gain") for r in acc]), "real_gain_rejected_mean": mean([r.get("real_fidelity_gain") for r in rej]),
         "accepted_local_gain_below_0.01": sum((r.get("local_fidelity_gain") or 0) < 0.01 for r in acc),
         "accepted_real_gain_nonpositive": sum((r.get("real_fidelity_gain") is not None and r["real_fidelity_gain"] <= 1e-12) for r in acc),
         "decision_sample_size_median": st.median([r["decision_sample_size"] for _, r in rows if r.get("decision_sample_size")] or [0]),
         "semantic_candidates_median": st.median([r["semantic_candidates_evaluated"] for _, r in rows if r.get("semantic_candidates_evaluated") is not None] or [0])}
    res["gate"] = g
    # tempos
    tm = defaultdict(lambda: defaultdict(list))
    for f in sorted((root / "audits").glob("*.json")):
        rec = json.loads(f.read_text())
        for k in TIMES:
            tm[(rec["arm"])][k].append(rec["timing"].get(k, 0.0))
    res["timing"] = {arm: {k: mean(v) for k, v in d.items()} for arm, d in tm.items()}
    out = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(out)
    print(out)


if __name__ == "__main__":
    main()
