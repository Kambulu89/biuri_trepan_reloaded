#!/usr/bin/env python3
"""Validação do gate de atribuição: o ganho do Reloaded tem de vir da ontologia.

* ``--mode real``      : datasets reais com a ontologia de benchmark (gate ligado).
* ``--mode synthetic`` : cenário declarado em que a ontologia codifica uma relação escondida
                         (professor semântico + espaço aumentado + gate).

Por semente reporta o modo escolhido, o ganho do Reloaded sobre o Original e, quando há semântica
ativa, o ganho ATRIBUÍVEL (real - controlos sem significado) medido uma vez no teste final.

    python scripts/run_attribution_gate_validation.py --mode real --dataset breast_cancer --out out.json
"""
from __future__ import annotations
import argparse, json, sys, tempfile, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
import numpy as np


def summarize(rep, seed, label):
    ev = rep["evaluation"]
    att = ev.get("semantic_attribution") or {}
    t = att.get("test_attribution") or {}
    row = {
        "seed": seed, "label": label, "mode": rep["manifest"]["reloaded_mode"],
        "ba_original": ev["models"]["original"]["balanced_accuracy"],
        "ba_reloaded": ev["models"]["reloaded"]["balanced_accuracy"],
        "fid_original": ev["models"]["original"]["oracle_fidelity"],
        "fid_reloaded": ev["models"]["reloaded"]["oracle_fidelity"],
        "gate": [{k: c.get(k) for k in ("mode", "status", "mean_gain_over_controls", "wins", "losses", "ties")}
                 for c in att.get("candidates", [])],
        "attributable_ba": t.get("attributable_balanced_accuracy"),
        "attributable_fid": t.get("attributable_oracle_fidelity"),
        "controls_ge_real_ba": t.get("controls_ge_real_balanced_accuracy"),
    }
    row["gain_reloaded_vs_original_ba"] = row["ba_reloaded"] - row["ba_original"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synthetic"], required=True)
    ap.add_argument("--dataset", default="breast_cancer"); ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--controls", type=int, default=3); ap.add_argument("--max-samples", type=int, default=800)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(); warnings.filterwarnings("ignore")
    from core.production_training import train_production_dataframe
    from core.semantic_attribution import AttributionConfig
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        gate = AttributionConfig(controls=args.controls, inner_folds=3, random_state=seed)
        with tempfile.TemporaryDirectory() as tmp:
            if args.mode == "real":
                from run_reloaded_attribution import load
                df = load(args.dataset, args.max_samples)
                owl = ROOT / "data" / "benchmark_ontologies" / f"{args.dataset}.owl"
                rep = train_production_dataframe(df, target="target", out_dir=tmp, seed=seed, owl_path=owl,
                                                 require_reasoner=True, scientific_tuning=False,
                                                 semantic_attribution=gate)
            else:
                from run_teacher_comparison import make_frame, make_ontology
                from core.semantic_enrichment import EnrichmentConfig
                df = make_frame(500, 20, "relation", seed)
                enrich = EnrichmentConfig(cv_folds=3, tuning_candidates=3, tuning_inner_folds=2, max_iter=200,
                                          n_bootstrap=300, random_state=seed)
                rep = train_production_dataframe(df, target="target", out_dir=tmp, seed=seed, ontology=make_ontology(20),
                                                 require_reasoner=False, scientific_tuning=False,
                                                 semantic_enrichment=enrich, semantic_attribution=gate)
        r = summarize(rep, seed, args.dataset if args.mode == "real" else "synthetic_hidden_relation")
        rows.append(r)
        print(f"{r['label']} seed={seed} | mode={r['mode']} | Reloaded-Original BA {r['gain_reloaded_vs_original_ba']:+.3f}"
              f" | attributable(test) BA {r['attributable_ba']} fid {r['attributable_fid']}", flush=True)
    G = [r["gain_reloaded_vs_original_ba"] for r in rows]
    A = [r["attributable_ba"] for r in rows if r["attributable_ba"] is not None]
    summary = {"n_seeds": len(rows), "modes": {m: sum(r["mode"] == m for r in rows) for m in {r["mode"] for r in rows}},
               "mean_gain_reloaded_vs_original_ba": float(np.mean(G)),
               "mean_attributable_ba_when_semantics_active": float(np.mean(A)) if A else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
