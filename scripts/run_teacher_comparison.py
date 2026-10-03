#!/usr/bin/env python3
"""Compara o pipeline de produção com e sem professor semântico (MLP+OWL) no TREPAN.

Cenário sintético declarado, com a relação worst/mean escondida entre atributos irrelevantes
(onde o enriquecimento tem evidência forte), mais um cenário "fácil" onde NÃO deve ser usado.
Mesma semente/dados nos dois braços; o teste externo só entra nas métricas finais.

    python scripts/run_teacher_comparison.py --out results/semantic_validation/teacher_comparison.json
"""
from __future__ import annotations
import argparse, json, sys, tempfile, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd


def make_ontology(noise):
    import owlready2
    onto = owlready2.World().get_ontology("http://example.org/teacher-cmp.owl")
    with onto:
        type("statisticRole", (owlready2.AnnotationProperty,), {})
        type("measurementFamily", (owlready2.AnnotationProperty,), {})
        Morph = type("Morphology", (owlready2.Thing,), {}); Bg = type("Background", (owlready2.Thing,), {})
        for role in ("mean", "error", "worst"):
            p = type(f"hasSize{role.capitalize()}", (owlready2.DataProperty,), {"range": [float], "domain": [Morph]})
            p.statisticRole = [role]; p.measurementFamily = ["Size"]
        for i in range(noise):
            type(f"hasBackgroundVariable{i:02d}Level", (owlready2.DataProperty,), {"range": [float], "domain": [Bg]})
    return onto


def make_frame(n, noise, signal, seed):
    rng = np.random.default_rng(seed)
    mean = rng.uniform(5, 15, n); delta = rng.uniform(0, 1, n)
    df = pd.DataFrame({"hasSizeMean": mean, "hasSizeError": rng.uniform(.1, 1, n), "hasSizeWorst": mean * (1 + delta)})
    for i in range(noise):
        df[f"hasBackgroundVariable{i:02d}Level"] = rng.normal(size=n)
    base = delta if signal == "relation" else mean - mean.mean()
    y = (base > np.median(base)).astype(int); y = np.where(rng.random(n) < .03, 1 - y, y)
    df["target"] = np.where(y == 1, "yes", "no")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/semantic_validation/teacher_comparison.json")
    ap.add_argument("--seeds", default="1,2,3,4,5"); ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args(); warnings.filterwarnings("ignore")
    from core.production_training import train_production_dataframe
    from core.semantic_enrichment import EnrichmentConfig
    cfg = EnrichmentConfig(cv_folds=3, tuning_candidates=3, tuning_inner_folds=2, max_iter=200, n_bootstrap=300)
    out = {"scenarios": {}}
    for scenario, noise, signal in (("hidden_relation_among_noise", 20, "relation"), ("easy_no_noise", 0, "relation")):
        rows = []
        for seed in [int(s) for s in args.seeds.split(",")]:
            df = make_frame(args.n, noise, signal, seed)
            res = {}
            arms = (("original_teacher", False, True),            # MLP original, árvores no espaço original
                    ("semantic_teacher_original_space", True, False),  # professor semântico, Reloaded no espaço original
                    ("semantic_teacher_augmented", True, True))        # professor semântico, Reloaded no espaço aumentado
            for label, use, augment in arms:
                with tempfile.TemporaryDirectory() as tmp:
                    rep = train_production_dataframe(
                        df, target="target", out_dir=tmp, seed=seed, ontology=make_ontology(noise),
                        require_reasoner=False, scientific_tuning=False,
                        semantic_enrichment=cfg, use_semantic_teacher=use, augment_reloaded_space=augment)
                ev = rep["evaluation"]
                res[label] = {
                    "teacher_used": ev["semantic_teacher"]["teacher"], "reason": ev["semantic_teacher"]["reason"],
                    "reloaded_space": ev["reloaded_feature_space"]["space"],
                    "decision": (ev["semantic_enrichment"] or {}).get("decision"),
                    "evidence": (ev["semantic_enrichment"] or {}).get("evidence_strength"),
                    "mlp_original": ev["models"]["mlp_original"], "mlp_semantic": ev["models"].get("mlp_semantic"),
                    "trepan_original": ev["models"]["original"], "trepan_reloaded": ev["models"]["reloaded"],
                }
            rows.append({"seed": seed, **res})
        out["scenarios"][scenario] = rows
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for sc, rows in out["scenarios"].items():
        for r in rows:
            o, s2, a = r["original_teacher"], r["semantic_teacher_original_space"], r["semantic_teacher_augmented"]
            b = lambda x: x["trepan_reloaded"]["balanced_accuracy"]
            fid = lambda x: x["trepan_reloaded"]["oracle_fidelity"]
            print(f"{sc} seed={r['seed']} | decision {a['decision']} ({a['evidence']}) teacher={a['teacher_used']} "
                  f"space={a['reloaded_space']} | Reloaded BA: orig-teacher {b(o):.3f} | sem-teacher/orig-space {b(s2):.3f} "
                  f"| sem-teacher/augmented {b(a):.3f} | fid(augm) {fid(a):.3f}")


if __name__ == "__main__":
    main()
