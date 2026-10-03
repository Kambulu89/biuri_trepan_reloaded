#!/usr/bin/env python3
"""Professor semântico + Reloaded aumentado em dados reais (ontologias de benchmark do repositório).

Três braços por semente, mesmos dados/semente: professor original; professor semântico com árvores no
espaço original; professor semântico com Reloaded no espaço aumentado. O professor semântico só é usado
se o gate o aceitar (aqui ``teacher_min_evidence`` configurável; "weak" permite aceitações marginais).

    python scripts/run_teacher_real_data.py --dataset breast_cancer --seeds 1,2,3 --out out.json
"""
from __future__ import annotations
import argparse, json, sys, tempfile, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="breast_cancer"); ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--evidence", default="weak"); ap.add_argument("--max-samples", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(); warnings.filterwarnings("ignore")
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from core.production_training import train_production_dataframe
    from core.semantic_enrichment import EnrichmentConfig
    bunch = getattr(datasets, {"breast_cancer": "load_breast_cancer", "wine": "load_wine", "iris": "load_iris"}[args.dataset])(as_frame=True)
    X = bunch.data.copy(); X.columns = [str(c) for c in X.columns]
    y = bunch.target.to_numpy()
    if len(X) > args.max_samples:
        X, _, y, _ = train_test_split(X, y, train_size=args.max_samples, random_state=0, stratify=y)
    df = X.reset_index(drop=True); df["target"] = np.where(y == 1, "c1", np.where(y == 0, "c0", "c2"))
    owl = ROOT / "data" / "benchmark_ontologies" / f"{args.dataset}.owl"
    out = {"dataset": args.dataset, "evidence_required": args.evidence, "runs": []}
    for seed in [int(s) for s in args.seeds.split(",")]:
        cfg = EnrichmentConfig(cv_folds=5, tuning_candidates=4, random_state=seed)
        arms = (("original_teacher", False, True), ("semantic_teacher_original_space", True, False),
                ("semantic_teacher_augmented", True, True))
        row = {"seed": seed}
        for label, use, augment in arms:
            with tempfile.TemporaryDirectory() as tmp:
                rep = train_production_dataframe(df, target="target", out_dir=tmp, seed=seed, owl_path=owl,
                                                 require_reasoner=True, scientific_tuning=False,
                                                 semantic_enrichment=cfg, use_semantic_teacher=use,
                                                 teacher_min_evidence=args.evidence, augment_reloaded_space=augment)
            ev = rep["evaluation"]
            row[label] = {
                "teacher": ev["semantic_teacher"]["teacher"], "reason": ev["semantic_teacher"]["reason"],
                "decision": (ev["semantic_enrichment"] or {}).get("decision"),
                "evidence": (ev["semantic_enrichment"] or {}).get("evidence_strength"),
                "space": ev["reloaded_feature_space"]["space"],
                "mlp_original": ev["models"]["mlp_original"], "mlp_semantic": ev["models"].get("mlp_semantic"),
                "trepan_original": ev["models"]["original"], "trepan_reloaded": ev["models"]["reloaded"],
            }
        out["runs"].append(row)
        a = row["semantic_teacher_augmented"]
        print(f"{args.dataset} seed={seed} | {a['decision']} ({a['evidence']}) teacher={a['teacher']} space={a['space']} | "
              f"BA Orig(sem) {a['trepan_original']['balanced_accuracy']:.3f} Reloaded(aug) {a['trepan_reloaded']['balanced_accuracy']:.3f}",
              flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
