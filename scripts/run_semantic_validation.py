#!/usr/bin/env python3
"""Validação experimental do enriquecimento semântico (reprodutível, sem teste externo).

Para cada dataset corre, sobre o conjunto de desenvolvimento apenas:

* a avaliação completa MLP base vs MLP + features OWL (``evaluate_semantic_enrichment``);
* a ablação por componente (agregados / relacionais / restrições / inferidas pelo reasoner);
* o controlo negativo: a mesma ontologia com a semântica baralhada (R repetições).

Os resultados são escritos tal como saem; nada é ajustado depois de os ver.
Os datasets são só dados de entrada deste script de validação, não do core.

    python scripts/run_semantic_validation.py --out results/semantic_validation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

LOADERS = {"iris": "load_iris", "wine": "load_wine", "breast_cancer": "load_breast_cancer", "digits": "load_digits"}
ONTOLOGY_DIR = ROOT / "data" / "benchmark_ontologies"
ABLATIONS = [
    ("B_aggregates_only", {"kinds": ["hierarchical_aggregate"]}),
    ("C_relational_only", {"kinds": ["relational"]}),
    ("D_constraints_only", {"kinds": ["constraint", "categorical_group"]}),
    ("E_reasoner_inferred_only", {"reasoner_inferred_only": True}),
]


def load_dataset(name: str, max_samples: int, seed: int):
    from sklearn import datasets
    bunch = getattr(datasets, LOADERS[name])(as_frame=True)
    X = bunch.data.copy(); X.columns = [str(c) for c in X.columns]
    y = np.asarray(bunch.target)
    if len(X) > max_samples:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(X, y, train_size=max_samples, random_state=seed, stratify=y)
    return X.reset_index(drop=True), y


def summarize(report: dict) -> dict:
    stages = report.get("stages", {})
    d = stages.get("D_mlp_comparison") or {}
    return {
        "decision": report.get("decision"), "reason": report.get("decision_reason"),
        "semantic_mlp_accepted": report.get("semantic_mlp_accepted"),
        "semantic_trepan_available": report.get("semantic_trepan_available"),
        "generated": (stages.get("B_novelty") or {}).get("generated"),
        "retained": (stages.get("B_novelty") or {}).get("retained"),
        "stable_features": (stages.get("C_screening") or {}).get("stable_features"),
        "selected_features": report.get("selected_semantic_features"),
        "base": d.get("base"), "with_owl": d.get("with_owl"),
        "utility_gain": d.get("utility_gain"), "utility_gain_ci": d.get("utility_gain_ci"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/semantic_validation")
    ap.add_argument("--datasets", default="iris,wine,breast_cancer,digits")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--max-samples", type=int, default=800)
    ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tuning-candidates", type=int, default=4)
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from owlready2 import World
    from core.ontology_quality import OntologyQualityGate
    from core.ontology_reasoner import run_owl_reasoner
    from core.semantic_controls import shuffle_entity_assignments
    from core.semantic_enrichment import EnrichmentConfig, evaluate_semantic_enrichment, export_feature_audit
    from core.semantic_version import SEMANTIC_PIPELINE_VERSION

    out = ROOT / args.out if not os.path.isabs(args.out) else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"semantic_pipeline_version": SEMANTIC_PIPELINE_VERSION, "datasets": {}}
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            t0 = time.time()
            X, y = load_dataset(name, args.max_samples, seed)
            onto = World().get_ontology(str((ONTOLOGY_DIR / f"{name}.owl").resolve())).load()
            reasoner = run_owl_reasoner(onto)
            quality = OntologyQualityGate().evaluate(list(X.columns), onto, reasoner_report=reasoner,
                                                     require_reasoner=True)
            cfg = EnrichmentConfig(cv_folds=args.folds, random_state=seed,
                                   tuning_candidates=args.tuning_candidates)
            row = {
                "n_samples": int(len(X)), "n_features": int(X.shape[1]), "n_classes": int(len(np.unique(y))),
                "quality": {k: quality.metrics.get(k) for k in (
                    "feature_coverage", "mapped_features", "total_features", "ambiguous_matches",
                    "entity_collisions", "generic_match_ratio", "mapping_entropy", "ontology_depth",
                    "semantic_richness")} | {"status": quality.status, "abox": quality.abox.get("status")},
                "reasoner": {k: reasoner.get(k) for k in (
                    "executed", "consistent", "reasoner_used", "duration_seconds", "inferred_axioms_count")},
            }
            full = evaluate_semantic_enrichment(X, y, onto, quality_report=quality, reasoner_report=reasoner, config=cfg)
            row["full"] = summarize(full.report)
            export_feature_audit(full.report, out / f"feature_audit_{name}_seed{seed}.csv")
            if not args.no_ablation:
                row["ablation"] = {}
                for label, kw in ABLATIONS:
                    row["ablation"][label] = summarize(evaluate_semantic_enrichment(
                        X, y, onto, quality_report=quality, reasoner_report=reasoner, config=cfg, **kw).report)
            controls = []
            for k in range(args.controls):
                rep = evaluate_semantic_enrichment(
                    X, y, onto, quality_report=quality, reasoner_report=reasoner, config=cfg,
                    matches_transform=lambda m, s=seed * 100 + k: shuffle_entity_assignments(m, seed=s))
                controls.append(summarize(rep.report))
            row["negative_control_shuffled"] = controls
            gains = [c["utility_gain"] for c in controls if c["utility_gain"] is not None]
            real = row["full"]["utility_gain"]
            row["control_summary"] = {
                "real_utility_gain": real, "control_gains": gains,
                "controls_ge_real": int(sum(g >= real for g in gains)) if real is not None else None,
            }
            row["seconds"] = round(time.time() - t0, 1)
            results["datasets"][f"{name}@seed{seed}"] = row
            (out / "semantic_validation.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            print(f"[{name} seed={seed}] {row['full']['decision']} gain={real} ({row['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
