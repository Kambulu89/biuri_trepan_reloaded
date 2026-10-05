#!/usr/bin/env python3
"""Ablação de engenharia V9.2 sem tocar no protocolo confirmatório congelado.

Neste ambiente, se owlready2/HermiT não estiver disponível, executa apenas o
braço controlado sem OWL e grava explicitamente o braço OWL como não executado.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.ablation_study import builtin_benchmark_datasets
from core.c45_j48_tree import C45Classifier
from core.controlled_trepan_experiment import (
    ControlledTrepanConfig,
    evaluate_controlled_trepan_pair,
    fit_controlled_trepan_pair,
    oracle_health_gate,
)
from core.mlp_factory import build_mlp_for_data

OUT = ROOT / "results" / "ablation_v9_2"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
owl_available = importlib.util.find_spec("owlready2") is not None

for name, (X, y, names) in builtin_benchmark_datasets().items():
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    names = [str(v) for v in names]
    seed = 42

    # Amostra estratificada quando o benchmark é grande; nunca pega simplesmente
    # as primeiras linhas.
    indices = np.arange(len(y))
    if len(indices) > 300:
        indices, _ = train_test_split(
            indices, train_size=300, random_state=seed, stratify=y
        )
    X_sub, y_sub = X[indices], y[indices]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.25, random_state=seed, stratify=y_sub
    )

    try:
        oracle_builder = build_mlp_for_data(X_train, y_train, random_state=seed)
        health = oracle_health_gate(
            oracle_builder,
            X_train,
            y_train,
            random_state=seed,
            cv_folds=3,
            dummy_margin=0.02,
            c45_margin=0.10,
        )
        oracle = build_mlp_for_data(X_train, y_train, random_state=seed).fit(X_train, y_train)

        min_sample = max(len(X_train), min(500, max(100, len(X_train) * 2)))
        config = ControlledTrepanConfig(
            max_nodes=7,
            max_depth=3,
            min_samples_leaf=2,
            min_sample=min_sample,
            max_n=2,
            beam_width=2,
            max_features_per_node=10,
            max_queries=max(700, min_sample * 2),
            random_state=seed,
        )
        pair = fit_controlled_trepan_pair(
            X_train,
            y_train,
            oracle=oracle,
            feature_names=names,
            config=config,
            run_id=f"ablation_v9_2_no_owl:{name}",
        )
        evaluation = evaluate_controlled_trepan_pair(pair, X_test, y_test)
        c45 = C45Classifier(
            confidence_factor=0.25, min_samples_leaf=2, random_state=seed
        ).fit(X_train, y_train)
        c45_pred = c45.predict(X_test)
        mlp_pred = oracle.predict(X_test)
        c45_metrics = {
            "accuracy": float(np.mean(c45_pred == y_test)),
            "fidelity_to_mlp": float(np.mean(c45_pred == mlp_pred)),
        }
        rows.append({
            "dataset": name,
            "arm": "without_owl",
            "status": "ok" if health.get("valid") else "oráculo_inválido",
            "oracle_health": health,
            "controlled_protocol": evaluation["experiment_audit"],
            "protocol_audit": evaluation["protocol_audit"],
            "metrics": evaluation["models"],
            "comparison_neutral": evaluation["comparison"],
            "c45_native": c45_metrics,
        })
    except Exception as exc:
        rows.append({
            "dataset": name,
            "arm": "without_owl",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        })

    rows.append({
        "dataset": name,
        "arm": "with_owl",
        "status": "not_executed",
        "reason": (
            "owlready2/HermiT indisponível no ambiente de validação"
            if not owl_available
            else "OWL disponível, mas execução real deve usar run_dataset_ablation com quality gate/reasoner"
        ),
    })

(OUT / "ablation_rows.json").write_text(
    json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
)
summary = {
    "protocol": "controlled_same_oracle_same_seed_same_budget_v9_2",
    "datasets": len(set(r["dataset"] for r in rows)),
    "without_owl_ok": sum(
        r["arm"] == "without_owl" and r["status"] == "ok" for r in rows
    ),
    "without_owl_invalid_oracle": sum(r["status"] == "oráculo_inválido" for r in rows),
    "without_owl_failed": sum(r["arm"] == "without_owl" and r["status"] == "failed" for r in rows),
    "with_owl_executed": 0,
    "owlready2_available": owl_available,
    "note": "Nenhum efeito OWL foi calculado porque o braço OWL não foi executado.",
}
(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False))
