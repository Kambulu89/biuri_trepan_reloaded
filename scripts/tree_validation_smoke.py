"""Smoke test end-to-end da construção de árvores (C4.5, TREPAN Original, TREPAN Reloaded).

Protocolo fixo e declarado ANTES de olhar para os resultados: mesma divisão
treino/teste, mesma seed, mesmo budget, mesma profundidade, mesmo max_nodes,
mesma poda. O teste só é usado para medir, nunca para escolher nada.
Uso: python scripts/tree_validation_smoke.py [saida.json]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from core.c45_j48_tree import C45Classifier
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier
from core.tree_build_report import c45_build_report, format_report, trepan_build_report

SEED = 42
PROTOCOL = dict(min_sample=1000, max_nodes=31, max_depth=8, max_n=3, beam_width=2, random_state=SEED)
BUDGETS = (2000, 5000, 10000, 20000)  # sensibilidade declarada a priori
MAIN_BUDGET = 10000


def main(out: str = "tree_validation_results.json") -> None:
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data)
    Xtr, Xte, ytr, yte = train_test_split(X, data.target, test_size=0.3, random_state=SEED, stratify=data.target)
    mlp = MLPClassifier((32,), max_iter=1000, random_state=SEED).fit(Xtr, ytr)
    oracle_acc = float(np.mean(mlp.predict(Xte) == yte))
    names = list(data.feature_names)
    y_oracle_te = mlp.predict(Xte)
    results: dict = {"dataset": "sklearn breast_cancer (smoke)", "seed": SEED, "protocol": PROTOCOL,
                     "oracle_accuracy_test": oracle_acc, "runs": {}, "budget_sensitivity": []}

    t0 = time.perf_counter()
    c45 = C45Classifier(random_state=SEED).fit(Xtr, ytr)
    rep = c45_build_report(c45, X_eval=Xte, y_real_eval=yte, oracle_eval=y_oracle_te, oracle_name="MLP Original")
    rep["TIME"] = {"training_time": time.perf_counter() - t0}
    results["runs"]["C4.5"] = rep

    for label, cls, extra in (
        ("TREPAN Original", TrepanOriginalClassifier, {}),
        ("TREPAN Reloaded (semantics OFF, same oracle)", TrepanReloadedClassifier, {"alpha": 0.0, "beta": 0.0}),
    ):
        tree = cls(max_queries=MAIN_BUDGET, **PROTOCOL, **extra).fit(Xtr, oracle=mlp, feature_names=names)
        results["runs"][label] = trepan_build_report(
            tree, algorithm=label, oracle_name="MLP Original", oracle_type="MLPClassifier",
            X_eval=Xte, y_oracle_eval=y_oracle_te, y_real_eval=yte, oracle_accuracy=oracle_acc)

    for budget in BUDGETS:
        tree = TrepanOriginalClassifier(max_queries=budget, **PROTOCOL).fit(Xtr, oracle=mlp, feature_names=names)
        r = trepan_build_report(tree, algorithm="TREPAN Original", oracle_name="MLP Original",
                                X_eval=Xte, y_oracle_eval=y_oracle_te, y_real_eval=yte)
        results["budget_sensitivity"].append({
            "query_budget": budget, "nodes": r["STRUCTURE"]["nodes_final"], "depth": r["STRUCTURE"]["depth"],
            "queries_used": r["CONSTRUCTION"]["query_budget_used"],
            "budget_exhausted": r["CONSTRUCTION"]["query_budget_exhausted"],
            "stop_reasons": r["STOP_REASONS"], "fidelity": r["METRICS"]["fidelity"],
            "accuracy_real": r["METRICS"]["accuracy_real"]})

    Path(out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    for name, rep in results["runs"].items():
        print("=" * 70); print(format_report(rep))
    print("=" * 70); print("BUDGET SENSITIVITY (TREPAN Original)")
    for row in results["budget_sensitivity"]:
        print(row)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tree_validation_results.json")
