"""Smoke test end-to-end da construção de árvores (C4.5, TREPAN Original, TREPAN Reloaded).

Protocolo fixo e declarado ANTES de olhar para os resultados: mesma divisão
treino/teste, mesma seed, mesmo budget, mesma profundidade, mesmo max_nodes,
mesma poda. O teste só é usado para medir, nunca para escolher nada.
Agnóstico a datasets: recebe um CSV (colunas numéricas; última coluna = alvo)
ou, sem argumento, gera dados sintéticos. Nenhum dataset é conhecido pelo código.
Uso: python scripts/tree_validation_smoke.py [--csv ficheiro.csv] [--out saida.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import make_classification
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


def load_data(csv: str | None):
    """(X, y, nomes). CSV genérico (última coluna = alvo) ou sintético."""
    if csv:
        import pandas as pd
        frame = pd.read_csv(csv)
        names = [str(c) for c in frame.columns[:-1]]
        X = frame.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        y = pd.factorize(frame.iloc[:, -1])[0]
        return X, y, names, f"csv:{Path(csv).name}"
    X, y = make_classification(n_samples=600, n_features=12, n_informative=5, n_redundant=2,
                               flip_y=0.03, random_state=SEED)
    return X, y, [f"x{i}" for i in range(X.shape[1])], "synthetic"


def main(csv: str | None = None, out: str = "tree_validation_results.json") -> None:
    Xraw, y, names, source = load_data(csv)
    X = StandardScaler().fit_transform(Xraw)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    mlp = MLPClassifier((32,), max_iter=1000, random_state=SEED).fit(Xtr, ytr)
    oracle_acc = float(np.mean(mlp.predict(Xte) == yte))
    y_oracle_te = mlp.predict(Xte)
    results: dict = {"dataset": source, "seed": SEED, "protocol": PROTOCOL,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--out", default="tree_validation_results.json")
    a = ap.parse_args()
    main(a.csv, a.out)
