"""Critério de elegibilidade da fase 6 (pré-registado): o MLP tem de superar o C4.5 em validação cruzada SÓ no treino de cada unidade.

Usa a mesma partição do benchmark (``split.json`` de cada unidade), o mesmo construtor de MLP do runner e o C4.5 canónico. O conjunto de
teste nunca é lido. Uso: python -m validation.benchmark.fase6_eligibility DIR_CAMPANHA SAIDA.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from validation.benchmark import run as R  # noqa: F401  (fixa o ambiente numérico)
from validation.benchmark import datasets as ds_mod
from validation.benchmark import datasets_categorical  # noqa: F401
from validation.benchmark import datasets_physics      # noqa: F401
from core.benchmark.runner import fit_mlp
from core.c45_j48_tree import C45Classifier

MARGIN = 0.02


def unit_gap(dataset: str, seed: int, split_file: Path) -> dict:
    ds, _, _ = ds_mod.REGISTRY[dataset].load()
    # só os ÍNDICES de teste são lidos, para obter o complemento (treino); rótulos e previsões de teste não são usados
    test_idx = json.loads((split_file.parent / "mlp_original.json").read_text(encoding="utf-8"))["predictions"]["test_row_index"]
    tr = np.setdiff1d(np.arange(len(ds.y)), np.asarray(test_idx, dtype=int))
    X, y = ds.X[tr], ds.y[tr]
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    mlp_acc, c45_acc = [], []
    for a, b in folds.split(X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp, _ = fit_mlp(X[a], y[a], seed, 0, 3)
        mlp_acc.append(float(np.mean(mlp.predict(X[b]) == y[b])))
        c45 = C45Classifier(confidence_factor=0.25, min_samples_leaf=2, random_state=seed).fit(X[a], y[a])
        c45_acc.append(float(np.mean(c45.predict(X[b]) == y[b])))
    return {"dataset": dataset, "seed": seed, "mlp_cv_accuracy": float(np.mean(mlp_acc)), "c45_cv_accuracy": float(np.mean(c45_acc)),
            "gap": float(np.mean(mlp_acc) - np.mean(c45_acc)), "test_used": False}


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [unit_gap(f.parent.parent.name, int(f.parent.name[4:]), f) for f in sorted(root.glob("raw/*/seed*/split.json"))]
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r["gap"])
    summary = {d: {"mean_gap": float(np.mean(g)), "units": len(g), "eligible": bool(np.mean(g) > MARGIN)} for d, g in by_ds.items()}
    out.write_text(json.dumps({"margin": MARGIN, "per_unit": rows, "per_dataset": summary}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
