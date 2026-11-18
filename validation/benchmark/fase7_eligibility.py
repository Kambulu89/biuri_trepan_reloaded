"""Elegibilidade da fase 7 (pré-registada): validação cruzada ANINHADA só no treino de cada unidade — em cada dobra externa, o MLP é
escolhido por ``mlp_selection_fase7.select_and_fit`` com dobras internas e comparado com o C4.5 canónico. O teste nunca é lido (só os índices).
Uso: python -m validation.benchmark.fase7_eligibility DIR_CAMPANHA SAIDA.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from validation.benchmark import run as R  # noqa: F401
from validation.benchmark import datasets as ds_mod
from validation.benchmark import datasets_categorical  # noqa: F401
from validation.benchmark import datasets_physics      # noqa: F401
from validation.benchmark.mlp_selection_fase7 import select_and_fit
from core.c45_j48_tree import C45Classifier

MARGIN = 0.02


def unit_gap(dataset: str, seed: int, unit_dir: Path) -> dict:
    ds, _, _ = ds_mod.REGISTRY[dataset].load()
    test_idx = json.loads((unit_dir / "mlp_original.json").read_text(encoding="utf-8"))["predictions"]["test_row_index"]
    tr = np.setdiff1d(np.arange(len(ds.y)), np.asarray(test_idx, dtype=int))
    X, y = ds.X[tr], ds.y[tr]
    mlp_acc, c45_acc, chosen = [], [], []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        model, info = select_and_fit(X[a], y[a], seed)
        mlp_acc.append(float(np.mean(model.predict(X[b]) == y[b]))); chosen.append(info["chosen"])
        c45 = C45Classifier(confidence_factor=0.25, min_samples_leaf=2, random_state=seed).fit(X[a], y[a])
        c45_acc.append(float(np.mean(c45.predict(X[b]) == y[b])))
    return {"dataset": dataset, "seed": seed, "mlp_nested_cv_accuracy": float(np.mean(mlp_acc)), "c45_cv_accuracy": float(np.mean(c45_acc)),
            "gap": float(np.mean(mlp_acc) - np.mean(c45_acc)), "chosen_per_outer_fold": chosen, "test_used": False}


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for unit in sorted(root.glob("raw/*/seed*")):
        rows.append(unit_gap(unit.parent.name, int(unit.name[4:]), unit)); print(json.dumps(rows[-1]), flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r["gap"])
    summary = {d: {"mean_gap": float(np.mean(g)), "units": len(g), "eligible": bool(np.mean(g) > MARGIN)} for d, g in by.items()}
    out.write_text(json.dumps({"margin": MARGIN, "per_unit": rows, "per_dataset": summary}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
