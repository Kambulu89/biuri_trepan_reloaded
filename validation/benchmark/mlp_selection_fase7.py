"""Seleção do MLP da fase 7 (EXPERIMENTAL, opt-in; não altera ``core/mlp_factory.py`` nem os resultados históricos).

Espaço CONGELADO no ``FASE7_PREREGISTO.md``: o perfil adaptativo por omissão (exatamente o MLP das fases anteriores) + 15 configurações lbfgs
(camadas {(16,), (64,), (32, 16)} × α {1e-3, 1e-2, 1e-1, 1, 10}). Seleção por validação cruzada estratificada de 5 dobras SÓ no treino
recebido (o escalonamento está dentro do pipeline, ajustado em cada dobra), critério = balanced accuracy média; empate (≤ 0,002) resolve-se
pelo α maior e depois pelo menor número de parâmetros. O modelo final é reajustado em todo o treino. Determinístico dada a semente.
"""
from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from core.mlp_factory import build_mlp_for_data

HIDDEN = [(16,), (64,), (32, 16)]
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
TIE = 0.002
INNER_FOLDS = 5


def candidates() -> List[Tuple[str, Dict[str, Any]]]:
    out = [("perfil_por_omissao", {})]
    for h in HIDDEN:
        for a in ALPHAS:
            out.append((f"lbfgs_h{'x'.join(map(str, h))}_a{a:g}", dict(solver="lbfgs", activation="relu", hidden_layer_sizes=h, alpha=a,
                                                                     max_iter=2000, tol=1e-5, early_stopping=False)))
    return out


def _n_params(model, n_features: int) -> int:
    sizes = [n_features] + list(model.named_steps["mlp"].hidden_layer_sizes) + [1]
    return int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])))


def select_and_fit(X, y, seed: int):
    t0 = time.perf_counter()
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    folds = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed).split(X, y))
    scores = []
    for name, params in candidates():
        vals = []
        for a, b in folds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = build_mlp_for_data(X[a], y[a], random_state=seed, **params).fit(X[a], y[a])
            vals.append(balanced_accuracy_score(y[b], m.predict(X[b])))
        probe = build_mlp_for_data(X, y, random_state=seed, **params)
        alpha = float(probe.named_steps["mlp"].alpha)
        scores.append(dict(name=name, params=params, cv_balanced_accuracy=float(np.mean(vals)), alpha=alpha, n_params=_n_params(probe, X.shape[1])))
    best = max(s["cv_balanced_accuracy"] for s in scores)
    tied = [s for s in scores if s["cv_balanced_accuracy"] >= best - TIE]
    chosen = sorted(tied, key=lambda s: (-s["alpha"], s["n_params"], s["name"]))[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_mlp_for_data(X, y, random_state=seed, **chosen["params"]).fit(X, y)
    info = {"trials": len(scores), "selection": "fase7_inner_cv_balanced_accuracy", "chosen": chosen["name"],
            "chosen_cv_balanced_accuracy": chosen["cv_balanced_accuracy"],
            "candidates": [{k: s[k] for k in ("name", "cv_balanced_accuracy", "alpha", "n_params")} for s in scores],
            "test_used": False, "train_time": time.perf_counter() - t0}
    return model, info
