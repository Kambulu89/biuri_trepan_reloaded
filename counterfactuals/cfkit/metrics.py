"""Métricas contrafactuais: proximidade (Gower), esparsidade, plausibilidade (kNN no treino), diversidade, dominância.

Tudo é ajustado SÓ no treino (``fit``). X_test nunca define escalas, densidade ou thresholds.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from counterfactuals.cfkit.schema import EPS, FeatureSchema


class GowerMetric:
    """Distância de Gower sobre unidades humanas (grupo one-hot = 1 unidade, 0/1).

    Contínuas/inteiras: |Δ| / amplitude de treino; binárias/códigos/one-hot: 0 ou 1.
    Evita a distância euclidiana crua em espaços mistos."""

    def __init__(self, schema: FeatureSchema):
        self.schema = schema
        self.fitted = False

    def fit(self, X_train) -> "GowerMetric":
        X = np.asarray(X_train, dtype=float)
        spans, weights = [], []
        for unit in self.schema.units:
            spec = self.schema.specs[unit.indices[0]]
            if unit.kind in ("continuous", "integer"):
                col = X[:, unit.indices[0]]
                col = col[np.isfinite(col)]
                span = float(col.max() - col.min()) if col.size else 1.0
                spans.append(max(span, EPS) if span > EPS else 1.0)
            else:
                spans.append(1.0)
            weights.append(float(np.mean([self.schema.specs[i].cost for i in unit.indices])))
        self.spans = np.asarray(spans)
        self.active = np.array([not all(self.schema.specs[i].derived for i in unit.indices) for unit in self.schema.units])   # derivadas não contam
        self.cost_weights = np.asarray(weights)
        self.fitted = True
        return self

    def _units(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        num = np.zeros((len(X), len(self.schema.units)))
        cat = np.zeros((len(X), len(self.schema.units)), dtype=float)
        is_cat = np.zeros(len(self.schema.units), dtype=bool)
        for u, unit in enumerate(self.schema.units):
            if unit.kind in ("continuous", "integer"):
                num[:, u] = X[:, unit.indices[0]] / self.spans[u]
            elif unit.kind == "onehot":
                cat[:, u] = np.argmax(X[:, unit.indices], axis=1)
                is_cat[u] = True
            else:  # binary, code
                cat[:, u] = X[:, unit.indices[0]]
                is_cat[u] = True
        self._is_cat = is_cat
        return num, cat

    def pairwise_to(self, a, B, weighted: bool = False) -> np.ndarray:
        """Distância do vector ``a`` a cada linha de ``B``."""
        if not self.fitted:
            raise RuntimeError("GowerMetric.fit(X_train) é obrigatório antes de medir distâncias.")
        na, ca = self._units(a)
        nb, cb = self._units(B)
        is_cat = self._is_cat
        d = np.where(is_cat[None, :], (ca[0][None, :] != cb).astype(float), np.abs(na[0][None, :] - nb))
        d = d[:, self.active] if self.active.any() else d
        if weighted:
            cw = self.cost_weights[self.active] if self.active.any() else self.cost_weights
            d = d * (cw / cw.sum() * len(cw))[None, :]
        return d.mean(axis=1)

    def distance(self, a, b, weighted: bool = False) -> float:
        return float(self.pairwise_to(a, np.atleast_2d(b), weighted)[0])

    def l1_changes(self, a, b) -> int:
        return len(self.schema.changed_units(a, b))


class DensityModel:
    """Plausibilidade = quão típico é o CF face ao TREINO (kNN em Gower).

    ``plausibility`` ∈ [0,1] é o percentil: fração das distâncias kNN leave-one-out do treino que são ≥ à do CF
    (1 = tão denso como o treino típico; ~0 = outlier). ``plausible`` é True se a distância kNN do CF ≤ quantil
    ``quantile`` (0,95 por defeito) da distribuição de referência do treino. Nenhum dado de teste entra aqui."""

    def __init__(self, metric: GowerMetric, k: int = 5, quantile: float = 0.95, max_reference: int = 2000, seed: int = 0):
        self.metric, self.k, self.quantile, self.max_reference, self.seed = metric, int(k), float(quantile), int(max_reference), int(seed)
        self.fitted = False

    def fit(self, X_train) -> "DensityModel":
        X = np.asarray(X_train, dtype=float)
        if len(X) < 3:
            self.fitted = False
            self.reason = "treino demasiado pequeno para estimar densidade"
            return self
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_reference:
            X = X[rng.choice(len(X), self.max_reference, replace=False)]
        self.reference = X
        k = min(self.k, len(X) - 1)
        self.k_eff = max(1, k)
        probe = X if len(X) <= 600 else X[rng.choice(len(X), 600, replace=False)]
        loo = []
        for row in probe:
            d = self.metric.pairwise_to(row, X)
            d = np.sort(d)[1:self.k_eff + 1]       # exclui o próprio ponto
            loo.append(float(d.mean()))
        self.loo = np.sort(np.asarray(loo))
        self.threshold = float(np.quantile(self.loo, self.quantile))
        self.fitted = True
        self.reason = ""
        return self

    def knn_distance(self, x) -> float:
        d = np.sort(self.metric.pairwise_to(x, self.reference))[:self.k_eff]
        return float(d.mean())

    def score(self, x) -> Dict[str, Any]:
        if not self.fitted:
            return {"available": False, "plausibility": None, "plausible": None, "knn_distance": None, "reason": getattr(self, "reason", "não ajustado")}
        d = self.knn_distance(x)
        pct = float(np.mean(self.loo >= d))
        return {"available": True, "plausibility": pct, "plausible": bool(d <= self.threshold), "knn_distance": d,
                "threshold": self.threshold, "k": self.k_eff, "quantile": self.quantile}


def diversity(metric: GowerMetric, vectors: Sequence[np.ndarray]) -> Dict[str, Optional[float]]:
    if len(vectors) < 2:
        return {"mean_pairwise": None, "min_pairwise": None}
    d = [metric.distance(vectors[i], vectors[j]) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    return {"mean_pairwise": float(np.mean(d)), "min_pairwise": float(np.min(d))}


def mark_dominance(candidates: List[Dict[str, Any]]) -> None:
    """A domina B se A é igualmente válido, muda ≤ features, está ≤ distante e é estritamente melhor em pelo menos um.
    Só marca (``dominated_by``); a remoção fica ao critério do chamador (diversidade pode justificar mantê-lo)."""
    for i, b in enumerate(candidates):
        b["dominated_by"] = []
    for i, a in enumerate(candidates):
        for j, b in enumerate(candidates):
            if i == j:
                continue
            same_valid = (a["model_valid"] and a["semantic_valid"] is not False) >= (b["model_valid"] and b["semantic_valid"] is not False)
            if not same_valid:
                continue
            if a["sparsity"] <= b["sparsity"] and a["proximity"] <= b["proximity"] + 1e-12 and (
                    a["sparsity"] < b["sparsity"] or a["proximity"] < b["proximity"] - 1e-12):
                b["dominated_by"].append(i)


def drop_near_duplicates(metric: GowerMetric, candidates: List[Dict[str, Any]], tol: float = 1e-3) -> Tuple[List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    removed = 0
    for item in candidates:
        vec = np.asarray(item["vector"], dtype=float)
        if any(metric.distance(vec, np.asarray(k["vector"], dtype=float)) <= tol for k in kept):
            removed += 1
            continue
        kept.append(item)
    return kept, removed
