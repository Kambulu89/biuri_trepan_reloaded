"""Auditoria comportamental: a implementação rotulada «C4.5» é realmente C4.5 (e não CART/entropy do scikit-learn)?

Cada verificação é um experimento mínimo com resposta conhecida. Um ``DecisionTreeClassifier(criterion='entropy')`` FALHA
(sem ganho normalizado, sem multirramo nominal, sem distribuição fracionária de ausentes, sem poda pessimista), pelo que o
rótulo «C4.5» não pode ser atribuído a uma implementação CART por engano.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

KNOWN_DEVIATIONS = {
    "subtree_raising": "não implementado (só substituição de subárvore por folha); J48/C4.5 R8 também faz subtree raising",
    "numeric_threshold": "ponto médio entre valores consecutivos (C4.5 R8 usa o maior valor observado <= ponto médio)",
    "categorical_grouping": "sem agrupamento de valores nominais (multirramo puro, como C4.5 sem -s)",
}


def _root(est):
    return getattr(est, "root_", None)


def _n_root_children(est) -> int:
    root = _root(est)
    if root is not None:
        return len(root.children)
    return 2 if getattr(est, "tree_", None) is not None and est.tree_.node_count > 1 else 0


def audit_c45_implementation(factory: Callable[..., Any]) -> Dict[str, Any]:
    """``factory(**kwargs)`` devolve um estimador (aceita ``min_samples_split``, ``min_samples_leaf``, ``confidence_factor``,
    ``feature_types`` quando existirem). Devolve ``{"is_c45": bool, "checks": {nome: {passed, detail}}}``."""
    checks: Dict[str, Dict[str, Any]] = {}

    def make(**kw):
        return factory(**kw)       # o factory decide que argumentos usa (um factory CART ignora feature_types/confidence_factor)

    def check(name, fn):
        try:
            passed, detail = fn()
        except Exception as exc:                                  # noqa: BLE001 - uma falha é um resultado da auditoria
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        checks[name] = {"passed": bool(passed), "detail": detail}

    # 1. multirramo em atributos categóricos (CART é sempre binário)
    def multiway():
        cat = np.repeat([0.0, 1.0, 2.0], 8)
        noise = np.tile([0.0, 1.0], 12)
        X = np.column_stack([cat, noise]); y = cat.astype(int)
        est = make(min_samples_split=2, min_samples_leaf=1, feature_types=["categorical", "numeric"]).fit(X, y)
        n = _n_root_children(est)
        return n == 3, f"filhos da raiz num atributo nominal de 3 valores: {n} (esperado 3)"
    check("categorical_multiway_split", multiway)

    # 2. contínuos: corte binário por limiar
    def numeric():
        x = np.r_[np.linspace(0, 1, 10), np.linspace(2, 3, 10)]
        est = make(min_samples_split=2, min_samples_leaf=1, feature_types=["numeric"]).fit(x.reshape(-1, 1), np.r_[np.zeros(10), np.ones(10)].astype(int))
        n = _n_root_children(est)
        return n == 2 and float(est.predict(np.array([[0.5], [2.5]]))[0]) == 0.0, f"filhos da raiz: {n}"
    check("numeric_binary_threshold", numeric)

    # 3. razão de ganho (não ganho de informação): um atributo tipo identificador tem ganho máximo mas razão de ganho mínima
    def gain_ratio():
        n = 16
        ident = np.arange(n, dtype=float)                      # 16 valores distintos: ganho de informação máximo
        signal = np.repeat([0.0, 1.0], n // 2)                  # binário e perfeitamente preditivo: razão de ganho 1
        y = signal.astype(int)
        est = make(min_samples_split=2, min_samples_leaf=1, feature_types=["categorical", "categorical"]).fit(np.column_stack([ident, signal]), y)
        root = _root(est)
        return root is not None and root.feature == 1, f"atributo da raiz: {getattr(root, 'feature', None)} (esperado 1 = binário; 0 = identificador)"
    check("gain_ratio_selection", gain_ratio)

    # 4. valores ausentes: distribuição fracionária pelos ramos (e não encaminhamento para um só lado)
    def missing():
        rng = np.random.default_rng(0)
        x = np.r_[np.zeros(20), np.ones(20)]
        y = x.astype(int)
        X = x.reshape(-1, 1).copy()
        X[[0, 1, 20, 21], 0] = np.nan
        est = make(min_samples_split=2, min_samples_leaf=1, feature_types=["numeric"]).fit(X, y)
        proba = est.predict_proba(np.array([[np.nan]]))[0]
        return bool(0.05 < proba.min() and proba.max() < 0.95), f"P(classe | ausente) = {np.round(proba, 3).tolist()} (esperado mistura)"
    check("missing_values_fractional_distribution", missing)

    # 5. poda pessimista com factor de confiança (mais conservador => árvore menor ou igual)
    def pruning():
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 4)); y = ((X[:, 0] + 0.8 * rng.normal(size=200)) > 0).astype(int)
        loose = make(confidence_factor=0.49, min_samples_split=2, min_samples_leaf=1, feature_types=["numeric"] * 4).fit(X, y)
        tight = make(confidence_factor=0.05, min_samples_split=2, min_samples_leaf=1, feature_types=["numeric"] * 4).fit(X, y)
        nl, nt = loose.get_n_leaves(), tight.get_n_leaves()
        audited = bool(getattr(tight, "pruning_audit_", None))
        return nt < nl and audited, f"folhas cf=0.49: {nl}; cf=0.05: {nt}; auditoria de poda: {audited}"
    check("pessimistic_error_pruning_confidence_factor", pruning)

    # 6. não é uma árvore do scikit-learn
    def not_sklearn_tree():
        est = make()
        mro = [c.__module__ + "." + c.__name__ for c in type(est).__mro__]
        bad = [m for m in mro if m.startswith("sklearn.tree")]
        return (not bad) and not hasattr(est, "tree_"), f"MRO: {mro[:3]}"
    check("not_a_sklearn_cart_tree", not_sklearn_tree)

    return {"is_c45": all(c["passed"] for c in checks.values()), "checks": checks, "known_deviations": dict(KNOWN_DEVIATIONS)}


def audit_native_c45() -> Dict[str, Any]:
    from core.c45_j48_tree import C45Classifier
    out = audit_c45_implementation(lambda **kw: C45Classifier(**kw))
    out.update(implementation="core.c45_j48_tree.C45Classifier", algorithm_name=C45Classifier.algorithm_name)
    return out
