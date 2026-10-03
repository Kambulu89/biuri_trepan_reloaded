"""TREPAN Original de Craven & Shavlik (implementação histórica V9.2).

Características implementadas:
- MLP tratado exclusivamente como oráculo de caixa-preta;
- membership queries geradas por nó;
- ``min_sample`` por nó antes de escolher um split;
- distribuições marginais: frequências para discretas e KDE para contínuas;
- restrições do caminho raiz->nó aplicadas a cada query;
- crescimento best-first por ``reach * (1 - fidelity)``;
- testes binários por information gain;
- procura m-of-n por beam search histórico (largura configurável, por defeito 2);
- teste estatístico de pureza e pruning sem alteração funcional.

O extractor CART legado permanece noutro módulo apenas por compatibilidade.
"""
from __future__ import annotations

import copy
import heapq
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Tuple
import os
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KernelDensity

try:  # scipy é dependência transitiva do scikit-learn
    from scipy.stats import chi2_contingency, ks_2samp
except Exception:  # pragma: no cover - fallback conservador
    chi2_contingency = ks_2samp = None


class StopReason:
    """Códigos explícitos de paragem de uma folha (nunca ``return leaf`` silencioso)."""

    PURE_NODE = "STOP_PURE_NODE"
    MAX_DEPTH = "STOP_MAX_DEPTH"
    MIN_SAMPLES = "STOP_MIN_SAMPLES"
    NO_VALID_SPLIT = "STOP_NO_VALID_SPLIT"
    MIN_GAIN = "STOP_MIN_GAIN"
    QUERY_BUDGET_EXHAUSTED = "STOP_QUERY_BUDGET_EXHAUSTED"
    MAX_NODES = "STOP_MAX_NODES"
    QUERY_GENERATION_FAILURE = "STOP_QUERY_GENERATION_FAILURE"
    NUMERICAL_FAILURE = "STOP_NUMERICAL_FAILURE"
    PRUNED = "STOP_PRUNED"

    ALL = (PURE_NODE, MAX_DEPTH, MIN_SAMPLES, NO_VALID_SPLIT, MIN_GAIN,
           QUERY_BUDGET_EXHAUSTED, MAX_NODES, QUERY_GENERATION_FAILURE,
           NUMERICAL_FAILURE, PRUNED)


@dataclass(frozen=True)
class Literal:
    feature: int
    threshold: float
    greater: bool = True

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        values = X[:, self.feature]
        return values > self.threshold if self.greater else values <= self.threshold

    def negate(self) -> "Literal":
        return Literal(self.feature, self.threshold, not self.greater)


@dataclass(frozen=True)
class MofNTest:
    m: int
    literals: Tuple[Literal, ...]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if not self.literals:
            return np.ones(len(X), dtype=bool)
        votes = np.column_stack([lit.evaluate(X) for lit in self.literals]).sum(axis=1)
        return votes >= self.m

    def text(self, feature_names: Optional[Sequence[str]] = None) -> str:
        names = feature_names or []
        parts = []
        for lit in self.literals:
            name = names[lit.feature] if lit.feature < len(names) else f"x{lit.feature}"
            parts.append(f"{name} {'>' if lit.greater else '<='} {lit.threshold:.6g}")
        return f"{self.m}-of-{len(parts)}(" + "; ".join(parts) + ")"


@dataclass
class ConstraintSet:
    """Conjunto ordenado de decisões tomadas no caminho raiz->nó."""

    clauses: list[tuple[MofNTest, bool]] = field(default_factory=list)

    def copy(self) -> "ConstraintSet":
        return ConstraintSet(list(self.clauses))

    def add(self, test: MofNTest, outcome: bool) -> None:
        self.clauses.append((test, bool(outcome)))

    def accepts(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        mask = np.ones(len(X), dtype=bool)
        for test, outcome in self.clauses:
            value = test.evaluate(X)
            mask &= value if outcome else ~value
        return mask

    def hard_literals(self) -> list[Literal]:
        """Literais impostos diretamente por testes 1-of-1 no caminho."""
        out: list[Literal] = []
        for test, outcome in self.clauses:
            if len(test.literals) == 1 and test.m == 1:
                lit = test.literals[0]
                out.append(lit if outcome else lit.negate())
        return out


@dataclass
class _FeatureDistribution:
    discrete: bool
    values: np.ndarray
    probabilities: Optional[np.ndarray] = None
    kde: Optional[KernelDensity] = None
    bandwidth: float = 1.0


class FeatureDistributionModel:
    """Modelo marginal usado pelo DrawInstance do TREPAN.

    Variáveis discretas são amostradas segundo frequências empíricas. Variáveis
    contínuas usam KDE gaussiano; as restrições do caminho são satisfeitas por
    amostragem condicionada/rejeição com fallback determinístico empírico.
    """

    def __init__(self, random_state: int = 42, discrete_max_unique: int = 20):
        self.random_state = random_state
        self.discrete_max_unique = discrete_max_unique

    def fit(self, X) -> "FeatureDistributionModel":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("FeatureDistributionModel exige uma matriz 2D não vazia.")
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.features_: list[_FeatureDistribution] = []
        self.training_columns_: list[np.ndarray] = []
        for j in range(X.shape[1]):
            col = X[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                finite = np.asarray([0.0])
            self.training_columns_.append(finite.copy())
            unique, counts = np.unique(finite, return_counts=True)
            threshold = min(self.discrete_max_unique, max(2, int(math.sqrt(len(finite)))))
            is_integerish = np.allclose(finite, np.round(finite), atol=1e-9)
            discrete = len(unique) <= threshold and (is_integerish or len(unique) <= 8)
            if discrete:
                p = counts.astype(float) / counts.sum()
                self.features_.append(_FeatureDistribution(True, unique, p, None, 0.0))
            else:
                std = float(np.std(finite))
                n = max(2, len(finite))
                bw = 1.06 * max(std, 1e-9) * (n ** (-1 / 5))
                bw = max(float(bw), 1e-6)
                kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(finite.reshape(-1, 1))
                self.features_.append(_FeatureDistribution(False, finite, None, kde, bw))
        return self

    def _sample_column(self, j: int, n: int) -> np.ndarray:
        spec = self.features_[j]
        if spec.discrete:
            return self._rng.choice(spec.values, size=n, replace=True, p=spec.probabilities)
        # KernelDensity.sample aceita int/RandomState, não Generator em versões antigas.
        seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
        return spec.kde.sample(n_samples=n, random_state=seed).reshape(-1)

    def _draw_unconstrained(self, n: int) -> np.ndarray:
        out = np.empty((n, self.n_features_in_), dtype=float)
        for j in range(self.n_features_in_):
            out[:, j] = self._sample_column(j, n)
        return out

    def draw(self, n: int, constraints: Optional[ConstraintSet] = None) -> np.ndarray:
        if n <= 0:
            return np.empty((0, self.n_features_in_), dtype=float)
        constraints = constraints or ConstraintSet()
        accepted: list[np.ndarray] = []
        remaining = int(n)
        # Rejeição condicionada em lotes preserva as marginais e satisfaz todo
        # o conjunto de restrições, incluindo m-of-n arbitrário.
        for _ in range(80):
            if remaining <= 0:
                break
            batch = self._draw_unconstrained(max(64, remaining * 6))
            keep = batch[constraints.accepts(batch)]
            self.n_drawn_ = getattr(self, "n_drawn_", 0) + len(batch)
            self.n_rejected_ = getattr(self, "n_rejected_", 0) + (len(batch) - len(keep))
            if len(keep):
                take = keep[:remaining]
                accepted.append(take)
                remaining -= len(take)
        if remaining > 0:
            # Fallback conservador: continua a amostrar das marginais aprendidas
            # (KDE/frequências) em lotes maiores. Nunca devolve uma linha que
            # viole as restrições apenas para cumprir a quota.
            for _ in range(160):
                batch = self._draw_unconstrained(max(128, remaining * 12))
                keep = batch[constraints.accepts(batch)]
                self.n_drawn_ = getattr(self, "n_drawn_", 0) + len(batch)
                self.n_rejected_ = getattr(self, "n_rejected_", 0) + (len(batch) - len(keep))
                if len(keep):
                    take = keep[:remaining]
                    accepted.append(take)
                    remaining -= len(take)
                if remaining <= 0:
                    break
        if remaining > 0:
            raise RuntimeError(
                "Não foi possível gerar membership queries que satisfaçam as restrições do nó; "
                "a região pode ser vazia ou demasiado rara."
            )
        return np.vstack(accepted)[:n]

    def differs_from(self, reference: "FeatureDistributionModel", X_local: np.ndarray, alpha: float = 0.10) -> bool:
        """Teste simples para decidir se um modelo local é justificável."""
        if reference is None or len(X_local) < 8:
            return False
        X_local = np.asarray(X_local, dtype=float)
        corrected = alpha / max(1, self.n_features_in_)
        for j, spec in enumerate(self.features_):
            local = X_local[:, j]
            local = local[np.isfinite(local)]
            global_col = reference.training_columns_[j]
            if len(local) < 4:
                continue
            if spec.discrete and chi2_contingency is not None:
                vals = np.union1d(np.unique(local), np.unique(global_col))
                a = np.array([(local == v).sum() for v in vals], dtype=float)
                b = np.array([(global_col == v).sum() for v in vals], dtype=float)
                table = np.vstack([a, b])
                valid = table.sum(axis=0) > 0
                if valid.sum() >= 2:
                    _, p, _, _ = chi2_contingency(table[:, valid], correction=False)
                    if p < corrected:
                        return True
            elif ks_2samp is not None:
                if ks_2samp(local, global_col).pvalue < corrected:
                    return True
        return False


def _entropy(y: np.ndarray, classes: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    counts = np.asarray([(y == c).sum() for c in classes], dtype=float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _information_gain(y: np.ndarray, mask: np.ndarray, classes: np.ndarray) -> float:
    if len(y) == 0 or not np.any(mask) or np.all(mask):
        return 0.0
    n = len(y)
    return float(
        _entropy(y, classes)
        - mask.sum() / n * _entropy(y[mask], classes)
        - (~mask).sum() / n * _entropy(y[~mask], classes)
    )


@dataclass
class _Node:
    real_X: np.ndarray
    real_y: np.ndarray
    depth: int
    constraints: ConstraintSet
    reach: float
    distribution: np.ndarray
    prediction: Any
    parent_model: Optional[FeatureDistributionModel] = None
    test: Optional[MofNTest] = None
    false_child: Optional["_Node"] = None
    true_child: Optional["_Node"] = None
    query_X: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    query_y: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=object))
    fidelity: float = 0.0
    node_id: int = -1
    stop_reason: Optional[str] = None
    stop_detail: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.test is None


class TrepanOriginalClassifier(ClassifierMixin, BaseEstimator):
    """TREPAN Original histórico, sem dependência de CART/sklearn trees."""

    PRESETS = {
        "nips_1995": dict(min_sample=1000, purity_epsilon=0.05, purity_alpha=0.05, max_nodes=31, beam_width=2),
        "thesis_1996": dict(min_sample=10000, purity_epsilon=0.05, purity_alpha=0.01, max_nodes=63, beam_width=2),
    }

    @classmethod
    def from_preset(cls, name: str, **overrides):
        key = str(name).strip().lower()
        if key not in cls.PRESETS:
            raise ValueError(f"Preset TREPAN desconhecido: {name!r}. Use 'nips_1995' ou 'thesis_1996'.")
        params = dict(cls.PRESETS[key])
        params.update(overrides)
        return cls(**params)

    def __init__(
        self,
        max_nodes: int = 31,
        max_depth: Optional[int] = 8,
        min_samples_leaf: int = 4,
        min_sample: int = 1000,
        max_n: int = 3,
        beam_width: int = 2,
        max_features_per_node: int = 12,
        max_queries: int = 10000,
        purity_epsilon: float = 0.05,
        purity_alpha: float = 0.05,
        mofn_alpha: float = 0.05,
        local_model_alpha: float = 0.10,
        random_state: int = 42,
        min_gain: float = 0.0,
    ):
        self.min_gain = min_gain
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_sample = min_sample
        self.max_n = max_n
        self.beam_width = beam_width
        self.max_features_per_node = max_features_per_node
        self.max_queries = max_queries
        self.purity_epsilon = purity_epsilon
        self.purity_alpha = purity_alpha
        self.mofn_alpha = mofn_alpha
        self.local_model_alpha = local_model_alpha
        self.random_state = random_state

    def _distribution(self, y: np.ndarray) -> np.ndarray:
        mass = np.asarray([(y == c).sum() for c in self.classes_], dtype=float)
        if mass.sum() == 0:
            mass[:] = 1.0
        return mass / mass.sum()

    def _fidelity(self, y: np.ndarray, prediction: Any) -> float:
        return float(np.mean(y == prediction)) if len(y) else 1.0

    def _priority(self, node: _Node) -> float:
        return float(node.reach * (1.0 - node.fidelity))

    def _is_statistically_pure(self, y: np.ndarray) -> bool:
        if len(y) == 0:
            return True
        counts = np.asarray([(y == c).sum() for c in self.classes_], dtype=float)
        p_hat = counts.max() / counts.sum()
        # Wilson lower bound; considera puro quando a confiança inferior excede 1-epsilon.
        z = 1.96 if self.purity_alpha <= 0.05 else 1.645
        n = counts.sum()
        denom = 1 + z * z / n
        centre = p_hat + z * z / (2 * n)
        radius = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)
        lower = (centre - radius) / denom
        return bool(lower >= 1.0 - self.purity_epsilon)

    def _feature_priority_scores(self, X: np.ndarray) -> np.ndarray:
        """Prioridade de features para gerar candidatos.

        No TREPAN Original histórico é apenas a variância observada. Extensões
        (como o Reloaded) podem sobrepor este hook sem alterar o motor base.
        """
        return np.nanvar(X, axis=0)

    def _split_selection_score(
        self, y: np.ndarray, mask: np.ndarray, test: Optional[MofNTest] = None
    ) -> float:
        """Score usado para seleccionar splits; no Original = Information Gain."""
        return _information_gain(y, mask, self.classes_)

    def _split_audit_metadata(self, test: MofNTest) -> dict:
        return {}

    def _candidate_literals(self, X: np.ndarray, y: np.ndarray) -> list[tuple[float, Literal]]:
        candidates: list[tuple[float, Literal]] = []
        priorities = self._feature_priority_scores(X)
        feature_order = np.argsort(-priorities, kind="stable")[: self.max_features_per_node]
        for j in feature_order:
            values = X[:, j]
            order = np.argsort(values, kind="stable")
            sv, sy = values[order], y[order]
            thresholds = []
            for i in range(len(sv) - 1):
                if not np.isfinite(sv[i]) or not np.isfinite(sv[i + 1]):
                    continue
                if sv[i] == sv[i + 1] or sy[i] == sy[i + 1]:
                    continue
                thresholds.append((sv[i] + sv[i + 1]) / 2.0)
            if not thresholds:
                uniq = np.unique(values[np.isfinite(values)])
                if len(uniq) > 1:
                    thresholds = ((uniq[:-1] + uniq[1:]) / 2.0).tolist()
            # limita custo mantendo cobertura determinística
            if len(thresholds) > 24:
                idx = np.linspace(0, len(thresholds) - 1, 24).round().astype(int)
                thresholds = [thresholds[i] for i in np.unique(idx)]
            for thr in thresholds:
                for greater in (True, False):
                    lit = Literal(int(j), float(thr), greater)
                    mask = lit.evaluate(X)
                    if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
                        continue
                    test = MofNTest(1, (lit,))
                    candidates.append((self._split_selection_score(y, mask, test), lit))
        candidates.sort(key=lambda x: (-x[0], x[1].feature, x[1].threshold, not x[1].greater))
        # evita duplicados exatos
        seen = set(); out = []
        for score, lit in candidates:
            key = (lit.feature, round(lit.threshold, 12), lit.greater)
            if key not in seen:
                seen.add(key); out.append((score, lit))
        self._count("simple_generated", len(candidates))
        self._count("simple_evaluated", len(out))
        return out

    def _count(self, name: str, amount: int = 1) -> None:
        """Contadores de auditoria activos apenas durante a procura de splits do nó."""
        counters = getattr(self, "_split_counters_", None)
        if counters is not None:
            counters[name] = counters.get(name, 0) + int(amount)

    def _partition_significantly_different(self, base: MofNTest, candidate: MofNTest, X: np.ndarray) -> bool:
        a = base.evaluate(X); b = candidate.evaluate(X)
        table = np.asarray([
            [np.sum(~a & ~b), np.sum(~a & b)],
            [np.sum(a & ~b), np.sum(a & b)],
        ], dtype=float)
        if chi2_contingency is None or np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
            return not np.array_equal(a, b)
        try:
            _, p, _, _ = chi2_contingency(table, correction=False)
            return bool(p < self.mofn_alpha)
        except ValueError:
            return not np.array_equal(a, b)

    def _best_mofn(self, X: np.ndarray, y: np.ndarray) -> Optional[tuple[MofNTest, float]]:
        ranked = self._candidate_literals(X, y)
        if not ranked:
            return None
        seed_score, seed_lit = ranked[0]
        best = MofNTest(1, (seed_lit,))
        best_score = seed_score
        beam = [(best_score, best)]
        literal_pool = [lit for _, lit in ranked[: max(8, self.max_features_per_node * 3)]]

        while beam:
            generated: list[tuple[float, MofNTest]] = []
            for _, test in beam:
                if len(test.literals) >= self.max_n:
                    continue
                used_features = {l.feature for l in test.literals}
                for lit in literal_pool:
                    if lit.feature in used_features:
                        continue
                    literals = tuple(list(test.literals) + [lit])
                    for m in {test.m, test.m + 1}:
                        if m < 1 or m > len(literals):
                            continue
                        cand = MofNTest(m, literals)
                        self._count("mofn_generated")
                        if not self._partition_significantly_different(test, cand, X):
                            self._count("mofn_rejected_not_distinct")
                            continue
                        mask = cand.evaluate(X)
                        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
                            self._count("mofn_rejected_min_samples_leaf")
                            continue
                        score = self._split_selection_score(y, mask, cand)
                        self._count("mofn_evaluated")
                        generated.append((score, cand))
            if not generated:
                break
            generated.sort(key=lambda x: (-x[0], len(x[1].literals), x[1].m))
            next_beam = generated[: max(1, self.beam_width)]
            if next_beam[0][0] <= best_score + 1e-12:
                break
            best_score, best = next_beam[0]
            beam = next_beam
        return best, float(best_score)

    def _draw_membership_queries(
        self, model: FeatureDistributionModel, n: int, constraints: ConstraintSet, node: _Node
    ) -> np.ndarray:
        """Hook de geração de membership queries.

        O Original histórico usa directamente o DrawInstance marginal. O
        Reloaded pode projectar as queries para um espaço ontológico coerente.
        """
        return model.draw(n, constraints)

    def _decision_sample(self, node: _Node) -> tuple[np.ndarray, np.ndarray, FeatureDistributionModel]:
        X = node.real_X
        y = node.real_y
        parent_model = node.parent_model or self.global_distribution_model_
        local_model = parent_model
        if len(X) >= 8:
            candidate = FeatureDistributionModel(
                random_state=self.random_state + max(0, node.node_id),
            ).fit(X)
            if candidate.differs_from(parent_model, X, alpha=self.local_model_alpha):
                local_model = candidate
        needed = max(0, int(self.effective_min_sample_) - len(X))
        remaining_budget = max(0, int(self.max_queries) - self.membership_queries_)
        requested = needed
        needed = min(needed, remaining_budget)
        node.stats.update({
            "queries_requested": int(requested),
            "queries_generated": 0, "queries_valid": 0, "queries_rejected": 0,
            "queries_used": 0, "query_budget_remaining_before": int(remaining_budget),
            "budget_truncated": bool(needed < requested),
        })
        if needed > 0 and self.oracle_ is not None:
            drawn_before = getattr(local_model, "n_drawn_", 0)
            rejected_before = getattr(local_model, "n_rejected_", 0)
            t0 = time.perf_counter()
            qx = self._draw_membership_queries(local_model, needed, node.constraints, node)
            qy = np.asarray(self.oracle_.predict(qx))
            self.query_time_ += time.perf_counter() - t0
            valid = int(np.sum(node.constraints.accepts(qx))) if len(qx) else 0
            node.stats.update({
                "queries_generated": int(len(qx)),
                "queries_valid": valid,
                # linhas candidatas descartadas pela amostragem por rejeição (restrições do caminho)
                "queries_rejected": int(getattr(local_model, "n_rejected_", 0) - rejected_before),
                "candidates_drawn": int(getattr(local_model, "n_drawn_", 0) - drawn_before),
                "queries_used": int(len(qx)),
                "queries_violating_path": int(len(qx) - valid),
            })
            self.membership_queries_ += len(qx)
            node.query_X = qx
            node.query_y = qy
            X = np.vstack([X, qx]) if len(X) else qx
            y = np.concatenate([y, qy]) if len(y) else qy
        node.stats["real_samples"] = int(len(node.real_y))
        node.stats["synthetic_samples"] = int(len(y) - len(node.real_y))
        node.stats["effective_samples"] = int(len(y))
        return X, y, local_model

    def fit(
        self,
        X,
        y=None,
        *,
        oracle=None,
        sample_weight=None,
        feature_names: Optional[Sequence[str]] = None,
    ):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("TREPAN Original exige X 2D não vazio.")
        if oracle is None and y is None:
            raise ValueError("TREPAN Original exige y do oráculo ou oracle=...")
        if sample_weight is not None:
            # API preservada; o algoritmo histórico não usa pesos externos.
            w = np.asarray(sample_weight)
            if w.shape != (len(X),):
                raise ValueError("sample_weight incompatível com X.")
        self.oracle_ = oracle
        y_oracle = np.asarray(oracle.predict(X) if oracle is not None else y)
        if len(y_oracle) != len(X):
            raise ValueError("X e y/oráculo têm tamanhos incompatíveis.")
        self.classes_ = np.unique(y_oracle)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(feature_names or [f"x{i}" for i in range(X.shape[1])])
        self._rng = np.random.default_rng(self.random_state)
        self.global_distribution_model_ = FeatureDistributionModel(random_state=self.random_state).fit(X)
        self.membership_queries_ = 0
        self._fit_started_ = time.perf_counter()
        self.query_time_ = 0.0
        self.split_search_time_ = 0.0
        self.m_of_n_search_time_ = 0.0
        self.pruning_time_ = 0.0
        self.expansion_log_: list[dict] = []
        self.candidate_totals_: dict = {}
        self.pruning_audit_: list[dict] = []
        self.budget_starved_nodes_: list[int] = []
        self.tree_raw_root_ = None
        self._split_counters_ = None
        self.oracle_info_ = {
            "oracle_type": type(oracle).__name__ if oracle is not None else "labels_from_y",
            "oracle_feature_space": int(X.shape[1]),
            "uses_real_labels": False if oracle is not None else None,
        }
        # Compatibilidade com chamadas antigas que forneciam um orçamento menor
        # do que o min_sample histórico: reduzimos uma única vez o alvo efectivo
        # e registamo-lo. Quando o orçamento é suficiente, effective == min_sample.
        self.effective_min_sample_ = int(min(self.min_sample, len(X) + max(0, int(self.max_queries))))
        self.oracle_query_count_ = len(X)  # compatibilidade: inclui consultas iniciais
        self.node_audit_: list[dict] = []
        self.split_audit_: list[dict] = []
        self.expansion_order_: list[int] = []
        self.local_model_count_ = 0

        root_dist = self._distribution(y_oracle)
        root_pred = self.classes_[int(np.argmax(root_dist))]
        root = _Node(
            X, y_oracle, 0, ConstraintSet(), 1.0, root_dist, root_pred,
            parent_model=self.global_distribution_model_, node_id=0,
        )
        root.fidelity = self._fidelity(y_oracle, root_pred)
        self.root_ = root

        queue: list[tuple[float, int, _Node]] = []
        serial = itertools.count()

        def push(node: _Node):
            priority = self._priority(node)
            node.stats["priority_at_push"] = float(priority)
            heapq.heappush(queue, (-priority, next(serial), node))
            self.node_audit_.append({
                "node_id": node.node_id,
                "depth": node.depth,
                "reach": float(node.reach),
                "fidelity": float(node.fidelity),
                "priority": float(priority),
                "expanded": False,
            })

        def stop(node: _Node, reason: str, **detail) -> None:
            node.stop_reason = reason
            node.stop_detail = dict(detail)

        self.nodes_created_ = 1
        self.nodes_expanded_ = 0
        push(root)
        next_node_id = 1
        node_count = 1
        self.max_nodes_reached_ = False

        while queue and node_count + 2 <= self.max_nodes:
            neg_priority, _, node = heapq.heappop(queue)
            # Prova best-first: a prioridade escolhida é o máximo da fila no momento.
            queue_max = max([-item[0] for item in queue], default=-neg_priority)
            order = len(self.expansion_log_) + 1
            self.expansion_log_.append({
                "selected_order": order,
                "candidate_node_id": node.node_id,
                "priority_score": float(-neg_priority),
                "queue_max_priority_other": float(queue_max),
                "is_best_first_choice": bool(-neg_priority >= queue_max - 1e-12),
                "queue_size_after_pop": len(queue),
                "depth": node.depth,
                "reach": float(node.reach),
                "estimated_error": float(1.0 - node.fidelity),
                "priority_components": {
                    "node_probability_mass": float(node.reach),
                    "estimated_error": float(1.0 - node.fidelity),
                },
                "samples": int(len(node.real_y)),
                "impurity": float(_entropy(node.real_y, self.classes_)) if len(node.real_y) else 0.0,
                "potential_fidelity_gain": float(node.reach * (1.0 - node.fidelity)),
            })
            if self.max_depth is not None and node.depth >= self.max_depth:
                stop(node, StopReason.MAX_DEPTH, configured_max_depth=self.max_depth, node_depth=node.depth)
                continue
            try:
                decision_X, decision_y, local_model = self._decision_sample(node)
            except RuntimeError as exc:
                stop(node, StopReason.QUERY_GENERATION_FAILURE, error=str(exc))
                continue
            node.distribution = self._distribution(decision_y)
            node.prediction = self.classes_[int(np.argmax(node.distribution))]
            node.fidelity = self._fidelity(decision_y, node.prediction)
            if len(decision_y) < int(self.effective_min_sample_):
                starved = self.oracle_ is not None
                if starved:
                    self.budget_starved_nodes_.append(node.node_id)
                    stop(node, StopReason.QUERY_BUDGET_EXHAUSTED,
                         query_budget=int(self.max_queries), queries_used=int(self.membership_queries_),
                         decision_sample_size=int(len(decision_y)),
                         required_sample_size=int(self.effective_min_sample_))
                else:
                    stop(node, StopReason.MIN_SAMPLES, decision_sample_size=int(len(decision_y)),
                         required_sample_size=int(self.effective_min_sample_))
                self.node_audit_.append({
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "reach": float(node.reach),
                    "fidelity": float(node.fidelity),
                    "priority": float(self._priority(node)),
                    "decision_sample_size": int(len(decision_y)),
                    "real_sample_size": int(len(node.real_y)),
                    "query_sample_size": int(len(decision_y) - len(node.real_y)),
                    "expanded": False,
                    "stop_reason": "query_budget_before_min_sample" if starved else "min_sample",
                    "stop_code": node.stop_reason,
                })
                continue
            audit = {
                "node_id": node.node_id,
                "depth": node.depth,
                "reach": float(node.reach),
                "fidelity": float(node.fidelity),
                "priority": float(self._priority(node)),
                "decision_sample_size": int(len(decision_y)),
                "real_sample_size": int(len(node.real_y)),
                "query_sample_size": int(len(decision_y) - len(node.real_y)),
                "expanded": False,
            }
            # substitui a entrada preliminar pela versão de decisão
            self.node_audit_.append(audit)
            node.stats["oracle_distribution"] = {
                str(c): float(p) for c, p in zip(self.classes_, node.distribution)
            }
            if len(np.unique(decision_y)) < 2:
                stop(node, StopReason.PURE_NODE, kind="single_class")
                continue
            if self._is_statistically_pure(decision_y):
                stop(node, StopReason.PURE_NODE, kind="wilson_lower_bound",
                     purity_epsilon=float(self.purity_epsilon), majority_fraction=float(node.distribution.max()))
                continue
            self._split_counters_ = {}
            t0 = time.perf_counter()
            try:
                best = self._best_mofn(decision_X, decision_y)
            except (FloatingPointError, ZeroDivisionError, ValueError) as exc:
                self._split_counters_ = None
                stop(node, StopReason.NUMERICAL_FAILURE, error=str(exc))
                continue
            elapsed = time.perf_counter() - t0
            counters, self._split_counters_ = self._split_counters_, None
            self.split_search_time_ += elapsed
            if counters.get("mofn_generated", 0):
                self.m_of_n_search_time_ += elapsed
            counters["semantic_candidates"] = int(getattr(self, "_semantic_candidates_last_", 0) or 0)
            node.stats["candidate_counters"] = counters
            for key, value in counters.items():
                self.candidate_totals_[key] = self.candidate_totals_.get(key, 0) + int(value)
            if best is None:
                stop(node, StopReason.NO_VALID_SPLIT, reason="no_candidate_satisfies_min_samples_leaf",
                     min_samples_leaf=int(self.min_samples_leaf), **counters)
                continue
            test, gain = best
            raw_gain = _information_gain(decision_y, test.evaluate(decision_X), self.classes_)
            if float(raw_gain) < float(self.min_gain):
                stop(node, StopReason.MIN_GAIN, best_candidate_gain=float(raw_gain),
                     required_min_gain=float(self.min_gain), best_test=test.text(self.feature_names_in_))
                continue
            real_mask = test.evaluate(node.real_X)
            dec_mask = test.evaluate(decision_X)
            if dec_mask.sum() < self.min_samples_leaf or (~dec_mask).sum() < self.min_samples_leaf:
                stop(node, StopReason.NO_VALID_SPLIT, reason="best_split_violates_min_samples_leaf",
                     best_test=test.text(self.feature_names_in_))
                continue
            # reach é estimado pela frequência do ramo no conjunto de decisão do nó.
            p_true = float(np.mean(dec_mask)); p_false = 1.0 - p_true
            false_X, true_X = node.real_X[~real_mask], node.real_X[real_mask]
            false_y, true_y = node.real_y[~real_mask], node.real_y[real_mask]
            # Se um ramo não contém exemplos reais, mantém pelo menos as queries desse ramo
            # como sementes para que o nó continue semanticamente definido.
            if len(false_X) == 0:
                false_X, false_y = decision_X[~dec_mask], decision_y[~dec_mask]
            if len(true_X) == 0:
                true_X, true_y = decision_X[dec_mask], decision_y[dec_mask]
            false_constraints = node.constraints.copy(); false_constraints.add(test, False)
            true_constraints = node.constraints.copy(); true_constraints.add(test, True)
            fd = self._distribution(false_y); td = self._distribution(true_y)
            false_node = _Node(false_X, false_y, node.depth + 1, false_constraints,
                               node.reach * p_false, fd, self.classes_[int(np.argmax(fd))],
                               parent_model=local_model, node_id=next_node_id)
            next_node_id += 1
            true_node = _Node(true_X, true_y, node.depth + 1, true_constraints,
                              node.reach * p_true, td, self.classes_[int(np.argmax(td))],
                              parent_model=local_model, node_id=next_node_id)
            next_node_id += 1
            false_node.fidelity = self._fidelity(false_y, false_node.prediction)
            true_node.fidelity = self._fidelity(true_y, true_node.prediction)
            node.test = test; node.false_child = false_node; node.true_child = true_node
            audit["expanded"] = True
            split_kind = "m_of_n" if (len(test.literals) > 1 or test.m > 1) else "simple"
            node.stats["split_type"] = split_kind
            split_record = {
                "node_id": node.node_id,
                "information_gain": float(raw_gain),
                "selection_score": float(gain),
                "test": test.text(self.feature_names_in_),
                "m": int(test.m),
                "n": int(len(test.literals)),
                "split_type": split_kind,
                "decision_sample_size": int(len(decision_y)),
                "left_count": int((~dec_mask).sum()),
                "right_count": int(dec_mask.sum()),
                "candidate_counters": dict(counters),
            }
            split_record.update(self._split_audit_metadata(test))
            self.split_audit_.append(split_record)
            self.expansion_order_.append(node.node_id)
            node_count += 2
            self.nodes_created_ += 2
            self.nodes_expanded_ += 1
            push(false_node); push(true_node)

        # Folhas ainda na fila quando o laço termina: o único motivo possível é max_nodes.
        for _, _, leftover in queue:
            if leftover.stop_reason is None:
                self.max_nodes_reached_ = True
                stop(leftover, StopReason.MAX_NODES, max_nodes=int(self.max_nodes),
                     nodes_created=int(node_count))

        self.node_count_ = node_count
        self.nodes_raw_ = node_count
        self.best_first_ = True
        self.m_of_n_ = any(s["n"] > 1 for s in self.split_audit_)
        self.oracle_query_count_ = int(len(X) + self.membership_queries_)
        self.query_budget_exhausted_ = bool(
            self.membership_queries_ >= int(self.max_queries) or self.budget_starved_nodes_
        )
        # Árvore bruta preservada (sem arrays de dados) antes de qualquer poda.
        self.tree_raw_root_ = self._snapshot(self.root_)
        raw_pred = self._predict_with_root(self.tree_raw_root_, X)
        t0 = time.perf_counter()
        self._prune_identical_subtrees(self.root_)
        self.pruning_time_ = time.perf_counter() - t0
        self.node_count_ = self._count_nodes(self.root_)
        final_pred = self._predict_with_root(self.root_, X)
        # Efeito da poda medido APENAS em dados de treino face ao oráculo (nunca no teste).
        self.pruning_summary_ = {
            "nodes_before_pruning": int(self.nodes_raw_),
            "nodes_after_pruning": int(self.node_count_),
            "depth_before": int(self._depth_of(self.tree_raw_root_)),
            "depth_after": int(self._depth_of(self.root_)),
            "pruned_nodes": int(self.nodes_raw_ - self.node_count_),
            "pruned_subtrees": int(len(self.pruning_audit_)),
            "fidelity_train_before": float(np.mean(raw_pred == y_oracle)),
            "fidelity_train_after": float(np.mean(final_pred == y_oracle)),
            "uses_test_data": False,
        }
        self.training_time_ = time.perf_counter() - self._fit_started_
        # O oráculo só é necessário durante a extracção. Não o persistir evita
        # acoplar o artefacto final ao MLP, ao extractor/GUI ou a callbacks de
        # treino, e permite carregar a árvore em qualquer directório/processo.
        self.oracle_ = None
        self.oracle_attached_ = False
        return self

    @staticmethod
    def _snapshot(node: "_Node") -> "_Node":
        """Cópia estrutural (sem matrizes de dados) usada como árvore bruta."""
        empty_X = np.empty((0, node.real_X.shape[1] if node.real_X.ndim == 2 else 0))
        clone = _Node(
            empty_X, np.empty((0,), dtype=node.real_y.dtype), node.depth, ConstraintSet(),
            node.reach, np.array(node.distribution, copy=True), node.prediction,
            test=node.test, fidelity=node.fidelity, node_id=node.node_id,
            stop_reason=node.stop_reason, stop_detail=dict(node.stop_detail),
            stats=copy.deepcopy(node.stats),
        )
        if not node.is_leaf:
            clone.false_child = TrepanOriginalClassifier._snapshot(node.false_child)
            clone.true_child = TrepanOriginalClassifier._snapshot(node.true_child)
        return clone

    def _predict_with_root(self, root: "_Node", X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = []
        for row in X:
            node = root
            r = row.reshape(1, -1)
            while not node.is_leaf:
                node = node.true_child if bool(node.test.evaluate(r)[0]) else node.false_child
            out.append(self.classes_[int(np.argmax(node.distribution))])
        return np.asarray(out)

    def predict_raw(self, X):
        """Predição da árvore bruta (antes da poda)."""
        return self._predict_with_root(self.tree_raw_root_, X)

    @staticmethod
    def _depth_of(node: "_Node") -> int:
        return int(node.depth if node.is_leaf else max(
            TrepanOriginalClassifier._depth_of(node.false_child),
            TrepanOriginalClassifier._depth_of(node.true_child)))

    def _prune_identical_subtrees(self, node: _Node) -> Any:
        if node.is_leaf:
            return node.prediction
        left = self._prune_identical_subtrees(node.false_child)
        right = self._prune_identical_subtrees(node.true_child)
        if left == right:
            removed = self._count_nodes(node) - 1
            self.pruning_audit_.append({
                "node_id": node.node_id,
                "reason": "identical_leaf_predictions",
                "removed_nodes": int(removed),
                "validation_effect": 0.0,
                "fidelity_before": float(node.fidelity),
                "fidelity_after": float(node.fidelity),
                "uses_test_data": False,
            })
            node.test = None; node.false_child = None; node.true_child = None
            node.prediction = left
            node.stop_reason = StopReason.PRUNED
            node.stop_detail = {"reason": "identical_leaf_predictions"}
            return left
        return object()

    def _count_nodes(self, node: _Node) -> int:
        if node.is_leaf:
            return 1
        return 1 + self._count_nodes(node.false_child) + self._count_nodes(node.true_child)

    def _leaf(self, row: np.ndarray) -> _Node:
        node = self.root_
        row = np.asarray(row, dtype=float).reshape(1, -1)
        while not node.is_leaf:
            node = node.true_child if bool(node.test.evaluate(row)[0]) else node.false_child
        return node

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.asarray([self._leaf(row).distribution for row in X], dtype=float)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]

    def get_depth(self):
        def rec(n):
            return n.depth if n.is_leaf else max(rec(n.false_child), rec(n.true_child))
        return int(rec(self.root_))

    def get_n_leaves(self):
        def rec(n):
            return 1 if n.is_leaf else rec(n.false_child) + rec(n.true_child)
        return int(rec(self.root_))

    def export_rules(self):
        rules = []
        def walk(node: _Node, conditions: list[str]):
            if node.is_leaf:
                rules.append({
                    "conditions": list(conditions),
                    "prediction": node.prediction.item() if isinstance(node.prediction, np.generic) else node.prediction,
                    "probabilities": [float(v) for v in node.distribution],
                    "support": int(len(node.real_y)),
                    "reach": float(node.reach),
                })
                return
            text = node.test.text(self.feature_names_in_)
            walk(node.false_child, conditions + [f"NÃO {text}"])
            walk(node.true_child, conditions + [text])
        walk(self.root_, [])
        return rules

    def export_text(self, class_names: Optional[Sequence[str]] = None) -> str:
        class_names = list(class_names or [])
        lines = []
        for idx, rule in enumerate(self.export_rules(), start=1):
            pred = rule["prediction"]
            try:
                pi = int(pred)
            except (TypeError, ValueError):
                pi = -1
            label = class_names[pi] if 0 <= pi < len(class_names) else str(pred)
            premise = " E ".join(rule["conditions"]) or "VERDADEIRO"
            confidence = max(rule.get("probabilities") or [0.0])
            lines.append(
                f"R{idx}: SE {premise}, ENTÃO classe {label} "
                f"(confiança={confidence:.3f}, suporte={rule.get('support', 0)})"
            )
        return "\n".join(lines)

    def iter_nodes(self):
        stack = [self.root_]
        while stack:
            node = stack.pop()
            yield node
            if not node.is_leaf:
                stack.append(node.true_child)
                stack.append(node.false_child)

    def iter_splits(self):
        for node in self.iter_nodes():
            if not node.is_leaf:
                yield node, node.test


class TrepanOriginalExtractor:
    """Adaptador de compatibilidade para o TREPAN Original histórico.

    Mantém a API ``extract_tree`` usada pela GUI e pelos pipelines antigos, mas
    o modelo produzido é sempre :class:`TrepanOriginalClassifier`. Não usa CART
    nem ``DecisionTreeClassifier``. O conjunto de teste, quando fornecido, serve
    apenas para a auditoria final e nunca para escolher parâmetros ou splits.
    """

    def __init__(self, *, preset: str | None = None, random_state: int = 42):
        self.preset = preset
        self.random_state = int(random_state)
        self.explainer_tree: TrepanOriginalClassifier | None = None
        self.last_audit: dict[str, Any] = {}
        self.last_exported_image_path: str | None = None

    @staticmethod
    def _limits(training_limits: Optional[dict[str, Any]], sample_size: int, n_train: int) -> dict[str, Any]:
        limits = dict(training_limits or {})
        max_queries = int(limits.get('max_queries') or sample_size or 2000)
        max_nodes = limits.get('max_nodes')
        if max_nodes is None:
            max_nodes = 31
        max_depth = limits.get('max_depth')
        if max_depth is None:
            max_depth = 8
        min_sample = limits.get('min_sample')
        if min_sample is None:
            # Configuração operacional adaptativa. Os presets históricos exactos
            # continuam disponíveis explicitamente em ``from_preset``.
            min_sample = max(n_train, min(1000, max(120, n_train * 3)))
        return {
            'max_queries': max(0, max_queries),
            'max_nodes': int(max_nodes),
            'max_depth': int(max_depth),
            'min_sample': int(min_sample),
            'min_samples_leaf': int(limits.get('min_samples_leaf', 4)),
            'max_n': int(limits.get('max_n', limits.get('m_of_n_max_n', 3))),
            'beam_width': int(limits.get('beam_width', 2)),
        }

    def _build_classifier(self, *, training_limits, sample_size, n_train):
        limits = self._limits(training_limits, sample_size, n_train)
        if self.preset:
            return TrepanOriginalClassifier.from_preset(
                self.preset, random_state=self.random_state, **limits
            )
        return TrepanOriginalClassifier(random_state=self.random_state, **limits)

    def _generate_intelligent_synthetic_data(self, mlp_model, X_encoded, sample_size):
        """Alias legado para geração marginal histórica sem recorrer a CART.

        Alguns módulos V8/V9 chamavam este helper privado do antigo extractor.
        O adaptador histórico mantém a compatibilidade, mas usa o mesmo modelo
        marginal KDE/frequências do TREPAN verdadeiro. ``mlp_model`` é aceite
        apenas para preservar a assinatura e não altera a distribuição gerada.
        """
        X = np.asarray(X_encoded, dtype=float)
        n = max(0, int(sample_size))
        if X.ndim != 2 or len(X) == 0:
            raise ValueError('A geração sintética exige matriz 2D não vazia.')
        if n == 0:
            return np.empty((0, X.shape[1]), dtype=float)
        model = FeatureDistributionModel(random_state=self.random_state).fit(X)
        return model.draw(n)

    @staticmethod
    def _rules_text(tree: TrepanOriginalClassifier, class_names: Optional[Sequence[str]]) -> str:
        class_names = list(class_names or [])
        lines = []
        for idx, rule in enumerate(tree.export_rules(), start=1):
            pred = rule['prediction']
            try:
                pi = int(pred)
            except (TypeError, ValueError):
                pi = -1
            label = class_names[pi] if 0 <= pi < len(class_names) else str(pred)
            premise = ' E '.join(rule['conditions']) or 'VERDADEIRO'
            confidence = max(rule.get('probabilities') or [0.0])
            lines.append(
                f"R{idx}: SE {premise}, ENTÃO classe {label} "
                f"(confiança={confidence:.3f}, suporte={rule.get('support', 0)})"
            )
        return '\n'.join(lines)

    def extract_tree(
        self,
        mlp_model,
        X_encoded,
        y_encoded,
        sample_size=2000,
        feature_names=None,
        class_names=None,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        training_limits=None,
        extra_X=None,
        extra_y=None,
        extra_weights=None,
    ):
        X_all = np.asarray(X_encoded, dtype=float)
        X_fit = np.asarray(X_train if X_train is not None else X_all, dtype=float)
        if X_fit.ndim != 2 or len(X_fit) == 0:
            raise ValueError('TREPAN Original exige dados de treino 2D não vazios.')
        names = list(feature_names or [f'x{i}' for i in range(X_fit.shape[1])])
        if len(names) != X_fit.shape[1]:
            raise ValueError(
                f'O schema do TREPAN Original tem {len(names)} nomes para {X_fit.shape[1]} colunas.'
            )
        extra_seed_samples = 0
        if extra_X is not None:
            extra = np.asarray(extra_X, dtype=float)
            if extra.ndim != 2 or extra.shape[1] != X_fit.shape[1]:
                raise ValueError(
                    'Os contrafactuais usados para melhorar o TREPAN Original não pertencem ao mesmo schema.'
                )
            if len(extra):
                X_fit = np.vstack([X_fit, extra])
                extra_seed_samples = int(len(extra))
        if extra_y is not None or extra_weights is not None:
            warnings.warn(
                'O TREPAN histórico volta a consultar o oráculo para as sementes contrafactuais; '
                'extra_y/extra_weights são ignorados para não substituir a autoridade do oráculo.',
                RuntimeWarning, stacklevel=2,
            )
        model = self._build_classifier(
            training_limits=training_limits, sample_size=int(sample_size), n_train=len(X_fit)
        )
        model.fit(X_fit, oracle=mlp_model, feature_names=names)
        self.explainer_tree = model

        y_oracle_train = np.asarray(mlp_model.predict(X_fit))
        y_tree_train = np.asarray(model.predict(X_fit))
        audit: dict[str, Any] = {
            'algorithm': 'TREPAN Original',
            'implementation': 'core.trepan_original.TrepanOriginalClassifier',
            'training_target': 'MLP Original predictions',
            'historical_core': True,
            'distilled_cart': False,
            'best_first': bool(getattr(model, 'best_first_', False)),
            'm_of_n_used': bool(getattr(model, 'm_of_n_', False)),
            'membership_queries': int(getattr(model, 'membership_queries_', 0)),
            'oracle_query_count': int(getattr(model, 'oracle_query_count_', len(X_fit))),
            'node_count': int(getattr(model, 'node_count_', 0)),
            'depth': int(model.get_depth()),
            'leaves': int(model.get_n_leaves()),
            'effective_min_sample': int(getattr(model, 'effective_min_sample_', model.min_sample)),
            'requested_min_sample': int(model.min_sample),
            'trepan_fidelity_train': float(accuracy_score(y_oracle_train, y_tree_train)),
            'configuration_selection_scope': 'training_only',
            'final_test_used_for_selection': False,
            'split_audit': list(getattr(model, 'split_audit_', [])),
            'expansion_order': list(getattr(model, 'expansion_order_', [])),
            'extra_seed_samples': extra_seed_samples,
        }

        # O teste externo só é medido depois de o modelo estar completamente ajustado.
        if X_test is not None:
            X_eval = np.asarray(X_test, dtype=float)
            y_oracle_eval = np.asarray(mlp_model.predict(X_eval))
            y_tree_eval = np.asarray(model.predict(X_eval))
            audit['trepan_fidelity'] = float(accuracy_score(y_oracle_eval, y_tree_eval))
            if y_test is not None:
                y_eval = np.asarray(y_test)
                audit.update({
                    'trepan_accuracy': float(accuracy_score(y_eval, y_tree_eval)),
                    'trepan_balanced_accuracy': float(balanced_accuracy_score(y_eval, y_tree_eval)),
                    'trepan_macro_f1': float(f1_score(y_eval, y_tree_eval, average='macro', zero_division=0)),
                    'trepan_precision_macro': float(precision_score(y_eval, y_tree_eval, average='macro', zero_division=0)),
                    'trepan_recall_macro': float(recall_score(y_eval, y_tree_eval, average='macro', zero_division=0)),
                    'mlp_accuracy': float(accuracy_score(y_eval, y_oracle_eval)),
                })
        else:
            audit['trepan_fidelity'] = audit['trepan_fidelity_train']
        # Relatório de construção auditável (queries, best-first, m-of-n, paragens, poda).
        from core.tree_build_report import trepan_build_report
        y_oracle_eval_report = None if X_test is None else np.asarray(mlp_model.predict(np.asarray(X_test, dtype=float)))
        report = trepan_build_report(
            model, algorithm='TREPAN Original', oracle_name='MLP Original',
            oracle_type=type(mlp_model).__name__,
            X_eval=None if X_test is None else np.asarray(X_test, dtype=float),
            y_oracle_eval=y_oracle_eval_report,
            y_real_eval=None if (X_test is None or y_test is None) else np.asarray(y_test),
        )
        audit['build_report'] = report
        audit['stop_reasons'] = report['STOP_REASONS']
        audit['query_budget_exhausted'] = report['CONSTRUCTION']['query_budget_exhausted']
        audit['stump_diagnostic'] = report['STUMP']
        audit['oracle_name'] = 'MLP Original'
        self.last_audit = audit

        rules = model.export_text(class_names)
        summary = (
            '\n\nTREPAN Original histórico:\n'
            f"• Fidelidade ao MLP (treino): {audit['trepan_fidelity_train']:.3f}\n"
            f"• Nós: {audit['node_count']} | profundidade: {audit['depth']} | folhas: {audit['leaves']}\n"
            f"• Membership queries: {audit['membership_queries']}\n"
            f"• m-of-n usado: {'sim' if audit['m_of_n_used'] else 'não'}\n"
        )
        return rules + summary

    def adopt_tree(self, tree: TrepanOriginalClassifier):
        if not isinstance(tree, TrepanOriginalClassifier):
            raise TypeError('Apenas TrepanOriginalClassifier pode ser adoptado como TREPAN Original histórico.')
        self.explainer_tree = tree
        return tree

    def export_tree_image(self, feature_names=None, class_names=None, output_file='trepan_original.dot', open_image=False):
        """Exporta DOT nativo, preservando testes m-of-n sem converter para CART."""
        if self.explainer_tree is None:
            raise ValueError('Árvore ainda não foi gerada. Execute extract_tree primeiro.')
        tree = self.explainer_tree
        names = list(feature_names or tree.feature_names_in_)
        labels = list(class_names or [])
        lines = ['digraph TrepanOriginal {', '  rankdir=TB;', '  node [shape=ellipse];']
        counter = itertools.count()

        def walk(node):
            node_id = next(counter)
            if node.is_leaf:
                pred = node.prediction
                try:
                    pi = int(pred)
                except (TypeError, ValueError):
                    pi = -1
                label = labels[pi] if 0 <= pi < len(labels) else str(pred)
                lines.append(f'  n{node_id} [label="classe={label}\nreach={node.reach:.3f}", shape=ellipse];')
                return node_id
            text = node.test.text(names).replace('\"', '\\"')
            lines.append(f'  n{node_id} [label="{text}\nreach={node.reach:.3f}"];')
            left = walk(node.false_child)
            right = walk(node.true_child)
            lines.append(f'  n{node_id} -> n{left} [label="não"];')
            lines.append(f'  n{node_id} -> n{right} [label="sim"];')
            return node_id

        walk(tree.root_)
        lines.append('}')
        path = os.path.abspath(output_file)
        if not path.lower().endswith('.dot'):
            path += '.dot'
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines))
        self.last_exported_image_path = path
        return f'TREPAN Original exportado em DOT para {path}'




__all__ = [
    "Literal", "MofNTest", "ConstraintSet", "FeatureDistributionModel",
    "TrepanOriginalClassifier", "TrepanOriginalExtractor",
]
