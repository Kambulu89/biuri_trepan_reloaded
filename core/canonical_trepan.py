"""TREPAN best-first com testes m-of-n e mérito multiobjetivo.

Esta implementação é independente do CART: os nós podem combinar vários
literais e a fila de expansão é ordenada pelo erro/fidelidade esperado.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


@dataclass(frozen=True)
class Literal:
    feature: int
    threshold: float
    greater: bool = True

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        values = X[:, self.feature]
        return values > self.threshold if self.greater else values <= self.threshold


@dataclass(frozen=True)
class MofNTest:
    m: int
    literals: Tuple[Literal, ...]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        votes = np.column_stack([literal.evaluate(X) for literal in self.literals]).sum(axis=1)
        return votes >= self.m

    def text(self, feature_names: Optional[Sequence[str]] = None) -> str:
        names = feature_names or []
        chunks = []
        for item in self.literals:
            name = names[item.feature] if item.feature < len(names) else f"x{item.feature}"
            chunks.append(f"{name} {'>' if item.greater else '<='} {item.threshold:.6g}")
        return f"{self.m}-of-{len(chunks)}(" + "; ".join(chunks) + ")"


@dataclass
class _Node:
    indices: np.ndarray
    depth: int
    distribution: np.ndarray
    prediction: Any
    test: Optional[MofNTest] = None
    false_child: Optional["_Node"] = None
    true_child: Optional["_Node"] = None
    split_audit: dict = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.test is None


def _entropy(y: np.ndarray, weights: np.ndarray, classes: np.ndarray) -> float:
    mass = np.asarray([weights[y == cls].sum() for cls in classes], dtype=float)
    if mass.sum() <= 0:
        return 0.0
    p = mass / mass.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _gain(y, weights, mask, classes) -> float:
    total = weights.sum()
    if total <= 0 or not np.any(mask) or np.all(mask):
        return 0.0
    left_w, right_w = weights[~mask], weights[mask]
    return float(
        _entropy(y, weights, classes)
        - left_w.sum() / total * _entropy(y[~mask], left_w, classes)
        - right_w.sum() / total * _entropy(y[mask], right_w, classes)
    )


class CanonicalTrepanClassifier(ClassifierMixin, BaseEstimator):
    """Best-first TREPAN com beam search de testes ``m``-of-``n``."""

    def __init__(
        self,
        max_nodes: int = 31,
        max_depth: int = 8,
        min_samples_leaf: int = 4,
        max_n: int = 3,
        beam_width: int = 14,
        max_features_per_node: int = 12,
        oracle_gain_weight: float = 0.20,
        label_gain_weight: float = 0.38,
        class_balance_weight: float = 0.20,
        semantic_weight: float = 0.12,
        stability_weight: float = 0.10,
        complexity_penalty: float = 0.015,
        random_state: int = 42,
    ):
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_n = max_n
        self.beam_width = beam_width
        self.max_features_per_node = max_features_per_node
        self.oracle_gain_weight = oracle_gain_weight
        self.label_gain_weight = label_gain_weight
        self.class_balance_weight = class_balance_weight
        self.semantic_weight = semantic_weight
        self.stability_weight = stability_weight
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state

    def _distribution(self, y, w):
        mass = np.asarray([w[y == cls].sum() for cls in self.classes_], dtype=float)
        if mass.sum() <= 0:
            mass[:] = 1.0
        return mass / mass.sum()

    def _candidate_literals(self, X, y_oracle, weights):
        variance = np.nanvar(X, axis=0)
        feature_order = np.argsort(-variance, kind="stable")[: self.max_features_per_node]
        candidates = []
        for feature in feature_order:
            values = X[:, feature]
            thresholds = np.unique(np.nanquantile(values, [0.2, 0.35, 0.5, 0.65, 0.8]))
            for threshold in thresholds:
                for greater in (True, False):
                    literal = Literal(int(feature), float(threshold), greater)
                    mask = literal.evaluate(X)
                    if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
                        continue
                    score = _gain(y_oracle, weights, mask, self.classes_)
                    candidates.append((score, literal))
        candidates.sort(key=lambda item: (-item[0], item[1].feature, item[1].threshold))
        # Evita a mesma fronteira nas duas direções dentro do beam.
        return [literal for _, literal in candidates[: self.beam_width]]

    def _best_test(self, X, y_oracle, y_true, weights):
        literals = self._candidate_literals(X, y_oracle, weights)
        semantic = self.semantic_feature_indices_
        labels, counts = np.unique(y_true, return_counts=True)
        inverse_frequency = {
            label: float(len(y_true) / (len(labels) * count))
            for label, count in zip(labels, counts)
        }
        balanced_weights = weights * np.asarray(
            [inverse_frequency[label] for label in y_true], dtype=float,
        )
        best = None
        max_n = min(max(1, int(self.max_n)), len(literals))
        for n in range(1, max_n + 1):
            combinations = itertools.combinations(literals, n)
            # O beam superior limita o custo combinatório deterministicamente.
            for combo_index, combo in enumerate(combinations):
                if combo_index >= self.beam_width * self.beam_width:
                    break
                if len({item.feature for item in combo}) != len(combo):
                    continue
                for m in range(1, n + 1):
                    test = MofNTest(m, tuple(combo))
                    mask = test.evaluate(X)
                    if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
                        continue
                    oracle_gain = _gain(y_oracle, weights, mask, self.classes_)
                    label_gain = _gain(y_true, weights, mask, self.true_classes_)
                    class_balance_gain = _gain(
                        y_true, balanced_weights, mask, self.true_classes_,
                    )
                    semantic_score = float(np.mean([
                        literal.feature in semantic for literal in combo
                    ])) if semantic else 0.0
                    # Margens maiores são menos sensíveis a pequenas perturbações.
                    margins = []
                    for literal in combo:
                        scale = np.std(X[:, literal.feature]) + 1e-12
                        margins.append(np.median(np.abs(X[:, literal.feature] - literal.threshold)) / scale)
                    stability = float(np.tanh(np.mean(margins)))
                    score = (
                        self.oracle_gain_weight * oracle_gain
                        + self.label_gain_weight * label_gain
                        + self.class_balance_weight * class_balance_gain
                        + self.semantic_weight * semantic_score
                        + self.stability_weight * stability
                        - self.complexity_penalty * max(0, n - 1)
                    )
                    record = (
                        score, oracle_gain, label_gain, class_balance_gain,
                        semantic_score, stability, test, mask,
                    )
                    if best is None or record[0] > best[0] + 1e-12:
                        best = record
        return best

    def fit(
        self,
        X,
        y,
        *,
        y_true=None,
        sample_weight=None,
        semantic_feature_indices: Optional[Iterable[int]] = None,
        feature_names: Optional[Sequence[str]] = None,
    ):
        X = np.asarray(X, dtype=float)
        y_oracle = np.asarray(y)
        y_true = y_oracle if y_true is None else np.asarray(y_true)
        if X.ndim != 2 or len(X) != len(y_oracle) or len(y_true) != len(X):
            raise ValueError("X/y incompatíveis no TREPAN canónico.")
        weights = np.ones(len(X), dtype=float) if sample_weight is None else np.asarray(
            sample_weight, dtype=float,
        )
        if weights.shape != (len(X),) or np.any(weights < 0):
            raise ValueError("sample_weight inválido no TREPAN canónico.")
        self.n_features_in_ = int(X.shape[1])
        self.classes_ = np.unique(y_oracle)
        self.true_classes_ = np.unique(y_true)
        self.semantic_feature_indices_ = set(int(v) for v in (semantic_feature_indices or []))
        self.feature_names_in_ = list(feature_names or [f"x{i}" for i in range(X.shape[1])])
        self._X_fit, self._y_oracle, self._y_true, self._weights = X, y_oracle, y_true, weights
        root_idx = np.arange(len(X))
        root_dist = self._distribution(y_oracle, weights)
        self.root_ = _Node(root_idx, 0, root_dist, self.classes_[int(np.argmax(root_dist))])
        heap = []
        counter = itertools.count()

        def push(node):
            oracle_impurity = _entropy(
                y_oracle[node.indices], weights[node.indices], self.classes_,
            )
            true_impurity = _entropy(
                y_true[node.indices], weights[node.indices], self.true_classes_,
            )
            impurity = 0.65 * true_impurity + 0.35 * oracle_impurity
            priority = impurity * float(weights[node.indices].sum())
            heapq.heappush(heap, (-priority, next(counter), node))

        push(self.root_)
        node_count = 1
        expanded = []
        while heap and node_count + 2 <= int(self.max_nodes):
            _, _, node = heapq.heappop(heap)
            if node.depth >= int(self.max_depth) or len(node.indices) < 2 * int(self.min_samples_leaf):
                continue
            idx = node.indices
            best = self._best_test(X[idx], y_oracle[idx], y_true[idx], weights[idx])
            if best is None or best[0] <= 1e-12:
                continue
            (
                score, oracle_gain, label_gain, class_balance_gain,
                semantic_score, stability, test, local_mask,
            ) = best
            false_idx, true_idx = idx[~local_mask], idx[local_mask]
            false_dist = self._distribution(y_oracle[false_idx], weights[false_idx])
            true_dist = self._distribution(y_oracle[true_idx], weights[true_idx])
            node.test = test
            node.false_child = _Node(
                false_idx, node.depth + 1, false_dist,
                self.classes_[int(np.argmax(false_dist))],
            )
            node.true_child = _Node(
                true_idx, node.depth + 1, true_dist,
                self.classes_[int(np.argmax(true_dist))],
            )
            node.split_audit = {
                "score": float(score), "oracle_gain": float(oracle_gain),
                "label_gain": float(label_gain), "semantic": float(semantic_score),
                "class_balance_gain": float(class_balance_gain),
                "stability": float(stability), "test": test.text(self.feature_names_in_),
                "m": int(test.m), "n": int(len(test.literals)),
            }
            expanded.append(node.split_audit)
            node_count += 2
            push(node.false_child)
            push(node.true_child)
        self.node_count_ = node_count
        self.split_audit_ = expanded
        self.best_first_ = True
        self.m_of_n_ = any(item["n"] > 1 for item in expanded)
        return self

    def _leaf(self, row):
        node = self.root_
        while not node.is_leaf:
            branch = bool(node.test.evaluate(np.asarray(row).reshape(1, -1))[0])
            node = node.true_child if branch else node.false_child
        return node

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.asarray([self._leaf(row).distribution for row in X], dtype=float)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def get_depth(self):
        def depth(node):
            return node.depth if node.is_leaf else max(depth(node.false_child), depth(node.true_child))
        return int(depth(self.root_))

    def get_n_leaves(self):
        def leaves(node):
            return 1 if node.is_leaf else leaves(node.false_child) + leaves(node.true_child)
        return int(leaves(self.root_))

    def export_rules(self):
        rules = []
        def walk(node, conditions):
            if node.is_leaf:
                rules.append({
                    "conditions": list(conditions), "prediction": node.prediction,
                    "probabilities": node.distribution.tolist(),
                    "support": int(len(node.indices)),
                })
                return
            text = node.test.text(self.feature_names_in_)
            walk(node.false_child, conditions + [f"NÃO {text}"])
            walk(node.true_child, conditions + [text])
        walk(self.root_, [])
        return rules


__all__ = ["Literal", "MofNTest", "CanonicalTrepanClassifier"]
