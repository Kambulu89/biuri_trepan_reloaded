"""Implementação nativa do algoritmo C4.5.

Não usa ``DecisionTreeClassifier`` nem Weka. Implementa gain ratio, splits
multirramos nominais, distribuição fracionária de valores ausentes e poda
pessimista. O nome histórico ``C45J48Tree`` é mantido só por compatibilidade;
J48 é uma implementação do Weka e não é utilizado neste projeto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


EPS = 1e-12
TREE_LEAF = -1
TREE_UNDEFINED = -2


def _entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= EPS:
        return 0.0
    probabilities = counts[counts > EPS] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return str(value).strip() in {"", "?"}


def _normalise_feature_type(value: Any) -> str:
    text = str(value or "numeric").strip().lower()
    if text.startswith("{") or any(
        token in text for token in ("nominal", "categorical", "string", "bool")
    ):
        return "categorical"
    return "numeric"


@dataclass
class _Split:
    feature: int
    kind: str
    gain: float
    gain_ratio: float
    threshold: Optional[float] = None
    categories: Tuple[float, ...] = ()
    branch_priors: Dict[Any, float] = field(default_factory=dict)


@dataclass
class _C45Node:
    class_counts: np.ndarray
    n_samples: float
    prediction_index: int
    depth: int
    feature: Optional[int] = None
    kind: Optional[str] = None
    threshold: Optional[float] = None
    children: Dict[Any, "_C45Node"] = field(default_factory=dict)
    branch_priors: Dict[Any, float] = field(default_factory=dict)
    default_branch: Any = None
    gain: float = 0.0
    gain_ratio: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature is None or not self.children


class C45Classifier(ClassifierMixin, BaseEstimator):
    """C4.5 nativo compatível com a API essencial do scikit-learn."""

    _estimator_type = "classifier"
    algorithm_name = "C4.5-Nativo"
    is_c45_native = True

    def __init__(
        self,
        *,
        confidence_factor: float = 0.25,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        max_depth: Optional[int] = None,
        min_gain_ratio: float = 1e-9,
        feature_types: Optional[Sequence[str]] = None,
        random_state: Optional[int] = 42,
    ):
        self.confidence_factor = confidence_factor
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_gain_ratio = min_gain_ratio
        self.feature_types = feature_types
        self.random_state = random_state

    def _validate_params(self) -> None:
        if not 0.0 < float(self.confidence_factor) < 0.5:
            raise ValueError("confidence_factor deve estar entre 0 e 0,5.")
        if int(self.min_samples_split) < 2:
            raise ValueError("min_samples_split deve ser >= 2.")
        if int(self.min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf deve ser >= 1.")
        if self.max_depth is not None and int(self.max_depth) < 1:
            raise ValueError("max_depth deve ser None ou >= 1.")

    @staticmethod
    def _as_2d_object(X: Any) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("X deve ser uma matriz bidimensional.")
        return arr

    def _resolve_feature_types(self, X: np.ndarray) -> Tuple[str, ...]:
        if self.feature_types is not None:
            types = tuple(_normalise_feature_type(v) for v in self.feature_types)
            if len(types) != X.shape[1]:
                raise ValueError(
                    f"feature_types contém {len(types)} entradas para {X.shape[1]} features."
                )
            return types
        inferred = []
        for col in range(X.shape[1]):
            values = [v for v in X[:, col] if not _is_missing(v)]
            numeric = True
            for value in values:
                try:
                    float(value)
                except (TypeError, ValueError):
                    numeric = False
                    break
            inferred.append("numeric" if numeric else "categorical")
        return tuple(inferred)

    def _fit_category_maps(self, X: np.ndarray) -> None:
        self.category_maps_: Dict[int, Dict[str, float]] = {}
        self.category_labels_: Dict[int, Dict[float, str]] = {}
        for col, kind in enumerate(self.feature_types_):
            if kind != "categorical":
                continue
            values = [v for v in X[:, col] if not _is_missing(v)]
            numeric = True
            numeric_values = []
            for value in values:
                try:
                    numeric_values.append(float(value))
                except (TypeError, ValueError):
                    numeric = False
                    break
            if numeric:
                unique = sorted(set(numeric_values))
                self.category_labels_[col] = {float(v): str(v) for v in unique}
            else:
                labels = sorted(set(str(v) for v in values))
                mapping = {label: float(i) for i, label in enumerate(labels)}
                self.category_maps_[col] = mapping
                self.category_labels_[col] = {code: label for label, code in mapping.items()}

    def _transform_X(self, X: Any) -> np.ndarray:
        raw = self._as_2d_object(X)
        if hasattr(self, "n_features_in_") and raw.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Schema incompatível: C4.5 espera {self.n_features_in_} features, "
                f"mas recebeu {raw.shape[1]}."
            )
        out = np.full(raw.shape, np.nan, dtype=float)
        for col in range(raw.shape[1]):
            mapping = getattr(self, "category_maps_", {}).get(col)
            for row, value in enumerate(raw[:, col]):
                if _is_missing(value):
                    continue
                if mapping is not None:
                    out[row, col] = mapping.get(str(value), np.nan)
                else:
                    try:
                        out[row, col] = float(value)
                    except (TypeError, ValueError):
                        out[row, col] = np.nan
        return out

    def fit(self, X: Any, y: Any, sample_weight: Optional[Any] = None):
        self._validate_params()
        raw = self._as_2d_object(X)
        y_arr = np.asarray(y)
        if len(raw) != len(y_arr) or len(raw) == 0:
            raise ValueError("X e y devem ter o mesmo número positivo de amostras.")
        self.n_features_in_ = int(raw.shape[1])
        self.feature_types_ = self._resolve_feature_types(raw)
        self._fit_category_maps(raw)
        X_arr = self._transform_X(raw)
        self.classes_, y_indices = np.unique(y_arr, return_inverse=True)
        self.n_classes_ = int(len(self.classes_))
        if self.n_classes_ < 2:
            raise ValueError("C4.5 requer pelo menos duas classes no treino.")
        weights = (
            np.ones(len(y_arr), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).copy()
        )
        if weights.shape != (len(y_arr),) or np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight inválido.")
        if float(weights.sum()) <= EPS:
            raise ValueError("A soma de sample_weight deve ser positiva.")

        self._X_fit_ = X_arr
        self._y_fit_indices_ = y_indices.astype(int)
        available_categorical = {
            i for i, kind in enumerate(self.feature_types_) if kind == "categorical"
        }
        indices = np.arange(len(y_arr), dtype=int)
        self.root_ = self._build_node(indices, weights, depth=0, available_categorical=available_categorical)
        self._prune(self.root_)
        self._build_tree_adapter()
        self.feature_importances_ = self._calculate_feature_importances()
        return self

    def _counts(self, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.bincount(
            self._y_fit_indices_[indices], weights=weights, minlength=self.n_classes_
        ).astype(float)

    def _numeric_candidate(
        self, feature: int, indices: np.ndarray, weights: np.ndarray
    ) -> Optional[_Split]:
        values = self._X_fit_[indices, feature]
        known_mask = np.isfinite(values)
        if int(np.sum(known_mask)) < 2:
            return None
        known_idx = indices[known_mask]
        known_weights = weights[known_mask]
        known_values = values[known_mask]
        order = np.argsort(known_values, kind="mergesort")
        known_values = known_values[order]
        known_idx = known_idx[order]
        known_weights = known_weights[order]
        known_y = self._y_fit_indices_[known_idx]
        total_known_weight = float(known_weights.sum())
        total_weight = float(weights.sum())
        missing_weight = max(0.0, total_weight - total_known_weight)
        if total_known_weight <= EPS:
            return None

        total_counts = np.bincount(
            known_y, weights=known_weights, minlength=self.n_classes_
        ).astype(float)
        cumulative = np.zeros((len(known_values), self.n_classes_), dtype=float)
        running = np.zeros(self.n_classes_, dtype=float)
        for pos, (label, weight) in enumerate(zip(known_y, known_weights)):
            running[int(label)] += float(weight)
            cumulative[pos] = running

        candidate_positions = [
            pos for pos in range(len(known_values) - 1)
            if known_values[pos] < known_values[pos + 1]
            and known_y[pos] != known_y[pos + 1]
        ]
        # Correção MDL de C4.5 para não favorecer atributos contínuos apenas
        # porque oferecem muitos limiares possíveis.
        threshold_penalty = (
            np.log2(len(candidate_positions)) / total_weight
            if len(candidate_positions) > 1 else 0.0
        )
        best: Optional[_Split] = None
        for pos in candidate_positions:
            left_counts = cumulative[pos]
            right_counts = total_counts - left_counts
            left_weight = float(left_counts.sum())
            right_weight = float(right_counts.sum())
            if left_weight + EPS < self.min_samples_leaf or right_weight + EPS < self.min_samples_leaf:
                continue
            child_entropy = (
                left_weight / total_known_weight * _entropy(left_counts)
                + right_weight / total_known_weight * _entropy(right_counts)
            )
            raw_gain = (total_known_weight / total_weight) * (
                _entropy(total_counts) - child_entropy
            )
            gain = max(0.0, float(raw_gain - threshold_penalty))
            split_info = _entropy(np.asarray([left_weight, right_weight, missing_weight]))
            ratio = gain / split_info if split_info > EPS else 0.0
            threshold = float((known_values[pos] + known_values[pos + 1]) / 2.0)
            candidate = _Split(
                feature=feature,
                kind="numeric",
                gain=float(gain),
                gain_ratio=float(ratio),
                threshold=threshold,
                branch_priors={
                    "left": left_weight / total_known_weight,
                    "right": right_weight / total_known_weight,
                },
            )
            if best is None or (candidate.gain_ratio, candidate.gain) > (best.gain_ratio, best.gain):
                best = candidate
        return best

    def _categorical_candidate(
        self, feature: int, indices: np.ndarray, weights: np.ndarray
    ) -> Optional[_Split]:
        values = self._X_fit_[indices, feature]
        known_mask = np.isfinite(values)
        known_values = values[known_mask]
        known_idx = indices[known_mask]
        known_weights = weights[known_mask]
        categories = tuple(float(v) for v in sorted(np.unique(known_values)))
        if len(categories) < 2:
            return None
        total_weight = float(weights.sum())
        known_weight = float(known_weights.sum())
        missing_weight = max(0.0, total_weight - known_weight)
        total_counts = self._counts(known_idx, known_weights)
        child_entropy = 0.0
        branch_weights: Dict[float, float] = {}
        for category in categories:
            mask = known_values == category
            branch_weight = float(known_weights[mask].sum())
            if branch_weight + EPS < self.min_samples_leaf:
                return None
            branch_counts = self._counts(known_idx[mask], known_weights[mask])
            child_entropy += branch_weight / known_weight * _entropy(branch_counts)
            branch_weights[category] = branch_weight
        gain = (known_weight / total_weight) * (_entropy(total_counts) - child_entropy)
        split_info = _entropy(np.asarray(list(branch_weights.values()) + [missing_weight]))
        ratio = gain / split_info if split_info > EPS else 0.0
        return _Split(
            feature=feature,
            kind="categorical",
            gain=float(gain),
            gain_ratio=float(ratio),
            categories=categories,
            branch_priors={k: v / known_weight for k, v in branch_weights.items()},
        )

    def _best_split(
        self,
        indices: np.ndarray,
        weights: np.ndarray,
        available_categorical: set,
    ) -> Optional[_Split]:
        candidates: List[_Split] = []
        for feature, kind in enumerate(self.feature_types_):
            if kind == "categorical" and feature not in available_categorical:
                continue
            candidate = (
                self._categorical_candidate(feature, indices, weights)
                if kind == "categorical"
                else self._numeric_candidate(feature, indices, weights)
            )
            if candidate is not None and candidate.gain > EPS:
                candidates.append(candidate)
        if not candidates:
            return None
        average_gain = float(np.mean([candidate.gain for candidate in candidates]))
        eligible = [candidate for candidate in candidates if candidate.gain + EPS >= average_gain]
        best = max(eligible or candidates, key=lambda item: (item.gain_ratio, item.gain))
        return best if best.gain_ratio + EPS >= float(self.min_gain_ratio) else None

    def _partition(
        self, split: _Split, indices: np.ndarray, weights: np.ndarray
    ) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
        values = self._X_fit_[indices, split.feature]
        missing_mask = ~np.isfinite(values)
        missing_indices = indices[missing_mask]
        missing_weights = weights[missing_mask]
        branches: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        keys: Iterable[Any] = ("left", "right") if split.kind == "numeric" else split.categories
        for key in keys:
            if split.kind == "numeric":
                known_mask = np.isfinite(values) & (
                    (values <= split.threshold) if key == "left" else (values > split.threshold)
                )
            else:
                known_mask = np.isfinite(values) & (values == float(key))
            branch_indices = indices[known_mask]
            branch_weights = weights[known_mask]
            prior = float(split.branch_priors[key])
            if len(missing_indices):
                branch_indices = np.concatenate([branch_indices, missing_indices])
                branch_weights = np.concatenate([branch_weights, missing_weights * prior])
            positive = branch_weights > EPS
            branches[key] = (branch_indices[positive], branch_weights[positive])
        return branches

    def _build_node(
        self,
        indices: np.ndarray,
        weights: np.ndarray,
        *,
        depth: int,
        available_categorical: set,
    ) -> _C45Node:
        counts = self._counts(indices, weights)
        node = _C45Node(
            class_counts=counts,
            n_samples=float(weights.sum()),
            prediction_index=int(np.argmax(counts)),
            depth=depth,
        )
        if (
            int(np.sum(counts > EPS)) <= 1
            or float(weights.sum()) + EPS < self.min_samples_split
            or (self.max_depth is not None and depth >= int(self.max_depth))
        ):
            return node
        split = self._best_split(indices, weights, available_categorical)
        if split is None:
            return node
        partitions = self._partition(split, indices, weights)
        if len(partitions) < 2 or any(len(part[0]) == 0 for part in partitions.values()):
            return node
        node.feature = split.feature
        node.kind = split.kind
        node.threshold = split.threshold
        node.branch_priors = dict(split.branch_priors)
        node.default_branch = max(split.branch_priors, key=split.branch_priors.get)
        node.gain = split.gain
        node.gain_ratio = split.gain_ratio
        next_categorical = set(available_categorical)
        if split.kind == "categorical":
            next_categorical.discard(split.feature)
        for key, (child_indices, child_weights) in partitions.items():
            node.children[key] = self._build_node(
                child_indices,
                child_weights,
                depth=depth + 1,
                available_categorical=set(next_categorical),
            )
        return node

    def _extra_errors(self, n: float, errors: float) -> float:
        """Correção de confiança usada na poda pessimista de Quinlan/J48."""
        n = float(max(n, EPS))
        errors = float(np.clip(errors, 0.0, n))
        cf = float(self.confidence_factor)
        if errors < 1.0:
            base = n * (1.0 - cf ** (1.0 / n))
            if errors <= EPS:
                return base
            return base + errors * (self._extra_errors(n, 1.0) - base)
        if errors + 0.5 >= n:
            return max(n - errors, 0.0)
        z = NormalDist().inv_cdf(1.0 - cf)
        f = (errors + 0.5) / n
        r = (
            f
            + z * z / (2.0 * n)
            + z * np.sqrt(max(f / n - f * f / n + z * z / (4.0 * n * n), 0.0))
        ) / (1.0 + z * z / n)
        return float(r * n - errors)

    def _leaf_estimated_error(self, node: _C45Node) -> float:
        errors = float(node.n_samples - np.max(node.class_counts))
        return errors + self._extra_errors(node.n_samples, errors)

    def _prune(self, node: _C45Node) -> float:
        if node.is_leaf:
            return self._leaf_estimated_error(node)
        subtree_error = sum(self._prune(child) for child in node.children.values())
        leaf_error = self._leaf_estimated_error(node)
        if leaf_error <= subtree_error + 0.1:
            node.feature = None
            node.kind = None
            node.threshold = None
            node.children = {}
            node.branch_priors = {}
            node.default_branch = None
            node.gain = 0.0
            node.gain_ratio = 0.0
            return leaf_error
        return subtree_error

    def _predict_node(self, row: np.ndarray) -> _C45Node:
        node = self.root_
        while not node.is_leaf:
            value = row[int(node.feature)]
            if not np.isfinite(value):
                key = node.default_branch
            elif node.kind == "numeric":
                key = "left" if value <= float(node.threshold) else "right"
            else:
                key = float(value)
                if key not in node.children:
                    key = node.default_branch
            node = node.children[key]
        return node

    def _predict_distribution(self, node: _C45Node, row: np.ndarray) -> np.ndarray:
        if node.is_leaf:
            counts = node.class_counts.astype(float)
            return (counts + 1.0) / (counts.sum() + self.n_classes_)
        value = row[int(node.feature)]
        if np.isfinite(value):
            if node.kind == "numeric":
                key = "left" if value <= float(node.threshold) else "right"
            else:
                key = float(value)
            if key in node.children:
                return self._predict_distribution(node.children[key], row)
        # C4.5 combina probabilidades dos ramos quando o valor está ausente ou
        # nunca foi observado, em vez de inventar uma categoria/valor.
        result = np.zeros(self.n_classes_, dtype=float)
        normaliser = 0.0
        for key, child in node.children.items():
            prior = float(node.branch_priors.get(key, 0.0))
            if prior <= 0.0:
                continue
            result += prior * self._predict_distribution(child, row)
            normaliser += prior
        if normaliser <= EPS:
            counts = node.class_counts.astype(float)
            return (counts + 1.0) / (counts.sum() + self.n_classes_)
        return result / normaliser

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "root_"):
            raise ValueError("C4.5 ainda não foi treinado.")
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[np.asarray(indices, dtype=int)]

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "root_"):
            raise ValueError("C4.5 ainda não foi treinado.")
        X_arr = self._transform_X(X)
        return np.asarray(
            [self._predict_distribution(self.root_, row) for row in X_arr],
            dtype=float,
        )

    def apply(self, X: Any) -> np.ndarray:
        X_arr = self._transform_X(X)
        return np.asarray([self._predict_node(row)._adapter_id for row in X_arr], dtype=int)

    def get_depth(self) -> int:
        def walk(node: _C45Node) -> int:
            return 0 if node.is_leaf else 1 + max(walk(child) for child in node.children.values())
        return int(walk(self.root_))

    def get_n_leaves(self) -> int:
        def walk(node: _C45Node) -> int:
            return 1 if node.is_leaf else sum(walk(child) for child in node.children.values())
        return int(walk(self.root_))

    def _calculate_feature_importances(self) -> np.ndarray:
        importance = np.zeros(self.n_features_in_, dtype=float)
        def walk(node: _C45Node) -> None:
            if node.is_leaf:
                return
            importance[int(node.feature)] += max(node.gain, 0.0) * node.n_samples
            for child in node.children.values():
                walk(child)
        walk(self.root_)
        total = float(importance.sum())
        return importance / total if total > EPS else importance

    def _build_tree_adapter(self) -> None:
        """Compila a árvore n-ária numa visão binária exata para a UI existente."""
        records: List[Dict[str, Any]] = []

        def add_record(node: _C45Node, *, feature=TREE_UNDEFINED, threshold=-2.0) -> int:
            idx = len(records)
            records.append({
                "left": TREE_LEAF,
                "right": TREE_LEAF,
                "feature": int(feature),
                "threshold": float(threshold),
                "samples": max(1, int(round(node.n_samples))),
                "weighted": float(node.n_samples),
                "value": node.class_counts.astype(float),
                "impurity": _entropy(node.class_counts),
            })
            return idx

        def combined_node(nodes: Sequence[_C45Node], depth: int) -> _C45Node:
            counts = np.sum([child.class_counts for child in nodes], axis=0)
            return _C45Node(
                class_counts=counts,
                n_samples=float(sum(child.n_samples for child in nodes)),
                prediction_index=int(np.argmax(counts)),
                depth=depth,
            )

        def compile_node(node: _C45Node) -> int:
            if node.is_leaf:
                idx = add_record(node)
                node._adapter_id = idx
                return idx
            if node.kind == "numeric":
                idx = add_record(node, feature=node.feature, threshold=node.threshold)
                left = compile_node(node.children["left"])
                right = compile_node(node.children["right"])
                records[idx]["left"], records[idx]["right"] = left, right
                node._adapter_id = idx
                return idx

            categories = sorted(node.children)
            def compile_chain(remaining: Sequence[float]) -> int:
                if len(remaining) == 1:
                    return compile_node(node.children[remaining[0]])
                chain_node = combined_node([node.children[v] for v in remaining], node.depth)
                boundary = float((remaining[0] + remaining[1]) / 2.0)
                idx = add_record(chain_node, feature=node.feature, threshold=boundary)
                left = compile_node(node.children[remaining[0]])
                right = compile_chain(remaining[1:])
                records[idx]["left"], records[idx]["right"] = left, right
                return idx
            idx = compile_chain(categories)
            node._adapter_id = idx
            return idx

        compile_node(self.root_)
        values = np.asarray([record["value"] for record in records], dtype=float)[:, None, :]
        self.tree_ = SimpleNamespace(
            children_left=np.asarray([record["left"] for record in records], dtype=np.int64),
            children_right=np.asarray([record["right"] for record in records], dtype=np.int64),
            feature=np.asarray([record["feature"] for record in records], dtype=np.int64),
            threshold=np.asarray([record["threshold"] for record in records], dtype=float),
            n_node_samples=np.asarray([record["samples"] for record in records], dtype=np.int64),
            weighted_n_node_samples=np.asarray([record["weighted"] for record in records], dtype=float),
            impurity=np.asarray([record["impurity"] for record in records], dtype=float),
            value=values,
            node_count=len(records),
            n_features=self.n_features_in_,
            max_depth=self.get_depth(),
            n_classes=np.asarray([self.n_classes_], dtype=np.int64),
            n_outputs=1,
        )

    def iter_rules(self) -> List[Dict[str, Any]]:
        rules: List[Dict[str, Any]] = []
        def walk(node: _C45Node, conditions: List[Tuple[int, str, Any]]) -> None:
            if node.is_leaf:
                rules.append({
                    "conditions": list(conditions),
                    "prediction": self.classes_[node.prediction_index],
                    "support": float(node.n_samples),
                    "confidence": float(np.max(node.class_counts) / max(node.class_counts.sum(), EPS)),
                })
                return
            if node.kind == "numeric":
                walk(node.children["left"], conditions + [(node.feature, "<=", node.threshold)])
                walk(node.children["right"], conditions + [(node.feature, ">", node.threshold)])
            else:
                labels = self.category_labels_.get(int(node.feature), {})
                for value, child in node.children.items():
                    walk(child, conditions + [(node.feature, "=", labels.get(value, value))])
        walk(self.root_, [])
        return rules


class C45Tree:
    """Fachada histórica que agora treina exclusivamente C4.5 nativo."""

    def __init__(self):
        self.tree_model: Optional[C45Classifier] = None
        self.feature_names: Optional[List[str]] = None
        self.class_names: Optional[List[str]] = None
        self.training_metrics: Dict[str, Any] = {}

    @staticmethod
    def _normalise_feature_types(
        feature_types: Optional[Sequence[Any]], n_features: int
    ) -> Optional[Tuple[str, ...]]:
        if feature_types is None:
            return None
        values = tuple(_normalise_feature_type(value) for value in feature_types)
        if len(values) != n_features:
            raise ValueError(
                f"Foram recebidos {len(values)} tipos ARFF para {n_features} features."
            )
        return values

    def train_c45_tree(
        self,
        X_train,
        y_train,
        feature_names=None,
        class_names=None,
        ontology_mode=False,
        feature_types=None,
    ):
        if ontology_mode:
            raise ValueError("C4.5 é um baseline no espaço ARFF original e não aceita ontologia.")
        X_arr = np.asarray(X_train, dtype=object)
        y_arr = np.asarray(y_train)
        if X_arr.ndim != 2:
            raise ValueError("X_train deve ser bidimensional.")
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(X_arr.shape[1])]
        )
        self.class_names = list(class_names) if class_names is not None else []
        types = self._normalise_feature_types(feature_types, X_arr.shape[1])

        _, counts = np.unique(y_arr, return_counts=True)
        scoring = "balanced_accuracy" if len(counts) > 1 and counts.max() / counts.min() > 1.5 else "accuracy"
        candidates = [
            {"confidence_factor": 0.25, "min_samples_leaf": 1, "max_depth": None},
            {"confidence_factor": 0.25, "min_samples_leaf": 2, "max_depth": None},
            {"confidence_factor": 0.10, "min_samples_leaf": 2, "max_depth": None},
            {"confidence_factor": 0.25, "min_samples_leaf": 2, "max_depth": 12},
        ]
        best_params = candidates[0]
        best_score = -np.inf
        if len(counts) > 1 and int(counts.min()) >= 2 and len(y_arr) >= 20:
            folds = min(5, int(counts.min()))
            cv = StratifiedKFold(folds, shuffle=True, random_state=42)
            for params in candidates:
                scores = []
                for train_idx, valid_idx in cv.split(X_arr, y_arr):
                    model = C45Classifier(
                        feature_types=types,
                        random_state=42,
                        min_samples_split=max(4, 2 * params["min_samples_leaf"]),
                        **params,
                    ).fit(X_arr[train_idx], y_arr[train_idx])
                    pred = model.predict(X_arr[valid_idx])
                    if scoring == "balanced_accuracy":
                        from sklearn.metrics import balanced_accuracy_score
                        score = balanced_accuracy_score(y_arr[valid_idx], pred)
                    else:
                        score = accuracy_score(y_arr[valid_idx], pred)
                    scores.append(float(score))
                mean_score = float(np.mean(scores))
                if mean_score > best_score + EPS:
                    best_score, best_params = mean_score, dict(params)

        self.tree_model = C45Classifier(
            feature_types=types,
            random_state=42,
            min_samples_split=max(4, 2 * best_params["min_samples_leaf"]),
            **best_params,
        ).fit(X_arr, y_arr)
        y_train_pred = self.tree_model.predict(X_arr)
        self.training_metrics = {
            "train_accuracy": float(accuracy_score(y_arr, y_train_pred)),
            "train_precision": float(precision_score(y_arr, y_train_pred, average="weighted", zero_division=0)),
            "train_recall": float(recall_score(y_arr, y_train_pred, average="weighted", zero_division=0)),
            "train_f1": float(f1_score(y_arr, y_train_pred, average="weighted", zero_division=0)),
            "tree_depth": self.tree_model.get_depth(),
            "n_leaves": self.tree_model.get_n_leaves(),
            "n_nodes": int(self.tree_model.tree_.node_count),
            "selection_scoring": scoring,
            "best_params": dict(best_params),
        }
        if np.isfinite(best_score):
            self.training_metrics["selection_cv_score"] = float(best_score)
        return self.tree_model

    def predict(self, X):
        if self.tree_model is None:
            raise ValueError("C4.5 ainda não foi treinado.")
        return self.tree_model.predict(X)

    def predict_proba(self, X):
        if self.tree_model is None:
            raise ValueError("C4.5 ainda não foi treinado.")
        return self.tree_model.predict_proba(X)

    def get_tree_info(self):
        if self.tree_model is None:
            return None
        return {
            "algorithm": "C4.5-Nativo",
            "implementation": "Python nativo; sem CART, Weka ou Java",
            "split_criterion": "gain_ratio",
            "missing_values": "fractional_instance_weighting",
            "pruning": "pessimistic_error_pruning",
            "confidence_factor": self.tree_model.confidence_factor,
            "depth": self.tree_model.get_depth(),
            "n_leaves": self.tree_model.get_n_leaves(),
            "n_nodes": int(self.tree_model.tree_.node_count),
            "training_metrics": dict(self.training_metrics),
        }

    def generate_tree_rules(self):
        if self.tree_model is None:
            return "❌ C4.5 ainda não foi treinado."
        lines = [
            "=" * 64,
            "🌳 C4.5 NATIVO — GAIN RATIO + PODA PESSIMISTA",
            "=" * 64,
            f"Profundidade: {self.tree_model.get_depth()}",
            f"Folhas: {self.tree_model.get_n_leaves()}",
            f"Nós (visão binária): {self.tree_model.tree_.node_count}",
            "Espaço: atributos ARFF originais; rótulos reais",
            "",
            "Regras:",
        ]
        for number, rule in enumerate(self.tree_model.iter_rules(), 1):
            conditions = []
            for feature, operator, value in rule["conditions"]:
                name = self.feature_names[int(feature)]
                rendered = f"{float(value):.6g}" if isinstance(value, (int, float, np.number)) else str(value)
                conditions.append(f"{name} {operator} {rendered}")
            predicted = rule["prediction"]
            try:
                class_index = int(predicted)
                if self.class_names and 0 <= class_index < len(self.class_names):
                    predicted = self.class_names[class_index]
            except (TypeError, ValueError):
                pass
            lines.append(
                f"R{number}: SE {' E '.join(conditions) if conditions else 'verdadeiro'} "
                f"ENTÃO classe={predicted} "
                f"[suporte={rule['support']:.2f}; confiança={rule['confidence']:.1%}]"
            )
        return "\n".join(lines)

    def calculate_complexity_metrics(self):
        if self.tree_model is None:
            return None
        depth = self.tree_model.get_depth()
        leaves = self.tree_model.get_n_leaves()
        path_lengths = [len(rule["conditions"]) for rule in self.tree_model.iter_rules()]
        return {
            "depth": depth,
            "n_leaves": leaves,
            "n_nodes": int(self.tree_model.tree_.node_count),
            "avg_path_length": float(np.mean(path_lengths)) if path_lengths else 0.0,
            "complexity_score": float(0.6 * min(depth / 15, 1.0) + 0.4 * min(leaves / 100, 1.0)),
            "interpretability_score": 0.6 if depth <= 3 and leaves <= 8 else (0.4 if depth <= 5 and leaves <= 16 else 0.2),
            "algorithm_type": "C4.5-Nativo",
            "explainability_level": "Global rule-based",
        }

    def compare_with_trepan(self, trepan_tree, mlp_model, X_test, y_test):
        if self.tree_model is None:
            return "❌ C4.5 ainda não foi treinado."
        y_c45 = self.predict(X_test)
        y_trepan = trepan_tree.predict(X_test)
        y_mlp = mlp_model.predict(X_test)
        metric = lambda truth, pred: {
            "accuracy": float(accuracy_score(truth, pred)),
            "precision": float(precision_score(truth, pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(truth, pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(truth, pred, average="weighted", zero_division=0)),
        }
        c45_metrics, trepan_metrics = metric(y_test, y_c45), metric(y_test, y_trepan)
        c45_fidelity = float(accuracy_score(y_mlp, y_c45))
        trepan_fidelity = float(accuracy_score(y_mlp, y_trepan))
        return {
            "c45_metrics": c45_metrics,
            "trepan_metrics": trepan_metrics,
            "c45_fidelity": c45_fidelity,
            "trepan_fidelity": trepan_fidelity,
            "accuracy_difference": trepan_metrics["accuracy"] - c45_metrics["accuracy"],
            "fidelity_difference": trepan_fidelity - c45_fidelity,
            "interpretability_advantage": "comparar complexidade observada",
        }

    def generate_comparison_report(self, trepan_tree, mlp_model, X_test, y_test, trepan_type="Trepan"):
        result = self.compare_with_trepan(trepan_tree, mlp_model, X_test, y_test)
        if isinstance(result, str):
            return result
        return (
            f"C4.5-Nativo vs {trepan_type}\n"
            f"Acurácia: {result['c45_metrics']['accuracy']:.3f} vs "
            f"{result['trepan_metrics']['accuracy']:.3f}\n"
            f"Fidelidade ao MLP: {result['c45_fidelity']:.3f} vs "
            f"{result['trepan_fidelity']:.3f}\n"
            "Comparação realizada nas mesmas amostras e no espaço ARFF original."
        )


# Alias legado: artefactos antigos podem ainda referenciar este símbolo.
C45J48Tree = C45Tree
