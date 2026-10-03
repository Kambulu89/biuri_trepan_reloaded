"""Motor contrafactual unificado e independente da interface gráfica.

O módulo mantém o subsistema pós-treino: não modifica MLPs, árvores TREPAN,
métricas preditivas nem a ontologia.  Implementa os contratos descritos no
artigo do projeto (DiCE/fallback, CLEAR, CoGS, LORE, restrições, validação,
métricas, robustez local e filtro inspirado em Rough Set Theory).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-9


def _as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def _predict(model: Any, values: Any) -> np.ndarray:
    return np.asarray(model.predict(_as_2d(values))).reshape(-1)


def _predict_proba(model: Any, values: Any) -> Optional[np.ndarray]:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        return np.asarray(model.predict_proba(_as_2d(values)), dtype=float)
    except Exception:
        return None


def model_n_features(model: Any) -> Optional[int]:
    """Obtém dimensionalidade sem depender de uma classe concreta de modelo."""
    for obj in (model, getattr(model, "model", None)):
        if obj is None:
            continue
        value = getattr(obj, "n_features_in_", None)
        if value is not None:
            return int(value)
        tree = getattr(obj, "tree_", None)
        if tree is not None and getattr(tree, "n_features", None) is not None:
            return int(tree.n_features)
    return None


def _normalise_feature_name(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


@dataclass
class CounterfactualConstraints:
    """Restrições de domínio e ação no espaço efetivamente usado pelo modelo."""

    feature_names: List[str]
    feature_ranges: Dict[str, Tuple[float, float]]
    immutable_features: set = field(default_factory=set)
    actionable_features: Optional[set] = None
    protected_features: set = field(default_factory=set)
    categorical_values: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    binary_features: set = field(default_factory=set)
    causal_rules: List[Mapping[str, Any]] = field(default_factory=list)

    @classmethod
    def from_data(
        cls,
        X: Any,
        feature_names: Sequence[str],
        config: Optional[Mapping[str, Any]] = None,
    ) -> "CounterfactualConstraints":
        config = dict(config or {})
        names = [str(name) for name in feature_names]
        arr = _as_2d(X)
        configured_ranges = dict(config.get("feature_ranges") or {})
        ranges: Dict[str, Tuple[float, float]] = {}
        binary = set(map(str, config.get("binary_features") or []))
        categorical_values = {
            str(key): tuple(float(v) for v in values)
            for key, values in (config.get("categorical_values") or {}).items()
        }

        for idx, name in enumerate(names):
            if name in configured_ranges:
                low, high = configured_ranges[name]
            else:
                column = arr[:, idx]
                finite = column[np.isfinite(column)]
                if finite.size:
                    low, high = float(np.min(finite)), float(np.max(finite))
                else:
                    low = high = 0.0
            if high < low:
                low, high = high, low
            ranges[name] = (float(low), float(high))
            unique = np.unique(arr[:, idx][np.isfinite(arr[:, idx])])
            if unique.size and set(np.round(unique, 12)).issubset({0.0, 1.0}):
                binary.add(name)

        actionable_raw = config.get("actionable_features")
        actionable = set(map(str, actionable_raw)) if actionable_raw is not None else None
        immutable = set(map(str, config.get("immutable_features") or []))
        protected = set(map(str, config.get("protected_features") or []))
        immutable |= protected

        return cls(
            feature_names=names,
            feature_ranges=ranges,
            immutable_features=immutable,
            actionable_features=actionable,
            protected_features=protected,
            categorical_values=categorical_values,
            binary_features=binary,
            causal_rules=list(config.get("causal_rules") or []),
        )

    def mutable_indices(self) -> List[int]:
        indices = []
        for idx, name in enumerate(self.feature_names):
            if name in self.immutable_features:
                continue
            if self.actionable_features is not None and name not in self.actionable_features:
                continue
            indices.append(idx)
        return indices

    def project(self, original: Any, candidate: Any) -> np.ndarray:
        original_arr = np.asarray(original, dtype=float).reshape(-1)
        projected = np.asarray(candidate, dtype=float).reshape(-1).copy()
        if len(projected) != len(self.feature_names):
            raise ValueError(
                f"Vetor contrafactual com {len(projected)} features; "
                f"esperadas {len(self.feature_names)}."
            )

        for idx, name in enumerate(self.feature_names):
            if name in self.immutable_features or (
                self.actionable_features is not None and name not in self.actionable_features
            ):
                projected[idx] = original_arr[idx]
                continue
            low, high = self.feature_ranges[name]
            projected[idx] = np.clip(projected[idx], low, high)
            if name in self.binary_features:
                projected[idx] = float(projected[idx] >= (low + high) / 2.0)
            allowed = self.categorical_values.get(name)
            if allowed:
                projected[idx] = min(allowed, key=lambda value: abs(value - projected[idx]))
        return projected

    def changed_mask(self, original: Any, candidate: Any) -> np.ndarray:
        original_arr = np.asarray(original, dtype=float).reshape(-1)
        candidate_arr = np.asarray(candidate, dtype=float).reshape(-1)
        scales = np.array(
            [max(abs(high - low), 1.0) for low, high in self.feature_ranges.values()],
            dtype=float,
        )
        return np.abs(candidate_arr - original_arr) > (EPS * scales)

    def validate(self, original: Any, candidate: Any) -> Dict[str, Any]:
        original_arr = np.asarray(original, dtype=float).reshape(-1)
        candidate_arr = np.asarray(candidate, dtype=float).reshape(-1)
        changed = self.changed_mask(original_arr, candidate_arr)
        violations: List[str] = []
        changed_names = [self.feature_names[i] for i in np.where(changed)[0]]

        for idx, name in enumerate(self.feature_names):
            value = candidate_arr[idx]
            low, high = self.feature_ranges[name]
            if not np.isfinite(value) or value < low - EPS or value > high + EPS:
                violations.append(f"{name}: valor {value!r} fora de [{low}, {high}]")
            if changed[idx] and name in self.immutable_features:
                violations.append(f"{name}: feature imutável/protegida")
            if changed[idx] and self.actionable_features is not None and name not in self.actionable_features:
                violations.append(f"{name}: feature não acionável")
            if name in self.binary_features and value not in (0, 1, 0.0, 1.0):
                violations.append(f"{name}: valor binário inválido")
            allowed = self.categorical_values.get(name)
            if allowed and not any(abs(value - item) <= EPS for item in allowed):
                violations.append(f"{name}: categoria não permitida")

        causal_violations = self._validate_causal_rules(original_arr, candidate_arr)
        violations.extend(causal_violations)
        actionable_changes = [
            name for name in changed_names
            if self.actionable_features is None or name in self.actionable_features
        ]
        actionability = (
            len(actionable_changes) / len(changed_names) if changed_names else 1.0
        )
        return {
            "plausible": not violations,
            "actionability": float(actionability),
            "causal_consistency": 1.0 if not causal_violations else 0.0,
            "violations": violations,
            "changed_features": changed_names,
        }

    def _validate_causal_rules(self, original: np.ndarray, candidate: np.ndarray) -> List[str]:
        violations: List[str] = []
        name_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}
        for rule in self.causal_rules:
            source = rule.get("if_feature")
            target = rule.get("then_feature")
            if source not in name_to_idx or target not in name_to_idx:
                continue
            source_delta = candidate[name_to_idx[source]] - original[name_to_idx[source]]
            target_delta = candidate[name_to_idx[target]] - original[name_to_idx[target]]
            direction = str(rule.get("direction", "same")).lower()
            tolerance = float(rule.get("tolerance", EPS))
            if abs(source_delta) <= tolerance:
                continue
            if direction == "same" and source_delta * target_delta < -tolerance:
                violations.append(f"regra causal violada: {source} e {target} devem variar no mesmo sentido")
            elif direction == "opposite" and source_delta * target_delta > tolerance:
                violations.append(f"regra causal violada: {source} e {target} devem variar em sentidos opostos")
            elif direction == "nondecreasing" and target_delta < -tolerance:
                violations.append(f"regra causal violada: {target} não pode diminuir quando {source} muda")
        return violations


class FeatureSpaceMismatchError(ValueError):
    """Erro controlado quando um vetor não pode ser aplicado a outro espaço."""


class FeatureSpaceAdapter:
    @staticmethod
    def adapt(
        vector: Any,
        source_names: Sequence[str],
        target_names: Sequence[str],
        *,
        reference_target: Optional[Any] = None,
        derive_fn=None,
    ) -> np.ndarray:
        source = np.asarray(vector, dtype=float).reshape(-1)
        source_names = [str(name) for name in source_names]
        target_names = [str(name) for name in target_names]
        if len(source) != len(source_names):
            raise FeatureSpaceMismatchError(
                f"Vetor tem {len(source)} valores mas o espaço de origem declara {len(source_names)}."
            )
        if source_names == target_names:
            return source.copy()
        if derive_fn is not None:
            derived = np.asarray(derive_fn(source.copy()), dtype=float).reshape(-1)
            if len(derived) != len(target_names):
                raise FeatureSpaceMismatchError(
                    "O adaptador ontológico devolveu dimensão incompatível: "
                    f"{len(derived)} em vez de {len(target_names)}."
                )
            return derived

        source_map = {name: source[idx] for idx, name in enumerate(source_names)}
        missing = [name for name in target_names if name not in source_map]
        if missing and reference_target is None:
            raise FeatureSpaceMismatchError(
                "O modelo de destino exige features ausentes no contrafactual: "
                + ", ".join(missing[:8])
                + ("..." if len(missing) > 8 else "")
                + ". Recalcule no espaço enriquecido ou forneça um adaptador ontológico."
            )
        if reference_target is not None:
            reference = np.asarray(reference_target, dtype=float).reshape(-1)
            if len(reference) != len(target_names):
                raise FeatureSpaceMismatchError("Linha de referência do destino tem dimensão inválida.")
            target = reference.copy()
        else:
            target = np.empty(len(target_names), dtype=float)
        for idx, name in enumerate(target_names):
            if name in source_map:
                target[idx] = source_map[name]
        return target


class OntologyCFValidator:
    """Validação conservadora: só declara o que a OWL e o mapeamento permitem."""

    def __init__(
        self,
        ontology: Any = None,
        feature_mapping: Optional[Mapping[str, Any]] = None,
        semantic_rules: Optional[Iterable[Any]] = None,
    ):
        self.ontology = ontology
        self.feature_mapping = dict(feature_mapping or {})
        self.semantic_rules = list(semantic_rules or [])
        self._entities = self._collect_entities(ontology)

    @staticmethod
    def _collect_entities(ontology: Any) -> Dict[str, Any]:
        if ontology is None:
            return {}
        entities: Dict[str, Any] = {}
        for accessor in ("classes", "data_properties", "object_properties"):
            fn = getattr(ontology, accessor, None)
            if not callable(fn):
                continue
            try:
                values = list(fn())
            except Exception:
                continue
            for entity in values:
                names = [getattr(entity, "name", "")]
                labels = getattr(entity, "label", None)
                if labels:
                    names.extend(str(label) for label in labels)
                for name in names:
                    if name:
                        entities[_normalise_feature_name(name)] = entity
        return entities

    def validate(
        self,
        original: Any,
        candidate: Any,
        feature_names: Sequence[str],
    ) -> Dict[str, Any]:
        original_arr = np.asarray(original, dtype=float).reshape(-1)
        candidate_arr = np.asarray(candidate, dtype=float).reshape(-1)
        changed = np.where(np.abs(candidate_arr - original_arr) > EPS)[0]
        if self.ontology is None:
            return {"applicable": False, "consistent": None, "score": None, "warnings": []}

        warnings: List[str] = []
        checked = 0
        valid = 0
        per_feature = []
        for idx in changed:
            name = str(feature_names[idx])
            mapped = self.feature_mapping.get(name)
            entity = mapped if mapped is not None and not isinstance(mapped, str) else None
            if entity is None:
                token = _normalise_feature_name(mapped or name)
                entity = self._entities.get(token)
            if entity is None:
                warnings.append(f"{name}: sem conceito/propriedade OWL mapeado")
                per_feature.append({"feature": name, "mapped": False, "consistent": None})
                continue
            checked += 1
            feature_ok = self._datatype_compatible(entity, candidate_arr[idx])
            if feature_ok:
                valid += 1
            else:
                warnings.append(f"{name}: valor incompatível com o range OWL")
            per_feature.append({"feature": name, "mapped": True, "consistent": feature_ok})

        for rule in self.semantic_rules:
            try:
                result = rule(original_arr, candidate_arr, list(feature_names)) if callable(rule) else True
            except Exception as exc:
                result = False
                warnings.append(f"regra semântica falhou: {exc}")
            checked += 1
            if result:
                valid += 1
            else:
                warnings.append("regra semântica explícita violada")

        score = (valid / checked) if checked else None
        return {
            "applicable": bool(checked),
            "consistent": (score == 1.0) if score is not None else None,
            "score": float(score) if score is not None else None,
            "warnings": warnings,
            "per_feature": per_feature,
        }

    @staticmethod
    def _datatype_compatible(entity: Any, value: float) -> bool:
        ranges = getattr(entity, "range", None) or []
        if not ranges:
            return np.isfinite(value)
        text = " ".join(str(item).lower() for item in ranges)
        if "bool" in text:
            return value in (0, 1, 0.0, 1.0)
        if "int" in text or "integer" in text:
            return np.isfinite(value) and abs(value - round(value)) <= EPS
        if any(token in text for token in ("float", "double", "decimal", "real")):
            return np.isfinite(value)
        return True


def _normalised_distance(
    first: Any,
    second: Any,
    constraints: CounterfactualConstraints,
) -> float:
    first_arr = np.asarray(first, dtype=float).reshape(-1)
    second_arr = np.asarray(second, dtype=float).reshape(-1)
    spans = np.array(
        [max(high - low, EPS) for low, high in constraints.feature_ranges.values()],
        dtype=float,
    )
    return float(np.mean(np.abs(first_arr - second_arr) / spans))


def evaluate_local_robustness(
    model: Any,
    counterfactual: Any,
    target_class: Any,
    constraints: CounterfactualConstraints,
    *,
    original: Optional[Any] = None,
    n_perturbations: int = 100,
    epsilon: float = 0.02,
    seed: int = 42,
    perturb_changed_only: bool = False,
) -> Dict[str, Any]:
    """Estima P(f(x' + delta) = y') com perturbações locais reproduzíveis."""
    cf = np.asarray(counterfactual, dtype=float).reshape(-1)
    base = cf if original is None else np.asarray(original, dtype=float).reshape(-1)
    mutable = constraints.mutable_indices()
    if perturb_changed_only and original is not None:
        changed = set(np.where(constraints.changed_mask(base, cf))[0])
        mutable = [idx for idx in mutable if idx in changed]
    if not mutable:
        valid = bool(_predict(model, cf)[0] == target_class)
        return {"score": float(valid), "valid": int(valid), "total": 1, "epsilon": epsilon}

    spans = np.array(
        [max(high - low, EPS) for low, high in constraints.feature_ranges.values()],
        dtype=float,
    )
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(max(1, int(n_perturbations))):
        perturbed = cf.copy()
        noise = rng.normal(0.0, epsilon, size=len(mutable))
        perturbed[mutable] += noise * spans[mutable]
        samples.append(constraints.project(base, perturbed))
    predictions = _predict(model, np.asarray(samples))
    count = int(np.sum(predictions == target_class))
    total = len(samples)
    return {
        "score": float(count / total) if total else 0.0,
        "valid": count,
        "total": total,
        "epsilon": float(epsilon),
    }


def rough_set_feature_scores(
    X: Any,
    y: Any,
    feature_names: Sequence[str],
    *,
    bins: int = 5,
) -> Dict[str, float]:
    """Dependência aproximada do rough set por atributo (regiões positivas)."""
    arr = _as_2d(X)
    labels = np.asarray(y).reshape(-1)
    scores: Dict[str, float] = {}
    for idx, name in enumerate(feature_names):
        column = arr[:, idx]
        unique = np.unique(column[np.isfinite(column)])
        if unique.size <= max(2, bins):
            discretised = column
        else:
            quantiles = np.unique(np.quantile(column, np.linspace(0, 1, bins + 1)))
            discretised = np.digitize(column, quantiles[1:-1], right=True)
        positive = 0
        for value in np.unique(discretised):
            mask = discretised == value
            if mask.any() and np.unique(labels[mask]).size == 1:
                positive += int(np.sum(mask))
        scores[str(name)] = float(positive / len(labels)) if len(labels) else 0.0
    return scores


def apply_rst_filter(
    candidates: List[Dict[str, Any]],
    X: Any,
    y: Any,
    feature_names: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Remove inválidos/implausíveis e anexa relevância RST às mudanças."""
    dependency = rough_set_feature_scores(X, y, feature_names)
    frequency = {str(name): 0 for name in feature_names}
    eligible = []
    for candidate in candidates:
        metrics = candidate.get("metrics", {})
        if not metrics.get("validity") or not metrics.get("plausibility"):
            continue
        eligible.append(candidate)
        for change in candidate.get("changes", []):
            frequency[change["feature"]] = frequency.get(change["feature"], 0) + 1
    denom = max(len(eligible), 1)
    scores = {
        name: float(dependency.get(name, 0.0) * (frequency.get(name, 0) / denom))
        for name in feature_names
    }
    for candidate in eligible:
        changed = [item["feature"] for item in candidate.get("changes", [])]
        candidate.setdefault("metrics", {})["rst_relevance"] = (
            float(np.mean([scores[name] for name in changed])) if changed else 0.0
        )
    return eligible, dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


class CounterfactualEngine:
    """Coordena geradores, validações e métricas com uma saída estável."""

    SUPPORTED_METHODS = (
        "AUTO", "DICE", "CLEAR", "COGS", "LORE-LOCAL", "LORE-GLOBAL"
    )
    # Rótulos honestos: nenhum destes geradores é uma réplica verificada do método da literatura.
    METHOD_LABELS = {
        "DICE": "DiCE (dice-ml opcional; fallback interno = DiCE-inspired)",
        "CLEAR": "CLEAR-inspired (regressão local; legado)",
        "COGS": "CoGS-inspired (pesquisa genética; legado)",
        "LORE-LOCAL": "LORE-inspired (substituto local TREPAN; legado)",
        "LORE-GLOBAL": "Tree-path (regras globais; não é LORE)",
    }

    def __init__(
        self,
        oracle: Any,
        X_reference: Any,
        feature_names: Sequence[str],
        *,
        y_reference: Optional[Any] = None,
        constraints: Optional[CounterfactualConstraints] = None,
        ontology_validator: Optional[OntologyCFValidator] = None,
        global_tree: Any = None,
        seed: int = 42,
    ):
        self.oracle = oracle
        self.X = _as_2d(X_reference)
        self.feature_names = [str(name) for name in feature_names]
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError("Nomes de features não coincidem com a matriz de referência.")
        expected_features = model_n_features(oracle)
        if expected_features is not None and self.X.shape[1] != expected_features:
            raise FeatureSpaceMismatchError(
                f"O oráculo espera {expected_features} features, mas o espaço "
                f"contrafactual contém {self.X.shape[1]}."
            )
        self.y = (
            np.asarray(y_reference).reshape(-1)
            if y_reference is not None
            else _predict(oracle, self.X)
        )
        self.constraints = constraints or CounterfactualConstraints.from_data(
            self.X, self.feature_names
        )
        self.ontology_validator = ontology_validator or OntologyCFValidator()
        self.global_tree = global_tree
        self.seed = int(seed)
        self.global_fidelity = None
        if global_tree is not None:
            try:
                self.global_fidelity = float(
                    np.mean(_predict(global_tree, self.X) == _predict(oracle, self.X))
                )
            except Exception:
                # A árvore pode pertencer a outro espaço; nesse caso a métrica
                # fica explicitamente N/A em vez de projectar colunas às cegas.
                self.global_fidelity = None

    def generate(
        self,
        instance: Any,
        *,
        desired_class: Any = None,
        method: str = "AUTO",
        model_type: str = "mlp",
        total_cfs: int = 5,
        apply_rst: bool = False,
        robustness_samples: int = 100,
        robustness_epsilon: float = 0.02,
    ) -> Dict[str, Any]:
        instance_arr = np.asarray(instance, dtype=float).reshape(-1)
        if len(instance_arr) != len(self.feature_names):
            raise ValueError("A instância não pertence ao espaço de features do oráculo.")
        factual = _predict(self.oracle, instance_arr)[0]
        desired = self._resolve_desired_class(instance_arr, factual, desired_class)
        method_key = str(method or "AUTO").upper().replace("_", "-")
        if method_key not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Método contrafactual desconhecido: {method}. "
                f"Opções: {', '.join(self.SUPPORTED_METHODS)}"
            )

        methods = self._automatic_methods(model_type) if method_key == "AUTO" else [method_key]
        raw_candidates: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for generator in methods:
            try:
                raw_candidates.extend(
                    self._run_generator(generator, instance_arr, factual, desired, total_cfs)
                )
            except Exception as exc:
                warnings.append(f"{generator}: {exc}")

        candidates = self._deduplicate(raw_candidates, instance_arr)
        evaluated = [
            self._evaluate_candidate(
                instance_arr,
                item,
                factual,
                desired,
                robustness_samples=robustness_samples,
                robustness_epsilon=robustness_epsilon,
            )
            for item in candidates
        ]
        if apply_rst:
            evaluated, rst_scores = apply_rst_filter(
                evaluated, self.X, self.y, self.feature_names
            )
        else:
            rst_scores = {}

        evaluated.sort(
            key=lambda item: (
                not item["metrics"]["validity"],
                not item["metrics"]["plausibility"],
                item["metrics"]["sparsity"],
                item["metrics"]["proximity"],
                -item["metrics"]["robustness"],
            )
        )
        # REGRA OBRIGATÓRIA: só se devolve como sucesso o que foi re-validado no modelo E cumpre as restrições de domínio.
        validated = [e for e in evaluated if e["metrics"]["validity"] and e["metrics"]["plausibility"]]
        rejected = [
            {"vector": e["vector"], "validity": e["metrics"]["validity"], "plausibility": e["metrics"]["plausibility"],
             "violations": (e.get("domain_validation") or {}).get("violations", [])}
            for e in evaluated if e not in validated
        ]
        evaluated = validated[: max(1, int(total_cfs))]
        best = evaluated[0] if evaluated else None
        aggregate = self._aggregate_metrics(evaluated)
        return {
            "status": "success" if best is not None else "no_counterfactual_found",
            "rejected_candidates": rejected,
            "method_labels": {m: self.METHOD_LABELS.get(m, m) for m in methods},
            "canonical": False,
            "causality": "NOT_CLAIMED",
            "method": method_key,
            "methods_executed": methods,
            "factual_prediction": self._scalar(factual),
            "desired_class": self._scalar(desired),
            "original_instance": instance_arr.tolist(),
            "feature_names": list(self.feature_names),
            "candidates": evaluated,
            "best_candidate": best,
            "aggregate_metrics": aggregate,
            "rst_feature_scores": rst_scores,
            "narrative": self._narrative(best, aggregate, factual, desired),
            "warnings": warnings,
        }

    def _automatic_methods(self, model_type: str) -> List[str]:
        if "tree" in str(model_type).lower() or "trepan" in str(model_type).lower():
            methods = ["LORE-LOCAL"]
            if self.global_tree is not None:
                methods.append("LORE-GLOBAL")
            methods.append("COGS")
            return methods
        return ["DICE", "CLEAR", "COGS"]

    def _resolve_desired_class(self, instance: np.ndarray, factual: Any, desired: Any) -> Any:
        if desired is not None and str(desired).lower() not in {"opposite", "any", "none"}:
            return desired
        probabilities = _predict_proba(self.oracle, instance)
        classes = getattr(self.oracle, "classes_", None)
        if classes is None:
            classes = getattr(getattr(self.oracle, "model", None), "classes_", None)
        if probabilities is not None:
            order = np.argsort(probabilities[0])[::-1]
            for index in order:
                label = classes[index] if classes is not None else index
                if label != factual:
                    return label
        for label in np.unique(self.y):
            if label != factual:
                return label
        raise ValueError("Não foi possível determinar uma classe contrafactual alternativa.")

    def _run_generator(
        self, method: str, instance: np.ndarray, factual: Any, desired: Any, total: int
    ) -> List[Dict[str, Any]]:
        if method == "DICE":
            return self._generate_dice(instance, factual, desired, total)
        if method == "CLEAR":
            return self._generate_clear(instance, factual, desired, total)
        if method == "COGS":
            return self._generate_cogs(instance, factual, desired, total)
        if method == "LORE-LOCAL":
            return self._generate_lore_local(instance, factual, desired, total)
        if method == "LORE-GLOBAL":
            if self.global_tree is None:
                raise ValueError("LORE-Global exige uma árvore substituta global.")
            return self._tree_counterfactuals(
                self.global_tree, instance, desired, total, "LORE-Global"
            )
        raise ValueError(method)

    def _generate_dice(
        self, instance: np.ndarray, factual: Any, desired: Any, total: int
    ) -> List[Dict[str, Any]]:
        try:
            import pandas as pd
            import dice_ml

            model = getattr(self.oracle, "model", self.oracle)
            if not hasattr(model, "predict_proba"):
                raise TypeError("modelo não compatível com o backend sklearn do DiCE")
            frame = pd.DataFrame(self.X, columns=self.feature_names)
            frame["__target__"] = self.y
            continuous = [
                name for name in self.feature_names
                if name not in self.constraints.binary_features
                and name not in self.constraints.categorical_values
            ]
            data = dice_ml.Data(
                dataframe=frame,
                continuous_features=continuous,
                outcome_name="__target__",
            )
            wrapped = dice_ml.Model(model=model, backend="sklearn")
            explainer = dice_ml.Dice(data, wrapped, method="random")
            query = pd.DataFrame([instance], columns=self.feature_names)
            features_to_vary = [self.feature_names[i] for i in self.constraints.mutable_indices()]
            permitted_range = {
                name: list(bounds) for name, bounds in self.constraints.feature_ranges.items()
            }
            result = explainer.generate_counterfactuals(
                query,
                total_CFs=max(1, int(total)),
                desired_class=desired,
                features_to_vary=features_to_vary,
                permitted_range=permitted_range,
            )
            final = result.cf_examples_list[0].final_cfs_df
            candidates = []
            if final is not None:
                for _, row in final.iterrows():
                    vector = row[self.feature_names].to_numpy(dtype=float)
                    candidates.append({"cf": vector, "method": "DiCE", "metadata": {}})
            if candidates:
                return candidates
        except Exception:
            pass
        return self._stochastic_search(
            instance, factual, desired, max(total * 4, total), "DiCE-Fallback", seed_offset=11
        )

    def _generate_clear(
        self, instance: np.ndarray, factual: Any, desired: Any, total: int
    ) -> List[Dict[str, Any]]:
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(self.seed + 23)
        spans = np.array(
            [max(high - low, EPS) for low, high in self.constraints.feature_ranges.values()]
        )
        neighbourhood = []
        for _ in range(max(300, total * 80)):
            candidate = instance + rng.normal(0, 0.15, len(instance)) * spans
            neighbourhood.append(self.constraints.project(instance, candidate))
        neighbourhood_arr = np.asarray(neighbourhood)
        labels = _predict(self.oracle, neighbourhood_arr)
        if np.unique(labels).size < 2:
            return self._stochastic_search(
                instance, factual, desired, max(total * 3, total), "CLEAR-Fallback", seed_offset=29
            )
        surrogate = LogisticRegression(max_iter=1000, random_state=self.seed)
        surrogate.fit(neighbourhood_arr, labels)
        surrogate_predictions = _predict(surrogate, neighbourhood_arr)
        local_fidelity = float(np.mean(surrogate_predictions == labels))
        candidates = []
        order = np.argsort([_normalised_distance(instance, row, self.constraints) for row in neighbourhood_arr])
        for idx in order:
            row = neighbourhood_arr[idx]
            if _predict(surrogate, row)[0] != desired:
                continue
            projected = self.constraints.project(instance, row)
            if _predict(self.oracle, projected)[0] == desired:
                candidates.append({
                    "cf": projected,
                    "method": "CLEAR",
                    "metadata": {"local_fidelity": local_fidelity},
                })
            if len(candidates) >= total * 3:
                break
        if not candidates:
            return self._stochastic_search(
                instance, factual, desired, max(total * 3, total), "CLEAR-Fallback", seed_offset=31
            )
        return candidates

    def _generate_cogs(
        self, instance: np.ndarray, factual: Any, desired: Any, total: int
    ) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(self.seed + 37)
        mutable = self.constraints.mutable_indices()
        if not mutable:
            return []
        spans = np.array(
            [max(high - low, EPS) for low, high in self.constraints.feature_ranges.values()]
        )
        population = []
        unlike = self.X[_predict(self.oracle, self.X) == desired]
        for idx in range(60):
            if len(unlike) and idx < 20:
                donor = unlike[rng.randint(len(unlike))]
                candidate = instance.copy()
                chosen = rng.choice(mutable, size=rng.randint(1, min(4, len(mutable)) + 1), replace=False)
                candidate[chosen] = donor[chosen]
            else:
                candidate = instance + rng.normal(0, 0.25, len(instance)) * spans
            population.append(self.constraints.project(instance, candidate))

        def fitness(vector: np.ndarray) -> float:
            prediction = _predict(self.oracle, vector)[0]
            validity = 4.0 if prediction == desired else 0.0
            proximity = _normalised_distance(instance, vector, self.constraints)
            sparsity = float(np.mean(self.constraints.changed_mask(instance, vector)))
            proba = _predict_proba(self.oracle, vector)
            confidence = 0.0
            if proba is not None:
                classes = getattr(self.oracle, "classes_", None)
                if classes is None:
                    classes = getattr(getattr(self.oracle, "model", None), "classes_", None)
                if classes is not None and desired in list(classes):
                    confidence = float(proba[0, list(classes).index(desired)])
                elif isinstance(desired, (int, np.integer)) and int(desired) < proba.shape[1]:
                    confidence = float(proba[0, int(desired)])
            return validity + confidence - 1.5 * proximity - 0.6 * sparsity

        for _ in range(35):
            population.sort(key=fitness, reverse=True)
            elite = population[:15]
            children = [item.copy() for item in elite]
            while len(children) < 60:
                first, second = elite[rng.randint(len(elite))], elite[rng.randint(len(elite))]
                mask = rng.rand(len(instance)) < 0.5
                child = np.where(mask, first, second)
                n_mutations = rng.randint(1, min(3, len(mutable)) + 1)
                for feature_idx in rng.choice(mutable, size=n_mutations, replace=False):
                    if rng.rand() < 0.35 and len(unlike):
                        child[feature_idx] = unlike[rng.randint(len(unlike)), feature_idx]
                    else:
                        child[feature_idx] += rng.normal(0, 0.15) * spans[feature_idx]
                children.append(self.constraints.project(instance, child))
            population = children
        population.sort(key=fitness, reverse=True)
        return [
            {"cf": vector, "method": "CoGS", "metadata": {"fitness": float(fitness(vector))}}
            for vector in population
            if _predict(self.oracle, vector)[0] == desired
        ][: max(total * 3, total)]

    def _generate_lore_local(
        self, instance: np.ndarray, factual: Any, desired: Any, total: int
    ) -> List[Dict[str, Any]]:
        """LORE local usando o mesmo núcleo TREPAN histórico da produção."""
        from core.trepan_original import TrepanOriginalClassifier

        rng = np.random.RandomState(self.seed + 43)
        spans = np.array(
            [max(high - low, EPS) for low, high in self.constraints.feature_ranges.values()]
        )
        neighbourhood = [instance.copy()]
        for _ in range(max(400, total * 100)):
            candidate = instance + rng.normal(0, 0.20, len(instance)) * spans
            neighbourhood.append(self.constraints.project(instance, candidate))
        neighbourhood_arr = np.asarray(neighbourhood)
        labels = _predict(self.oracle, neighbourhood_arr)
        if np.unique(labels).size < 2:
            return self._stochastic_search(
                instance, factual, desired, max(total * 3, total),
                'LORE-Local-Fallback', seed_offset=47
            )

        min_sample = max(len(neighbourhood_arr), min(500, max(120, len(neighbourhood_arr) * 2)))
        local_tree = TrepanOriginalClassifier(
            max_nodes=31, max_depth=5, min_samples_leaf=3,
            min_sample=min_sample, max_n=3, beam_width=2,
            max_features_per_node=min(12, neighbourhood_arr.shape[1]),
            max_queries=max(600, min_sample * 2), random_state=self.seed,
        ).fit(neighbourhood_arr, oracle=self.oracle, feature_names=self.feature_names)
        fidelity = float(np.mean(local_tree.predict(neighbourhood_arr) == labels))
        candidates = self._tree_counterfactuals(
            local_tree, instance, desired, total * 3, 'LORE-Local'
        )
        for candidate in candidates:
            candidate.setdefault('metadata', {})['local_fidelity'] = fidelity
            candidate['metadata']['local_tree_family'] = 'historical_trepan'
        return candidates

    def _tree_counterfactuals(
        self,
        tree_model: Any,
        instance: np.ndarray,
        desired: Any,
        total: int,
        method: str,
    ) -> List[Dict[str, Any]]:
        """Gera CFs a partir das regras do modelo alvo sem assumir sklearn.tree.

        Suporta TREPAN histórico/reloaded e modelos compatíveis com o extractor
        global de regras. Nenhuma árvore auxiliar de outra família é criada.
        """
        from counterfactuals.global_rules import extract_global_rules

        rules = extract_global_rules(tree_model, self.feature_names)
        desired_rules = [r for r in rules if r.predicted_class == desired]
        if not desired_rules:
            return []

        spans = np.asarray([
            max(self.constraints.feature_ranges[name][1] - self.constraints.feature_ranges[name][0], EPS)
            for name in self.feature_names
        ], dtype=float)

        def set_literal(vector, feature, op, threshold, truth=True):
            feature = int(feature)
            threshold = float(threshold)
            value = float(vector[feature])
            if op == '<=':
                satisfied = value <= threshold
                if satisfied != truth:
                    vector[feature] = np.nextafter(threshold, -np.inf if truth else np.inf)
            elif op == '>':
                satisfied = value > threshold
                if satisfied != truth:
                    vector[feature] = np.nextafter(threshold, np.inf if truth else -np.inf)
            return vector

        def literal_cost(vector, literal, target_truth):
            feature, _name, op, threshold = literal
            feature = int(feature)
            threshold = float(threshold)
            value = float(vector[feature])
            if op == '<=':
                target = np.nextafter(threshold, -np.inf if target_truth else np.inf)
            else:
                target = np.nextafter(threshold, np.inf if target_truth else -np.inf)
            return abs(target - value) / max(spans[feature], EPS)

        def satisfy_condition(vector, condition):
            if condition.kind != 'm_of_n':
                return set_literal(vector, condition.feature_index, condition.operator, condition.threshold, True)
            literals = list(condition.literals)
            truths = []
            for feature, _name, op, threshold in literals:
                value = float(vector[int(feature)])
                truths.append(value > threshold if op == '>' else value <= threshold)
            votes = int(sum(truths))
            if not condition.negated:
                need = max(0, int(condition.m) - votes)
                choices = [i for i, ok in enumerate(truths) if not ok]
                choices.sort(key=lambda i: literal_cost(vector, literals[i], True))
                for i in choices[:need]:
                    feature, _name, op, threshold = literals[i]
                    set_literal(vector, feature, op, threshold, True)
            else:
                need = max(0, votes - (int(condition.m) - 1))
                choices = [i for i, ok in enumerate(truths) if ok]
                choices.sort(key=lambda i: literal_cost(vector, literals[i], False))
                for i in choices[:need]:
                    feature, _name, op, threshold = literals[i]
                    set_literal(vector, feature, op, threshold, False)
            return vector

        candidates = []
        for rule in desired_rules:
            candidate = np.asarray(instance, dtype=float).copy()
            for condition in rule.conditions:
                candidate = satisfy_condition(candidate, condition)
            candidate = self.constraints.project(instance, candidate)
            if _predict(self.oracle, candidate)[0] != desired:
                continue
            candidates.append({
                'cf': candidate,
                'method': method,
                'metadata': {
                    'coverage': float(rule.support),
                    'confidence': float(rule.confidence),
                    'rule': rule.text(),
                    'tree_family_preserved': True,
                },
            })
        candidates.sort(key=lambda item: _normalised_distance(instance, item['cf'], self.constraints))
        return candidates[: max(total, 1)]

    def _stochastic_search(
        self,
        instance: np.ndarray,
        factual: Any,
        desired: Any,
        total: int,
        method: str,
        *,
        seed_offset: int,
    ) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(self.seed + seed_offset)
        mutable = self.constraints.mutable_indices()
        if not mutable:
            return []
        spans = np.array(
            [max(high - low, EPS) for low, high in self.constraints.feature_ranges.values()]
        )
        predictions = _predict(self.oracle, self.X)
        unlike = self.X[predictions == desired]
        candidates = []
        trials = max(500, total * 120)
        for trial in range(trials):
            candidate = instance.copy()
            max_changes = min(max(1, len(mutable)), 6)
            n_changes = 1 + (trial % max_changes)
            chosen = rng.choice(mutable, size=n_changes, replace=False)
            if len(unlike) and trial % 2 == 0:
                donor = unlike[rng.randint(len(unlike))]
                candidate[chosen] = donor[chosen]
            else:
                candidate[chosen] += rng.normal(0, 0.30, n_changes) * spans[chosen]
            candidate = self.constraints.project(instance, candidate)
            if _predict(self.oracle, candidate)[0] == desired:
                candidates.append({"cf": candidate, "method": method, "metadata": {}})
                if len(candidates) >= total * 8:
                    break
        candidates.sort(key=lambda item: (
            int(np.sum(self.constraints.changed_mask(instance, item["cf"]))),
            _normalised_distance(instance, item["cf"], self.constraints),
        ))
        if method.startswith("DiCE"):
            candidates = self._select_diverse(candidates, instance, total * 3)
        return candidates[: max(total * 3, total)]

    def _select_diverse(
        self, candidates: List[Dict[str, Any]], instance: np.ndarray, limit: int
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        selected = [candidates[0]]
        remaining = candidates[1:]
        while remaining and len(selected) < limit:
            def score(item):
                diversity = min(
                    _normalised_distance(item["cf"], chosen["cf"], self.constraints)
                    for chosen in selected
                )
                proximity = _normalised_distance(instance, item["cf"], self.constraints)
                return diversity - 0.2 * proximity
            best_index = max(range(len(remaining)), key=lambda idx: score(remaining[idx]))
            selected.append(remaining.pop(best_index))
        return selected

    def _deduplicate(
        self, candidates: List[Dict[str, Any]], original: np.ndarray,
    ) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for item in candidates:
            vector = self.constraints.project(original, item["cf"])
            key = tuple(np.round(vector, 10))
            if key in seen:
                continue
            seen.add(key)
            cloned = dict(item)
            cloned["cf"] = vector
            unique.append(cloned)
        return unique

    def _evaluate_candidate(
        self,
        original: np.ndarray,
        item: Dict[str, Any],
        factual: Any,
        desired: Any,
        *,
        robustness_samples: int,
        robustness_epsilon: float,
    ) -> Dict[str, Any]:
        candidate = self.constraints.project(original, item["cf"])
        prediction = _predict(self.oracle, candidate)[0]
        domain = self.constraints.validate(original, candidate)
        ontology = self.ontology_validator.validate(original, candidate, self.feature_names)
        changed_mask = self.constraints.changed_mask(original, candidate)
        changes = []
        for idx in np.where(changed_mask)[0]:
            changes.append({
                "feature": self.feature_names[idx],
                "original": float(original[idx]),
                "counterfactual": float(candidate[idx]),
                "delta": float(candidate[idx] - original[idx]),
                "actionable": (
                    self.constraints.actionable_features is None
                    or self.feature_names[idx] in self.constraints.actionable_features
                ),
            })
        robustness = evaluate_local_robustness(
            self.oracle,
            candidate,
            desired,
            self.constraints,
            original=original,
            n_perturbations=robustness_samples,
            epsilon=robustness_epsilon,
            seed=self.seed + len(changes),
        )
        metadata = dict(item.get("metadata") or {})
        metrics = {
            "validity": bool(prediction == desired and prediction != factual),
            "proximity": _normalised_distance(original, candidate, self.constraints),
            "sparsity": int(np.sum(changed_mask)),
            "plausibility": bool(domain["plausible"]),
            "actionability": float(domain["actionability"]),
            "causal_consistency": float(domain["causal_consistency"]),
            "ontology_consistency": ontology.get("score"),
            "robustness": float(robustness["score"]),
            "coverage": metadata.get("coverage"),
            "local_fidelity": metadata.get("local_fidelity"),
            "global_fidelity": metadata.get("global_fidelity", self.global_fidelity),
        }
        return {
            "method": item.get("method", "unknown"),
            "prediction": self._scalar(prediction),
            "vector": candidate.tolist(),
            "changes": changes,
            "rule": metadata.get("rule") or self._change_rule(changes, prediction),
            "metrics": metrics,
            "domain_validation": domain,
            "ontology_validation": ontology,
            "robustness_details": robustness,
            "metadata": metadata,
        }

    def _aggregate_metrics(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candidates:
            return {
                "validity": 0.0, "proximity": None, "sparsity": None,
                "diversity": None, "stability": None, "robustness": None,
                "plausibility": 0.0, "actionability": None,
                "ontology_consistency": None, "causal_consistency": None,
                "global_fidelity": None,
            }
        metrics = [item["metrics"] for item in candidates]
        vectors = [np.asarray(item["vector"], dtype=float) for item in candidates]
        masks = [set(change["feature"] for change in item["changes"]) for item in candidates]
        pairwise_distance = []
        pairwise_jaccard = []
        for first, second in combinations(range(len(candidates)), 2):
            pairwise_distance.append(
                _normalised_distance(vectors[first], vectors[second], self.constraints)
            )
            union = masks[first] | masks[second]
            pairwise_jaccard.append(
                len(masks[first] & masks[second]) / len(union) if union else 1.0
            )

        def mean_value(key: str) -> Optional[float]:
            values = [row.get(key) for row in metrics if row.get(key) is not None]
            return float(np.mean(values)) if values else None

        return {
            "validity": mean_value("validity"),
            "proximity": mean_value("proximity"),
            "sparsity": mean_value("sparsity"),
            "diversity": float(np.mean(pairwise_distance)) if pairwise_distance else 0.0,
            "stability": float(np.mean(pairwise_jaccard)) if pairwise_jaccard else 1.0,
            "robustness": mean_value("robustness"),
            "plausibility": mean_value("plausibility"),
            "actionability": mean_value("actionability"),
            "ontology_consistency": mean_value("ontology_consistency"),
            "causal_consistency": mean_value("causal_consistency"),
            "coverage": mean_value("coverage"),
            "local_fidelity": mean_value("local_fidelity"),
            "global_fidelity": mean_value("global_fidelity"),
        }

    @staticmethod
    def _change_rule(changes: List[Dict[str, Any]], prediction: Any) -> str:
        if not changes:
            return f"sem alterações → classe {prediction}"
        clauses = [f"{row['feature']} = {row['counterfactual']:.6g}" for row in changes]
        return "se " + " e ".join(clauses) + f", então classe {prediction}"

    @staticmethod
    def _scalar(value: Any) -> Any:
        return value.item() if isinstance(value, np.generic) else value

    @staticmethod
    def _narrative(
        best: Optional[Dict[str, Any]],
        aggregate: Dict[str, Any],
        factual: Any,
        desired: Any,
    ) -> str:
        if best is None:
            return (
                f"Não foi encontrado um contrafactual válido para mudar a classe "
                f"{factual} para {desired} dentro das restrições definidas."
            )
        metrics = best["metrics"]
        changes = ", ".join(change["feature"] for change in best["changes"])
        return (
            f"A previsão factual é {factual}. O melhor contrafactual muda para {desired} "
            f"alterando {metrics['sparsity']} feature(s): {changes or 'nenhuma'}. "
            f"Distância normalizada={metrics['proximity']:.4f}; "
            f"robustez local={metrics['robustness']:.1%}; "
            f"plausível={'sim' if metrics['plausibility'] else 'não'}."
        )
