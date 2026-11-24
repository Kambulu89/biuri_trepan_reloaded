"""Protocolo seguro de melhoria de árvores substitutas com contrafactuais.

Adapta o método descrito na tese de Eldis Dennys González Pérez ao protocolo
científico do BIURI. Os contrafactuais são gerados apenas no subtreino de
desenvolvimento, ponderados por confiança/plausibilidade e avaliados numa
validação interna. O teste bloqueado nunca participa nesta operação.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from core.evaluation_protocol import EvaluationProtocolGuard, PartitionRole


EPS = 1e-12


@dataclass(frozen=True)
class CandidateParameters:
    """Parâmetros que variam durante a busca na validação interna."""

    max_cfs: int = 15
    cf_types: str = "A+B+C"
    cf_weight_ratio: float = 0.05
    # Tese (González Pérez, 2026): o fator de fronteira é uma dimensão da grelha. Por omissão, sempre ativo.
    use_frontier_weight: bool = True


PROFILE_BIURI = "biuri"
PROFILE_THESIS = "tese"
SELECTION_HOLDOUT = "holdout"
SELECTION_CV2 = "cv2"
OBJECTIVE_MULTI = "multiobjective"
OBJECTIVE_FIDELITY = "fidelity"


@dataclass(frozen=True)
class ImprovementSearchConfig:
    """Configuração auditável do experimento de melhoria.

    Os valores por omissão são o protocolo BIURI. ``thesis_profile()`` devolve o método da tese de Eldis Dennys
    González Pérez (2026) com as duas correções de fuga de informação descritas em ``THESIS_PROFILE_NOTES``.
    """

    profile: str = PROFILE_BIURI
    validation_fraction: float = 0.25
    acceptance_fraction: float = 0.20
    random_state: int = 42
    counterfactual_method: str = "COGS"
    origin_fraction: float = 0.33
    max_origins: int = 24
    cfs_per_origin: int = 3
    confidence_threshold: float = 0.60
    density_percentile: float = 95.0
    strict_density_gate: bool = True
    max_cf_ratio: float = 0.20
    anchor_ratio: float = 0.20
    max_cf_weight: float = 20.0
    sample_size: int = 1200
    max_cfs_grid: Tuple[int, ...] = (8, 15, 30)
    cf_types_grid: Tuple[str, ...] = ("A+B", "A+B+C")
    cf_weight_ratio_grid: Tuple[float, ...] = (0.03, 0.05)
    fidelity_tolerance: float = 0.01
    real_metric_tolerance: float = 0.005
    minority_recall_tolerance: float = 0.02
    minimum_real_gain: float = 0.001
    attribution_margin: float = 0.001
    max_complexity_ratio: float = 1.75
    # --- opções do método da tese (os valores por omissão reproduzem exatamente o protocolo BIURI)
    use_frontier_weight_grid: Tuple[bool, ...] = (True,)
    divergent_boost: str = "fixed"          # "fixed": peso 2,0 | "adaptive": 1 + min(3, 10 / n_B) (tese)
    divergent_fixed_weight: float = 2.0
    gap_weight: float = 0.30                # peso base dos CFs de brecha (C)
    min_anchors: int = 1                    # tese: 50 (limitado ao nº de linhas)
    min_cf_weight: float = 0.0              # tese: 0,1
    adjust_ratio_by_oracle_accuracy: bool = False   # tese: ratio · acc/0,9 (≤ 0,10); aqui acc só do treino
    selection_protocol: str = SELECTION_HOLDOUT     # tese: "cv2" (validação cruzada estratificada, 2 partições)
    selection_objective: str = OBJECTIVE_MULTI      # tese: "fidelity"
    # Orçamento que escala os pesos dos CFs (desired_total = ratio · orçamento). "sample_size": o mesmo da amostra
    # sintética do extrator (BIURI). "thesis": min(5000, max(3000, 5·n)), independente do extrator (tese).
    cf_weight_budget_rule: str = "sample_size"

    @classmethod
    def thesis_profile(cls, **overrides: Any) -> "ImprovementSearchConfig":
        """Método da tese, Cap. 2 e ``pipeline_improve.py``: grelha max_cfs {5, 10} × tipos {A+B, A+B+C} ×
        fronteira {não, sim}, seleção por validação cruzada de 2 partições a maximizar a fidelidade, reforço
        adaptativo dos divergentes, peso C 0,3, ≥ 50 âncoras, peso mínimo 0,1, orçamento dos pesos
        min(5000, max(3000, 5·n)), amostra sintética do extrator 2000, filtro de confiança 0,6 e de densidade P95 (rejeição, passo 5.4).
        O gate de aceitação do BIURI mantém-se: a tese não o tem, mas sem ele uma árvore pior seria implantada."""
        base = dict(
            profile=PROFILE_THESIS,
            max_origins=0,                      # tese: 33 % das instâncias de treino, sem teto
            max_cfs_grid=(5, 10),
            cf_types_grid=("A+B", "A+B+C"),
            cf_weight_ratio_grid=(0.05,),
            use_frontier_weight_grid=(False, True),
            divergent_boost="adaptive",
            gap_weight=0.30,
            min_anchors=50,
            min_cf_weight=0.1,
            adjust_ratio_by_oracle_accuracy=True,
            selection_protocol=SELECTION_CV2,
            selection_objective=OBJECTIVE_FIDELITY,
            confidence_threshold=0.60,
            density_percentile=95.0,
            strict_density_gate=True,
            sample_size=2000,
            cf_weight_budget_rule="thesis",
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def from_options(cls, options: Optional[Mapping[str, Any]]) -> "ImprovementSearchConfig":
        values = dict(options or {})
        allowed = set(cls.__dataclass_fields__)
        clean = {key: value for key, value in values.items() if key in allowed}
        for key in ("max_cfs_grid", "cf_types_grid", "cf_weight_ratio_grid", "use_frontier_weight_grid"):
            if key in clean and not isinstance(clean[key], tuple):
                clean[key] = tuple(clean[key])
        if str(clean.get("profile", PROFILE_BIURI)).lower() == PROFILE_THESIS:
            clean.pop("profile", None)
            return cls.thesis_profile(**clean)
        return cls(**clean)


THESIS_PROFILE_NOTES = (
    "Correções obrigatórias face ao código da tese: (1) os CFs de cada partição da validação cruzada são gerados só a "
    "partir das linhas de ajuste dessa partição (na tese eram gerados em todo o treino e reutilizados nas validações); "
    "(2) a accuracy do oráculo usada para ajustar cf_weight_ratio é medida no treino (na tese vinha de uma tabela "
    "por dataset com a accuracy no teste). Configurações por dataset (MLP, CLEAR) não são importadas: o BIURI é "
    "agnóstico ao dataset."
)


def _effective_sample_size(config: "ImprovementSearchConfig", n_rows: int) -> int:
    """Orçamento que escala os pesos dos CFs: a amostra sintética (BIURI) ou min(5000, max(3000, 5·n)) (tese)."""
    if str(config.cf_weight_budget_rule).lower() == "thesis":
        return int(min(5000, max(3000, 5 * int(n_rows))))
    return int(config.sample_size)


@dataclass
class AugmentationBatch:
    X: Optional[np.ndarray]
    y: Optional[np.ndarray]
    weights: Optional[np.ndarray]
    stats: Dict[str, Any]


@dataclass
class TrainingRequest:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    extra_X: Optional[np.ndarray]
    extra_y: Optional[np.ndarray]
    extra_weights: Optional[np.ndarray]
    parameters: Optional[CandidateParameters]
    seed: int
    final_refit: bool = False


CandidateBuilder = Callable[[TrainingRequest], Any]


def _as_2d(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} deve ser uma matriz bidimensional.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contém NaN ou infinito.")
    return arr


def _predict_confidence(model: Any, matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        return np.ones(len(matrix), dtype=float)
    probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
    if probabilities.ndim != 2 or len(probabilities) != len(matrix):
        raise ValueError("predict_proba devolveu uma forma incompatível.")
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = getattr(getattr(model, "model", None), "classes_", None)
    if classes is None:
        classes = np.arange(probabilities.shape[1])
    mapping = {value: index for index, value in enumerate(np.asarray(classes).tolist())}
    output = []
    for row, label in zip(probabilities, labels):
        if label not in mapping:
            raise ValueError(f"Classe {label!r} ausente em predict_proba/classes_.")
        output.append(float(row[mapping[label]]))
    return np.asarray(output, dtype=float)


def _stratified_indices(labels: np.ndarray, fraction: float, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    selected: List[int] = []
    labels = np.asarray(labels)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        requested = max(1, int(np.ceil(float(fraction) * len(indices))))
        requested = min(requested, len(indices))
        selected.extend(rng.choice(indices, size=requested, replace=False).tolist())
    rng.shuffle(selected)
    if maximum > 0:
        selected = selected[: int(maximum)]
    return np.asarray(selected, dtype=int)


def _stratified_holdout_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Divide índices garantindo espaço para todas as classes nos dois lados."""

    indices = np.asarray(indices, dtype=int)
    subset_labels = np.asarray(labels)[indices]
    classes, counts = np.unique(subset_labels, return_counts=True)
    if not 0.0 < float(fraction) < 0.5:
        raise ValueError("As frações internas devem estar entre 0 e 0,5.")
    stratify = subset_labels if len(classes) > 1 and counts.min() >= 2 else None
    requested = int(np.ceil(float(fraction) * len(indices)))
    if stratify is not None:
        requested = max(requested, len(classes))
        requested = min(requested, len(indices) - len(classes))
    requested = max(1, min(requested, len(indices) - 1))
    first, second = train_test_split(
        indices,
        test_size=requested,
        random_state=int(seed),
        stratify=stratify,
    )
    return np.asarray(first, dtype=int), np.asarray(second, dtype=int)


def generate_counterfactual_pool(
    oracle: Any,
    X_fit: Any,
    y_fit: Any,
    feature_names: Sequence[str],
    *,
    original_tree: Any,
    config: ImprovementSearchConfig,
    constraints_config: Optional[Mapping[str, Any]] = None,
    ontology_validator: Any = None,
    progress_fn: Optional[Callable[[str, int, str], None]] = None,
    cancel_fn: Optional[Callable[[], bool]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Gera CFs exclusivamente no subtreino usado pelo seletor."""

    from counterfactuals.engine import CounterfactualConstraints, CounterfactualEngine

    X = _as_2d(X_fit, name="X_fit")
    y = np.asarray(y_fit)
    names = list(feature_names)
    if len(y) != len(X) or len(names) != X.shape[1]:
        raise ValueError("Schema X/y/feature_names incompatível na geração de CFs.")
    factual_labels = np.asarray(oracle.predict(X))
    origins = _stratified_indices(
        factual_labels, config.origin_fraction, config.max_origins, config.random_state
    )
    constraints = CounterfactualConstraints.from_data(X, names, dict(constraints_config or {}))
    engine = CounterfactualEngine(
        oracle,
        X,
        names,
        # A geração aprende exclusivamente a fronteira do professor. Os
        # rótulos reais ficam reservados para avaliar o desempenho preditivo.
        y_reference=factual_labels,
        constraints=constraints,
        ontology_validator=ontology_validator,
        global_tree=original_tree,
        seed=config.random_state,
    )
    method_token = str(config.counterfactual_method or "COGS").upper()
    methods = ("CLEAR", "COGS") if method_token in {"COMBINED", "CLEAR+COGS"} else (method_token,)
    pool: List[Dict[str, Any]] = []
    warnings: List[str] = []
    rejected_engine = 0
    total_jobs = max(1, len(origins) * len(methods))
    completed = 0
    for source_index in origins:
        if cancel_fn and cancel_fn():
            raise InterruptedError("Melhoria cancelada.")
        instance = X[source_index]
        factual = factual_labels[source_index]
        for method in methods:
            try:
                generated = engine.generate(
                    instance,
                    method=method,
                    model_type="mlp",
                    total_cfs=config.cfs_per_origin,
                    robustness_samples=30,
                    robustness_epsilon=0.02,
                )
            except Exception as exc:
                warnings.append(f"{method} origem {int(source_index)}: {exc}")
                generated = {"candidates": []}
            for candidate in generated.get("candidates", []):
                metrics = dict(candidate.get("metrics") or {})
                semantic = dict(
                    candidate.get("ontology_validation")
                    or candidate.get("semantic_validation")
                    or {}
                )
                if not (
                    bool(metrics.get("validity"))
                    and bool(metrics.get("plausibility", True))
                    and semantic.get("consistent") is not False
                ):
                    rejected_engine += 1
                    continue
                vector = np.asarray(
                    candidate.get("vector", candidate.get("cf")), dtype=float
                ).reshape(-1)
                if vector.shape != (X.shape[1],) or not np.isfinite(vector).all():
                    rejected_engine += 1
                    continue
                pool.append({
                    "cf": vector,
                    "original_class": factual,
                    "source_index": int(source_index),
                    "source_instance": instance.copy(),
                    "method": candidate.get("method", method),
                    "engine_metrics": metrics,
                    "semantic_validation": semantic,
                    "data_role": "development_fit_only",
                })
            completed += 1
            if progress_fn:
                progress_fn(
                    "improvement_cf",
                    10 + int(25 * completed / total_jobs),
                    "Gerando CFs no subtreino de desenvolvimento...",
                )

    unique: List[Dict[str, Any]] = []
    seen = set()
    for item in pool:
        key = tuple(np.round(np.asarray(item["cf"], dtype=float), 10).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique, {
        "data_role": "development_fit_only",
        "n_origins": int(len(origins)),
        "methods": list(methods),
        "generated_valid_plausible": int(len(unique)),
        "duplicates_removed": int(len(pool) - len(unique)),
        "engine_rejected": int(rejected_engine),
        "warnings": warnings,
        "test_queried": False,
    }


def _density_index(
    X: np.ndarray, factual_labels: np.ndarray, percentile: float
) -> Tuple[Dict[Any, Tuple[Optional[NearestNeighbors], float]], np.ndarray]:
    scale = np.std(X, axis=0)
    scale = np.where(scale > EPS, scale, 1.0)
    normalised = X / scale
    result: Dict[Any, Tuple[Optional[NearestNeighbors], float]] = {}
    for label in np.unique(factual_labels):
        subset = normalised[factual_labels == label]
        if len(subset) <= 1:
            result[label] = (None, float("inf"))
            continue
        model = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(subset)
        distances, _ = model.kneighbors(subset)
        threshold = float(np.percentile(distances[:, 1], percentile))
        result[label] = (model, max(threshold, EPS))
    return result, scale


def _type_allowed(cf_type: str, policy: str) -> bool:
    token = str(policy or "A+B+C").upper().replace(" ", "")
    if token in {"ALL", "A+B+C"}:
        return cf_type in {"A", "B", "C"}
    return cf_type in set(token.split("+"))


def prepare_counterfactual_augmentation(
    oracle: Any,
    original_tree: Any,
    X_train: Any,
    y_train: Any,
    cf_list: Sequence[Mapping[str, Any]],
    *,
    parameters: CandidateParameters,
    config: ImprovementSearchConfig,
) -> AugmentationBatch:
    """Filtra, etiqueta e pondera CFs + âncoras reais."""

    X = _as_2d(X_train, name="X_train")
    y = np.asarray(y_train)
    if len(y) != len(X):
        raise ValueError("X_train/y_train incompatíveis.")
    factual_train = np.asarray(oracle.predict(X))
    density_models, scale = _density_index(X, factual_train, config.density_percentile)
    stats: Dict[str, Any] = {
        "total_cfs": int(len(cf_list)),
        "valid_oracle": 0,
        "confidence_rejected": 0,
        "density_rejected": 0,
        "schema_rejected": 0,
        "semantic_rejected": 0,
        "coincident": 0,
        "divergent": 0,
        "gap": 0,
        "used": 0,
        "used_real_anchor": 0,
        "total_cf_weight": 0.0,
        "cf_weight_cap": 0.0,
        "data_role": "development_fit_only",
    }
    eligible: List[Dict[str, Any]] = []
    for item in cf_list:
        vector = np.asarray(item.get("cf"), dtype=float).reshape(-1)
        if vector.shape != (X.shape[1],) or not np.isfinite(vector).all():
            stats["schema_rejected"] += 1
            continue
        semantic = dict(item.get("semantic_validation") or {})
        if semantic.get("consistent") is False:
            stats["semantic_rejected"] += 1
            continue
        original_class = item.get("original_class")
        oracle_prediction = np.asarray(oracle.predict(vector.reshape(1, -1)))[0]
        if oracle_prediction == original_class:
            continue
        stats["valid_oracle"] += 1
        confidence = float(
            _predict_confidence(oracle, vector.reshape(1, -1), np.asarray([oracle_prediction]))[0]
        )
        if confidence < config.confidence_threshold:
            stats["confidence_rejected"] += 1
            continue

        density_factor = 1.0
        model, threshold = density_models.get(original_class, (None, float("inf")))
        if model is not None:
            distance = float(
                model.kneighbors((vector / scale).reshape(1, -1), n_neighbors=1)[0][0, 0]
            )
            if distance > threshold and config.strict_density_gate:
                stats["density_rejected"] += 1
                continue
            density_factor = max(0.10, 1.0 - 0.50 * min(distance / threshold, 1.8))

        tree_prediction = np.asarray(original_tree.predict(vector.reshape(1, -1)))[0]
        if tree_prediction == original_class:
            cf_type = "C"
            stats["gap"] += 1
        elif tree_prediction == oracle_prediction:
            cf_type = "A"
            stats["coincident"] += 1
        else:
            cf_type = "B"
            stats["divergent"] += 1
        if not _type_allowed(cf_type, parameters.cf_types):
            continue

        frontier = 1.0
        if parameters.use_frontier_weight and hasattr(original_tree, "predict_proba"):
            tree_proba = np.asarray(original_tree.predict_proba(vector.reshape(1, -1)))[0]
            frontier = 0.5 + (1.0 - float(np.max(tree_proba)))
        eligible.append({
            "cf": vector,
            "label": oracle_prediction,
            "confidence": confidence,
            "type": cf_type,
            "density_factor": density_factor,
            "frontier": frontier,
        })

    # Peso por tipo. Divergentes: fixo (BIURI, 2,0) ou adaptativo à sua escassez (tese: 1 + min(3, 10 / n_B)).
    n_divergent = sum(item["type"] == "B" for item in eligible)
    if str(config.divergent_boost).lower() == "adaptive":
        boost_b = 1.0 + min(3.0, 10.0 / n_divergent) if n_divergent else 2.0
    else:
        boost_b = float(config.divergent_fixed_weight)
    type_weights = {"A": 1.0, "B": boost_b, "C": float(config.gap_weight)}
    for item in eligible:
        # mesma ordem de multiplicação de sempre: tipo · confiança · densidade · fronteira
        item["raw_weight"] = type_weights[item["type"]] * item["confidence"] * item["density_factor"] * item["frontier"]
    stats["divergent_weight"] = float(boost_b)
    stats["eligible_by_type"] = {key: int(sum(item["type"] == key for item in eligible)) for key in ("A", "B", "C")}

    max_by_ratio = max(1, int(np.floor(config.max_cf_ratio * len(X))))
    maximum = max(0, min(int(parameters.max_cfs), max_by_ratio))
    eligible.sort(key=lambda item: (item["raw_weight"], item["confidence"]), reverse=True)
    selected = eligible[:maximum]
    stats["used"] = int(len(selected))
    if not selected:
        return AugmentationBatch(None, None, None, stats)

    cf_X = np.asarray([item["cf"] for item in selected], dtype=float)
    cf_y = np.asarray([item["label"] for item in selected])
    raw = np.asarray([item["raw_weight"] for item in selected], dtype=float)
    ratio = float(parameters.cf_weight_ratio)
    if config.adjust_ratio_by_oracle_accuracy:
        # Tese: cf_weight_ratio · (acc_MLP / 0,9), no máximo 0,10. Aqui a accuracy é medida só nas linhas recebidas
        # (treino), nunca no teste.
        oracle_accuracy = float(accuracy_score(y, factual_train))
        ratio = min(ratio * (oracle_accuracy / 0.9), 0.10)
        stats["oracle_accuracy_on_fit"] = oracle_accuracy
    sample_size = _effective_sample_size(config, len(X))
    desired_total = ratio * float(sample_size)
    cf_weights = (
        np.full(len(raw), desired_total / len(raw), dtype=float)
        if raw.sum() <= EPS else raw * (desired_total / raw.sum())
    )
    if float(config.min_cf_weight) > 0.0:
        cf_weights = np.clip(cf_weights, float(config.min_cf_weight), float(config.max_cf_weight))
    else:
        cf_weights = np.minimum(cf_weights, float(config.max_cf_weight))
    if cf_weights.sum() > desired_total + EPS:
        cf_weights *= desired_total / cf_weights.sum()
    stats["effective_cf_weight_ratio"] = float(ratio)
    stats["effective_sample_size"] = int(sample_size)

    rng = np.random.RandomState(config.random_state)
    n_anchor = min(len(X), max(int(config.min_anchors), int(np.ceil(config.anchor_ratio * len(X)))))
    anchor_indices = rng.choice(len(X), size=n_anchor, replace=False)
    stats["used_real_anchor"] = int(n_anchor)
    stats["total_cf_weight"] = float(cf_weights.sum())
    stats["cf_weight_cap"] = float(desired_total)
    stats["used_by_type"] = {
        key: int(sum(item["type"] == key for item in selected)) for key in ("A", "B", "C")
    }
    return AugmentationBatch(
        np.vstack([cf_X, X[anchor_indices]]),
        np.concatenate([cf_y, y[anchor_indices]]),
        np.concatenate([cf_weights, np.ones(n_anchor, dtype=float)]),
        stats,
    )


def evaluate_surrogate(oracle: Any, X: Any, y_true: Any, tree: Any) -> Dict[str, float]:
    matrix = _as_2d(X, name="X_evaluation")
    y = np.asarray(y_true)
    if len(y) != len(matrix):
        raise ValueError("X/y incompatíveis na avaliação.")
    oracle_predictions = np.asarray(oracle.predict(matrix))
    predictions = np.asarray(tree.predict(matrix))
    labels = np.unique(y)
    recalls = recall_score(y, predictions, labels=labels, average=None, zero_division=0)
    depth = int(tree.get_depth()) if hasattr(tree, "get_depth") else 0
    leaves = int(tree.get_n_leaves()) if hasattr(tree, "get_n_leaves") else 1
    nodes = int(getattr(tree, "node_count_", getattr(getattr(tree, "tree_", None), "node_count", leaves)))
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "minority_recall": float(np.min(recalls)) if len(recalls) else 0.0,
        "fidelity": float(accuracy_score(oracle_predictions, predictions)),
        "depth": depth,
        "leaves": leaves,
        "nodes": nodes,
    }


def _multiobjective_score(metrics: Mapping[str, float], baseline_nodes: int) -> float:
    complexity_ratio = float(metrics["nodes"]) / max(1.0, float(baseline_nodes))
    complexity_penalty = 0.015 * max(0.0, complexity_ratio - 1.0)
    return float(
        0.34 * metrics["balanced_accuracy"]
        + 0.26 * metrics["macro_f1"]
        + 0.26 * metrics["fidelity"]
        + 0.14 * metrics["minority_recall"]
        - complexity_penalty
    )


def _acceptance_gate(
    matched_control: Mapping[str, float],
    candidate: Mapping[str, float],
    config: ImprovementSearchConfig,
) -> Dict[str, Any]:
    deltas = {
        key: float(candidate[key] - matched_control[key])
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "minority_recall", "fidelity")
    }
    real_gain = max(deltas["accuracy"], deltas["balanced_accuracy"], deltas["macro_f1"])
    attributed_gain = max(
        deltas["balanced_accuracy"],
        deltas["macro_f1"],
        deltas["fidelity"],
    )
    complexity_ratio = float(candidate["nodes"]) / max(1.0, float(matched_control["nodes"]))
    checks = {
        "real_label_improvement": real_gain >= config.minimum_real_gain,
        "balanced_accuracy_noninferior": candidate["balanced_accuracy"] >= matched_control["balanced_accuracy"] - config.real_metric_tolerance,
        "macro_f1_noninferior": candidate["macro_f1"] >= matched_control["macro_f1"] - config.real_metric_tolerance,
        "minority_recall_noninferior": candidate["minority_recall"] >= matched_control["minority_recall"] - config.minority_recall_tolerance,
        "fidelity_noninferior": candidate["fidelity"] >= matched_control["fidelity"] - config.fidelity_tolerance,
        "counterfactual_contribution": attributed_gain >= config.attribution_margin,
        "complexity_guard": complexity_ratio <= config.max_complexity_ratio,
    }
    return {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "blockers": [name for name, passed in checks.items() if not passed],
        "deltas_vs_matched_no_cf": deltas,
        "complexity_ratio": complexity_ratio,
        "decision_scope": "independent_internal_acceptance_holdout",
        "test_used": False,
    }


def _unwrap_builder_result(value: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], dict(value[1] or {})
    return value, {}


def consistency_indicators(oracle: Any, tree: Any, cf_list: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Estudo de consistência da tese (secção 2.6.3), sobre os CFs válidos no oráculo.

    A (coincidência) = válidos na árvore com a mesma nova classe do oráculo / válidos no oráculo;
    B (divergência)  = válidos na árvore com outra nova classe / válidos no oráculo;
    C (brecha)       = válidos no oráculo que a árvore não consegue mudar de classe / válidos no oráculo.
    Diagnóstico descritivo: a tese mostra que uma brecha alta é condição necessária mas não suficiente para melhorar.
    """
    n_generated = len(cf_list)
    vectors, origins = [], []
    for item in cf_list:
        vector = np.asarray(item.get("cf"), dtype=float).reshape(-1)
        if vector.size and np.isfinite(vector).all():
            vectors.append(vector)
            origins.append(item.get("original_class"))
    if not vectors:
        return {"total_cfs": int(n_generated), "valid_oracle": 0, "valid_tree": 0, "coincident": 0, "divergent": 0,
                "gap": 0, "indicador_a": float("nan"), "indicador_b": float("nan"), "indicador_c": float("nan")}
    matrix = np.vstack(vectors)
    origins_arr = np.asarray(origins, dtype=object)
    oracle_pred = np.asarray(oracle.predict(matrix), dtype=object)
    tree_pred = np.asarray(tree.predict(matrix), dtype=object)
    valid_oracle = oracle_pred != origins_arr
    valid_tree = valid_oracle & (tree_pred != origins_arr)
    coincident = valid_tree & (tree_pred == oracle_pred)
    n_mlp, n_tree, n_same = int(valid_oracle.sum()), int(valid_tree.sum()), int(coincident.sum())
    ratio = (lambda value: float(value) / n_mlp) if n_mlp else (lambda value: float("nan"))
    return {
        "total_cfs": int(n_generated), "valid_oracle": n_mlp, "valid_tree": n_tree,
        "coincident": n_same, "divergent": n_tree - n_same, "gap": n_mlp - n_tree,
        "indicador_a": ratio(n_same), "indicador_b": ratio(n_tree - n_same), "indicador_c": ratio(n_mlp - n_tree),
    }


def _parameter_grid(config: "ImprovementSearchConfig") -> List[CandidateParameters]:
    return [
        CandidateParameters(int(max_cfs), str(types), float(ratio), bool(frontier))
        for max_cfs in config.max_cfs_grid
        for types in config.cf_types_grid
        for ratio in config.cf_weight_ratio_grid
        for frontier in config.use_frontier_weight_grid
    ]


def _selection_score(metrics: Mapping[str, float], baseline_nodes: float, config: "ImprovementSearchConfig") -> float:
    if str(config.selection_objective).lower() == OBJECTIVE_FIDELITY:
        return float(metrics["fidelity"])                      # tese: maximizar a fidelidade média
    return _multiobjective_score(metrics, int(round(baseline_nodes)))


_METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "minority_recall", "fidelity", "depth", "leaves", "nodes")


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in _METRIC_KEYS}


def _select_by_two_fold_cv(
    *, oracle, X, y, work_idx, names, candidate_builder, config, constraints_config, ontology_validator,
    progress_fn, cancel_fn, guard,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Any], List[Dict[str, Any]], List[List[int]]]:
    """Seleção da tese: validação cruzada estratificada de 2 partições no bloco de trabalho.

    Correção face ao código da tese: em cada partição, a árvore de controlo e o conjunto de CFs vêm SÓ das linhas
    de ajuste dessa partição; a outra metade nunca origina CFs. Uma configuração sem CFs utilizáveis numa partição
    conta, nessa partição, com a árvore de controlo (como na tese, em que a melhoria devolve a árvore original).
    """
    from sklearn.model_selection import StratifiedKFold

    y_work = np.asarray(y)[work_idx]
    if np.unique(y_work, return_counts=True)[1].min() < 2:
        raise ValueError("A validação cruzada de 2 partições exige pelo menos 2 exemplos por classe no bloco de trabalho.")
    folds = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=int(config.random_state)).split(work_idx, y_work))
    grid = _parameter_grid(config)
    per_param: List[List[Dict[str, Any]]] = [[] for _ in grid]
    control_rows, pool_audits, consistency, fold_sizes = [], [], [], []
    for fold_no, (a, b) in enumerate(folds):
        tr, va = work_idx[a], work_idx[b]
        fold_sizes.append([int(len(tr)), int(len(va))])
        control, _ = _unwrap_builder_result(candidate_builder(TrainingRequest(
            X[tr], y[tr], X[va], y[va], tr, va, None, None, None, None, config.random_state)))
        control_metrics = evaluate_surrogate(oracle, X[va], y[va], control)
        control_rows.append(control_metrics)
        guard.record_selection(PartitionRole.TRAIN, f"generate_counterfactual_pool_cv_fold_{fold_no}", n_samples=int(len(tr)))
        pool, audit = generate_counterfactual_pool(
            oracle, X[tr], y[tr], names, original_tree=control, config=replace(config, random_state=config.random_state + fold_no),
            constraints_config=constraints_config, ontology_validator=ontology_validator, progress_fn=progress_fn, cancel_fn=cancel_fn)
        pool_audits.append(audit)
        consistency.append(dict(consistency_indicators(oracle, control, pool), fold=fold_no))
        guard.record_selection(PartitionRole.VALIDATION, f"cv_fold_{fold_no}_configuration_selection", n_samples=int(len(va)),
                               subrole="configuration_selection")
        for position, parameters in enumerate(grid):
            if cancel_fn and cancel_fn():
                raise InterruptedError("Melhoria cancelada.")
            batch = prepare_counterfactual_augmentation(oracle, control, X[tr], y[tr], pool, parameters=parameters, config=config)
            if batch.X is None:
                per_param[position].append({"metrics": control_metrics, "used_cfs": False, "augmentation": batch.stats})
                continue
            tree, builder_audit = _unwrap_builder_result(candidate_builder(TrainingRequest(
                X[tr], y[tr], X[va], y[va], tr, va, batch.X, batch.y, batch.weights, parameters, config.random_state)))
            per_param[position].append({"metrics": evaluate_surrogate(oracle, X[va], y[va], tree), "used_cfs": True,
                                        "augmentation": batch.stats, "builder_audit": builder_audit})
        if progress_fn:
            progress_fn("improvement_search", 35 + int(30 * (fold_no + 1) / len(folds)), "Validação cruzada (2 partições)...")
    control_mean = _mean_metrics(control_rows)
    candidates = []
    for parameters, rows in zip(grid, per_param):
        if not any(row["used_cfs"] for row in rows):
            continue                                            # nenhuma partição teve CFs: não é uma configuração com CF
        metrics = _mean_metrics([row["metrics"] for row in rows])
        candidates.append({
            "parameters": asdict(parameters), "metrics": metrics,
            "score": _selection_score(metrics, control_mean["nodes"], config),
            "augmentation": [row["augmentation"] for row in rows],
            "builder_audit": [row.get("builder_audit", {}) for row in rows],
            "folds_with_cfs": int(sum(row["used_cfs"] for row in rows)),
            "_tree": None,
        })
    return candidates, control_mean, {"protocol": SELECTION_CV2, "folds": pool_audits}, consistency, fold_sizes


def select_and_refit_improved_surrogate(
    *,
    model_name: str,
    oracle: Any,
    original_tree: Any,
    X_development: Any,
    y_development: Any,
    feature_names: Sequence[str],
    candidate_builder: CandidateBuilder,
    config: ImprovementSearchConfig,
    constraints_config: Optional[Mapping[str, Any]] = None,
    ontology_validator: Any = None,
    progress_fn: Optional[Callable[[str, int, str], None]] = None,
    cancel_fn: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Seleciona, audita num holdout independente e refaz no desenvolvimento.

    A partição de desenvolvimento é dividida em três blocos. O primeiro ajusta
    os substitutos, o segundo escolhe os hiperparâmetros dos CFs e o terceiro
    decide se o ganho é aceite. O teste externo nunca é recebido pela função.
    """

    X = _as_2d(X_development, name="X_development")
    y = np.asarray(y_development)
    names = list(feature_names)
    if len(y) != len(X) or len(names) != X.shape[1]:
        raise ValueError(f"Contexto de melhoria incompatível para {model_name}.")
    classes, counts = np.unique(y, return_counts=True)
    if len(X) < 12 or len(classes) < 2 or counts.min() < 3:
        raise ValueError(
            "A melhoria controlada exige pelo menos 12 amostras, duas classes "
            "e três exemplos de desenvolvimento por classe."
        )

    all_idx = np.arange(len(X))
    work_idx, audit_idx = _stratified_holdout_indices(
        all_idx,
        y,
        fraction=config.acceptance_fraction,
        seed=config.random_state + 701,
    )
    fit_idx, selection_idx = _stratified_holdout_indices(
        work_idx,
        y,
        fraction=config.validation_fraction,
        seed=config.random_state,
    )
    X_fit, y_fit = X[fit_idx], y[fit_idx]
    X_selection, y_selection = X[selection_idx], y[selection_idx]
    X_work, y_work = X[work_idx], y[work_idx]
    X_audit, y_audit = X[audit_idx], y[audit_idx]
    guard = EvaluationProtocolGuard(run_id=f"cf-improvement:{model_name}")

    consistency_selection: Any = None
    configuration_folds: Optional[List[List[int]]] = None
    selection_control_audit: Dict[str, Any] = {}
    if str(config.selection_protocol).lower() == SELECTION_CV2:
        candidates, selection_control_metrics, pool_audit, consistency_selection, configuration_folds = _select_by_two_fold_cv(
            oracle=oracle, X=X, y=y, work_idx=work_idx, names=names, candidate_builder=candidate_builder, config=config,
            constraints_config=constraints_config, ontology_validator=ontology_validator, progress_fn=progress_fn,
            cancel_fn=cancel_fn, guard=guard,
        )
    else:
        candidates, selection_control_metrics, pool_audit, consistency_selection, selection_control_audit = _select_by_holdout(
            oracle=oracle, X_fit=X_fit, y_fit=y_fit, X_selection=X_selection, y_selection=y_selection, fit_idx=fit_idx,
            selection_idx=selection_idx, names=names, candidate_builder=candidate_builder, config=config,
            constraints_config=constraints_config, ontology_validator=ontology_validator, progress_fn=progress_fn,
            cancel_fn=cancel_fn, guard=guard, model_name=model_name,
        )
    return _finish_selection(
        model_name=model_name, oracle=oracle, original_tree=original_tree, X=X, y=y, names=names, all_idx=all_idx,
        work_idx=work_idx, audit_idx=audit_idx, fit_idx=fit_idx, selection_idx=selection_idx, X_work=X_work, y_work=y_work,
        X_audit=X_audit, y_audit=y_audit, candidates=candidates, selection_control_metrics=selection_control_metrics,
        pool_audit=pool_audit, consistency_selection=consistency_selection, configuration_folds=configuration_folds,
        selection_control_audit=selection_control_audit, candidate_builder=candidate_builder, config=config, constraints_config=constraints_config,
        ontology_validator=ontology_validator, progress_fn=progress_fn, cancel_fn=cancel_fn, guard=guard,
    )


def _select_by_holdout(
    *, oracle, X_fit, y_fit, X_selection, y_selection, fit_idx, selection_idx, names, candidate_builder, config,
    constraints_config, ontology_validator, progress_fn, cancel_fn, guard, model_name,
):
    """Protocolo BIURI (por omissão): uma validação interna escolhe a configuração."""
    # Controlo pareado: mesma reextração e mesma semente, mas sem CFs do tutor.
    selection_control_request = TrainingRequest(
        X_fit,
        y_fit,
        X_selection,
        y_selection,
        fit_idx,
        selection_idx,
        None,
        None,
        None,
        None,
        config.random_state,
    )
    selection_control, selection_control_audit = _unwrap_builder_result(
        candidate_builder(selection_control_request)
    )
    selection_control_metrics = evaluate_surrogate(
        oracle, X_selection, y_selection, selection_control
    )
    guard.record_selection(
        PartitionRole.TRAIN,
        "generate_counterfactual_pool_for_configuration",
        n_samples=int(len(X_fit)),
    )
    pool, pool_audit = generate_counterfactual_pool(
        oracle,
        X_fit,
        y_fit,
        names,
        original_tree=selection_control,
        config=config,
        constraints_config=constraints_config,
        ontology_validator=ontology_validator,
        progress_fn=progress_fn,
        cancel_fn=cancel_fn,
    )
    guard.record_selection(
        PartitionRole.VALIDATION,
        "multiobjective_configuration_selection",
        n_samples=int(len(X_selection)),
        subrole="configuration_selection",
    )

    candidates: List[Dict[str, Any]] = []
    grid = _parameter_grid(config)
    for position, parameters in enumerate(grid):
        if cancel_fn and cancel_fn():
            raise InterruptedError("Melhoria cancelada.")
        batch = prepare_counterfactual_augmentation(
            oracle,
            selection_control,
            X_fit,
            y_fit,
            pool,
            parameters=parameters,
            config=config,
        )
        if batch.X is None:
            continue
        request = TrainingRequest(
            X_fit,
            y_fit,
            X_selection,
            y_selection,
            fit_idx,
            selection_idx,
            batch.X,
            batch.y,
            batch.weights,
            parameters,
            # Semente idêntica: a única diferença causal é o lote CF.
            config.random_state,
        )
        tree, builder_audit = _unwrap_builder_result(candidate_builder(request))
        metrics = evaluate_surrogate(oracle, X_selection, y_selection, tree)
        candidates.append({
            "parameters": asdict(parameters),
            "metrics": metrics,
            "score": _selection_score(
                metrics, selection_control_metrics["nodes"], config
            ),
            "augmentation": batch.stats,
            "builder_audit": builder_audit,
            "_tree": tree,
        })
        if progress_fn:
            progress_fn(
                "improvement_search",
                35 + int(30 * (position + 1) / max(1, len(grid))),
                f"Selecionando configuração CF para {model_name}...",
            )
    consistency = consistency_indicators(oracle, selection_control, pool)
    return candidates, selection_control_metrics, pool_audit, consistency, selection_control_audit


def _finish_selection(
    *, model_name, oracle, original_tree, X, y, names, all_idx, work_idx, audit_idx, fit_idx, selection_idx, X_work,
    y_work, X_audit, y_audit, candidates, selection_control_metrics, pool_audit, consistency_selection,
    configuration_folds, selection_control_audit, candidate_builder, config, constraints_config, ontology_validator,
    progress_fn, cancel_fn, guard,
):
    """Gate de aceitação num bloco independente e reajuste final (igual para os dois protocolos de seleção)."""
    partition_sizes = {
        "configuration_fit": int(len(fit_idx)),
        "configuration_selection": int(len(selection_idx)),
        "acceptance_fit": int(len(work_idx)),
        "acceptance_holdout": int(len(audit_idx)),
        "final_refit_development": int(len(all_idx)),
    }
    if configuration_folds is not None:
        partition_sizes["configuration_protocol"] = SELECTION_CV2
        partition_sizes["configuration_folds"] = configuration_folds
    consistency_diagnostic: Dict[str, Any] = {
        "selection": consistency_selection,
        "definition": "A=coincidência, B=divergência, C=brecha (proporções dos CFs válidos no oráculo; tese, 2.6.3)",
        "role": "descriptive_only: não altera a decisão de aceitação",
    }

    current_tree_metrics = evaluate_surrogate(oracle, X_audit, y_audit, original_tree)
    current_tree_note = (
        "descriptive_only: a árvore atual pode ter sido ajustada com todo o "
        "desenvolvimento e não participa no gate"
    )
    if not candidates:
        return {
            "model_name": model_name,
            "status": "rejected_no_valid_counterfactuals",
            "accepted": False,
            "original_tree": original_tree,
            "candidate_tree": None,
            "deployed_tree": original_tree,
            "baseline_metrics": current_tree_metrics,
            "baseline_metrics_role": current_tree_note,
            "matched_no_cf_metrics": selection_control_metrics,
            "matched_no_cf_metrics_role": "configuration_selection_holdout",
            "candidate_metrics": None,
            "pool_audit": pool_audit,
            "partition_sizes": partition_sizes,
            "consistency_diagnostic": consistency_diagnostic,
            "protocol_audit": guard.audit(),
            "locked_test_used": False,
            "final_test_evaluated": False,
        }

    best = max(candidates, key=lambda item: (item["score"], -item["metrics"]["nodes"]))
    best_parameters = CandidateParameters(**best["parameters"])

    # O gate é executado num terceiro bloco que não escolheu a configuração.
    gate_seed = config.random_state + 503
    audit_control_request = TrainingRequest(
        X_work,
        y_work,
        X_audit,
        y_audit,
        work_idx,
        audit_idx,
        None,
        None,
        None,
        None,
        gate_seed,
    )
    audit_control, audit_control_builder = _unwrap_builder_result(
        candidate_builder(audit_control_request)
    )
    audit_control_metrics = evaluate_surrogate(oracle, X_audit, y_audit, audit_control)
    gate_config = replace(config, random_state=gate_seed)
    guard.record_selection(
        PartitionRole.TRAIN,
        "generate_counterfactual_pool_for_acceptance",
        n_samples=int(len(X_work)),
    )
    gate_pool, gate_pool_audit = generate_counterfactual_pool(
        oracle,
        X_work,
        y_work,
        names,
        original_tree=audit_control,
        config=gate_config,
        constraints_config=constraints_config,
        ontology_validator=ontology_validator,
        progress_fn=progress_fn,
        cancel_fn=cancel_fn,
    )
    consistency_diagnostic["acceptance"] = consistency_indicators(oracle, audit_control, gate_pool)
    gate_batch = prepare_counterfactual_augmentation(
        oracle,
        audit_control,
        X_work,
        y_work,
        gate_pool,
        parameters=best_parameters,
        config=gate_config,
    )
    if gate_batch.X is None:
        gate_candidate = None
        gate_candidate_metrics = None
        gate = {
            "accepted": False,
            "checks": {"valid_counterfactuals_on_acceptance_fit": False},
            "blockers": ["valid_counterfactuals_on_acceptance_fit"],
            "decision_scope": "independent_internal_acceptance_holdout",
            "test_used": False,
        }
        gate_candidate_builder: Dict[str, Any] = {}
    else:
        audit_candidate_request = TrainingRequest(
            X_work,
            y_work,
            X_audit,
            y_audit,
            work_idx,
            audit_idx,
            gate_batch.X,
            gate_batch.y,
            gate_batch.weights,
            best_parameters,
            gate_seed,
        )
        gate_candidate, gate_candidate_builder = _unwrap_builder_result(
            candidate_builder(audit_candidate_request)
        )
        gate_candidate_metrics = evaluate_surrogate(
            oracle, X_audit, y_audit, gate_candidate
        )
        guard.record_selection(
            PartitionRole.VALIDATION,
            "independent_counterfactual_acceptance_gate",
            n_samples=int(len(X_audit)),
            subrole="acceptance_holdout",
        )
        gate = _acceptance_gate(
            audit_control_metrics, gate_candidate_metrics, config
        )

    final_tree = original_tree
    final_audit: Dict[str, Any] = {}
    final_pool_audit: Dict[str, Any] = {}
    final_batch_stats: Dict[str, Any] = {}
    if gate["accepted"]:
        final_config = replace(config, random_state=config.random_state + 1000)
        guard.record_selection(
            PartitionRole.TRAIN,
            "generate_counterfactual_pool_for_final_refit",
            n_samples=int(len(X)),
        )
        final_pool, final_pool_audit = generate_counterfactual_pool(
            oracle,
            X,
            y,
            names,
            original_tree=original_tree,
            config=final_config,
            constraints_config=constraints_config,
            ontology_validator=ontology_validator,
            progress_fn=progress_fn,
            cancel_fn=cancel_fn,
        )
        consistency_diagnostic["final"] = consistency_indicators(oracle, original_tree, final_pool)
        final_batch = prepare_counterfactual_augmentation(
            oracle,
            original_tree,
            X,
            y,
            final_pool,
            parameters=best_parameters,
            config=final_config,
        )
        final_batch_stats = final_batch.stats
        if final_batch.X is None:
            gate["accepted"] = False
            gate.setdefault("checks", {})["valid_counterfactuals_for_final_refit"] = False
            gate.setdefault("blockers", []).append("valid_counterfactuals_for_final_refit")
        else:
            final_request = TrainingRequest(
                X,
                y,
                X,
                y,
                all_idx,
                all_idx,
                final_batch.X,
                final_batch.y,
                final_batch.weights,
                best_parameters,
                final_config.random_state,
                final_refit=True,
            )
            final_tree, final_audit = _unwrap_builder_result(
                candidate_builder(final_request)
            )

    public_candidates = [
        {key: value for key, value in item.items() if key != "_tree"}
        for item in candidates
    ]
    return {
        "model_name": model_name,
        "status": "accepted" if gate["accepted"] else "rejected_by_quality_gate",
        "accepted": bool(gate["accepted"]),
        "original_tree": original_tree,
        "candidate_tree": gate_candidate,
        "deployed_tree": final_tree,
        "baseline_metrics": current_tree_metrics,
        "baseline_metrics_role": current_tree_note,
        "matched_no_cf_metrics": audit_control_metrics,
        "candidate_metrics": gate_candidate_metrics,
        "selection_matched_no_cf_metrics": selection_control_metrics,
        "selection_candidate_metrics": best["metrics"],
        "selected_parameters": best["parameters"],
        "selected_augmentation": gate_batch.stats,
        "final_augmentation": final_batch_stats,
        "selection_score": float(best["score"]),
        "acceptance_gate": gate,
        "pool_audit": pool_audit,
        "acceptance_pool_audit": gate_pool_audit,
        "final_pool_audit": final_pool_audit,
        "matched_control_audit": audit_control_builder,
        "candidate_builder_audit": gate_candidate_builder,
        "selection_control_audit": selection_control_audit,
        "final_refit_audit": final_audit,
        "candidate_table": public_candidates,
        "config": asdict(config),
        "profile_notes": THESIS_PROFILE_NOTES if config.profile == PROFILE_THESIS else None,
        "partition_sizes": partition_sizes,
        "consistency_diagnostic": consistency_diagnostic,
        "protocol_audit": guard.audit(),
        "selection_estimate_only": True,
        "final_model_refit_on_all_development": bool(gate["accepted"]),
        "locked_test_used": False,
        "final_test_evaluated": False,
    }


def improve_surrogate(
    mlp_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tree_extractor: Any,
    cf_list: List[Dict[str, Any]],
    feature_names: List[str],
    class_names: List[str],
    sample_size: Optional[int] = None,
    cf_types: str = "A+B",
    include_type_C: bool = True,
    confidence_threshold: float = 0.6,
    max_cfs: int = 15,
    max_cf_ratio: float = 0.2,
    extract_method: str = "extract_tree",
    cf_weight_ratio: float = 0.05,
    mlp_accuracy: float = 0.8,
    max_cf_weight: float = 20.0,
    boost_B: Optional[float] = None,
    weight_C: float = 0.3,
    anchor_ratio: float = 0.2,
    anchor_mix_ratio: float = 0.0,
    use_density_as_weight: bool = True,
    class_density_percentile: float = 95.0,
    class_nn_dict: Optional[Dict[Any, Any]] = None,
    use_frontier_weight: bool = True,
    feature_intervals: Optional[np.ndarray] = None,
    seed: int = 42,
    X_validation: Optional[np.ndarray] = None,
    y_validation: Optional[np.ndarray] = None,
    **extract_kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """API histórica para uma única reextração; a busca segura usa o seletor."""

    del mlp_accuracy, boost_B, weight_C, anchor_mix_ratio, class_nn_dict, use_frontier_weight
    original_tree = getattr(tree_extractor, "explainer_tree", None)
    if original_tree is None:
        raise ValueError("O extrator não contém uma árvore substituta original.")
    policy = "A+B+C" if include_type_C else cf_types
    effective_sample_size = int(sample_size or min(5000, max(1200, 5 * len(X_train))))
    config = ImprovementSearchConfig(
        random_state=seed,
        confidence_threshold=confidence_threshold,
        density_percentile=class_density_percentile,
        strict_density_gate=bool(use_density_as_weight),
        max_cf_ratio=max_cf_ratio,
        anchor_ratio=anchor_ratio,
        max_cf_weight=max_cf_weight,
        sample_size=effective_sample_size,
        max_cfs_grid=(max_cfs,),
        cf_types_grid=(policy,),
        cf_weight_ratio_grid=(cf_weight_ratio,),
    )
    parameters = CandidateParameters(max_cfs, policy, cf_weight_ratio)
    batch = prepare_counterfactual_augmentation(
        mlp_model, original_tree, X_train, y_train, cf_list,
        parameters=parameters, config=config,
    )
    if batch.X is None:
        return original_tree, batch.stats
    if feature_intervals is not None:
        intervals = np.asarray(feature_intervals, dtype=object)
        if intervals.shape != (len(feature_names), 2):
            raise ValueError("feature_intervals incompatível com feature_names.")
        for index, (low, high) in enumerate(intervals):
            batch.X[:, index] = np.clip(batch.X[:, index], float(low), float(high))

    validation_X = np.asarray(X_validation if X_validation is not None else X_train, dtype=float)
    validation_y = np.asarray(y_validation if y_validation is not None else y_train)
    common = dict(
        sample_size=effective_sample_size,
        feature_names=feature_names,
        class_names=class_names,
        X_train=np.asarray(X_train, dtype=float),
        X_test=validation_X,
        y_train=np.asarray(y_train),
        y_test=validation_y,
        extra_X=batch.X,
        extra_y=batch.y,
        extra_weights=batch.weights,
    )
    common.update(extract_kwargs)
    if extract_method == "extract_tree":
        tree_extractor.extract_tree(mlp_model, X_train, y_train, **common)
    elif extract_method == "extract_tree_with_ontology":
        reduced = {key: value for key, value in common.items() if key not in {"feature_names", "class_names"}}
        tree_extractor.extract_tree_with_ontology(
            mlp_model, X_train, y_train, feature_names, class_names, **reduced
        )
    else:
        raise ValueError(f"Método de extração não suportado: {extract_method}")
    return tree_extractor.explainer_tree, batch.stats


def evaluate_improvement(
    mlp_model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    original_tree: Any,
    improved_tree: Any,
) -> Dict[str, float]:
    """Métricas antes/depois; o chamador deve declarar o papel da partição."""

    original = evaluate_surrogate(mlp_model, X_test, y_test, original_tree)
    improved = evaluate_surrogate(mlp_model, X_test, y_test, improved_tree)
    relative = (
        (improved["accuracy"] - original["accuracy"]) / original["accuracy"]
        if original["accuracy"] > 0 else float("nan")
    )
    return {
        "fidelity_original": original["fidelity"],
        "fidelity_improved": improved["fidelity"],
        "accuracy_original": original["accuracy"],
        "accuracy_improved": improved["accuracy"],
        "balanced_accuracy_original": original["balanced_accuracy"],
        "balanced_accuracy_improved": improved["balanced_accuracy"],
        "macro_f1_original": original["macro_f1"],
        "macro_f1_improved": improved["macro_f1"],
        "relative_improvement": float(relative),
    }


__all__ = [
    "THESIS_PROFILE_NOTES",
    "consistency_indicators",
    "AugmentationBatch",
    "CandidateParameters",
    "ImprovementSearchConfig",
    "TrainingRequest",
    "evaluate_improvement",
    "evaluate_surrogate",
    "generate_counterfactual_pool",
    "improve_surrogate",
    "prepare_counterfactual_augmentation",
    "select_and_refit_improved_surrogate",
]
