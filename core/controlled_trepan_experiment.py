"""Protocolo pareado e controlado para TREPAN Original vs TREPAN Reloaded.

Este módulo existe para impedir uma comparação confundida: os dois braços usam
exactamente o mesmo MLP-oráculo, seed, orçamento de queries, limites de árvore e
amostras de treino. A única variável experimental permitida é a extensão
semântica/ontológica do Reloaded (features OWL, pesos semânticos e projecção das
membership queries).

O conjunto de teste não faz parte de ``fit_controlled_trepan_pair``. Só pode ser
consultado por ``evaluate_controlled_trepan_pair`` depois de ambos os modelos
estarem congelados.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from core.c45_j48_tree import C45Classifier

from core.evaluation_protocol import (
    EvaluationProtocolGuard,
    PartitionRole,
    classification_metrics,
)
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier


@dataclass(frozen=True)
class ControlledTrepanConfig:
    max_nodes: int = 31
    max_depth: int = 8
    min_samples_leaf: int = 2
    min_sample: int = 1000
    max_n: int = 3
    beam_width: int = 2
    max_features_per_node: int = 12
    max_queries: int = 10000
    purity_epsilon: float = 0.05
    purity_alpha: float = 0.05
    mofn_alpha: float = 0.05
    local_model_alpha: float = 0.10
    random_state: int = 42
    semantic_gain_strength: float = 1.0
    semantic_group_strength: float = 0.15
    semantic_candidate_budget: int = 24
    semantic_relation_threshold: float = 0.35
    semantic_active_query_fraction: float = 0.65
    semantic_active_pool_multiplier: int = 4
    error_focused_refinement: bool = True
    error_focus_min_disagreement: float = 0.05
    error_focus_strength: float = 1.0
    error_focus_semantic_weight: float = 0.50
    error_focus_uncertainty_weight: float = 0.35
    error_focus_min_local_fidelity_gain: float = 0.002
    error_focus_min_real_fidelity_gain: float = 0.0
    error_focus_anchor_k: int = 24
    error_focus_top_k: int = 3
    error_focus_min_regions: int = 1
    mirror_when_no_semantic_effect: bool = True
    # Gain_Reloaded(A) = GainRatio(A) + alpha*OntoDepth(A) + beta*ErrorCoverage(A)
    alpha: float = 0.35
    beta: float = 0.20
    gain_criterion: str = "normalized_information_gain"
    semantic_query_projection: bool = True
    error_focus_fidelity_tolerance: float = 0.015

    def common_tree_kwargs(self) -> dict[str, Any]:
        values = asdict(self)
        for key in (
            "semantic_gain_strength", "semantic_group_strength",
            "semantic_candidate_budget", "semantic_relation_threshold",
            "semantic_active_query_fraction", "semantic_active_pool_multiplier",
            "error_focused_refinement", "error_focus_min_disagreement",
            "error_focus_strength", "error_focus_semantic_weight",
            "error_focus_uncertainty_weight", "error_focus_min_local_fidelity_gain",
            "error_focus_min_real_fidelity_gain", "error_focus_anchor_k",
            "error_focus_top_k", "error_focus_min_regions",
            "mirror_when_no_semantic_effect", "alpha", "beta", "gain_criterion",
            "semantic_query_projection", "error_focus_fidelity_tolerance",
        ):
            values.pop(key, None)
        return values


class OriginalOracleProjection:
    """Expõe o MESMO oráculo original num espaço de features aumentado.

    O adaptador não aprende nada e não altera a decisão do MLP. Apenas selecciona
    as colunas que pertencem ao espaço original antes de chamar ``predict`` ou
    ``predict_proba``. Assim, adicionar OWL não muda o professor da experiência.
    """

    ORACLE_TYPE = "projected_original_same_oracle"

    def __init__(self, oracle, original_feature_indices: Sequence[int]):
        self.oracle = oracle
        self.original_feature_indices = tuple(int(i) for i in original_feature_indices)
        self.classes_ = getattr(oracle, "classes_", None)
        self.base_oracle_identity_ = id(oracle)

    def _base(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("O oráculo projectado exige uma matriz 2D.")
        if not self.original_feature_indices:
            raise ValueError("original_feature_indices não pode estar vazio.")
        if max(self.original_feature_indices) >= X.shape[1]:
            raise ValueError(
                "original_feature_indices contém uma coluna fora do schema aumentado."
            )
        return X[:, self.original_feature_indices]

    def predict(self, X):
        return np.asarray(self.oracle.predict(self._base(X)))

    def predict_proba(self, X):
        if not hasattr(self.oracle, "predict_proba"):
            raise AttributeError("O oráculo original não implementa predict_proba().")
        return np.asarray(self.oracle.predict_proba(self._base(X)))


@dataclass
class ControlledTrepanPair:
    original: TrepanOriginalClassifier
    reloaded: TrepanReloadedClassifier
    base_oracle: Any
    reloaded_oracle_for_audit: Any
    guard: EvaluationProtocolGuard
    audit: dict[str, Any]
    base_feature_names: list[str]
    reloaded_feature_names: list[str]
    original_feature_indices: tuple[int, ...]


def _validate_training_matrix(X, y, names: Sequence[str], label: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError(f"{label}: X deve ser uma matriz 2D não vazia.")
    if len(X) != len(y):
        raise ValueError(f"{label}: X e y têm tamanhos incompatíveis.")
    if X.shape[1] != len(names):
        raise ValueError(f"{label}: feature_names não coincide com o número de colunas.")
    if not np.isfinite(X).all():
        raise ValueError(f"{label}: a matriz entregue ao TREPAN deve estar pré-processada e finita.")
    return X, y



def oracle_health_gate(
    estimator,
    X_train,
    y_train,
    *,
    random_state: int = 42,
    cv_folds: int = 3,
    dummy_margin: float = 0.02,
    c45_margin: float = 0.10,
) -> dict[str, Any]:
    """Valida o oráculo exclusivamente por CV interno do treino.

    O MLP deve (a) superar o DummyClassifier por ``dummy_margin`` e (b) não
    ficar abaixo do C4.5-Nativo por mais de ``c45_margin``. Nenhum holdout/teste
    é aceite por esta função.
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X_train/y_train inválidos no gate do oráculo.")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return {
            "status": "dados_insuficientes",
            "valid": False,
            "evaluation_scope": "training_cv_only",
            "reason": "classe_com_menos_de_duas_amostras",
        }
    folds = max(2, min(int(cv_folds), int(counts.min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(random_state))
    c45 = C45Classifier(
        confidence_factor=0.25, min_samples_leaf=2, random_state=int(random_state)
    )
    dummy = DummyClassifier(strategy="most_frequent")
    mlp_scores = cross_val_score(estimator, X, y, cv=cv, scoring="balanced_accuracy")
    c45_scores = cross_val_score(c45, X, y, cv=cv, scoring="balanced_accuracy")
    dummy_scores = cross_val_score(dummy, X, y, cv=cv, scoring="balanced_accuracy")
    mlp_mean = float(np.mean(mlp_scores))
    c45_mean = float(np.mean(c45_scores))
    dummy_mean = float(np.mean(dummy_scores))
    dummy_delta = mlp_mean - dummy_mean
    c45_delta = mlp_mean - c45_mean
    beats_dummy = dummy_delta >= float(dummy_margin)
    within_c45 = c45_delta >= -float(c45_margin)
    valid = bool(beats_dummy and within_c45)
    reasons = []
    if not beats_dummy:
        reasons.append("mlp_nao_supera_dummy")
    if not within_c45:
        reasons.append("mlp_muito_abaixo_do_c45")
    return {
        "status": "oraculo_valido" if valid else "oraculo_invalido",
        "valid": valid,
        "evaluation_scope": "training_cv_only",
        "metric": "balanced_accuracy",
        "folds": int(folds),
        "mlp_cv_balanced_accuracy": mlp_mean,
        "mlp_cv_std": float(np.std(mlp_scores)),
        "c45_cv_balanced_accuracy": c45_mean,
        "c45_cv_std": float(np.std(c45_scores)),
        "dummy_cv_balanced_accuracy": dummy_mean,
        "dummy_cv_std": float(np.std(dummy_scores)),
        "delta_vs_dummy": float(dummy_delta),
        "delta_vs_c45": float(c45_delta),
        "required_dummy_margin": float(dummy_margin),
        "allowed_c45_deficit": float(c45_margin),
        "reasons": reasons,
        "test_used": False,
    }

def fit_reloaded_arm(
    X,
    *,
    oracle,
    feature_names: Sequence[str],
    config: ControlledTrepanConfig,
    semantic_feature_weights=None,
    semantic_feature_groups=None,
    semantic_relatedness_matrix=None,
    query_projector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ontology_graph=None,
    semantic_feature_entities=None,
) -> TrepanReloadedClassifier:
    """Ajusta SÓ o braço Reloaded (mesma construção usada no par controlado).

    Permite treinar vários Reloaded (real vs controlos) sem treinar o Original de cada vez.
    """
    X = np.asarray(X, dtype=float)
    common = config.common_tree_kwargs()
    if semantic_feature_weights is None:
        semantic_feature_weights = np.ones(X.shape[1], dtype=float)
    return TrepanReloadedClassifier(
        **common,
        semantic_gain_strength=float(config.semantic_gain_strength),
        semantic_group_strength=float(config.semantic_group_strength),
        semantic_candidate_budget=int(config.semantic_candidate_budget),
        semantic_relation_threshold=float(config.semantic_relation_threshold),
        semantic_active_query_fraction=float(config.semantic_active_query_fraction),
        semantic_active_pool_multiplier=int(config.semantic_active_pool_multiplier),
        error_focused_refinement=bool(config.error_focused_refinement),
        error_focus_min_disagreement=float(config.error_focus_min_disagreement),
        error_focus_strength=float(config.error_focus_strength),
        error_focus_semantic_weight=float(config.error_focus_semantic_weight),
        error_focus_uncertainty_weight=float(config.error_focus_uncertainty_weight),
        error_focus_min_local_fidelity_gain=float(config.error_focus_min_local_fidelity_gain),
        error_focus_min_real_fidelity_gain=float(config.error_focus_min_real_fidelity_gain),
        error_focus_anchor_k=int(config.error_focus_anchor_k),
        error_focus_top_k=int(config.error_focus_top_k),
        error_focus_min_regions=int(config.error_focus_min_regions),
        mirror_when_no_semantic_effect=bool(config.mirror_when_no_semantic_effect),
        alpha=float(config.alpha),
        beta=float(config.beta),
        gain_criterion=str(config.gain_criterion),
        semantic_query_projection=bool(config.semantic_query_projection),
        error_focus_fidelity_tolerance=float(config.error_focus_fidelity_tolerance),
    ).fit(
        X,
        oracle=oracle,
        feature_names=[str(v) for v in feature_names],
        semantic_feature_weights=semantic_feature_weights,
        semantic_feature_groups=semantic_feature_groups,
        semantic_relatedness_matrix=semantic_relatedness_matrix,
        query_projector=query_projector,
        ontology_graph=ontology_graph,
        semantic_feature_entities=semantic_feature_entities,
    )


def fit_controlled_trepan_pair(
    X_train,
    y_train_real,
    *,
    oracle,
    feature_names: Sequence[str],
    config: ControlledTrepanConfig = ControlledTrepanConfig(),
    reloaded_X_train=None,
    reloaded_feature_names: Optional[Sequence[str]] = None,
    original_feature_indices: Optional[Sequence[int]] = None,
    semantic_feature_weights=None,
    semantic_feature_groups=None,
    semantic_relatedness_matrix=None,
    query_projector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    run_id: str = "controlled_trepan_pair",
    ontology_graph=None,
    semantic_feature_entities=None,
) -> ControlledTrepanPair:
    """Ajusta os dois braços sem aceitar qualquer conjunto de teste.

    ``y_train_real`` é mantido apenas para validar o alinhamento da amostra e para
    auditoria. Os rótulos usados pela árvore substituta vêm exclusivamente do
    mesmo ``oracle``.
    """
    base_names = [str(v) for v in feature_names]
    X_base, y_train_real = _validate_training_matrix(
        X_train, y_train_real, base_names, "TREPAN Original"
    )
    X_rel = X_base if reloaded_X_train is None else np.asarray(reloaded_X_train, dtype=float)
    rel_names = base_names if reloaded_feature_names is None else [str(v) for v in reloaded_feature_names]
    X_rel, _ = _validate_training_matrix(
        X_rel, y_train_real, rel_names, "TREPAN Reloaded"
    )

    if original_feature_indices is None:
        if X_rel.shape[1] != X_base.shape[1]:
            raise ValueError(
                "Espaço Reloaded aumentado exige original_feature_indices explícito."
            )
        original_feature_indices = tuple(range(X_base.shape[1]))
    else:
        original_feature_indices = tuple(int(i) for i in original_feature_indices)
    if len(original_feature_indices) != X_base.shape[1]:
        raise ValueError(
            "original_feature_indices deve mapear exactamente todas as features do espaço original."
        )

    guard = EvaluationProtocolGuard(run_id=run_id)
    guard.record_selection(
        PartitionRole.TRAIN,
        "fit_same_oracle_original_and_reloaded",
        seed=int(config.random_state),
        max_queries=int(config.max_queries),
        min_sample=int(config.min_sample),
    )

    common = config.common_tree_kwargs()
    original = TrepanOriginalClassifier(**common).fit(
        X_base, oracle=oracle, feature_names=base_names
    )

    if X_rel.shape[1] == X_base.shape[1] and original_feature_indices == tuple(range(X_base.shape[1])):
        reloaded_oracle = oracle
        oracle_adapter = "identity_same_oracle"
    else:
        reloaded_oracle = OriginalOracleProjection(oracle, original_feature_indices)
        oracle_adapter = "OriginalOracleProjection"

    if semantic_feature_weights is None:
        semantic_feature_weights = np.ones(X_rel.shape[1], dtype=float)

    reloaded = fit_reloaded_arm(
        X_rel, oracle=reloaded_oracle, feature_names=rel_names, config=config,
        semantic_feature_weights=semantic_feature_weights,
        semantic_feature_groups=semantic_feature_groups,
        semantic_relatedness_matrix=semantic_relatedness_matrix,
        query_projector=query_projector, ontology_graph=ontology_graph,
        semantic_feature_entities=semantic_feature_entities,
    )

    # O professor é o mesmo objecto lógico; no braço enriquecido pode existir
    # apenas um adaptador determinístico de colunas.
    same_oracle = (
        reloaded_oracle is oracle
        or (
            isinstance(reloaded_oracle, OriginalOracleProjection)
            and reloaded_oracle.oracle is oracle
        )
    )
    audit = {
        "protocol": "paired_controlled_original_vs_reloaded_v9_2",
        "only_experimental_variable": "semantic_ontology_extension",
        "same_oracle": bool(same_oracle),
        "base_oracle_identity": int(id(oracle)),
        "reloaded_oracle_adapter": oracle_adapter,
        "same_seed": original.random_state == reloaded.random_state == config.random_state,
        "same_tree_budget": bool(
            original.max_nodes == reloaded.max_nodes
            and original.max_depth == reloaded.max_depth
            and original.min_sample == reloaded.min_sample
            and original.max_queries == reloaded.max_queries
            and original.max_n == reloaded.max_n
            and original.beam_width == reloaded.beam_width
            and original.min_samples_leaf == reloaded.min_samples_leaf
        ),
        "config": asdict(config),
        "base_feature_count": int(X_base.shape[1]),
        "reloaded_feature_count": int(X_rel.shape[1]),
        "semantic_extension_active": bool(reloaded.reload_extension_active_),
        "semantic_graph_active": bool(semantic_relatedness_matrix is not None),
        "error_focused_semantic_refinement": bool(config.error_focused_refinement),
        "error_focus_min_disagreement": float(config.error_focus_min_disagreement),
        "error_focus_min_local_fidelity_gain": float(config.error_focus_min_local_fidelity_gain),
        "error_focus_min_real_fidelity_gain": float(config.error_focus_min_real_fidelity_gain),
        "error_focus_top_k": int(config.error_focus_top_k),
        "error_focus_min_regions": int(config.error_focus_min_regions),
        "semantic_effect_mirror_applied": bool(getattr(reloaded, 'semantic_effect_mirror_applied_', False)),
        "original_membership_queries": int(original.membership_queries_),
        "reloaded_membership_queries": int(reloaded.membership_queries_),
        "test_available_during_fit": False,
        "test_used_for_selection": False,
    }
    if not audit["same_oracle"] or not audit["same_seed"] or not audit["same_tree_budget"]:
        raise RuntimeError(
            "Protocolo pareado inválido: oráculo, seed ou orçamento divergem entre os braços."
        )

    return ControlledTrepanPair(
        original=original,
        reloaded=reloaded,
        base_oracle=oracle,
        reloaded_oracle_for_audit=reloaded_oracle,
        guard=guard,
        audit=audit,
        base_feature_names=base_names,
        reloaded_feature_names=rel_names,
        original_feature_indices=tuple(original_feature_indices),
    )


def _tree_metrics(model, X, y_true, oracle_pred) -> dict[str, Any]:
    pred = np.asarray(model.predict(X))
    metrics = classification_metrics(y_true, pred)
    metrics.update({
        "oracle_fidelity": float(accuracy_score(oracle_pred, pred)),
        "nodes": int(getattr(model, "node_count_", 0)),
        "depth": int(model.get_depth()),
        "leaves": int(model.get_n_leaves()),
        "membership_queries": int(getattr(model, "membership_queries_", 0)),
        "oracle_query_count": int(getattr(model, "oracle_query_count_", 0)),
    })
    return metrics


def evaluate_controlled_trepan_pair(
    pair: ControlledTrepanPair,
    X_test,
    y_test,
    *,
    reloaded_X_test=None,
) -> dict[str, Any]:
    """Avalia os dois modelos uma única vez sobre o mesmo teste bloqueado."""
    X_base = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test)
    if X_base.ndim != 2 or len(X_base) != len(y_test):
        raise ValueError("X_test/y_test inválidos para avaliação pareada.")
    X_rel = X_base if reloaded_X_test is None else np.asarray(reloaded_X_test, dtype=float)
    if len(X_rel) != len(y_test):
        raise ValueError("reloaded_X_test/y_test têm tamanhos incompatíveis.")

    pair.guard.record_final_evaluation(
        PartitionRole.EXTERNAL_TEST,
        "paired_original_vs_reloaded_final_evaluation",
        n_samples=int(len(y_test)),
    )

    original_oracle_pred = np.asarray(pair.base_oracle.predict(X_base))
    reloaded_oracle_pred = np.asarray(pair.reloaded_oracle_for_audit.predict(X_rel))
    if not np.array_equal(original_oracle_pred, reloaded_oracle_pred):
        raise RuntimeError(
            "O braço Reloaded deixou de consultar o mesmo MLP Original; experiência confundida."
        )

    original_metrics = _tree_metrics(pair.original, X_base, y_test, original_oracle_pred)
    reloaded_metrics = _tree_metrics(pair.reloaded, X_rel, y_test, reloaded_oracle_pred)
    comparison = {
        "delta_accuracy": float(reloaded_metrics["accuracy"] - original_metrics["accuracy"]),
        "delta_balanced_accuracy": float(
            reloaded_metrics["balanced_accuracy"] - original_metrics["balanced_accuracy"]
        ),
        "delta_macro_f1": float(reloaded_metrics["macro_f1"] - original_metrics["macro_f1"]),
        "delta_oracle_fidelity": float(
            reloaded_metrics["oracle_fidelity"] - original_metrics["oracle_fidelity"]
        ),
        "delta_nodes": int(reloaded_metrics["nodes"] - original_metrics["nodes"]),
        "delta_depth": int(reloaded_metrics["depth"] - original_metrics["depth"]),
    }
    semantic_audit = pair.reloaded.semantic_audit_summary()
    return {
        "experiment_audit": dict(pair.audit),
        "protocol_audit": pair.guard.audit(),
        "models": {
            "original": original_metrics,
            "reloaded": reloaded_metrics,
        },
        "comparison": comparison,
        "semantic_audit": semantic_audit,
        "semantic_split_audit": list(getattr(pair.reloaded, "semantic_split_audit_", []) or []),
        "error_region_audit": list(getattr(pair.reloaded, "error_region_audit_", []) or []),
        # Porque cada árvore parou (observabilidade; chave separada para não tocar nas métricas).
        "tree_diagnostics": {
            "original": dict(getattr(pair.original, "stop_summary_", {}) or {}),
            "reloaded": dict(getattr(pair.reloaded, "stop_summary_", {}) or {}),
        },
    }


__all__ = [
    "ControlledTrepanConfig",
    "ControlledTrepanPair",
    "OriginalOracleProjection",
    "fit_controlled_trepan_pair",
    "evaluate_controlled_trepan_pair",
    "oracle_health_gate", "fit_reloaded_arm",
]
