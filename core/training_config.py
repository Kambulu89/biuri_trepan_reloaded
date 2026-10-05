"""
Presets de treino BIURI / TREPAN Reloaded.

A aplicação de produção V9.2 é bloqueada no modo científico. Os presets
``fast`` e ``balanced`` permanecem apenas para testes de regressão e
desenvolvimento interno; não são seleccionáveis nos fluxos de treino públicos.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Callable, Dict, List, Optional


CancelFn = Optional[Callable[[], bool]]
ProgressFn = Optional[Callable[[str, int, str], None]]

PRODUCTION_TRAINING_PRESET = "scientific"
SCIENTIFIC_TRAINING_ONLY = True


DEFAULT_QUERY_BUDGET_CAP = 250_000
DEFAULT_MIN_SAMPLE_CAP = 5000

@dataclass
class TrainingPreset:
    """Configuração unificada de treino — MLP, Trepan e ontologia."""

    key: str
    display_name: str
    description: str

    # MLP
    run_baseline: bool = True
    run_grid: bool = True
    run_optuna: bool = True
    optuna_trials: int = 50
    mlp_max_iter: int = 1000
    mlp_cv_folds: int = 5
    mlp_solver: str = "adam"
    n_iter_no_change: int = 10
    validation_fraction: float = 0.1
    # Limiares adaptativos agnósticos ao dataset
    small_sample_threshold: int = 500
    wide_feature_ratio: float = 1.0
    hidden_units_min: int = 8
    hidden_units_max: int = 128
    strong_regularization_alpha: float = 0.01
    default_regularization_alpha: float = 0.001
    min_validation_samples_per_class: int = 5
    minimum_signal_margin: float = 0.03
    n_jobs: int = 1
    calibrate_probabilities: bool = True
    calibration_method: str = "sigmoid"
    calibration_cv_folds: int = 5
    mlp_timeout_seconds: Optional[int] = None
    grid_timeout_seconds: Optional[int] = None
    optuna_timeout_seconds: Optional[int] = None

    # Trepan Original
    trepan_sample_size: int = 2000
    trepan_max_queries: int = 2000
    # Teto do orçamento de queries escalado com o nº de nós: mantém datasets grandes tratáveis (agnóstico ao tamanho).
    trepan_query_budget_cap: int = 250_000
    # Teto da amostra de decisão por nó (o TREPAN original usa 1000-10000); 'n_train' sem limite não escala.
    trepan_min_sample_cap: int = 5000
    trepan_max_depth: Optional[int] = None
    trepan_max_nodes: Optional[int] = None
    trepan_max_time_seconds: Optional[int] = None
    trepan_fidelity_target: float = 0.90
    trepan_fidelity_early_stop: float = 0.95
    trepan_min_samples_leaf: int = 4
    # Tolerância de impureza para declarar um nó "puro" e deixar de o expandir. Canónico (NIPS 1995): 0.05.
    # Valor por omissão e de fallback; o tuning científico pode escolher outro por CV interna no treino.
    trepan_purity_epsilon: float = 0.05
    trepan_scientific_tuning: bool = True
    trepan_tuning_cv_folds: int = 3
    trepan_tuning_capacity_candidates: int = 6
    trepan_fidelity_tuning_target: float = 0.95

    # Trepan Reloaded
    reloaded_sample_size: int = 5000
    reloaded_max_time_seconds: Optional[int] = None
    reloaded_fidelity_target: float = 0.90
    reloaded_fidelity_early_stop: float = 0.95
    hybrid_label_weight: float = 0.60
    hybrid_teacher_weight: float = 0.30
    hybrid_semantic_weight: float = 0.10
    canonical_trepan_enabled: bool = True
    canonical_m_of_n_max_n: int = 3
    canonical_max_nodes: int = 31
    global_soft_tree_enabled: bool = False
    semantic_gain_strength: float = 1.0
    semantic_group_strength: float = 0.15
    semantic_candidate_budget: int = 24
    semantic_relation_threshold: float = 0.35
    semantic_active_query_fraction: float = 0.65
    semantic_active_pool_multiplier: int = 4
    # Error-Focused Semantic Refinement (EFSR) — actua apenas em regiões onde
    # o surrogate discorda do MLP e exige ganho real de fidelidade local.
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
    semantic_tuning_candidates: int = 6

    # Ontologia
    top_k_onto_features: Optional[int] = None  # None = todas
    ontology_acceptance_tolerance: float = 0.01  # tolerância máxima vs MLP Original

    # Cache e logs
    use_cache: bool = True
    full_logs: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _grid_minimal() -> Dict[str, List[Any]]:
    return {
        "mlp__hidden_layer_sizes": [(64, 32), (100,)],
        "mlp__activation": ["relu"],
        "mlp__solver": ["adam"],
        "mlp__alpha": [0.001],
        "mlp__learning_rate_init": [0.001],
        "mlp__max_iter": [500],
        "mlp__early_stopping": [True],
        "mlp__validation_fraction": [0.1],
    }


def _grid_small() -> Dict[str, List[Any]]:
    return {
        "mlp__hidden_layer_sizes": [(50,), (100,), (100, 50), (128, 64)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__solver": ["adam"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__learning_rate_init": [0.001, 0.0005],
        "mlp__max_iter": [1000],
        "mlp__early_stopping": [True],
        "mlp__validation_fraction": [0.1],
    }


TRAINING_PRESETS: Dict[str, TrainingPreset] = {
    "fast": TrainingPreset(
        key="fast",
        display_name="Rápido",
        description="Exploración rápida — Optuna corto, grid mínimo, menos consultas Trepan.",
        run_grid=False,
        run_optuna=True,
        optuna_trials=10,
        mlp_max_iter=500,
        mlp_cv_folds=3,
        mlp_timeout_seconds=120,
        grid_timeout_seconds=60,
        optuna_timeout_seconds=90,
        trepan_sample_size=800,
        trepan_max_queries=800,
        trepan_max_depth=8,
        trepan_max_time_seconds=60,
        reloaded_sample_size=1500,
        reloaded_max_time_seconds=90,
        top_k_onto_features=30,
        use_cache=True,
        full_logs=False,
    ),
    "balanced": TrainingPreset(
        key="balanced",
        display_name="Equilibrado",
        description="Compromiso entre tiempo y rigor — preset recomendado.",
        run_grid=True,
        run_optuna=True,
        optuna_trials=30,
        mlp_max_iter=1000,
        mlp_cv_folds=5,
        mlp_timeout_seconds=300,
        grid_timeout_seconds=180,
        optuna_timeout_seconds=240,
        trepan_sample_size=2000,
        trepan_max_queries=2000,
        trepan_max_depth=12,
        trepan_max_time_seconds=120,
        reloaded_sample_size=3000,
        reloaded_max_time_seconds=180,
        top_k_onto_features=100,
        use_cache=True,
        full_logs=False,
    ),
    "scientific": TrainingPreset(
        key="scientific",
        display_name="Científico",
        description="Resultados finales para artículo — optimización completa y registros detallados.",
        run_grid=True,
        run_optuna=True,
        optuna_trials=50,
        mlp_max_iter=2000,
        mlp_cv_folds=5,
        mlp_timeout_seconds=900,
        grid_timeout_seconds=600,
        optuna_timeout_seconds=600,
        trepan_sample_size=2000,
        trepan_max_queries=2000,
        trepan_max_depth=None,
        # Estrutura canónica (31 nós, purity_epsilon 0.05): é o valor por omissão e o de fallback. O tuning científico
        # escolhe purity_epsilon {0.05, 0.02, 0.01} x max_nodes {31, 63} por CV interna, só no treino.
        trepan_max_time_seconds=300,
        reloaded_sample_size=5000,
        reloaded_max_time_seconds=300,
        top_k_onto_features=None,
        use_cache=True,
        full_logs=True,
    ),
}


def get_training_preset(key: str = PRODUCTION_TRAINING_PRESET) -> TrainingPreset:
    """Obtém um preset conhecido.

    O default é científico. Presets não científicos continuam acessíveis apenas
    para testes/desenvolvimento legado; os fluxos públicos devem chamar
    :func:`enforce_scientific_preset`.
    """
    return TRAINING_PRESETS.get(key, TRAINING_PRESETS[PRODUCTION_TRAINING_PRESET])


def enforce_scientific_preset(
    preset: Optional[TrainingPreset] = None, *, use_cache: Optional[bool] = None
) -> TrainingPreset:
    """Força o contrato de treino de produção para o preset científico.

    Apenas a opção operacional de cache pode ser preservada/alterada. Nenhum
    outro hiperparâmetro de ``fast``/``balanced`` atravessa esta fronteira.
    """
    scientific = TRAINING_PRESETS[PRODUCTION_TRAINING_PRESET]
    cache_value = scientific.use_cache
    if use_cache is not None:
        cache_value = bool(use_cache)
    elif preset is not None:
        cache_value = bool(preset.use_cache)
    return replace(scientific, use_cache=cache_value)


def resolve_trepan_structure_limits(preset: TrainingPreset) -> Dict[str, int]:
    """Resolve limites opcionais para inteiros seguros e comparáveis.

    ``trepan_max_depth=None`` significa "sem limite mais apertado que o número
    máximo de nós". Isto evita ``int(None)`` e mantém Original/Reloaded sob o
    mesmo contrato estrutural.
    """
    max_nodes = int(preset.trepan_max_nodes or preset.canonical_max_nodes or 31)
    max_depth = preset.trepan_max_depth
    if max_depth is None:
        max_depth = max_nodes
    return {"max_nodes": max_nodes, "max_depth": int(max_depth)}


def resolve_trepan_min_sample(n_train: int, cap: Optional[int] = DEFAULT_MIN_SAMPLE_CAP) -> int:
    """Amostra mínima de decisão por nó: fórmula da GUI (>= n_train), limitada por ``cap``.

    Até ``cap`` linhas de treino o resultado é o de sempre; acima disso a amostra deixa de crescer com o dataset.
    """
    n = int(n_train)
    value = max(n, min(1000, max(120, n * 3)))
    return value if cap is None else min(value, int(cap))


def required_query_budget(max_nodes: int, min_sample: int, cap: Optional[int] = None) -> int:
    """Orçamento de queries suficiente para expandir todos os nós internos possíveis.

    Cada nó expandido precisa de, no máximo, ``min_sample`` exemplos de decisão (reais + queries);
    uma árvore binária com ``max_nodes`` nós tem no máximo ``(max_nodes - 1) // 2`` nós internos.
    É um teto: o TREPAN só gasta as queries de que precisa.
    """
    internal = max(1, (int(max_nodes) - 1) // 2)
    need = internal * int(min_sample)
    return need if cap is None else min(need, int(cap))


def non_binding_query_budget(max_nodes: int, min_sample: int) -> int:
    """Orçamento de queries que NUNCA limita um TREPAN com ``max_nodes`` nós.

    Todo nó retirado da fila (expandido ou não) consome, no máximo, ``min_sample`` queries e nunca são retirados
    mais nós do que os criados (<= ``max_nodes``). Logo ``max_nodes * min_sample`` é um majorante do consumo;
    o ``+ 1`` garante que o orçamento nunca é atingido (``budget_exhausted`` fica falso por construção).
    """
    return int(max_nodes) * int(min_sample) + 1


def resolve_trepan_query_budget(preset: TrainingPreset, n_train: int) -> int:
    """Orçamento efetivo de queries: nunca inferior ao do preset, mas suficiente para ``max_nodes``."""
    structure = resolve_trepan_structure_limits(preset)
    cap = max(int(preset.trepan_max_queries), int(getattr(preset, "trepan_query_budget_cap", DEFAULT_QUERY_BUDGET_CAP)))
    needed = required_query_budget(
        structure["max_nodes"],
        resolve_trepan_min_sample(n_train, cap=getattr(preset, "trepan_min_sample_cap", DEFAULT_MIN_SAMPLE_CAP)),
        cap=cap,
    )
    return max(int(preset.trepan_max_queries), needed)


def grid_param_grid_for_preset(preset: TrainingPreset) -> Optional[Dict[str, List[Any]]]:
    if not preset.run_grid:
        return None
    if preset.key == "fast":
        return _grid_minimal()
    if preset.key == "balanced":
        return _grid_small()
    return None  # full grid from mlp_optimizer.MLP_GRID_PARAM_GRID


def preset_config_hash(preset: TrainingPreset) -> str:
    import hashlib
    import json

    payload = json.dumps(preset.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
