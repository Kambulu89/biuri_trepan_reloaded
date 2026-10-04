"""Modelo de dados de uma experiência BIURI: a fonte de verdade única da interface.

A GUI **não calcula** ciência: recebe um ``ExperimentResult`` montado a partir do que o
backend já produziu (ver ``core/experiment_builders.py`` e ``gui/result_builder.py``) e
limita-se a apresentá-lo. Este módulo só contém dados e utilitários de serialização, sem Qt
e sem lógica estatística.

Valores em falta nunca são ``0`` nem um ``N/A`` mudo: um ``Measure`` sem valor traz sempre a
razão (``Reason``) pela qual não existe.
"""
from __future__ import annotations

import enum
import math
import time
import uuid
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"


class ExperimentState(str, enum.Enum):
    NO_DATA = "NO_DATA"
    DATA_LOADED = "DATA_LOADED"
    MODEL_TRAINED = "MODEL_TRAINED"
    SEMANTIC_VALIDATED = "SEMANTIC_VALIDATED"
    TREES_BUILT = "TREES_BUILT"
    RESULTS_READY = "RESULTS_READY"
    ERROR = "ERROR"


class Reason:
    """Razões canónicas para um valor inexistente (texto humano em ``gui/strings.py``)."""

    TEACHER_REJECTED = "teacher_rejected"          # Não calculado — professor semântico rejeitado
    NO_ORACLE = "no_oracle"                        # Não aplicável — o modelo não tem oráculo
    NO_ONTOLOGY = "no_ontology"                    # Indisponível — ontologia não carregada
    TREE_NOT_BUILT = "tree_not_built"              # Não executado — árvore não construída
    NOT_TRAINED = "not_trained"                    # Não executado — modelo não treinado
    COMPARISON_NOT_RUN = "comparison_not_run"      # Não executado — comparação de métricas pendente
    ENRICHMENT_NOT_EVALUATED = "enrichment_not_evaluated"
    NOT_RENDERED = "not_rendered"                  # Não executado — árvore ainda não desenhada
    NOT_REPORTED = "not_reported_by_backend"       # O backend não reportou este valor
    NO_SEMANTIC_FEATURES = "no_semantic_features"  # Nenhuma feature semântica gerada

    TUNING_NOT_RUN = "tuning_not_run"              # Não executado — tuning científico não corrido
    NO_ORACLE_CONTRACT = "no_oracle_contract"      # Não aplicável — modo exploratório sem contrato do oráculo

    CATEGORY = {
        TUNING_NOT_RUN: "not_executed",
        NO_ORACLE_CONTRACT: "not_applicable",
        TEACHER_REJECTED: "not_calculated",
        NO_ORACLE: "not_applicable",
        NO_ONTOLOGY: "not_available",
        TREE_NOT_BUILT: "not_executed",
        NOT_RENDERED: "not_executed",
        NOT_TRAINED: "not_executed",
        COMPARISON_NOT_RUN: "not_executed",
        ENRICHMENT_NOT_EVALUATED: "not_executed",
        NOT_REPORTED: "not_available",
        NO_SEMANTIC_FEATURES: "not_available",
    }


@dataclass(frozen=True)
class Measure:
    """Um número com razão explícita quando não existe. Nunca ``0`` por omissão."""

    value: Optional[float] = None
    reason: Optional[str] = None

    @classmethod
    def of(cls, value: Any, reason: str = Reason.NOT_REPORTED) -> "Measure":
        """``None``/NaN/não numérico -> sem valor com ``reason``; caso contrário o número."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return cls(None, reason)
        if math.isnan(number) or math.isinf(number):
            return cls(None, reason)
        return cls(number, None)

    @classmethod
    def na(cls, reason: str) -> "Measure":
        return cls(None, reason)

    @property
    def available(self) -> bool:
        return self.value is not None


@dataclass
class DatasetInfo:
    name: str = ""
    path: Optional[str] = None
    rows: Optional[int] = None
    features: Optional[int] = None
    classes: Optional[int] = None
    class_names: List[str] = field(default_factory=list)
    train_rows: Optional[int] = None
    test_rows: Optional[int] = None
    seed: Optional[int] = None
    hash: Optional[str] = None
    target: Optional[str] = None


@dataclass
class OntologyInfo:
    """Estado da ontologia, separado em eixos independentes (nunca um parágrafo único)."""

    loaded: bool = False
    path: Optional[str] = None
    hash: Optional[str] = None
    structural_status: str = "NOT_EVALUATED"
    reasoner_status: str = "NOT_EXECUTED"          # CONSISTENT / INCONSISTENT / NOT_EXECUTED
    reasoner_engine: Optional[str] = None
    reasoner_seconds: Optional[float] = None
    reasoner_inferred_axioms: Optional[int] = None
    reasoner_fallback: Optional[str] = None
    mapped: Optional[int] = None
    total: Optional[int] = None
    coverage: Optional[float] = None
    ambiguous: Optional[int] = None
    collisions: Optional[int] = None
    tbox_status: str = "NOT_EVALUATED"
    abox_status: str = "NOT_EVALUATED"
    richness: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class EnrichmentInfo:
    """Três estados independentes: validade da ontologia, enriquecimento do MLP e uso no TREPAN."""

    mlp_status: str = "NOT_EVALUATED"              # ACCEPTED / REJECTED / NOT_EVALUATED / NOT_AVAILABLE
    decision: Optional[str] = None                 # código técnico (ex. REJECT_NO_INFORMATIONAL_GAIN)
    reason_text: Optional[str] = None              # explicação humana
    evidence_strength: Optional[str] = None
    base_utility: Measure = field(default_factory=Measure)
    owl_utility: Measure = field(default_factory=Measure)
    delta_utility: Measure = field(default_factory=Measure)
    utility_ci: Optional[List[float]] = None
    generated: Optional[int] = None
    stable: Optional[int] = None
    selected: Optional[int] = None
    trepan_semantics_available: Optional[bool] = None
    trepan_semantics_reason: Optional[str] = None
    teacher: Optional[str] = None                  # mlp_original / mlp_ontological
    reloaded_mode: Optional[str] = None            # augmented / original_space / neutral / none
    attribution_status: Optional[str] = None


@dataclass
class ModelCard:
    key: str                                       # mlp_original, mlp_ontological, c45, trepan_original, trepan_reloaded
    status: str = "NOT_AVAILABLE"                  # AVAILABLE / ACCEPTED / REJECTED / NOT_AVAILABLE
    status_reason: Optional[str] = None
    oracle: Optional[str] = None                   # modelo-oráculo usado (None se não aplicável)
    metrics: Dict[str, Measure] = field(default_factory=dict)      # desempenho preditivo vs rótulos reais
    fidelity: Measure = field(default_factory=Measure)             # concordância com o oráculo
    agreement_with_mlp: Measure = field(default_factory=Measure)   # diagnóstico (ex. C4.5); NÃO é fidelidade
    complexity: Dict[str, Measure] = field(default_factory=dict)   # nodes, depth, leaves, queries
    evaluation_samples: Optional[int] = None
    cached: bool = False
    cache_key: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class TreeDiagnostics:
    tree: str
    logical_nodes: Measure = field(default_factory=Measure)
    rendered_nodes: Measure = field(default_factory=Measure)
    depth: Measure = field(default_factory=Measure)
    leaves: Measure = field(default_factory=Measure)
    queries_used: Measure = field(default_factory=Measure)
    query_budget: Measure = field(default_factory=Measure)
    query_budget_exhausted: Optional[bool] = None
    node_budget: Measure = field(default_factory=Measure)
    nodes_before_pruning: Measure = field(default_factory=Measure)
    nodes_after_pruning: Measure = field(default_factory=Measure)
    loop_end_reason: Optional[str] = None
    stop_reasons: Dict[str, int] = field(default_factory=dict)
    m_of_n_splits: Optional[int] = None
    available: bool = False


@dataclass
class SemanticFeatureRow:
    name: str
    type: str = ""
    source: str = ""
    stability: Measure = field(default_factory=Measure)
    selected: bool = False
    reason: str = ""


@dataclass
class SemanticSplitRow:
    node: Optional[int] = None
    feature: str = ""
    base_score: Measure = field(default_factory=Measure)
    semantic_bonus: Measure = field(default_factory=Measure)
    final_score: Measure = field(default_factory=Measure)
    reason: str = ""
    decision_changed: Optional[bool] = None


@dataclass
class ControlRow:
    """Linha do controlo negativo: sem semântica / OWL real / OWL baralhada, lado a lado."""

    arm: str                                       # no_semantics / real_owl / shuffled_owl
    accuracy: Measure = field(default_factory=Measure)
    fidelity: Measure = field(default_factory=Measure)
    nodes: Measure = field(default_factory=Measure)
    n_runs: int = 1


@dataclass
class AblationRow:
    configuration: str
    accuracy: Measure = field(default_factory=Measure)
    fidelity: Measure = field(default_factory=Measure)
    nodes: Measure = field(default_factory=Measure)
    semantic_contribution: Measure = field(default_factory=Measure)


@dataclass
class BenchmarkSummary:
    metric: str
    rows: List[Dict[str, Any]] = field(default_factory=list)   # {name, mean, std, n}
    n_runs: int = 0
    detail_path: Optional[str] = None


@dataclass
class Message:
    """Mensagem de diagnóstico rastreável a uma experiência (ver ``gui/messages.py``)."""

    level: str                                     # INFO / WARNING / ERROR / SCIENTIFIC_WARNING
    code: str
    text: str
    experiment_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    where: Optional[str] = None
    details: Optional[str] = None
    action: Optional[str] = None


@dataclass
class Provenance:
    experiment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    seed: Optional[int] = None
    dataset_hash: Optional[str] = None
    owl_hash: Optional[str] = None
    config_hash: Optional[str] = None
    build: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    cache_used: bool = False
    cache_key: Optional[str] = None
    split: Optional[str] = None
    training_mode: Optional[str] = None
    source: str = "gui"                            # gui / headless
    execution_mode: Optional[str] = None           # SCIENTIFIC_BENCHMARK / INTERACTIVE_EXPLORATORY (core.execution_mode)


@dataclass
class ScientificDiagnostics:
    """Diagnóstico científico do tuning e do oráculo (só relata o que o backend calculou; nada é recalculado)."""

    execution_mode: Optional[str] = None
    benchmark_eligible: bool = False               # só SCIENTIFIC_BENCHMARK com contrato cumprido
    oracle_id: Optional[str] = None
    oracle_builder: Optional[str] = None
    same_oracle_original_reloaded: Optional[bool] = None
    seed: Optional[int] = None
    cv_plan: Optional[str] = None
    selected_config: Optional[str] = None
    fidelity_mean: Measure = field(default_factory=Measure)
    fidelity_std: Measure = field(default_factory=Measure)
    predictive_stability: Measure = field(default_factory=Measure)       # desvio-padrão da fidelity entre partições (menor = mais estável)
    structural_stability: Measure = field(default_factory=Measure)       # índice de instabilidade estrutural (menor = mais estável)
    selection_probability: Measure = field(default_factory=Measure)         # P(configuração escolhida pela CV completa)
    selected_config_full_cv: Optional[str] = None
    bootstrap_modal_config: Optional[str] = None
    bootstrap_modal_probability: Measure = field(default_factory=Measure)
    selection_runner_up: Optional[str] = None                               # 2.º da distribuição bootstrap (<= 1.º)
    selection_margin: Measure = field(default_factory=Measure)              # top1 − top2 (>= 0)
    full_cv_selection_fragile: Optional[bool] = None                        # a escolhida pela CV completa não é a moda
    bootstrap_method: Optional[str] = None                                  # exact / monte_carlo
    selection_resamples: Optional[int] = None
    structural_stability_evidence: Optional[str] = None                     # observed / censored_by_node_cap
    expansion_triggered: Optional[bool] = None
    capacity_expansion_rounds: Optional[int] = None
    initial_node_grid: List[int] = field(default_factory=list)
    final_node_grid: List[int] = field(default_factory=list)
    expansion_stop_reason: Optional[str] = None
    expansion_interpretation: Optional[str] = None
    last_capacity_step: Dict[str, Any] = field(default_factory=dict)        # previous/candidate max_nodes, fidelity, delta, teste, ganho
    equivalent_candidate_count: Optional[int] = None
    equivalent_candidate_ids: List[str] = field(default_factory=list)
    equivalent_set_probability: Measure = field(default_factory=Measure)
    tree_behavior_unstable: Optional[bool] = None
    tuning_status_reason: Optional[str] = None
    fraction_at_node_cap: Measure = field(default_factory=Measure)
    node_cap_censored: Optional[bool] = None
    queries_used: Measure = field(default_factory=Measure)
    query_budget: Measure = field(default_factory=Measure)
    budget_exhausted: Optional[bool] = None
    test_used_for_selection: Optional[bool] = None
    tuning_status: str = "not_run"                 # stable_exact / stable_equivalent_set / tuning_uncertain / not_run / failed


@dataclass
class ExperimentResult:
    schema_version: str = SCHEMA_VERSION
    state: str = ExperimentState.NO_DATA.value
    provenance: Provenance = field(default_factory=Provenance)
    dataset: Optional[DatasetInfo] = None
    ontology: Optional[OntologyInfo] = None
    enrichment: EnrichmentInfo = field(default_factory=EnrichmentInfo)
    models: Dict[str, ModelCard] = field(default_factory=dict)
    trees: Dict[str, TreeDiagnostics] = field(default_factory=dict)
    semantic_features: List[SemanticFeatureRow] = field(default_factory=list)
    semantic_splits: List[SemanticSplitRow] = field(default_factory=list)
    controls: List[ControlRow] = field(default_factory=list)
    ablation: List[AblationRow] = field(default_factory=list)
    benchmark: Optional[BenchmarkSummary] = None
    counterfactual: Optional[Dict[str, Any]] = None
    messages: List[Message] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    scientific: Optional[ScientificDiagnostics] = None
    stale: bool = False
    stale_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def to_jsonable(obj: Any) -> Any:
    """Converte dataclasses/enums/numpy em estruturas JSON; ``Measure`` -> ``{value, reason}``."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return to_jsonable(obj.item())
        if isinstance(obj, np.ndarray):
            return to_jsonable(obj.tolist())
    except ImportError:  # pragma: no cover
        pass
    return obj


MODEL_ORDER = ("mlp_original", "mlp_ontological", "c45", "trepan_original", "trepan_reloaded")
