"""Extensão ontológica do TREPAN histórico para o BIURI/TREPAN Reloaded V9.2.

A classe deste módulo NÃO troca a família algorítmica do TREPAN. O crescimento,
membership queries, best-first, min_sample, m-of-n, pureza e pruning continuam
sendo fornecidos por :class:`core.trepan_original.TrepanOriginalClassifier`.

A extensão Reloaded entra apenas em dois hooks:
1. prioridade/score semântico na escolha de splits;
2. projecção das membership queries para um espaço ontológico coerente.

Com pesos semânticos neutros e sem projector, o comportamento é exactamente o
mesmo do TREPAN Original histórico.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence
import hashlib
import itertools
import math

import numpy as np

from core.trepan_original import (
    ConstraintSet,
    Literal,
    FeatureDistributionModel,
    MofNTest,
    TrepanOriginalClassifier,
    _Node,
    _entropy,
)
from core.probabilistic_distillation import (
    build_semantic_query_projector,
    constrain_synthetic_samples,
)
from core.error_focused_semantic_refinement import (
    combine_error_and_semantics,
    nearest_anchor_similarity,
    normalized_entropy,
    robust_error_feature_scores,
    semantic_support_vector,
    split_surrogate_fidelity,
)


def _exact_call_key(method: str, X) -> tuple:
    """Chave de uma consulta ao oráculo: método + forma + dtype + SHA-256 dos bytes EXATOS (sem arredondamentos nem aproximações)."""
    a = np.ascontiguousarray(np.asarray(X, dtype=float))
    return (method, a.shape, a.dtype.str, hashlib.sha256(a.tobytes()).hexdigest())


class _ExactCallLedger:
    """Delega no oráculo e regista, por método, chamadas/amostras e as respostas de ``predict`` indexadas pelo conteúdo exato da entrada.

    Só observa: não altera argumentos nem respostas. ``predict_proba`` é exposto apenas se o oráculo o tiver (preserva ``hasattr``).
    """

    def __init__(self, inner, keep_responses: bool = True):
        self.inner = inner
        self.keep_responses = bool(keep_responses)      # False: só contabiliza (sem hashing nem memória)
        self.responses: dict = {}
        self.stats = {"predict": [0, 0], "predict_proba": [0, 0]}
        # B2-C: atribuição (só relato) das chamadas à porção Original ("original") ou à camada semântica ("semantic"); None fora de B2-C.
        self.tag = None
        self.tagged = {"original": {"predict": [0, 0], "predict_proba": [0, 0]}, "semantic": {"predict": [0, 0], "predict_proba": [0, 0]}}

    def _tag_count(self, method, n):
        if self.tag in self.tagged:
            self.tagged[self.tag][method][0] += 1
            self.tagged[self.tag][method][1] += int(n)

    def predict(self, X):
        out = self.inner.predict(X)
        self.stats["predict"][0] += 1
        self.stats["predict"][1] += int(len(X))
        self._tag_count("predict", len(X))
        if self.keep_responses:
            self.responses[_exact_call_key("predict", X)] = np.array(out, copy=True)
        return out

    def __getattr__(self, name):
        if name in ("inner", "stats", "tagged", "tag", "responses", "keep_responses") or name.startswith("__"):
            raise AttributeError(name)         # evita recursão infinita em cópia/pickle antes de o estado existir
        attr = getattr(self.inner, name)
        if name == "predict_proba":
            def counted(X, *a, **k):
                self.stats["predict_proba"][0] += 1
                self.stats["predict_proba"][1] += int(len(X))
                self._tag_count("predict_proba", len(X))
                return attr(X, *a, **k)
            return counted
        return attr


class _ExactReplayOracle:
    """Reexecução do Original (mirror): serve de uma cópia da resposta SÓ as chamadas ``predict`` cuja entrada é byte-idêntica a uma já feita ao
    MESMO oráculo neste mesmo ``fit``; tudo o resto vai ao oráculo. Pressupõe oráculo determinístico (contrato FrozenOracle)."""

    def __init__(self, ledger: _ExactCallLedger):
        self.ledger = ledger
        self.inner = ledger.inner
        self.stats = {"predict": [0, 0], "predict_proba": [0, 0]}
        self.reused = {"calls": 0, "samples": 0}

    def predict(self, X):
        stored = self.ledger.responses.get(_exact_call_key("predict", X))
        if stored is not None:
            self.reused["calls"] += 1
            self.reused["samples"] += int(len(X))
            return np.array(stored, copy=True)
        self.stats["predict"][0] += 1
        self.stats["predict"][1] += int(len(X))
        return self.inner.predict(X)

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if name == "predict_proba":
            def counted(X, *a, **k):
                self.stats["predict_proba"][0] += 1
                self.stats["predict_proba"][1] += int(len(X))
                return attr(X, *a, **k)
            return counted
        return attr


class TrepanReloadedClassifier(TrepanOriginalClassifier):
    """TREPAN Reloaded construído directamente sobre o motor histórico.

    ``GAIN_CRITERIA`` lista as bases aceites para Gain_Reloaded.

    ``semantic_feature_weights`` e ``query_projector`` são argumentos de
    ``fit`` e não do construtor para que closures/callbacks de treino não fiquem
    persistidos no artefacto final.
    """

    GAIN_CRITERIA = ("normalized_information_gain", "information_gain", "gain_ratio")
    SEMANTIC_CALIBRATIONS = (None, "tiebreak", "bounded", "calibrated")
    # Fase 1b (V2): no mirror, a reexecução do Original reutiliza as respostas de chamadas ``predict`` byte-idênticas já feitas ao mesmo
    # oráculo neste fit (hoje: a rotulagem inicial do treino). ``False`` restaura o comportamento anterior (útil para auditoria/ablação).
    REUSE_IDENTICAL_ORACLE_CALLS = True
    # Contrato de equivalência EQ v2.0 (Doc. 7 §9): S0 ontologia ausente, S1 rejeitada, S2 válida sem mecanismo semântico executado
    # (≡ Original, E1–E9 integrais); S3 válida com tentativa semântica executada (porção Original ≡ Original isolado + tentativa auditada).
    EQ_CONTRACT_VERSION = "2.0-S0S3"

    def __init__(
        self,
        max_nodes: int = 31,
        max_depth: int = 8,
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
        semantic_gain_strength: float = 1.0,
        semantic_group_strength: float = 0.15,
        semantic_candidate_budget: int = 24,
        semantic_relation_threshold: float = 0.35,
        semantic_active_query_fraction: float = 0.65,
        semantic_active_pool_multiplier: int = 4,
        error_focused_refinement: bool = True,
        error_focus_min_disagreement: float = 0.05,
        error_focus_strength: float = 1.0,
        error_focus_semantic_weight: float = 0.50,
        error_focus_uncertainty_weight: float = 0.35,
        error_focus_min_local_fidelity_gain: float = 0.002,
        error_focus_min_real_fidelity_gain: float = 0.0,
        error_focus_anchor_k: int = 24,
        error_focus_top_k: int = 3,
        error_focus_min_regions: int = 1,
        mirror_when_no_semantic_effect: bool = True,
        alpha: float = 0.35,
        beta: float = 0.20,
        gain_criterion: str = "normalized_information_gain",
        semantic_query_projection: bool = True,
        error_focus_fidelity_tolerance: float = 0.015,
        semantic_calibration: Optional[str] = None,
        semantic_lambda_cap: float = 0.20,
        semantic_tie_se_multiplier: float = 1.0,
        log_split_candidates: bool = False,
        statistical_fallback: bool = False,
        b2c_semantic_node_priority: bool = False,
    ):
        super().__init__(
            max_nodes=max_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_sample=min_sample,
            max_n=max_n,
            beam_width=beam_width,
            max_features_per_node=max_features_per_node,
            max_queries=max_queries,
            purity_epsilon=purity_epsilon,
            purity_alpha=purity_alpha,
            mofn_alpha=mofn_alpha,
            local_model_alpha=local_model_alpha,
            random_state=random_state,
        )
        self.semantic_gain_strength = semantic_gain_strength
        self.semantic_group_strength = float(semantic_group_strength)
        self.semantic_candidate_budget = int(semantic_candidate_budget)
        self.semantic_relation_threshold = float(semantic_relation_threshold)
        self.semantic_active_query_fraction = float(semantic_active_query_fraction)
        self.semantic_active_pool_multiplier = int(semantic_active_pool_multiplier)
        self.error_focused_refinement = bool(error_focused_refinement)
        self.error_focus_min_disagreement = float(error_focus_min_disagreement)
        self.error_focus_strength = float(error_focus_strength)
        self.error_focus_semantic_weight = float(error_focus_semantic_weight)
        self.error_focus_uncertainty_weight = float(error_focus_uncertainty_weight)
        self.error_focus_min_local_fidelity_gain = float(error_focus_min_local_fidelity_gain)
        self.error_focus_min_real_fidelity_gain = float(error_focus_min_real_fidelity_gain)
        self.error_focus_anchor_k = int(error_focus_anchor_k)
        self.error_focus_top_k = max(0, int(error_focus_top_k))
        self.error_focus_min_regions = max(0, int(error_focus_min_regions))
        self.mirror_when_no_semantic_effect = bool(mirror_when_no_semantic_effect)
        # Gain_Reloaded(A) = GainRatio(A) + alpha*OntoDepth(A) + beta*ErrorCoverage(A)
        self.alpha = float(alpha)
        self.beta = float(beta)
        # Base do Gain_Reloaded. Por omissão IG/H(nó) em [0,1], na mesma escala
        # de OntoDepth e ErrorCoverage. "information_gain" (IG cru do TREPAN) e
        # "gain_ratio" ficam disponíveis para ablação; na calibração V9.2 o
        # GainRatio baixou a fidelidade ao oráculo abaixo do Original.
        if gain_criterion not in self.GAIN_CRITERIA:
            raise ValueError(f"gain_criterion deve ser um de {self.GAIN_CRITERIA}.")
        self.gain_criterion = gain_criterion
        # Membership queries projectadas para o domínio OWL antes do oráculo.
        self.semantic_query_projection = bool(semantic_query_projection)
        # EFSR: perda de fidelidade local/real tolerada quando a coerência
        # semântica do split aumenta. 0 recupera o gate estrito anterior.
        self.error_focus_fidelity_tolerance = float(error_focus_fidelity_tolerance)
        # B1 (EXPERIMENTAL, desligado por omissão): calibração do contributo semântico no score do split. ``None`` = score histórico
        # (bit-a-bit). Só atua com ontologia efetiva (S3); nunca altera o caminho ≡ Original (S0–S2).
        if semantic_calibration not in self.SEMANTIC_CALIBRATIONS:
            raise ValueError(f"semantic_calibration deve ser um de {self.SEMANTIC_CALIBRATIONS}.")
        if float(semantic_lambda_cap) < 0.0 or float(semantic_tie_se_multiplier) < 0.0:
            raise ValueError("semantic_lambda_cap e semantic_tie_se_multiplier devem ser >= 0.")
        self.semantic_calibration = semantic_calibration
        self.semantic_lambda_cap = float(semantic_lambda_cap)
        self.semantic_tie_se_multiplier = float(semantic_tie_se_multiplier)
        # Instrumentação (sem efeito nas decisões, nas consultas nem no RNG): regista TODOS os candidatos pontuados, não só o vencedor.
        self.log_split_candidates = bool(log_split_candidates)
        # B2-C (EXPERIMENTAL, desligado por omissão; Doc. 20): a amostra-base de decisão, a prioridade de expansão e a pesquisa estatística
        # são as do Original (corrente ``rng_original`` = RNG do modelo de distribuição, intocada); a camada ontológica usa uma amostra
        # ADICIONAL com corrente ``rng_semantic`` separada, e o fallback de uma proposta semântica rejeitada é a escolha estatística obtida
        # no contexto-base. Só atua com ontologia efetiva (S3). ``b2c_semantic_node_priority=True`` reintroduz a prioridade semântica de nós
        # (fator experimental separado, que quebra a preservação da ordem de expansão do Original).
        self.statistical_fallback = bool(statistical_fallback)
        self.b2c_semantic_node_priority = bool(b2c_semantic_node_priority)

    def _b2c_active(self) -> bool:
        return bool(self.statistical_fallback and self._ontology_effective())

    def fit(
        self,
        X,
        y=None,
        *,
        oracle=None,
        sample_weight=None,
        feature_names: Optional[Sequence[str]] = None,
        semantic_feature_weights=None,
        semantic_feature_groups: Optional[Sequence[Optional[str]]] = None,
        semantic_relatedness_matrix=None,
        query_projector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        semantic_feature_depths=None,
        ontology_graph=None,
        semantic_feature_entities: Optional[Sequence[Optional[str]]] = None,
    ):
        X_arr = np.asarray(X, dtype=float)
        if semantic_feature_entities is not None and len(semantic_feature_entities) != X_arr.shape[1]:
            raise ValueError(
                "semantic_feature_entities incompatível com o número de features: "
                f"{len(semantic_feature_entities)} != {X_arr.shape[1]}."
            )
        self.semantic_feature_entities_ = (
            [str(v) if v not in (None, '') else None for v in semantic_feature_entities]
            if semantic_feature_entities is not None else None
        )
        graph_active = bool(ontology_graph is not None and getattr(ontology_graph, "is_active", False))
        if semantic_feature_depths is None and graph_active:
            names = list(feature_names or [f"x{i}" for i in range(X_arr.shape[1])])
            semantic_feature_depths = [ontology_graph.get_feature_depth(name) for name in names]
        if semantic_feature_depths is None:
            depths = np.zeros(X_arr.shape[1], dtype=float)
        else:
            depths = np.asarray(semantic_feature_depths, dtype=float).reshape(-1)
            if len(depths) != X_arr.shape[1]:
                raise ValueError(
                    "semantic_feature_depths incompatível com o número de features: "
                    f"{len(depths)} != {X_arr.shape[1]}."
                )
            if not np.isfinite(depths).all() or np.any(depths < 0) or np.any(depths > 1):
                raise ValueError("semantic_feature_depths deve conter valores finitos em [0,1].")
        self.semantic_feature_depths_ = depths
        self.ontology_active_ = graph_active
        self.ontology_sampling_audit_ = {
            "batches": 0, "candidates": 0, "rejected": 0, "fallback_filled": 0,
        }
        schema_names = list(feature_names or [f"x{i}" for i in range(X_arr.shape[1])])
        self._ontology_domain_fit = (
            ontology_graph.domain_constraints(schema_names, X_arr) if graph_active else None
        )
        if self._ontology_domain_fit is not None and not self._ontology_domain_fit.is_active:
            self._ontology_domain_fit = None
        query_projector, _domain, projection_audit = build_semantic_query_projector(
            ontology_graph if graph_active else None,
            schema_names,
            X_arr,
            semantic_query_projection=self.semantic_query_projection,
            base_projector=query_projector,
        )
        self.semantic_query_projection_audit_ = projection_audit
        if semantic_feature_weights is None:
            weights = np.ones(X_arr.shape[1], dtype=float)
        else:
            weights = np.asarray(semantic_feature_weights, dtype=float).reshape(-1)
            if len(weights) != X_arr.shape[1]:
                raise ValueError(
                    "semantic_feature_weights incompatível com o número de features: "
                    f"{len(weights)} != {X_arr.shape[1]}."
                )
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise ValueError("semantic_feature_weights deve conter apenas valores positivos e finitos.")

        # Normaliza em torno de 1 para evitar que a escala absoluta domine o IG.
        mean = float(np.mean(weights)) if len(weights) else 1.0
        self.semantic_feature_weights_ = weights / max(mean, 1e-12)
        if semantic_feature_groups is None:
            self.semantic_feature_groups_ = [None] * X_arr.shape[1]
        else:
            if len(semantic_feature_groups) != X_arr.shape[1]:
                raise ValueError(
                    "semantic_feature_groups incompatível com o número de features: "
                    f"{len(semantic_feature_groups)} != {X_arr.shape[1]}."
                )
            self.semantic_feature_groups_ = [
                str(value).strip() if value not in (None, '') else None
                for value in semantic_feature_groups
            ]
        if semantic_relatedness_matrix is None:
            self.semantic_relatedness_matrix_ = None
        else:
            rel = np.asarray(semantic_relatedness_matrix, dtype=float)
            if rel.shape != (X_arr.shape[1], X_arr.shape[1]):
                raise ValueError("semantic_relatedness_matrix deve ser quadrada e compatível com as features.")
            if not np.isfinite(rel).all() or np.any(rel < 0) or np.any(rel > 1):
                raise ValueError("semantic_relatedness_matrix deve conter valores finitos em [0,1].")
            self.semantic_relatedness_matrix_ = rel

        repeated_groups = {
            g for g in self.semantic_feature_groups_
            if g and self.semantic_feature_groups_.count(g) > 1
        }
        related_off_diagonal = False
        if self.semantic_relatedness_matrix_ is not None:
            off = np.asarray(self.semantic_relatedness_matrix_, dtype=float).copy()
            np.fill_diagonal(off, 0.0)
            related_off_diagonal = bool(np.any(off > float(self.semantic_relation_threshold)))
        self.reload_extension_active_ = bool(
            query_projector is not None
            or (self.alpha > 0 and np.any(self.semantic_feature_depths_ > 0))
            or not np.allclose(self.semantic_feature_weights_, 1.0, atol=1e-12)
            or repeated_groups
            or related_off_diagonal
        )
        # Predicado único de «ontologia efetiva» (contrato EQ, Doc. 7): a extensão ontológica altera algum mecanismo do Original.
        # Inclui o domínio OWL (filtragem das membership queries), que atua mesmo com a projeção desligada.
        self.ontology_effective_ = bool(self.reload_extension_active_ or self._ontology_domain_fit is not None)
        self.split_candidate_log_: list[dict] = []
        self.b2c_audit_: list[dict] = []
        self._b2c_base_draw = False
        self._b2c_sem_X = None
        self._b2c_sem_y = None
        self._b2c_semantic_tests_: set = set()
        self._b2c_first_divergence_order_ = None
        self.semantic_membership_queries_ = 0
        semantic_inputs_given = any(v is not None for v in (
            semantic_feature_weights, semantic_feature_groups, semantic_relatedness_matrix, semantic_feature_depths, query_projector,
            semantic_feature_entities,
        ))
        if self.ontology_effective_:
            self.ontology_state_ = "S3"        # válida com tentativa semântica executada
        elif ontology_graph is not None and not graph_active:
            self.ontology_state_ = "S1"        # rejeitada / inativa
        elif ontology_graph is None and not semantic_inputs_given:
            self.ontology_state_ = "S0"        # ausente
        else:
            self.ontology_state_ = "S2"        # válida sem mecanismo semântico executado
        self.algorithm_family_ = "historical_trepan"
        self.semantic_projected_query_count_ = 0
        self.semantic_projection_batches_ = 0
        self.semantic_active_query_batches_ = 0
        self.semantic_active_query_selected_ = 0
        self.semantic_candidate_generation_count_ = 0
        self.semantic_split_audit_: list[dict] = []
        self._semantic_candidates_last_ = 0
        self.error_region_audit_: list[dict] = []
        self.error_focused_interventions_attempted_ = 0
        self.error_focused_interventions_accepted_ = 0
        self.error_focused_interventions_rejected_ = 0
        self.error_focused_query_batches_ = 0
        self.error_focused_query_selected_ = 0
        self.probability_rows_reused_for_uncertainty_ = 0
        self.probability_batches_reused_for_uncertainty_ = 0
        self._current_error_profile_ = None
        self._current_real_validation_X_ = None
        self._current_real_validation_y_ = None
        self._error_focus_fallback_used_ = 0
        self._error_focus_fallback_nodes_: set[int] = set()
        self.semantic_effect_mirror_applied_ = False
        self.semantic_effect_mirror_reason_ = None
        self._query_projector_fit = query_projector
        # Como o projector, o grafo é só de treino e não fica no artefacto.
        self._ontology_graph_fit = ontology_graph if graph_active else None
        # A estrutura semântica é fixa durante o fit; o score de cada split
        # candidato consulta-a, por isso é calculada uma única vez.
        self._semantic_structure_cache_ = None
        self._semantic_structure_cache_enabled_ = True
        ledger = None
        pass_oracle = oracle
        if oracle is not None:
            # Observador (nunca altera argumentos nem respostas): contabilidade do contrato S0–S3 em TODOS os estados; só guarda respostas
            # (reutilização exata V2) quando o mirror é possível.
            mirror_possible = bool(self.mirror_when_no_semantic_effect and self.reload_extension_active_)
            ledger = _ExactCallLedger(oracle, keep_responses=mirror_possible)
            pass_oracle = ledger
            if self.statistical_fallback:
                ledger.tag = "original"          # B2-C: por omissão a porção Original; a camada semântica marca-se explicitamente
        self._ledger = ledger if self.statistical_fallback else None      # só durante o fit e só em B2-C (nunca persiste no artefacto)
        try:
            fitted = super().fit(
                X_arr,
                y=y,
                oracle=pass_oracle,
                sample_weight=sample_weight,
                feature_names=feature_names,
            )
        finally:
            # Não persistir callback/closure de treino no artefacto.
            self._ledger = None
            self._b2c_sem_X = None
            self._b2c_sem_y = None
            self._query_projector_fit = None
            self._ontology_graph_fit = None
            self._ontology_domain_fit = None
            self._semantic_structure_cache_enabled_ = False
            self._semantic_structure_cache_ = None
        if ledger is not None:
            self.oracle_info_["oracle_type"] = type(oracle).__name__      # o invólucro de observação não é visível no relato
        self.semantic_split_audit_ = []
        for item in self.split_audit_:
            row = dict(item)
            row['semantic_weight'] = float(row.get('semantic_weight', 1.0))
            row['semantic_features'] = list(row.get('semantic_features', []))
            row['selection_score'] = float(row.get('selection_score', row.get('information_gain', 0.0)))
            row['information_gain'] = float(row.get('information_gain', 0.0))
            row['semantic_bonus'] = float(row.get('semantic_bonus', row['selection_score'] - row['information_gain']) or 0.0)
            row['decision_changed'] = bool(row.get('decision_changed', False))
            row['decision_reinforced'] = bool(row.get('decision_reinforced', False))
            row['ontology_influenced'] = bool(row.get('ontology_influenced', row['decision_changed'] or row['decision_reinforced']))
            self._annotate_semantic_split_row(row)
            self.semantic_split_audit_.append(row)
        preliminary_summary = self.semantic_audit_summary()
        measurable_effect = bool(
            preliminary_summary.get('ontology_influenced_splits', 0)
            or preliminary_summary.get('error_focused_interventions_accepted', 0)
        )
        if (
            self.mirror_when_no_semantic_effect
            and self.reload_extension_active_
            and not measurable_effect
            and not self.statistical_fallback        # B2-C já preserva a porção Original por construção; o mirror é redundante
        ):
            # Produção: uma OWL carregada não pode alterar silenciosamente a
            # árvore quando nenhuma contribuição semântica mensurável foi
            # aceite. Reexecuta exactamente o TREPAN Original com o mesmo
            # oráculo, seed e orçamento e adopta esse estado final.
            rerun_oracle = (
                _ExactReplayOracle(ledger) if (ledger is not None and self.REUSE_IDENTICAL_ORACLE_CALLS) else
                (ledger.inner if ledger is not None else oracle)
            )
            original = TrepanOriginalClassifier(
                max_nodes=self.max_nodes,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_sample=self.min_sample,
                max_n=self.max_n,
                beam_width=self.beam_width,
                max_features_per_node=self.max_features_per_node,
                max_queries=self.max_queries,
                purity_epsilon=self.purity_epsilon,
                purity_alpha=self.purity_alpha,
                mofn_alpha=self.mofn_alpha,
                local_model_alpha=self.local_model_alpha,
                random_state=self.random_state,
            ).fit(
                X_arr, y=y, oracle=rerun_oracle, sample_weight=sample_weight,
                feature_names=feature_names,
            )
            original.oracle_info_["oracle_type"] = type(oracle).__name__
            if ledger is not None:
                replay = rerun_oracle if isinstance(rerun_oracle, _ExactReplayOracle) else None
                self._mirror_rerun_accounting = {
                    "stats": dict(replay.stats) if replay is not None else None,
                    "reused": dict(replay.reused) if replay is not None else {"calls": 0, "samples": 0},
                }
            self.pre_mirror_semantic_audit_ = dict(preliminary_summary)
            self.pre_mirror_semantic_split_audit_ = list(self.semantic_split_audit_)
            self._adopt_original_state(original)
            self.semantic_effect_mirror_applied_ = True
            self.semantic_effect_mirror_reason_ = 'no_measurable_semantic_effect'
            # Mantém a auditoria das tentativas semânticas, mas deixa explícito
            # que nenhuma delas chegou ao modelo final espelhado.
            mirrored_rows = []
            for row in self.pre_mirror_semantic_split_audit_:
                item = dict(row)
                item['decision_changed'] = False
                item['decision_reinforced'] = False
                item['ontology_influenced'] = False
                item['final_model_mirrored_to_original'] = True
                mirrored_rows.append(item)
            self.semantic_split_audit_ = mirrored_rows
            fitted = self
        self._current_real_validation_X_ = None
        self._current_real_validation_y_ = None
        if ledger is not None:
            self.oracle_accounting_ = self._build_oracle_accounting(ledger)
        self.semantic_audit_summary_ = self.semantic_audit_summary()
        return fitted

    def _build_oracle_accounting(self, ledger: _ExactCallLedger) -> dict:
        """Contabilidade auditável do consumo do oráculo (só relato; não influencia decisões), contrato EQ v2.0.

        * ``semantic_oracle_*``: consultas da tentativa semântica (passagem semântica; 0 nos estados S0–S2, em que o fit É o Original).
        * ``original_oracle_*``: pedidos da porção Original (reexecução do ``mirror``: enviados + reutilizados = Original isolado);
          0 e ``original_portion_executed=False`` se não houve reexecução.
        * ``reused_oracle_*``: chamadas byte-idênticas servidas da passagem semântica em vez de repetidas.
        * ``total_oracle_*`` = original + semantic − reused = o que o oráculo realmente recebeu.
        Estado S0–S2: toda a passagem é o Original (``original`` = total, ``semantic`` = 0). S3 sem mirror: a passagem é a tentativa
        semântica (a porção Original isolada é o braço de referência, medido à parte)."""
        state = getattr(self, "ontology_state_", "S3")
        attempt = {m: [int(v[0]), int(v[1])] for m, v in ledger.stats.items()}
        rerun = getattr(self, "_mirror_rerun_accounting", None)
        reused = dict(rerun["reused"]) if rerun else {"calls": 0, "samples": 0}
        calls_att = sum(v[0] for v in attempt.values())
        samples_att = sum(v[1] for v in attempt.values())
        if state != "S3":                                   # S0–S2: o fit é o Original
            orig_calls, orig_samples, sem_calls, sem_samples, orig_exec = calls_att, samples_att, 0, 0, True
        else:
            sem_calls, sem_samples = calls_att, samples_att
            if rerun and rerun["stats"] is not None:
                orig_calls = sum(v[0] for v in rerun["stats"].values()) + reused["calls"]
                orig_samples = sum(v[1] for v in rerun["stats"].values()) + reused["samples"]
                orig_exec = True
            else:
                orig_calls = orig_samples = 0
                orig_exec = False
        tagged = getattr(ledger, "tagged", None)
        b2c_split = bool(state == "S3" and getattr(self, "statistical_fallback", False) and tagged is not None and not rerun)
        if b2c_split:                                        # B2-C: a mesma passagem contém a porção Original e a camada semântica
            orig_calls = sum(v[0] for v in tagged["original"].values()); orig_samples = sum(v[1] for v in tagged["original"].values())
            sem_calls = sum(v[0] for v in tagged["semantic"].values()); sem_samples = sum(v[1] for v in tagged["semantic"].values())
            orig_exec = True
        total_calls = (orig_calls + sem_calls - reused["calls"]) if state == "S3" else calls_att
        total_samples = (orig_samples + sem_samples - reused["samples"]) if state == "S3" else samples_att
        summary = self.semantic_audit_summary()
        effect_observed = bool(
            state == "S3" and not getattr(self, "semantic_effect_mirror_applied_", False)
            and (summary.get("ontology_influenced_splits", 0) or summary.get("error_focused_interventions_accepted", 0))
        )
        if state != "S3":
            status = "NO_SEMANTIC_ATTEMPT"
        elif effect_observed:
            status = "SEMANTIC_EFFECT_STRUCTURAL_GAIN_UNEVALUATED"     # ganho em métricas só se avalia fora do fit (conjuntos de avaliação)
        else:
            status = "SEMANTIC_ATTEMPT_NO_OBSERVED_GAIN"
        out = {
            "eq_contract_version": self.EQ_CONTRACT_VERSION,
            "ontology_state": state,
            "semantic_attempt_executed": bool(state == "S3"),
            "semantic_effect_observed": effect_observed,
            "semantic_attempt_status": status,
            "mirror_applied": bool(self.semantic_effect_mirror_applied_),
            "original_portion_executed": bool(orig_exec),
            "original_oracle_calls": int(orig_calls), "original_oracle_samples": int(orig_samples),
            "semantic_oracle_calls": int(sem_calls), "semantic_oracle_samples": int(sem_samples),
            "reused_oracle_calls": int(reused["calls"]), "reused_oracle_samples": int(reused["samples"]),
            "total_oracle_calls": int(total_calls), "total_oracle_samples": int(total_samples),
            "semantic_attempt": {m: {"calls": v[0], "samples": v[1]} for m, v in attempt.items()} if state == "S3" else None,
            "original_rerun": ({m: {"calls": int(v[0]), "samples": int(v[1])} for m, v in rerun["stats"].items()}
                               if rerun and rerun["stats"] is not None else None),
            "reused_identical": reused,
            "reuse_policy": "byte_identical_predict_same_oracle_same_fit" if self.REUSE_IDENTICAL_ORACLE_CALLS else "disabled",
            "b2c_split_attribution": ({tag: {m: {"calls": int(v[0]), "samples": int(v[1])} for m, v in tagged[tag].items()}
                                       for tag in ("original", "semantic")} if b2c_split else None),
        }
        self._mirror_rerun_accounting = None
        return out

    def _ontology_effective(self) -> bool:
        """``False`` ⇒ o Reloaded delega no TREPAN Original (equivalência EQ: mesma árvore e mesmo fluxo de consultas ao oráculo).

        Antes de ``fit`` assume-se efetiva (comportamento histórico preservado).
        """
        return bool(getattr(self, "ontology_effective_", True))

    def _adopt_original_state(self, original: TrepanOriginalClassifier) -> None:
        """Adopta apenas o estado aprendido do TREPAN Original de referência."""
        for name in (
            'root_', 'classes_', 'n_features_in_', 'feature_names_in_',
            'global_distribution_model_', 'membership_queries_', 'effective_min_sample_',
            'oracle_query_count_', 'node_audit_', 'split_audit_', 'expansion_order_',
            'local_model_count_', 'node_count_', 'best_first_', 'm_of_n_',
            'oracle_', 'oracle_attached_',
            'expansion_log_', 'pruning_audit_', 'pruning_summary_', 'tree_raw_root_', 'nodes_raw_',
            'nodes_created_', 'nodes_expanded_', 'candidate_totals_', 'budget_starved_nodes_',
            'query_budget_exhausted_', 'max_nodes_reached_', 'oracle_info_', 'training_time_',
            'query_time_', 'split_search_time_', 'm_of_n_search_time_', 'pruning_time_',
        ):
            if hasattr(original, name):
                setattr(self, name, getattr(original, name))

    def _semantic_support_vector(self) -> np.ndarray:
        return semantic_support_vector(
            int(getattr(self, "n_features_in_", len(getattr(self, "semantic_feature_weights_", [])))),
            feature_weights=getattr(self, "semantic_feature_weights_", None),
            feature_groups=getattr(self, "semantic_feature_groups_", None),
            relatedness_matrix=getattr(self, "semantic_relatedness_matrix_", None),
        )

    def _known_row_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, str]:
        """Incerteza do oráculo apenas para linhas JÁ conhecidas pelo TREPAN.

        Não gera novas membership queries e não incrementa ``membership_queries_``.
        O uso é auditado separadamente porque fornece probabilidades do mesmo MLP
        nas linhas já consultadas, nunca no teste externo.
        """
        X = np.asarray(X, dtype=float)
        oracle = getattr(self, "oracle_", None)
        if len(X) == 0 or oracle is None or not hasattr(oracle, "predict_proba"):
            return np.zeros(len(X), dtype=float), "unavailable"
        try:
            probabilities = np.asarray(oracle.predict_proba(X), dtype=float)
            uncertainty = normalized_entropy(probabilities)
            self.probability_batches_reused_for_uncertainty_ += 1
            self.probability_rows_reused_for_uncertainty_ += int(len(X))
            return uncertainty, "oracle_predict_proba_known_rows"
        except (ValueError, TypeError, AttributeError, RuntimeError):
            return np.zeros(len(X), dtype=float), "unavailable"

    def _build_error_region_profile(
        self,
        node: _Node,
        X: np.ndarray,
        y: np.ndarray,
        *,
        include_uncertainty: bool = True,
        prediction_override=None,
    ) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("Perfil EFSR exige X/y alinhados.")
        if len(y) == 0:
            disagreement = np.zeros(0, dtype=bool)
        else:
            current_prediction = node.prediction if prediction_override is None else prediction_override
            disagreement = np.asarray(y != current_prediction, dtype=bool)
        disagreement_rate = float(np.mean(disagreement)) if len(disagreement) else 0.0
        feature_error = robust_error_feature_scores(X, disagreement)
        sem_support = self._semantic_support_vector()
        if len(sem_support) != X.shape[1]:
            sem_support = np.zeros(X.shape[1], dtype=float)
        semantic_scores = combine_error_and_semantics(
            feature_error,
            sem_support,
            semantic_weight=float(self.error_focus_semantic_weight),
        ) if X.shape[1] else np.zeros(0, dtype=float)
        nonzero = np.sort(semantic_scores[semantic_scores > 0.0])
        semantic_opportunity = float(np.mean(nonzero[-min(3, len(nonzero)):])) if len(nonzero) else 0.0

        if include_uncertainty:
            uncertainty, probability_source = self._known_row_uncertainty(X)
        else:
            uncertainty = np.zeros(len(X), dtype=float)
            probability_source = "disabled"
        mean_uncertainty = float(np.mean(uncertainty)) if len(uncertainty) else 0.0
        disagreement_uncertainty = (
            float(np.mean(uncertainty[disagreement]))
            if len(uncertainty) and np.any(disagreement)
            else 0.0
        )
        error_indices = np.flatnonzero(disagreement)
        if len(error_indices):
            # Prioriza anchors que combinam erro + incerteza do MLP.
            anchor_priority = 1.0 + float(self.error_focus_uncertainty_weight) * uncertainty[error_indices]
            order = np.argsort(-anchor_priority, kind="stable")
            error_indices = error_indices[order[: max(1, int(self.error_focus_anchor_k))]]

        reach = float(getattr(node, "reach", 0.0))
        priority_score = float(
            reach
            * disagreement_rate
            * (1.0 + float(self.error_focus_uncertainty_weight) * disagreement_uncertainty)
            * (1.0 + float(self.error_focus_semantic_weight) * semantic_opportunity)
        )
        profile = {
            "node_id": int(getattr(node, "node_id", -1)),
            "depth": int(getattr(node, "depth", 0)),
            "reach": reach,
            "sample_count": int(len(y)),
            "disagreement_count": int(np.sum(disagreement)),
            "disagreement_rate": disagreement_rate,
            "fidelity": float(1.0 - disagreement_rate),
            "mean_uncertainty": mean_uncertainty,
            "disagreement_uncertainty": disagreement_uncertainty,
            "semantic_opportunity": semantic_opportunity,
            "error_region_priority": priority_score,
            "feature_error_scores": [float(v) for v in feature_error],
            "semantic_feature_scores": [float(v) for v in semantic_scores],
            "anchor_indices": [int(v) for v in error_indices],
            "probability_source": probability_source,
        }
        eligible, reason = self._resolve_error_region_eligibility(profile, consume=False)
        profile["eligible_for_semantic_refinement"] = bool(eligible)
        profile["eligibility_reason"] = reason
        return profile

    def _resolve_error_region_eligibility(self, profile: dict, *, consume: bool = False) -> tuple[bool, str]:
        """Decide se uma região recebe EFSR sem depender de um limiar fixo único.

        Regiões acima de ``error_focus_min_disagreement`` são elegíveis normalmente.
        Se nenhuma região atingir esse limiar, o modo científico reserva até
        ``error_focus_top_k`` regiões reais com erro para um fallback best-first.
        Isto evita o estado patológico ``regiões avaliadas > 0 / elegíveis = 0``
        quando ainda existe discordância MLP↔TREPAN e estrutura OWL utilizável.
        """
        if not bool(self.error_focused_refinement):
            return False, "error_focused_refinement_disabled"
        if not self._semantic_structure_active():
            return False, "semantic_structure_unavailable"
        disagreement = float(profile.get("disagreement_rate", 0.0) or 0.0)
        if int(profile.get("disagreement_count", 0) or 0) <= 0 or disagreement <= 0.0:
            return False, "no_local_disagreement"
        if disagreement >= float(self.error_focus_min_disagreement):
            return True, "disagreement_threshold"

        # Fallback online: o TREPAN já percorre nós best-first. Logo, reservar as
        # primeiras regiões de erro abaixo do limiar equivale a dar prioridade
        # às regiões de maior alcance/erro visitadas pelo motor histórico.
        fallback_limit = min(max(0, int(self.error_focus_top_k)), max(0, int(self.error_focus_min_regions)))
        node_id = int(profile.get("node_id", -1))
        fallback_nodes = getattr(self, "_error_focus_fallback_nodes_", None)
        if fallback_nodes is None:
            fallback_nodes = set()
            self._error_focus_fallback_nodes_ = fallback_nodes
        already = node_id in fallback_nodes
        used = int(getattr(self, "_error_focus_fallback_used_", 0))
        if already or (fallback_limit > 0 and used < fallback_limit):
            if consume and not already:
                fallback_nodes.add(node_id)
                self._error_focus_fallback_used_ = used + 1
            return True, "top_k_error_region_fallback"
        return False, "below_disagreement_threshold"

    def _priority(self, node: _Node) -> float:
        if not self._ontology_effective():
            return TrepanOriginalClassifier._priority(self, node)
        base = TrepanOriginalClassifier._priority(self, node)
        if self._b2c_active() and not self.b2c_semantic_node_priority:
            return base                 # B2-C: ordem de expansão do Original preservada (fator experimental separado)
        if not bool(self.error_focused_refinement) or not self._semantic_structure_active():
            return base
        try:
            profile = self._build_error_region_profile(
                node, node.real_X, node.real_y, include_uncertainty=False,
            )
        except (ValueError, TypeError):
            return base
        eligible, _ = self._resolve_error_region_eligibility(profile, consume=False)
        if not eligible:
            return base
        # A prioridade histórica reach*(1-fidelity) permanece a base. A semântica
        # apenas redistribui a ordem entre regiões que já apresentam erro local.
        multiplier = 1.0 + max(0.0, float(self.error_focus_strength)) * float(profile["semantic_opportunity"])
        return float(base * multiplier)

    def _decision_sample(self, node: _Node):
        if not self._ontology_effective():
            return super()._decision_sample(node)
        if self._b2c_active():
            return self._decision_sample_b2c(node)
        X, y, model = super()._decision_sample(node)
        self._register_semantic_profile(node, X, y)
        return X, y, model

    # ---- B2-C: amostra-base do Original + amostra semântica adicional (rng_semantic separada) -----------------------------------
    def _b2c_semantic_rng_seed(self, node: _Node) -> np.random.SeedSequence:
        """Derivação documentada e estável da corrente ``rng_semantic``: SeedSequence([random_state, 0xB2C, node_id]). Não depende do dataset."""
        return np.random.SeedSequence([int(self.random_state), 0xB2C, int(max(0, node.node_id))])

    def _b2c_semantic_model(self, model: FeatureDistributionModel, node: _Node) -> FeatureDistributionModel:
        """Cópia superficial do modelo de distribuição com RNG PRÓPRIA: sorteios semânticos nunca avançam o estado do ``rng_original``."""
        import copy
        clone = copy.copy(model)
        clone._rng = np.random.default_rng(self._b2c_semantic_rng_seed(node))
        return clone

    def _decision_sample_b2c(self, node: _Node):
        ledger = getattr(self, "_ledger", None)
        self._b2c_base_draw = True
        if ledger is not None:
            ledger.tag = "original"
        try:
            Xo, yo, model = TrepanOriginalClassifier._decision_sample(self, node)    # exatamente o contexto-base do Original
        finally:
            self._b2c_base_draw = False
        n_base = int(len(Xo) - len(node.real_y))
        Xs, ys = Xo, yo
        if ledger is not None:
            ledger.tag = "semantic"
        try:
            if n_base > 0 and self.oracle_ is not None:
                sem_model = self._b2c_semantic_model(model, node)
                qx = self._draw_membership_queries(sem_model, n_base, node.constraints, node)   # caminho semântico (ativo/projetado/domínio)
                qy = np.asarray(self.oracle_.predict(qx))
                self.semantic_membership_queries_ += int(len(qx))
                node.stats["b2c_semantic_queries"] = int(len(qx))
                Xs = np.vstack([node.real_X, qx]) if len(node.real_X) else qx
                ys = np.concatenate([node.real_y, qy]) if len(node.real_y) else qy
            counts = np.asarray([(ys == c).sum() for c in self.classes_], dtype=int)
            sem_prediction = self.classes_[int(np.argmax(counts))] if len(ys) else node.prediction
            self._b2c_sem_X, self._b2c_sem_y = Xs, ys
            self._register_semantic_profile(node, Xs, ys, prediction_override=sem_prediction)
        finally:
            if ledger is not None:
                ledger.tag = "original"
        return Xo, yo, model

    def _register_semantic_profile(self, node: _Node, X, y, prediction_override=None) -> None:
        # O TREPAN Original actualiza a predição da folha com a maioria do
        # conjunto de decisão imediatamente depois deste hook. O perfil EFSR
        # usa a mesma predição para medir exactamente o erro que será refinado.
        if prediction_override is None:
            counts = np.asarray([(y == c).sum() for c in self.classes_], dtype=int)
            prediction_override = self.classes_[int(np.argmax(counts))] if len(y) else node.prediction
        profile = self._build_error_region_profile(
            node, X, y, include_uncertainty=True, prediction_override=prediction_override
        )
        eligible, reason = self._resolve_error_region_eligibility(profile, consume=True)
        profile["eligible_for_semantic_refinement"] = bool(eligible)
        profile["eligibility_reason"] = reason
        self._current_error_profile_ = profile
        # Gate de generalização local: estas são linhas REAIS do treino que
        # chegaram ao nó, nunca membership queries nem o teste externo.
        self._current_real_validation_X_ = np.asarray(node.real_X, dtype=float)
        self._current_real_validation_y_ = np.asarray(node.real_y)
        audit_profile = dict(profile)
        # Mantém o relatório compacto: scores completos ficam só no split audit.
        audit_profile["top_error_features"] = [
            int(i) for i in np.argsort(-np.asarray(profile["semantic_feature_scores"], dtype=float), kind="stable")
            if float(profile["semantic_feature_scores"][int(i)]) > 0.0
        ][: min(8, int(getattr(self, "n_features_in_", 0)))]
        audit_profile.pop("feature_error_scores", None)
        audit_profile.pop("semantic_feature_scores", None)
        audit_profile.pop("anchor_indices", None)
        self.error_region_audit_.append(audit_profile)

    def _evaluate_error_focused_intervention(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        data_test: Optional[MofNTest],
        semantic_test: Optional[MofNTest],
        disagreement_rate: float,
    ) -> dict:
        result = {
            "attempted": False,
            "accepted": False,
            "reason": "no_semantic_change",
            "data_only_local_fidelity": None,
            "semantic_local_fidelity": None,
            "local_fidelity_gain": 0.0,
            "required_local_fidelity_gain": float(self.error_focus_min_local_fidelity_gain),
            "data_only_real_fidelity": None,
            "semantic_real_fidelity": None,
            "real_fidelity_gain": None,
            "required_real_fidelity_gain": float(self.error_focus_min_real_fidelity_gain),
            "real_training_gate_evaluated": False,
            "fidelity_tolerance": float(self.error_focus_fidelity_tolerance),
            "semantic_coherence_gain": 0.0,
            "accepted_within_tolerance": False,
        }
        if semantic_test is None or self._test_signature(semantic_test) == self._test_signature(data_test):
            return result
        if not bool(self.error_focused_refinement):
            result["reason"] = "error_focused_refinement_disabled"
            return result

        profile = dict(getattr(self, "_current_error_profile_", {}) or {})
        eligible = bool(profile.get("eligible_for_semantic_refinement", False))
        if not eligible:
            # Compatibilidade para chamadas unitárias directas fora do ciclo do nó.
            eligible = float(disagreement_rate) >= float(self.error_focus_min_disagreement)
        if not eligible:
            result["reason"] = "region_below_disagreement_threshold"
            return result

        result["attempted"] = True
        if data_test is None:
            data_fidelity = float(1.0 - disagreement_rate)
        else:
            data_fidelity = split_surrogate_fidelity(y, data_test.evaluate(X), self.classes_)
        semantic_fidelity = split_surrogate_fidelity(y, semantic_test.evaluate(X), self.classes_)
        gain = float(semantic_fidelity - data_fidelity)
        coherence_gain = float(
            self._semantic_coherence(semantic_test) - self._semantic_coherence(data_test)
        )
        tolerance = max(0.0, float(self.error_focus_fidelity_tolerance))
        # A tolerância só compra coerência semântica: sem aumento de coerência
        # o gate continua estrito.
        tolerable = bool(tolerance > 0.0 and coherence_gain > 1e-12)
        result.update({
            "data_only_local_fidelity": float(data_fidelity),
            "semantic_local_fidelity": float(semantic_fidelity),
            "local_fidelity_gain": gain,
            "semantic_coherence_gain": coherence_gain,
        })
        within_tolerance = False
        if gain + 1e-12 < float(self.error_focus_min_local_fidelity_gain):
            if not (tolerable and gain + tolerance + 1e-12 >= 0.0):
                result["reason"] = "semantic_candidate_no_local_fidelity_gain"
                return result
            within_tolerance = True

        # Segundo gate: o candidato precisa também melhorar (ou pelo menos não
        # degradar quando não há resolução suficiente) as linhas REAIS de treino
        # que chegaram ao nó. Isto impede aceitar uma regra que só funciona nas
        # membership queries sintéticas.
        real_X = getattr(self, "_current_real_validation_X_", None)
        real_y = getattr(self, "_current_real_validation_y_", None)
        if real_X is not None and real_y is not None:
            real_X = np.asarray(real_X, dtype=float)
            real_y = np.asarray(real_y)
            if len(real_X) == len(real_y) and len(real_y) >= max(4, int(self.min_samples_leaf)):
                result["real_training_gate_evaluated"] = True
                if data_test is None:
                    # Sem split data-only, baseline local é a folha corrente.
                    counts = np.asarray([(real_y == c).sum() for c in self.classes_], dtype=int)
                    pred = self.classes_[int(np.argmax(counts))]
                    real_data_fidelity = float(np.mean(real_y == pred))
                else:
                    real_data_fidelity = split_surrogate_fidelity(
                        real_y, data_test.evaluate(real_X), self.classes_
                    )
                real_semantic_fidelity = split_surrogate_fidelity(
                    real_y, semantic_test.evaluate(real_X), self.classes_
                )
                real_gain = float(real_semantic_fidelity - real_data_fidelity)
                result.update({
                    "data_only_real_fidelity": float(real_data_fidelity),
                    "semantic_real_fidelity": float(real_semantic_fidelity),
                    "real_fidelity_gain": real_gain,
                })
                required_real = float(self.error_focus_min_real_fidelity_gain)
                # ``0`` significa ganho estritamente positivo. Uma perda real só
                # é aceite dentro da tolerância e com ganho de coerência OWL.
                real_ok = real_gain > 1e-12 if required_real <= 0.0 else real_gain + 1e-12 >= required_real
                if not real_ok:
                    if not (tolerable and real_gain + tolerance + 1e-12 >= 0.0):
                        result["reason"] = "semantic_candidate_fails_real_training_gate"
                        return result
                    within_tolerance = True

        result["accepted"] = True
        if within_tolerance:
            result["accepted_within_tolerance"] = True
            result["reason"] = "semantic_coherence_gain_within_fidelity_tolerance"
        else:
            result["reason"] = "semantic_candidate_improves_local_and_real_training_fidelity"
        return result

    def _feature_priority_scores(self, X: np.ndarray) -> np.ndarray:
        if not self._ontology_effective():
            return super()._feature_priority_scores(X)
        base = super()._feature_priority_scores(X)
        weights = getattr(self, "semantic_feature_weights_", None)
        if weights is None or np.allclose(weights, 1.0, atol=1e-12):
            return base
        # O expoente permite desligar/reduzir o viés sem mudar o algoritmo.
        factor = np.power(weights, float(self.semantic_gain_strength))
        return base * factor

    def _annotate_semantic_split_row(self, row: dict) -> None:
        """Torna cada split auditável: o que foi escolhido, qual a entidade OWL e porquê.

        ``base_score`` é o ganho estatístico antes do bónus; ``final_score`` o score
        usado na seleção. O score do TREPAN não tem termo de complexidade, por isso
        ``complexity_penalty`` é 0.0 e declarado como não modelado (não é inventado).
        """
        names = list(getattr(self, 'feature_names_in_', None) or [])
        features = [int(i) for i in row.get('semantic_features', [])]
        entities = getattr(self, 'semantic_feature_entities_', None)
        groups = getattr(self, 'semantic_feature_groups_', None) or []
        rel = getattr(self, 'semantic_relatedness_matrix_', None)
        row['base_score'] = float(row['information_gain'])
        row['final_score'] = float(row['selection_score'])
        row['complexity_penalty'] = 0.0
        row['complexity_penalty_note'] = 'not_modelled_in_trepan_selection_score'
        row['selected_feature'] = [names[i] for i in features if i < len(names)] or None
        row['ontology_entity'] = (
            [entities[i] for i in features if i < len(entities) and entities[i]] or None
            if entities is not None else None
        )
        reasons = []
        bonus = row['semantic_bonus']
        if features and abs(bonus) > 1e-12:
            reasons.append(f"bónus semântico {bonus:+.6f} sobre o ganho estatístico {row['base_score']:.6f}")
        used_groups = sorted({groups[i] for i in features if i < len(groups) and groups[i]})
        if used_groups:
            reasons.append("grupo(s) ontológico(s): " + ", ".join(used_groups))
        if rel is not None and len(features) >= 2:
            pair = [float(rel[a][b]) for k, a in enumerate(features) for b in features[k + 1:]
                    if a < len(rel) and b < len(rel)]
            if pair and max(pair) > 0:
                reasons.append(f"relatedness máxima entre literais {max(pair):.2f}")
        if row.get('decision_changed'):
            reasons.append("a decisão do split mudou face ao TREPAN data-only")
        elif row.get('decision_reinforced'):
            reasons.append("o vencedor data-only foi mantido e reforçado")
        row['semantic_reason'] = "; ".join(reasons) if reasons else "sem influência ontológica neste split"

    def _semantic_group_coherence_factor(self, test: Optional[MofNTest]) -> float:
        """Bónus genérico para regras m-of-n semanticamente coesas.

        Não usa nomes de datasets nem thresholds externos. Só considera os grupos
        fornecidos pela ontologia (por exemplo, subPropertyOf/domínio/classe pai).
        Testes univariados não recebem bónus de grupo.
        """
        if test is None or len(test.literals) < 2:
            return 1.0
        groups = getattr(self, 'semantic_feature_groups_', None)
        group_coherence = 0.0
        if groups:
            selected = [groups[lit.feature] for lit in test.literals if lit.feature < len(groups)]
            selected = [g for g in selected if g and str(g).lower() not in {'general', 'none'}]
            if len(selected) >= 2:
                counts = {}
                for group in selected:
                    counts[group] = counts.get(group, 0) + 1
                pair_total = len(selected) * (len(selected) - 1) / 2.0
                same_pairs = sum(count * (count - 1) / 2.0 for count in counts.values())
                group_coherence = float(same_pairs / pair_total) if pair_total else 0.0
        rel = getattr(self, "semantic_relatedness_matrix_", None)
        relation_values = []
        if rel is not None:
            indices = [int(lit.feature) for lit in test.literals]
            for i, left in enumerate(indices):
                for right in indices[i+1:]:
                    relation_values.append(float(rel[left, right]))
        relation_coherence = float(np.mean(relation_values)) if relation_values else 0.0
        coherence = max(group_coherence, relation_coherence)
        return float(1.0 + max(0.0, self.semantic_group_strength) * coherence)

    def _semantic_weight_for_test(self, test: Optional[MofNTest]) -> float:
        if test is None or not test.literals:
            return 1.0
        weights = getattr(self, "semantic_feature_weights_", None)
        if weights is None:
            return 1.0
        selected = [float(weights[lit.feature]) for lit in test.literals]
        # Média geométrica: evita um único literal de peso extremo dominar uma
        # regra m-of-n inteira, mas ainda recompensa regras semanticamente fortes.
        feature_factor = float(np.exp(np.mean(np.log(np.clip(selected, 1e-12, None)))))
        return float(feature_factor * self._semantic_group_coherence_factor(test))

    def _ontology_gain_active(self) -> bool:
        """Gain_Reloaded só substitui o IG quando a semântica está ligada.

        Com alpha=beta=0, OWL inactiva ou durante a busca data-only de
        referência, o score é exactamente o do TREPAN Original.
        """
        if getattr(self, "_semantic_search_disabled", False):
            return False
        if float(self.alpha) <= 0.0 and float(self.beta) <= 0.0:
            return False
        return self._semantic_structure_active()

    @staticmethod
    def _gain_ratio(information_gain: float, mask: np.ndarray) -> float:
        """GainRatio de Quinlan para a partição binária induzida pelo teste."""
        split_info = _entropy(np.asarray(mask, dtype=bool), np.asarray([False, True]))
        if split_info <= 1e-12:
            return 0.0
        return float(information_gain / split_info)

    def _normalized_information_gain(self, information_gain: float, y: np.ndarray) -> float:
        """IG/H(y) do nó: fracção da incerteza local removida, em [0, 1]."""
        node_entropy = _entropy(np.asarray(y), self.classes_)
        if node_entropy <= 1e-12:
            return 0.0
        return float(np.clip(information_gain / node_entropy, 0.0, 1.0))

    def _semantic_coherence(self, test: Optional[MofNTest]) -> float:
        """Coerência/interpretabilidade OWL de um teste, sem olhar para y.

        Média do suporte estrutural e da profundidade OWL dos literais, mais a
        coesão de grupo das regras m-of-n (0 para testes univariados).
        """
        if test is None or not test.literals:
            return 0.0
        support = self._semantic_support_vector()
        depths = np.asarray(getattr(self, "semantic_feature_depths_", np.zeros(len(support))), dtype=float)
        per_literal = []
        for lit in test.literals:
            s = float(support[lit.feature]) if lit.feature < len(support) else 0.0
            d = float(depths[lit.feature]) if lit.feature < len(depths) else 0.0
            per_literal.append(0.5 * (s + d))
        group_bonus = max(0.0, self._semantic_group_coherence_factor(test) - 1.0)
        return float(np.mean(per_literal) + group_bonus)

    def _test_feature_mean(self, values, test: Optional[MofNTest]) -> float:
        if test is None or not test.literals or values is None:
            return 0.0
        values = np.asarray(values, dtype=float)
        selected = [float(values[lit.feature]) for lit in test.literals if lit.feature < len(values)]
        return float(np.mean(selected)) if selected else 0.0

    def _onto_depth_for_test(self, test: Optional[MofNTest]) -> float:
        return self._test_feature_mean(getattr(self, "semantic_feature_depths_", None), test)

    def _error_coverage_for_test(self, test: Optional[MofNTest]) -> float:
        profile = getattr(self, "_current_error_profile_", None) or {}
        return self._test_feature_mean(profile.get("feature_error_scores"), test)

    def _split_selection_score(self, y, mask, test=None) -> float:
        if not self._ontology_effective():
            return super()._split_selection_score(y, mask, test)
        raw = super()._split_selection_score(y, mask, test)
        weight = self._semantic_weight_for_test(test)
        calibration = self.semantic_calibration
        if calibration is not None and not getattr(self, "_semantic_search_disabled", False) and self._semantic_structure_active():
            return self._calibrated_split_score(raw, weight, y, mask, test, calibration)
        if not self._ontology_gain_active():
            if np.isclose(weight, 1.0, atol=1e-12):
                return raw
            score = float(raw * (weight ** float(self.semantic_gain_strength)))
            if self.log_split_candidates:
                self._log_candidate(test, raw, raw, weight, 0.0, 0.0, score, "legacy_weight_only")
            return score
        # Um split sem informação nunca vence só pelos bónus ontológicos.
        if raw <= 1e-12:
            return 0.0
        if self.gain_criterion == "gain_ratio":
            base_gain = self._gain_ratio(raw, mask)
        elif self.gain_criterion == "normalized_information_gain":
            base_gain = self._normalized_information_gain(raw, y)
        else:
            base_gain = raw
        stat_gain = base_gain
        if not np.isclose(weight, 1.0, atol=1e-12):
            base_gain *= weight ** float(self.semantic_gain_strength)
        onto_depth = self._onto_depth_for_test(test)
        error_cov = self._error_coverage_for_test(test)
        gain_reloaded = base_gain + (self.alpha * onto_depth) + (self.beta * error_cov)
        if self.log_split_candidates:
            self._log_candidate(test, raw, stat_gain, weight, onto_depth, error_cov, float(gain_reloaded), "legacy")
        return float(gain_reloaded)

    # ---- B1: calibração do contributo semântico (EXPERIMENTAL) ------------------------------------------------------------
    def _calibrated_split_score(self, raw, weight, y, mask, test, calibration: str) -> float:
        """Score calibrado. ``s`` = componente estatística (ganho normalizado do nó, ∈[0,1]); ``m`` = contributo semântico líquido, definido
        para que o score histórico seja exatamente ``s + m``:  m = s·(w^k − 1) + α·OntoDepth + β·ErrorCoverage.

        * ``bounded``   : score = s + clip(m, −λ·s, +λ·s)            (λ = ``semantic_lambda_cap``; λ=0 ⇒ ranking puramente estatístico)
        * ``calibrated``: score = s + λ·s·tanh(m / (λ·s))             (saturação suave com o mesmo teto λ·s)
        * ``tiebreak``  : score = s + τ·clip(q, −1, 1), τ = c/√n_real, q = (w^k − 1) + α·OntoDepth + β·ErrorCoverage  (a semântica só reordena
          candidatos estatisticamente indistinguíveis: |Δs| < 2τ; n_real = linhas REAIS do nó, nunca as consultas sintéticas)
        """
        if raw <= 1e-12:
            return 0.0
        if self.gain_criterion == "gain_ratio":
            s = self._gain_ratio(raw, mask)
        elif self.gain_criterion == "normalized_information_gain":
            s = self._normalized_information_gain(raw, y)
        else:
            s = raw
        k = float(self.semantic_gain_strength)
        w_term = (weight ** k) - 1.0 if not np.isclose(weight, 1.0, atol=1e-12) else 0.0
        onto_depth = self._onto_depth_for_test(test)
        error_cov = self._error_coverage_for_test(test)
        sem_add = float(self.alpha) * onto_depth + float(self.beta) * error_cov
        lam = float(self.semantic_lambda_cap)
        if calibration == "bounded":
            m = s * w_term + sem_add
            score = s + float(np.clip(m, -lam * s, lam * s))
        elif calibration == "calibrated":
            m = s * w_term + sem_add
            score = s if lam <= 0.0 or s <= 0.0 else s + lam * s * float(np.tanh(m / (lam * s)))
        else:   # tiebreak
            ry = getattr(self, "_current_real_validation_y_", None)
            n_real = int(len(ry)) if ry is not None and len(ry) else int(len(y))
            tau = float(self.semantic_tie_se_multiplier) / math.sqrt(max(1, n_real))
            score = s + tau * float(np.clip(w_term + sem_add, -1.0, 1.0))
        if self.log_split_candidates:
            self._log_candidate(test, raw, s, weight, onto_depth, error_cov, float(score), calibration)
        return float(score)

    def candidate_log_summary(self) -> dict:
        """Sumário por nó do log de candidatos (exige ``log_split_candidates``): escalas reais e efeito do contributo semântico na ESCOLHA.

        Por nó (fase ``semantic``): ``stat_top`` = melhor componente estatística entre os candidatos; ``winner_stat`` = componente estatística do
        teste finalmente escolhido (``test`` do audit); ``regret`` = (stat_top − winner_stat)/stat_top; ``switched`` = o vencedor não é o melhor
        estatístico (decisão alterada pela semântica ou pelo gate EFSR); ``ratio_m_over_s`` = mediana de (score − s)/s dos candidatos do nó.
        """
        from collections import defaultdict
        log = getattr(self, "split_candidate_log_", None) or []
        by_node: dict = defaultdict(list)
        for r in log:
            if r["phase"] == "semantic" and r["stat_component"] > 1e-12:
                by_node[r["node_id"]].append(r)
        chosen = {int(row["node_id"]): str(row.get("test")) for row in (getattr(self, "split_audit_", None) or []) if "node_id" in row}
        nodes = []
        for nid, rows in sorted(by_node.items()):
            top = max(r["stat_component"] for r in rows)
            top_test = max(rows, key=lambda r: r["stat_component"])
            win_txt = chosen.get(nid)
            win = [r for r in rows if win_txt is not None and MofNTest(r["m"], tuple(Literal(f, t, g) for f, t, g in r["literals"])).text(self.feature_names_in_) == win_txt]
            winner_stat = max((r["stat_component"] for r in win), default=None)
            nodes.append({
                "node_id": nid, "n_candidates": len(rows), "stat_top": float(top),
                "winner_stat": None if winner_stat is None else float(winner_stat),
                "regret": None if winner_stat is None else float((top - winner_stat) / top),
                "switched": None if winner_stat is None else bool(winner_stat < top - 1e-12),
                "ratio_m_over_s_median": float(np.median([(r["score"] - r["stat_component"]) / r["stat_component"] for r in rows])),
                "top_stat_literals": top_test["literals"],
            })
        regrets = [n["regret"] for n in nodes if n["regret"] is not None]
        sw = [n["switched"] for n in nodes if n["switched"] is not None]
        return {
            "nodes": nodes, "candidates_logged": len(log), "nodes_with_candidates": len(nodes),
            "switch_rate": float(np.mean(sw)) if sw else None,
            "regret_median": float(np.median(regrets)) if regrets else None, "regret_max": float(np.max(regrets)) if regrets else None,
            "ratio_m_over_s_median": float(np.median([n["ratio_m_over_s_median"] for n in nodes])) if nodes else None,
        }

    def _log_candidate(self, test, raw, stat_gain, weight, onto_depth, error_cov, score, variant) -> None:
        """Registo de TODOS os candidatos pontuados (instrumentação pura)."""
        profile = getattr(self, "_current_error_profile_", None) or {}
        lits = [] if test is None else [(int(l.feature), float(l.threshold), bool(l.greater)) for l in test.literals]
        self.split_candidate_log_.append({
            "node_id": int(profile.get("node_id", -1)), "m": None if test is None else int(test.m), "n": len(lits), "literals": lits,
            "raw_information_gain": float(raw), "stat_component": float(stat_gain), "weight": float(weight), "onto_depth": float(onto_depth),
            "error_coverage": float(error_cov), "score": float(score), "variant": variant,
            "lambda_cap": float(self.semantic_lambda_cap), "phase": "data_only" if getattr(self, "_semantic_search_disabled", False) else "semantic",
        })

    def _semantic_structure_active(self) -> bool:
        if getattr(self, "_semantic_search_disabled", False):
            return False
        caching = bool(getattr(self, "_semantic_structure_cache_enabled_", False))
        if caching and getattr(self, "_semantic_structure_cache_", None) is not None:
            return bool(self._semantic_structure_cache_)
        active = self._compute_semantic_structure_active()
        if caching:
            self._semantic_structure_cache_ = active
        return active

    def _compute_semantic_structure_active(self) -> bool:
        groups = list(getattr(self, "semantic_feature_groups_", []) or [])
        repeated = any(g and groups.count(g) > 1 for g in set(groups) if g)
        rel = getattr(self, "semantic_relatedness_matrix_", None)
        related = False
        if rel is not None:
            off = np.asarray(rel, dtype=float).copy()
            np.fill_diagonal(off, 0.0)
            related = bool(np.any(off > float(self.semantic_relation_threshold)))
        weights = np.asarray(getattr(self, "semantic_feature_weights_", []), dtype=float)
        weighted = bool(weights.size and not np.allclose(weights, 1.0, atol=1e-12))
        depths = np.asarray(getattr(self, "semantic_feature_depths_", []), dtype=float)
        deep = bool(float(self.alpha) > 0.0 and depths.size and np.any(depths > 0.0))
        return bool(repeated or related or weighted or deep)

    def _ontology_guided_candidates(self, X: np.ndarray, y: np.ndarray) -> list[tuple[float, MofNTest]]:
        """Gera candidatos m-of-n que o beam estatístico pode não visitar.

        A geração é totalmente agnóstica ao dataset: utiliza apenas grupos e
        relações fornecidos pelo grafo OWL. Os thresholds continuam a vir dos
        dados de treino do nó e cada candidato é validado pelo mesmo score do
        TREPAN histórico.
        """
        budget = max(0, int(self.semantic_candidate_budget))
        if budget == 0 or not self._semantic_structure_active() or self.max_n < 2:
            return []
        profile = dict(getattr(self, "_current_error_profile_", {}) or {})
        if bool(self.error_focused_refinement):
            eligible = bool(profile.get("eligible_for_semantic_refinement", False))
            if not eligible:
                eligible, _ = self._resolve_error_region_eligibility(profile, consume=False)
            if not eligible:
                return []
        ranked = self._candidate_literals(X, y)
        if not ranked:
            return []
        literal_options_by_feature: dict[int, list] = {}
        for _score, lit in ranked:
            bucket = literal_options_by_feature.setdefault(int(lit.feature), [])
            # Preserva no máximo uma opção por orientação. Information Gain de
            # partições complementares pode empatar; excluir uma orientação pode
            # tornar impossível descobrir o m-of-n correto.
            if not any(bool(existing.greater) == bool(lit.greater) for existing in bucket):
                bucket.append(lit)
            if len(bucket) >= 2:
                continue
        focus_scores = np.asarray(profile.get("semantic_feature_scores", np.zeros(X.shape[1])), dtype=float)
        if len(focus_scores) != X.shape[1]:
            focus_scores = np.zeros(X.shape[1], dtype=float)
        features = sorted(literal_options_by_feature, key=lambda f: (-float(focus_scores[f]), int(f)))
        # Limite de trabalho independente do dataset: o EFSR deve procurar mais
        # fundo nas features relacionadas com o erro, não fazer explosão combinatória.
        feature_limit = min(len(features), max(4, min(8, int(self.max_features_per_node))))
        features = features[:feature_limit]
        if len(features) < 2:
            return []

        # Componentes semânticos: mesmo grupo OU relação OWL acima do limiar.
        parent = {f: f for f in features}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        groups = list(getattr(self, "semantic_feature_groups_", []) or [])
        by_group = {}
        for f in features:
            g = groups[f] if f < len(groups) else None
            if g and str(g).lower() not in {"general", "none"}:
                by_group.setdefault(str(g), []).append(f)
        for members in by_group.values():
            for a, b in zip(members, members[1:]):
                union(a, b)
        rel = getattr(self, "semantic_relatedness_matrix_", None)
        if rel is not None:
            rel = np.asarray(rel, dtype=float)
            threshold = float(self.semantic_relation_threshold)
            for pos, a in enumerate(features):
                for b in features[pos + 1:]:
                    if rel[a, b] >= threshold:
                        union(a, b)
        components = {}
        for f in features:
            components.setdefault(find(f), []).append(f)

        generated = []
        seen = set()
        for members in sorted(components.values(), key=lambda xs: (-len(xs), xs)):
            if len(members) < 2:
                continue
            max_k = min(int(self.max_n), len(members))
            for k in range(2, max_k + 1):
                for combo in itertools.combinations(members, k):
                    option_sets = [literal_options_by_feature[f] for f in combo]
                    for chosen_literals in itertools.product(*option_sets):
                        literals = tuple(chosen_literals)
                        # Maioria e conjunção são as hipóteses m-of-n semanticamente
                        # mais interpretáveis; 1-of-n é incluído para disjunções.
                        m_values = sorted({1, int(math.ceil(k / 2.0)), k})
                        for m in m_values:
                            test = MofNTest(int(m), literals)
                            sig = self._test_signature(test)
                            if sig in seen:
                                continue
                            seen.add(sig)
                            mask = test.evaluate(X)
                            if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
                                continue
                            score = float(self._split_selection_score(y, mask, test))
                            if bool(self.error_focused_refinement) and len(focus_scores):
                                local_focus = float(np.mean([focus_scores[f] for f in combo]))
                                score *= 1.0 + max(0.0, float(self.error_focus_strength)) * local_focus * float(profile.get("disagreement_rate", 0.0))
                            generated.append((score, test))
        generated.sort(key=lambda item: (-item[0], len(item[1].literals), item[1].m, self._test_signature(item[1])))
        selected = generated[:budget]
        self.semantic_candidate_generation_count_ += len(selected)
        return selected

    @staticmethod
    def _test_signature(test: Optional[MofNTest]):
        if test is None:
            return None
        return (
            int(test.m),
            tuple(
                (int(lit.feature), round(float(lit.threshold), 12), bool(lit.greater))
                for lit in test.literals
            ),
        )

    def _best_mofn(self, X: np.ndarray, y: np.ndarray):
        """Selecciona split Reloaded com gate local de fidelidade EFSR.

        A ontologia pode propor/reordenar candidatos, mas uma mudança relativamente
        ao TREPAN Original só é aceite numa região de erro elegível e quando a
        fidelidade local ao MLP melhora pelo mínimo configurado. Isto impede que
        coerência semântica, isoladamente, force uma decisão pior.
        """
        if not self._ontology_effective():
            self._semantic_candidates_last_ = 0
            return super()._best_mofn(X, y)
        # B2-C: ``X, y`` = amostra-base (contexto do Original); ``Xs, ys`` = amostra semântica adicional. Fora de B2-C ambas coincidem.
        b2c = self._b2c_active()
        Xs, ys = (self._b2c_sem_X, self._b2c_sem_y) if (b2c and self._b2c_sem_X is not None) else (X, y)
        semantic_result = super()._best_mofn(Xs, ys)
        generated = self._ontology_guided_candidates(Xs, ys)
        self._semantic_candidates_last_ = len(generated)
        if generated:
            best_generated_score, best_generated_test = generated[0]
            if semantic_result is None or best_generated_score > float(semantic_result[1]) + 1e-12:
                semantic_result = (best_generated_test, float(best_generated_score))

        weights = np.asarray(
            getattr(self, 'semantic_feature_weights_', np.ones(X.shape[1])),
            dtype=float,
        )
        groups = list(getattr(self, 'semantic_feature_groups_', []) or [])
        group_active = len({g for g in groups if g and groups.count(g) > 1}) > 0
        rel = getattr(self, 'semantic_relatedness_matrix_', None)
        relation_active = False
        if rel is not None:
            off = np.asarray(rel, dtype=float).copy()
            np.fill_diagonal(off, 0.0)
            relation_active = bool(np.any(off > float(self.semantic_relation_threshold)))
        neutral = (
            np.allclose(weights, 1.0, atol=1e-12)
            and not group_active and not relation_active
            and not self._ontology_gain_active()
        )
        if neutral and not b2c:
            data_result = semantic_result
        else:
            saved_groups = list(getattr(self, 'semantic_feature_groups_', []))
            saved_rel = getattr(self, 'semantic_relatedness_matrix_', None)
            self.semantic_feature_weights_ = np.ones_like(weights)
            self.semantic_feature_groups_ = [None] * len(weights)
            self.semantic_relatedness_matrix_ = None
            self._semantic_search_disabled = True
            try:
                data_result = super()._best_mofn(X, y)
            finally:
                self._semantic_search_disabled = False
                self.semantic_feature_weights_ = weights
                self.semantic_feature_groups_ = saved_groups
                self.semantic_relatedness_matrix_ = saved_rel

        proposed_semantic_test = semantic_result[0] if semantic_result is not None else None
        proposed_semantic_score = float(semantic_result[1]) if semantic_result is not None else None
        data_test = data_result[0] if data_result is not None else None
        data_score = float(data_result[1]) if data_result is not None else None

        profile = dict(getattr(self, '_current_error_profile_', {}) or {})
        disagreement_rate = float(profile.get('disagreement_rate', 0.0))
        intervention = self._evaluate_error_focused_intervention(
            Xs, ys,
            data_test=data_test,
            semantic_test=proposed_semantic_test,
            disagreement_rate=disagreement_rate,
        )
        if b2c and intervention['accepted'] and self._test_signature(proposed_semantic_test) != self._test_signature(data_test):
            ok, why = self._b2c_valid_in_base_context(proposed_semantic_test, X, y)
            if not ok:       # a proposta semântica tem de ser também uma divisão válida no contexto-base (mesmos critérios do laço do Original)
                intervention = dict(intervention)
                intervention['accepted'] = False
                intervention['accepted_within_tolerance'] = False
                intervention['reason'] = why
        if intervention['attempted']:
            self.error_focused_interventions_attempted_ += 1
            if intervention['accepted']:
                self.error_focused_interventions_accepted_ += 1
            else:
                self.error_focused_interventions_rejected_ += 1

        # Se a semântica propôs uma decisão diferente mas não provou ganho local,
        # volta exactamente ao vencedor data-only do mesmo motor TREPAN.
        proposed_changed = self._test_signature(proposed_semantic_test) != self._test_signature(data_test)
        if proposed_changed and not intervention['accepted']:
            final_result = data_result
        else:
            final_result = semantic_result

        final_test = final_result[0] if final_result is not None else None
        final_score = float(final_result[1]) if final_result is not None else None
        final_changed = self._test_signature(final_test) != self._test_signature(data_test)

        if proposed_semantic_test is not None:
            proposed_mask = proposed_semantic_test.evaluate(Xs)
            proposed_raw = float(
                TrepanOriginalClassifier._split_selection_score(
                    self, ys, proposed_mask, proposed_semantic_test
                )
            )
            semantic_weight = float(self._semantic_weight_for_test(proposed_semantic_test))
            group_factor = float(self._semantic_group_coherence_factor(proposed_semantic_test))
            semantic_features = sorted({
                int(lit.feature)
                for lit in proposed_semantic_test.literals
                if (
                    not np.isclose(float(weights[lit.feature]), 1.0, atol=1e-12)
                    or not np.isclose(group_factor, 1.0, atol=1e-12)
                    or float(profile.get('semantic_feature_scores', [0.0] * len(weights))[lit.feature]) > 0.0
                )
            })
        else:
            proposed_raw = None
            semantic_weight = 1.0
            group_factor = 1.0
            semantic_features = []

        data_raw = None
        if data_test is not None:
            data_raw = float(
                TrepanOriginalClassifier._split_selection_score(
                    self, y, data_test.evaluate(X), data_test
                )
            )

        semantic_bonus = (
            proposed_semantic_score - proposed_raw
            if proposed_semantic_score is not None and proposed_raw is not None
            else 0.0
        )
        reinforced = (
            bool(semantic_features)
            and not proposed_changed
            and not np.isclose(semantic_bonus, 0.0, atol=1e-12)
        )
        focus_scores = list(profile.get('semantic_feature_scores', []) or [])
        top_error_features = [
            int(i) for i in np.argsort(-np.asarray(focus_scores, dtype=float), kind='stable')
            if i < len(focus_scores) and float(focus_scores[int(i)]) > 0.0
        ][:8] if focus_scores else []

        self._pending_semantic_decision_audit = {
            'data_only_test': data_test.text(self.feature_names_in_) if data_test is not None else None,
            'data_only_information_gain': data_raw,
            'data_only_selection_score': data_score,
            'semantic_test': proposed_semantic_test.text(self.feature_names_in_) if proposed_semantic_test is not None else None,
            'semantic_information_gain': proposed_raw,
            'semantic_selection_score': proposed_semantic_score,
            'selected_test_after_error_gate': final_test.text(self.feature_names_in_) if final_test is not None else None,
            'selected_score_after_error_gate': final_score,
            'semantic_bonus': float(semantic_bonus),
            'gain_reloaded_active': bool(self._ontology_gain_active()),
            'gain_ratio': (
                float(self._gain_ratio(proposed_raw, proposed_semantic_test.evaluate(Xs)))
                if proposed_semantic_test is not None and proposed_raw is not None else None
            ),
            'onto_depth': float(self._onto_depth_for_test(proposed_semantic_test)),
            'error_coverage': float(self._error_coverage_for_test(proposed_semantic_test)),
            'alpha': float(self.alpha),
            'beta': float(self.beta),
            'gain_criterion': self.gain_criterion,
            'semantic_weight': semantic_weight,
            'semantic_group_factor': float(group_factor),
            'semantic_groups': sorted({groups[i] for i in semantic_features if i < len(groups) and groups[i]}),
            'semantic_features': semantic_features,
            'proposed_decision_changed': bool(proposed_changed),
            'decision_changed': bool(final_changed),
            'decision_reinforced': bool(reinforced),
            'ontology_influenced': bool(final_changed or reinforced),
            'query_projection_active': bool(getattr(self, '_query_projector_fit', None) is not None),
            'semantic_candidates_evaluated': int(getattr(self, '_semantic_candidates_last_', 0)),
            'error_focused_refinement': bool(self.error_focused_refinement),
            'error_region_disagreement_rate': disagreement_rate,
            'error_region_semantic_opportunity': float(profile.get('semantic_opportunity', 0.0)),
            'error_region_mean_uncertainty': float(profile.get('mean_uncertainty', 0.0)),
            'error_region_disagreement_uncertainty': float(profile.get('disagreement_uncertainty', 0.0)),
            'top_error_semantic_features': top_error_features,
            'error_focused_intervention_attempted': bool(intervention['attempted']),
            'error_focused_intervention_accepted': bool(intervention['accepted']),
            'error_focused_intervention_reason': intervention['reason'],
            'data_only_local_fidelity': intervention['data_only_local_fidelity'],
            'semantic_local_fidelity': intervention['semantic_local_fidelity'],
            'local_fidelity_gain': float(intervention['local_fidelity_gain']),
            'required_local_fidelity_gain': float(intervention['required_local_fidelity_gain']),
            'data_only_real_fidelity': intervention.get('data_only_real_fidelity'),
            'semantic_real_fidelity': intervention.get('semantic_real_fidelity'),
            'real_fidelity_gain': intervention.get('real_fidelity_gain'),
            'required_real_fidelity_gain': float(intervention.get('required_real_fidelity_gain', self.error_focus_min_real_fidelity_gain)),
            'real_training_gate_evaluated': bool(intervention.get('real_training_gate_evaluated', False)),
            'fidelity_tolerance': float(intervention.get('fidelity_tolerance', 0.0)),
            'semantic_coherence_gain': float(intervention.get('semantic_coherence_gain', 0.0)),
            'error_focused_accepted_within_tolerance': bool(intervention.get('accepted_within_tolerance', False)),
            'error_region_eligibility_reason': profile.get('eligibility_reason'),
        }
        if b2c:
            self._pending_semantic_decision_audit.update(
                self._b2c_record(profile, X, y, Xs, ys, data_test, proposed_semantic_test, final_test, intervention,
                                 proposed_changed, proposed_semantic_score, data_raw)
            )
        return final_result

    def _split_audit_metadata(self, test: MofNTest) -> dict:
        if not self._ontology_effective():
            return super()._split_audit_metadata(test)
        weight = self._semantic_weight_for_test(test)
        weights = getattr(self, "semantic_feature_weights_", np.ones(self.n_features_in_))
        weight_features = {
            int(lit.feature)
            for lit in test.literals
            if not np.isclose(float(weights[lit.feature]), 1.0, atol=1e-12)
        }
        decision = dict(getattr(self, '_pending_semantic_decision_audit', {}) or {})
        # Não perder features registadas pelo EFSR/grupo semântico quando os pesos
        # individuais são todos 1.0 (caso comum com matching 100%).
        semantic_features = sorted(weight_features.union(
            int(v) for v in (decision.get('semantic_features') or [])
        ))
        groups = list(getattr(self, 'semantic_feature_groups_', []) or [])
        final_group_factor = float(self._semantic_group_coherence_factor(test))
        region_eligible = (
            not bool(self.error_focused_refinement)
            or float(decision.get('error_region_disagreement_rate', 1.0)) >= float(self.error_focus_min_disagreement)
        )
        reinforced_final = bool(
            region_eligible
            and not bool(decision.get('decision_changed'))
            and (
                not np.isclose(float(weight), 1.0, atol=1e-12)
                or not np.isclose(final_group_factor, 1.0, atol=1e-12)
            )
        )
        if reinforced_final:
            decision['decision_reinforced'] = True
            decision['ontology_influenced'] = True
            semantic_features = sorted(set(semantic_features).union(int(lit.feature) for lit in test.literals))
        decision.update({
            "semantic_weight": float(weight),
            "semantic_group_factor": final_group_factor,
            "semantic_groups": sorted({groups[i] for i in semantic_features if i < len(groups) and groups[i]}),
            "semantic_features": semantic_features,
            "reloaded_extension": bool(semantic_features) or bool(decision.get('query_projection_active')),
        })
        self._pending_semantic_decision_audit = {}
        return decision

    def _b2c_valid_in_base_context(self, test: MofNTest, X: np.ndarray, y: np.ndarray) -> tuple[bool, str]:
        """Critérios que o laço do Original aplica à divisão escolhida, avaliados na amostra-base (ganho mínimo e tamanho mínimo de folha)."""
        mask = test.evaluate(X)
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return False, "semantic_invalid_in_base_context_min_samples_leaf"
        raw = TrepanOriginalClassifier._split_selection_score(self, y, mask, test)
        if float(raw) < float(self.min_gain):
            return False, "semantic_invalid_in_base_context_min_gain"
        return True, ""

    def _b2c_record(self, profile, X, y, Xs, ys, data_test, semantic_test, final_test, intervention, proposed_changed,
                    semantic_score, data_raw) -> dict:
        """Auditoria B2-C por nó (observacional). Classes de contexto (NUNCA equivalência estrutural global):
        * ``in_context_statistical_fallback``: o fallback é a escolha estatística (pesquisa do Original) no contexto-base ATUAL.
        * ``original_reproduction_conditions_met``: condições NECESSÁRIAS para que esta escolha seja a do Original isolado na mesma
          representação (nenhuma divisão semântica aceite antes, ordem de expansão do Original, corrente-base intacta); só o teste
          diferencial contra o Original prova a reprodução.
        * ``context_diverged_not_reproducible``: já houve divergência da árvore (ou prioridade semântica ligada) ⇒ a reprodução não é presumível."""
        node_id = int(profile.get("node_id", -1))
        accepted = bool(intervention["accepted"]) and bool(proposed_changed)
        fallback_used = bool(proposed_changed and not intervention["accepted"])
        prefix_identical = self._b2c_first_divergence_order_ is None
        order_preserved = not bool(self.b2c_semantic_node_priority)
        if prefix_identical and order_preserved:
            context_class = "original_reproduction_conditions_met"
        else:
            context_class = "context_diverged_not_reproducible"
        final_sig = self._test_signature(final_test)
        data_sig = self._test_signature(data_test)
        if final_sig != data_sig and self._b2c_first_divergence_order_ is None:
            self._b2c_first_divergence_order_ = node_id
        if accepted:
            self._b2c_semantic_tests_.add(final_sig)
        if not proposed_changed:
            chosen = "identical_candidates" if data_test is not None else "none"
        else:
            chosen = "semantic" if accepted else ("statistical" if data_test is not None else "none")
        row = {
            "node_id": node_id,
            "candidate_ids": {"statistical": "stat:" + (data_test.text(self.feature_names_in_) if data_test is not None else "none"),
                              "semantic": "sem:" + (semantic_test.text(self.feature_names_in_) if semantic_test is not None else "none")},
            "candidate_type": "semantic_scored_or_guided" if semantic_test is not None else "none",
            "ontology_state": getattr(self, "ontology_state_", "S3"),
            "statistical_gain": data_raw,
            "semantic_score": semantic_score,
            "local_fidelity_gain": float(intervention["local_fidelity_gain"]),
            "validated_fidelity_gain": None,                # B2-A (não implementado)
            "uncertainty_estimate": None,                   # B2-B (não implementado)
            "effective_sample_count": int(len(X)),
            "base_sample_size": int(len(X)), "semantic_sample_size": int(len(Xs)),
            "oracle_query_count": {"base": int(len(X) - len(self._current_real_validation_y_)),
                                   "semantic": int(len(Xs) - len(self._current_real_validation_y_))},
            "accepted": accepted,
            "rejection_reason": None if (accepted or not proposed_changed) else intervention["reason"],
            "fallback_used": fallback_used,
            "chosen_candidate_id": chosen,
            "tree_complexity_delta": (int(len(final_test.literals)) - int(len(data_test.literals))) if (final_test is not None and data_test is not None) else 0,
            "context_class": context_class,
            "prefix_identical_so_far": bool(prefix_identical),
            "base_stream_preserved": True,
            "node_priority": "original" if order_preserved else "semantic",
            "reference_representation": "same tree space as this fit (reference = Original fitted on the identical representation)",
        }
        self.b2c_audit_.append(row)
        return {"b2c": row}

    def semantic_audit_summary(self) -> dict:
        """Resumo auditável do impacto ontológico nas decisões da árvore."""
        rows = list(getattr(self, 'semantic_split_audit_', []) or [])
        evaluated = len(rows)
        changed = sum(bool(row.get('decision_changed')) for row in rows)
        reinforced = sum(bool(row.get('decision_reinforced')) for row in rows)
        influenced = sum(bool(row.get('ontology_influenced')) for row in rows)
        bonuses = [
            float(row.get('semantic_bonus', 0.0) or 0.0)
            for row in rows
            if row.get('semantic_bonus') is not None
        ]
        return {
            'evaluated_splits': int(evaluated),
            'ontology_influenced_splits': int(influenced),
            'semantic_changed_splits': int(changed),
            'semantic_reinforced_splits': int(reinforced),
            'ontology_usage_rate': float(influenced / evaluated) if evaluated else 0.0,
            'semantic_decision_impact': float(changed / evaluated) if evaluated else 0.0,
            'mean_semantic_bonus': float(np.mean(bonuses)) if bonuses else 0.0,
            'query_projection_active': bool(getattr(self, 'semantic_projection_batches_', 0) > 0),
            'semantic_projection_batches': int(getattr(self, 'semantic_projection_batches_', 0)),
            'semantic_projected_query_count': int(getattr(self, 'semantic_projected_query_count_', 0)),
            'semantic_candidate_generation_count': int(getattr(self, 'semantic_candidate_generation_count_', 0)),
            'active_query_batches': int(getattr(self, 'semantic_active_query_batches_', 0)),
            'active_query_selected': int(getattr(self, 'semantic_active_query_selected_', 0)),
            'error_regions_evaluated': int(len(getattr(self, 'error_region_audit_', []) or [])),
            'error_regions_eligible': int(sum(bool(row.get('eligible_for_semantic_refinement')) for row in (getattr(self, 'error_region_audit_', []) or []))),
            'error_focused_interventions_attempted': int(getattr(self, 'error_focused_interventions_attempted_', 0)),
            'error_focused_interventions_accepted': int(getattr(self, 'error_focused_interventions_accepted_', 0)),
            'error_focused_interventions_rejected': int(getattr(self, 'error_focused_interventions_rejected_', 0)),
            'error_focused_query_batches': int(getattr(self, 'error_focused_query_batches_', 0)),
            'error_focused_query_selected': int(getattr(self, 'error_focused_query_selected_', 0)),
            'probability_batches_reused_for_uncertainty': int(getattr(self, 'probability_batches_reused_for_uncertainty_', 0)),
            'probability_rows_reused_for_uncertainty': int(getattr(self, 'probability_rows_reused_for_uncertainty_', 0)),
            'error_focus_min_disagreement': float(self.error_focus_min_disagreement),
            'error_focus_min_local_fidelity_gain': float(self.error_focus_min_local_fidelity_gain),
            'error_focus_min_real_fidelity_gain': float(self.error_focus_min_real_fidelity_gain),
            'error_focus_top_k': int(self.error_focus_top_k),
            'error_focus_min_regions': int(self.error_focus_min_regions),
            'error_focus_top_k_fallback_used': int(getattr(self, '_error_focus_fallback_used_', 0)),
            'semantic_effect_mirror_applied': bool(getattr(self, 'semantic_effect_mirror_applied_', False)),
            'semantic_effect_mirror_reason': getattr(self, 'semantic_effect_mirror_reason_', None),
            'ontology_active': bool(getattr(self, 'ontology_active_', False)),
            'alpha': float(self.alpha),
            'beta': float(self.beta),
            'gain_criterion': self.gain_criterion,
            'error_focus_fidelity_tolerance': float(self.error_focus_fidelity_tolerance),
            'error_focused_accepted_within_tolerance': int(sum(
                bool(row.get('error_focused_accepted_within_tolerance'))
                for row in (getattr(self, 'semantic_split_audit_', []) or [])
            )),
            'semantic_query_projection': dict(getattr(self, 'semantic_query_projection_audit_', {}) or {}),
            'ontology_sampling': dict(getattr(self, 'ontology_sampling_audit_', {}) or {}),
        }

    def _draw_membership_queries(
        self,
        model: FeatureDistributionModel,
        n: int,
        constraints: ConstraintSet,
        node: _Node,
    ) -> np.ndarray:
        if not self._ontology_effective() or getattr(self, "_b2c_base_draw", False):
            return super()._draw_membership_queries(model, n, constraints, node)     # contexto-base: sorteio do Original, sem hooks semânticos
        if n <= 0:
            return np.empty((0, model.n_features_in_), dtype=float)
        domain = getattr(self, "_ontology_domain_fit", None)
        if domain is not None:
            # Toda a query passa pela validação OWL antes do oráculo. Com
            # projecção activa já é válida por construção; sem ela, sobre-amostra
            # e o fallback de constrain_synthetic_samples completa o lote.
            projected = bool(self.semantic_query_projection_audit_.get("enabled"))
            pool_size = int(n) if projected else int(n) * 2
            pool = self._draw_membership_queries_unfiltered(model, pool_size, constraints, node)
            kept, audit = constrain_synthetic_samples(
                pool, min_samples=n, ontology_graph=domain,
                feature_names=self.feature_names_in_,
            )
            stats = self.ontology_sampling_audit_
            stats["batches"] += 1
            stats["candidates"] += int(audit["candidates"])
            stats["rejected"] += int(audit["rejected"])
            stats["fallback_filled"] += int(audit["fallback_filled"])
            return kept[:n]
        return self._draw_membership_queries_unfiltered(model, n, constraints, node)

    def _draw_membership_queries_unfiltered(
        self,
        model: FeatureDistributionModel,
        n: int,
        constraints: ConstraintSet,
        node: _Node,
    ) -> np.ndarray:
        projector = getattr(self, "_query_projector_fit", None)
        profile = self._build_error_region_profile(
            node, node.real_X, node.real_y, include_uncertainty=True,
        )
        error_active = False
        if self.error_focused_refinement and self._semantic_structure_active():
            error_active, _ = self._resolve_error_region_eligibility(profile, consume=False)
        semantic_active = bool(
            self._semantic_structure_active()
            and float(self.semantic_active_query_fraction) > 0.0
            and int(self.semantic_active_pool_multiplier) > 1
        )
        # Com EFSR ligado, active sampling semântico só acontece onde há erro.
        active = bool(semantic_active and (error_active or not self.error_focused_refinement))
        if not active and projector is None:
            return super()._draw_membership_queries(model, n, constraints, node)

        target_pool = max(int(n), int(n) * max(1, int(self.semantic_active_pool_multiplier))) if active else int(n)
        pool = []
        remaining = target_pool
        for _ in range(180):
            if remaining <= 0:
                break
            raw = model._draw_unconstrained(max(64, remaining * 5))
            if projector is not None:
                projected = np.asarray(projector(raw), dtype=float)
                if projected.shape != raw.shape:
                    raise ValueError(
                        "query_projector alterou o schema das membership queries: "
                        f"{raw.shape} -> {projected.shape}."
                    )
                if not np.isfinite(projected).all():
                    raise ValueError("query_projector produziu NaN ou infinito.")
                self.semantic_projection_batches_ += 1
                self.semantic_projected_query_count_ += len(projected)
                raw = projected
            keep = raw[constraints.accepts(raw)]
            if len(keep):
                take = keep[:remaining]
                pool.append(take)
                remaining -= len(take)
        if remaining > 0:
            raise RuntimeError(
                "Não foi possível gerar membership queries coerentes que satisfaçam "
                "as restrições do nó."
            )
        candidates = np.vstack(pool)[:target_pool]
        if not active or len(candidates) <= n:
            return candidates[:n]

        reference_X = np.asarray(node.real_X, dtype=float)
        reference_y = np.asarray(node.real_y)
        ranked = self._candidate_literals(reference_X, reference_y) if len(reference_X) >= 4 else []
        boundary = np.zeros(len(candidates), dtype=float)
        scales = np.nanstd(reference_X, axis=0) if len(reference_X) else np.ones(candidates.shape[1])
        scales = np.where(scales > 1e-9, scales, 1.0)
        for _score, lit in ranked[: max(6, min(24, self.max_features_per_node * 2))]:
            distance = np.abs(candidates[:, lit.feature] - float(lit.threshold)) / scales[lit.feature]
            sem = float(self.semantic_feature_weights_[lit.feature])
            rel = getattr(self, 'semantic_relatedness_matrix_', None)
            if rel is not None:
                off = np.asarray(rel[lit.feature], dtype=float).copy()
                if lit.feature < len(off):
                    off[lit.feature] = 0.0
                sem *= 1.0 + float(np.mean(off))
            boundary = np.maximum(boundary, np.exp(-distance) * sem)
        lo, hi = float(np.min(boundary)), float(np.max(boundary))
        if hi > lo:
            boundary = (boundary - lo) / (hi - lo)

        combined = boundary.copy()
        if error_active:
            focus = np.asarray(profile.get('semantic_feature_scores', []), dtype=float)
            top_features = [
                int(i) for i in np.argsort(-focus, kind='stable')
                if i < len(focus) and float(focus[int(i)]) > 0.0
            ][: max(2, min(int(self.max_features_per_node), 12))]
            anchor_indices = [int(i) for i in profile.get('anchor_indices', []) if int(i) < len(reference_X)]
            anchors = reference_X[anchor_indices] if anchor_indices else np.empty((0, reference_X.shape[1]))
            anchor_similarity = nearest_anchor_similarity(
                candidates,
                anchors,
                feature_indices=top_features,
                scales=scales,
            )
            # A incerteza serve para tornar os próprios anchors mais importantes;
            # candidatos nunca são consultados ao MLP antes da selecção.
            known_uncertainty, _ = self._known_row_uncertainty(reference_X)
            if anchor_indices and len(known_uncertainty):
                anchor_uncertainty = float(np.mean(known_uncertainty[anchor_indices]))
            else:
                anchor_uncertainty = 0.0
            error_signal = anchor_similarity * (
                1.0 + float(self.error_focus_uncertainty_weight) * anchor_uncertainty
            ) * (1.0 + float(profile.get('semantic_opportunity', 0.0)))
            combined = boundary + max(0.0, float(self.error_focus_strength)) * error_signal
            self.error_focused_query_batches_ += 1

        if not np.any(combined):
            centre = np.nanmedian(reference_X, axis=0) if len(reference_X) else np.zeros(candidates.shape[1])
            z = np.abs((candidates - centre) / scales)
            axis = np.asarray(self.semantic_feature_weights_, dtype=float)
            combined = np.average(z, axis=1, weights=np.clip(axis, 1e-9, None))

        order = np.argsort(-combined, kind='stable')
        selected = candidates[order[:n]]
        self.semantic_active_query_batches_ += 1
        self.semantic_active_query_selected_ += len(selected)
        if error_active:
            self.error_focused_query_selected_ += len(selected)
        return selected


__all__ = ["TrepanReloadedClassifier"]
