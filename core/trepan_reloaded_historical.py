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
import itertools
import math

import numpy as np

from core.trepan_original import (
    ConstraintSet,
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


class TrepanReloadedClassifier(TrepanOriginalClassifier):
    """TREPAN Reloaded construído directamente sobre o motor histórico.

    ``GAIN_CRITERIA`` lista as bases aceites para Gain_Reloaded.

    ``semantic_feature_weights`` e ``query_projector`` são argumentos de
    ``fit`` e não do construtor para que closures/callbacks de treino não fiquem
    persistidos no artefacto final.
    """

    GAIN_CRITERIA = ("normalized_information_gain", "information_gain", "gain_ratio")

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
    ):
        X_arr = np.asarray(X, dtype=float)
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
        try:
            fitted = super().fit(
                X_arr,
                y=y,
                oracle=oracle,
                sample_weight=sample_weight,
                feature_names=feature_names,
            )
        finally:
            # Não persistir callback/closure de treino no artefacto.
            self._query_projector_fit = None
            self._ontology_graph_fit = None
            self._ontology_domain_fit = None
            self._semantic_structure_cache_enabled_ = False
            self._semantic_structure_cache_ = None
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
        ):
            # Produção: uma OWL carregada não pode alterar silenciosamente a
            # árvore quando nenhuma contribuição semântica mensurável foi
            # aceite. Reexecuta exactamente o TREPAN Original com o mesmo
            # oráculo, seed e orçamento e adopta esse estado final.
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
                X_arr, y=y, oracle=oracle, sample_weight=sample_weight,
                feature_names=feature_names,
            )
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
        self.semantic_audit_summary_ = self.semantic_audit_summary()
        return fitted

    def _adopt_original_state(self, original: TrepanOriginalClassifier) -> None:
        """Adopta apenas o estado aprendido do TREPAN Original de referência."""
        for name in (
            'root_', 'classes_', 'n_features_in_', 'feature_names_in_',
            'global_distribution_model_', 'membership_queries_', 'effective_min_sample_',
            'oracle_query_count_', 'node_audit_', 'split_audit_', 'expansion_order_',
            'local_model_count_', 'node_count_', 'best_first_', 'm_of_n_',
            'oracle_', 'oracle_attached_',
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
        base = TrepanOriginalClassifier._priority(self, node)
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
        X, y, model = super()._decision_sample(node)
        # O TREPAN Original actualiza a predição da folha com a maioria do
        # conjunto de decisão imediatamente depois deste hook. O perfil EFSR
        # usa a mesma predição para medir exactamente o erro que será refinado.
        counts = np.asarray([(y == c).sum() for c in self.classes_], dtype=int)
        decision_prediction = self.classes_[int(np.argmax(counts))] if len(y) else node.prediction
        profile = self._build_error_region_profile(
            node, X, y, include_uncertainty=True, prediction_override=decision_prediction
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
        return X, y, model

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
        base = super()._feature_priority_scores(X)
        weights = getattr(self, "semantic_feature_weights_", None)
        if weights is None or np.allclose(weights, 1.0, atol=1e-12):
            return base
        # O expoente permite desligar/reduzir o viés sem mudar o algoritmo.
        factor = np.power(weights, float(self.semantic_gain_strength))
        return base * factor

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
        raw = super()._split_selection_score(y, mask, test)
        weight = self._semantic_weight_for_test(test)
        if not self._ontology_gain_active():
            if np.isclose(weight, 1.0, atol=1e-12):
                return raw
            return float(raw * (weight ** float(self.semantic_gain_strength)))
        # Um split sem informação nunca vence só pelos bónus ontológicos.
        if raw <= 1e-12:
            return 0.0
        if self.gain_criterion == "gain_ratio":
            base_gain = self._gain_ratio(raw, mask)
        elif self.gain_criterion == "normalized_information_gain":
            base_gain = self._normalized_information_gain(raw, y)
        else:
            base_gain = raw
        if not np.isclose(weight, 1.0, atol=1e-12):
            base_gain *= weight ** float(self.semantic_gain_strength)
        onto_depth = self._onto_depth_for_test(test)
        error_cov = self._error_coverage_for_test(test)
        gain_reloaded = base_gain + (self.alpha * onto_depth) + (self.beta * error_cov)
        return float(gain_reloaded)


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
        semantic_result = super()._best_mofn(X, y)
        generated = self._ontology_guided_candidates(X, y)
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
        if neutral:
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
            X, y,
            data_test=data_test,
            semantic_test=proposed_semantic_test,
            disagreement_rate=disagreement_rate,
        )
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
            proposed_mask = proposed_semantic_test.evaluate(X)
            proposed_raw = float(
                TrepanOriginalClassifier._split_selection_score(
                    self, y, proposed_mask, proposed_semantic_test
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
                float(self._gain_ratio(proposed_raw, proposed_semantic_test.evaluate(X)))
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
        return final_result

    def _split_audit_metadata(self, test: MofNTest) -> dict:
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
