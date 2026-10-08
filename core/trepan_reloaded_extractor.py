from core.tree_stop_summary import stop_summary_of
import logging
import os
import webbrowser
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
)
try:
    from dtreeviz import dtreeviz
except ImportError:  # exportação opcional
    dtreeviz = None
try:
    import owlready2
    from owlready2 import get_ontology, OwlReadyOntologyParsingError, IRIS as OWLREADY_IRIS, owl
except ImportError:  # modo sem OWL continua utilizável
    owlready2 = None
    get_ontology = None
    OWLREADY_IRIS = {}
    owl = None

    class OwlReadyOntologyParsingError(RuntimeError):
        pass
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import difflib
import pickle
import json
import copy
import time
from pathlib import Path
from core.trepan_original import TrepanOriginalClassifier, TrepanOriginalExtractor
from core.trepan_reloaded_historical import TrepanReloadedClassifier
from core.training_config import resolve_trepan_structure_limits
from core.controlled_trepan_experiment import ControlledTrepanConfig
from core.trepan_scientific_tuning import ScientificTrepanSearchConfig, tune_scientific_trepan
from core.ontology_semantic_graph import OntologySemanticGraph


from core.ontology_processor import OntologyProcessor
from core.feature_alignment import align_feature_spaces, oracle_n_features
from core.probabilistic_distillation import (
    DistillationConfig, expand_soft_targets, expand_hybrid_targets,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class _AlignedOracleView:
    """Expõe um oráculo no schema enriquecido por projeção nominal explícita."""

    def __init__(self, extractor, oracle, feature_names):
        self.extractor = extractor
        self.oracle = oracle
        self.feature_names = list(feature_names)
        self.n_features_in_ = len(self.feature_names)
        self.classes_ = getattr(oracle, 'classes_', None)

    def _matrix(self, X):
        X = np.asarray(X, dtype=float)
        expected = oracle_n_features(self.oracle)
        if expected is None:
            raise ValueError("Oráculo sem n_features_in_ no adaptador nominal.")
        if X.shape[1] == expected:
            return X
        original_names = (
            getattr(self.extractor, '_original_feature_names', None)
            or getattr(self.extractor, '_base_feature_names', None)
        )
        return align_feature_spaces(
            self.oracle, X, feature_names=self.feature_names,
            original_feature_names=original_names,
            oracle_type='MLP Original', log=False,
        )

    def predict(self, X):
        return self.oracle.predict(self._matrix(X))

    def predict_proba(self, X):
        if not hasattr(self.oracle, 'predict_proba'):
            raise AttributeError('predict_proba')
        return self.oracle.predict_proba(self._matrix(X))

class TrepanReloadedExtractor:
    """TREPAN Reloaded sobre o mesmo núcleo histórico do TREPAN Original.

    Sem OWL, espelha o TREPAN Original histórico. Com OWL, aplica o mesmo
    crescimento best-first/m-of-n no espaço enriquecido e consulta o oráculo
    activo. O CART destilado legado não é usado como TREPAN Original.
    """

    ONTO_FEATURE_PREFIX = 'onto_'
    ONTO_METADATA_TOKENS = (
        '_concept_score', '_value_score', '_num_children', '_num_properties',
        '_value_concept', '_concept', '_depth', '_parents',
    )
    ONTO_INFERRED_MARKERS = ('_High', '_Low', '_rel_', 'onto_context_')
    ONTO_FEATURE_WEIGHT = 2.5  # valor por omissão; preferir onto_feature_bias_weight
    ONTO_FEATURE_BIAS_WEIGHT_DEFAULT = 2.5
    TREE_CRITERION = 'entropy'
    CRITERION_SEARCH_ORDER = ('entropy', 'gini')
    ONTOLOGY_SAMPLE_SIZE_MIN = 5000
    ONTOLOGY_SAMPLE_SIZE_DEFAULT = 5000
    ONTOLOGY_SAMPLE_SIZE_MAX = 5000
    SEMANTIC_TUNING_MAX_CONFIGS = 24
    PARTIAL_MATCH_THRESHOLD = 0.72
    MATCH_AMBIGUITY_MARGIN = 0.08
    MAPPED_FEATURE_GAIN_BOOST = 1.2
    FEATURE_PREFIXES = ('val_', 'feat_', 'feature_', 'attr_', 'f_', 'x_')
    FEATURE_SUFFIXES = ('_score', '_value', '_val', '_feat', '_attr', '_num')
    # Papéis genéricos de alvo/classe — nunca usados no matching de features. Só vocabulário neutro quanto ao domínio:
    # os nomes do alvo e das classes REAIS do dataset vêm dos metadados em tempo de execução (``set_target_metadata``).
    FEATURE_MATCH_EXCLUDED_NAMES = frozenset({
        'class', 'targetclass', 'target', 'labelclass', 'outcome', 'classlabel', 'label',
        'negative', 'positive',  # rótulos binários frequentes
    })
    FEATURE_MATCH_EXCLUDED_OBJECT_PROPERTIES = frozenset()
    DOMINANCE_MARGIN = 0.001
    DOMINANCE_CHECK_ENABLED = False
    PRECISION_MARGIN = 0.002
    DOMINANCE_EXPANDED_DEPTHS = (5, 7, 8, 10, 12, 15, 20, None)
    FIDELITY_TARGET = 0.90
    FIDELITY_EARLY_STOP = 0.95
    RELOADED_FIDELITY_CONFIGS = (
        {'depth_grid': [5, 7, 10], 'min_samples_leaf': 4, 'min_samples_split': 8},
        {'depth_grid': [8, 12, 16, 20], 'min_samples_leaf': 2, 'min_samples_split': 4},
        {'depth_grid': [12, 16, 20, None], 'min_samples_leaf': 1, 'min_samples_split': 2},
    )

    def __init__(self, ontology=None, onto_feature_bias_weight=None, alpha=0.35, beta=0.20):
        # Pesos de Gain_Reloaded(A) = GainRatio(A) + alpha*OntoDepth(A) + beta*ErrorCoverage(A).
        # Podem ser sobrepostos por _training_limits['alpha'/'beta'].
        self.alpha = float(alpha)
        self.beta = float(beta)
        bias_env = os.environ.get('ONTO_FEATURE_BIAS_WEIGHT')
        bias_source = 'caller' if onto_feature_bias_weight is not None else 'default'
        if onto_feature_bias_weight is None and bias_env is not None:
            try:
                onto_feature_bias_weight = float(bias_env)
                bias_source = 'environment'
            except ValueError:
                onto_feature_bias_weight = self.ONTO_FEATURE_BIAS_WEIGHT_DEFAULT
                bias_source = 'default_after_invalid_environment'
        if onto_feature_bias_weight is None:
            onto_feature_bias_weight = self.ONTO_FEATURE_BIAS_WEIGHT_DEFAULT
        self.onto_feature_bias_weight = float(onto_feature_bias_weight)
        self._onto_bias_source = bias_source
        self._onto_bias_calibration_audit = None

        self.explainer_tree = None
        self.last_audit = {}
        self.ontology = ontology
        self.ontology_active = ontology is not None
        self._original_extractor = TrepanOriginalExtractor()
        self.domain_knowledge = {}
        self.semantic_rules = []
        self.last_exported_image_path = None
        self.feature_semantics = {}  # Mapeia features para conceitos semânticos
        self.class_semantics = {}    # Mapeia classes para conceitos semânticos
        self.mapping_cache = {}      # Cache de mapeamentos
        self.unmapped_features = []  # Features que não foram mapeadas
        self.unmapped_classes = []   # Classes que não foram mapeadas
        self.mapping_scores = {}     # Scores de qualidade dos mapeamentos
        self.ontology_hierarchy = {} # Hierarquia de classes extraída
        self.property_constraints = {} # Restrições de propriedades (domain/range)
        self.datatype_ranges = {}    # Ranges de datatypes extraídos
        self.feature_centrality = {}  # Importância ontológica por índice de feature
        self._real_data_bounds = {}   # Limites empíricos por feature (validação)
        self._training_cache = {}     # Artefactos do último treino (refino de dominância)
        self._training_limits = {}
        self._ontology_split_multipliers = None  # Escala onto_* para treino/inferência da árvore
        self.ontology_processor = None
        self.semantic_feature_engineering_stats = {}
        self._matrix_feature_names = []
        self._base_feature_names = []
        self._oracle_feature_names = []
        self._mlp_oracle_n_features = None
        self._active_oracle_model = None
        self.ontology_quality_report = None
        self.semantic_graph = None
        self.semantic_graph_summary = {}
        self.reasoner_report = None
        self.c45_baseline = None
        self._last_canonical_info = None
        self._last_soft_global_info = None
        self._last_plausible_cf_info = None

    @property
    def ONTO_FEATURE_BIAS_WEIGHT(self):
        """Alias legível para o peso configurável de viés ontológico nos splits."""
        return self.onto_feature_bias_weight

    def set_onto_feature_bias_weight(self, weight):
        """Actualiza o viés ontológico (GUI / runtime) e invalida multiplicadores em cache."""
        self.onto_feature_bias_weight = float(weight)
        self._onto_bias_source = 'runtime_user'
        self._onto_bias_calibration_audit = {
            'selection_scope': 'user_configuration',
            'test_used': False,
            'selected_weight': self.onto_feature_bias_weight,
            'formal_weighted_information_gain': False,
            'method': 'heuristic_sample_and_feature_priority_weight',
        }
        self._ontology_split_multipliers = None

    def _calibrate_onto_feature_bias_weight_internal(self, *args, **kwargs):
        """Compatibilidade: calibração por árvore auxiliar foi removida da produção.

        O peso semântico passa a ser um parâmetro científico explícito e auditável;
        nenhuma árvore auxiliar pode seleccionar este valor.
        """
        self._onto_bias_calibration_audit = {
            'selection_scope': self._onto_bias_source,
            'test_used': False,
            'status': 'explicit_scientific_parameter',
            'selected_weight': float(self.onto_feature_bias_weight),
            'method': 'historical_trepan_semantic_prior',
            'auxiliary_tree_used': False,
        }
        return dict(self._onto_bias_calibration_audit)

    def apply_training_preset(self, preset) -> None:
        """Aplica limites de tempo/queries/fidelidade do preset de treino."""
        structure = resolve_trepan_structure_limits(preset)
        self._training_limits = {
            'fidelity_target': preset.reloaded_fidelity_target,
            'fidelity_early_stop': preset.reloaded_fidelity_early_stop,
            'max_time_seconds': preset.reloaded_max_time_seconds,
            # Comparação controlada: o núcleo histórico Original e Reloaded
            # recebem exactamente o mesmo orçamento e limites estruturais.
            'sample_size': preset.trepan_sample_size,
            'historical_max_queries': preset.trepan_max_queries,
            'canonical_max_nodes': structure['max_nodes'],
            'canonical_max_depth': structure['max_depth'],
            'historical_min_samples_leaf': preset.trepan_min_samples_leaf,
            'historical_purity_epsilon': float(getattr(preset, 'trepan_purity_epsilon', 0.05)),
            'random_state': 42,
            'hybrid_label_weight': getattr(preset, 'hybrid_label_weight', 0.60),
            'hybrid_teacher_weight': getattr(preset, 'hybrid_teacher_weight', 0.30),
            'hybrid_semantic_weight': getattr(preset, 'hybrid_semantic_weight', 0.10),
            'canonical_trepan_enabled': getattr(preset, 'canonical_trepan_enabled', True),
            'canonical_m_of_n_max_n': getattr(preset, 'canonical_m_of_n_max_n', 3),
            'global_soft_tree_enabled': False,
            'trepan_scientific_tuning': bool(getattr(preset, 'trepan_scientific_tuning', True)),
            'trepan_tuning_cv_folds': int(getattr(preset, 'trepan_tuning_cv_folds', 3)),
            'trepan_tuning_capacity_candidates': int(getattr(preset, 'trepan_tuning_capacity_candidates', 6)),
            'trepan_fidelity_tuning_target': float(getattr(preset, 'trepan_fidelity_tuning_target', 0.95)),
            'semantic_gain_strength': float(getattr(preset, 'semantic_gain_strength', 1.0)),
            'semantic_group_strength': float(getattr(preset, 'semantic_group_strength', 0.15)),
            'semantic_candidate_budget': int(getattr(preset, 'semantic_candidate_budget', 24)),
            'semantic_relation_threshold': float(getattr(preset, 'semantic_relation_threshold', 0.35)),
            'semantic_active_query_fraction': float(getattr(preset, 'semantic_active_query_fraction', 0.65)),
            'semantic_active_pool_multiplier': int(getattr(preset, 'semantic_active_pool_multiplier', 4)),
            'error_focused_refinement': bool(getattr(preset, 'error_focused_refinement', True)),
            'error_focus_min_disagreement': float(getattr(preset, 'error_focus_min_disagreement', 0.05)),
            'error_focus_strength': float(getattr(preset, 'error_focus_strength', 1.0)),
            'error_focus_semantic_weight': float(getattr(preset, 'error_focus_semantic_weight', 0.50)),
            'error_focus_uncertainty_weight': float(getattr(preset, 'error_focus_uncertainty_weight', 0.35)),
            'error_focus_min_local_fidelity_gain': float(getattr(preset, 'error_focus_min_local_fidelity_gain', 0.002)),
            'error_focus_min_real_fidelity_gain': float(getattr(preset, 'error_focus_min_real_fidelity_gain', 0.0)),
            'error_focus_anchor_k': int(getattr(preset, 'error_focus_anchor_k', 24)),
            'error_focus_top_k': int(getattr(preset, 'error_focus_top_k', 3)),
            'error_focus_min_regions': int(getattr(preset, 'error_focus_min_regions', 1)),
            'mirror_when_no_semantic_effect': bool(getattr(preset, 'mirror_when_no_semantic_effect', True)),
            'semantic_tuning_candidates': int(getattr(preset, 'semantic_tuning_candidates', 6)),
        }
        if preset.key == 'fast':
            self.RELOADED_FIDELITY_CONFIGS = (
                {'depth_grid': [5, 8], 'min_samples_leaf': 4, 'min_samples_split': 8},
                {'depth_grid': [8, 10], 'min_samples_leaf': 2, 'min_samples_split': 4},
            )
        elif preset.key == 'balanced':
            self.RELOADED_FIDELITY_CONFIGS = (
                {'depth_grid': [5, 7, 10], 'min_samples_leaf': 4, 'min_samples_split': 8},
                {'depth_grid': [8, 12, 16], 'min_samples_leaf': 2, 'min_samples_split': 4},
            )

    def _select_oracle_model(self, mlp_model, mlp_model_onto, n_matrix_features):
        """
        Escolhe oráculo MLP original, MLP Residual Ontológico ou MLP_Onto conforme dimensões.
        Usar apenas quando X já está enriquecido (|cols| = n_matrix_features).
        """
        if mlp_model_onto is not None:
            n_onto = getattr(mlp_model_onto, 'n_features_in_', None)
            if n_onto is not None and n_onto == n_matrix_features:
                return mlp_model_onto
            if getattr(mlp_model_onto, 'ORACLE_TYPE', None) == 'residual_ontological':
                if getattr(mlp_model_onto, 'n_features_in_', None) == n_matrix_features:
                    return mlp_model_onto
        return mlp_model

    def _oracle_for_base_sampling(self, mlp_model=None):
        """Oráculo MLP original — único válido para amostragem no espaço ARFF."""
        if mlp_model is not None:
            return mlp_model
        return getattr(self, '_mlp_model_original', None)

    def _oracle_for_enriched_matrix(self, mlp_model, mlp_model_onto, n_matrix_features):
        """Oráculo para rotulagem/fidelidade em matriz já expandida (original + onto_*)."""
        return self._select_oracle_model(mlp_model, mlp_model_onto, n_matrix_features)

    def _resolve_oracle_for_matrix(self, X, mlp_model_hint=None):
        """
        Escolhe MLP original ou MLP_Onto conforme n_features_in_ vs colunas de X.
        Evita passar MLP_Onto a matrizes ARFF (ou vice-versa).
        """
        if X is None:
            return mlp_model_hint
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            n_cols = X.shape[0]
        else:
            n_cols = X.shape[1]

        mlp_orig = self._oracle_for_base_sampling()
        mlp_onto = getattr(self, '_mlp_model_onto', None)
        candidates = []

        for mlp in (mlp_onto, mlp_orig, mlp_model_hint):
            if mlp is None:
                continue
            n_exp = getattr(mlp, 'n_features_in_', None)
            if n_exp is not None and n_exp == n_cols:
                if mlp not in candidates:
                    candidates.append(mlp)

        if candidates:
            return candidates[0]

        # Matriz enriquecida sem MLP_Onto: usar MLP original nas colunas ARFF base
        if mlp_orig is not None:
            n_orig = getattr(mlp_orig, 'n_features_in_', None)
            if n_orig is not None and n_cols > n_orig:
                return mlp_orig

        raise ValueError(
            f"Alinhamento oráculo/matriz falhou: nenhum MLP aceita {n_cols} colunas. "
            f"Original={getattr(mlp_orig, 'n_features_in_', '?')}, "
            f"Onto={getattr(mlp_onto, 'n_features_in_', '?')}. "
            "Use matriz enriquecida para MLP_Onto ou ARFF para MLP original."
        )

    def _oracle_names_for_model(self, mlp_model, matrix_names):
        """Nomes de features esperados pelo oráculo para esta matriz (por dimensão)."""
        n_exp = getattr(mlp_model, 'n_features_in_', None)
        names = list(matrix_names)
        if n_exp is None:
            return names
        if n_exp == len(names):
            return names
        for cand in (
            getattr(self, '_original_feature_names', None),
            getattr(self, '_base_feature_names', None),
        ):
            if cand and n_exp == len(cand):
                return list(cand)
        stored = getattr(self, '_oracle_feature_names', None)
        if stored and n_exp == len(stored):
            return list(stored)
        return None

    @staticmethod
    def load_ontology_file(file_path):
        """
        Carrega OWL/RDF local (RDF/XML ou Turtle). Para Turtle, usa o namespace
        declarado no ficheiro (ex.: http://example.org/dominio#).
        """
        if get_ontology is None:
            raise ImportError(
                "owlready2 não está instalado. Instale requirements-runtime.txt "
                "para carregar ontologias OWL/RDF."
            )
        path = Path(file_path).resolve()
        last_error = None

        from core.owl_runtime import load_ontology_isolated
        try:
            return load_ontology_isolated(path)            # mundo OWL próprio por carga (nada de default_world partilhado)
        except (OwlReadyOntologyParsingError, Exception) as exc:
            last_error = exc

        head = path.read_text(encoding='utf-8', errors='ignore')[:8192]
        is_turtle = '@prefix' in head or (
            'syntax-ns#' in head and 'xml version' not in head[:80].lower()
        )
        if not is_turtle:
            if last_error:
                raise last_error
            raise OwlReadyOntologyParsingError(f"Formato não suportado: {path}")

        ns_iri = None
        onto_iri_match = re.search(r'<([^>]+)>\s*\n?\s*a\s+owl:Ontology', head, re.I)
        if onto_iri_match:
            base = onto_iri_match.group(1).rstrip('/')
            ns_iri = base if base.endswith('#') else f"{base}#"
        if ns_iri is None:
            prefix_match = re.search(r'@prefix\s+\w+:\s+<([^>#]+)#>\s*\.', head)
            if prefix_match:
                ns_iri = f"{prefix_match.group(1)}#"

        if not ns_iri:
            raise OwlReadyOntologyParsingError(
                f"Não foi possível determinar o IRI da ontologia Turtle: {path}"
            )

        onto = load_ontology_isolated(path, iri=ns_iri, fileobj_format='turtle')
        if not list(onto.classes()) and not list(onto.data_properties()):
            tbox_candidate = path.with_name(f"{path.stem}_TBox{path.suffix}")
            if tbox_candidate.exists() and tbox_candidate != path:
                return load_ontology_isolated(tbox_candidate)
            raise OwlReadyOntologyParsingError(
                f"Turtle carregado sem entidades reconhecíveis: {path}. "
                "Use RDF/XML, exporte um TBox (ex.: *_TBox.owl) ou verifique o namespace."
            )
        return onto

    def set_ontology(self, ontology):
        self.ontology = ontology
        self.ontology_active = ontology is not None
        if not self.ontology_active:
            self._reset_semantic_state()
        self.ontology_processor = None
        self.ontology_quality_report = None
        self.semantic_graph = None
        self.semantic_graph_summary = {}
        self.reasoner_report = None

    def _reset_semantic_state(self):
        self.domain_knowledge = {}
        self.semantic_rules = []
        self.feature_semantics = {}
        self.class_semantics = {}
        self.mapping_cache = {}
        self.unmapped_features = []
        self.unmapped_classes = []
        self.mapping_scores = {}
        self.ontology_hierarchy = {}
        self.property_constraints = {}
        self.datatype_ranges = {}
        self.feature_centrality = {}
        self._real_data_bounds = {}
        self._training_cache = {}

    @property
    def has_active_ontology(self):
        return self.ontology_active and self.ontology is not None

    def adopt_original_tree(self, tree, feature_names=None):
        """Sem ontologia: espelha a árvore do Trepan-Original (mesma instância)."""
        self.explainer_tree = tree
        self._original_extractor.explainer_tree = tree
        if feature_names is not None:
            names = list(feature_names)
            self._matrix_feature_names = names
            self._base_feature_names = names
            self._oracle_feature_names = names

    def _extract_tree_original_mode(
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
        extra_X=None,
        extra_y=None,
        extra_weights=None,
    ):
        """Sem ontologia: delega ao TREPAN Original histórico, sem CART."""
        result = self._original_extractor.extract_tree(
            mlp_model, X_encoded, y_encoded,
            sample_size=sample_size,
            feature_names=feature_names,
            class_names=class_names,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            extra_X=extra_X,
            extra_y=extra_y,
            extra_weights=extra_weights,
        )
        self.last_audit = dict(getattr(self._original_extractor, 'last_audit', {}) or {})
        self.explainer_tree = self._original_extractor.explainer_tree
        self.last_exported_image_path = getattr(
            self._original_extractor, 'last_exported_image_path', None
        )
        if feature_names is not None:
            names = list(feature_names)
            self._matrix_feature_names = names
            self._base_feature_names = names
            self._oracle_feature_names = names
        return result

    def extract_tree(
        self,
        mlp_model,
        X_encoded,
        y_encoded,
        sample_size=2000,
        feature_names=None,
        class_names=None,
        original_feature_names=None,
        mlp_model_onto=None,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        extra_X=None,
        extra_y=None,
        extra_weights=None,
    ):
        if not self.has_active_ontology:
            return self._extract_tree_original_mode(
                mlp_model, X_encoded, y_encoded, sample_size, feature_names, class_names,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                extra_X=extra_X, extra_y=extra_y, extra_weights=extra_weights,
            )
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_encoded.shape[1])]
        if class_names is None:
            class_names = [f"class_{int(c)}" for c in np.unique(y_encoded)]
        return self.extract_tree_with_ontology(
            mlp_model,
            X_encoded,
            y_encoded,
            feature_names,
            class_names,
            sample_size,
            original_feature_names=original_feature_names,
            mlp_model_onto=mlp_model_onto,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            extra_X=extra_X,
            extra_y=extra_y,
            extra_weights=extra_weights,
        )

    def _apply_reloaded_context(self, ctx: dict) -> None:
        """Regista oráculo e feature_space activos — roteamento explícito."""
        self._reloaded_context = ctx
        self._active_oracle_model = ctx.get("oracle")
        self._active_oracle_bundle = ctx.get("oracle_bundle")
        self._active_feature_space = ctx.get("feature_space")
        self._mlp_model_original = ctx.get("mlp_original_ref")
        self._mlp_model_onto = ctx.get("mlp_model_onto")
        self._labeling_oracle = ctx.get("oracle")
        self._matrix_feature_names = list(ctx.get("feature_names") or [])
        self._oof_teacher_probabilities = ctx.get("oof_teacher_probabilities")

    def _extract_tree_in_oracle_space(
        self,
        ctx: dict,
        class_names,
        sample_size=2000,
        extra_X=None,
        extra_y=None,
        extra_weights=None,
    ):
        """Trepan-Reloaded em espaço original ou residual — oráculo alinhado à matriz."""
        self._apply_reloaded_context(ctx)
        oracle = ctx["oracle"]
        feature_names = ctx["feature_names"]
        y_train = ctx["y_train"]
        y_test = ctx["y_test"]

        from core.mlp_diagnostic import diagnose_oracle, log_oracle_diagnostic
        diag = diagnose_oracle(
            oracle,
            X_train=ctx["X_train"],
            X_test=ctx["X_test"],
            oracle_label=ctx.get("oracle_name", "Oracle"),
            bundle=ctx.get("oracle_bundle"),
        )
        self._last_oracle_diagnostic = diag
        log_oracle_diagnostic(diag)

        structural_owl = bool(
            ctx.get("use_ontology_semantic_pipeline")
            and self.has_active_ontology
            and (getattr(self, 'ontology_quality_report', None) or {}).get('accepted')
        )
        if structural_owl:
            self._matrix_feature_names = list(feature_names)
            self._base_feature_names = list(feature_names)
            self._original_feature_names = list(feature_names)
            result = self._extract_historical_reloaded_core(
                labeling_oracle=oracle,
                mlp_model=(self._mlp_model_original or oracle),
                X_train_augmented=ctx["X_train"],
                y_train_real=y_train,
                X_eval=ctx["X_test"],
                y_eval_real=y_test,
                tree_feature_names=feature_names,
                base_feature_names=feature_names,
                class_names=class_names,
                sample_size=sample_size,
                extra_X=extra_X,
                extra_weights=extra_weights,
            )
            self.last_audit = dict(getattr(self, 'last_audit', {}) or {})
            self.last_audit.update({
                'ontology_structural_only': True,
                'ontology_feature_engineering_required': False,
                'feature_space': ctx.get('feature_space', 'original'),
            })
            return result

        result = self._original_extractor.extract_tree(
            oracle,
            ctx["X_train"],
            y_train,
            sample_size=sample_size,
            feature_names=feature_names,
            class_names=class_names,
            X_train=ctx["X_train"],
            X_test=ctx["X_test"],
            y_train=y_train,
            y_test=y_test,
            extra_X=extra_X,
            extra_y=extra_y,
            extra_weights=extra_weights,
        )
        self.last_audit = dict(getattr(self._original_extractor, "last_audit", {}) or {})
        self.explainer_tree = self._original_extractor.explainer_tree
        self._matrix_feature_names = list(feature_names)
        return result

    def extract_tree_with_ontology(
        self,
        mlp_model,
        X_encoded,
        y_encoded,
        feature_names,
        class_names,
        sample_size=2000,
        original_feature_names=None,
        mlp_model_onto=None,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        reloaded_context=None,
        extra_X=None,
        extra_y=None,
        extra_weights=None,
    ):
        if reloaded_context is not None:
            self._apply_reloaded_context(reloaded_context)
            mlp_model = reloaded_context.get("mlp_original_ref") or mlp_model
            if reloaded_context.get("feature_space") in ("original", "residual"):
                return self._extract_tree_in_oracle_space(
                    reloaded_context,
                    class_names,
                    sample_size=sample_size,
                    extra_X=extra_X,
                    extra_y=extra_y,
                    extra_weights=extra_weights,
                )
            X_train = reloaded_context["X_train"]
            X_test = reloaded_context["X_test"]
            y_train = reloaded_context["y_train"]
            y_test = reloaded_context["y_test"]
            feature_names = reloaded_context["feature_names"]
            mlp_model_onto = reloaded_context.get("mlp_model_onto")
            X_encoded = X_train

        if not self.has_active_ontology:
            return self._extract_tree_original_mode(
                mlp_model, X_encoded, y_encoded, sample_size, feature_names, class_names,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            )

        X_matrix = np.asarray(
            X_train if X_train is not None else X_encoded, dtype=float
        )
        X_eval_matrix = np.asarray(
            X_test if X_test is not None else X_encoded, dtype=float
        )
        y_eval_real = np.asarray(
            y_test if y_test is not None else y_encoded[: len(X_eval_matrix)]
        )

        if reloaded_context is not None and reloaded_context.get("feature_space") == "enriched":
            labeling_oracle = reloaded_context["oracle"]
            oracle_label = reloaded_context.get("oracle_name", "MLP_Onto")
        else:
            labeling_oracle = self._oracle_for_enriched_matrix(
                mlp_model, mlp_model_onto, X_matrix.shape[1]
            )
            oracle_label = (
                'MLP_Onto (augmented)'
                if labeling_oracle is mlp_model_onto and mlp_model_onto is not None
                else 'MLP original (ARFF)'
            )

        self._mlp_model_original = mlp_model
        self._mlp_model_onto = mlp_model_onto
        self._labeling_oracle = labeling_oracle
        self._active_oracle_model = labeling_oracle
        self._active_feature_space = (
            reloaded_context.get("feature_space")
            if reloaded_context is not None
            else None
        )
        self._active_oracle_bundle = (
            reloaded_context.get("oracle_bundle") if reloaded_context else None
        )
        matrix_names, base_names, oracle_names = self._resolve_feature_schemas(
            X_matrix, feature_names, mlp_model, original_feature_names,
            mlp_model_onto=mlp_model_onto,
        )
        self._original_feature_names = list(
            original_feature_names if original_feature_names else base_names
        )
        self._mlp_oracle_n_features = getattr(labeling_oracle, 'n_features_in_', None)

        from core.mlp_diagnostic import diagnose_oracle, log_oracle_diagnostic
        splits = {
            "original": {
                "X_train": X_matrix, "X_test": X_eval_matrix,
            } if reloaded_context and reloaded_context.get("feature_space") == "original" else None,
            "enriched": {
                "X_train": X_matrix, "X_test": X_eval_matrix,
            },
            "augmented": {
                "X_train": X_matrix, "X_test": X_eval_matrix,
            },
        }
        if reloaded_context and reloaded_context.get("feature_space") == "enriched":
            self._last_oracle_diagnostic = diagnose_oracle(
                labeling_oracle,
                splits={"enriched": {"X_train": X_matrix, "X_test": X_eval_matrix}},
                oracle_label=oracle_label,
                bundle=reloaded_context.get("oracle_bundle"),
            )
        else:
            self._last_oracle_diagnostic = diagnose_oracle(
                labeling_oracle,
                X_matrix,
                X_eval_matrix,
                oracle_label=oracle_label,
            )
        log_oracle_diagnostic(self._last_oracle_diagnostic)

        base_idx = self._resolve_base_column_indices(base_names, matrix_names)
        X_base = X_matrix[:, base_idx] if base_idx else X_matrix

        effective_sample_size = self._resolve_ontology_sample_size(
            len(X_base), sample_size
        )
        self._cache_real_data_bounds(X_base, base_names)

        print(
            f"[INFO] Dataset: {len(base_names)} features ARFF originais, "
            f"{X_matrix.shape[1]} colunas na matriz, "
            f"oráculo={oracle_label} ({len(oracle_names)} features)"
        )

        # Fase 1: conhecimento de domínio apenas sobre features ARFF originais
        self._extract_domain_knowledge(base_names, class_names, X_base)
        # O extractor legado acima pode reconstruir ``feature_semantics``. Reaplica
        # o matching aprovado para garantir que o grafo OWL validado é a fonte
        # final de grupos/relações usados pelo TREPAN histórico.
        quality = dict(getattr(self, 'ontology_quality_report', None) or {})
        if quality.get('accepted') and quality.get('matches'):
            self.register_quality_matches(base_names, quality.get('matches'))

        # Fase 1b: matriz já enriquecida (GUI onto_* ou inferidas) vs engenharia em X_base
        pre_augmented = (
            self._has_semantic_inferred_columns(matrix_names)
            or self._is_pre_augmented_matrix(matrix_names, base_names)
        )
        if pre_augmented:
            X_augmented = X_matrix
            tree_feature_names = matrix_names
            self._ensure_ontology_processor()
            inferred_cols = [
                n for n in matrix_names
                if n not in set(base_names) and self._is_ontology_derived_feature(n)
            ]
            if inferred_cols:
                self._register_inferred_feature_semantics(
                    base_names, matrix_names, inferred_cols
                )
            print(
                "[INFO] Matriz enriquecida já presente "
                f"({len(base_names)} ARFF + {len(matrix_names) - len(base_names)} onto_*); "
                "engenharia semântica adicional omitida."
            )
        else:
            X_augmented, tree_feature_names = self._apply_semantic_feature_engineering(
                X_base, base_names
            )

        self._matrix_feature_names = tree_feature_names

        self._X_base = X_base
        self._X_augmented_reference = X_augmented

        self.feature_centrality = self._compute_feature_centrality_scores(tree_feature_names)
        y_development = np.asarray(
            y_train if y_train is not None else y_encoded[: len(X_augmented)]
        )
        # No caminho histórico primário, o peso semântico é um prior explícito
        # derivado do quality/matching ontológico; não é calibrado por uma árvore
        # CART proxy. Isto mantém Original e Reloaded na mesma família algorítmica.
        self._onto_bias_calibration_audit = {
            'selection_scope': self._onto_bias_source,
            'test_used': False,
            'selected_weight': float(self.onto_feature_bias_weight),
            'formal_weighted_information_gain': False,
            'method': 'historical_trepan_semantic_prior_no_cart',
        }
        self._build_ontology_split_multipliers(tree_feature_names)

        return self._extract_historical_reloaded_core(
            labeling_oracle=labeling_oracle,
            mlp_model=mlp_model,
            X_train_augmented=X_augmented,
            y_train_real=y_development,
            X_eval=X_eval_matrix,
            y_eval_real=y_eval_real,
            tree_feature_names=tree_feature_names,
            base_feature_names=base_names,
            class_names=class_names,
            sample_size=sample_size,
            extra_X=extra_X,
            extra_weights=extra_weights,
        )
    
    def _build_historical_query_projector(
        self, X_augmented, base_feature_names, tree_feature_names
    ):
        """Projecta membership queries para um schema OWL semanticamente coerente.

        Quando o ``OntologyProcessor`` foi ajustado, recompõe as colunas onto_*
        a partir das features base. Para matrizes já enriquecidas fornecidas pelo
        utilizador, usa a linha de treino mais próxima no espaço base apenas para
        copiar as colunas semânticas; as features originais amostradas são
        preservadas. Nenhum dado de teste entra nesta projecção.
        """
        full_names = list(tree_feature_names or [])
        base_names = list(base_feature_names or [])
        if not full_names or len(full_names) <= len(base_names):
            return None, {'enabled': False, 'reason': 'no_semantic_columns'}

        positions = []
        for name in base_names:
            if name not in full_names:
                return None, {
                    'enabled': False,
                    'reason': 'base_feature_missing_from_augmented_schema',
                    'feature': name,
                }
            positions.append(full_names.index(name))
        semantic_positions = [i for i in range(len(full_names)) if i not in set(positions)]
        if not semantic_positions:
            return None, {'enabled': False, 'reason': 'no_semantic_columns'}

        processor = getattr(self, 'ontology_processor', None)
        processor_ready = bool(processor is not None and getattr(processor, 'is_fitted_', False))
        reference = np.asarray(X_augmented, dtype=float)
        ref_base = reference[:, positions]

        if processor_ready:
            def projector(rows):
                rows = np.asarray(rows, dtype=float)
                base = rows[:, positions]
                transformed, names = processor.transform_matrix(base, base_names)
                if list(names) != full_names:
                    raise ValueError(
                        'OntologyProcessor produziu schema diferente durante membership query: '
                        f'esperado={full_names}; produzido={list(names)}.'
                    )
                return np.asarray(transformed, dtype=float)

            return projector, {
                'enabled': True,
                'method': 'ontology_processor_transform',
                'semantic_columns': len(semantic_positions),
                'test_used': False,
            }

        # Fallback para dados já enriquecidos: aproximação empírica condicionada
        # exclusivamente ao conjunto de treino. Limita referências para manter
        # custo previsível em datasets grandes.
        rng = np.random.default_rng(42)
        if len(reference) > 512:
            ref_idx = np.sort(rng.choice(len(reference), size=512, replace=False))
            reference = reference[ref_idx]
            ref_base = ref_base[ref_idx]
        centre = np.nanmedian(ref_base, axis=0)
        scale = np.nanstd(ref_base, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        ref_scaled = np.nan_to_num((ref_base - centre) / scale, nan=0.0)

        def projector(rows):
            rows = np.asarray(rows, dtype=float)
            out = np.array(rows, copy=True)
            base = out[:, positions]
            scaled = np.nan_to_num((base - centre) / scale, nan=0.0)
            nearest = np.empty(len(out), dtype=int)
            for lo in range(0, len(out), 64):
                hi = min(len(out), lo + 64)
                distances = ((scaled[lo:hi, None, :] - ref_scaled[None, :, :]) ** 2).sum(axis=2)
                nearest[lo:hi] = np.argmin(distances, axis=1)
            out[:, semantic_positions] = reference[nearest][:, semantic_positions]
            return out

        return projector, {
            'enabled': True,
            'method': 'training_nearest_semantic_projection',
            'semantic_columns': len(semantic_positions),
            'reference_rows': int(len(reference)),
            'test_used': False,
        }

    def _extract_historical_reloaded_core(
        self, *, labeling_oracle, mlp_model, X_train_augmented, y_train_real,
        X_eval, y_eval_real, tree_feature_names, base_feature_names, class_names,
        sample_size=2000, extra_X=None, extra_weights=None,
    ):
        """Extrai o Reloaded directamente com o motor histórico, sem CART intermédio."""
        X_train_augmented = np.asarray(X_train_augmented, dtype=float)
        y_train_real = np.asarray(y_train_real)
        X_eval = np.asarray(X_eval, dtype=float)
        y_eval_real = np.asarray(y_eval_real)
        limits = getattr(self, '_training_limits', {}) or {}

        active_oracle_view = _AlignedOracleView(self, labeling_oracle, tree_feature_names)
        semantic_weights = np.asarray(
            self._build_ontology_split_multipliers(tree_feature_names), dtype=float
        )
        projector, projection_audit = self._build_historical_query_projector(
            X_train_augmented, base_feature_names, tree_feature_names
        )

        # A fonte primária da estrutura semântica passa a ser o grafo OWL
        # construído a partir do quality gate. O caminho antigo por
        # ``feature_semantics`` é apenas fallback de compatibilidade.
        graph = getattr(self, 'semantic_graph', None)
        if graph is not None and getattr(graph, 'is_active', False):
            logger.info(
                "TREPAN Reloaded: ontologia activa (is_active=True, %d features mapeadas, alpha=%.3f, beta=%.3f).",
                len(graph.feature_to_entity), float(limits.get('alpha', self.alpha)),
                float(limits.get('beta', self.beta)),
            )
        else:
            logger.warning(
                "TREPAN Reloaded: ontologia inactiva (is_active=False); "
                "Gain_Reloaded degrada para o critério do TREPAN Original."
            )
        semantic_groups = []
        graph_feature_keys = []
        for idx, name in enumerate(tree_feature_names):
            base_name = self._ontology_feature_base_name(name)
            if graph is not None and graph.entity_for_feature(base_name) is None:
                if idx < len(base_feature_names):
                    candidate = self._ontology_feature_base_name(base_feature_names[idx])
                    if graph.entity_for_feature(candidate) is not None:
                        base_name = candidate
            graph_feature_keys.append(base_name)
            info = self.feature_semantics.get(idx, {})
            group = (
                graph.primary_group(base_name) if graph is not None else None
            ) or info.get('semantic_group') \
              or ((info.get('concept_info') or {}).get('parents') or [None])[0] \
              or info.get('ontology_concept')
            semantic_groups.append(group)

        n_sem = len(tree_feature_names)
        semantic_relatedness = np.eye(n_sem, dtype=float)
        if graph is not None and getattr(graph, 'feature_to_entity', None):
            for i, left in enumerate(graph_feature_keys):
                left_entity = graph.entity_for_feature(left)
                for j in range(i + 1, n_sem):
                    right = graph_feature_keys[j]
                    right_entity = graph.entity_for_feature(right)
                    related = (
                        graph.relatedness(left_entity, right_entity)
                        if left_entity and right_entity else 0.0
                    )
                    semantic_relatedness[i, j] = semantic_relatedness[j, i] = float(related)
            self.semantic_graph_summary = graph.summary()
            self.domain_knowledge = dict(self.domain_knowledge or {})
            self.domain_knowledge['relationships'] = graph.relationship_records()
            self.domain_knowledge['semantic_graph'] = graph.to_dict()
        else:
            for i in range(n_sem):
                info_i = self.feature_semantics.get(i, {})
                parents_i = set((info_i.get('concept_info') or {}).get('parents') or [])
                group_i = semantic_groups[i] if i < len(semantic_groups) else None
                for j in range(i + 1, n_sem):
                    info_j = self.feature_semantics.get(j, {})
                    parents_j = set((info_j.get('concept_info') or {}).get('parents') or [])
                    group_j = semantic_groups[j] if j < len(semantic_groups) else None
                    related = 0.0
                    if group_i and group_j and group_i == group_j:
                        related = 1.0
                    elif parents_i and parents_j and parents_i.intersection(parents_j):
                        related = 0.80
                    semantic_relatedness[i, j] = semantic_relatedness[j, i] = related

        X_seed = X_train_augmented
        extra_seed_count = 0
        if extra_X is not None:
            external = np.asarray(extra_X, dtype=float)
            if external.ndim != 2 or external.shape[1] != X_train_augmented.shape[1]:
                raise ValueError('Contrafactuais externos incompatíveis com o schema OWL.')
            X_seed = np.vstack([X_seed, external])
            extra_seed_count = int(len(external))

        historical_min_sample = int(limits.get(
            'historical_min_sample',
            max(len(X_train_augmented), min(1000, max(120, len(X_train_augmented) * 3))),
        ))
        historical_query_budget = int(limits.get(
            'historical_max_queries',
            max(int(sample_size), historical_min_sample * 2),
        ))
        base_cfg = ControlledTrepanConfig(
            max_nodes=int(limits.get('canonical_max_nodes') or 31),
            max_depth=int(limits.get('canonical_max_depth') or limits.get('canonical_max_nodes') or 31),
            min_samples_leaf=int(limits.get('historical_min_samples_leaf', 2)),
            min_sample=historical_min_sample,
            max_n=int(limits.get('canonical_m_of_n_max_n', 3)),
            beam_width=int(limits.get('historical_beam_width', 2)),
            max_features_per_node=int(limits.get('historical_max_features_per_node', 12)),
            max_queries=historical_query_budget,
            purity_epsilon=float(limits.get('historical_purity_epsilon', 0.05)),
            purity_alpha=float(limits.get('historical_purity_alpha', 0.05)),
            mofn_alpha=float(limits.get('historical_mofn_alpha', 0.05)),
            local_model_alpha=float(limits.get('historical_local_model_alpha', 0.10)),
            random_state=int(limits.get('random_state', 42)),
            semantic_gain_strength=float(limits.get('semantic_gain_strength', 1.0)),
            semantic_group_strength=float(limits.get('semantic_group_strength', 0.15)),
            semantic_candidate_budget=int(limits.get('semantic_candidate_budget', 24)),
            semantic_relation_threshold=float(limits.get('semantic_relation_threshold', 0.35)),
            semantic_active_query_fraction=float(limits.get('semantic_active_query_fraction', 0.65)),
            semantic_active_pool_multiplier=int(limits.get('semantic_active_pool_multiplier', 4)),
            error_focused_refinement=bool(limits.get('error_focused_refinement', True)),
            error_focus_min_disagreement=float(limits.get('error_focus_min_disagreement', 0.05)),
            error_focus_strength=float(limits.get('error_focus_strength', 1.0)),
            error_focus_semantic_weight=float(limits.get('error_focus_semantic_weight', 0.50)),
            error_focus_uncertainty_weight=float(limits.get('error_focus_uncertainty_weight', 0.35)),
            error_focus_min_local_fidelity_gain=float(limits.get('error_focus_min_local_fidelity_gain', 0.002)),
            error_focus_min_real_fidelity_gain=float(limits.get('error_focus_min_real_fidelity_gain', 0.0)),
            error_focus_anchor_k=int(limits.get('error_focus_anchor_k', 24)),
            error_focus_top_k=int(limits.get('error_focus_top_k', 3)),
            error_focus_min_regions=int(limits.get('error_focus_min_regions', 1)),
            mirror_when_no_semantic_effect=bool(limits.get('mirror_when_no_semantic_effect', True)),
            alpha=float(limits.get('alpha', self.alpha)),
            beta=float(limits.get('beta', self.beta)),
            gain_criterion=str(limits.get('gain_criterion', 'normalized_information_gain')),
            semantic_query_projection=bool(limits.get('semantic_query_projection', True)),
            error_focus_fidelity_tolerance=float(limits.get('error_focus_fidelity_tolerance', 0.015)),
        )
        semantic_tuning = None
        selected_cfg = base_cfg
        if bool(limits.get('trepan_scientific_tuning', True)):
            semantic_tuning = tune_scientific_trepan(
                X_train_augmented, y_train_real, oracle=active_oracle_view,
                feature_names=tree_feature_names, base_config=base_cfg,
                semantic_feature_weights=semantic_weights,
                semantic_feature_groups=semantic_groups,
                semantic_relatedness_matrix=semantic_relatedness,
                # O projector final usa todo o treino como referência. Não é
                # usado na CV para evitar contaminação entre folds.
                query_projector=None,
                ontology_graph=graph,
                search=ScientificTrepanSearchConfig(
                    cv_folds=int(limits.get('trepan_tuning_cv_folds', 3)),
                    max_capacity_candidates=1,  # capacidade já comum ao Original
                    tune_structure=False,       # a estrutura já foi escolhida (no Original); não se volta a afinar
                    cv_repeats=1,
                    max_semantic_candidates=int(limits.get('semantic_tuning_candidates', 6)),
                    fidelity_target=float(limits.get('trepan_fidelity_tuning_target', 0.95)),
                ),
            )
            selected_cfg = ControlledTrepanConfig(**semantic_tuning['selected_config'])

        model = TrepanReloadedClassifier(
            **selected_cfg.common_tree_kwargs(),
            semantic_gain_strength=float(selected_cfg.semantic_gain_strength),
            semantic_group_strength=float(selected_cfg.semantic_group_strength),
            semantic_candidate_budget=int(selected_cfg.semantic_candidate_budget),
            semantic_relation_threshold=float(selected_cfg.semantic_relation_threshold),
            semantic_active_query_fraction=float(selected_cfg.semantic_active_query_fraction),
            semantic_active_pool_multiplier=int(selected_cfg.semantic_active_pool_multiplier),
            error_focused_refinement=bool(selected_cfg.error_focused_refinement),
            error_focus_min_disagreement=float(selected_cfg.error_focus_min_disagreement),
            error_focus_strength=float(selected_cfg.error_focus_strength),
            error_focus_semantic_weight=float(selected_cfg.error_focus_semantic_weight),
            error_focus_uncertainty_weight=float(selected_cfg.error_focus_uncertainty_weight),
            error_focus_min_local_fidelity_gain=float(selected_cfg.error_focus_min_local_fidelity_gain),
            error_focus_min_real_fidelity_gain=float(selected_cfg.error_focus_min_real_fidelity_gain),
            error_focus_anchor_k=int(selected_cfg.error_focus_anchor_k),
            error_focus_top_k=int(selected_cfg.error_focus_top_k),
            error_focus_min_regions=int(selected_cfg.error_focus_min_regions),
            mirror_when_no_semantic_effect=bool(selected_cfg.mirror_when_no_semantic_effect),
            alpha=float(selected_cfg.alpha),
            beta=float(selected_cfg.beta),
            gain_criterion=str(selected_cfg.gain_criterion),
            semantic_query_projection=bool(selected_cfg.semantic_query_projection),
            error_focus_fidelity_tolerance=float(selected_cfg.error_focus_fidelity_tolerance),
        ).fit(
            X_seed,
            oracle=active_oracle_view,
            feature_names=tree_feature_names,
            semantic_feature_weights=semantic_weights,
            semantic_feature_groups=semantic_groups,
            semantic_relatedness_matrix=semantic_relatedness,
            query_projector=projector,
            ontology_graph=graph,
        )
        self.explainer_tree = model

        y_oracle_eval = self._mlp_predict(labeling_oracle, X_eval, tree_feature_names)
        X_eval_tree = self._prepare_tree_matrix(model, X_eval, tree_feature_names)
        y_pred = model.predict(X_eval_tree)
        trepan_acc = float(accuracy_score(y_eval_real, y_pred))
        fidelity_eval = float(accuracy_score(y_oracle_eval, y_pred))

        self._last_active_query_info = {
            'enabled': True,
            'implementation': 'historical_membership_queries_per_node',
            'membership_queries': int(model.membership_queries_),
            'semantic_projected_query_count': int(model.semantic_projected_query_count_),
            'projection': projection_audit,
            'test_queried': False,
        }
        self._last_canonical_info = {
            'enabled': True,
            'implementation': 'core.trepan_reloaded_historical.TrepanReloadedClassifier',
            'historical_core': True,
            'reloaded_extension_active': bool(model.reload_extension_active_),
            'best_first': True,
            'm_of_n_used': bool(model.m_of_n_),
            'nodes': int(model.node_count_),
            'depth': int(model.get_depth()),
            'membership_queries': int(model.membership_queries_),
            'effective_min_sample': int(model.effective_min_sample_),
            'semantic_weights': [float(v) for v in model.semantic_feature_weights_],
            'semantic_groups': list(getattr(model, 'semantic_feature_groups_', [])),
            'semantic_relation_edges': int(np.sum(semantic_relatedness > 0.0) - len(semantic_relatedness)),
            'scientific_semantic_tuning': semantic_tuning,
            'splits': model.split_audit_,
            'node_audit': list(getattr(model, 'node_audit_', []) or []),
            'stop_summary': stop_summary_of(model),
            'semantic_split_audit': list(getattr(model, 'semantic_split_audit_', []) or []),
            'semantic_audit_summary': dict(getattr(model, 'semantic_audit_summary_', {}) or {}),
            'error_region_audit': list(getattr(model, 'error_region_audit_', []) or []),
            'error_focused_refinement': True,
            'test_used': False,
        }
        self._last_soft_global_info = {
            'enabled': False,
            'reason': 'auxiliary_cart_and_soft_tree_not_in_primary_reloaded_path',
            'final_model_family': 'historical_trepan_reloaded',
            'cart_used_for_final': False,
            'soft_tree_used_for_final': False,
            'test_used': False,
        }
        self._last_multiobjective_info = {
            'enabled': False,
            'reason': 'sklearn_tree_pruning_not_applicable_to_historical_trepan',
            'historical_pruning': True,
            'test_used': False,
        }
        self._last_plausible_cf_info = {
            'enabled': False,
            'reason': 'counterfactuals_are_auxiliary_and_not_required_to_fit_primary_historical_core',
            'test_used': False,
        }

        if getattr(labeling_oracle, 'ORACLE_TYPE', None) == 'validated_hybrid_ontological':
            oracle_name = 'Oráculo Híbrido Ontológico OOF'
        elif getattr(labeling_oracle, 'ORACLE_TYPE', None) == 'projected_original':
            oracle_name = 'MLP Original projetado no espaço OWL'
        elif getattr(labeling_oracle, 'ORACLE_TYPE', None) == 'residual_ontological':
            oracle_name = 'MLP Residual Ontológico'
        else:
            oracle_name = 'MLP Ontológico' if labeling_oracle is not mlp_model else 'MLP Original'

        self._log_trepan_reloaded_audit(
            oracle=oracle_name,
            X_train_shape=tuple(X_train_augmented.shape),
            X_test_shape=tuple(X_eval.shape),
            n_features=len(tree_feature_names),
            trepan_accuracy=trepan_acc,
            trepan_fidelity=fidelity_eval,
            config={
                'family': 'historical_trepan_reloaded',
                'max_nodes': model.max_nodes,
                'max_depth': model.max_depth,
                'min_sample': model.min_sample,
                'effective_min_sample': model.effective_min_sample_,
                'max_queries': model.max_queries,
                'semantic_gain_strength': model.semantic_gain_strength,
                'alpha': model.alpha,
                'beta': model.beta,
                'ontology_active': bool(getattr(model, 'ontology_active_', False)),
                'extra_seed_count': extra_seed_count,
            },
        )
        self.last_audit.update({
            'final_tree_family': 'historical_trepan_reloaded',
            'historical_core': True,
            'cart_used_for_final': False,
            'soft_tree_used_for_final': False,
            'semantic_query_projection': projection_audit,
            'semantic_split_audit': model.semantic_split_audit_,
            'semantic_audit_summary': dict(getattr(model, 'semantic_audit_summary_', {}) or {}),
            'error_region_audit': list(getattr(model, 'error_region_audit_', []) or []),
            'scientific_semantic_tuning': semantic_tuning,
            'external_test_used_for_selection': False,
        })
        try:
            from core.tree_build_report import trepan_build_report
            _oracle_label = str(self.last_audit.get('oracle') or 'selected oracle')
            _y_oracle_eval = np.asarray(active_oracle_view.predict(X_eval))
            _report = trepan_build_report(
                model, algorithm='TREPAN Reloaded', oracle_name=_oracle_label,
                oracle_type=type(labeling_oracle).__name__,
                X_eval=X_eval, y_oracle_eval=_y_oracle_eval, y_real_eval=y_eval_real,
            )
            self.last_audit['build_report'] = _report
            self.last_audit['stop_reasons'] = _report['STOP_REASONS']
            self.last_audit['query_budget_exhausted'] = _report['CONSTRUCTION']['query_budget_exhausted']
            self.last_audit['stump_diagnostic'] = _report['STUMP']
        except Exception as exc:  # o relatório nunca deve derrubar a extracção
            self.last_audit['build_report_error'] = f'{type(exc).__name__}: {exc}'

        self._training_cache = {
            'X_training': X_train_augmented,
            'feature_names': list(tree_feature_names),
            'class_names': list(class_names),
            'tree_family': 'historical_trepan_reloaded',
        }
        fidelity_metrics = self._calculate_ontology_aware_fidelity(
            model, labeling_oracle, X_eval, y_eval_real,
            X_train_augmented, active_oracle_view.predict(X_train_augmented),
        )
        export_class_names = self._class_names_for_tree(model, class_names)
        rules = model.export_text(export_class_names)
        coherence = self._validate_semantic_coherence(
            model, tree_feature_names, class_names, verbose=False
        )
        if self.ontology_processor is not None:
            rules, _flagged = self.ontology_processor.prune_semantically_incoherent_rules(
                rules, tree_feature_names, self.feature_semantics, coherence
            )
        explanation = self._generate_ontology_aware_report(
            fidelity_metrics, X_eval.shape[0], tree_feature_names, class_names
        )
        return rules + explanation

    def _collect_ontology_properties(self):
        """Recolhe propriedades OWL sem assumir uma API/ontologia específica.

        Owlready2 expõe diferentes iteradores conforme a versão e o tipo de
        propriedade. A união evita que uma ontologia composta apenas por
        DatatypeProperty seja reportada como tendo zero relações.
        """
        if self.ontology is None:
            return []
        collected = []
        for accessor in ('properties', 'data_properties', 'object_properties', 'annotation_properties'):
            fn = getattr(self.ontology, accessor, None)
            if not callable(fn):
                continue
            try:
                collected.extend(list(fn()))
            except (AttributeError, TypeError, RuntimeError):
                continue
        unique = []
        seen = set()
        for entity in collected:
            key = getattr(entity, 'iri', None) or (type(entity).__name__, getattr(entity, 'name', None), id(entity))
            if key in seen:
                continue
            seen.add(key)
            unique.append(entity)
        return unique

    def _extract_domain_knowledge(self, feature_names, class_names, X_encoded=None):
        if not self.has_active_ontology:
            return
        
        try:
            self.unmapped_features = []
            self.set_target_metadata(class_names)          # exclusões do alvo derivadas dos metadados observados
            ontology_classes = list(self.ontology.classes())
            ontology_properties = self._collect_ontology_properties()
            matching_pairs, entity_stats = self._get_ontology_matching_entities()
            matching_entities = self._unwrap_matching_entities(matching_pairs)
            self._log_ontology_matching_entities(entity_stats)

            # Mapeia features para entidades OWL (Class, DatatypeProperty, ObjectProperty)
            self._map_features_to_concepts(
                feature_names,
                matching_entities,
                ontology_properties,
                X_encoded,
            )
            
            # Mapeia classes-alvo apenas contra owl:Class
            self._map_classes_to_concepts(class_names, ontology_classes)
            
            # Extrai relacionamentos semânticos
            self._extract_semantic_relationships(ontology_classes, ontology_properties)
            
            # Inicializa domain_knowledge com estrutura completa
            self.domain_knowledge = {
                'features': self.feature_semantics,
                'classes': self.class_semantics,
                'relationships': self.domain_knowledge.get('relationships', []),
                'hierarchy': self.ontology_hierarchy,
                'property_constraints': self.property_constraints,
                'datatype_ranges': self.datatype_ranges,
                'mapping_stats': {
                    'total_features': len(feature_names),
                    'mapped_features': sum(
                        1 for f in self.feature_semantics.values()
                        if f.get('ontology_mapped')
                    ),
                    'general_fallback_features': sum(
                        1 for f in self.feature_semantics.values()
                        if f.get('general_fallback')
                    ),
                    'unmapped_features': len(self.unmapped_features),
                    'avg_mapping_score': np.mean(list(self.mapping_scores.values())) if self.mapping_scores else 0.0
                }
            }
            
            mapped_count = sum(
                1 for f in self.feature_semantics.values()
                if f.get('ontology_mapped')
            )
            unmapped_count = len(self.unmapped_features)
            base_feature_names = [
                self._ontology_feature_base_name(f)
                for f in feature_names
                if not self._is_ontology_derived_feature(f)
            ]
            mapping_lines = []
            detected_types = set()
            for info in self.feature_semantics.values():
                if info.get('ontology_derived'):
                    continue
                feat = info.get('name', '')
                lookup = self._ontology_feature_base_name(feat)
                if lookup not in base_feature_names and feat not in base_feature_names:
                    continue
                if not info.get('ontology_mapped'):
                    continue
                entity = info.get('matched_entity') or info.get('ontology_concept') or 'UNMAPPED'
                etype = info.get('entity_type', 'unknown')
                if etype and etype != 'unknown':
                    detected_types.add(etype)
                score = info.get('mapping_score', 0.0)
                mapping_lines.append(f"{lookup} -> {entity} [{etype}] score={score:.2f}")
            self._log_feature_mapping_summary(
                base_feature_names, mapping_lines, mapped_count, detected_types
            )
            print(
                f"[INFO] Conhecimento de dominio extraido: "
                f"{len(self.feature_semantics)} features, "
                f"{len(self.class_semantics)} classes, "
                f"{len(self.domain_knowledge.get('relationships', []))} relacoes, "
                f"{mapped_count} mapeadas, {unmapped_count} nao mapeadas"
            )
            if mapped_count == 0 and base_feature_names:
                print(
                    "[WARN] A ontologia foi carregada, mas nenhuma feature do ARFF foi mapeada "
                    "semanticamente. Verifique se as features estão representadas como owl:Class, "
                    "owl:DatatypeProperty ou owl:ObjectProperty e se os nomes/labels estão "
                    "alinhados com o ARFF."
                )

        except Exception as e:
            print(f"[WARN] Erro ao extrair conhecimento de dominio: {e}")
            self._create_basic_domain_knowledge(feature_names, class_names)

    def _apply_semantic_feature_engineering(self, X_encoded, feature_names):
        """
        Enriquece a matriz de treino com features onto_* inferidas (hierárquicas,
        relacionais e de restrição) via OntologyProcessor.
        """
        if not self.has_active_ontology:
            return X_encoded, feature_names

        self.ontology_processor = OntologyProcessor(self.ontology, matcher=self)
        X_new, new_names, stats = self.ontology_processor.enrich_training_matrix(
            X_encoded, feature_names, self.feature_semantics, log=True,
        )
        self.semantic_feature_engineering_stats = stats
        if len(new_names) > len(feature_names):
            self._register_inferred_feature_semantics(
                feature_names, new_names, stats.get("inferred_columns", [])
            )
        return X_new, new_names

    def _register_inferred_feature_semantics(
        self, old_names, new_names, inferred_columns
    ):
        """Atribui semântica às colunas onto_* inferidas para splits guiados."""
        old_set = set(old_names)
        for idx, fname in enumerate(new_names):
            if fname in old_set or idx in self.feature_semantics:
                continue
            if not self._is_ontology_derived_feature(fname):
                continue
            base = fname[len(self.ONTO_FEATURE_PREFIX):]
            parent_score = 0.72
            for old_idx, info in self.feature_semantics.items():
                ent = info.get("matched_entity") or info.get("ontology_concept")
                if ent and (ent in base or ent in fname):
                    parent_score = max(parent_score, float(info.get("mapping_score", 0.72)))
            self.feature_semantics[idx] = {
                'name': fname,
                'ontology_concept': base,
                'matched_entity': base,
                'entity_type': 'inferred',
                'mapping_score': parent_score,
                'onto_concept_score': parent_score,
                'semantic_properties': [],
                'concept_type': 'quantitative',
                'ontology_derived': True,
                'split_priority': 1.35 + parent_score * 0.5,
                'concept_info': {},
                'ontology_mapped': True,
                'inferred_feature': True,
            }
            self.mapping_scores[fname] = parent_score

    def _expand_synthetic_with_semantic_features(
        self, X_synthetic_base, base_feature_names, full_feature_names
    ):
        """Aplica a mesma engenharia onto_* aos sintéticos gerados no espaço ARFF."""
        if X_synthetic_base is None or len(X_synthetic_base) == 0:
            return X_synthetic_base
        X_base = np.asarray(X_synthetic_base, dtype=float)
        full_feature_names = list(full_feature_names)
        if X_base.shape[1] == len(full_feature_names):
            return X_base

        self._ensure_ontology_processor()
        if self.ontology_processor is None or not self.ontology_processor.is_fitted_:
            raise ValueError(
                "Transformador ontológico não foi ajustado no treino antes dos sintéticos."
            )
        X_syn, syn_names = self.ontology_processor.transform_matrix(
            X_base, base_feature_names
        )
        if list(syn_names) != list(full_feature_names):
            raise ValueError(
                "Schema sintético ontológico difere do schema ajustado no treino; "
                "zero-padding e reindexação silenciosa estão proibidos."
            )
        return X_syn
    
    def _create_basic_domain_knowledge(self, feature_names, class_names):
        
        # Analisa nomes de features para inferir conceitos
        for i, feature in enumerate(feature_names):
            concept_type = self._infer_concept_type(feature)
            self.feature_semantics[i] = {
                'name': feature,
                'concept_type': concept_type,
                'semantic_group': self._group_by_semantics(feature)
            }
        
        # Analisa nomes de classes para inferir conceitos
        for i, class_name in enumerate(class_names):
            concept_type = self._infer_concept_type(class_name)
            self.class_semantics[i] = {
                'name': class_name,
                'concept_type': concept_type,
                'semantic_group': self._group_by_semantics(class_name)
            }
        
        self.domain_knowledge = {
            'features': self.feature_semantics,
            'classes': self.class_semantics,
            'relationships': []
        }
    
    def _infer_concept_type(self, name):
        
        name_lower = name.lower()
        
        # Padrões para diferentes tipos de conceitos
        if any(word in name_lower for word in ['age', 'idade', 'tempo', 'time', 'duration']):
            return 'temporal'
        elif any(word in name_lower for word in ['size', 'tamanho', 'length', 'width', 'height']):
            return 'dimensional'
        elif any(word in name_lower for word in ['count', 'number', 'quantidade', 'num']):
            return 'quantitative'
        elif any(word in name_lower for word in ['color', 'cor', 'type', 'tipo', 'category']):
            return 'categorical'
        elif any(word in name_lower for word in ['score', 'rating', 'pontuação', 'nota']):
            return 'evaluative'
        else:
            return 'general'
    
    def _group_by_semantics(self, name):
        """Grupo semântico de uma feature SEM ontologia: nenhum.

        Antes adivinhava domínios (medical/financial/social/technical) por palavras no
        nome da feature. Esse "conhecimento" não vem da ontologia e chegava ao bónus de
        coesão de grupo do TREPAN. Os grupos reais vêm do grafo OWL
        (``OntologySemanticGraph.primary_group``) ou dos pais do conceito casado;
        sem isso a feature fica em ``general``, que o TREPAN ignora.
        """
        return 'general'

    def _get_ontology_entity_type(self, entity):
        """Tipo OWL: class, datatype_property ou object_property."""
        if entity is None:
            return 'unknown'
        if owl is None:
            return 'unknown'
        try:
            if hasattr(entity, 'is_a'):
                for parent in entity.is_a:
                    if parent is owl.DatatypeProperty:
                        return 'datatype_property'
                    if parent is owl.ObjectProperty:
                        return 'object_property'
                    if parent is owl.Class:
                        return 'class'
        except Exception:
            pass
        name = getattr(entity, 'name', '') or ''
        if name and self.ontology is not None:
            try:
                if name in [p.name for p in self.ontology.data_properties()]:
                    return 'datatype_property'
                if name in [p.name for p in self.ontology.object_properties()]:
                    return 'object_property'
                if name in [c.name for c in self.ontology.classes()]:
                    return 'class'
            except Exception:
                pass
        return 'unknown'

    def set_target_metadata(self, class_names=None, target_name=None):
        """Regista, a partir dos METADADOS do dataset, os nomes que representam o alvo e as suas classes.

        Entidades OWL com esses nomes, os seus pais diretos (o conceito "alvo") e os irmãos (outros valores do alvo)
        deixam de ser candidatas a features. Nada aqui depende da identidade do dataset.
        """
        names = {self._normalize_name(str(n)) for n in (class_names or []) if str(n)}
        if target_name:
            names.add(self._normalize_name(str(target_name)))
        names.discard('')
        self._target_metadata_names = names
        self._target_exclusion_names = set(names)
        onto = getattr(self, 'ontology', None)
        if onto is None or not names:
            return
        try:
            for cls in onto.classes():
                if self._normalize_name(getattr(cls, 'name', '') or '') not in names:
                    continue
                for parent in getattr(cls, 'is_a', []):
                    pname = getattr(parent, 'name', None)
                    if not pname or pname == 'Thing' or not hasattr(parent, 'subclasses'):
                        continue
                    self._target_exclusion_names.add(self._normalize_name(pname))
                    for sibling in parent.subclasses():
                        self._target_exclusion_names.add(self._normalize_name(getattr(sibling, 'name', '') or ''))
        except Exception:  # a derivação é um reforço: nunca impede o pipeline
            pass
        self._target_exclusion_names.discard('')

    def _is_excluded_from_feature_matching(self, entity):
        """Exclui classes de alvo (genéricas ou derivadas dos metadados) e propriedades que ligam ao alvo."""
        if entity is None:
            return True
        name = getattr(entity, 'name', '') or ''
        norm = self._normalize_name(name)
        runtime = getattr(self, '_target_exclusion_names', set())
        if norm in self.FEATURE_MATCH_EXCLUDED_NAMES or norm in runtime:
            return True
        if norm in self.FEATURE_MATCH_EXCLUDED_OBJECT_PROPERTIES:
            return True
        entity_type = self._get_ontology_entity_type(entity)
        if entity_type == 'object_property':
            if any(token in norm for token in ('target', 'classlabel', 'label')):
                return True
            try:  # propriedade cujo domínio/contradomínio é um conceito de alvo
                linked = list(getattr(entity, 'range', []) or []) + list(getattr(entity, 'domain', []) or [])
                if any(self._normalize_name(getattr(c, 'name', '') or '') in runtime for c in linked):
                    return True
            except Exception:
                pass
        if entity_type != 'class':
            return False
        try:
            blocked = self.FEATURE_MATCH_EXCLUDED_NAMES | runtime
            for parent in getattr(entity, 'is_a', []):
                parent_name = getattr(parent, 'name', None)
                if parent_name and self._normalize_name(parent_name) in blocked:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _unwrap_matching_entities(matching_pairs):
        """Extrai entidades OWL de pares (entity, entity_type)."""
        if not matching_pairs:
            return []
        if isinstance(matching_pairs[0], tuple) and len(matching_pairs[0]) == 2:
            return [entity for entity, _etype in matching_pairs]
        return list(matching_pairs)

    def _get_ontology_matching_entities(self, ontology=None):
        """
        Retorna entidades OWL que podem representar features de qualquer ARFF:
        owl:Class, owl:DatatypeProperty e owl:ObjectProperty (excepto alvo/diagnóstico).

        Returns:
            matching_pairs: lista de (entity, entity_type) com entity_type em
                {'class', 'datatype_property', 'object_property'}
            stats: contagens para logging
        """
        onto = ontology if ontology is not None else self.ontology
        empty_stats = {
            'classes': 0,
            'datatype_properties': 0,
            'object_properties': 0,
            'matching_classes': 0,
            'matching_datatype_properties': 0,
            'matching_object_properties': 0,
            'total': 0,
        }
        if onto is None:
            return [], empty_stats

        try:
            all_classes = list(onto.classes())
        except Exception:
            all_classes = []
        try:
            data_props = list(onto.data_properties())
        except Exception:
            data_props = []
        try:
            object_props = list(onto.object_properties())
        except Exception:
            object_props = []

        matching_pairs = []
        for entity in all_classes:
            if not self._is_excluded_from_feature_matching(entity):
                matching_pairs.append((entity, 'class'))
        for entity in data_props:
            if not self._is_excluded_from_feature_matching(entity):
                matching_pairs.append((entity, 'datatype_property'))
        for entity in object_props:
            if not self._is_excluded_from_feature_matching(entity):
                matching_pairs.append((entity, 'object_property'))

        stats = {
            'classes': len(all_classes),
            'datatype_properties': len(data_props),
            'object_properties': len(object_props),
            'matching_classes': sum(1 for _, t in matching_pairs if t == 'class'),
            'matching_datatype_properties': sum(
                1 for _, t in matching_pairs if t == 'datatype_property'
            ),
            'matching_object_properties': sum(
                1 for _, t in matching_pairs if t == 'object_property'
            ),
            'total': len(matching_pairs),
        }
        return matching_pairs, stats

    def _get_ontology_class_entities_for_value_matching(self, ontology=None):
        """Classes e indivíduos OWL para mapear categorias nominais."""
        onto = ontology if ontology is not None else self.ontology
        if onto is None:
            return []
        try:
            return list(onto.classes()) + list(onto.individuals())
        except Exception:
            return []

    def _get_ontology_individual_entities_for_value_matching(self, ontology=None):
        """Indivíduos nomeados da ABox de vocabulário, nunca registos do dataset."""
        onto = ontology if ontology is not None else self.ontology
        if onto is None:
            return []
        try:
            return list(onto.individuals())
        except Exception:
            return []

    def _log_ontology_matching_entities(self, stats):
        print('[INFO] Ontology matching entities:')
        print(f"   - Classes: {stats.get('classes', 0)}")
        print(f"   - DatatypeProperties: {stats.get('datatype_properties', 0)}")
        print(f"   - ObjectProperties: {stats.get('object_properties', 0)}")
        print(
            f"   - Elegíveis para features: {stats.get('total', 0)} "
            f"(classes={stats.get('matching_classes', 0)}, "
            f"datatype={stats.get('matching_datatype_properties', 0)}, "
            f"object={stats.get('matching_object_properties', 0)})"
        )

    def _log_feature_mapping_summary(
        self, feature_names, mapping_lines, mapped_count, detected_types=None
    ):
        print('[INFO] Feature mapping:')
        for line in mapping_lines:
            print(f"   - {line}")
        unmapped = max(0, len(feature_names) - mapped_count)
        enriched = 'yes' if mapped_count > 0 else 'no'
        types_list = sorted(detected_types or [])
        print('[INFO] Summary:')
        print(f"   - Total de features no ARFF: {len(feature_names)}")
        print(f"   - Features mapeadas semanticamente: {mapped_count}/{len(feature_names)}")
        print(f"   - Features não mapeadas: {unmapped}")
        if types_list:
            print(f"   - Tipos detetados: {types_list}")
        print(f"   - Colunas de enriquecimento criadas: {enriched}")

    def _is_ontology_derived_feature(self, feature_name):
        return str(feature_name).lower().startswith(self.ONTO_FEATURE_PREFIX)

    def _is_ontology_metadata_column(self, feature_name):
        """Colunas onto_* de metadados da GUI (concept, depth, score, ...)."""
        name = str(feature_name).lower()
        if not name.startswith(self.ONTO_FEATURE_PREFIX):
            return False
        return any(token in name for token in self.ONTO_METADATA_TOKENS)

    def _is_semantic_inferred_column(self, feature_name):
        """Features onto_* criadas pelo OntologyProcessor (não metadados GUI)."""
        if not self._is_ontology_derived_feature(feature_name):
            return False
        if self._is_ontology_metadata_column(feature_name):
            return False
        name = str(feature_name)
        if any(marker in name for marker in self.ONTO_INFERRED_MARKERS):
            return True
        # Abstrações hierárquicas: onto_Cell_Characteristic (sem sufixo de metadado)
        base_stripped = name[len(self.ONTO_FEATURE_PREFIX):]
        return '_' not in base_stripped or base_stripped.count('_') <= 2

    def _identify_base_feature_names(self, feature_names, original_feature_names=None):
        """Features ARFF originais — agnóstico ao número de colunas."""
        if original_feature_names:
            return list(original_feature_names)
        return [
            n for n in feature_names
            if not self._is_ontology_derived_feature(n)
        ]

    def _has_semantic_inferred_columns(self, feature_names):
        return any(self._is_semantic_inferred_column(n) for n in feature_names)

    def _is_pre_augmented_matrix(self, matrix_names, base_names):
        """Matriz já enriquecida (GUI onto_* ou OntologyProcessor) — não re-engenharia só em X_base."""
        if not matrix_names or not base_names:
            return False
        if len(matrix_names) <= len(base_names):
            return False
        base_set = set(base_names)
        return any(n not in base_set for n in matrix_names)

    def _ensure_ontology_processor(self):
        if self.ontology_processor is None and self.has_active_ontology:
            self.ontology_processor = OntologyProcessor(
                self.ontology, matcher=self
            )

    def _expand_synthetic_by_schema(
        self, X_base, base_names, full_names, reference_X=None
    ):
        """Transforma ARFF para OWL apenas com o transformer ajustado no treino."""
        X_base = np.asarray(X_base, dtype=float)
        full_names = list(full_names)
        if X_base.shape[1] == len(full_names):
            return X_base
        self._ensure_ontology_processor()
        if self.ontology_processor is None or not self.ontology_processor.is_fitted_:
            raise ValueError("Expansão ARFF→OWL exige OntologyProcessor fitted no treino.")
        transformed, names = self.ontology_processor.transform_matrix(X_base, base_names)
        if list(names) != full_names:
            raise ValueError(
                f"Schema OWL incompatível: esperado={full_names}, produzido={names}."
            )
        return transformed

    def _align_matrix_to_feature_schema(self, X, feature_names, reference_X=None):
        """
        Garante X.shape[1] == len(feature_names) para treino/predict da árvore Reloaded.
        Expande ARFF → schema enriquecido quando necessário (ex.: 14 → 44 cols).
        """
        if X is None:
            return X
        X = np.asarray(X, dtype=float)
        if not feature_names:
            return X
        feature_names = list(feature_names)
        n_full = len(feature_names)
        if X.shape[1] == n_full:
            return X
        ref = reference_X if reference_X is not None else getattr(
            self, '_X_augmented_reference', None
        )
        base_names = (
            getattr(self, '_base_feature_names', None)
            or getattr(self, '_original_feature_names', None)
            or []
        )
        if base_names and X.shape[1] == len(base_names) and n_full > X.shape[1]:
            return self._expand_synthetic_by_schema(
                X, list(base_names), feature_names, reference_X=ref
            )
        raise ValueError(
            f"Schema da árvore incompatível: matriz={X.shape[1]} colunas, "
            f"schema={n_full}. Apenas expansão nominal ARFF→OWL é permitida."
        )

    def _prepare_tree_matrix(self, tree, X, feature_names=None):
        """Alinha colunas de X ao n_features da árvore e aplica viés de split."""
        if tree is None or X is None:
            return X
        if hasattr(tree, 'tree_'):
            n_tree = int(getattr(tree.tree_, 'n_features', 0) or 0)
        else:
            n_tree = int(getattr(tree, 'n_features_in_', 0) or 0)
        names = list(
            feature_names
            or getattr(self, '_matrix_feature_names', None)
            or []
        )
        if n_tree and (not names or len(names) != n_tree):
            original_names = list(
                getattr(self, '_original_feature_names', None)
                or getattr(self, '_base_feature_names', None)
                or []
            )
            if len(original_names) != n_tree or not all(
                name in names for name in original_names
            ):
                raise ValueError(
                    f"Schema da árvore ausente/incompatível: árvore={n_tree}, "
                    f"nomes={len(names)}."
                )
            X = align_feature_spaces(
                type('_TreeShape', (), {'n_features_in_': n_tree})(),
                X, feature_names=names, original_feature_names=original_names,
                oracle_type='árvore em espaço original', log=False,
            )
            names = original_names
        if names:
            X = self._align_matrix_to_feature_schema(
                X, names, getattr(self, '_X_augmented_reference', None)
            )
        elif n_tree and X.shape[1] != n_tree:
            raise ValueError(
                f"Matriz incompatível com a árvore: {X.shape[1]} != {n_tree}."
            )
        return self.apply_ontology_split_bias(X)

    def _tree_schema_feature_names(self, feature_names=None):
        """Nomes completos da matriz de treino da árvore (ARFF + onto_*)."""
        candidates = [
            getattr(self, '_matrix_feature_names', None),
            (getattr(self, '_training_cache', None) or {}).get('feature_names'),
            feature_names,
        ]
        best = []
        for names in candidates:
            if names and len(list(names)) > len(best):
                best = list(names)
        return best

    def _align_real_for_tree(self, X, feature_names=None):
        """Expande matriz ARFF para o schema completo da árvore Reloaded."""
        names = self._tree_schema_feature_names(feature_names)
        if not names:
            return np.asarray(X, dtype=float)
        return self._align_matrix_to_feature_schema(
            X, names, getattr(self, '_X_augmented_reference', None)
        )

    def _valid_column_indices(self, indices, n_columns):
        return [int(i) for i in indices if 0 <= int(i) < n_columns]

    def _class_names_for_tree(self, tree, class_names=None):
        """
        Nomes humanos alinhados a tree.classes_ (export_text/graphviz exigem o mesmo tamanho).
        class_names referencia rótulos inteiros 0..K-1 do label encoder.
        """
        if tree is None:
            return list(class_names or [])
        classes = np.asarray(getattr(tree, 'classes_', []))
        if classes.size == 0:
            return list(class_names or [])
        if class_names is None:
            return [f"class_{int(c)}" for c in classes]
        names = list(class_names)
        out = []
        for c in classes:
            ci = int(c)
            if 0 <= ci < len(names):
                out.append(str(names[ci]))
            else:
                out.append(f"class_{ci}")
        return out

    def _ensure_training_label_diversity(
        self, y_train, X_train, X_real, y_real, mlp_model, feature_names
    ):
        """Evita árvore com uma única classe quando o problema é multi-classe."""
        y_train = np.asarray(y_train)
        y_real = np.asarray(y_real)
        if len(np.unique(y_real)) < 2:
            return y_train
        if len(np.unique(y_train)) >= 2:
            return y_train

        print(
            "[WARN] Rótulos sintéticos com uma única classe; a aplicar fallback de diversidade."
        )
        mlp_orig = self._oracle_for_base_sampling()
        if mlp_orig is not None and X_train is not None:
            try:
                X_base, _, _ = self._base_matrix_and_names(X_train, feature_names)
                y_base = mlp_orig.predict(X_base)
                if len(np.unique(y_base)) >= 2:
                    print("[INFO] Fallback: rótulos do MLP original (espaço ARFF).")
                    return y_base
            except Exception:
                pass

        if mlp_model is not None and X_real is not None:
            y_mlp_real = self._mlp_predict(mlp_model, X_real, feature_names)
            if len(np.unique(y_mlp_real)) >= 2:
                n = len(y_train)
                n_mix = min(len(y_mlp_real), max(100, n // 5))
                rng = np.random.default_rng(42)
                idx = rng.choice(n, size=n_mix, replace=False)
                y_mix = y_train.copy()
                y_mix[idx] = rng.choice(y_mlp_real, size=n_mix)
                if len(np.unique(y_mix)) >= 2:
                    return y_mix

        if mlp_model is not None and X_train is not None:
            try:
                y_oracle_full = self._mlp_predict(mlp_model, X_train, feature_names)
                if len(np.unique(y_oracle_full)) >= 2:
                    n = len(y_train)
                    if len(y_oracle_full) >= n:
                        print("[INFO] Fallback: rótulos sintéticos substituídos pelo oráculo MLP.")
                        return np.asarray(y_oracle_full[:n])
            except Exception:
                pass

        print(
            "[WARN] Diversidade de classes não alcançada nos rótulos sintéticos; "
            "mantendo rótulos do oráculo (nunca rótulos reais y)."
        )
        return y_train

    def _retain_multiclass_tree(self, tree, *args, **kwargs):
        """Compatibilidade sem refit: preserva o modelo histórico recebido.

        A versão de produção não cria uma segunda família de árvore para
        "corrigir" classes ausentes. O próprio TREPAN histórico é a autoridade.
        """
        return tree

    def _align_matrix_feature_names(self, feature_names, n_columns):
        names = list(feature_names) if feature_names else []
        if not names:
            names = [f"feature_{i}" for i in range(n_columns)]
        if len(names) != n_columns:
            raise ValueError(
                f"Schema nominal incompatível: {len(names)} nomes para "
                f"{n_columns} colunas."
            )
        if len(set(names)) != len(names):
            raise ValueError("Schema nominal contém atributos duplicados.")
        return names

    def _column_indices_for_names(self, names, matrix_feature_names):
        """Índices de colunas por nome (nunca por contagem fixa)."""
        if not names:
            return []
        name_to_idx = {n: i for i, n in enumerate(matrix_feature_names)}
        indices = [name_to_idx[n] for n in names if n in name_to_idx]
        missing = [n for n in names if n not in name_to_idx]
        if missing:
            raise ValueError(
                f"{len(missing)} coluna(s) do oráculo/base ausentes na matriz: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return indices

    def _resolve_base_column_indices(self, base_names, matrix_names):
        """
        Índices das colunas ARFF na matriz enriquecida.
        Fallback posicional nas primeiras colunas não-onto_* quando nomes divergem
        (ex.: mean_radius no ARFF vs feature_0 na matriz GUI).
        """
        indices = self._column_indices_for_names(base_names, matrix_names)
        if base_names and len(indices) == len(base_names):
            return indices
        non_onto = [
            i for i, n in enumerate(matrix_names)
            if not self._is_ontology_derived_feature(n)
        ]
        if base_names and len(non_onto) >= len(base_names):
            if len(indices) < len(base_names):
                print(
                    f"[WARN] Alinhamento base por posição: "
                    f"{len(base_names)} colunas não-onto (nomes ARFF ≠ matriz)."
                )
            return non_onto[: len(base_names)]
        return indices

    def _pick_oracle_column_indices(
        self, matrix_names, mlp_model, oracle_names=None, base_names=None
    ):
        """
        Resolve índices de colunas para o oráculo MLP exclusivamente por nome.
        Agnóstico ao número de features — nunca assume [:N] posicional.
        """
        names = self._align_matrix_feature_names(
            list(matrix_names), len(matrix_names)
        )
        n_exp = getattr(mlp_model, 'n_features_in_', None)
        name_to_idx = {n: i for i, n in enumerate(names)}

        candidate_lists = []
        for cand in (
            oracle_names,
            base_names,
            getattr(self, '_oracle_feature_names', None),
            getattr(self, '_base_feature_names', None),
            getattr(self, '_original_feature_names', None),
        ):
            if cand:
                candidate_lists.append(list(cand))

        seen = set()
        unique_lists = []
        for cand in candidate_lists:
            key = tuple(cand)
            if key not in seen:
                seen.add(key)
                unique_lists.append(cand)

        for cand in unique_lists:
            indices = [name_to_idx[n] for n in cand if n in name_to_idx]
            if not indices:
                continue
            if n_exp is None:
                return indices
            if len(indices) == n_exp:
                return indices

        if n_exp is not None and len(names) == n_exp:
            return list(range(len(names)))

        non_onto_names = [n for n in names if not self._is_ontology_derived_feature(n)]
        non_onto_idx = [name_to_idx[n] for n in non_onto_names if n in name_to_idx]
        if n_exp is not None and len(non_onto_idx) == n_exp:
            return non_onto_idx

        orig_names = getattr(self, '_original_feature_names', None)
        if orig_names and n_exp == len(orig_names):
            indices = [name_to_idx[n] for n in orig_names if n in name_to_idx]
            if len(indices) == n_exp:
                return indices

        if n_exp is not None and len(non_onto_idx) >= n_exp:
            return non_onto_idx[:n_exp]
        if n_exp is not None and len(names) >= n_exp:
            print(
                f"[WARNING] Alinhamento oráculo por posição: "
                f"{n_exp} de {len(names)} colunas."
            )
            return list(range(n_exp))
        return list(range(len(names)))

    def _validate_oracle_width(self, X, mlp_model=None):
        """Garante largura compatível com o oráculo (alinhamento automático se necessário)."""
        if X is None:
            return X
        X = np.asarray(X, dtype=float)
        oracle = mlp_model
        n_exp = oracle_n_features(oracle)
        if n_exp is None:
            n_exp = getattr(self, '_mlp_oracle_n_features', None)
        if n_exp is not None and X.shape[1] != n_exp:
            matrix_names = getattr(self, '_matrix_feature_names', None)
            base_names = (
                getattr(self, '_original_feature_names', None)
                or getattr(self, '_base_feature_names', None)
            )
            return align_feature_spaces(
                oracle,
                X,
                feature_names=matrix_names,
                original_feature_names=base_names,
            )
        return X

    def _enforce_oracle_width(self, X, mlp_model=None):
        """Compat: valida largura; não corta colunas por posição."""
        return self._validate_oracle_width(X, mlp_model)

    def _resolve_feature_schemas(
        self, X, feature_names, mlp_model, original_feature_names=None,
        mlp_model_onto=None,
    ):
        """
        Separa nomes da matriz, features ARFF originais e features do oráculo MLP.
        O oráculo usa exactamente n_features_in_ do MLP activo (original ou MLP_Onto).
        """
        matrix_names = self._align_matrix_feature_names(feature_names, X.shape[1])
        base_names = self._identify_base_feature_names(
            matrix_names, original_feature_names
        )
        active_oracle = self._select_oracle_model(
            mlp_model, mlp_model_onto, len(matrix_names)
        )
        n_mlp = getattr(active_oracle, 'n_features_in_', None)

        if n_mlp is None:
            oracle_names = list(base_names)
        elif n_mlp == len(matrix_names):
            oracle_names = list(matrix_names)
        elif n_mlp == len(base_names):
            # MLP treinado só em ARFF; matriz pode ter colunas extra (ruído/onto_*)
            oracle_names = list(base_names)
        else:
            in_matrix = [n for n in base_names if n in matrix_names]
            if n_mlp is not None and n_mlp == len(in_matrix):
                oracle_names = in_matrix
            else:
                raise ValueError(
                    f"Não é possível identificar nominalmente o espaço do oráculo: "
                    f"MLP={n_mlp}, matriz={len(matrix_names)}, base={len(base_names)}."
                )

        if not base_names:
            base_names = list(oracle_names)
        n_stored = n_mlp if n_mlp is not None else getattr(active_oracle, 'n_features_in_', None)
        self._matrix_feature_names = matrix_names
        self._base_feature_names = base_names
        self._oracle_feature_names = oracle_names
        self._mlp_oracle_n_features = n_stored
        return matrix_names, base_names, oracle_names

    def _base_matrix_and_names(self, X, feature_names=None):
        """Sub-matriz ARFF original (espaço do oráculo MLP) a partir de matriz enriquecida."""
        if X is None:
            return None, [], []
        X = np.asarray(X, dtype=float)
        matrix_names = self._align_matrix_feature_names(
            feature_names or getattr(self, '_matrix_feature_names', None) or [],
            X.shape[1],
        )
        base_names = getattr(self, '_base_feature_names', None)
        if not base_names:
            base_names = self._identify_base_feature_names(matrix_names)
        orig_names = getattr(self, '_original_feature_names', None) or base_names
        indices = self._resolve_base_column_indices(orig_names, matrix_names)
        if indices and len(indices) == len(orig_names):
            return X[:, indices], list(orig_names), matrix_names
        if indices:
            raise ValueError(
                f"Base ARFF parcial: {len(indices)}/{len(orig_names)} colunas por nome."
            )
        n_base = len(orig_names)
        aligned = align_feature_spaces(
            type('_Shape', (), {'n_features_in_': n_base})(),
            X,
            feature_names=matrix_names,
            original_feature_names=orig_names,
            oracle_type='MLP Original',
        )
        return aligned, list(orig_names), matrix_names

    def _matrix_for_oracle(self, X, matrix_feature_names=None, mlp_model=None):
        """Extrai sub-matriz alinhada ao oráculo MLP (original ou MLP_Onto)."""
        if X is None:
            return X
        X = np.asarray(X, dtype=float)
        oracle = self._resolve_oracle_for_matrix(X, mlp_model)
        names = matrix_feature_names or getattr(self, '_matrix_feature_names', None)
        base_names = (
            getattr(self, '_original_feature_names', None)
            or getattr(self, '_base_feature_names', None)
        )
        n_oracle = oracle_n_features(oracle)
        n_target = X.shape[1]
        if n_oracle == n_target:
            return X
        oracle_type = 'MLP_Onto' if n_oracle is not None and n_oracle == len(names or []) else 'MLP Original'
        return align_feature_spaces(
            oracle,
            X,
            feature_names=names,
            original_feature_names=base_names,
            oracle_type=oracle_type,
        )

    def _features_for_mlp_oracle(self, X, matrix_feature_names=None, mlp_model=None):
        """Oráculo MLP: colunas determinadas por n_features_in_ e nomes, sem hardcoding."""
        return self._matrix_for_oracle(X, matrix_feature_names, mlp_model)

    def _mlp_predict(self, mlp_model, X, matrix_feature_names=None):
        active = getattr(self, "_active_oracle_model", None)
        bundle = getattr(self, "_active_oracle_bundle", None)
        space = getattr(self, "_active_feature_space", None)
        if active is not None and bundle is not None:
            from core.model_bundle import safe_oracle_predict
            return safe_oracle_predict(
                active,
                X,
                bundle=bundle,
                current_feature_space=space,
            )
        oracle = self._resolve_oracle_for_matrix(X, mlp_model)
        return oracle.predict(
            self._features_for_mlp_oracle(X, matrix_feature_names, oracle)
        )

    def _mlp_predict_proba(self, mlp_model, X, matrix_feature_names=None):
        active = getattr(self, "_active_oracle_model", None)
        bundle = getattr(self, "_active_oracle_bundle", None)
        if active is not None and bundle is not None:
            from core.model_bundle import validate_model_input
            X_arr = np.asarray(X, dtype=float)
            validate_model_input(
                bundle, X_arr,
                current_feature_space=getattr(self, "_active_feature_space", None),
            )
            return active.predict_proba(X_arr)
        oracle = self._resolve_oracle_for_matrix(X, mlp_model)
        return oracle.predict_proba(
            self._features_for_mlp_oracle(X, matrix_feature_names, oracle)
        )

    def _add_soft_label_distillation(
        self, mlp_model, X, hard_labels, sample_weights, feature_names,
        temperature=2.0, soft_weight=0.35, min_probability=0.03,
    ):
        """Representa probabilidades do oraculo como alvos ponderados da arvore."""
        if not hasattr(mlp_model, 'predict_proba') or soft_weight <= 0:
            return X, hard_labels, sample_weights
        X_arr = np.asarray(X, dtype=float)
        y_hard = np.asarray(hard_labels)
        base_w = (np.asarray(sample_weights, dtype=float)
                  if sample_weights is not None else np.ones(len(X_arr)))
        try:
            probabilities = np.asarray(
                self._mlp_predict_proba(mlp_model, X_arr, feature_names), dtype=float
            )
        except Exception as exc:
            print(f"[WARN] Destilacao soft omitida: {exc}")
            return X_arr, y_hard, base_w
        classes = np.asarray(
            getattr(mlp_model, 'classes_', np.arange(probabilities.shape[1]))
        )
        X_out, y_out, w_out, audit = expand_soft_targets(
            X_arr, y_hard, probabilities, classes, base_w,
            DistillationConfig(
                temperature=float(temperature), soft_weight=float(soft_weight),
                min_probability=float(min_probability),
            ),
        )
        self._last_distillation_info = audit
        return X_out, y_out, w_out

    def _ontology_feature_base_name(self, feature_name):
        name = str(feature_name)
        if self._is_ontology_derived_feature(name):
            return name[len(self.ONTO_FEATURE_PREFIX):]
        return name

    def _map_features_to_concepts(self, feature_names, ontology_entities, ontology_properties, X_encoded=None):
        for i, feature in enumerate(feature_names):
            lookup_name = self._ontology_feature_base_name(feature)
            match_result = self._find_matching_concept(lookup_name, ontology_entities)
            matching_concept = match_result['concept'] if match_result else None
            matching_score = match_result['score'] if match_result else 0.0
            match_type = match_result.get('match_type') if match_result else None
            entity_type = match_result.get('entity_type') if match_result else None

            if self._is_ontology_derived_feature(feature) and matching_concept:
                matching_score = min(1.0, matching_score + 0.1)
                self.mapping_scores[feature] = matching_score

            semantic_props = self._extract_semantic_properties(lookup_name, ontology_properties)

            is_general_fallback = False
            if not matching_concept:
                best_suggestion = self._find_best_fallback_match(lookup_name, ontology_entities)
                if best_suggestion:
                    matching_concept = best_suggestion['name']
                    matching_score = best_suggestion['score'] * 0.85
                    match_type = 'fallback_' + best_suggestion.get('type', 'similarity')
                    entity_type = best_suggestion.get('entity_type')

            if not matching_concept:
                general_category = self._assign_general_category(i, feature, X_encoded)
                matching_concept = general_category
                matching_score = 0.35
                match_type = 'general_category'
                entity_type = None
                is_general_fallback = True
                self.unmapped_features.append({
                    'feature': feature,
                    'suggested_match': general_category,
                    'confidence': matching_score,
                    'general_category': True,
                })
            
            concept_info = {}
            if matching_concept and not is_general_fallback:
                concept_obj = self._get_entity_object(matching_concept, ontology_entities)
                if concept_obj:
                    concept_info = self._extract_entity_details(concept_obj)
                    if entity_type is None:
                        entity_type = self._get_ontology_entity_type(concept_obj)
            
            is_onto = self._is_ontology_derived_feature(feature)
            onto_score = float(matching_score)
            split_priority = 1.0 + (onto_score * 1.2)
            if is_onto and matching_concept:
                split_priority += 0.35
            if is_general_fallback:
                split_priority = 0.85

            concept_type = self._infer_concept_type(lookup_name)
            if is_general_fallback:
                concept_type = 'categorical' if 'Categorical' in matching_concept else 'quantitative'

            self.feature_semantics[i] = {
                'name': feature,
                'ontology_concept': matching_concept,
                'matched_entity': matching_concept,
                'entity_type': entity_type,
                'mapping_score': matching_score,
                'onto_concept_score': onto_score,
                'semantic_properties': semantic_props,
                'concept_type': concept_type,
                'ontology_derived': is_onto,
                'split_priority': split_priority,
                'concept_info': concept_info,
                'rdfs_labels': concept_info.get('labels', []),
                'rdfs_comment': concept_info.get('comment', ''),
                'semantic_group': (concept_info.get('parents') or [matching_concept])[0] if matching_concept else concept_type,
                'match_type': match_type,
                'general_fallback': is_general_fallback,
                'ontology_mapped': not is_general_fallback and matching_concept is not None,
            }
            if matching_concept and not is_general_fallback:
                self.mapping_scores[feature] = matching_score

    def _assign_general_category(self, feature_idx, feature_name, X_encoded):
        """Categoria sintética para features sem match OWL — ainda usada na geração de dados."""
        if X_encoded is not None and feature_idx < X_encoded.shape[1]:
            col = X_encoded[:, feature_idx]
            n_unique = len(np.unique(col))
            if n_unique <= 10:
                return 'General_Categorical'
            return 'General_Numeric'
        name_lower = feature_name.lower()
        if any(w in name_lower for w in ['type', 'tipo', 'category', 'class', 'label']):
            return 'General_Categorical'
        return 'General_Numeric'
    
    def _map_classes_to_concepts(self, class_names, ontology_classes):
        
        for i, class_name in enumerate(class_names):
            match_result = self._find_matching_concept(
                class_name, ontology_classes, allow_target_classes=True
            )
            matching_concept = match_result['concept'] if match_result else None
            
            self.class_semantics[i] = {
                'name': class_name,
                'ontology_concept': matching_concept,
                'concept_type': self._infer_concept_type(class_name),
                'mapping_score': match_result['score'] if match_result else 0.0,
            }
    
    def _camel_to_snake(self, name):
        """Converte CamelCase para snake_case antes da normalização."""
        if not name:
            return name
        s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', str(name))
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
        return s.lower()

    def _strip_feature_affixes(self, name):
        """Remove prefixos/sufixos técnicos comuns antes do matching."""
        cleaned = name.lower().strip()
        changed = True
        while changed:
            changed = False
            for prefix in self.FEATURE_PREFIXES:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
                    changed = True
            for suffix in self.FEATURE_SUFFIXES:
                if cleaned.endswith(suffix):
                    cleaned = cleaned[:-len(suffix)]
                    changed = True
        return cleaned

    def _extract_rdfs_comment_text(self, concept):
        """Texto completo rdfs:comment (sem tokenizar em keywords)."""
        try:
            if hasattr(concept, 'comment'):
                if isinstance(concept.comment, list):
                    return ' '.join(str(c) for c in concept.comment).lower()
                return str(concept.comment).lower()
        except Exception:
            pass
        return ''

    def _feature_in_ontology_text(self, name_clean, concept, labels, comment_text):
        """Verifica se tokens da feature aparecem em labels ou comentários OWL."""
        if not name_clean or len(name_clean) < 3:
            return 0.0, None

        search_tokens = set()
        stripped = self._strip_feature_affixes(name_clean)
        for token_source in (name_clean, stripped):
            search_tokens.update(re.findall(r'[a-z0-9]{3,}', token_source))

        if not search_tokens:
            return 0.0, None

        best_score = 0.0
        best_type = None

        for label in labels:
            label_norm = self._normalize_name(label)
            label_tokens = set(re.findall(r'[a-z0-9]{3,}', label.lower()))
            if name_clean in label_norm or stripped in label_norm:
                return 0.80, 'label_substring'
            if len(stripped) >= 4 and stripped in label_norm:
                return 0.80, 'feature_in_label'
            if len(stripped) >= 4 and label_norm in stripped:
                return 0.78, 'label_in_feature'
            overlap = search_tokens & label_tokens
            if overlap:
                ratio = len(overlap) / max(len(search_tokens), 1)
                score = 0.55 + ratio * 0.25
                if score > best_score:
                    best_score = score
                    best_type = 'label_token_overlap'

        if comment_text:
            comment_norm = re.sub(r'[^a-z0-9\s]', ' ', comment_text)
            if name_clean in comment_norm.replace(' ', '') or stripped in comment_norm.replace(' ', ''):
                return 0.75, 'comment_substring'
            comment_tokens = set(re.findall(r'[a-z0-9]{3,}', comment_norm))
            overlap = search_tokens & comment_tokens
            if overlap:
                ratio = len(overlap) / max(len(search_tokens), 1)
                score = 0.50 + ratio * 0.30
                if score > best_score:
                    best_score = score
                    best_type = 'comment_token_overlap'

        return best_score, best_type

    def _find_matching_concept(self, name, ontology_entities, allow_target_classes=False):
        """Matching fuzzy: Class, DatatypeProperty ou ObjectProperty (nome + rdfs:label)."""
        cache_key = f"match_{'target' if allow_target_classes else 'feat'}_{name}"
        if cache_key in self.mapping_cache:
            cached = self.mapping_cache[cache_key]
            if cached is None:
                return None
            if isinstance(cached, dict):
                return cached
            return {
                'concept': cached,
                'matched_entity': cached,
                'entity_type': self.mapping_cache.get(f"{cache_key}_type"),
                'score': self.mapping_scores.get(name, 0.5),
                'match_type': 'cached',
            }

        name_clean = self._normalize_name(name)
        name_stripped = self._normalize_name(self._strip_feature_affixes(name))
        candidates = []

        for concept in ontology_entities:
            if not allow_target_classes and self._is_excluded_from_feature_matching(concept):
                continue
            concept_name = concept.name.lower()
            concept_clean = self._normalize_name(concept.name)
            entity_type = self._get_ontology_entity_type(concept)
            labels = self._extract_rdfs_labels(concept)
            label_matches = [self._normalize_name(label) for label in labels]
            comment_text = self._extract_rdfs_comment_text(concept)

            score = 0.0
            match_type = None

            for probe in (name_clean, name_stripped):
                if not probe:
                    continue
                if probe == concept_clean:
                    score = max(score, 1.0)
                    match_type = 'exact'
                elif any(probe == label for label in label_matches):
                    score = max(score, 0.95)
                    match_type = 'rdfs_label'
                elif len(probe) >= 4 and probe in concept_clean:
                    score = max(score, 0.80)
                    match_type = 'feature_in_concept'
                elif len(probe) >= 4 and concept_clean in probe:
                    score = max(score, 0.78)
                    match_type = 'concept_in_feature'
                elif probe in concept_clean or concept_clean in probe:
                    overlap_ratio = min(len(probe), len(concept_clean)) / max(len(probe), len(concept_clean))
                    score = max(score, 0.58 + overlap_ratio * 0.22)
                    match_type = 'substring'
                else:
                    similarity = difflib.SequenceMatcher(None, probe, concept_clean).ratio()
                    if similarity > 0.55:
                        score = max(score, similarity * 0.88)
                        match_type = 'similarity'

                for label in label_matches:
                    if len(probe) >= 4 and probe in label:
                        score = max(score, 0.80)
                        match_type = 'feature_in_label'
                    label_similarity = difflib.SequenceMatcher(None, probe, label).ratio()
                    if label_similarity > score:
                        score = label_similarity * 0.90
                        match_type = 'label_similarity'

                text_score, text_type = self._feature_in_ontology_text(
                    probe, concept, labels, comment_text
                )
                if text_score > score:
                    score = text_score
                    match_type = text_type

                probe_words = set(re.findall(r'[a-z0-9]{3,}', probe))
                concept_words = set(re.findall(r'[a-z0-9]{3,}', concept_clean))
                if probe_words and concept_words:
                    word_overlap = len(probe_words & concept_words) / max(len(probe_words), len(concept_words))
                    if word_overlap > 0.35:
                        score = max(score, word_overlap * 0.78)
                        match_type = match_type or 'word_overlap'

            if score >= self.PARTIAL_MATCH_THRESHOLD:
                if score < 0.65:
                    score *= 0.85
                    match_type = (match_type or 'partial') + '_low_confidence'
                candidates.append({
                    'concept': concept.name,
                    'matched_entity': concept.name,
                    'entity_type': entity_type,
                    'score': score,
                    'match_type': match_type,
                    'labels': labels,
                })

        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_match = dict(candidates[0])
            runner_up_score = float(candidates[1]['score']) if len(candidates) > 1 else 0.0
            best_match['runner_up_score'] = runner_up_score
            best_match['ambiguity_margin'] = float(best_match['score']) - runner_up_score
            best_match['accepted'] = bool(
                best_match['score'] >= self.PARTIAL_MATCH_THRESHOLD
                and (
                    best_match['score'] >= 0.98
                    or best_match['ambiguity_margin'] >= self.MATCH_AMBIGUITY_MARGIN
                )
            )
            if not best_match['accepted']:
                self.mapping_cache[cache_key] = None
                return None
            self.mapping_cache[cache_key] = best_match
            self.mapping_cache[f"{cache_key}_type"] = best_match.get('entity_type')
            self.mapping_scores[name] = best_match['score']
            return best_match

        self.mapping_cache[cache_key] = None
        return None
    
    def _normalize_name(self, name):

        name = self._camel_to_snake(name)
        normalized = re.sub(r'[^a-z0-9]', '', name.lower())
        normalized = re.sub(r'\d+$', '', normalized)
        normalized = self._strip_feature_affixes(normalized)
        normalized = re.sub(r'\d+$', '', normalized)
        normalized = re.sub(r'^(has|is|contains|hasvalue|hasproperty)', '', normalized)
        return normalized
    
    def _extract_rdfs_labels(self, concept):
        labels = []
        try:
            # rdfs:label
            if hasattr(concept, 'label'):
                if isinstance(concept.label, list):
                    labels.extend([str(l) for l in concept.label])
                else:
                    labels.append(str(concept.label))
            
            # skos:prefLabel
            if hasattr(concept, 'prefLabel'):
                if isinstance(concept.prefLabel, list):
                    labels.extend([str(l) for l in concept.prefLabel])
                else:
                    labels.append(str(concept.prefLabel))
            
            # rdfs:comment (pode conter descrições úteis)
            if hasattr(concept, 'comment'):
                comment = str(concept.comment) if not isinstance(concept.comment, list) else str(concept.comment[0])
                # Extrai palavras-chave do comentário
                words = re.findall(r'\b[a-z]{3,}\b', comment.lower())
                labels.extend(words[:5])  # Limita a 5 palavras-chave
                
        except Exception:
            pass
        
        return [l.lower().strip() for l in labels if l]
    
    def _extract_semantic_properties(self, feature_name, ontology_properties):
        properties = []
        feature_lower = feature_name.lower()
        
        for prop in ontology_properties:
            prop_name = prop.name.lower()
            if feature_lower in prop_name or prop_name in feature_lower:
                properties.append(prop.name)
        
        return properties
    
    def _extract_semantic_relationships(self, ontology_classes, ontology_properties):
        relationships = []
        hierarchy_map = defaultdict(list)
        
        # Extrai relacionamentos hierárquicos (subclasse de)
        for concept in ontology_classes:
            concept_name = concept.name
            if hasattr(concept, 'is_a'):
                for parent in concept.is_a:
                    parent_name = parent.name if hasattr(parent, 'name') else str(parent)
                    hierarchy_map[parent_name].append(concept_name)
                    relationships.append({
                        'type': 'hierarchy',
                        'child': concept_name,
                        'parent': parent_name,
                        'relation': 'subClassOf'
                    })
            
            # Extrai equivalent classes
            if hasattr(concept, 'equivalent_to'):
                for equiv in concept.equivalent_to:
                    equiv_name = equiv.name if hasattr(equiv, 'name') else str(equiv)
                    relationships.append({
                        'type': 'equivalence',
                        'class1': concept_name,
                        'class2': equiv_name,
                        'relation': 'equivalentClass'
                    })
            
            # Extrai disjoint classes
            if hasattr(concept, 'disjoint_with'):
                for disjoint in concept.disjoint_with:
                    disjoint_name = disjoint.name if hasattr(disjoint, 'name') else str(disjoint)
                    relationships.append({
                        'type': 'disjoint',
                        'class1': concept_name,
                        'class2': disjoint_name,
                        'relation': 'disjointWith'
                    })
        
        # Extrai relacionamentos de propriedade com mais detalhes
        for prop in ontology_properties:
            prop_name = prop.name
            prop_info = {
                'type': 'property',
                'property': prop_name,
                'relation': 'property'
            }
            property_parents = []
            try:
                for parent in list(getattr(prop, 'is_a', []) or []):
                    parent_name = getattr(parent, 'name', None)
                    if not parent_name or parent_name == prop_name:
                        continue
                    property_parents.append(parent_name)
                    hierarchy_map[parent_name].append(prop_name)
                    relationships.append({
                        'type': 'property_hierarchy',
                        'child': prop_name,
                        'parent': parent_name,
                        'relation': 'subPropertyOf',
                    })
            except (AttributeError, TypeError, RuntimeError):
                pass
            if property_parents:
                prop_info['parents'] = property_parents
            
            # Domain
            if hasattr(prop, 'domain'):
                domain_list = prop.domain if isinstance(prop.domain, list) else [prop.domain]
                domains = []
                for domain in domain_list:
                    domain_name = domain.name if hasattr(domain, 'name') else str(domain)
                    domains.append(domain_name)
                prop_info['domain'] = domains[0] if len(domains) == 1 else domains
                prop_info['domains'] = domains
            
            # Range
            if hasattr(prop, 'range'):
                range_val = prop.range
                if hasattr(range_val, 'name'):
                    prop_info['range'] = range_val.name
                    prop_info['range_type'] = 'class'
                else:
                    # Pode ser um datatype (xsd:integer, xsd:float, etc)
                    range_str = str(range_val)
                    prop_info['range'] = range_str
                    prop_info['range_type'] = 'datatype'
                    # Extrai informações do datatype
                    self._extract_datatype_info(range_str, prop_name)
                prop_info['range_obj'] = range_val
            
            relationships.append(prop_info)
            
            # Guarda restrições de propriedade
            if 'domain' in prop_info:
                if prop_name not in self.property_constraints:
                    self.property_constraints[prop_name] = {}
                self.property_constraints[prop_name]['domain'] = prop_info.get('domain')
                self.property_constraints[prop_name]['range'] = prop_info.get('range')
                self.property_constraints[prop_name]['range_type'] = prop_info.get('range_type', 'class')
            
            # Verifica propriedades inversas
            if hasattr(prop, 'inverse_property'):
                inverse = prop.inverse_property
                relationships.append({
                    'type': 'inverse_property',
                    'property1': prop_name,
                    'property2': inverse.name if hasattr(inverse, 'name') else str(inverse),
                    'relation': 'inverseOf'
                })
        
        self.domain_knowledge['relationships'] = relationships
        self.ontology_hierarchy = dict(hierarchy_map)
    
    def _extract_datatype_info(self, datatype_str, prop_name):
        # Padrões comuns de datatypes
        if 'integer' in datatype_str.lower() or 'int' in datatype_str.lower():
            self.datatype_ranges[prop_name] = {
                'type': 'integer',
                'min': None,
                'max': None
            }
        elif 'float' in datatype_str.lower() or 'double' in datatype_str.lower() or 'decimal' in datatype_str.lower():
            self.datatype_ranges[prop_name] = {
                'type': 'float',
                'min': None,
                'max': None
            }
        elif 'boolean' in datatype_str.lower() or 'bool' in datatype_str.lower():
            self.datatype_ranges[prop_name] = {
                'type': 'boolean',
                'min': 0,
                'max': 1
            }
        elif 'nonNegativeInteger' in datatype_str.lower():
            self.datatype_ranges[prop_name] = {
                'type': 'integer',
                'min': 0,
                'max': None
            }
        elif 'positiveInteger' in datatype_str.lower():
            self.datatype_ranges[prop_name] = {
                'type': 'integer',
                'min': 1,
                'max': None
            }
    
    def _get_entity_object(self, entity_name, ontology_entities):
        for entity in ontology_entities:
            if entity.name == entity_name:
                return entity
        return None

    def _get_concept_object(self, concept_name, ontology_entities):
        return self._get_entity_object(concept_name, ontology_entities)
    
    def _extract_entity_details(self, entity):
        details = self._extract_concept_details(entity)
        details['entity_type'] = self._get_ontology_entity_type(entity)
        if details['entity_type'] in ('datatype_property', 'object_property'):
            domains = []
            try:
                if hasattr(entity, 'domain'):
                    domain_list = entity.domain if isinstance(entity.domain, list) else [entity.domain]
                    for domain in domain_list:
                        if hasattr(domain, 'name'):
                            domains.append(domain.name)
            except Exception:
                pass
            details['domains'] = domains
            parents = []
            try:
                for parent in list(getattr(entity, 'is_a', []) or []):
                    parent_name = getattr(parent, 'name', None)
                    if parent_name:
                        parents.append(parent_name)
            except (AttributeError, TypeError, RuntimeError):
                pass
            details['parents'] = parents
            try:
                if hasattr(entity, 'range'):
                    range_val = entity.range
                    if hasattr(range_val, 'name'):
                        details['range'] = range_val.name
                    else:
                        details['range'] = str(range_val)
            except Exception:
                pass
        return details

    def _extract_concept_details(self, concept):
        details = {
            'labels': self._extract_rdfs_labels(concept),
            'comment': '',
            'properties': [],
            'entity_type': self._get_ontology_entity_type(concept),
        }
        
        # Comentário RDFS
        try:
            if hasattr(concept, 'comment'):
                if isinstance(concept.comment, list):
                    details['comment'] = ' '.join([str(c) for c in concept.comment])
                else:
                    details['comment'] = str(concept.comment)
        except Exception:
            pass
        
        return details
    
    def _find_best_fallback_match(self, name, ontology_entities, min_score=None):
        if min_score is None:
            min_score = self.PARTIAL_MATCH_THRESHOLD
        name_clean = self._normalize_name(name)
        name_stripped = self._normalize_name(self._strip_feature_affixes(name))
        candidates = []
        
        for concept in ontology_entities:
            if self._is_excluded_from_feature_matching(concept):
                continue
            concept_clean = self._normalize_name(concept.name)
            entity_type = self._get_ontology_entity_type(concept)
            labels = self._extract_rdfs_labels(concept)
            comment_text = self._extract_rdfs_comment_text(concept)

            for probe in (name_clean, name_stripped):
                if not probe:
                    continue
                similarity = difflib.SequenceMatcher(None, probe, concept_clean).ratio()
                if similarity >= min_score:
                    candidates.append({
                        'name': concept.name,
                        'score': similarity,
                        'type': 'name_similarity',
                        'entity_type': entity_type,
                    })
                
                for label in labels:
                    label_norm = self._normalize_name(label)
                    label_sim = difflib.SequenceMatcher(None, probe, label_norm).ratio()
                    if label_sim >= min_score:
                        candidates.append({
                            'name': concept.name,
                            'score': label_sim,
                            'type': 'label_similarity',
                            'entity_type': entity_type,
                        })

                text_score, text_type = self._feature_in_ontology_text(
                    probe, concept, labels, comment_text
                )
                if text_score >= min_score:
                    candidates.append({
                        'name': concept.name,
                        'score': text_score,
                        'type': text_type or 'comment_match',
                        'entity_type': entity_type,
                    })
        
        if candidates:
            return max(candidates, key=lambda x: x['score'])
        return None

    def _resolve_ontology_sample_size(self, n_real_samples, requested=None):
        """Volume sintético conforme preset ou default ontologia."""
        limits = getattr(self, '_training_limits', {}) or {}
        if limits.get('sample_size'):
            return int(limits['sample_size'])
        target = self.ONTOLOGY_SAMPLE_SIZE_DEFAULT
        if requested is not None and requested >= self.ONTOLOGY_SAMPLE_SIZE_MIN:
            target = int(min(self.ONTOLOGY_SAMPLE_SIZE_MAX, max(requested, self.ONTOLOGY_SAMPLE_SIZE_MIN)))
        return target

    def _cache_real_data_bounds(self, X_encoded, feature_names):
        self._real_data_bounds = {}
        for idx in range(X_encoded.shape[1]):
            col = X_encoded[:, idx]
            self._real_data_bounds[idx] = {
                'min': float(np.min(col)),
                'max': float(np.max(col)),
                'p05': float(np.percentile(col, 5)),
                'p95': float(np.percentile(col, 95)),
                'name': feature_names[idx] if feature_names and idx < len(feature_names) else f'feature_{idx}',
            }

    def _compute_feature_centrality_scores(self, feature_names):
        """Features mapeadas a conceitos centrais na hierarquia OWL recebem score mais alto."""
        child_counts = {parent: len(children) for parent, children in self.ontology_hierarchy.items()}
        max_children = max(child_counts.values()) if child_counts else 1
        scores = {}
        for idx, fname in enumerate(feature_names):
            info = self.feature_semantics.get(idx, {})
            concept = info.get('ontology_concept')
            mapping = float(info.get('mapping_score', 0.0))
            hierarchy_boost = 0.0
            if concept:
                hierarchy_boost = child_counts.get(concept, 0) / max(max_children, 1)
            onto_boost = 0.12 if self._is_ontology_derived_feature(fname) else 0.0
            scores[idx] = float(min(1.0, 0.42 * mapping + 0.38 * hierarchy_boost + onto_boost + 0.08))
        return scores

    def _sample_by_ontology_central_density(self, X_encoded, n_samples, centrality):
        """Amostras com menor ruído nas dimensões ontológicamente centrais."""
        if n_samples <= 0:
            return np.empty((0, X_encoded.shape[1]))
        if not centrality:
            return self._sample_by_density(X_encoded, n_samples)

        n_features = X_encoded.shape[1]
        weights = np.array([centrality.get(i, 0.1) for i in range(n_features)], dtype=float)
        weights = weights / (weights.sum() + 1e-9)

        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_encoded), size=n_samples, replace=True)
        X_central = X_encoded[indices].astype(float, copy=True)
        col_std = np.std(X_encoded, axis=0) + 1e-6

        for col_idx in range(n_features):
            noise_scale = col_std[col_idx] * (0.04 + 0.14 * (1.0 - weights[col_idx]))
            X_central[:, col_idx] += rng.normal(0.0, noise_scale, size=n_samples)
        return X_central

    def _sample_mlp_decision_boundary(self, mlp_model, X_encoded, n_samples, centrality):
        """Combina incerteza do MLP com ênfase em features centrais."""
        if n_samples <= 0:
            return np.empty((0, X_encoded.shape[1]))
        X_unc = self._sample_uncertainty_regions(mlp_model, X_encoded, n_samples)
        if not centrality:
            return X_unc

        n_features = X_encoded.shape[1]
        weights = np.array([centrality.get(i, 0.1) for i in range(n_features)], dtype=float)
        weights = weights / (weights.sum() + 1e-9)
        rng = np.random.default_rng(43)
        col_std = np.std(X_encoded, axis=0) + 1e-6
        n_rows = len(X_unc)
        for col_idx in range(n_features):
            X_unc[:, col_idx] += rng.normal(
                0.0, col_std[col_idx] * weights[col_idx] * 0.05, size=n_rows
            )
        return X_unc

    def _generate_ontology_aware_synthetic_data(
        self, mlp_original, X_encoded, sample_size, feature_names
    ):
        """
        Gera sintéticos no espaço ARFF original.
        Regra: usar sempre MLP original para amostragem/incerteza/fronteira MLP.
        MLP_Onto só entra depois da expansão onto_* (rotulagem em extract_tree).
        """
        X_encoded, feature_names, _ = self._base_matrix_and_names(X_encoded, feature_names)
        sampling_mlp = self._oracle_for_base_sampling(mlp_original)
        if sampling_mlp is None:
            raise ValueError(
                "Oráculo de amostragem indisponível: é necessário MLP original (espaço ARFF)."
            )
        n_base = X_encoded.shape[1]
        n_mlp = getattr(sampling_mlp, 'n_features_in_', None)
        if n_mlp is not None and n_mlp != n_base:
            raise ValueError(
                f"Oráculo de amostragem incompatível: MLP espera {n_mlp} features, "
                f"matriz ARFF tem {n_base}. Use MLP original, não MLP_Onto."
            )

        if not self.has_active_ontology:
            return self._original_extractor._generate_intelligent_synthetic_data(
                sampling_mlp, X_encoded, sample_size
            )

        centrality = self.feature_centrality or self._compute_feature_centrality_scores(feature_names)
        n_features = X_encoded.shape[1]

        # Premium mix: centrality + domínio + fronteira MLP + exploração + amostras Trepan-Original
        legacy_samples = int(sample_size * 0.20)
        central_samples = int(sample_size * 0.24)
        domain_samples = int(sample_size * 0.24)
        uncertainty_samples = int(sample_size * 0.12)
        boundary_samples = int(sample_size * 0.12)
        random_samples = (
            sample_size
            - legacy_samples
            - central_samples
            - domain_samples
            - uncertainty_samples
            - boundary_samples
        )

        X_legacy = self._original_extractor._generate_intelligent_synthetic_data(
            sampling_mlp, X_encoded, legacy_samples
        )
        X_central = self._sample_by_ontology_central_density(
            X_encoded, central_samples, centrality
        )
        X_domain = self._sample_by_domain_knowledge(X_encoded, domain_samples, feature_names)
        X_uncertainty = self._sample_uncertainty_regions(
            sampling_mlp, X_encoded, uncertainty_samples
        )
        X_boundary = self._sample_mlp_decision_boundary(
            sampling_mlp, X_encoded, boundary_samples, centrality
        )
        X_random = np.random.uniform(
            low=np.min(X_encoded, axis=0),
            high=np.max(X_encoded, axis=0),
            size=(max(0, random_samples), n_features),
        )

        parts = [p for p in [X_legacy, X_central, X_domain, X_uncertainty, X_boundary, X_random] if len(p) > 0]
        X_combined = np.vstack(parts) if parts else X_central
        X_combined = self._apply_ontology_correlations(X_combined, feature_names)
        return X_combined

    def _concept_to_feature_indices(self, feature_names):
        """Mapeia nome de conceito OWL → índices de features associadas."""
        concept_map = defaultdict(list)
        for idx, info in self.feature_semantics.items():
            concept = info.get('ontology_concept')
            if concept and not info.get('general_fallback'):
                concept_map[concept].append(idx)
            for prop in info.get('semantic_properties', []):
                concept_map[prop].append(idx)
        for idx, fname in enumerate(feature_names):
            base = self._ontology_feature_base_name(fname).lower()
            concept_map[base].append(idx)
        return concept_map

    def _build_ontology_feature_dependencies(self, feature_names):
        """Extrai pares (dependente, base) a partir das relações OWL."""
        dependencies = []
        seen = set()
        concept_map = self._concept_to_feature_indices(feature_names)

        for rel in self.domain_knowledge.get('relationships', []):
            rel_type = rel.get('type')
            if rel_type == 'hierarchy':
                parent, child = rel.get('parent'), rel.get('child')
                for c_idx in concept_map.get(child, []):
                    for p_idx in concept_map.get(parent, []):
                        if c_idx != p_idx:
                            key = (c_idx, p_idx)
                            if key not in seen:
                                seen.add(key)
                                dependencies.append({
                                    'dependent': c_idx,
                                    'base': p_idx,
                                    'strength': 0.55,
                                    'relation': 'subClassOf',
                                })
            elif rel_type == 'property':
                prop_name = rel.get('property', '')
                domain = rel.get('domain')
                domains = rel.get('domains', [])
                if isinstance(domain, list):
                    domains = domain
                elif domain:
                    domains = [domain]

                prop_indices = concept_map.get(prop_name, [])
                for d in domains:
                    domain_indices = concept_map.get(d, [])
                    for p_idx in prop_indices:
                        for d_idx in domain_indices:
                            if p_idx != d_idx:
                                key = (p_idx, d_idx)
                                if key not in seen:
                                    seen.add(key)
                                    dependencies.append({
                                        'dependent': p_idx,
                                        'base': d_idx,
                                        'strength': 0.65,
                                        'relation': 'property_domain',
                                    })

        for prop_name, prop_constraints in self.property_constraints.items():
            domain = prop_constraints.get('domain')
            domains = [domain] if isinstance(domain, str) else (domain or [])
            for d in domains:
                for idx, fname in enumerate(feature_names):
                    if prop_name.lower() in fname.lower() or fname.lower() in prop_name.lower():
                        for d_idx in concept_map.get(d, []):
                            if idx != d_idx:
                                key = (idx, d_idx)
                                if key not in seen:
                                    seen.add(key)
                                    dependencies.append({
                                        'dependent': idx,
                                        'base': d_idx,
                                        'strength': 0.50,
                                        'relation': 'property_name',
                                    })
        return dependencies

    def _apply_ontology_correlations(self, X, feature_names):
        """Aplica correlações entre colunas com base nas relações ontológicas."""
        if X is None or len(X) == 0:
            return X
        dependencies = self._build_ontology_feature_dependencies(feature_names)
        if not dependencies:
            return X

        X_corr = np.array(X, dtype=float, copy=True)
        n_rows, n_features = X_corr.shape
        col_std = np.std(X_corr, axis=0) + 1e-6
        rng = np.random.default_rng(44)

        for dep in dependencies:
            dep_idx, base_idx = dep['dependent'], dep['base']
            if dep_idx >= n_features or base_idx >= n_features:
                continue
            strength = dep['strength']
            base_vals = X_corr[:, base_idx]
            base_norm = (base_vals - np.mean(base_vals)) / (np.std(base_vals) + 1e-6)
            noise = rng.normal(0, 0.12, size=n_rows)
            adjustment = strength * base_norm * col_std[dep_idx] + noise * col_std[dep_idx] * 0.08
            X_corr[:, dep_idx] = X_corr[:, dep_idx] * (1.0 - strength * 0.45) + adjustment

        bounds = self._real_data_bounds
        for j in range(n_features):
            b = bounds.get(j, {})
            if b:
                lo, hi = b.get('min'), b.get('max')
                if lo is not None and hi is not None:
                    X_corr[:, j] = np.clip(X_corr[:, j], lo, hi)
        return X_corr
    
    def _sample_by_domain_knowledge(self, X_encoded, n_samples, feature_names):
        n_features = X_encoded.shape[1]
        X_domain = np.zeros((n_samples, n_features))
        
        # Identifica features relacionadas através de propriedades ontológicas
        feature_relationships = self._identify_related_features(feature_names)
        
        for i in range(n_samples):
            # Determina valores baseados em grupos relacionados primeiro
            assigned_features = set()
            
            # Processa grupos de features relacionadas
            for group in feature_relationships:
                if not group:  # Skip empty groups
                    continue
                
                # Gera valores para features relacionadas considerando restrições
                group_values = self._generate_related_feature_values(
                    group, X_encoded, feature_names
                )
                
                for feat_idx, value in group_values.items():
                    if feat_idx not in assigned_features:
                        X_domain[i, feat_idx] = value
                        assigned_features.add(feat_idx)
            
            # Para features não relacionadas ou não em grupos, gera individualmente
            for j in range(n_features):
                if j in assigned_features:
                    continue
                    
                feature_info = self.feature_semantics.get(j, {})
                concept_type = feature_info.get('concept_type', 'general')
                
                # Usa informações de datatype se disponível
                datatype_info = None
                ontology_concept = feature_info.get('ontology_concept')
                if ontology_concept:
                    # Verifica se há restrições de datatype nas propriedades
                    for prop_name, prop_constraints in self.property_constraints.items():
                        if prop_name.lower() in feature_names[j].lower():
                            range_type = prop_constraints.get('range_type')
                            if range_type == 'datatype':
                                datatype_info = self.datatype_ranges.get(prop_name)
                                break
                
                # Gera valor baseado no tipo de conceito e restrições
                value = self._generate_value_by_concept_type(
                    j, concept_type, X_encoded, datatype_info
                )
                X_domain[i, j] = value
        
        return X_domain
    
    def _identify_related_features(self, feature_names):
        groups = []
        processed = set()
        
        # Agrupa features que compartilham propriedades ontológicas
        for prop_name, prop_constraints in self.property_constraints.items():
            domain = prop_constraints.get('domain')
            if not domain:
                continue
                
            # Encontra features relacionadas a este domínio
            related_indices = []
            for i, feature in enumerate(feature_names):
                if i in processed:
                    continue
                    
                feature_info = self.feature_semantics.get(i, {})
                ontology_concept = feature_info.get('ontology_concept', '')
                
                # Verifica se a feature pertence ao domínio da propriedade
                if isinstance(domain, str):
                    if domain.lower() in feature.lower() or feature.lower() in domain.lower():
                        related_indices.append(i)
                        processed.add(i)
                elif isinstance(domain, list):
                    for d in domain:
                        if d.lower() in feature.lower() or feature.lower() in d.lower():
                            related_indices.append(i)
                            processed.add(i)
                            break
            
            if len(related_indices) > 1:
                groups.append(related_indices)
        
        # Se não encontrou grupos, agrupa por hierarquia ontológica
        if not groups:
            n_feat = len(feature_names)
            for i, feature_info in self.feature_semantics.items():
                if i >= n_feat:
                    continue
                ontology_concept = feature_info.get('ontology_concept')
                if ontology_concept and ontology_concept in self.ontology_hierarchy:
                    related = []
                    for j, other_info in self.feature_semantics.items():
                        if j >= n_feat or i == j:
                            continue
                        other_concept = other_info.get('ontology_concept')
                        if other_concept in self.ontology_hierarchy.get(ontology_concept, []):
                            related.append(j)
                    if related:
                        groups.append([i] + related)
        
        return groups if groups else [[i] for i in range(len(feature_names))]
    
    def _generate_related_feature_values(self, feature_indices, X_encoded, feature_names):
        values = {}
        dependencies = self._build_ontology_feature_dependencies(feature_names)
        dep_by_dependent = {d['dependent']: d for d in dependencies}

        n_cols = X_encoded.shape[1]
        for idx in feature_indices:
            if idx >= n_cols:
                continue
            feature_info = self.feature_semantics.get(idx, {})
            concept_type = feature_info.get('concept_type', 'general')
            value = self._generate_value_by_concept_type(idx, concept_type, X_encoded, None)

            if idx in dep_by_dependent:
                dep = dep_by_dependent[idx]
                base_idx = dep['base']
                if base_idx >= n_cols:
                    continue
                if base_idx in values:
                    base_val = values[base_idx]
                elif base_idx in feature_indices:
                    base_val = self._generate_value_by_concept_type(
                        base_idx,
                        self.feature_semantics.get(base_idx, {}).get('concept_type', 'general'),
                        X_encoded, None,
                    )
                else:
                    col = X_encoded[:, base_idx]
                    base_val = float(np.random.choice(col))

                col_dep = X_encoded[:, idx]
                col_std = float(np.std(col_dep)) + 1e-6
                col_mean = float(np.mean(col_dep))
                base_std = float(np.std(X_encoded[:, base_idx])) + 1e-6
                base_mean = float(np.mean(X_encoded[:, base_idx]))
                normalized_base = (base_val - base_mean) / base_std
                value = col_mean + dep['strength'] * normalized_base * col_std
                value += np.random.normal(0, col_std * 0.08)
                value = float(np.clip(value, np.min(col_dep), np.max(col_dep)))

            values[idx] = value

        return values
    
    def _generate_value_by_concept_type(self, feature_idx, concept_type, X_encoded, datatype_info):
        col_data = X_encoded[:, feature_idx]
        col_min = np.min(col_data)
        col_max = np.max(col_data)
        col_mean = np.mean(col_data)
        col_std = np.std(col_data)
        
        # Aplica restrições de datatype se disponível
        if datatype_info:
            dtype_type = datatype_info.get('type')
            dtype_min = datatype_info.get('min')
            dtype_max = datatype_info.get('max')
            
            if dtype_min is not None:
                col_min = max(col_min, dtype_min)
            if dtype_max is not None:
                col_max = min(col_max, dtype_max)
        
        # Gera valor baseado no tipo de conceito
        if concept_type == 'temporal':
            value = np.random.normal(col_mean, col_std * 0.5)
            value = max(col_min, min(col_max, value))
        elif concept_type == 'dimensional':
            # Valores dimensionais tendem a ser positivos
            if col_min >= 0:
                value = np.random.exponential(max(col_mean, 1))
                value = min(col_max, value)
            else:
                value = np.random.normal(col_mean, col_std * 0.7)
        elif concept_type == 'quantitative':
            value = np.random.normal(col_mean, col_std)
            value = max(col_min, min(col_max, value))
        elif concept_type == 'categorical':
            unique_values = np.unique(col_data)
            value = np.random.choice(unique_values)
        elif concept_type in ('general', 'quantitative') or concept_type.startswith('General_'):
            if 'Categorical' in str(concept_type):
                unique_values = np.unique(col_data)
                value = np.random.choice(unique_values)
            else:
                value = np.random.uniform(col_min, col_max)
        else:
            value = np.random.uniform(col_min, col_max)
        
        # Aplica restrições de datatype novamente
        if datatype_info:
            dtype_type = datatype_info.get('type')
            if dtype_type == 'integer':
                value = int(round(value))
            elif dtype_type == 'boolean':
                value = 1.0 if value > 0.5 else 0.0
        
        return value
    
    def _sample_by_density(self, X_encoded, n_samples):
        from sklearn.neighbors import KernelDensity
        
        # Estima densidade dos dados originais
        kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
        kde.fit(X_encoded)
        
        # Gera amostras baseadas na densidade estimada
        X_density = kde.sample(n_samples)
        return X_density
    
    def _sample_uncertainty_regions(self, mlp_model, X_encoded, n_samples):
        sampling_mlp = self._resolve_oracle_for_matrix(X_encoded, mlp_model)
        if hasattr(sampling_mlp, 'predict_proba'):
            probas = self._mlp_predict_proba(sampling_mlp, X_encoded)
            # Calcula entropia como medida de incerteza
            entropy = -np.sum(probas * np.log(probas + 1e-10), axis=1)
            
            rng = np.random.default_rng(42)
            if n_samples <= len(X_encoded):
                high_uncertainty_indices = np.argsort(entropy)[-n_samples:]
                X_uncertainty_base = X_encoded[high_uncertainty_indices]
            else:
                ranked = np.argsort(entropy)
                high_uncertainty_indices = rng.choice(ranked, size=n_samples, replace=True)
                X_uncertainty_base = X_encoded[high_uncertainty_indices]

            noise = rng.normal(0, 0.1, X_uncertainty_base.shape)
            return X_uncertainty_base + noise
        else:
            # Fallback para geração aleatória se não há predict_proba
            return np.random.uniform(
                low=np.min(X_encoded, axis=0),
                high=np.max(X_encoded, axis=0),
                size=(n_samples, X_encoded.shape[1])
            )
    
    def _augment_synthetic_for_onto_features(self, X, y, weights, feature_names):
        """Duplica amostras com ênfase em features onto_* (sem alterar escala de inferência)."""
        if feature_names is None or not self.has_active_ontology:
            return X, y, weights
        X = np.asarray(X, dtype=float)
        n_cols = X.shape[1]
        onto_indices = self._valid_column_indices([
            i for i, name in enumerate(feature_names)
            if self._is_ontology_derived_feature(name)
        ], n_cols)
        if not onto_indices:
            return X, y, weights
        boost = int(max(1, round(self.onto_feature_bias_weight)))
        X_parts = [X]
        y_parts = [y]
        w_parts = [weights]
        rng = np.random.default_rng(42)
        for _ in range(boost - 1):
            X_extra = np.array(X, copy=True)
            noise = rng.normal(0, 0.03, size=(len(X), len(onto_indices)))
            for col, idx in enumerate(onto_indices):
                X_extra[:, idx] += noise[:, col]
            X_parts.append(X_extra)
            y_parts.append(y)
            w_parts.append(weights * 1.15)
        return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(w_parts)

    def _build_ontology_sample_weights(self, X_synthetic, feature_names):
        """Prioriza amostras alinhadas a features com alto onto_concept_score / centralidade."""
        weights = np.ones(len(X_synthetic), dtype=float)
        if not self.has_active_ontology or feature_names is None:
            return weights

        X_synthetic = np.asarray(X_synthetic, dtype=float)
        n_cols = X_synthetic.shape[1]
        feature_names = self._align_matrix_feature_names(feature_names, n_cols)

        centrality = self.feature_centrality
        priority_indices = []
        for idx, name in enumerate(feature_names):
            if idx >= n_cols:
                continue
            info = self.feature_semantics.get(idx, {})
            mapping = float(info.get('mapping_score', 0.0))
            central = float(centrality.get(idx, 0.0))
            if mapping >= 0.55 or central >= 0.45 or self._is_ontology_derived_feature(name):
                priority_indices.append((idx, mapping, central))

        if not priority_indices:
            return weights

        col_means = np.mean(X_synthetic, axis=0)
        col_stds = np.std(X_synthetic, axis=0) + 1e-6
        for row in range(len(X_synthetic)):
            boost = 0.0
            for idx, mapping, central in priority_indices:
                z = abs(X_synthetic[row, idx] - col_means[idx]) / col_stds[idx]
                if z <= 1.5:
                    boost += 0.25 * mapping + 0.20 * central
            weights[row] += boost

        return weights / max(np.mean(weights), 1e-6)

    def _build_feature_gain_weights(self, feature_names):
        """Peso elevado no ganho de informação para colunas onto_* e features mapeadas na OWL."""
        weights = np.ones(len(feature_names), dtype=float)
        for idx, name in enumerate(feature_names):
            info = self.feature_semantics.get(idx, {})
            if self._is_ontology_derived_feature(name) or info.get('ontology_derived'):
                weights[idx] = self.onto_feature_bias_weight
            elif info.get('ontology_mapped') and info.get('mapping_score', 0) >= self.PARTIAL_MATCH_THRESHOLD:
                weights[idx] = self.MAPPED_FEATURE_GAIN_BOOST
        return weights

    def _build_ontology_split_multipliers(self, feature_names):
        """
        Mantem prioridades semanticas como metadata.

        Escalar monotonamente uma coluna nao altera a ordem das observacoes nem
        o ganho de informacao de uma arvore univariada; portanto nao constitui
        um bias de split real. O efeito semantico e implementado por amostragem,
        pesos e penalizacao de coerencia, sem deformar valores/limiares.
        """
        multipliers = self._build_feature_gain_weights(feature_names)
        for idx, name in enumerate(feature_names):
            if self._is_ontology_derived_feature(name):
                prio = float(self.feature_semantics.get(idx, {}).get('split_priority', 1.0))
                multipliers[idx] = max(
                    multipliers[idx],
                    self.onto_feature_bias_weight * min(1.35, 0.85 + 0.15 * prio),
                )
        self._ontology_split_multipliers = multipliers
        return multipliers

    def apply_ontology_split_bias(self, X):
        """Compatibilidade: preserva a escala original das features."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return X_arr

    def _boost_sample_weights_for_mapped_features(self, X, sample_weights, feature_names, y=None):
        """Reforça amostras onde features mapeadas separam classes (efeito ~1.2x no ganho)."""
        X = np.asarray(X, dtype=float)
        n_cols = X.shape[1]
        feature_names = self._align_matrix_feature_names(feature_names or [], n_cols)
        gain_weights = self._build_feature_gain_weights(feature_names)[:n_cols]
        mapped_indices = self._valid_column_indices(
            np.where(gain_weights > 1.0)[0], n_cols
        )
        if len(mapped_indices) == 0:
            return sample_weights

        col_means = np.mean(X, axis=0)
        col_stds = np.std(X, axis=0) + 1e-6
        z_mapped = np.mean(
            np.abs((X[:, mapped_indices] - col_means[mapped_indices]) / col_stds[mapped_indices]),
            axis=1,
        )
        boost = 1.0 + (self.MAPPED_FEATURE_GAIN_BOOST - 1.0) * np.clip(z_mapped / 2.0, 0.0, 1.0)

        if y is not None and len(np.unique(y)) > 1:
            for idx in mapped_indices:
                col = X[:, idx]
                classes = np.unique(y)
                if len(classes) < 2:
                    continue
                class_means = np.array([np.mean(col[y == c]) for c in classes])
                spread = np.std(class_means) / (col_stds[idx] + 1e-6)
                if spread > 0.05:
                    centered = np.abs(col - col_means[idx]) / col_stds[idx]
                    boost += (self.MAPPED_FEATURE_GAIN_BOOST - 1.0) * spread * np.clip(centered, 0, 1.5) * 0.35

        return sample_weights * boost

    def _build_depth_fidelity_grid(self):
        """GridSearch interno: 3 profundidades fixas orientadas à fidelidade MLP."""
        return [5, 7, 10]

    def _score_tree_fidelity(self, tree, mlp_model, X_real, feature_names=None):
        """Score principal: fidelidade da árvore face ao MLP em dados reais."""
        names = feature_names or getattr(self, '_matrix_feature_names', None)
        X_oracle = self._align_matrix_to_feature_schema(
            X_real, names, getattr(self, '_X_augmented_reference', None)
        )
        y_mlp = self._mlp_predict(mlp_model, X_oracle, names)
        y_tree = tree.predict(self._prepare_tree_matrix(tree, X_real, names))
        return accuracy_score(y_mlp, y_tree)

    def _matrix_for_tree(self, tree, X, matrix_feature_names=None):
        """Sub-matriz com n_features da árvore (ARFF vs enriquecido)."""
        if tree is None or X is None:
            return X
        names = matrix_feature_names or getattr(self, '_matrix_feature_names', None)
        X = self._align_matrix_to_feature_schema(
            X, names, getattr(self, '_X_augmented_reference', None)
        ) if names else np.asarray(X, dtype=float)
        if hasattr(tree, 'tree_'):
            n_tree = int(getattr(tree.tree_, 'n_features', X.shape[1]))
        else:
            n_tree = int(getattr(tree, 'n_features_in_', X.shape[1]))
        if X.shape[1] == n_tree:
            return X
        raise ValueError(
            f"Árvore espera {n_tree} atributos, mas recebeu {X.shape[1]}; "
            "truncamento/zero-padding estão desativados."
        )

    def _measure_tree_performance(
        self, tree, mlp_model, X, y_true,
        matrix_feature_names=None, oracle_X=None,
    ):
        """Fidelidade (vs MLP) e precisão (vs rótulos reais) numa árvore."""
        if tree is None:
            return {'fidelity': 0.0, 'precision': 0.0, 'precision_macro': 0.0, 'precision_weighted': 0.0, 'recall_macro': 0.0, 'macro_f1': 0.0, 'balanced_accuracy': 0.0, 'accuracy': 0.0}
        X_arr = np.asarray(X, dtype=float)
        if matrix_feature_names is None:
            stored_original = list(
                getattr(self, '_original_feature_names', None)
                or getattr(self, '_base_feature_names', None)
                or []
            )
            names = (
                stored_original if len(stored_original) == X_arr.shape[1]
                else [f"feature_{i}" for i in range(X_arr.shape[1])]
            )
            X_aligned = X_arr
        else:
            names = self._tree_schema_feature_names(matrix_feature_names)
            X_aligned = self._align_real_for_tree(X_arr, names)
        X_oracle = oracle_X if oracle_X is not None else X_aligned
        if oracle_X is None and names and matrix_feature_names is not None:
            X_oracle = self._align_real_for_tree(X_oracle, names)
        y_mlp = self._mlp_predict(mlp_model, X_oracle, names)
        X_tree = self._prepare_tree_matrix(tree, X_aligned, names)
        y_tree = tree.predict(X_tree)
        # Métricas contra rótulos reais: Precision Macro é a métrica primária
        # do protocolo. Mantemos Precision Weighted apenas para auditoria.
        precision_macro = float(
            precision_score(y_true, y_tree, average='macro', zero_division=0)
        )
        return {
            'fidelity': float(accuracy_score(y_mlp, y_tree)),
            'precision': precision_macro,  # alias legado = Precision Macro
            'precision_macro': precision_macro,
            'precision_weighted': float(
                precision_score(y_true, y_tree, average='weighted', zero_division=0)
            ),
            'recall_macro': float(
                recall_score(y_true, y_tree, average='macro', zero_division=0)
            ),
            'macro_f1': float(
                f1_score(y_true, y_tree, average='macro', zero_division=0)
            ),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_tree)),
            'accuracy': float(accuracy_score(y_true, y_tree)),
            'metric_semantics': {
                'precision': 'precision_macro_alias',
                'precision_macro': 'precision_score_average_macro_against_real_labels',
                'precision_weighted': 'audit_only',
                'fidelity': 'agreement_with_explicit_oracle_argument',
            },
        }

    def _tree_dominates(self, reloaded_metrics, original_metrics, c45_metrics,
                        margin=None):
        """Critério de busca: fidelidade vs TREPAN Original e Precision Macro vs ambos.

        C4.5 é baseline supervisionado pelos rótulos reais e NÃO é oráculo. Por
        isso a sua fidelidade nunca participa da decisão de dominância.
        """
        if margin is None:
            margin = self.DOMINANCE_MARGIN
        return (
            reloaded_metrics['fidelity'] > original_metrics['fidelity'] + margin - 1e-9
            and reloaded_metrics['precision_macro'] > original_metrics['precision_macro'] + margin - 1e-9
            and reloaded_metrics['precision_macro'] > c45_metrics['precision_macro'] + margin - 1e-9
        )

    def _score_tree_dominance(self, tree, mlp_model, X_eval, y_eval, competitor_trees):
        """Score composto: fidelidade vs Original e Precision Macro vs Original/C4.5."""
        metrics = self._measure_tree_performance(tree, mlp_model, X_eval, y_eval)
        orig = self._measure_tree_performance(
            competitor_trees.get('trepan_original'), mlp_model, X_eval, y_eval
        )
        c45 = self._measure_tree_performance(
            competitor_trees.get('c45_j48'), mlp_model, X_eval, y_eval
        )
        if self._tree_dominates(metrics, orig, c45, margin=0.0):
            return 1000.0 + metrics['fidelity'] * 3.0 + metrics['precision_macro']

        fid_gap_orig = metrics['fidelity'] - orig['fidelity']
        prec_gap = metrics['precision_macro'] - max(
            orig['precision_macro'], c45['precision_macro']
        )
        penalty = 0.0
        if fid_gap_orig < 0:
            penalty += abs(fid_gap_orig) * 8.0
        if prec_gap < 0:
            penalty += abs(prec_gap) * 2.0
        return (
            fid_gap_orig * 4.0 + prec_gap
            - penalty + metrics['fidelity'] * 0.10
        )

    def _meets_precision_floor(self, metrics, orig_m, c45_m, margin=None):
        """Precision Macro mínima exigida para candidatos orientados à fidelidade."""
        if margin is None:
            margin = self.DOMINANCE_MARGIN
        floor = max(orig_m['precision_macro'], c45_m['precision_macro']) + margin - 1e-9
        return metrics['precision_macro'] >= floor

    def _select_fidelity_constrained_best(self, candidates, mlp_model, X_eval, y_eval,
                                          competitor_trees):
        """Escolhe árvore com maior fidelidade que ainda supera a precisão dos concorrentes."""
        orig_m = self._measure_tree_performance(
            competitor_trees.get('trepan_original'), mlp_model, X_eval, y_eval
        )
        c45_m = self._measure_tree_performance(
            competitor_trees.get('c45_j48'), mlp_model, X_eval, y_eval
        )
        dominating = []
        precision_safe = []
        for tree in candidates:
            if tree is None:
                continue
            m = self._measure_tree_performance(tree, mlp_model, X_eval, y_eval)
            if self._tree_dominates(m, orig_m, c45_m):
                dominating.append((tree, m))
            elif self._meets_precision_floor(m, orig_m, c45_m):
                precision_safe.append((tree, m))

        if dominating:
            return max(dominating, key=lambda x: x[1]['fidelity'] + x[1]['precision_macro'] * 0.05)[0]
        if precision_safe:
            return max(precision_safe, key=lambda x: x[1]['fidelity'])[0]
        return None

    def _build_mlp_distillation_candidates(self, *args, **kwargs):
        raise RuntimeError(
            'Caminho legado de destilação por árvores auxiliares removido. '
            'Use o núcleo TREPAN histórico de produção.'
        )

    def _build_original_style_candidates(self, mlp_model, X_real, y_real, X_syn=None, y_syn=None):
        """Candidatos com pipeline Trepan-Original (piso de fidelidade comprovado)."""
        candidates = []
        y_mlp_real = self._mlp_predict(mlp_model, X_real)
        X_base, base_names, _ = self._base_matrix_and_names(X_real)
        sampling_mlp = self._oracle_for_base_sampling()
        for sample_size in (2000, 5000):
            try:
                X_int = self._original_extractor._generate_intelligent_synthetic_data(
                    sampling_mlp, X_base, sample_size
                )
                y_int = self._mlp_predict(mlp_model, X_int, base_names)
                candidates.append(
                    self._original_extractor._train_trepan_tree(
                        X_int, y_int, X_real, y_mlp_real
                    )
                )
            except Exception:
                pass
        if X_syn is not None and y_syn is not None:
            try:
                candidates.append(
                    self._original_extractor._train_trepan_tree(
                        X_syn, y_syn, X_real, y_mlp_real
                    )
                )
            except Exception:
                pass
        return candidates

    def _ontology_fidelity_refinement(self, mlp_model, X_syn, y_syn, sample_weights,
                                      X_real, y_real, X_eval, y_eval,
                                      feature_names, competitor_trees, seed_tree=None):
        """Refino principal: destilação MLP + seleção por fidelidade com piso de precisão."""
        candidates = self._build_mlp_distillation_candidates(
            mlp_model, X_syn, y_syn, sample_weights, feature_names, X_real
        )
        if seed_tree is not None:
            candidates.insert(0, seed_tree)
        if self.explainer_tree is not None:
            candidates.insert(0, self.explainer_tree)

        best = self._select_fidelity_constrained_best(
            candidates, mlp_model, X_eval, y_eval, competitor_trees
        )
        return best if best is not None else self.explainer_tree

    def _build_dominance_hyperparameter_grid(self, n_samples):
        """Grid expandido para garantir dominância sobre Original e C4.5."""
        leaves = sorted({1, 2, 4, max(4, n_samples // 200)})
        splits = sorted({max(4, n_samples // 150), max(8, n_samples // 100), max(12, n_samples // 70)})
        grid = []
        for max_depth in self.DOMINANCE_EXPANDED_DEPTHS:
            for min_samples_leaf in leaves:
                for min_samples_split in splits:
                    if min_samples_split > min_samples_leaf:
                        for criterion in self.CRITERION_SEARCH_ORDER:
                            grid.append((max_depth, min_samples_split, min_samples_leaf, criterion))
        return grid[:48]

    def _fit_tree_dominance_grid(self, *args, **kwargs):
        raise RuntimeError(
            'Pesquisa legada de dominância por árvores auxiliares foi removida da produção.'
        )

    def _exhaustive_dominance_refinement(self, *args, **kwargs):
        raise RuntimeError(
            'Refino legado por árvores auxiliares foi removido da produção.'
        )

    def _fidelity_boost_refinement(self, *args, **kwargs):
        raise RuntimeError(
            'Refino legado de fidelidade por árvores auxiliares foi removido da produção.'
        )

    def _beats_competitors_on_precision(self, metrics, orig_m, c45_m, margin=None):
        """True se Precision Macro alcança/supera Original e C4.5."""
        if margin is None:
            margin = self.PRECISION_MARGIN
        target = max(orig_m['precision_macro'], c45_m['precision_macro']) + margin - 1e-9
        return metrics['precision_macro'] >= target

    def _precision_boost_refinement(self, *args, **kwargs):
        raise RuntimeError(
            'Refino legado de precisão por árvores auxiliares foi removido da produção.'
        )

    def _dominance_composite(self, metrics, orig_m, c45_m):
        """Score único para comparar candidatos (positivo = melhor que concorrentes)."""
        return (
            # Fidelidade é comparada apenas à referência TREPAN Original quando
            # ambas foram medidas contra o mesmo oracle argument. C4.5 entra
            # somente como baseline de qualidade contra rótulos reais.
            (metrics['fidelity'] - orig_m['fidelity']) * 0.55
            + (metrics['precision_macro'] - max(orig_m['precision_macro'], c45_m['precision_macro'])) * 0.45
        )

    def _pick_best_dominance_candidate(self, candidates, mlp_model, X_eval, y_eval,
                                       competitor_trees, baseline_tree=None):
        """Escolhe a melhor árvore; nunca ignora a baseline se for superior."""
        orig_m = self._measure_tree_performance(
            competitor_trees.get('trepan_original'), mlp_model, X_eval, y_eval
        )
        c45_m = self._measure_tree_performance(
            competitor_trees.get('c45_j48'), mlp_model, X_eval, y_eval
        )

        pool = list(candidates)
        if baseline_tree is not None:
            pool.insert(0, baseline_tree)

        dominating = []
        scored = []
        for tree in pool:
            if tree is None:
                continue
            m = self._measure_tree_performance(tree, mlp_model, X_eval, y_eval)
            composite = self._dominance_composite(m, orig_m, c45_m)
            scored.append((tree, m, composite))
            if self._tree_dominates(m, orig_m, c45_m):
                dominating.append((tree, m, composite))

        if dominating:
            return max(
                dominating,
                key=lambda x: x[1]['fidelity'] * 0.65 + x[1]['precision_macro'] * 0.35,
            )[0]

        if not scored:
            return baseline_tree

        best_tree, best_m, best_composite = max(scored, key=lambda x: x[2])
        if baseline_tree is not None:
            base_m = self._measure_tree_performance(baseline_tree, mlp_model, X_eval, y_eval)
            base_composite = self._dominance_composite(base_m, orig_m, c45_m)
            if base_composite > best_composite + 1e-9:
                return baseline_tree
        return best_tree

    def ensure_ontology_dominance(self, mlp_model, X_real, y_real, feature_names,
                                class_names, trepan_original_tree, c45_tree,
                                X_eval=None, y_eval=None, mlp_model_onto=None):
        """
        Avaliação informativa (não bloqueante): mede fidelidade/precisão da árvore
        Trepan-Reloaded já extraída, sem substituí-la nem exigir dominância sobre
        Trepan-Original ou C4.5-Nativo (oráculos distintos são comparáveis apenas
        de forma exploratória em «Comparar Métricas»).
        """
        return self.report_ontology_dominance_status(
            mlp_model,
            X_real,
            y_real,
            feature_names,
            class_names,
            trepan_original_tree,
            c45_tree,
            X_eval=X_eval,
            y_eval=y_eval,
            mlp_model_onto=mlp_model_onto,
        )

    def report_ontology_dominance_status(
        self,
        mlp_model,
        X_real,
        y_real,
        feature_names,
        class_names,
        trepan_original_tree,
        c45_tree,
        X_eval=None,
        y_eval=None,
        mlp_model_onto=None,
        dominance_check_enabled=None,
    ):
        """Regista métricas de dominância sem alterar a árvore Trepan-Reloaded."""
        if not self.has_active_ontology or self.explainer_tree is None:
            return self.explainer_tree

        enabled = (
            self.DOMINANCE_CHECK_ENABLED
            if dominance_check_enabled is None
            else dominance_check_enabled
        )
        if not enabled:
            self._last_dominance_metrics = None
            self._last_dominance_achieved = None
            self._last_dominance_informational = False
            return self.explainer_tree

        mlp_original = self._oracle_for_base_sampling(mlp_model) or mlp_model
        X_check = X_eval if X_eval is not None else X_real
        y_check = y_eval if y_eval is not None else y_real
        onto_ref = mlp_model_onto if mlp_model_onto is not None else getattr(
            self, '_mlp_model_onto', None
        )
        labeling_oracle = self._oracle_for_enriched_matrix(
            mlp_original, onto_ref, np.asarray(X_check).shape[1]
        )
        tree_feature_names = (
            self._training_cache.get('feature_names')
            or getattr(self, '_matrix_feature_names', None)
            or feature_names
        )

        reloaded_m = self._measure_tree_performance(
            self.explainer_tree,
            labeling_oracle,
            X_check,
            y_check,
            matrix_feature_names=tree_feature_names,
        )

        X_base_check, _, _ = self._base_matrix_and_names(X_check, feature_names)
        orig_m = {'fidelity': 0.0, 'precision': 0.0, 'precision_macro': 0.0}
        c45_m = {'fidelity': 0.0, 'precision': 0.0, 'precision_macro': 0.0}
        if trepan_original_tree is not None and X_base_check is not None:
            orig_m = self._measure_tree_performance(
                trepan_original_tree, mlp_original, X_base_check, y_check,
            )
        if c45_tree is not None and X_base_check is not None:
            c45_m = self._measure_tree_performance(
                c45_tree, mlp_original, X_base_check, y_check,
            )

        dominates = False
        if trepan_original_tree is not None and c45_tree is not None:
            dominates = self._tree_dominates(reloaded_m, orig_m, c45_m)

        print(
            f"[INFO] Trepan-Reloaded (mantida): "
            f"fid={reloaded_m['fidelity']:.4f} prec={reloaded_m['precision_macro']:.4f}"
        )
        if trepan_original_tree is not None:
            print(
                f"[INFO] Referência Trepan-Original: "
                f"fid={orig_m['fidelity']:.4f} prec={orig_m['precision_macro']:.4f}"
            )
        if c45_tree is not None:
            print(
                f"[INFO] Referência C4.5-Nativo: "
                f"fid={c45_m['fidelity']:.4f} prec={c45_m['precision_macro']:.4f}"
            )
        print(
            f"[INFO] Dominância simultânea (informativa): "
            f"{'SIM' if dominates else 'NAO'}"
        )
        if not dominates:
            print(
                "[INFO] A árvore Trepan-Reloaded foi mantida. A dominância não foi "
                "alcançada, mas isto não invalida o modelo, porque ele pode estar a "
                "imitar um oráculo ontológico diferente."
            )

        self._last_dominance_metrics = reloaded_m
        self._last_dominance_achieved = dominates
        self._last_dominance_informational = True
        return self.explainer_tree

    def _precision_ceiling_refinement(self, *args, **kwargs):
        raise RuntimeError(
            'Refino legado de precisão por árvores auxiliares foi removido da produção.'
        )

    def _build_semantic_hyperparameter_grid(self, n_samples):
        log_n = int(np.log2(max(n_samples, 2)))
        depths = sorted({min(5, log_n), min(6, log_n + 1), min(7, log_n + 2), min(8, log_n + 3)})
        leaves = sorted({2, 4, max(4, n_samples // 200), max(6, n_samples // 150)})
        splits = sorted({max(6, n_samples // 120), max(8, n_samples // 90), max(12, n_samples // 70)})
        grid = []
        for max_depth in depths:
            for min_samples_leaf in leaves:
                for min_samples_split in splits:
                    if min_samples_split > min_samples_leaf * 2:
                        grid.append((max_depth, min_samples_split, min_samples_leaf))
        return grid[: self.SEMANTIC_TUNING_MAX_CONFIGS]

    def _score_tree_candidate(self, tree, mlp_model, X_real, X_syn, y_syn, feature_names, class_names):
        validation = self._validate_semantic_coherence(
            tree, feature_names, class_names, verbose=False
        )
        n_issues = len(validation.get('issues', []))
        if n_issues > 3:
            return -1.0

        names = self._tree_schema_feature_names(feature_names)
        X_real_a = self._align_real_for_tree(X_real, names)
        X_syn_a = self._align_real_for_tree(X_syn, names)
        y_mlp_real = self._mlp_predict(mlp_model, X_real_a, names)
        y_tree_real = tree.predict(self._prepare_tree_matrix(tree, X_real_a, names))
        fid_real = accuracy_score(y_mlp_real, y_tree_real)

        subsample = min(2500, len(X_syn_a))
        if subsample > 0:
            idx = np.random.default_rng(42).choice(len(X_syn_a), size=subsample, replace=False)
            y_mlp_syn = self._mlp_predict(mlp_model, X_syn_a[idx], names)
            y_tree_syn = tree.predict(
                self._prepare_tree_matrix(tree, X_syn_a[idx], names)
            )
            fid_syn = accuracy_score(y_mlp_syn, y_tree_syn)
        else:
            fid_syn = fid_real

        penalty = 0.10 * n_issues
        return 0.70 * fid_real + 0.30 * fid_syn - penalty

    def _fit_tree_with_semantic_tuning(self, *args, **kwargs):
        raise RuntimeError(
            'Tuning legado baseado noutra família de árvore foi removido. '
            'A semântica é aplicada directamente no TrepanReloadedClassifier histórico.'
        )

    def _extract_faithful_reloaded_tree(
        self,
        mlp_model,
        X_syn,
        y_syn,
        X_real,
        feature_names,
        class_names,
        X_eval=None,
        sample_weights=None,
    ):
        """Treina árvore Reloaded com retry automático se fidelidade < alvo."""
        if sample_weights is None:
            sample_weights = self._build_ontology_sample_weights(X_syn, feature_names)

        limits = getattr(self, '_training_limits', {}) or {}
        fidelity_target = limits.get('fidelity_target', self.FIDELITY_TARGET)
        fidelity_early_stop = limits.get(
            'fidelity_early_stop', self.FIDELITY_EARLY_STOP
        )
        max_time = limits.get('max_time_seconds')
        configs = self.RELOADED_FIDELITY_CONFIGS

        # Nunca escolher configuracao no teste externo. A validacao e retirada
        # de X_real (treino externo) e estratificada pelas previsoes do oraculo.
        X_real_arr = np.asarray(X_real, dtype=float)
        oracle_labels = self._mlp_predict(mlp_model, X_real_arr, feature_names)
        if len(X_real_arr) >= 10:
            try:
                X_fit_real, eval_X, _, _ = train_test_split(
                    X_real_arr,
                    oracle_labels,
                    test_size=0.2,
                    random_state=42,
                    stratify=oracle_labels,
                )
            except ValueError:
                X_fit_real, eval_X = train_test_split(
                    X_real_arr, test_size=0.2, random_state=42
                )
        else:
            X_fit_real = eval_X = X_real_arr
        best_tree = None
        best_fidelity = -1.0
        chosen_cfg = configs[0]
        t0 = time.perf_counter()

        for cfg in configs:
            if max_time is not None and (time.perf_counter() - t0) >= max_time:
                print("[WARN] Trepan Reloaded: timeout; mantendo melhor árvore.")
                break

            candidate = self._fit_tree_with_semantic_tuning(
                mlp_model, X_syn, y_syn, sample_weights,
                X_fit_real, None, feature_names, class_names,
                tree_config=cfg,
            )
            candidate = self._retain_multiclass_tree(
                candidate, X_syn, y_syn, X_fit_real, None,
                mlp_model, feature_names, sample_weights=sample_weights,
            )
            fidelity = self._score_tree_fidelity(
                candidate, mlp_model, eval_X, feature_names
            )
            print(
                f"[INFO] Trepan-Reloaded fidelidade (config={cfg.get('depth_grid')}): "
                f"{fidelity:.4f}"
            )
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_tree = candidate
                chosen_cfg = cfg
            if fidelity >= fidelity_early_stop:
                print(
                    f"[INFO] Trepan Reloaded: fidelidade {fidelity:.3f} — paragem antecipada."
                )
                break
            if fidelity >= fidelity_target:
                break

        if best_tree is None:
            best_tree = self._train_ontology_constrained_tree(
                X_syn, y_syn, X_fit_real, None,
                feature_names, class_names, mlp_model=mlp_model,
            )
            best_fidelity = self._score_tree_fidelity(
                best_tree, mlp_model, eval_X, feature_names
            )

        return best_tree, float(best_fidelity), chosen_cfg

    def _log_trepan_reloaded_audit(
        self,
        oracle,
        X_train_shape,
        X_test_shape,
        n_features,
        trepan_accuracy,
        trepan_fidelity,
        config=None,
    ):
        audit = {
            'oracle': oracle,
            'trepan_accuracy': trepan_accuracy,
            'trepan_fidelity': trepan_fidelity,
            'n_features': n_features,
            'X_train_shape': X_train_shape,
            'X_test_shape': X_test_shape,
            'training_target': f'{oracle} predictions',
            'config': config,
        }
        if self.explainer_tree is not None:
            audit['tree_n_nodes'] = int(
                getattr(getattr(self.explainer_tree, 'tree_', None), 'node_count', 0)
                or getattr(self.explainer_tree, 'node_count_', 0) or 0
            )
            audit['tree_n_leaves'] = int(self.explainer_tree.get_n_leaves())
            audit['tree_is_trivial'] = audit['tree_n_leaves'] <= 1
        oracle_diag = getattr(self, '_last_oracle_diagnostic', None) or {}
        audit['oracle_unique_predictions'] = oracle_diag.get('unique_pred_test')
        audit['configuration_selection_scope'] = 'internal_training_validation'
        audit['final_test_used_for_selection'] = False
        audit['onto_feature_bias'] = self._onto_bias_calibration_audit or {
            'selection_scope': self._onto_bias_source,
            'test_used': False,
            'selected_weight': float(self.onto_feature_bias_weight),
            'formal_weighted_information_gain': False,
            'method': 'heuristic_sample_and_feature_priority_weight',
            'note': 'Heurística documentada; não é ganho de informação ponderado formal.',
        }
        audit['distillation'] = getattr(self, '_last_distillation_info', None)
        audit['hybrid_distillation'] = getattr(
            self, '_last_hybrid_distillation_info', None
        )
        audit['active_queries'] = getattr(self, '_last_active_query_info', None)
        audit['plausible_counterfactuals'] = getattr(
            self, '_last_plausible_cf_info', None
        )
        audit['canonical_trepan'] = getattr(self, '_last_canonical_info', None)
        audit['global_soft_tree'] = getattr(
            self, '_last_soft_global_info', None
        )
        audit['multiobjective_selection'] = getattr(
            self, '_last_multiobjective_info', None
        )
        if audit.get('tree_is_trivial') and oracle_diag.get('unique_pred_test', 0) > 1:
            audit['trivial_reason'] = 'extraction_bug'
        elif audit.get('tree_is_trivial'):
            audit['trivial_reason'] = 'constant_oracle'
        else:
            audit['trivial_reason'] = 'ok'
        self.last_audit = audit

        print("\n[TREPAN RELOADED AUDIT]")
        print(f"oracle: {oracle}")
        print(f"Trepan Reloaded accuracy: {trepan_accuracy:.4f}")
        print(f"Trepan Reloaded fidelity: {trepan_fidelity:.4f}")
        print(f"Trepan training target: {audit['training_target']}")
        print(f"X_train shape: {X_train_shape}")
        print(f"X_test shape: {X_test_shape}")
        print(f"n_features: {n_features}")

    def _train_ontology_constrained_tree(self, *args, **kwargs):
        raise RuntimeError(
            'Treino ontológico legado removido. Use _extract_historical_reloaded_core().'
        )
    
    def validate_semantic_coherence(self, mlp_model, X_encoded, y_encoded, feature_names, class_names):
        if not self.has_active_ontology:
            return "ℹ️ Validação semântica desativada (modo compatibilidade Trepan-Original)."
        if self.explainer_tree is None:
            return "❌ Nenhuma árvore gerada. Execute a extração primeiro."
        result = self._validate_semantic_coherence(self.explainer_tree, feature_names, class_names)
        issues = len(result.get('issues', []))
        warnings = len(result.get('warnings', []))
        if issues == 0 and warnings == 0:
            return "✅ Coerência semântica validada sem problemas detectados."
        return (
            f"⚠️ Coerência semântica: {issues} problema(s), {warnings} aviso(s). "
            "Consulte o console para detalhes."
        )

    def _validate_semantic_coherence(self, tree, feature_names, class_names, verbose=True):
        if not self.has_active_ontology:
            return {'issues': [], 'warnings': [], 'total_nodes': 0, 'validated_nodes': 0}

        validation_issues = []
        validation_warnings = []
        reasoner = dict(self.reasoner_report or {})
        if not reasoner.get('executed'):
            validation_issues.append(
                '[reasoner] Validação OWL DL não executada; coerência lógica não demonstrada.'
            )
        elif reasoner.get('consistent') is False:
            validation_issues.append('[reasoner] Ontologia logicamente inconsistente.')
        for class_name in reasoner.get('unsatisfiable_classes', []) or []:
            validation_issues.append(f'[reasoner] Classe insatisfazível: {class_name}.')

        split_literals = []
        if hasattr(tree, 'iter_splits'):
            # TREPAN histórico: um nó pode conter vários literais num teste m-of-n.
            for node, test in tree.iter_splits():
                for literal in test.literals:
                    split_literals.append((int(literal.feature), float(literal.threshold), node.node_id))
            total_nodes = int(getattr(tree, 'node_count_', 0))
            validated_nodes = len(list(tree.iter_splits()))
        elif hasattr(tree, 'tree_'):
            structure = tree.tree_
            total_nodes = int(structure.node_count)
            validated_nodes = 0
            for i in range(structure.node_count):
                if structure.children_left[i] != -1:
                    validated_nodes += 1
                    split_literals.append((
                        int(structure.feature[i]), float(structure.threshold[i]), int(i)
                    ))
        else:
            raise TypeError('Modelo de árvore não suportado pela validação semântica.')

        for feature_idx, threshold, node_id in split_literals:
            feature_name = (
                feature_names[feature_idx]
                if 0 <= feature_idx < len(feature_names)
                else f'feature_{feature_idx}'
            )
            feature_info = self.feature_semantics.get(feature_idx, {})
            concept_type = feature_info.get('concept_type', 'general')
            ontology_concept = feature_info.get('ontology_concept')
            bounds_feat = self._real_data_bounds.get(feature_idx, {})
            col_min_empirical = bounds_feat.get('min')

            if concept_type in {'temporal', 'dimensional'} and col_min_empirical is not None and col_min_empirical >= 0 and threshold < 0:
                validation_warnings.append(
                    f'[warn] Threshold negativo em {feature_name} ({threshold:.3f}), nó {node_id}.'
                )

            if ontology_concept:
                for prop_name, prop_constraints in self.property_constraints.items():
                    if prop_name.lower() not in feature_name.lower():
                        continue
                    if prop_constraints.get('range_type') != 'datatype':
                        continue
                    datatype_info = self.datatype_ranges.get(prop_name) or {}
                    dtype_min = datatype_info.get('min')
                    dtype_max = datatype_info.get('max')
                    if dtype_min is not None and threshold < dtype_min:
                        validation_issues.append(
                            f'[axioma] Threshold {threshold:.3f} em {feature_name} viola min={dtype_min}'
                        )
                    if dtype_max is not None and threshold > dtype_max:
                        validation_issues.append(
                            f'[axioma] Threshold {threshold:.3f} em {feature_name} viola max={dtype_max}'
                        )

            bounds = self._real_data_bounds.get(feature_idx)
            if bounds:
                hard_min, hard_max = bounds['min'], bounds['max']
                if threshold < hard_min or threshold > hard_max:
                    validation_issues.append(
                        f'[axioma] Threshold {threshold:.3f} em {feature_name} fora do domínio '
                        f'empírico [{hard_min:.3f}, {hard_max:.3f}]'
                    )
                elif threshold < bounds['p05'] or threshold > bounds['p95']:
                    validation_warnings.append(
                        f'[warn] Threshold {threshold:.3f} em {feature_name} fora do intervalo '
                        f'plausível [{bounds["p05"]:.3f}, {bounds["p95"]:.3f}]'
                    )

        if verbose:
            if validation_issues:
                print(f'\n[WARN] Problemas de coerência semântica ({len(validation_issues)}):')
                for issue in validation_issues[:10]:
                    print(f'  {issue}')
            if validation_warnings:
                print(f'\n[WARN] Avisos de coerência semântica ({len(validation_warnings)}):')
                for warning in validation_warnings[:10]:
                    print(f'  {warning}')

        return {
            'issues': validation_issues,
            'warnings': validation_warnings,
            'reasoner': reasoner,
            'total_nodes': total_nodes,
            'validated_nodes': validated_nodes,
            'tree_family': 'trepan_historical' if hasattr(tree, 'iter_splits') else 'sklearn_tree',
        }

    def _calculate_ontology_aware_fidelity(self, tree, mlp_model, X_real, y_real, 
                                          X_synthetic, y_synthetic):
        # Fidelidade básica
        fidelity_real = self._calculate_fidelity(tree, mlp_model, X_real, y_real)
        fidelity_synthetic = self._calculate_fidelity(tree, mlp_model, X_synthetic, y_synthetic)
        
        # Fidelidade por grupo semântico
        semantic_fidelity = self._calculate_semantic_group_fidelity(tree, mlp_model, X_real, y_real)
        
        return {
            'overall_fidelity': fidelity_real,
            'overall_real': fidelity_real,
            'overall_synthetic': fidelity_synthetic,
            'semantic_groups': semantic_fidelity,
            'ontology_active': self.has_active_ontology,
            'n_real_samples': len(X_real),
            'n_synthetic_samples': len(X_synthetic),
        }
    
    def _calculate_semantic_group_fidelity(self, tree, mlp_model, X_real, y_real):
        """Fidelidade do próprio TREPAN em regiões activadas por grupo semântico.

        V9.2: não treina CARTs auxiliares. Para cada grupo, selecciona as linhas
        onde as respectivas features apresentam maior activação/desvio robusto e
        mede directamente a concordância TREPAN↔oráculo nessa região.
        """
        X_real = np.asarray(X_real, dtype=float)
        names = self._tree_schema_feature_names(None)
        X_aligned = self._align_real_for_tree(X_real, names)
        X_tree = self._prepare_tree_matrix(tree, X_aligned, names)
        y_tree_pred = np.asarray(tree.predict(X_tree))
        y_mlp_pred = np.asarray(self._mlp_predict(mlp_model, X_aligned, names))

        semantic_groups = {}
        for feature_idx, feature_info in self.feature_semantics.items():
            group = feature_info.get('semantic_group', 'general')
            semantic_groups.setdefault(group, []).append(feature_idx)

        results = {}
        n_cols = X_aligned.shape[1]
        for group_name, feature_indices in semantic_groups.items():
            valid_idx = self._valid_column_indices(feature_indices, n_cols)
            if not valid_idx:
                continue
            group_matrix = np.asarray(X_aligned[:, valid_idx], dtype=float)
            centre = np.nanmedian(group_matrix, axis=0)
            scale = np.nanstd(group_matrix, axis=0)
            scale = np.where(scale > 1e-12, scale, 1.0)
            activation = np.nanmean(np.abs((group_matrix - centre) / scale), axis=1)
            finite = np.isfinite(activation)
            if not np.any(finite):
                mask = np.ones(len(X_aligned), dtype=bool)
            else:
                threshold = float(np.nanmedian(activation[finite]))
                mask = finite & (activation >= threshold)
                if int(mask.sum()) < min(5, len(X_aligned)):
                    mask = finite
            if np.any(mask):
                results[group_name] = float(accuracy_score(
                    y_mlp_pred[mask], y_tree_pred[mask]
                ))
        return results

    def _calculate_fidelity(self, tree, mlp_model, X_data, y_data, feature_names=None):
        names = self._tree_schema_feature_names(feature_names)
        X_aligned = self._align_real_for_tree(X_data, names)
        X_tree = self._prepare_tree_matrix(tree, X_aligned, names)
        y_tree_pred = tree.predict(X_tree)
        y_mlp_pred = self._mlp_predict(mlp_model, X_aligned, names)
        return accuracy_score(y_mlp_pred, y_tree_pred)
    
    def _generate_ontology_aware_report(self, fidelity_metrics, n_real_samples, 
                                       feature_names, class_names):
        report = f"\n{'='*70}\n"
        report += f"🧠 RELATÓRIO DE INTERPRETABILIDADE TREPAN-RELOADED\n"
        report += f"{'='*70}\n\n"
        
        overall = fidelity_metrics.get(
            'overall_fidelity', fidelity_metrics.get('overall_real', 0.0)
        )
        report += f"🎯 Fidelidade da Árvore ao MLP (overall_fidelity):\n"
        report += f"   • Métrica principal (dados reais): {overall:.3f} ({overall*100:.1f}%)\n"
        report += f"   • Sobre dados sintéticos: {fidelity_metrics['overall_synthetic']:.3f} ({fidelity_metrics['overall_synthetic']*100:.1f}%)\n"
        report += f"   • Sobre dados reais: {fidelity_metrics['overall_real']:.3f} ({fidelity_metrics['overall_real']*100:.1f}%)\n"
        report += f"   • Modo ontologia: {'ativo (Reloaded Full)' if self.has_active_ontology else 'inativo'}\n"
        report += f"   • Amostras sintéticas destiladas: {fidelity_metrics.get('n_synthetic_samples', 'N/A')}\n"
        if self.feature_centrality:
            top_central = sorted(
                self.feature_centrality.items(), key=lambda x: x[1], reverse=True
            )[:3]
            report += f"   • Features centrais OWL: {', '.join(f'#{i}({s:.2f})' for i, s in top_central)}\n"
        if hasattr(self, '_last_dominance_metrics') and self._last_dominance_metrics:
            dm = self._last_dominance_metrics
            report += (
                f"   • Métricas Trepan-Reloaded (informativo): "
                f"fid={dm.get('fidelity', 0):.3f}, prec={dm.get('precision', 0):.3f}\n"
            )
            if hasattr(self, '_last_dominance_achieved'):
                if self._last_dominance_achieved:
                    report += "   • Dominância vs Original/C4.5: sim (informativo)\n"
                else:
                    report += (
                        "   • Dominância vs Original/C4.5: não (informativo; a árvore "
                        "foi mantida — oráculos distintos)\n"
                    )
        report += f"   • Amostras reais validadas: {n_real_samples}\n\n"

        semantic_audit = dict(
            getattr(getattr(self, 'explainer_tree', None), 'semantic_audit_summary_', {}) or {}
        )
        if semantic_audit:
            evaluated = int(semantic_audit.get('evaluated_splits', 0))
            influenced = int(semantic_audit.get('ontology_influenced_splits', 0))
            changed = int(semantic_audit.get('semantic_changed_splits', 0))
            reinforced = int(semantic_audit.get('semantic_reinforced_splits', 0))
            report += "🔎 Auditoria de Impacto Ontológico nos Splits:\n"
            report += f"   • Splits avaliados: {evaluated}\n"
            report += f"   • Splits influenciados/reforçados pela OWL: {influenced}\n"
            report += f"   • Splits cuja decisão mudou: {changed}\n"
            report += f"   • Splits com vencedor mantido mas reforçado: {reinforced}\n"
            report += f"   • Ontology Usage Rate: {semantic_audit.get('ontology_usage_rate', 0.0)*100:.1f}%\n"
            report += f"   • Semantic Decision Impact: {semantic_audit.get('semantic_decision_impact', 0.0)*100:.1f}%\n"
            report += f"   • Bónus semântico médio: {semantic_audit.get('mean_semantic_bonus', 0.0):+.6f}\n"
            report += f"   • Regiões de erro avaliadas (EFSR): {int(semantic_audit.get('error_regions_evaluated', 0))}\n"
            report += f"   • Regiões elegíveis para refinamento: {int(semantic_audit.get('error_regions_eligible', 0))}\n"
            report += f"   • Intervenções EFSR tentadas: {int(semantic_audit.get('error_focused_interventions_attempted', 0))}\n"
            report += f"   • Intervenções EFSR aceites: {int(semantic_audit.get('error_focused_interventions_accepted', 0))}\n"
            report += f"   • Intervenções EFSR rejeitadas: {int(semantic_audit.get('error_focused_interventions_rejected', 0))}\n"
            report += f"   • Queries focadas em erro seleccionadas: {int(semantic_audit.get('error_focused_query_selected', 0))}\n"
            report += (
                f"   • Projecção semântica de membership queries: "
                f"{'sim' if semantic_audit.get('query_projection_active') else 'não'}\n"
            )
            split_rows = list(getattr(getattr(self, 'explainer_tree', None), 'semantic_split_audit_', []) or [])
            if split_rows:
                report += "   • Auditoria por nó:\n"
                for row in split_rows:
                    report += (
                        f"     - nó {row.get('node_id')}: IG={float(row.get('information_gain', 0.0)):.6f}; "
                        f"score semântico={float(row.get('selection_score', 0.0)):.6f}; "
                        f"bónus={float(row.get('semantic_bonus', 0.0)):+.6f}; "
                        f"mudou={'sim' if row.get('decision_changed') else 'não'}; "
                        f"EFSR={'aceite' if row.get('error_focused_intervention_accepted') else ('rejeitado' if row.get('error_focused_intervention_attempted') else 'não necessário')}\n"
                    )
                    if row.get('error_focused_intervention_attempted'):
                        report += (
                            f"       fidelidade local data-only={row.get('data_only_local_fidelity')}; "
                            f"semântica={row.get('semantic_local_fidelity')}; "
                            f"Δ={float(row.get('local_fidelity_gain', 0.0)):+.4f}\n"
                        )
                        if row.get('real_training_gate_evaluated'):
                            real_gain = row.get('real_fidelity_gain')
                            report += (
                                f"       gate real-train: data-only={row.get('data_only_real_fidelity')}; "
                                f"semântica={row.get('semantic_real_fidelity')}; "
                                f"Δ={float(real_gain or 0.0):+.4f}\n"
                            )
                    if row.get('decision_changed'):
                        report += f"       data-only: {row.get('data_only_test')}\n"
                        report += f"       Reloaded: {row.get('semantic_test') or row.get('test')}\n"
            report += "\n"

        # Fidelidade por grupo semântico
        if fidelity_metrics['semantic_groups']:
            report += f"📊 Fidelidade por Grupo Semântico:\n"
            for group, fidelity in fidelity_metrics['semantic_groups'].items():
                report += f"   • {group}: {fidelity:.3f} ({fidelity*100:.1f}%)\n"
            report += f"\n"
        
        # Conhecimento de domínio extraído
        report += f"🧠 Conhecimento de Domínio Mapeado / Disponível:\n"
        report += f"   • Features analisadas: {len(self.feature_semantics)}\n"
        semantic_graph_summary = dict(getattr(self, 'semantic_graph_summary', {}) or {})
        owl_class_count = int(semantic_graph_summary.get('class_count', 0) or 0)
        report += f"   • Classes OWL disponíveis: {owl_class_count}\n"
        report += f"   • Classes-alvo mapeadas (informativo): {len(self.class_semantics)}\n"
        rel_count = max(
            len(self.domain_knowledge.get('relationships', [])),
            int(semantic_graph_summary.get('edges', 0) or 0),
        )
        report += f"   • Relacionamentos semânticos: {rel_count}\n"
        if semantic_graph_summary:
            report += f"   • Grafo OWL: {int(semantic_graph_summary.get('nodes', 0) or 0)} nós; {int(semantic_graph_summary.get('edges', 0) or 0)} arestas\n"
            report += f"   • Features com grupos semânticos: {int(semantic_graph_summary.get('features_with_groups', 0) or 0)}\n"
        
        # Features mapeadas vs não mapeadas
        mapped_features = sum(
            1 for f in self.feature_semantics.values() if f.get('ontology_mapped')
        )
        general_fallback = sum(
            1 for f in self.feature_semantics.values() if f.get('general_fallback')
        )
        unmapped_count = len(self.unmapped_features)
        report += f"   • Features mapeadas para ontologia: {mapped_features}/{len(self.feature_semantics)}\n"
        if general_fallback:
            report += f"   • Features com categoria geral (General_Numeric/Categorical): {general_fallback}\n"
        report += f"   • Features não mapeadas: {unmapped_count}\n"
        if unmapped_count > 0 and unmapped_count <= 5:
            for unmapped in self.unmapped_features:
                if unmapped.get('suggested_match'):
                    report += f"     - '{unmapped['feature']}' (sugestão: {unmapped['suggested_match']}, confiança: {unmapped['confidence']:.2f})\n"
                else:
                    report += f"     - '{unmapped['feature']}' (sem match encontrado)\n"
        if hasattr(self, '_last_gridsearch_depth'):
            report += f"   • GridSearch fidelidade: profundidade={self._last_gridsearch_depth}\n"
        if hasattr(self, '_last_gridsearch_fidelity'):
            report += f"   • Fidelidade pós-GridSearch: {self._last_gridsearch_fidelity:.3f}\n"
        
        # Qualidade dos mapeamentos
        if self.mapping_scores:
            avg_score = np.mean(list(self.mapping_scores.values()))
            report += f"   • Score médio de mapeamento: {avg_score:.2f}\n"
        
        # Hierarquias e restrições
        if self.ontology_hierarchy:
            report += f"   • Hierarquias ontológicas extraídas: {len(self.ontology_hierarchy)}\n"
        if self.property_constraints:
            report += f"   • Restrições de propriedades: {len(self.property_constraints)}\n"
        if self.datatype_ranges:
            report += f"   • Datatypes identificados: {len(self.datatype_ranges)}\n"
        eng = self.semantic_feature_engineering_stats or {}
        if eng:
            report += (
                f"   • Engenharia semântica: +{len(eng.get('inferred_columns', []))} features "
                f"(hierárquicas={eng.get('hierarchical_features', 0)}, "
                f"relacionais={eng.get('relational_features', 0)}, "
                f"restrição={eng.get('constraint_features', 0)})\n"
            )
        
        report += f"\n"
        
        # Interpretação da fidelidade e da contribuição semântica são independentes.
        avg_fidelity = (fidelity_metrics['overall_synthetic'] + fidelity_metrics['overall_real']) / 2
        if avg_fidelity > 0.9:
            report += "🌟 EXCELENTE fidelidade ao MLP.\n"
        elif avg_fidelity > 0.8:
            report += "✅ MUITO BOA fidelidade ao MLP.\n"
        elif avg_fidelity > 0.7:
            report += "👍 BOA fidelidade ao MLP.\n"
        else:
            report += "⚠️ Fidelidade MODERADA ao MLP.\n"

        usage_rate = float(semantic_audit.get('ontology_usage_rate', 0.0)) if semantic_audit else 0.0
        relationships_n = max(len(self.domain_knowledge.get('relationships', [])), int((getattr(self, 'semantic_graph_summary', {}) or {}).get('edges', 0) or 0))
        if usage_rate > 0:
            report += (
                f"✓ Contribuição semântica observada em {usage_rate*100:.1f}% dos splits avaliados.\n"
            )
        elif self.has_active_ontology:
            report += (
                "⚠️ Ontologia válida/mapeada, mas esta execução não demonstrou alteração "
                "ou reforço mensurável dos splits pela semântica.\n"
            )

        report += "\n💡 Estado semântico do Trepan-Reloaded:\n"
        report += "   • Mapeamento ontológico disponível para auditoria das features\n"
        report += (
            "   • Influência nos splits: demonstrada\n" if usage_rate > 0
            else "   • Influência nos splits: não demonstrada nesta execução\n"
        )
        report += (
            f"   • Relacionamentos ontológicos extraídos: {relationships_n}\n"
        )
        report += "   • Coerência semântica é validada separadamente da fidelidade preditiva\n\n"
        
        report += f"🌳 Próximos passos:\n"
        report += f"   • Use '🌳 Visualizar Árvore' para explorar as regras\n"
        report += f"   • Use '💡 Explicações Semânticas' para análises detalhadas\n"
        report += f"   • Use '📊 Comparar Modelos' para análise comparativa\n"
        
        return report
    
    def generate_semantic_explanations(self, feature_names, class_names):
        if not self.has_active_ontology:
            return "ℹ️ Explicações semânticas requerem ontologia OWL carregada na GUI."
        if self.explainer_tree is None:
            return "❌ Nenhuma árvore foi gerada ainda. Execute extract_tree_with_ontology primeiro."
        
        rules = []
        export_class_names = self._class_names_for_tree(self.explainer_tree, class_names)
        if hasattr(self.explainer_tree, 'export_rules'):
            for raw in self.explainer_tree.export_rules():
                pred = raw.get('prediction')
                try:
                    pred_idx = int(pred)
                except (TypeError, ValueError):
                    pred_idx = -1
                class_name = (
                    export_class_names[pred_idx]
                    if 0 <= pred_idx < len(export_class_names)
                    else str(pred)
                )
                conditions = [{
                    'feature': condition,
                    'operator': '',
                    'threshold': 0.0,
                    'feature_context': {},
                    'semantic_meaning': condition,
                } for condition in raw.get('conditions', [])]
                probabilities = list(raw.get('probabilities') or [0.0])
                rules.append({
                    'conditions': conditions,
                    'class': class_name,
                    'confidence': float(max(probabilities)),
                    'samples': int(raw.get('support', 0)),
                    'class_context': self.class_semantics.get(pred_idx, {}),
                    'semantic_interpretation': self._generate_semantic_interpretation(
                        conditions, class_name
                    ),
                })
        else:
            self._extract_semantic_rules_recursive(
                self.explainer_tree.tree_, 0, [], rules, feature_names, export_class_names
            )

        return self._format_semantic_rules(rules, export_class_names)
    
    def _extract_semantic_rules_recursive(self, tree, node_id, conditions, rules, 
                                        feature_names, class_names):
        left_id = int(tree.children_left[node_id])
        right_id = int(tree.children_right[node_id])
        if left_id < 0 or left_id == right_id:
            # Nó folha - adiciona regra completa com contexto semântico
            class_id = np.argmax(tree.value[node_id][0])
            class_name = class_names[class_id]
            confidence = tree.value[node_id][0][class_id] / np.sum(tree.value[node_id][0])
            samples = tree.n_node_samples[node_id]
            
            # Adiciona contexto semântico da classe
            class_info = self.class_semantics.get(class_id, {})
            
            rules.append({
                'conditions': conditions.copy(),
                'class': class_name,
                'confidence': confidence,
                'samples': samples,
                'class_context': class_info,
                'semantic_interpretation': self._generate_semantic_interpretation(conditions, class_name)
            })
            return
        
        # Nó interno - continua recursão com contexto semântico
        feature_name = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        
        # Adiciona contexto semântico da feature
        feature_info = self.feature_semantics.get(tree.feature[node_id], {})
        
        # Condição para filho esquerdo (<=)
        left_conditions = conditions + [{
            'feature': feature_name,
            'operator': '≤',
            'threshold': threshold,
            'feature_context': feature_info,
            'semantic_meaning': self._get_semantic_meaning(feature_name, '≤', threshold)
        }]
        self._extract_semantic_rules_recursive(
            tree, tree.children_left[node_id], left_conditions, rules, feature_names, class_names
        )
        
        # Condição para filho direito (>)
        right_conditions = conditions + [{
            'feature': feature_name,
            'operator': '>',
            'threshold': threshold,
            'feature_context': feature_info,
            'semantic_meaning': self._get_semantic_meaning(feature_name, '>', threshold)
        }]
        self._extract_semantic_rules_recursive(
            tree, tree.children_right[node_id], right_conditions, rules, feature_names, class_names
        )
    
    def _get_semantic_meaning(self, feature_name, operator, threshold):
        feature_info = None
        for info in self.feature_semantics.values():
            if info.get('name') == feature_name:
                feature_info = info
                break
        
        if not feature_info:
            return f"{feature_name} {operator} {threshold:.3f}"
        
        concept_type = feature_info.get('concept_type', 'general')
        
        # Gera significado baseado no tipo de conceito
        if concept_type == 'temporal':
            if operator == '≤':
                return f"até {threshold:.1f} unidades de tempo"
            else:
                return f"após {threshold:.1f} unidades de tempo"
        elif concept_type == 'dimensional':
            if operator == '≤':
                return f"até {threshold:.1f} unidades de medida"
            else:
                return f"maior que {threshold:.1f} unidades de medida"
        elif concept_type == 'quantitative':
            if operator == '≤':
                return f"até {threshold:.1f}"
            else:
                return f"maior que {threshold:.1f}"
        elif concept_type == 'categorical':
            return f"categoria {threshold:.0f}"
        else:
            return f"{feature_name} {operator} {threshold:.3f}"
    
    def _generate_semantic_interpretation(self, conditions, class_name):
        if not conditions:
            return f"Classificação padrão como '{class_name}'"
        
        # Agrupa condições por tipo semântico
        temporal_conditions = []
        dimensional_conditions = []
        quantitative_conditions = []
        categorical_conditions = []
        
        for condition in conditions:
            feature_context = condition.get('feature_context', {})
            concept_type = feature_context.get('concept_type', 'general')
            
            if concept_type == 'temporal':
                temporal_conditions.append(condition)
            elif concept_type == 'dimensional':
                dimensional_conditions.append(condition)
            elif concept_type == 'quantitative':
                quantitative_conditions.append(condition)
            elif concept_type == 'categorical':
                categorical_conditions.append(condition)
        
        # Gera interpretação semântica
        interpretation_parts = []
        
        if temporal_conditions:
            interpretation_parts.append("considerando aspectos temporais")
        if dimensional_conditions:
            interpretation_parts.append("considerando dimensões físicas")
        if quantitative_conditions:
            interpretation_parts.append("considerando valores quantitativos")
        if categorical_conditions:
            interpretation_parts.append("considerando categorias")
        
        if interpretation_parts:
            return f"Classificação como '{class_name}' " + " e ".join(interpretation_parts)
        else:
            return f"Classificação como '{class_name}' baseada em múltiplos critérios"
    
    def _format_semantic_rules(self, rules, class_names):
        if not rules:
            return "❌ Nenhuma regra foi extraída da árvore."
        
        formatted_rules = f"\n{'='*70}\n"
        formatted_rules += f"🧠 EXPLICAÇÕES SEMÂNTICAS TREPAN-RELOADED\n"
        formatted_rules += f"{'='*70}\n\n"
        
        formatted_rules += f"📋 O modelo toma decisões baseado em conhecimento de domínio:\n\n"
        
        for i, rule in enumerate(rules, 1):
            formatted_rules += f"🔹 Regra Semântica {i}:\n"
            
            if rule['conditions']:
                # Formata condições com contexto semântico
                condition_texts = []
                for condition in rule['conditions']:
                    semantic_meaning = condition.get('semantic_meaning', 
                                                   f"{condition['feature']} {condition['operator']} {condition['threshold']:.3f}")
                    condition_texts.append(semantic_meaning)
                
                formatted_rules += f"   Se {' E '.join(condition_texts)}\n"
            else:
                formatted_rules += f"   Se nenhuma condição específica\n"
            
            formatted_rules += f"   Então: Classe '{rule['class']}'\n"
            formatted_rules += f"   Confiança: {rule['confidence']:.1%}\n"
            formatted_rules += f"   Amostras: {rule['samples']}\n"
            formatted_rules += f"   Interpretação: {rule['semantic_interpretation']}\n\n"
        
        # Análise de complexidade semântica
        avg_conditions = np.mean([len(rule['conditions']) for rule in rules])
        formatted_rules += f"📊 Análise de Complexidade Semântica:\n"
        formatted_rules += f"   • Total de regras: {len(rules)}\n"
        formatted_rules += f"   • Condições médias por regra: {avg_conditions:.1f}\n"
        
        # Análise de tipos de conceitos utilizados
        concept_types_used = set()
        for rule in rules:
            for condition in rule['conditions']:
                feature_context = condition.get('feature_context', {})
                concept_type = feature_context.get('concept_type', 'general')
                concept_types_used.add(concept_type)
        
        formatted_rules += f"   • Tipos de conceitos utilizados: {', '.join(concept_types_used)}\n"
        
        if avg_conditions <= 2:
            formatted_rules += f"   ✅ Regras simples e semanticamente claras\n"
        elif avg_conditions <= 4:
            formatted_rules += f"   ⚠️ Regras moderadamente complexas\n"
        else:
            formatted_rules += f"   ❌ Regras complexas, considere simplificar\n"
        
        formatted_rules += f"\n💡 Vantagens das Explicações Semânticas:\n"
        formatted_rules += f"   • Incorporam conhecimento de domínio\n"
        formatted_rules += f"   • Fornecem contexto semântico rico\n"
        formatted_rules += f"   • Consideram relacionamentos entre conceitos\n"
        formatted_rules += f"   • São mais compreensíveis para especialistas do domínio\n"
        
        return formatted_rules
    
    def export_tree_image(self, feature_names, class_names, output_file="tree_visualization.png", open_image=True):
        if self.explainer_tree is None:
            raise ValueError("Árvore ainda não foi gerada. Execute extract_tree_with_ontology primeiro.")

        # Adiciona contexto semântico aos nomes das features
        enhanced_feature_names = []
        for i, feature_name in enumerate(feature_names):
            feature_info = self.feature_semantics.get(i, {})
            concept_type = feature_info.get('concept_type', 'general')
            semantic_group = feature_info.get('semantic_group', 'general')
            
            enhanced_name = f"{feature_name}\n[{concept_type}]"
            enhanced_feature_names.append(enhanced_name)

        if isinstance(self.explainer_tree, TrepanOriginalClassifier):
            adapter = TrepanOriginalExtractor()
            adapter.adopt_tree(self.explainer_tree)
            requested = Path(output_file)
            stem = requested.with_suffix('') if requested.suffix else requested
            dot_path = str(stem) + '.dot'
            adapter.export_tree_image(
                enhanced_feature_names, class_names, output_file=dot_path, open_image=False
            )
            dot_data = Path(dot_path).read_text(encoding='utf-8')
            try:
                import graphviz                      # visualização opcional: só a exportação para imagem precisa dela
            except ImportError as exc:
                raise RuntimeError("Exportar a árvore como imagem requer o pacote Python 'graphviz' (e o binário 'dot'); "
                                   "o treino e a avaliação funcionam sem eles. pip install graphviz") from exc
            graph = graphviz.Source(dot_data)
            graph.format = 'png'
            rendered = graph.render(filename=str(stem), cleanup=True)
            file_path = os.path.abspath(rendered)
        else:
            raise TypeError(
                "A build de produção exporta apenas árvores TREPAN históricas neste extractor. "
                "Use o visualizador C4.5 dedicado para o baseline supervisionado."
            )
        self.last_exported_image_path = file_path

        if open_image:
            webbrowser.open(f"file://{file_path}")

        return f"Árvore com contexto semântico exportada como imagem para {file_path}"
    
    def save_domain_knowledge(self, file_path):
        try:
            save_data = {
                'feature_semantics': self.feature_semantics,
                'class_semantics': self.class_semantics,
                'domain_knowledge': self.domain_knowledge,
                'mapping_cache': self.mapping_cache,
                'mapping_scores': self.mapping_scores,
                'ontology_hierarchy': self.ontology_hierarchy,
                'property_constraints': self.property_constraints,
                'datatype_ranges': self.datatype_ranges,
                'unmapped_features': self.unmapped_features,
                'unmapped_classes': self.unmapped_classes
            }
            
            # Converte objetos owlready2 para strings onde necessário
            serializable_data = self._make_serializable(save_data)
            
            with open(file_path, 'wb') as f:
                pickle.dump(serializable_data, f)
            
            print(f"✅ Conhecimento de domínio salvo em {file_path}")
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar conhecimento de domínio: {e}")
            return False
    
    def load_domain_knowledge(self, file_path):
        
        try:
            with open(file_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            self.feature_semantics = loaded_data.get('feature_semantics', {})
            self.class_semantics = loaded_data.get('class_semantics', {})
            self.domain_knowledge = loaded_data.get('domain_knowledge', {})
            self.mapping_cache = loaded_data.get('mapping_cache', {})
            self.mapping_scores = loaded_data.get('mapping_scores', {})
            self.ontology_hierarchy = loaded_data.get('ontology_hierarchy', {})
            self.property_constraints = loaded_data.get('property_constraints', {})
            self.datatype_ranges = loaded_data.get('datatype_ranges', {})
            self.unmapped_features = loaded_data.get('unmapped_features', [])
            self.unmapped_classes = loaded_data.get('unmapped_classes', [])
            
            print(f"✅ Conhecimento de domínio carregado de {file_path}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar conhecimento de domínio: {e}")
            return False
    
    def _make_serializable(self, data):
        
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            # Tenta converter para string
            try:
                return str(data)
            except (TypeError, ValueError, RuntimeError):
                return None
    
    def register_quality_matches(self, feature_names, matches):
        """Promove o matching validado a uma estrutura semântica executável.

        V9.2 produção: não basta guardar ``feature -> entity``. Quando a OWL está
        disponível, constrói também :class:`OntologySemanticGraph`, propaga
        grupos/parentes para ``feature_semantics`` e publica as relações no
        ``domain_knowledge``. Tudo é derivado da TBox; não lê target nem teste.
        """
        names = list(map(str, feature_names or []))
        accepted_by_feature = {
            str(row.get('feature')): row
            for row in (matches or [])
            if row.get('accepted')
        }
        previous = dict(self.feature_semantics or {})
        self.feature_semantics = {}
        self.mapping_scores = {}
        self.unmapped_features = []
        for idx, name in enumerate(names):
            row = accepted_by_feature.get(name)
            old = dict(previous.get(idx) or {})
            if row is None:
                old.update({
                    'feature_name': name, 'feature': name,
                    'ontology_mapped': False, 'mapping_score': 0.0,
                    'matched_entity': None, 'entity_type': None,
                    'general_fallback': False,
                })
                self.feature_semantics[idx] = old
                self.unmapped_features.append({'feature': name, 'reason': 'quality_gate_unmapped'})
                continue
            score = float(row.get('score', 0.0) or 0.0)
            entity_name = row.get('entity_name') or row.get('matched_entity')
            old.update({
                'feature_name': name, 'feature': name,
                'ontology_mapped': True, 'mapping_score': score,
                'matched_entity': entity_name, 'concept': entity_name,
                'ontology_concept': entity_name, 'entity_type': row.get('entity_type'),
                'general_fallback': False, 'split_priority': max(1.0, score),
                'semantic_group': entity_name or name,
            })
            self.feature_semantics[idx] = old
            self.mapping_scores[name] = score

        self.semantic_graph = None
        self.semantic_graph_summary = {}
        if self.ontology is not None and accepted_by_feature:
            try:
                graph = OntologySemanticGraph.from_ontology(
                    self.ontology, accepted_matches=list(matches or [])
                )
                self.semantic_graph = graph
                self.semantic_graph_summary = graph.summary()
                for idx, name in enumerate(names):
                    info = self.feature_semantics.get(idx, {})
                    if not info.get('ontology_mapped'):
                        continue
                    group = graph.primary_group(name)
                    parents = list(graph.feature_groups.get(name, []) or [])
                    if group:
                        info['semantic_group'] = group
                    concept_info = dict(info.get('concept_info') or {})
                    concept_info['parents'] = parents
                    concept_info['semantic_graph_entity'] = graph.feature_to_entity.get(name)
                    info['concept_info'] = concept_info
                    self.feature_semantics[idx] = info

                relationships = graph.relationship_records()
                self.domain_knowledge = dict(self.domain_knowledge or {})
                self.domain_knowledge['relationships'] = relationships
                self.domain_knowledge['semantic_graph'] = graph.to_dict()
                self.ontology_hierarchy = {
                    key: sorted(values)
                    for key, values in graph.edges.items()
                    if values
                }
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                # O matching continua válido; a ausência de grafo é explicitamente
                # auditada em vez de ser silenciosamente tratada como semântica.
                self.semantic_graph = None
                self.semantic_graph_summary = {
                    'build_failed': True, 'error': str(exc),
                    'mapped_features': len(accepted_by_feature),
                }

        self._ontology_split_multipliers = None
        return self.feature_semantics

    def get_mapping_statistics(self):
        quality = dict(getattr(self, 'ontology_quality_report', None) or {})
        quality_metrics = dict(quality.get('metrics') or {})
        quality_matches = list(quality.get('matches') or [])
        if quality_metrics.get('total_features') is not None:
            total_features = int(quality_metrics.get('total_features') or 0)
            mapped_features = int(quality_metrics.get('mapped_features') or 0)
            accepted_scores = [
                float(row.get('score', 0.0)) for row in quality_matches
                if row.get('accepted')
            ]
            feature_stats = {
                'total': total_features,
                'mapped': mapped_features,
                'general_fallback': 0,
                'unmapped': max(0, total_features - mapped_features),
                'avg_score': (
                    float(np.mean(accepted_scores)) if accepted_scores
                    else float(quality_metrics.get('mean_matching_confidence', 0.0) or 0.0)
                ),
                'coverage': float(quality_metrics.get('feature_coverage', 0.0) or 0.0),
                'source': 'ontology_quality_gate',
            }
        else:
            feature_stats = {
                'total': len(self.feature_semantics),
                'mapped': sum(
                    1 for f in self.feature_semantics.values() if f.get('ontology_mapped')
                ),
                'general_fallback': sum(
                    1 for f in self.feature_semantics.values() if f.get('general_fallback')
                ),
                'unmapped': len(self.unmapped_features),
                'avg_score': float(np.mean(list(self.mapping_scores.values()))) if self.mapping_scores else 0.0,
                'coverage': (
                    sum(1 for f in self.feature_semantics.values() if f.get('ontology_mapped'))
                    / max(len(self.feature_semantics), 1)
                ) if self.feature_semantics else 0.0,
                'source': 'legacy_feature_semantics',
            }

        stats = {
            'features': feature_stats,
            'classes': {
                'total': len(self.class_semantics),
                'mapped': sum(1 for c in self.class_semantics.values() 
                             if c.get('ontology_concept') is not None),
                'unmapped': len(self.unmapped_classes),
                'avg_score': np.mean([v for k, v in self.mapping_scores.items() 
                                    if k in [c.get('name') for c in self.class_semantics.values()]]) 
                           if self.mapping_scores else 0.0
            },
            'relationships': {
                'hierarchies': len(self.ontology_hierarchy),
                'property_constraints': len(self.property_constraints),
                'total_relationships': max(
                    len(self.domain_knowledge.get('relationships', [])),
                    int((getattr(self, 'semantic_graph_summary', {}) or {}).get('edges', 0) or 0),
                ),
                'semantic_graph_edges': int((getattr(self, 'semantic_graph_summary', {}) or {}).get('edges', 0) or 0),
                'features_with_groups': int((getattr(self, 'semantic_graph_summary', {}) or {}).get('features_with_groups', 0) or 0),
            }
        }
        return stats
