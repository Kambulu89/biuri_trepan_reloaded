import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, log_loss, brier_score_loss,
    classification_report, cohen_kappa_score, balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from scipy.stats import ttest_rel, mannwhitneyu, wilcoxon, chi2, binom, t
from core.c45_j48_tree import C45Tree
from core.biomedical_validation import validate_biomedical_model
import functools
import hashlib
import pickle
import json
import csv
from datetime import datetime
from pathlib import Path
from core.mlp_optimizer import log_metrics_check
from core.feature_alignment import align_feature_spaces, oracle_n_features
from core.scientific_validation import build_scientific_validation_report
from core.protocol_audit import build_protocol_audit
from core.surrogate_acceptance import evaluate_surrogate_acceptance_metrics
from core.c45_baseline_gate import evaluate_c45_baseline_pair

class MetricsComparator:
    
    def __init__(self, n_bootstrap=1000, random_state=42, enable_cache=True):
        self.comparison_results = {}
        self.statistical_tests = {}
        self.visualization_data = {}
        self.c45_tree = C45Tree()
        self.n_bootstrap = n_bootstrap  # Número de iterações bootstrap
        self.random_state = random_state  # Para reprodutibilidade
        self.class_balance_info = {}  # Informações sobre balanceamento de classes
        self.enable_cache = enable_cache  # Habilitar/desabilitar cache
        self._prediction_cache = {}  # Cache manual de predições
        self._execution_history = []  # Histórico de execuções para análise de estabilidade
        self._stability_metrics = {}  # Métricas de estabilidade calculadas
        
    def _hash_array(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.size > 100000:  # Para arrays muito grandes, usar amostra
            # Hash de uma amostra estratificada
            sample_size = min(1000, X.shape[0])
            indices = np.linspace(0, X.shape[0] - 1, sample_size, dtype=int)
            sample = X[indices]
            hash_data = f"{X.shape}_{X.dtype}_{hashlib.md5(sample.tobytes()).hexdigest()}"
        else:
            # Para arrays menores, usar hash completo
            hash_data = f"{X.shape}_{X.dtype}_{hashlib.md5(X.tobytes()).hexdigest()}"
        
        return hash_data
    
    def _get_model_id(self, model):

        if model is None:
            return "None"
        
        # Usar tipo do modelo e ID do objeto como identificador
        model_type = type(model).__name__
        model_id = id(model)
        
        # Para modelos sklearn, tentar obter parâmetros se disponível
        try:
            if hasattr(model, 'get_params'):
                params_str = str(sorted(model.get_params().items()))[:100]  # Limitar tamanho
                return f"{model_type}_{hashlib.md5(params_str.encode()).hexdigest()[:8]}_{model_id}"
        except Exception:
            pass
        
        return f"{model_type}_{model_id}"
    
    def _cached_predict(self, model, X, model_name=None):

        if not self.enable_cache or model is None:
            return model.predict(X)
        
        # Criar chave de cache
        model_id = self._get_model_id(model)
        X_hash = self._hash_array(X)
        cache_key = f"{model_id}_{X_hash}"
        
        # Verificar cache
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        
        # Calcular predições
        y_pred = model.predict(X)
        
        # Armazenar no cache
        self._prediction_cache[cache_key] = y_pred
        
        # Limitar tamanho do cache (remover itens mais antigos se exceder 100)
        if len(self._prediction_cache) > 100:
            # Remover item mais antigo (FIFO - primeiro item adicionado)
            oldest_key = next(iter(self._prediction_cache))
            del self._prediction_cache[oldest_key]
        
        return y_pred
    
    def clear_cache(self):
        
        self._prediction_cache.clear()
    
    def get_cache_stats(self):

        return {
            'cache_enabled': self.enable_cache,
            'cache_size': len(self._prediction_cache),
            'cache_keys': list(self._prediction_cache.keys())[:10]  # Primeiras 10 chaves
        }
        
    def _mlp_oracle_matrix(self, mlp_model, X, matrix_feature_names=None, oracle_feature_names=None):
        """Sub-matriz para o oráculo MLP — alinhamento robusto entre espaços distintos."""
        if X is None or mlp_model is None:
            return X
        n_expected = oracle_n_features(mlp_model)
        oracle_type = "MLP_Onto" if (
            n_expected is not None
            and matrix_feature_names
            and n_expected == len(matrix_feature_names)
            and X.shape[1] == n_expected
        ) else "MLP Original"
        return align_feature_spaces(
            mlp_model,
            X,
            feature_names=matrix_feature_names,
            original_feature_names=oracle_feature_names,
            oracle_type=oracle_type,
        )

    def _prepare_reloaded_predict_matrix(
        self, tree, X, X_reloaded=None, reload_test_transform=None
    ):
        """Alinha X ao n_features da árvore Reloaded (ex.: 14 ARFF → 44 enriquecidas)."""
        if tree is None:
            return X
        X_use = np.asarray(X_reloaded if X_reloaded is not None else X, dtype=float)
        if reload_test_transform is not None:
            try:
                X_use = reload_test_transform(tree, X_use)
            except TypeError:
                X_use = reload_test_transform(X_use)
            X_use = np.asarray(X_use, dtype=float)
        n_tree = int(getattr(getattr(tree, 'tree_', None), 'n_features', X_use.shape[1]))
        if n_tree and X_use.shape[1] != n_tree:
            raise ValueError(
                "Schema incompatível para Trepan Reloaded: a árvore espera "
                f"{n_tree} features, mas recebeu {X_use.shape[1]}. "
                "Forneça X_test_reloaded ou reload_test_transform com os mesmos "
                "nomes e a mesma ordem usados no treino; truncar/preencher colunas "
                "silenciosamente invalidaria as métricas."
            )
        return X_use

    @staticmethod
    def _should_expose_ontological_mlp(fidelity_ref, mlp_model, mlp_model_reloaded):
        """Só expõe o professor ontológico quando ele é realmente o oráculo activo.

        Um espaço Reloaded enriquecido pode existir mesmo quando o gate OOF rejeita
        o professor ontológico. Nesse caso ``mlp_model_reloaded`` pode apontar para
        o MLP original/fallback e não deve originar uma barra duplicada na GUI.
        """
        return bool(
            fidelity_ref == 'mlp_onto'
            and mlp_model_reloaded is not None
            and mlp_model_reloaded is not mlp_model
        )

    @staticmethod
    def _reloaded_fidelity_reference(
        mlp_model_reloaded,
        feature_names_reloaded,
        n_matrix_features=None,
    ):
        """Determina se a fidelidade Reloaded deve referenciar MLP_Onto ou MLP original."""
        if mlp_model_reloaded is None:
            return 'mlp_original'
        oracle_type = getattr(mlp_model_reloaded, 'ORACLE_TYPE', None)
        if oracle_type in {'projected_original', 'mlp_original', 'original'}:
            return 'mlp_original'
        if oracle_type in {'residual_ontological', 'validated_hybrid_ontological', 'mlp_ontological'}:
            return 'mlp_onto'
        n = oracle_n_features(mlp_model_reloaded)
        if n is None:
            return 'mlp_original'
        if n_matrix_features is not None and n == n_matrix_features:
            return 'mlp_onto'
        if feature_names_reloaded is not None and n == len(feature_names_reloaded):
            return 'mlp_onto'
        if (
            feature_names_reloaded is not None
            and n != len(feature_names_reloaded)
        ):
            return 'mlp_original'
        if feature_names_reloaded and n_matrix_features is not None:
            n_base = len([
                f for f in feature_names_reloaded if not str(f).startswith('onto_')
            ])
            if n > n_base and n == n_matrix_features:
                return 'mlp_onto'
        return 'mlp_original'

    def compare_all_models(self, mlp_model, trepan_original_tree, trepan_reloaded_tree,
                          X_test, y_test, X_train=None, y_train=None, 
                          feature_names=None, class_names=None, ontology_active=False,
                          mlp_model_reloaded=None, X_test_reloaded=None, X_train_reloaded=None,
                          feature_names_reloaded=None, reload_test_transform=None,
                          original_feature_names=None, clinical_context=None,
                          feature_types=None, test_row_ids=None,
                          preprocessing_id='caller_provided_preprocessed_matrix',
                          repeat_count=1, evaluation_seed=None,
                          ontology_feature_metadata=None):
        
        self.comparison_results = {}

        use_dual = (
            X_test_reloaded is not None
            and trepan_reloaded_tree is not None
        )
        oracle_names = original_feature_names or feature_names

        print(
            f"[DIAG] compare_all_models: n_test={len(X_test)}, "
            f"n_features={X_test.shape[1]}, n_classes={len(np.unique(y_test))}"
            + (f", dual_reloaded n_features={X_test_reloaded.shape[1]}" if use_dual else "")
        )

        # 0. C4.5 nativo: treina só no conjunto de treino (evita vazamento no teste).
        c45_tree = None
        if X_train is not None and y_train is not None:
            self.c45_tree.train_c45_tree(
                X_train, y_train, feature_names, class_names,
                ontology_mode=False,
                feature_types=feature_types,
            )
            c45_tree = self.c45_tree.tree_model
        
        # Calcular predições do MLP UMA VEZ com cache (evita recalcular múltiplas vezes)
        y_mlp_pred = self._cached_predict(mlp_model, X_test, 'MLP')
        y_mlp_pred_reloaded = None
        fidelity_ref = 'mlp_original'
        if use_dual:
            mlp_oracle = mlp_model_reloaded if mlp_model_reloaded is not None else mlp_model
            fidelity_ref = self._reloaded_fidelity_reference(
                mlp_model_reloaded,
                feature_names_reloaded,
                n_matrix_features=X_test_reloaded.shape[1],
            )
            if fidelity_ref == 'mlp_onto':
                oracle_names_reloaded = feature_names_reloaded
                mlp_oracle = mlp_model_reloaded
            else:
                oracle_names_reloaded = oracle_names
                mlp_oracle = mlp_model
            X_oracle = self._mlp_oracle_matrix(
                mlp_oracle,
                X_test_reloaded,
                feature_names_reloaded,
                oracle_names_reloaded,
            )
            y_mlp_pred_reloaded = self._cached_predict(
                mlp_oracle, X_oracle, 'MLP-Reloaded-oracle'
            )
        
        # 1. Calcular métricas básicas de precisão
        precision_metrics = self._calculate_precision_metrics(
            mlp_model, trepan_original_tree, trepan_reloaded_tree, c45_tree, X_test, y_test,
            y_mlp_pred=y_mlp_pred,
            X_test_reloaded=X_test_reloaded if use_dual else None,
            y_mlp_pred_reloaded=y_mlp_pred_reloaded,
            reload_test_transform=reload_test_transform,
        )
        # Expor o MLP Ontológico/ativo como modelo preditivo separado quando
        # existe espaço ontológico. A comparação visual usa Precisão Macro, não Accuracy.
        if (
            use_dual
            and y_mlp_pred_reloaded is not None
            and self._should_expose_ontological_mlp(fidelity_ref, mlp_model, mlp_model_reloaded)
        ):
            onto_ci = self._calculate_metrics_with_ci(y_test, y_mlp_pred_reloaded)
            precision_metrics['mlp_ontological'] = {
                'accuracy': onto_ci['accuracy']['mean'],
                'precision': onto_ci['precision']['mean'],
                'precision_macro': onto_ci.get('precision_macro', {}).get('mean'),
                'recall': onto_ci['recall']['mean'],
                'recall_macro': onto_ci.get('recall_macro', {}).get('mean'),
                'f1': onto_ci['f1']['mean'],
                'f1_macro': onto_ci.get('f1_macro', {}).get('mean'),
                'balanced_accuracy': float(balanced_accuracy_score(y_test, y_mlp_pred_reloaded)),
                'predictions': y_mlp_pred_reloaded,
                'accuracy_ci': onto_ci['accuracy'],
                'precision_ci': onto_ci['precision'],
                'precision_macro_ci': onto_ci.get('precision_macro'),
                'recall_ci': onto_ci['recall'],
                'recall_macro_ci': onto_ci.get('recall_macro'),
                'f1_ci': onto_ci['f1'],
                'f1_macro_ci': onto_ci.get('f1_macro'),
                'metric_semantics': {
                    'precision_macro': 'precision_score_average_macro',
                    'accuracy': 'accuracy_score',
                },
            }
        
        # 2. Calcular métricas de fidelidade
        fidelity_metrics = self._calculate_fidelity_metrics(
            mlp_model, trepan_original_tree, trepan_reloaded_tree, c45_tree, X_test, y_test,
            feature_names, y_mlp_pred=y_mlp_pred,
            X_test_reloaded=X_test_reloaded if use_dual else None,
            mlp_model_reloaded=mlp_model_reloaded if use_dual else None,
            y_mlp_pred_reloaded=y_mlp_pred_reloaded,
            feature_names_reloaded=feature_names_reloaded,
            reload_test_transform=reload_test_transform,
        )

        for key in ('mlp', 'trepan_original', 'trepan_reloaded', 'c45_j48'):
            p = precision_metrics.get(key)
            f = fidelity_metrics.get(key)
            if p:
                print(
                    f"[DIAG] {key}: precision_macro={p.get('precision_macro', p['precision']) * 100:.1f}%, "
                    f"accuracy_audit={p['accuracy'] * 100:.1f}%"
                )
            if f:
                ref = f.get('fidelity_reference', '')
                ref_note = f" (ref={ref})" if ref else ""
                print(
                    f"[DIAG] {key}: overall_fidelity={f['overall_fidelity'] * 100:.1f}%{ref_note}"
                )

        log_metrics_check(precision_metrics, fidelity_metrics)
        
        # 3. Calcular métricas por classe
        per_class_metrics = self._calculate_per_class_metrics(
            mlp_model, trepan_original_tree, trepan_reloaded_tree, c45_tree, X_test, y_test, class_names,
            y_mlp_pred=y_mlp_pred,
            X_test_reloaded=X_test_reloaded if use_dual else None,
            reload_test_transform=reload_test_transform,
        )
        
        # 4. Validação cruzada (se dados de treino disponíveis)
        cross_val_metrics = {}
        if X_train is not None and y_train is not None:
            cross_val_metrics = self._calculate_cross_validation_metrics(
                mlp_model, trepan_original_tree, trepan_reloaded_tree, c45_tree, X_train, y_train,
                X_train_reloaded=X_train_reloaded if use_dual else None,
                reload_test_transform=reload_test_transform,
            )
        
        # 5. Análise estatística
        statistical_analysis = self._perform_statistical_analysis(
            precision_metrics, fidelity_metrics, X_test, y_test
        )

        paired_predictions = {
            key: (block or {}).get('predictions')
            for key, block in precision_metrics.items()
            if key in {'trepan_original', 'trepan_reloaded', 'c45_j48'}
        }
        scientific_validation = build_scientific_validation_report(
            y_test,
            y_mlp_pred,
            paired_predictions,
            n_bootstrap=max(1000, int(self.n_bootstrap)),
            random_state=self.random_state,
        )

        # Prova de identidade do protocolo: hashes completos do holdout, schema,
        # classes e linhas lógicas. Estes valores são apenas de auditoria.
        active_oracle_model = mlp_model_reloaded if use_dual and mlp_model_reloaded is not None else mlp_model
        protocol_audit = build_protocol_audit(
            X_test,
            y_test,
            X_test_reloaded=X_test_reloaded if use_dual else None,
            feature_names=feature_names,
            feature_names_reloaded=feature_names_reloaded if use_dual else None,
            class_order_original=getattr(mlp_model, 'classes_', None),
            class_order_active_oracle=getattr(active_oracle_model, 'classes_', None),
            test_row_ids=test_row_ids,
            seed=self.random_state if evaluation_seed is None else evaluation_seed,
            repeat_count=repeat_count,
            preprocessing_id=preprocessing_id,
        )

        # Qualidade do professor activo no MESMO teste bloqueado. Isto explica
        # casos em que a árvore tem fidelidade alta mas accuracy baixa.
        active_oracle_test = None
        if y_mlp_pred_reloaded is not None:
            from core.classification_metrics import compute_classification_metrics
            active_oracle_test = compute_classification_metrics(y_test, y_mlp_pred_reloaded)
            active_oracle_test.update({
                'role': 'locked_test_reporting_only',
                'used_for_selection': False,
            })

        # Diagnóstico linha-a-linha pedido na auditoria. As regras OWL são
        # opcionais e só aparecem quando o chamador fornece metadados de origem.
        row_diagnostics = []
        orig_tree_pred = (precision_metrics.get('trepan_original') or {}).get('predictions')
        reloaded_tree_pred = (precision_metrics.get('trepan_reloaded') or {}).get('predictions')
        metadata = ontology_feature_metadata or {}
        semantic_positions = []
        if use_dual and feature_names_reloaded:
            semantic_positions = [
                (i, str(name)) for i, name in enumerate(feature_names_reloaded)
                if str(name).startswith('onto_')
            ]
        row_ids = np.arange(len(y_test)) if test_row_ids is None else np.asarray(test_row_ids)
        for i in range(len(y_test)):
            onto_values = {}
            onto_rules = {}
            if use_dual and X_test_reloaded is not None:
                for pos, name in semantic_positions:
                    value = np.asarray(X_test_reloaded)[i, pos]
                    scalar = value.item() if hasattr(value, 'item') else value
                    if isinstance(scalar, np.generic):
                        scalar = scalar.item()
                    onto_values[name] = scalar
                    info = metadata.get(name) if isinstance(metadata, dict) else None
                    if info:
                        onto_rules[name] = info
            row_diagnostics.append({
                'row_id': (row_ids[i].item() if hasattr(row_ids[i], 'item') else row_ids[i]),
                'true_label': (np.asarray(y_test)[i].item() if hasattr(np.asarray(y_test)[i], 'item') else np.asarray(y_test)[i]),
                'mlp_original_prediction': (y_mlp_pred[i].item() if hasattr(y_mlp_pred[i], 'item') else y_mlp_pred[i]),
                'active_oracle_prediction': (
                    (y_mlp_pred_reloaded[i].item() if hasattr(y_mlp_pred_reloaded[i], 'item') else y_mlp_pred_reloaded[i])
                    if y_mlp_pred_reloaded is not None else
                    (y_mlp_pred[i].item() if hasattr(y_mlp_pred[i], 'item') else y_mlp_pred[i])
                ),
                'trepan_original_prediction': (
                    None if orig_tree_pred is None else
                    (orig_tree_pred[i].item() if hasattr(orig_tree_pred[i], 'item') else orig_tree_pred[i])
                ),
                'trepan_reloaded_prediction': (
                    None if reloaded_tree_pred is None else
                    (reloaded_tree_pred[i].item() if hasattr(reloaded_tree_pred[i], 'item') else reloaded_tree_pred[i])
                ),
                'onto_feature_values': onto_values,
                'owl_rule_or_concept': onto_rules,
            })

        # O teste final pode auditar o contrato da árvore, mas NÃO pode alterar
        # o modelo. O fallback deve ter sido decidido num holdout de aceitação.
        surrogate_quality_audit = None
        if precision_metrics.get('trepan_reloaded') and precision_metrics.get('trepan_original'):
            rel = precision_metrics['trepan_reloaded']
            base = precision_metrics['trepan_original']
            rel_fid = fidelity_metrics.get('trepan_reloaded') or {}
            candidate_complexity = (
                int(trepan_reloaded_tree.get_n_leaves())
                if trepan_reloaded_tree is not None and hasattr(trepan_reloaded_tree, 'get_n_leaves') else None
            )
            baseline_complexity = (
                int(trepan_original_tree.get_n_leaves())
                if trepan_original_tree is not None and hasattr(trepan_original_tree, 'get_n_leaves') else None
            )
            c45_block = precision_metrics.get('c45_j48')
            surrogate_quality_audit = evaluate_surrogate_acceptance_metrics(
                {
                    'accuracy': rel.get('accuracy', 0.0),
                    'precision_macro': rel.get('precision_macro', 0.0),
                    'recall_macro': rel.get('recall_macro', 0.0),
                    'balanced_accuracy': rel.get('balanced_accuracy', 0.0),
                    'macro_f1': rel.get('f1_macro', 0.0),
                },
                {
                    'accuracy': base.get('accuracy', 0.0),
                    'precision_macro': base.get('precision_macro', 0.0),
                    'recall_macro': base.get('recall_macro', 0.0),
                    'balanced_accuracy': base.get('balanced_accuracy', 0.0),
                    'macro_f1': base.get('f1_macro', 0.0),
                },
                fidelity_to_active_oracle=rel_fid.get('active_oracle_fidelity'),
                candidate_complexity=candidate_complexity,
                baseline_complexity=baseline_complexity,
                c45_metrics=(
                    {
                        'accuracy': c45_block.get('accuracy', 0.0),
                        'precision_macro': c45_block.get('precision_macro', 0.0),
                        'recall_macro': c45_block.get('recall_macro', 0.0),
                        'balanced_accuracy': c45_block.get('balanced_accuracy', 0.0),
                        'macro_f1': c45_block.get('f1_macro', 0.0),
                    } if c45_block else None
                ),
                require_c45_noninferiority=bool(c45_block),
                selection_role='locked_test_audit_only',
            )

        biomedical_validation = {}
        # Sempre calcula o que é possível. Sem identificadores de paciente/site,
        # o claim guard permanece bloqueado e explica os campos em falta.
        if True:
            clinical_context = dict(clinical_context or {})
            model_inputs = {
                'mlp': (mlp_model, X_test),
                'trepan_original': (trepan_original_tree, X_test),
                'trepan_reloaded': (
                    trepan_reloaded_tree,
                    self._prepare_reloaded_predict_matrix(
                        trepan_reloaded_tree, X_test, X_test_reloaded,
                        reload_test_transform,
                    ) if trepan_reloaded_tree is not None else None,
                ),
                'c45_j48': (c45_tree, X_test),
            }
            for key, (model, matrix) in model_inputs.items():
                block = precision_metrics.get(key)
                if model is None or matrix is None or not block:
                    continue
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(matrix)
                biomedical_validation[key] = validate_biomedical_model(
                    y_test, block['predictions'], y_proba=probabilities,
                    classes=getattr(model, 'classes_', None), **clinical_context,
                )
        
        # 6. Detectar classes desbalanceadas
        self.class_balance_info = self._detect_class_imbalance(y_test, class_names)
        
        # Gate C4.5 separado: C4.5 continua baseline supervisionado, nunca oráculo.
        # No teste bloqueado este resultado é apenas auditoria; não seleciona modelos.
        c45_baseline_gate = evaluate_c45_baseline_pair(
            precision_metrics.get('trepan_original'),
            precision_metrics.get('trepan_reloaded'),
            precision_metrics.get('c45_j48'),
            tolerance=0.0,
            selection_role='locked_test_audit_only',
        )

        # 7. Compilar resultados
        self.comparison_results = {
            'precision': precision_metrics,
            'fidelity': fidelity_metrics,
            'per_class': per_class_metrics,
            'cross_validation': cross_val_metrics,
            'statistical_analysis': statistical_analysis,
            'scientific_validation': scientific_validation,
            'protocol_audit': protocol_audit,
            'active_oracle_locked_test_metrics': active_oracle_test,
            'row_diagnostics': row_diagnostics,
            'surrogate_quality_audit': surrogate_quality_audit,
            'c45_baseline_gate': c45_baseline_gate,
            'biomedical_validation': biomedical_validation,
            'class_balance': self.class_balance_info,
            'model_info': {
                'mlp_type': type(mlp_model).__name__,
                'trepan_original_depth': trepan_original_tree.get_depth() if trepan_original_tree else None,
                'trepan_reloaded_depth': trepan_reloaded_tree.get_depth() if trepan_reloaded_tree else None,
                'n_test_samples': len(X_test),
                'n_features': X_test.shape[1],
                'n_classes': len(np.unique(y_test))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 8. Registrar execução para análise de estabilidade
        self._record_execution(precision_metrics, fidelity_metrics, cross_val_metrics)
        
        # 9. Calcular métricas de estabilidade se houver múltiplas execuções
        if len(self._execution_history) > 1:
            self._calculate_stability_metrics()
        
        return self.comparison_results
    
    def _calculate_additional_metrics(self, model, X_test, y_test, y_pred, model_name=''):

        additional_metrics = {}
        
        # Matriz de confusão
        try:
            # sklearn.confusion_matrix ordena classes automaticamente
            unique_classes = np.unique(np.concatenate([y_test, y_pred]))
            cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
            additional_metrics['confusion_matrix'] = cm.tolist()
            additional_metrics['confusion_matrix_classes'] = unique_classes.tolist()
            
            # Calcular sensibilidade e especificidade (para cada classe)
            n_classes = len(unique_classes)
            sensitivity_list = []
            specificity_list = []
            
            for i in range(n_classes):
                # Verdadeiros positivos, falsos negativos, falsos positivos, verdadeiros negativos
                tp = int(cm[i, i])
                fn = int(np.sum(cm[i, :]) - tp)
                fp = int(np.sum(cm[:, i]) - tp)
                tn = int(np.sum(cm) - (tp + fn + fp))
                
                # Sensibilidade (Recall) = TP / (TP + FN)
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                sensitivity_list.append(float(sens))
                
                # Especificidade = TN / (TN + FP)
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificity_list.append(float(spec))
            
            additional_metrics['sensitivity'] = sensitivity_list
            additional_metrics['specificity'] = specificity_list
            additional_metrics['mean_sensitivity'] = float(np.mean(sensitivity_list))
            additional_metrics['mean_specificity'] = float(np.mean(specificity_list))
        except Exception as e:
            additional_metrics['confusion_matrix'] = None
            additional_metrics['error_cm'] = str(e)
        
        # AUC-ROC (para classes multiclasse: one-vs-rest)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                n_classes = len(np.unique(y_test))
                
                if n_classes == 2:
                    # Binário: usar diretamente
                    try:
                        auc_roc = roc_auc_score(y_test, y_proba[:, 1])
                        additional_metrics['auc_roc'] = float(auc_roc)
                        additional_metrics['auc_roc_method'] = 'binary'
                    except Exception:
                        additional_metrics['auc_roc'] = None
                else:
                    # Multiclasse: one-vs-rest
                    try:
                        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
                        # Se há apenas 1 classe em y_test, usar todas as classes
                        if y_test_binarized.shape[1] == 1:
                            y_test_binarized = np.hstack([1 - y_test_binarized, y_test_binarized])
                            # Ajustar y_proba se necessário
                            if y_proba.shape[1] == 1:
                                y_proba = np.hstack([1 - y_proba, y_proba])
                        
                        auc_roc_ovr = roc_auc_score(y_test_binarized, y_proba, average='macro', multi_class='ovr')
                        additional_metrics['auc_roc'] = float(auc_roc_ovr)
                        additional_metrics['auc_roc_method'] = 'one-vs-rest (macro)'
                        
                        # AUC por classe (individual)
                        auc_per_class = []
                        for i in range(y_test_binarized.shape[1]):
                            try:
                                auc_class = roc_auc_score(y_test_binarized[:, i], y_proba[:, i])
                                auc_per_class.append(float(auc_class))
                            except Exception:
                                auc_per_class.append(None)
                        additional_metrics['auc_roc_per_class'] = auc_per_class
                    except Exception as e:
                        additional_metrics['auc_roc'] = None
                        additional_metrics['auc_roc_error'] = str(e)
            else:
                additional_metrics['auc_roc'] = None
                additional_metrics['auc_roc_note'] = 'Modelo não suporta predict_proba'
        except Exception as e:
            additional_metrics['auc_roc'] = None
            additional_metrics['auc_roc_error'] = str(e)
        
        # Log-loss (entropia cruzada)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                # Garantir que probabilidades estão normalizadas e não têm zeros
                y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
                y_proba = y_proba / y_proba.sum(axis=1, keepdims=1)
                
                log_loss_value = log_loss(y_test, y_proba)
                additional_metrics['log_loss'] = float(log_loss_value)
            else:
                additional_metrics['log_loss'] = None
                additional_metrics['log_loss_note'] = 'Modelo não suporta predict_proba'
        except Exception as e:
            additional_metrics['log_loss'] = None
            additional_metrics['log_loss_error'] = str(e)
        
        # Brier Score (para cada classe, média ponderada)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                n_classes = len(np.unique(y_test))
                brier_scores = []
                
                # Binarizar y_test para calcular Brier Score por classe
                y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
                
                for i in range(n_classes):
                    try:
                        if y_test_binarized.shape[1] > i:
                            brier = brier_score_loss(y_test_binarized[:, i], y_proba[:, i])
                            brier_scores.append(float(brier))
                    except Exception:
                        brier_scores.append(None)
                
                if brier_scores and all(x is not None for x in brier_scores):
                    # Média ponderada pelo número de amostras de cada classe
                    class_counts = [np.sum(y_test == class_label) for class_label in np.unique(y_test)]
                    weights = np.array(class_counts) / len(y_test)
                    mean_brier = np.average(brier_scores, weights=weights)
                    additional_metrics['brier_score'] = float(mean_brier)
                    additional_metrics['brier_score_per_class'] = brier_scores
                else:
                    additional_metrics['brier_score'] = None
            else:
                additional_metrics['brier_score'] = None
                additional_metrics['brier_score_note'] = 'Modelo não suporta predict_proba'
        except Exception as e:
            additional_metrics['brier_score'] = None
            additional_metrics['brier_score_error'] = str(e)
        
        # Cohen's Kappa (concordância ajustada ao acaso)
        try:
            kappa = cohen_kappa_score(y_test, y_pred)
            additional_metrics['cohens_kappa'] = float(kappa)
        except Exception as e:
            additional_metrics['cohens_kappa'] = None
            additional_metrics['kappa_error'] = str(e)
        
        return additional_metrics
    
    def _calculate_precision_metrics(self, mlp_model, trepan_original_tree, trepan_reloaded_tree, c45_tree, X_test, y_test, y_mlp_pred=None,
                                   X_test_reloaded=None, y_mlp_pred_reloaded=None, reload_test_transform=None):

        metrics = {}
        
        # MLP Original - usar predições já calculadas se disponíveis ou cache
        if y_mlp_pred is None:
            y_mlp_pred = self._cached_predict(mlp_model, X_test, 'MLP')
        metrics_with_ci = self._calculate_metrics_with_ci(y_test, y_mlp_pred)
        
        # Métricas adicionais para MLP
        additional_metrics_mlp = self._calculate_additional_metrics(mlp_model, X_test, y_test, y_mlp_pred, 'MLP')
        
        metrics['mlp'] = {
            'accuracy': metrics_with_ci['accuracy']['mean'],
            'precision': metrics_with_ci['precision']['mean'],
            'recall': metrics_with_ci['recall']['mean'],
            'f1': metrics_with_ci['f1']['mean'],
            'predictions': y_mlp_pred,
            # Intervalos de confiança
            'accuracy_ci': metrics_with_ci['accuracy'],
            'precision_ci': metrics_with_ci['precision'],
            'precision_macro_ci': metrics_with_ci.get('precision_macro'),
            'recall_ci': metrics_with_ci['recall'],
            'recall_macro_ci': metrics_with_ci.get('recall_macro'),
            'f1_ci': metrics_with_ci['f1'],
            'f1_macro_ci': metrics_with_ci.get('f1_macro'),
            # Métricas adicionais
            'additional_metrics': additional_metrics_mlp
        }
        
        # Trepan-Original
        if trepan_original_tree is not None:
            y_trepan_orig_pred = self._cached_predict(trepan_original_tree, X_test, 'Trepan-Original')
            metrics_with_ci = self._calculate_metrics_with_ci(y_test, y_trepan_orig_pred)
            additional_metrics_trepan_orig = self._calculate_additional_metrics(
                trepan_original_tree, X_test, y_test, y_trepan_orig_pred, 'Trepan-Original'
            )
            metrics['trepan_original'] = {
                'accuracy': metrics_with_ci['accuracy']['mean'],
                'precision': metrics_with_ci['precision']['mean'],
                'recall': metrics_with_ci['recall']['mean'],
                'f1': metrics_with_ci['f1']['mean'],
                'predictions': y_trepan_orig_pred,
                # Intervalos de confiança
                'accuracy_ci': metrics_with_ci['accuracy'],
                'precision_ci': metrics_with_ci['precision'],
            'precision_macro_ci': metrics_with_ci.get('precision_macro'),
                'recall_ci': metrics_with_ci['recall'],
            'recall_macro_ci': metrics_with_ci.get('recall_macro'),
                'f1_ci': metrics_with_ci['f1'],
            'f1_macro_ci': metrics_with_ci.get('f1_macro'),
                # Métricas adicionais
                'additional_metrics': additional_metrics_trepan_orig
            }
        else:
            metrics['trepan_original'] = None
        
        # Trepan-Reloaded
        if trepan_reloaded_tree is not None:
            X_rel = self._prepare_reloaded_predict_matrix(
                trepan_reloaded_tree, X_test, X_test_reloaded, reload_test_transform
            )
            y_trepan_reloaded_pred = self._cached_predict(trepan_reloaded_tree, X_rel, 'Trepan-Reloaded')
            metrics_with_ci = self._calculate_metrics_with_ci(y_test, y_trepan_reloaded_pred)
            additional_metrics_trepan_reloaded = self._calculate_additional_metrics(
                trepan_reloaded_tree, X_rel, y_test, y_trepan_reloaded_pred, 'Trepan-Reloaded'
            )
            metrics['trepan_reloaded'] = {
                'accuracy': metrics_with_ci['accuracy']['mean'],
                'precision': metrics_with_ci['precision']['mean'],
                'recall': metrics_with_ci['recall']['mean'],
                'f1': metrics_with_ci['f1']['mean'],
                'predictions': y_trepan_reloaded_pred,
                # Intervalos de confiança
                'accuracy_ci': metrics_with_ci['accuracy'],
                'precision_ci': metrics_with_ci['precision'],
            'precision_macro_ci': metrics_with_ci.get('precision_macro'),
                'recall_ci': metrics_with_ci['recall'],
            'recall_macro_ci': metrics_with_ci.get('recall_macro'),
                'f1_ci': metrics_with_ci['f1'],
            'f1_macro_ci': metrics_with_ci.get('f1_macro'),
                # Métricas adicionais
                'additional_metrics': additional_metrics_trepan_reloaded
            }
        else:
            metrics['trepan_reloaded'] = None
        
        # C4.5 nativo
        if c45_tree is not None:
            y_c45_pred = self._cached_predict(c45_tree, X_test, 'C4.5-Nativo')
            metrics_with_ci = self._calculate_metrics_with_ci(y_test, y_c45_pred)
            additional_metrics_c45 = self._calculate_additional_metrics(
                c45_tree, X_test, y_test, y_c45_pred, 'C4.5-Nativo'
            )
            metrics['c45_j48'] = {
                'accuracy': metrics_with_ci['accuracy']['mean'],
                'precision': metrics_with_ci['precision']['mean'],
                'recall': metrics_with_ci['recall']['mean'],
                'f1': metrics_with_ci['f1']['mean'],
                'predictions': y_c45_pred,
                # Intervalos de confiança
                'accuracy_ci': metrics_with_ci['accuracy'],
                'precision_ci': metrics_with_ci['precision'],
            'precision_macro_ci': metrics_with_ci.get('precision_macro'),
                'recall_ci': metrics_with_ci['recall'],
            'recall_macro_ci': metrics_with_ci.get('recall_macro'),
                'f1_ci': metrics_with_ci['f1'],
            'f1_macro_ci': metrics_with_ci.get('f1_macro'),
                # Métricas adicionais
                'additional_metrics': additional_metrics_c45
            }
        else:
            metrics['c45_j48'] = None

        # Métricas adequadas a desbalanceamento, expostas para auditoria
        # clínica e para evitar que "precisão" weighted seja confundida com
        # accuracy ou desempenho da classe minoritária.
        for block in metrics.values():
            if not block or block.get('predictions') is None:
                continue
            pred = block['predictions']
            block['balanced_accuracy'] = float(balanced_accuracy_score(y_test, pred))
            block['precision_macro'] = float(
                precision_score(y_test, pred, average='macro', zero_division=0)
            )
            block['recall_macro'] = float(
                recall_score(y_test, pred, average='macro', zero_division=0)
            )
            block['f1_macro'] = float(
                f1_score(y_test, pred, average='macro', zero_division=0)
            )
            block['metric_semantics'] = {
                'precision': 'weighted_precision',
                'accuracy': 'overall_accuracy',
                'primary_imbalanced': 'balanced_accuracy_and_macro_f1',
            }
        return metrics
    
    def _analyze_fidelity_by_confidence_region(self, mlp_model, y_mlp_pred, y_tree_pred, X_test):

        try:
            if not hasattr(mlp_model, 'predict_proba'):
                return None
            
            # Calcular probabilidades do MLP
            mlp_probas = mlp_model.predict_proba(X_test)
            
            # Calcular confiança (max probabilidade) e entropia
            max_proba = np.max(mlp_probas, axis=1)
            entropy = -np.sum(mlp_probas * np.log(mlp_probas + 1e-10), axis=1)
            
            # Definir limites: alta confiança = max_proba > 0.7, baixa = max_proba <= 0.7
            high_confidence_mask = max_proba > 0.7
            low_confidence_mask = max_proba <= 0.7
            
            # Calcular fidelidade por região
            agreements = (y_mlp_pred == y_tree_pred)
            
            high_conf_fidelity = np.mean(agreements[high_confidence_mask]) if np.any(high_confidence_mask) else None
            low_conf_fidelity = np.mean(agreements[low_confidence_mask]) if np.any(low_confidence_mask) else None
            
            return {
                'high_confidence': {
                    'fidelity': float(high_conf_fidelity) if high_conf_fidelity is not None else None,
                    'n_samples': int(np.sum(high_confidence_mask)),
                    'threshold': 0.7
                },
                'low_confidence': {
                    'fidelity': float(low_conf_fidelity) if low_conf_fidelity is not None else None,
                    'n_samples': int(np.sum(low_confidence_mask)),
                    'threshold': 0.7
                },
                'mean_confidence': float(np.mean(max_proba)),
                'mean_entropy': float(np.mean(entropy))
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_fidelity_by_error_type(self, y_mlp_pred, y_tree_pred, y_true=None):

        agreements = (y_mlp_pred == y_tree_pred)
        
        # Se y_true disponível, podemos categorizar melhor
        if y_true is not None:
            mlp_correct = (y_mlp_pred == y_true)
            tree_correct = (y_tree_pred == y_true)
            
            # Categorias
            both_correct = mlp_correct & tree_correct & agreements
            both_wrong = ~mlp_correct & ~tree_correct  # Ambos erram (pode ou não concordar)
            mlp_correct_tree_wrong = mlp_correct & ~tree_correct & ~agreements  # MLP acerta, árvore erra
            mlp_wrong_tree_correct = ~mlp_correct & tree_correct & ~agreements  # MLP erra, árvore acerta
            disagreement = ~agreements  # Qualquer discordância
            
            return {
                'both_correct': {
                    'count': int(np.sum(both_correct)),
                    'percentage': float(np.mean(both_correct)) * 100
                },
                'both_wrong': {
                    'count': int(np.sum(both_wrong)),
                    'percentage': float(np.mean(both_wrong)) * 100
                },
                'mlp_correct_tree_wrong': {
                    'count': int(np.sum(mlp_correct_tree_wrong)),
                    'percentage': float(np.mean(mlp_correct_tree_wrong)) * 100,
                    'description': 'MLP acertó, árbol falló en relación al MLP (falso negativo de fidelidad)'
                },
                'mlp_wrong_tree_correct': {
                    'count': int(np.sum(mlp_wrong_tree_correct)),
                    'percentage': float(np.mean(mlp_wrong_tree_correct)) * 100,
                    'description': 'MLP falló, árbol acertó (pero divergió del MLP)'
                },
                'total_disagreements': {
                    'count': int(np.sum(disagreement)),
                    'percentage': float(np.mean(disagreement)) * 100
                }
            }
        else:
            # Sem y_true, apenas análise de concordância
            return {
                'agreements': {
                    'count': int(np.sum(agreements)),
                    'percentage': float(np.mean(agreements)) * 100
                },
                'disagreements': {
                    'count': int(np.sum(~agreements)),
                    'percentage': float(np.mean(~agreements)) * 100
                }
            }
    
    def _analyze_fidelity_by_feature_region(self, X_test, y_mlp_pred, y_tree_pred, n_regions=4):

        try:
            # Calcular centro dos dados (mediana para ser robusto a outliers)
            center = np.median(X_test, axis=0)
            
            # Calcular distância de cada ponto ao centro
            distances = np.linalg.norm(X_test - center, axis=1)
            
            # Dividir em quartis (4 regiões)
            quartiles = np.percentile(distances, [25, 50, 75])
            
            # Criar máscaras para cada região
            agreements = (y_mlp_pred == y_tree_pred)
            
            region_0 = distances <= quartiles[0]  # Mais próximo do centro
            region_1 = (distances > quartiles[0]) & (distances <= quartiles[1])
            region_2 = (distances > quartiles[1]) & (distances <= quartiles[2])
            region_3 = distances > quartiles[2]  # Mais distante do centro (possíveis outliers)
            
            regions = [
                {'mask': region_0, 'name': 'Región Central (Q1)', 'fidelity': np.mean(agreements[region_0]) if np.any(region_0) else None, 'n_samples': int(np.sum(region_0))},
                {'mask': region_1, 'name': 'Región Media-Baja (Q2)', 'fidelity': np.mean(agreements[region_1]) if np.any(region_1) else None, 'n_samples': int(np.sum(region_1))},
                {'mask': region_2, 'name': 'Región Media-Alta (Q3)', 'fidelity': np.mean(agreements[region_2]) if np.any(region_2) else None, 'n_samples': int(np.sum(region_2))},
                {'mask': region_3, 'name': 'Región Periférica (Q4)', 'fidelity': np.mean(agreements[region_3]) if np.any(region_3) else None, 'n_samples': int(np.sum(region_3))}
            ]
            
            return {
                'regions': [
                    {
                        'name': r['name'],
                        'fidelity': float(r['fidelity']) if r['fidelity'] is not None else None,
                        'n_samples': r['n_samples'],
                        'percentage': (r['n_samples'] / len(X_test)) * 100
                    }
                    for r in regions
                ],
                'mean_distance': float(np.mean(distances)),
                'median_distance': float(np.median(distances))
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_discrepancies(self, mlp_model, tree_model, X_test, y_mlp_pred, y_tree_pred, 
                                y_true=None, feature_names=None, max_examples=10):

        discrepancies_mask = (y_mlp_pred != y_tree_pred)
        
        if not np.any(discrepancies_mask):
            return {
                'total': 0,
                'percentage': 0.0,
                'examples': [],
                'characteristics': {},
                'common_patterns': {},
                'feature_importance_in_disagreements': {}
            }
        
        X_discrepant = X_test[discrepancies_mask]
        y_mlp_discrepant = y_mlp_pred[discrepancies_mask]
        y_tree_discrepant = y_tree_pred[discrepancies_mask]
        indices_discrepant = np.where(discrepancies_mask)[0]
        
        X_agreement = X_test[~discrepancies_mask]
        
        examples = []
        
        try:
            # Obter probabilidades do MLP se disponível
            has_proba = hasattr(mlp_model, 'predict_proba')
            if has_proba:
                mlp_probas_discrepant = mlp_model.predict_proba(X_discrepant)
                mlp_max_proba = np.max(mlp_probas_discrepant, axis=1)
            else:
                mlp_max_proba = None
                mlp_probas_discrepant = None
            
            # Limitar número de exemplos analisados
            n_examples = min(max_examples, len(X_discrepant))
            np.random.seed(self.random_state)
            example_indices = np.random.choice(len(X_discrepant), size=n_examples, replace=False)
            
            for idx in example_indices:
                example = {
                    'sample_index': int(indices_discrepant[idx]),
                    'mlp_prediction': str(y_mlp_discrepant[idx]),
                    'tree_prediction': str(y_tree_discrepant[idx]),
                }
                
                if has_proba:
                    example['mlp_confidence'] = float(mlp_max_proba[idx])
                    example['mlp_probabilities'] = {
                        f'class_{i}': float(prob) 
                        for i, prob in enumerate(mlp_probas_discrepant[idx])
                    }
                
                # Adicionar valores das features principais (se feature_names disponível)
                if feature_names is not None and len(feature_names) > 0:
                    n_features_to_show = min(5, len(feature_names))
                    top_indices = np.argsort(np.abs(X_discrepant[idx]))[-n_features_to_show:][::-1]
                    example['key_features'] = {
                        feature_names[i]: float(X_discrepant[idx][i])
                        for i in top_indices if i < len(feature_names)
                    }
                
                examples.append(example)
            
            # Análise de características dos casos de discordância
            characteristics = self._analyze_disagreement_characteristics(
                X_discrepant, X_agreement, y_mlp_discrepant, y_tree_discrepant,
                mlp_max_proba, feature_names
            )
            
            # Análise de padrões comuns
            # Filtrar y_true para casos de discrepância se disponível
            y_true_discrepant = None
            if y_true is not None and len(y_true) == len(X_test):
                y_true_discrepant = y_true[discrepancies_mask]
            elif y_true is not None and len(y_true) == len(y_mlp_discrepant):
                # y_true já está filtrado
                y_true_discrepant = y_true
            
            common_patterns = self._analyze_common_disagreement_patterns(
                X_discrepant, y_mlp_discrepant, y_tree_discrepant, y_true_discrepant,
                feature_names, mlp_model, tree_model
            )
            
            # Importância de features nos casos problemáticos
            feature_importance = self._analyze_feature_importance_in_disagreements(
                X_discrepant, X_agreement, y_mlp_discrepant, y_tree_discrepant,
                feature_names, tree_model
            )
        
        except Exception as e:
            return {
                'total': int(np.sum(discrepancies_mask)),
                'percentage': float(np.mean(discrepancies_mask)) * 100,
                'error': str(e),
                'examples': [],
                'characteristics': {},
                'common_patterns': {},
                'feature_importance_in_disagreements': {}
            }
        
        return {
            'total': int(np.sum(discrepancies_mask)),
            'percentage': float(np.mean(discrepancies_mask)) * 100,
            'examples': examples,
            'mean_mlp_confidence': float(np.mean(mlp_max_proba)) if mlp_max_proba is not None else None,
            'characteristics': characteristics,
            'common_patterns': common_patterns,
            'feature_importance_in_disagreements': feature_importance
        }
    
    def _analyze_disagreement_characteristics(self, X_discrepant, X_agreement, y_mlp_discrepant, 
                                               y_tree_discrepant, mlp_confidence_discrepant, 
                                               feature_names=None):

        try:
            characteristics = {}
            
            # Comparar estatísticas descritivas
            if len(X_agreement) > 0 and X_discrepant.shape[1] > 0:
                # Estatísticas das features em casos de discordância vs concordância
                mean_discrepant = np.mean(X_discrepant, axis=0)
                std_discrepant = np.std(X_discrepant, axis=0)
                mean_agreement = np.mean(X_agreement, axis=0)
                std_agreement = np.std(X_agreement, axis=0)
                
                # Diferenças significativas
                feature_differences = np.abs(mean_discrepant - mean_agreement)
                std_differences = np.abs(std_discrepant - std_agreement)
                
                # Features com maior diferença (potenciais causas)
                top_diff_indices = np.argsort(feature_differences)[-5:][::-1]
                
                characteristics['mean_differences'] = {
                    f'feature_{i}' if feature_names is None or i >= len(feature_names) 
                    else feature_names[i]: float(feature_differences[i])
                    for i in top_diff_indices
                }
                
                # Outliers detectados (usando IQR)
                n_outliers_per_feature = []
                for feat_idx in range(X_discrepant.shape[1]):
                    if len(X_agreement) > 0:
                        Q1_agree = np.percentile(X_agreement[:, feat_idx], 25)
                        Q3_agree = np.percentile(X_agreement[:, feat_idx], 75)
                        IQR = Q3_agree - Q1_agree
                        lower_bound = Q1_agree - 1.5 * IQR
                        upper_bound = Q3_agree + 1.5 * IQR
                        
                        outliers = np.sum((X_discrepant[:, feat_idx] < lower_bound) | 
                                         (X_discrepant[:, feat_idx] > upper_bound))
                        n_outliers_per_feature.append(int(outliers))
                    else:
                        n_outliers_per_feature.append(0)
                
                top_outlier_features = np.argsort(n_outliers_per_feature)[-5:][::-1]
                characteristics['outlier_features'] = {
                    f'feature_{i}' if feature_names is None or i >= len(feature_names) 
                    else feature_names[i]: n_outliers_per_feature[i]
                    for i in top_outlier_features if n_outliers_per_feature[i] > 0
                }
                
                # Distância média ao centro dos dados de concordância
                if len(X_agreement) > 0:
                    center_agreement = np.median(X_agreement, axis=0)
                    distances_discrepant = np.linalg.norm(X_discrepant - center_agreement, axis=1)
                    distances_agreement_samples = np.linalg.norm(X_agreement - center_agreement, axis=1)
                    
                    characteristics['distance_from_center'] = {
                        'mean_discrepant': float(np.mean(distances_discrepant)),
                        'mean_agreement': float(np.mean(distances_agreement_samples)),
                        'difference': float(np.mean(distances_discrepant) - np.mean(distances_agreement_samples))
                    }
            
            # Análise de confiança do MLP
            if mlp_confidence_discrepant is not None:
                characteristics['mlp_confidence_stats'] = {
                    'mean': float(np.mean(mlp_confidence_discrepant)),
                    'std': float(np.std(mlp_confidence_discrepant)),
                    'min': float(np.min(mlp_confidence_discrepant)),
                    'max': float(np.max(mlp_confidence_discrepant)),
                    'median': float(np.median(mlp_confidence_discrepant))
                }
            
        except Exception as e:
            characteristics['error'] = str(e)
        
        return characteristics
    
    def _analyze_common_disagreement_patterns(self, X_discrepant, y_mlp_discrepant, y_tree_discrepant,
                                               y_true, feature_names, mlp_model, tree_model):

        try:
            patterns = {}
            
            # Padrões de transição de classes (MLP -> Árvore)
            if len(y_mlp_discrepant) > 0 and len(y_tree_discrepant) > 0:
                transitions = {}
                for mlp_cls, tree_cls in zip(y_mlp_discrepant, y_tree_discrepant):
                    transition = f"{mlp_cls} -> {tree_cls}"
                    transitions[transition] = transitions.get(transition, 0) + 1
                
                # Ordenar por frequência
                sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
                patterns['class_transitions'] = {
                    trans: count 
                    for trans, count in sorted_transitions[:10]  # Top 10 transições
                }
            
            # Se tem o mesmo tamanho de y_mlp_discrepant, assume que já está filtrado
            if y_true is not None:
                # Se y_true tem o mesmo tamanho que y_mlp_discrepant, assume que já está filtrado
                if len(y_true) == len(y_mlp_discrepant):
                    discrepant_y_true = y_true
                else:
                    # y_true é completo, então não podemos filtrar diretamente sem a máscara original
                    # Neste caso, não podemos analisar classes verdadeiras (necessitaria índice da discrepância)
                    discrepant_y_true = None
                
                if discrepant_y_true is not None and len(discrepant_y_true) > 0:
                    # Quais classes verdadeiras levam mais a discordâncias?
                    true_classes_in_disagreement = {}
                    for true_cls in discrepant_y_true:
                        true_classes_in_disagreement[true_cls] = true_classes_in_disagreement.get(true_cls, 0) + 1
                    
                    patterns['problematic_true_classes'] = dict(
                        sorted(true_classes_in_disagreement.items(), key=lambda x: x[1], reverse=True)[:5]
                    )
                    
                    # Quando MLP ou árvore acertam o verdadeiro
                    mlp_correct_in_disagreement = np.sum(y_mlp_discrepant == discrepant_y_true)
                    tree_correct_in_disagreement = np.sum(y_tree_discrepant == discrepant_y_true)
                    
                    patterns['accuracy_in_disagreement'] = {
                        'mlp_correct_count': int(mlp_correct_in_disagreement),
                        'tree_correct_count': int(tree_correct_in_disagreement),
                        'mlp_correct_pct': float(mlp_correct_in_disagreement / len(discrepant_y_true) * 100) if len(discrepant_y_true) > 0 else 0,
                        'tree_correct_pct': float(tree_correct_in_disagreement / len(discrepant_y_true) * 100) if len(discrepant_y_true) > 0 else 0
                    }
            
            # Padrões de valores de features (clusters de discrepâncias)
            if X_discrepant.shape[0] > 3 and X_discrepant.shape[1] > 0:
                # Detectar features com valores consistentemente diferentes
                feature_ranges_discrepant = {
                    i: {
                        'min': float(np.min(X_discrepant[:, i])),
                        'max': float(np.max(X_discrepant[:, i])),
                        'mean': float(np.mean(X_discrepant[:, i])),
                        'median': float(np.median(X_discrepant[:, i]))
                    }
                    for i in range(min(10, X_discrepant.shape[1]))  # Limitar a 10 features
                }
                
                patterns['feature_value_ranges_in_disagreement'] = {
                    (feature_names[i] if feature_names and i < len(feature_names) else f'feature_{i}'): ranges
                    for i, ranges in feature_ranges_discrepant.items()
                }
            
        except Exception as e:
            patterns['error'] = str(e)
        
        return patterns
    
    def _analyze_feature_importance_in_disagreements(self, X_discrepant, X_agreement, 
                                                     y_mlp_discrepant, y_tree_discrepant,
                                                     feature_names, tree_model):

        try:
            importance_analysis = {}
            
            # 1. Feature importance da árvore (se disponível)
            if tree_model is not None and hasattr(tree_model, 'feature_importances_'):
                tree_importance = tree_model.feature_importances_
                
                # Feature importance top
                top_indices = np.argsort(tree_importance)[-10:][::-1]
                importance_analysis['tree_feature_importance'] = {
                    (feature_names[i] if feature_names and i < len(feature_names) else f'feature_{i}'): 
                    float(tree_importance[i])
                    for i in top_indices
                }
            
            # 2. Análise de variância: features com maior diferença entre grupos
            if len(X_agreement) > 0 and X_discrepant.shape[0] > 0:
                # Calcular F-statistic (análise de variância)
                f_scores = []
                for feat_idx in range(min(X_discrepant.shape[1], 50)):  # Limitar a 50 features
                    try:
                        # Calcular variância entre grupos vs dentro dos grupos
                        mean_discrepant = np.mean(X_discrepant[:, feat_idx])
                        mean_agreement = np.mean(X_agreement[:, feat_idx])
                        var_discrepant = np.var(X_discrepant[:, feat_idx])
                        var_agreement = np.var(X_agreement[:, feat_idx])
                        
                        # F-statistic aproximado
                        between_group_var = (mean_discrepant - mean_agreement) ** 2
                        within_group_var = (var_discrepant + var_agreement) / 2
                        
                        if within_group_var > 1e-10:
                            f_score = between_group_var / within_group_var
                            f_scores.append((feat_idx, float(f_score)))
                    except Exception:
                        continue
                
                # Ordenar por F-score
                f_scores.sort(key=lambda x: x[1], reverse=True)
                
                importance_analysis['discriminative_features'] = {
                    (feature_names[i] if feature_names and i < len(feature_names) else f'feature_{i}'): 
                    f_score
                    for i, f_score in f_scores[:10]  # Top 10 features discriminativas
                }
            
            # 3. Correlação: features mais correlacionadas com ocorrência de discordâncias
            if X_discrepant.shape[1] > 0:
                # Para cada feature, calcular correlação com "é discrepância"
                # Criar vetor binário: 1 = discrepância, 0 = concordância
                all_X = np.vstack([X_discrepant, X_agreement]) if len(X_agreement) > 0 else X_discrepant
                is_discrepancy = np.concatenate([
                    np.ones(len(X_discrepant)),
                    np.zeros(len(X_agreement)) if len(X_agreement) > 0 else []
                ])
                
                correlations = []
                for feat_idx in range(min(all_X.shape[1], 50)):  # Limitar a 50 features
                    try:
                        from scipy.stats import pearsonr
                        corr, _ = pearsonr(all_X[:, feat_idx], is_discrepancy)
                        if not np.isnan(corr):
                            correlations.append((feat_idx, float(corr)))
                    except Exception:
                        continue
                
                # Ordenar por correlação absoluta
                correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                
                importance_analysis['correlation_with_disagreement'] = {
                    (feature_names[i] if feature_names and i < len(feature_names) else f'feature_{i}'): 
                    corr
                    for i, corr in correlations[:10]  # Top 10 correlações
                }
            
        except Exception as e:
            importance_analysis['error'] = str(e)
        
        return importance_analysis
    
    def _calculate_fidelity_metrics(self, mlp_model, trepan_original_tree, trepan_reloaded_tree, c45_tree, X_test, y_test, feature_names=None, y_mlp_pred=None,
                                    X_test_reloaded=None, mlp_model_reloaded=None, y_mlp_pred_reloaded=None, feature_names_reloaded=None,
                                    reload_test_transform=None):

        fidelity = {}
        
        # Calcular predições do MLP UMA VEZ no início com cache (evita recalcular múltiplas vezes)
        if y_mlp_pred is None:
            y_mlp_pred = self._cached_predict(mlp_model, X_test, 'MLP')
        
        # Fidelidade Trepan-Original
        if trepan_original_tree is not None:
            y_trepan_orig_pred = self._cached_predict(trepan_original_tree, X_test, 'Trepan-Original')
            
            # Calcular métricas com IC usando bootstrap
            overall_fidelity_ci = self._bootstrap_confidence_interval(
                y_mlp_pred, y_trepan_orig_pred, accuracy_score
            )
            precision_fidelity_ci = self._bootstrap_confidence_interval(
                y_mlp_pred, y_trepan_orig_pred,
                lambda y1, y2: precision_score(y1, y2, average='weighted', zero_division=0)
            )
            
            # Análises granulares
            confidence_region_analysis = self._analyze_fidelity_by_confidence_region(
                mlp_model, y_mlp_pred, y_trepan_orig_pred, X_test
            )
            error_type_analysis = self._analyze_fidelity_by_error_type(
                y_mlp_pred, y_trepan_orig_pred, y_true=y_test
            )
            feature_region_analysis = self._analyze_fidelity_by_feature_region(
                X_test, y_mlp_pred, y_trepan_orig_pred
            )
            discrepancies_analysis = self._analyze_discrepancies(
                mlp_model, trepan_original_tree, X_test, 
                y_mlp_pred, y_trepan_orig_pred, y_true=y_test,
                feature_names=feature_names
            )
            
            exact_fidelity = float(accuracy_score(y_mlp_pred, y_trepan_orig_pred))
            overall_fidelity_ci['point_estimate'] = exact_fidelity
            fidelity['trepan_original'] = {
                'overall_fidelity': exact_fidelity,
                'precision_fidelity': precision_fidelity_ci['mean'],
                'recall_fidelity': recall_score(y_mlp_pred, y_trepan_orig_pred, average='weighted', zero_division=0),
                'f1_fidelity': f1_score(y_mlp_pred, y_trepan_orig_pred, average='weighted', zero_division=0),
                'agreement_rate': np.mean(y_mlp_pred == y_trepan_orig_pred),
                'fidelity_reference': 'mlp_original',
                'fidelity_to_mlp_original': exact_fidelity,
                'active_oracle_fidelity': exact_fidelity,
                'fidelity_to_active_oracle': exact_fidelity,
                'feature_space': 'original',
                # Intervalos de confiança
                'overall_fidelity_ci': overall_fidelity_ci,
                'precision_fidelity_ci': precision_fidelity_ci,
                # Análises granulares
                'by_confidence_region': confidence_region_analysis,
                'by_error_type': error_type_analysis,
                'by_feature_region': feature_region_analysis,
                'discrepancies': discrepancies_analysis
            }
        else:
            fidelity['trepan_original'] = None
        
        # Fidelidade Trepan-Reloaded
        if trepan_reloaded_tree is not None:
            X_rel = self._prepare_reloaded_predict_matrix(
                trepan_reloaded_tree, X_test, X_test_reloaded, reload_test_transform
            )
            fidelity_ref = self._reloaded_fidelity_reference(
                mlp_model_reloaded,
                feature_names_reloaded,
                n_matrix_features=X_rel.shape[1],
            )
            if fidelity_ref == 'mlp_onto':
                mlp_ref_model = mlp_model_reloaded if mlp_model_reloaded is not None else mlp_model
                X_mlp_ref = X_test_reloaded if X_test_reloaded is not None else X_test
                if y_mlp_pred_reloaded is None and mlp_ref_model is not None:
                    X_oracle = self._mlp_oracle_matrix(
                        mlp_ref_model,
                        X_mlp_ref,
                        feature_names_reloaded,
                        feature_names_reloaded,
                    )
                    y_mlp_pred_reloaded = self._cached_predict(
                        mlp_ref_model, X_oracle, 'MLP-Reloaded-oracle'
                    )
                y_mlp_ref = y_mlp_pred_reloaded if y_mlp_pred_reloaded is not None else y_mlp_pred
            else:
                y_mlp_ref = y_mlp_pred
                mlp_ref_model = mlp_model
                X_mlp_ref = X_test

            y_trepan_reloaded_pred = self._cached_predict(trepan_reloaded_tree, X_rel, 'Trepan-Reloaded')
            
            # Calcular métricas com IC usando bootstrap (referência = oráculo activo)
            overall_fidelity_ci = self._bootstrap_confidence_interval(
                y_mlp_ref, y_trepan_reloaded_pred, accuracy_score
            )
            precision_fidelity_ci = self._bootstrap_confidence_interval(
                y_mlp_ref, y_trepan_reloaded_pred,
                lambda y1, y2: precision_score(y1, y2, average='weighted', zero_division=0)
            )
            
            confidence_region_analysis = self._analyze_fidelity_by_confidence_region(
                mlp_ref_model, y_mlp_ref, y_trepan_reloaded_pred, X_mlp_ref
            )
            error_type_analysis = self._analyze_fidelity_by_error_type(
                y_mlp_ref, y_trepan_reloaded_pred, y_true=y_test
            )
            feature_region_analysis = self._analyze_fidelity_by_feature_region(
                X_mlp_ref, y_mlp_ref, y_trepan_reloaded_pred
            )
            discrepancies_analysis = self._analyze_discrepancies(
                mlp_ref_model, trepan_reloaded_tree, X_mlp_ref,
                y_mlp_ref, y_trepan_reloaded_pred, y_true=y_test,
                feature_names=feature_names_reloaded if fidelity_ref == 'mlp_onto' else feature_names
            )
            
            exact_active_fidelity = float(accuracy_score(y_mlp_ref, y_trepan_reloaded_pred))
            exact_control_fidelity = float(accuracy_score(y_mlp_pred, y_trepan_reloaded_pred))
            overall_fidelity_ci['point_estimate'] = exact_active_fidelity
            fidelity['trepan_reloaded'] = {
                'overall_fidelity': exact_active_fidelity,
                'precision_fidelity': precision_fidelity_ci['mean'],
                'recall_fidelity': recall_score(y_mlp_ref, y_trepan_reloaded_pred, average='weighted', zero_division=0),
                'f1_fidelity': f1_score(y_mlp_ref, y_trepan_reloaded_pred, average='weighted', zero_division=0),
                'agreement_rate': np.mean(y_mlp_ref == y_trepan_reloaded_pred),
                'fidelity_reference': fidelity_ref,
                'feature_space': 'enriched' if fidelity_ref == 'mlp_onto' else 'original',
                # Comparacao controlada contra o MESMO MLP Original usado pelo
                # Trepan Original e pelo baseline C4.5 nativo.
                'fidelity_to_mlp_original': exact_control_fidelity,
                'active_oracle_fidelity': exact_active_fidelity,
                'fidelity_to_active_oracle': exact_active_fidelity,
                'tree_eval_space': 'augmented_test_scaled',
                # Intervalos de confiança
                'overall_fidelity_ci': overall_fidelity_ci,
                'precision_fidelity_ci': precision_fidelity_ci,
                # Análises granulares
                'by_confidence_region': confidence_region_analysis,
                'by_error_type': error_type_analysis,
                'by_feature_region': feature_region_analysis,
                'discrepancies': discrepancies_analysis
            }
        else:
            fidelity['trepan_reloaded'] = None
        
        # Fidelidade C4.5 nativo
        if c45_tree is not None:
            y_c45_pred = self._cached_predict(c45_tree, X_test, 'C4.5-Nativo')
            
            # Calcular métricas com IC usando bootstrap
            overall_fidelity_ci = self._bootstrap_confidence_interval(
                y_mlp_pred, y_c45_pred, accuracy_score
            )
            precision_fidelity_ci = self._bootstrap_confidence_interval(
                y_mlp_pred, y_c45_pred,
                lambda y1, y2: precision_score(y1, y2, average='weighted', zero_division=0)
            )
            
            # Análises granulares
            confidence_region_analysis = self._analyze_fidelity_by_confidence_region(
                mlp_model, y_mlp_pred, y_c45_pred, X_test
            )
            error_type_analysis = self._analyze_fidelity_by_error_type(
                y_mlp_pred, y_c45_pred, y_true=y_test
            )
            feature_region_analysis = self._analyze_fidelity_by_feature_region(
                X_test, y_mlp_pred, y_c45_pred
            )
            discrepancies_analysis = self._analyze_discrepancies(
                mlp_model, c45_tree, X_test, 
                y_mlp_pred, y_c45_pred, y_true=y_test,
                feature_names=feature_names
            )
            
            exact_fidelity = float(accuracy_score(y_mlp_pred, y_c45_pred))
            overall_fidelity_ci['point_estimate'] = exact_fidelity
            fidelity['c45_j48'] = {
                'overall_fidelity': exact_fidelity,
                'precision_fidelity': precision_fidelity_ci['mean'],
                'recall_fidelity': recall_score(y_mlp_pred, y_c45_pred, average='weighted', zero_division=0),
                'f1_fidelity': f1_score(y_mlp_pred, y_c45_pred, average='weighted', zero_division=0),
                'agreement_rate': np.mean(y_mlp_pred == y_c45_pred),
                'fidelity_reference': 'mlp_original',
                'fidelity_to_mlp_original': exact_fidelity,
                'active_oracle_fidelity': exact_fidelity,
                'fidelity_to_active_oracle': exact_fidelity,
                'feature_space': 'original',
                'fidelity_role': 'auxiliary_agreement_only_supervised_baseline',
                # Intervalos de confiança
                'overall_fidelity_ci': overall_fidelity_ci,
                'precision_fidelity_ci': precision_fidelity_ci,
                # Análises granulares
                'by_confidence_region': confidence_region_analysis,
                'by_error_type': error_type_analysis,
                'by_feature_region': feature_region_analysis,
                'discrepancies': discrepancies_analysis
            }
        else:
            fidelity['c45_j48'] = None
        
        return fidelity
    
    def _calculate_per_class_metrics(self, mlp_model, trepan_original_tree, trepan_reloaded_tree, c45_tree,
                                   X_test, y_test, class_names, y_mlp_pred=None, X_test_reloaded=None,
                                   reload_test_transform=None):

        per_class = {}
        
        # MLP Original - usar predições já calculadas se disponíveis ou cache
        if y_mlp_pred is None:
            y_mlp_pred = self._cached_predict(mlp_model, X_test, 'MLP')
        per_class['mlp'] = self._calculate_class_metrics(y_test, y_mlp_pred, class_names)
        
        # Trepan-Original
        if trepan_original_tree is not None:
            y_trepan_orig_pred = self._cached_predict(trepan_original_tree, X_test, 'Trepan-Original')
            per_class['trepan_original'] = self._calculate_class_metrics(y_test, y_trepan_orig_pred, class_names)
        else:
            per_class['trepan_original'] = None
        
        # Trepan-Reloaded
        if trepan_reloaded_tree is not None:
            X_rel = self._prepare_reloaded_predict_matrix(
                trepan_reloaded_tree, X_test, X_test_reloaded, reload_test_transform
            )
            y_trepan_reloaded_pred = self._cached_predict(trepan_reloaded_tree, X_rel, 'Trepan-Reloaded')
            per_class['trepan_reloaded'] = self._calculate_class_metrics(y_test, y_trepan_reloaded_pred, class_names)
        else:
            per_class['trepan_reloaded'] = None
        
        # C4.5 nativo
        if c45_tree is not None:
            y_c45_pred = self._cached_predict(c45_tree, X_test, 'C4.5-Nativo')
            per_class['c45_j48'] = self._calculate_class_metrics(y_test, y_c45_pred, class_names)
        else:
            per_class['c45_j48'] = None
        
        return per_class
    
    def _calculate_class_metrics(self, y_true, y_pred, class_names):
        
        unique_classes = np.unique(y_true)
        class_metrics = {}
        
        for i, class_label in enumerate(unique_classes):
            if class_names and i < len(class_names):
                class_name = class_names[i]
            else:
                class_name = f"Class_{class_label}"
            
            # Métricas binárias para esta classe
            y_true_binary = (y_true == class_label).astype(int)
            y_pred_binary = (y_pred == class_label).astype(int)
            
            if len(np.unique(y_true_binary)) > 1:  # Só calcula se há ambas as classes
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            else:
                precision = recall = f1 = 0.0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': np.sum(y_true == class_label),
                'true_positives': np.sum((y_true == class_label) & (y_pred == class_label)),
                'false_positives': np.sum((y_true != class_label) & (y_pred == class_label)),
                'false_negatives': np.sum((y_true == class_label) & (y_pred != class_label))
            }
        
        return class_metrics
    
    def _calculate_cross_validation_metrics(self, mlp_model, trepan_original_tree, trepan_reloaded_tree, c45_tree, X_train, y_train,
                                            X_train_reloaded=None, reload_test_transform=None):

        cv_metrics = {}
        
        # Criar StratifiedKFold limitado pela classe minoritaria.
        _, counts = np.unique(y_train, return_counts=True)
        n_splits = max(2, min(5, int(counts.min()))) if len(counts) > 1 else 2
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Validação cruzada MLP
        try:
            cv_scores_mlp = cross_val_score(mlp_model, X_train, y_train, cv=skf, scoring='accuracy')
            cv_metrics['mlp'] = {
                'mean_accuracy': np.mean(cv_scores_mlp),
                'std_accuracy': np.std(cv_scores_mlp),
                'scores': cv_scores_mlp
            }
        except Exception as e:
            cv_metrics['mlp'] = {'error': str(e)}
        
        # Uma árvore substituta pré-extraída NÃO pode ser submetida a
        # cross_val_score contra y_real: isso a transforma numa árvore
        # supervisionada comum e deixa de medir o pipeline TREPAN.
        if trepan_original_tree is not None:
            cv_metrics['trepan_original'] = {
                'not_applicable': True,
                'reason': 'requires_nested_refit_of_oracle_and_extractor_per_fold',
            }
        else:
            cv_metrics['trepan_original'] = None
        
        # Validação cruzada Trepan-Reloaded
        if trepan_reloaded_tree is not None:
            cv_metrics['trepan_reloaded'] = {
                'not_applicable': True,
                'reason': 'requires_nested_refit_of_ontology_oracle_and_extractor_per_fold',
            }
        else:
            cv_metrics['trepan_reloaded'] = None
        
        # Validação cruzada C4.5 nativo
        if c45_tree is not None:
            try:
                cv_scores_c45 = cross_val_score(c45_tree, X_train, y_train, cv=skf, scoring='accuracy')
                cv_metrics['c45_j48'] = {
                    'mean_accuracy': np.mean(cv_scores_c45),
                    'std_accuracy': np.std(cv_scores_c45),
                    'scores': cv_scores_c45
                }
            except Exception as e:
                cv_metrics['c45_j48'] = {'error': str(e)}
        else:
            cv_metrics['c45_j48'] = None
        
        return cv_metrics
    
    def _bootstrap_confidence_interval(self, y_true, y_pred, metric_func, n_bootstrap=None, confidence=0.95):

        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap
        
        np.random.seed(self.random_state)
        
        n_samples = len(y_true)
        bootstrap_scores = []
        
        # Bootstrap: reamostragem com reposição
        for _ in range(n_bootstrap):
            # Reamostrar índices com reposição
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = np.array(y_true)[indices]
            y_pred_boot = np.array(y_pred)[indices]
            
            # Calcular métrica na amostra bootstrap
            try:
                score = metric_func(y_true_boot, y_pred_boot)
                if not np.isnan(score):
                    bootstrap_scores.append(score)
            except Exception:
                continue
        
        if len(bootstrap_scores) == 0:
            return {
                'mean': None,
                'std': None,
                'ci_lower': None,
                'ci_upper': None,
                'confidence': confidence
            }
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calcular percentis para intervalo de confiança
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        mean_score = np.mean(bootstrap_scores)
        std_score = np.std(bootstrap_scores)
        
        return {
            'mean': float(mean_score),
            'std': float(std_score),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'confidence': confidence,
            'n_bootstrap': len(bootstrap_scores)
        }
    
    def _detect_class_imbalance(self, y_true, class_names=None, imbalance_threshold=0.05):

        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        n_samples = len(y_true)
        class_proportions = class_counts / n_samples
        
        imbalance_info = {
            'n_classes': len(unique_classes),
            'n_samples': n_samples,
            'class_distribution': {},
            'imbalanced_classes': [],
            'warnings': []
        }
        
        # Identificar classes desbalanceadas (< threshold)
        for i, cls in enumerate(unique_classes):
            prop = class_proportions[i]
            count = class_counts[i]
            
            cls_name = class_names[i] if class_names and i < len(class_names) else f"Class_{cls}"
            
            imbalance_info['class_distribution'][cls_name] = {
                'count': int(count),
                'proportion': float(prop),
                'percentage': float(prop * 100)
            }
            
            if prop < imbalance_threshold:
                imbalance_info['imbalanced_classes'].append({
                    'class': cls_name,
                    'count': int(count),
                    'proportion': float(prop),
                    'percentage': float(prop * 100)
                })
                imbalance_info['warnings'].append(
                    f"⚠️ Classe '{cls_name}' tem apenas {prop*100:.2f}% dos dados ({count} amostras)"
                )
        
        return imbalance_info
    
    def _calculate_metrics_with_ci(self, y_true, y_pred, include_macro=True):

        metrics_with_ci = {}
        
        # Acurácia com IC
        metrics_with_ci['accuracy'] = self._bootstrap_confidence_interval(
            y_true, y_pred, accuracy_score
        )
        
        # Precisão weighted com IC
        try:
            metrics_with_ci['precision_weighted'] = self._bootstrap_confidence_interval(
                y_true, y_pred, 
                lambda yt, yp: precision_score(yt, yp, average='weighted', zero_division=0)
            )
            # Compatibilidade: manter 'precision' como weighted
            metrics_with_ci['precision'] = metrics_with_ci['precision_weighted']
        except Exception:
            metrics_with_ci['precision'] = metrics_with_ci['precision_weighted'] = {'mean': None, 'std': None, 'ci_lower': None, 'ci_upper': None}
        
        # Precisão macro com IC (se solicitado)
        if include_macro:
            try:
                metrics_with_ci['precision_macro'] = self._bootstrap_confidence_interval(
                    y_true, y_pred, 
                    lambda yt, yp: precision_score(yt, yp, average='macro', zero_division=0)
                )
            except Exception:
                metrics_with_ci['precision_macro'] = {'mean': None, 'std': None, 'ci_lower': None, 'ci_upper': None}
        
        # Recall weighted com IC
        try:
            metrics_with_ci['recall_weighted'] = self._bootstrap_confidence_interval(
                y_true, y_pred,
                lambda yt, yp: recall_score(yt, yp, average='weighted', zero_division=0)
            )
            # Compatibilidade: manter 'recall' como weighted
            metrics_with_ci['recall'] = metrics_with_ci['recall_weighted']
        except Exception:
            metrics_with_ci['recall'] = metrics_with_ci['recall_weighted'] = {'mean': None, 'std': None, 'ci_lower': None, 'ci_upper': None}
        
        # Recall macro com IC (se solicitado)
        if include_macro:
            try:
                metrics_with_ci['recall_macro'] = self._bootstrap_confidence_interval(
                    y_true, y_pred,
                    lambda yt, yp: recall_score(yt, yp, average='macro', zero_division=0)
                )
            except Exception:
                metrics_with_ci['recall_macro'] = {'mean': None, 'std': None, 'ci_lower': None, 'ci_upper': None}
        
        # F1-Score weighted com IC
        try:
            metrics_with_ci['f1_weighted'] = self._bootstrap_confidence_interval(
                y_true, y_pred,
                lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0)
            )
            # Compatibilidade: manter 'f1' como weighted
            metrics_with_ci['f1'] = metrics_with_ci['f1_weighted']
        except Exception:
            metrics_with_ci['f1'] = metrics_with_ci['f1_weighted'] = {'mean': None, 'std': None, 'ci_lower': None, 'ci_upper': None}
        
        # F1-Score macro com IC (se solicitado)
        if include_macro:
            try:
                metrics_with_ci['f1_macro'] = self._bootstrap_confidence_interval(
                    y_true, y_pred,
                    lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0)
                )
            except Exception:
                metrics_with_ci['f1_macro'] = {'mean': None, 'std': None, 'ci_lower': None, 'ci_upper': None}
        
        return metrics_with_ci
    
    def _format_metric_with_ci(self, metric_dict, metric_name='accuracy'):

        if metric_dict is None:
            return "N/A"
        
        mean = metric_dict.get('mean')
        if mean is None:
            return "N/A"
        
        # Garantir que todos os valores são numéricos
        try:
            mean = float(mean)
            std = float(metric_dict.get('std', 0) or 0)
            ci_lower = float(metric_dict.get('ci_lower', mean) or mean)
            ci_upper = float(metric_dict.get('ci_upper', mean) or mean)
            confidence = float(metric_dict.get('confidence', 0.95) or 0.95)
            confidence_pct = int(confidence * 100)
            
            # Formato: 0.85 ± 0.02 (IC 95%: [0.83, 0.87])
            return f"{mean:.4f} ± {std:.4f} (IC {confidence_pct}%: [{ci_lower:.4f}, {ci_upper:.4f}])"
        except (TypeError, ValueError) as e:
            # Se algum valor não puder ser convertido, retorna formato simplificado
            try:
                return f"{mean:.4f}"
            except (TypeError, ValueError):
                return "N/A"
    
    def _perform_statistical_analysis(self, precision_metrics, fidelity_metrics, X_test, y_test):

        analysis = {}
        
        # Obter previsões para testes estatísticos
        y_mlp_pred = precision_metrics['mlp']['predictions']
        
        # Análise de diferenças de precisão com testes estatísticos
        if precision_metrics['trepan_original'] is not None:
            mlp_acc = precision_metrics['mlp']['accuracy']
            trepan_orig_acc = precision_metrics['trepan_original']['accuracy']
            y_trepan_orig_pred = precision_metrics['trepan_original']['predictions']
            
            # Teste de McNemar (para comparar dois classificadores)
            mcnemar_result = self._perform_mcnemar_test(y_mlp_pred, y_trepan_orig_pred, y_test)
            
            # Teste t pareado (para métricas contínuas - usando diferenças individuais)
            ttest_result = self._perform_paired_ttest(y_mlp_pred, y_trepan_orig_pred, y_test)
            
            # Teste de Wilcoxon signed-rank (para distribuições não-normais)
            wilcoxon_result = self._perform_wilcoxon_test(y_mlp_pred, y_trepan_orig_pred, y_test)
            
            analysis['trepan_original_vs_mlp'] = {
                'accuracy_difference': trepan_orig_acc - mlp_acc,
                'accuracy_relative_difference': (trepan_orig_acc - mlp_acc) / mlp_acc * 100,
                'fidelity': fidelity_metrics['trepan_original']['overall_fidelity'] if fidelity_metrics['trepan_original'] else None,
                'mcnemar_test': mcnemar_result,
                'paired_ttest': ttest_result,
                'wilcoxon_test': wilcoxon_result
            }
        
        if precision_metrics['trepan_reloaded'] is not None:
            mlp_acc = precision_metrics['mlp']['accuracy']
            trepan_reloaded_acc = precision_metrics['trepan_reloaded']['accuracy']
            y_trepan_reloaded_pred = precision_metrics['trepan_reloaded']['predictions']
            
            # Teste de McNemar
            mcnemar_result = self._perform_mcnemar_test(y_mlp_pred, y_trepan_reloaded_pred, y_test)
            
            # Teste t pareado
            ttest_result = self._perform_paired_ttest(y_mlp_pred, y_trepan_reloaded_pred, y_test)
            
            # Teste de Wilcoxon signed-rank
            wilcoxon_result = self._perform_wilcoxon_test(y_mlp_pred, y_trepan_reloaded_pred, y_test)
            
            analysis['trepan_reloaded_vs_mlp'] = {
                'accuracy_difference': trepan_reloaded_acc - mlp_acc,
                'accuracy_relative_difference': (trepan_reloaded_acc - mlp_acc) / mlp_acc * 100,
                'fidelity': fidelity_metrics['trepan_reloaded']['overall_fidelity'] if fidelity_metrics['trepan_reloaded'] else None,
                'mcnemar_test': mcnemar_result,
                'paired_ttest': ttest_result,
                'wilcoxon_test': wilcoxon_result
            }
        
        # Comparação entre Trepan-Original e Trepan-Reloaded
        if (precision_metrics['trepan_original'] is not None and 
            precision_metrics['trepan_reloaded'] is not None):
            
            trepan_orig_acc = precision_metrics['trepan_original']['accuracy']
            trepan_reloaded_acc = precision_metrics['trepan_reloaded']['accuracy']
            y_trepan_orig_pred = precision_metrics['trepan_original']['predictions']
            y_trepan_reloaded_pred = precision_metrics['trepan_reloaded']['predictions']
            
            # Teste de McNemar
            mcnemar_result = self._perform_mcnemar_test(y_trepan_orig_pred, y_trepan_reloaded_pred, y_test)
            
            # Teste t pareado
            ttest_result = self._perform_paired_ttest(y_trepan_orig_pred, y_trepan_reloaded_pred, y_test)
            
            # Teste de Wilcoxon signed-rank
            wilcoxon_result = self._perform_wilcoxon_test(y_trepan_orig_pred, y_trepan_reloaded_pred, y_test)
            
            analysis['trepan_original_vs_reloaded'] = {
                'accuracy_difference': trepan_reloaded_acc - trepan_orig_acc,
                'accuracy_relative_difference': (trepan_reloaded_acc - trepan_orig_acc) / trepan_orig_acc * 100,
                'fidelity_difference': (fidelity_metrics['trepan_reloaded']['overall_fidelity'] - 
                                      fidelity_metrics['trepan_original']['overall_fidelity']) if (
                    fidelity_metrics['trepan_reloaded'] and fidelity_metrics['trepan_original']) else None,
                'mcnemar_test': mcnemar_result,
                'paired_ttest': ttest_result,
                'wilcoxon_test': wilcoxon_result
            }

        # Comparações pareadas do C4.5 com as duas árvores TREPAN. A diferença
        # é sempre "segundo menos primeiro", tornando o sinal inequívoco.
        pair_specs = (
            ('trepan_original_vs_c45', 'trepan_original', 'c45_j48'),
            ('c45_vs_reloaded', 'c45_j48', 'trepan_reloaded'),
        )
        for result_key, first_key, second_key in pair_specs:
            first = precision_metrics.get(first_key)
            second = precision_metrics.get(second_key)
            if not first or not second:
                continue
            first_pred = first['predictions']
            second_pred = second['predictions']
            first_fid = fidelity_metrics.get(first_key) or {}
            second_fid = fidelity_metrics.get(second_key) or {}
            first_acc = float(first['accuracy'])
            second_acc = float(second['accuracy'])
            analysis[result_key] = {
                'accuracy_difference': second_acc - first_acc,
                'accuracy_relative_difference': (
                    (second_acc - first_acc) / first_acc * 100 if first_acc else None
                ),
                'fidelity_difference': (
                    float(second_fid['overall_fidelity']) - float(first_fid['overall_fidelity'])
                    if second_fid.get('overall_fidelity') is not None
                    and first_fid.get('overall_fidelity') is not None else None
                ),
                'mcnemar_test': self._perform_mcnemar_test(first_pred, second_pred, y_test),
                'paired_ttest': self._perform_paired_ttest(first_pred, second_pred, y_test),
                'wilcoxon_test': self._perform_wilcoxon_test(first_pred, second_pred, y_test),
            }
        
        return analysis
    
    def _perform_mcnemar_test(self, y_pred1, y_pred2, y_true):

        try:
            # Construir tabela de contingência 2x2
            # a: ambos corretos
            # b: modelo1 correto, modelo2 incorreto
            # c: modelo1 incorreto, modelo2 correto
            # d: ambos incorretos
            
            correct1 = (y_pred1 == y_true)
            correct2 = (y_pred2 == y_true)
            
            a = np.sum(correct1 & correct2)
            b = np.sum(correct1 & ~correct2)
            c = np.sum(~correct1 & correct2)
            d = np.sum(~correct1 & ~correct2)
            
            # Tabela de contingência para McNemar
            # McNemar requer b + c >= 25 para uso da distribuição qui-quadrado
            # Para amostras menores, usa correção de continuidade
            contingency_table = np.array([[a, b], [c, d]])
            
            # Realizar teste de McNemar
            # Se b + c < 25, usa correção de continuidade
            if b + c < 25:
                # Correção de continuidade (teste binomial)
                n = b + c
                if n == 0:
                    return {
                        'statistic': 0.0,
                        'p_value': 1.0,
                        'significant': False,
                        'note': 'Nenhuma discordância entre modelos (b+c=0)'
                    }
                # Sob H0 (p=0.5), b ~ Binomial(n, 0.5)
                p_value = 2 * min(binom.cdf(min(b, c), n, 0.5), 1 - binom.cdf(min(b, c)-1, n, 0.5))
                statistic = min(b, c)
            else:
                # Usar qui-quadrado (distribuição assintótica)
                statistic = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0
                p_value = 1 - chi2.cdf(statistic, df=1)
            
            alpha = 0.05
            significant = p_value < alpha
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': significant,
                'contingency_table': [[int(a), int(b)], [int(c), int(d)]],
                'interpretation': self._interpret_significance(p_value, 'McNemar')
            }
        except Exception as e:
            return {
                'statistic': None,
                'p_value': None,
                'significant': None,
                'error': str(e)
            }
    
    def _perform_paired_ttest(self, y_pred1, y_pred2, y_true):

        try:
            # Calcular acertos/erros para cada amostra (0 = erro, 1 = acerto)
            correct1 = (y_pred1 == y_true).astype(float)
            correct2 = (y_pred2 == y_true).astype(float)
            
            # Calcular diferenças pareadas
            differences = correct2 - correct1
            
            # Se todas as diferenças são iguais (variação zero), não podemos fazer o teste
            if np.std(differences) == 0:
                mean_diff = np.mean(differences)
                return {
                    'statistic': 0.0,
                    'p_value': 1.0 if abs(mean_diff) < 1e-10 else 0.0,
                    'significant': False,
                    'mean_difference': float(mean_diff),
                    'note': 'Nenhuma variabilidade nas diferenças'
                }
            
            # Realizar teste t pareado
            statistic, p_value = ttest_rel(correct2, correct1)
            
            alpha = 0.05
            significant = p_value < alpha
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': significant,
                'mean_difference': float(np.mean(differences)),
                'interpretation': self._interpret_significance(p_value, 't-test pareado')
            }
        except Exception as e:
            return {
                'statistic': None,
                'p_value': None,
                'significant': None,
                'error': str(e)
            }
    
    def _perform_wilcoxon_test(self, y_pred1, y_pred2, y_true):

        try:
            # Calcular acertos/erros para cada amostra (0 = erro, 1 = acerto)
            correct1 = (y_pred1 == y_true).astype(float)
            correct2 = (y_pred2 == y_true).astype(float)
            
            # Calcular diferenças pareadas
            differences = correct2 - correct1
            
            # Remover zeros (sem diferença)
            nonzero_diffs = differences[differences != 0]
            
            if len(nonzero_diffs) == 0:
                return {
                    'statistic': 0.0,
                    'p_value': 1.0,
                    'significant': False,
                    'note': 'Nenhuma diferença entre modelos'
                }
            
            # Realizar teste de Wilcoxon signed-rank
            statistic, p_value = wilcoxon(correct2, correct1, alternative='two-sided')
            
            alpha = 0.05
            significant = p_value < alpha
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': significant,
                'mean_difference': float(np.mean(differences)),
                'interpretation': self._interpret_significance(p_value, 'Wilcoxon signed-rank')
            }
        except Exception as e:
            return {
                'statistic': None,
                'p_value': None,
                'significant': None,
                'error': str(e)
            }
    
    def _interpret_significance(self, p_value, test_name):
        
        if p_value is None:
            return "La prueba no pudo realizarse"
        
        if p_value < 0.001:
            return f"*** Diferencia altamente significativa (p < 0.001)"
        elif p_value < 0.01:
            return f"** Diferencia muy significativa (p < 0.01)"
        elif p_value < 0.05:
            return f"* Diferencia significativa (p < 0.05)"
        elif p_value < 0.10:
            return f"Diferencia marginalmente significativa (p < 0.10)"
        else:
            return f"Diferencia no significativa (p >= 0.05)"
    
    def generate_comparison_report(self):
        
        if not self.comparison_results:
            return "❌ Ninguna comparación se ha realizado aún."
        
        report = f"\n{'='*80}\n"
        report += f"📊 INFORME COMPARATIVO DE PRECISÃO MACRO E FIDELIDADE\n"
        report += f"Trepan-Original vs C4.5-Nativo vs Trepan-Reloaded\n"
        report += f"{'='*80}\n\n"
        
        # Protocolo: nenhuma hierarquia e pressuposta antes dos resultados.
        report += f"🔬 PROTOCOLO DE INTERPRETACIÓN:\n"
        report += f"{'='*50}\n"
        report += f"   • Sin ganador predefinido.\n"
        report += f"   • Test final reservado para una única evaluación.\n"
        report += f"   • Comparaciones pareadas sobre las mismas muestras.\n"
        report += f"   • Fidelidad comparativa calculada contra el mismo MLP Original.\n"
        report += f"   • C4.5 é uma implementação nativa: gain ratio, missing fracionário e poda pessimista.\n"
        report += f"   • A MLP não é candidata no ranking; atua somente como oráculo de fidelidade.\n\n"

        protocol = self.comparison_results.get('protocol_audit') or {}
        if protocol:
            report += "🔐 IDENTIDADE DO PROTOCOLO BLOQUEADO:\n"
            report += f"{'='*50}\n"
            report += f"   • Linhas de teste: {protocol.get('n_test_rows')}\n"
            report += f"   • Hash linhas lógicas: {protocol.get('logical_test_rows_sha256')}\n"
            report += f"   • Hash X original: {protocol.get('X_test_original_sha256')}\n"
            if protocol.get('X_test_reloaded_sha256'):
                report += f"   • Hash X Reloaded: {protocol.get('X_test_reloaded_sha256')}\n"
            report += f"   • Hash schema original: {protocol.get('schema_original_sha256')}\n"
            if protocol.get('schema_reloaded_sha256'):
                report += f"   • Hash schema Reloaded: {protocol.get('schema_reloaded_sha256')}\n"
            report += f"   • Seed: {protocol.get('seed')} | repetições: {protocol.get('repeat_count')}\n"
            report += "   • O teste bloqueado não participa na seleção dos modelos.\n\n"

        teacher = self.comparison_results.get('active_oracle_locked_test_metrics')
        if teacher:
            report += "🧠 PROFESSOR ATIVO NO TESTE BLOQUEADO (APENAS RELATO):\n"
            report += f"{'='*50}\n"
            report += f"   • Accuracy: {teacher.get('accuracy', 0):.1%}\n"
            report += f"   • Balanced accuracy: {teacher.get('balanced_accuracy', 0):.1%}\n"
            report += f"   • Macro-F1: {teacher.get('macro_f1', 0):.1%}\n\n"
        
        # Informações do modelo
        model_info = self.comparison_results['model_info']
        report += f"📋 Información del Modelo:\n"
        report += f"   • Tipo MLP: {model_info['mlp_type']}\n"
        report += f"   • Muestras de prueba: {model_info['n_test_samples']}\n"
        report += f"   • Features: {model_info['n_features']}\n"
        report += f"   • Classes: {model_info['n_classes']}\n"
        if model_info['trepan_original_depth']:
            report += f"   • Profundidad Trepan-Original: {model_info['trepan_original_depth']}\n"
        if model_info['trepan_reloaded_depth']:
            report += f"   • Profundidad Trepan-Reloaded: {model_info['trepan_reloaded_depth']}\n"
        
        # Informações sobre balanceamento de classes
        if 'class_balance' in self.comparison_results:
            balance_info = self.comparison_results['class_balance']
            report += f"\n⚖️ ANÁLISIS DE EQUILIBRIO DE CLASES:\n"
            report += f"{'='*50}\n"
            report += f"   • Total de classes: {balance_info['n_classes']}\n"
            report += f"   • Total de muestras: {balance_info['n_samples']}\n\n"
            
            if balance_info.get('imbalanced_classes'):
                report += f"⚠️ CLASES DESBALANCEADAS DETECTADAS (< 5% de los datos):\n"
                for imbalanced in balance_info['imbalanced_classes']:
                    report += f"   • {imbalanced['class']}: {imbalanced['percentage']:.2f}% ({imbalanced['count']} muestras)\n"
                report += f"\n💡 Aviso: Las clases desbalanceadas pueden llevar a métricas engañosas.\n"
                report += f"   Considere usar métricas macro además de weighted para una evaluación más justa.\n\n"
            else:
                report += f"✅ Clases relativamente balanceadas (todas tienen ≥ 5% de los datos)\n\n"
            
            report += f"📊 Distribución de Clases:\n"
            for cls_name, dist in balance_info['class_distribution'].items():
                report += f"   • {cls_name}: {dist['percentage']:.2f}% ({dist['count']} muestras)\n"
        
        report += f"\n"
        
        # Métricas de precisão
        try:
            precision_report = self._format_precision_report()
            report += precision_report if precision_report is not None else ""
            report += self._format_c45_baseline_gate_report()
        except Exception as e:
            report += f"\n⚠️ Error al generar informe de precisión: {str(e)}\n"
        
        # Métricas de fidelidade
        try:
            fidelity_report = self._format_fidelity_report()
            report += fidelity_report if fidelity_report is not None else ""
        except Exception as e:
            report += f"\n⚠️ Error al generar informe de fidelidad: {str(e)}\n"
        
        # Métricas por classe
        try:
            per_class_report = self._format_per_class_report()
            report += per_class_report if per_class_report is not None else ""
        except Exception as e:
            report += f"\n⚠️ Error al generar informe por clase: {str(e)}\n"
        
        # Validação cruzada
        if self.comparison_results.get('cross_validation'):
            try:
                cv_report = self._format_cross_validation_report()
                report += cv_report if cv_report is not None else ""
            except Exception as e:
                report += f"\n⚠️ Error al generar informe de validación cruzada: {str(e)}\n"
        
        # Análise estatística
        try:
            stat_report = self._format_statistical_analysis_report()
            report += stat_report if stat_report is not None else ""
        except Exception as e:
            report += f"\n⚠️ Error al generar análisis estadístico: {str(e)}\n"

        report += self._format_scientific_validation_report()
        
        # Conclusões
        try:
            conclusions_report = self._format_conclusions_report()
            report += conclusions_report if conclusions_report is not None else ""
        except Exception as e:
            report += f"\n⚠️ Error al generar conclusiones: {str(e)}\n"
        
        return report
    
    def _format_precision_report(self):
        
        report = f"🎯 MÉTRICAS PREDICTIVAS OBSERVADAS:\n"
        report += f"{'='*60}\n"
        report += "Na comparação principal, Precisão significa Precision Macro (precision_score(..., average=\"macro\")); Accuracy/Exatidão é mantida apenas como métrica adicional de auditoria.\n\n"
        
        precision_metrics = self.comparison_results['precision']
        
        # MLP Original
        mlp_metrics = precision_metrics['mlp']
        report += f"📊 MLP Original (modelo predictivo de referencia):\n"
        report += f"   • Exactitud: {self._format_metric_with_ci(mlp_metrics.get('accuracy_ci'), 'Exactitud')}\n"
        report += f"   • Precisión: {self._format_metric_with_ci(mlp_metrics.get('precision_macro_ci') or mlp_metrics.get('precision_ci'), 'Precisão Macro')}\n"
        report += f"   • Recall: {self._format_metric_with_ci(mlp_metrics.get('recall_ci'), 'Recall')}\n"
        report += f"   • F1-Score: {self._format_metric_with_ci(mlp_metrics.get('f1_ci'), 'F1-Score')}\n"
        
        # Métricas adicionais
        if mlp_metrics.get('additional_metrics'):
            report += self._format_additional_metrics(mlp_metrics['additional_metrics'], "MLP Original")
        
        report += f"\n"
        
        # Trepan-Original
        if precision_metrics['trepan_original'] is not None:
            trepan_orig_metrics = precision_metrics['trepan_original']
            report += f"🌳 Trepan-Original:\n"
            report += f"   • Exactitud: {self._format_metric_with_ci(trepan_orig_metrics.get('accuracy_ci'), 'Exactitud')}\n"
            report += f"   • Precisión: {self._format_metric_with_ci(trepan_orig_metrics.get('precision_macro_ci') or trepan_orig_metrics.get('precision_ci'), 'Precisão Macro')}\n"
            report += f"   • Recall: {self._format_metric_with_ci(trepan_orig_metrics.get('recall_ci'), 'Recall')}\n"
            report += f"   • F1-Score: {self._format_metric_with_ci(trepan_orig_metrics.get('f1_ci'), 'F1-Score')}\n"
            
            # Métricas adicionais
            if trepan_orig_metrics.get('additional_metrics'):
                report += self._format_additional_metrics(trepan_orig_metrics['additional_metrics'], "Trepan-Original")
            
            report += f"\n"
        else:
            report += f"🌳 Trepan-Original: ❌ No disponible\n\n"
        
        # Trepan-Reloaded
        if precision_metrics['trepan_reloaded'] is not None:
            trepan_reloaded_metrics = precision_metrics['trepan_reloaded']
            report += f"🧠 Trepan-Reloaded:\n"
            report += f"   • Exactitud: {self._format_metric_with_ci(trepan_reloaded_metrics.get('accuracy_ci'), 'Exactitud')}\n"
            report += f"   • Precisión: {self._format_metric_with_ci(trepan_reloaded_metrics.get('precision_macro_ci') or trepan_reloaded_metrics.get('precision_ci'), 'Precisão Macro')}\n"
            report += f"   • Recall: {self._format_metric_with_ci(trepan_reloaded_metrics.get('recall_ci'), 'Recall')}\n"
            report += f"   • F1-Score: {self._format_metric_with_ci(trepan_reloaded_metrics.get('f1_ci'), 'F1-Score')}\n"
            
            # Métricas adicionais
            if trepan_reloaded_metrics.get('additional_metrics'):
                report += self._format_additional_metrics(trepan_reloaded_metrics['additional_metrics'], "Trepan-Reloaded")
            
            report += f"\n"
        else:
            report += f"🌳 Trepan-Reloaded: ❌ No disponible\n\n"
        
        # C4.5 nativo
        if precision_metrics['c45_j48'] is not None:
            c45_metrics = precision_metrics['c45_j48']
            report += f"🌿 C4.5-Nativo (Gain Ratio + poda pessimista):\n"
            report += f"   • Exactitud: {self._format_metric_with_ci(c45_metrics.get('accuracy_ci'), 'Exactitud')}\n"
            report += f"   • Precisión: {self._format_metric_with_ci(c45_metrics.get('precision_macro_ci') or c45_metrics.get('precision_ci'), 'Precisão Macro')}\n"
            report += f"   • Recall: {self._format_metric_with_ci(c45_metrics.get('recall_ci'), 'Recall')}\n"
            report += f"   • F1-Score: {self._format_metric_with_ci(c45_metrics.get('f1_ci'), 'F1-Score')}\n"
            
            # Métricas adicionais
            if c45_metrics.get('additional_metrics'):
                report += self._format_additional_metrics(c45_metrics['additional_metrics'], "C4.5-Nativo")
            
            report += f"\n"
        else:
            report += f"C4.5-Nativo: ❌ No disponible\n\n"
        
        return report
    
    def _format_c45_baseline_gate_report(self):
        gate = self.comparison_results.get('c45_baseline_gate') or {}
        if not gate.get('available'):
            return "\n📏 GATE C4.5: indisponível nesta execução.\n\n"
        report = "\n📏 GATE C4.5 — BASELINE SUPERVISIONADO (NÃO É ORÁCULO):\n"
        report += f"{'='*60}\n"
        for key, label in (("trepan_original", "TREPAN Original"), ("trepan_reloaded", "TREPAN Reloaded")):
            item = gate.get(key) or {}
            if not item:
                report += f"   • {label}: não avaliado.\n"
                continue
            delta = (item.get('deltas') or {}).get('precision_macro')
            status = "ATINGIU" if item.get('accepted') else "NÃO ATINGIU"
            delta_text = "n/a" if delta is None else f"{float(delta) * 100:+.2f} pp"
            report += (
                f"   • {label}: {status} o baseline | "
                f"Δ Precision Macro vs C4.5 = {delta_text}.\n"
            )
        report += "   • Nenhuma métrica é alterada para satisfazer este gate.\n\n"
        return report

    def _format_additional_metrics(self, additional_metrics, model_name):
        
        if not additional_metrics:
            return ""
        
        report = f"\n   📊 Métricas Adicionales ({model_name}):\n"
        
        # AUC-ROC
        if additional_metrics.get('auc_roc') is not None:
            auc_method = additional_metrics.get('auc_roc_method', 'N/A')
            report += f"      • AUC-ROC: {additional_metrics['auc_roc']:.4f} ({auc_method})\n"
            if additional_metrics.get('auc_roc_per_class'):
                auc_per_class = additional_metrics['auc_roc_per_class']
                report += f"        AUC por clase: {', '.join([f'{x:.3f}' if x is not None else 'N/A' for x in auc_per_class])}\n"
        elif additional_metrics.get('auc_roc_note'):
            report += f"      • AUC-ROC: {additional_metrics['auc_roc_note']}\n"
        
        # Log-loss
        if additional_metrics.get('log_loss') is not None:
            report += f"      • Log-Loss: {additional_metrics['log_loss']:.4f}\n"
        elif additional_metrics.get('log_loss_note'):
            report += f"      • Log-Loss: {additional_metrics['log_loss_note']}\n"
        
        # Brier Score
        if additional_metrics.get('brier_score') is not None:
            report += f"      • Brier Score: {additional_metrics['brier_score']:.4f}\n"
            if additional_metrics.get('brier_score_per_class'):
                brier_per_class = additional_metrics['brier_score_per_class']
                report += f"        Brier por clase: {', '.join([f'{x:.3f}' if x is not None else 'N/A' for x in brier_per_class])}\n"
        elif additional_metrics.get('brier_score_note'):
            report += f"      • Brier Score: {additional_metrics['brier_score_note']}\n"
        
        # Cohen's Kappa
        if additional_metrics.get('cohens_kappa') is not None:
            kappa = additional_metrics['cohens_kappa']
            kappa_interpretation = "Excelente" if kappa > 0.75 else "Boa" if kappa > 0.40 else "Débil"
            report += f"      • Cohen's Kappa: {kappa:.4f} ({kappa_interpretation} concordancia)\n"
        
        # Sensibilidade e Especificidade
        if additional_metrics.get('mean_sensitivity') is not None:
            report += f"      • Sensibilidad Media: {additional_metrics['mean_sensitivity']:.4f}\n"
            if additional_metrics.get('sensitivity'):
                sens_per_class = additional_metrics['sensitivity']
                report += f"        Sensibilidad por clase: {', '.join([f'{x:.3f}' for x in sens_per_class])}\n"
        if additional_metrics.get('mean_specificity') is not None:
            report += f"      • Especificidad Media: {additional_metrics['mean_specificity']:.4f}\n"
            if additional_metrics.get('specificity'):
                spec_per_class = additional_metrics['specificity']
                report += f"        Especificidad por clase: {', '.join([f'{x:.3f}' for x in spec_per_class])}\n"
        
        # Matriz de Confusão
        if additional_metrics.get('confusion_matrix') is not None:
            cm = np.array(additional_metrics['confusion_matrix'])
            report += f"      • Matriz de Confusión ({cm.shape[0]}x{cm.shape[1]}):\n"
            # Mostrar apenas matriz pequena (até 5x5) ou resumo
            if cm.shape[0] <= 5:
                report += f"        {str(cm).replace(chr(10), chr(10) + '        ')}\n"
            else:
                report += f"        Matriz {cm.shape[0]}x{cm.shape[1]} (demasiado grande para mostrar)\n"
                report += f"        Diagonal (VP): {np.diag(cm).tolist()}\n"
                report += f"        Suma por fila: {cm.sum(axis=1).tolist()}\n"
        
        return report
    
    def _format_fidelity_report(self):
        
        report = f"🔗 MÉTRICAS DE FIDELIDAD (SEM HIERARQUIA PRÉ-DEFINIDA):\n"
        report += f"{'='*60}\n"
        report += f"Compare apenas fidelidades calculadas contra o mesmo oráculo.\n"
        report += f"Fidelidade = concordância da árvore com o oráculo identificado.\n\n"
        
        fidelity_metrics = self.comparison_results['fidelity']
        
        # Trepan-Original
        if fidelity_metrics['trepan_original'] is not None:
            trepan_orig_fidelity = fidelity_metrics['trepan_original']
            report += f"Trepan-Original vs MLP Original:\n"
            report += f"   • Fidelidad General: {self._format_metric_with_ci(trepan_orig_fidelity.get('overall_fidelity_ci'), 'Fidelidad')}\n"
            report += f"   • Tasa de Concordancia: {trepan_orig_fidelity['agreement_rate']:.4f}\n"
            report += f"   • F1-Fidelidad: {trepan_orig_fidelity['f1_fidelity']:.4f}\n"
            
            # Análises granulares (incluindo análise de discrepâncias melhorada)
            report += self._format_granular_fidelity_analysis(trepan_orig_fidelity, "Trepan-Original")
            
            report += f"\n"
        else:
            report += f"🌳 Trepan-Original vs MLP: ❌ No disponible\n\n"
        
        # Trepan-Reloaded
        if fidelity_metrics['trepan_reloaded'] is not None:
            trepan_reloaded_fidelity = fidelity_metrics['trepan_reloaded']
            ref = trepan_reloaded_fidelity.get('fidelity_reference', 'oráculo ativo')
            report += f"Trepan-Reloaded vs {ref}:\n"
            report += f"   • Fidelidad General: {self._format_metric_with_ci(trepan_reloaded_fidelity.get('overall_fidelity_ci'), 'Fidelidad')}\n"
            report += f"   • Tasa de Concordancia: {trepan_reloaded_fidelity['agreement_rate']:.4f}\n"
            report += f"   • F1-Fidelidad: {trepan_reloaded_fidelity['f1_fidelity']:.4f}\n"
            
            # Análises granulares (incluindo análise de discrepâncias melhorada)
            report += self._format_granular_fidelity_analysis(trepan_reloaded_fidelity, "Trepan-Reloaded")
            
            report += f"\n"
        else:
            report += f"🌳 Trepan-Reloaded vs MLP: ❌ No disponible\n\n"
        
        return report
    
    def _format_granular_fidelity_analysis(self, fidelity_data, model_name):
        
        report = ""
        
        # 1. Análise por região de confiança do MLP
        if fidelity_data.get('by_confidence_region') and 'error' not in fidelity_data['by_confidence_region']:
            conf_region = fidelity_data['by_confidence_region']
            report += f"\n   📊 Fidelidad por Región de Confianza del MLP:\n"
            if conf_region.get('high_confidence'):
                hc = conf_region['high_confidence']
                report += f"      • Alta Confianza (≥0.7): {hc.get('fidelity', 0):.4f} ({hc.get('n_samples', 0)} muestras)\n"
            if conf_region.get('low_confidence'):
                lc = conf_region['low_confidence']
                report += f"      • Baja Confianza (<0.7): {lc.get('fidelity', 0):.4f} ({lc.get('n_samples', 0)} muestras)\n"
            if conf_region.get('mean_confidence'):
                report += f"      • Confianza Media del MLP: {conf_region['mean_confidence']:.4f}\n"
        
        # 2. Análise por tipo de erro
        if fidelity_data.get('by_error_type') and 'error' not in fidelity_data['by_error_type']:
            error_type = fidelity_data['by_error_type']
            report += f"\n   🔍 Fidelidad por Tipo de Error:\n"
            if error_type.get('both_correct'):
                report += f"      • Ambos acertaron: {error_type['both_correct']['count']} ({error_type['both_correct']['percentage']:.1f}%)\n"
            if error_type.get('both_wrong'):
                report += f"      • Ambos fallaron: {error_type['both_wrong']['count']} ({error_type['both_wrong']['percentage']:.1f}%)\n"
            if error_type.get('mlp_correct_tree_wrong'):
                mlp_ok = error_type['mlp_correct_tree_wrong']
                report += f"      • MLP Correcto, Árbol Incorrecto: {mlp_ok['count']} ({mlp_ok['percentage']:.1f}%)\n"
            if error_type.get('mlp_wrong_tree_correct'):
                tree_ok = error_type['mlp_wrong_tree_correct']
                report += f"      • MLP Incorrecto, Árbol Correcto: {tree_ok['count']} ({tree_ok['percentage']:.1f}%)\n"
        
        # 3. Análise por região do espaço de features
        if fidelity_data.get('by_feature_region') and 'error' not in fidelity_data['by_feature_region']:
            feat_region = fidelity_data['by_feature_region']
            report += f"\n   🗺️  Fidelidad por Región del Espacio de Features:\n"
            if isinstance(feat_region, list):
                for region_info in feat_region[:4]:  # Mostrar até 4 regiões
                    if isinstance(region_info, dict) and region_info.get('fidelity') is not None:
                        report += f"      • {region_info.get('name', 'Región')}: {region_info['fidelity']:.4f} ({region_info.get('n_samples', 0)} muestras)\n"
        
        # 4. Análise de discrepâncias melhorada (com características, padrões e importância de features)
        if fidelity_data.get('discrepancies'):
            disc_analysis = fidelity_data['discrepancies']
            report += f"\n   ⚠️ Análisis de Discrepancias Detalladas:\n"
            report += f"      • Total de Discrepancias: {disc_analysis.get('total', 0)} ({disc_analysis.get('percentage', 0):.2f}%)\n"
            if disc_analysis.get('mean_mlp_confidence') is not None:
                report += f"      • Confianza Media del MLP en Discrepancias: {disc_analysis['mean_mlp_confidence']:.4f}\n"
            
            # Características dos casos de discordância
            if disc_analysis.get('characteristics'):
                char = disc_analysis['characteristics']
                report += f"\n      📊 Características de los Casos de Discordancia:\n"
                if char.get('mean_differences'):
                    report += f"         • Features con Mayor Diferencia Media:\n"
                    for feat_name, diff in list(char['mean_differences'].items())[:3]:
                        report += f"           - {feat_name}: {diff:.4f}\n"
                if char.get('outlier_features'):
                    report += f"         • Features con Más Outliers:\n"
                    for feat_name, n_outliers in list(char['outlier_features'].items())[:3]:
                        report += f"           - {feat_name}: {n_outliers} outliers\n"
                if char.get('distance_from_center'):
                    dist_info = char['distance_from_center']
                    report += f"         • Distancia del Centro: Discordantes={dist_info.get('mean_discrepant', 0):.3f}, Concordantes={dist_info.get('mean_agreement', 0):.3f}\n"
                if char.get('mlp_confidence_stats'):
                    conf_stats = char['mlp_confidence_stats']
                    report += f"         • Confianza MLP: media={conf_stats.get('mean', 0):.3f}, mediana={conf_stats.get('median', 0):.3f}\n"
            
            # Padrões comuns
            if disc_analysis.get('common_patterns'):
                patterns = disc_analysis['common_patterns']
                report += f"\n      🔍 Patrones Comunes en las Discrepancias:\n"
                if patterns.get('class_transitions'):
                    report += f"         • Transiciones de Clase Más Frecuentes:\n"
                    for trans, count in list(patterns['class_transitions'].items())[:5]:
                        report += f"           - {trans}: {count} ocurrencias\n"
                if patterns.get('problematic_true_classes'):
                    report += f"         • Clases Verdaderas que Más Causan Discordancias:\n"
                    for cls, count in list(patterns['problematic_true_classes'].items())[:3]:
                        report += f"           - Classe {cls}: {count} casos\n"
                if patterns.get('accuracy_in_disagreement'):
                    acc_info = patterns['accuracy_in_disagreement']
                    report += f"         • Precisión en Discrepancias: MLP={acc_info.get('mlp_correct_pct', 0):.1f}%, Árbol={acc_info.get('tree_correct_pct', 0):.1f}%\n"
            
            # Importância de features
            if disc_analysis.get('feature_importance_in_disagreements'):
                feat_imp = disc_analysis['feature_importance_in_disagreements']
                report += f"\n      🎯 Importancia de Features en los Casos Problemáticos:\n"
                if feat_imp.get('tree_feature_importance'):
                    report += f"         • Top Features por el Árbol:\n"
                    for feat_name, importance in list(feat_imp['tree_feature_importance'].items())[:5]:
                        report += f"           - {feat_name}: {importance:.4f}\n"
                if feat_imp.get('discriminative_features'):
                    report += f"         • Features Más Discriminativas (F-statistic):\n"
                    for feat_name, f_score in list(feat_imp['discriminative_features'].items())[:5]:
                        report += f"           - {feat_name}: {f_score:.4f}\n"
                if feat_imp.get('correlation_with_disagreement'):
                    report += f"         • Features Correlacionadas con Discordancias:\n"
                    for feat_name, corr in list(feat_imp['correlation_with_disagreement'].items())[:5]:
                        report += f"           - {feat_name}: {corr:+.4f}\n"
            
            # Exemplos
            if disc_analysis.get('examples') and len(disc_analysis['examples']) > 0:
                report += f"\n      📝 Ejemplos de Discrepancias (muestra de {len(disc_analysis['examples'])}):\n"
                for i, ex in enumerate(disc_analysis['examples'][:3], 1):  # Mostrar apenas 3 exemplos
                    report += f"         {i}. Muestra #{ex.get('sample_index', 'N/A')}: MLP={ex.get('mlp_prediction')} vs Árbol={ex.get('tree_prediction')}"
                    if ex.get('mlp_confidence') is not None:
                        report += f" (confianza MLP: {ex['mlp_confidence']:.3f})"
                    if ex.get('key_features'):
                        key_feats = list(ex['key_features'].items())[:2]  # Top 2 features
                        feat_str = ", ".join([f"{k}={v:.2f}" for k, v in key_feats])
                        report += f"\n           Características principales: {feat_str}"
                    report += f"\n"
        
        return report
    
    def _format_per_class_report(self):
        
        report = f"📊 MÉTRICAS POR CLASE:\n"
        report += f"{'='*50}\n\n"
        
        per_class_metrics = self.comparison_results['per_class']
        
        # Obter todas as classes
        all_classes = set()
        for model_metrics in per_class_metrics.values():
            if model_metrics is not None:
                all_classes.update(model_metrics.keys())
        
        for class_name in sorted(all_classes):
            report += f"🏷️ Clase '{class_name}':\n"
            
            # MLP
            if per_class_metrics['mlp'] and class_name in per_class_metrics['mlp']:
                mlp_class = per_class_metrics['mlp'][class_name]
                report += f"   🤖 MLP: P={mlp_class['precision']:.3f}, R={mlp_class['recall']:.3f}, F1={mlp_class['f1']:.3f}\n"
            
            # Trepan-Original
            if (per_class_metrics['trepan_original'] and 
                class_name in per_class_metrics['trepan_original']):
                trepan_orig_class = per_class_metrics['trepan_original'][class_name]
                report += f"   🌳 T-Orig: P={trepan_orig_class['precision']:.3f}, R={trepan_orig_class['recall']:.3f}, F1={trepan_orig_class['f1']:.3f}\n"
            
            # Trepan-Reloaded
            if (per_class_metrics['trepan_reloaded'] and 
                class_name in per_class_metrics['trepan_reloaded']):
                trepan_reloaded_class = per_class_metrics['trepan_reloaded'][class_name]
                report += f"   🧠 T-Reload: P={trepan_reloaded_class['precision']:.3f}, R={trepan_reloaded_class['recall']:.3f}, F1={trepan_reloaded_class['f1']:.3f}\n"
            
            report += f"\n"
        
        return report
    
    def _format_cross_validation_report(self):
        
        report = f"🔄 VALIDACIÓN CRUZADA (5-Fold) con Intervalos de Confianza:\n"
        report += f"{'='*50}\n\n"
        report += f"💡 La validación cruzada proporciona estimaciones de variabilidad usando diferentes folds.\n\n"
        
        cv_metrics = self.comparison_results['cross_validation']
        
        # MLP
        if 'mlp' in cv_metrics and 'error' not in cv_metrics['mlp']:
            mlp_cv = cv_metrics['mlp']
            mean_acc = mlp_cv['mean_accuracy']
            std_acc = mlp_cv['std_accuracy']
            # IC95% usando distribuição t (para pequenas amostras)
            # Para 5 folds: df = 5-1 = 4
            df = 5 - 1  # graus de liberdade
            t_critical = t.ppf(0.975, df)  # IC95% (0.975 para cauda superior)
            ci_lower = mean_acc - t_critical * std_acc / np.sqrt(5)
            ci_upper = mean_acc + t_critical * std_acc / np.sqrt(5)
            # Garantir que IC está no intervalo válido [0, 1]
            ci_lower = max(0.0, min(1.0, ci_lower))
            ci_upper = max(0.0, min(1.0, ci_upper))
            report += f"🤖 MLP: {mean_acc:.4f} ± {std_acc:.4f} (IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}])\n"
        
        # Trepan-Original
        if ('trepan_original' in cv_metrics and cv_metrics['trepan_original'] is not None and 
            'error' not in cv_metrics['trepan_original']):
            trepan_orig_cv = cv_metrics['trepan_original']
            mean_acc = trepan_orig_cv['mean_accuracy']
            std_acc = trepan_orig_cv['std_accuracy']
            df = 5 - 1
            t_critical = t.ppf(0.975, df)
            ci_lower = mean_acc - t_critical * std_acc / np.sqrt(5)
            ci_upper = mean_acc + t_critical * std_acc / np.sqrt(5)
            ci_lower = max(0.0, min(1.0, ci_lower))
            ci_upper = max(0.0, min(1.0, ci_upper))
            report += f"🌳 Trepan-Original: {mean_acc:.4f} ± {std_acc:.4f} (IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}])\n"
        
        # Trepan-Reloaded
        if ('trepan_reloaded' in cv_metrics and cv_metrics['trepan_reloaded'] is not None and 
            'error' not in cv_metrics['trepan_reloaded']):
            trepan_reloaded_cv = cv_metrics['trepan_reloaded']
            mean_acc = trepan_reloaded_cv['mean_accuracy']
            std_acc = trepan_reloaded_cv['std_accuracy']
            df = 5 - 1
            t_critical = t.ppf(0.975, df)
            ci_lower = mean_acc - t_critical * std_acc / np.sqrt(5)
            ci_upper = mean_acc + t_critical * std_acc / np.sqrt(5)
            ci_lower = max(0.0, min(1.0, ci_lower))
            ci_upper = max(0.0, min(1.0, ci_upper))
            report += f"🧠 Trepan-Reloaded: {mean_acc:.4f} ± {std_acc:.4f} (IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}])\n"
        
        report += f"\n"
        return report
    
    def _format_statistical_analysis_report(self):
        
        report = f"📈 ANÁLISIS ESTADÍSTICO CON PRUEBAS DE SIGNIFICANCIA:\n"
        report += f"{'='*60}\n\n"
        report += f"💡 Pruebas realizadas:\n"
        report += f"   • Prueba de McNemar: Compara dos clasificadores en datos pareados\n"
        report += f"   • Prueba t pareada: Compara métricas continuas entre modelos\n"
        report += f"   • Wilcoxon signed-rank: Prueba no paramétrica para distribuciones no normales\n\n"
        report += f"📊 Interpretación de los p-values:\n"
        report += f"   • p < 0.001: *** Altamente significativo\n"
        report += f"   • p < 0.01: ** Muy significativo\n"
        report += f"   • p < 0.05: * Significativo\n"
        report += f"   • p < 0.10: Marginalmente significativo\n"
        report += f"   • p >= 0.05: No significativo\n\n"
        report += f"{'='*60}\n\n"
        
        statistical_analysis = self.comparison_results['statistical_analysis']
        
        # Trepan-Original vs MLP
        if 'trepan_original_vs_mlp' in statistical_analysis:
            analysis = statistical_analysis['trepan_original_vs_mlp']
            report += f"🌳 Trepan-Original vs MLP:\n"
            report += f"{'─'*60}\n"
            report += f"📊 Métricas Básicas:\n"
            report += f"   • Diferencia de Exactitud: {analysis['accuracy_difference']:+.4f}\n"
            report += f"   • Diferencia Relativa: {analysis['accuracy_relative_difference']:+.2f}%\n"
            if analysis['fidelity'] is not None:
                report += f"   • Fidelidad: {analysis['fidelity']:.4f} ({analysis['fidelity']*100:.2f}%)\n"
            
            # Testes estatísticos
            report += f"\n📊 Pruebas de Significancia Estadística:\n"
            
            # McNemar
            if 'mcnemar_test' in analysis and analysis['mcnemar_test']:
                mcnemar = analysis['mcnemar_test']
                if 'error' not in mcnemar and mcnemar.get('p_value') is not None:
                    sig_symbol = "***" if mcnemar['significant'] else ""
                    report += f"   • McNemar Test: p-value = {mcnemar['p_value']:.6f} {sig_symbol}\n"
                    report += f"     → {mcnemar.get('interpretation', 'N/A')}\n"
                    if 'contingency_table' in mcnemar:
                        ct = mcnemar['contingency_table']
                        report += f"     Tabla de contingencia: [[{ct[0][0]}, {ct[0][1]}], [{ct[1][0]}, {ct[1][1]}]]\n"
                elif 'error' in mcnemar:
                    report += f"   • McNemar Test: ❌ Error - {mcnemar['error']}\n"
            
            # T-test pareado
            if 'paired_ttest' in analysis and analysis['paired_ttest']:
                ttest = analysis['paired_ttest']
                if 'error' not in ttest and ttest.get('p_value') is not None:
                    sig_symbol = "***" if ttest['significant'] else ""
                    report += f"   • T-test Pareado: p-value = {ttest['p_value']:.6f} {sig_symbol}\n"
                    report += f"     → {ttest.get('interpretation', 'N/A')}\n"
                    if 'mean_difference' in ttest:
                        report += f"     Diferencia media: {ttest['mean_difference']:+.6f}\n"
                elif 'error' in ttest:
                    report += f"   • T-test Pareado: ❌ Error - {ttest['error']}\n"
            
            # Wilcoxon
            if 'wilcoxon_test' in analysis and analysis['wilcoxon_test']:
                wilcoxon = analysis['wilcoxon_test']
                if 'error' not in wilcoxon and wilcoxon.get('p_value') is not None:
                    sig_symbol = "***" if wilcoxon['significant'] else ""
                    report += f"   • Wilcoxon Signed-Rank: p-value = {wilcoxon['p_value']:.6f} {sig_symbol}\n"
                    report += f"     → {wilcoxon.get('interpretation', 'N/A')}\n"
                    if 'mean_difference' in wilcoxon:
                        report += f"     Diferencia media: {wilcoxon['mean_difference']:+.6f}\n"
                elif 'error' in wilcoxon:
                    report += f"   • Wilcoxon Signed-Rank: ❌ Error - {wilcoxon['error']}\n"
            
            report += f"\n"
        
        # Trepan-Reloaded vs MLP
        if 'trepan_reloaded_vs_mlp' in statistical_analysis:
            analysis = statistical_analysis['trepan_reloaded_vs_mlp']
            report += f"🧠 Trepan-Reloaded vs MLP:\n"
            report += f"{'─'*60}\n"
            report += f"📊 Métricas Básicas:\n"
            report += f"   • Diferencia de Exactitud: {analysis['accuracy_difference']:+.4f}\n"
            report += f"   • Diferencia Relativa: {analysis['accuracy_relative_difference']:+.2f}%\n"
            if analysis['fidelity'] is not None:
                report += f"   • Fidelidad: {analysis['fidelity']:.4f} ({analysis['fidelity']*100:.2f}%)\n"
            
            # Testes estatísticos
            report += f"\n📊 Pruebas de Significancia Estadística:\n"
            
            # McNemar
            if 'mcnemar_test' in analysis and analysis['mcnemar_test']:
                mcnemar = analysis['mcnemar_test']
                if 'error' not in mcnemar and mcnemar.get('p_value') is not None:
                    sig_symbol = "***" if mcnemar['significant'] else ""
                    report += f"   • McNemar Test: p-value = {mcnemar['p_value']:.6f} {sig_symbol}\n"
                    report += f"     → {mcnemar.get('interpretation', 'N/A')}\n"
                    if 'contingency_table' in mcnemar:
                        ct = mcnemar['contingency_table']
                        report += f"     Tabla de contingencia: [[{ct[0][0]}, {ct[0][1]}], [{ct[1][0]}, {ct[1][1]}]]\n"
                elif 'error' in mcnemar:
                    report += f"   • McNemar Test: ❌ Error - {mcnemar['error']}\n"
            
            # T-test pareado
            if 'paired_ttest' in analysis and analysis['paired_ttest']:
                ttest = analysis['paired_ttest']
                if 'error' not in ttest and ttest.get('p_value') is not None:
                    sig_symbol = "***" if ttest['significant'] else ""
                    report += f"   • T-test Pareado: p-value = {ttest['p_value']:.6f} {sig_symbol}\n"
                    report += f"     → {ttest.get('interpretation', 'N/A')}\n"
                    if 'mean_difference' in ttest:
                        report += f"     Diferencia media: {ttest['mean_difference']:+.6f}\n"
                elif 'error' in ttest:
                    report += f"   • T-test Pareado: ❌ Error - {ttest['error']}\n"
            
            # Wilcoxon
            if 'wilcoxon_test' in analysis and analysis['wilcoxon_test']:
                wilcoxon = analysis['wilcoxon_test']
                if 'error' not in wilcoxon and wilcoxon.get('p_value') is not None:
                    sig_symbol = "***" if wilcoxon['significant'] else ""
                    report += f"   • Wilcoxon Signed-Rank: p-value = {wilcoxon['p_value']:.6f} {sig_symbol}\n"
                    report += f"     → {wilcoxon.get('interpretation', 'N/A')}\n"
                    if 'mean_difference' in wilcoxon:
                        report += f"     Diferencia media: {wilcoxon['mean_difference']:+.6f}\n"
                elif 'error' in wilcoxon:
                    report += f"   • Wilcoxon Signed-Rank: ❌ Error - {wilcoxon['error']}\n"
            
            report += f"\n"
        
        # Trepan-Original vs Trepan-Reloaded
        if 'trepan_original_vs_reloaded' in statistical_analysis:
            analysis = statistical_analysis['trepan_original_vs_reloaded']
            report += f"🌳 vs 🧠 Trepan-Original vs Trepan-Reloaded:\n"
            report += f"{'─'*60}\n"
            report += f"📊 Métricas Básicas:\n"
            report += f"   • Diferencia de Exactitud: {analysis['accuracy_difference']:+.4f}\n"
            report += f"   • Diferencia Relativa: {analysis['accuracy_relative_difference']:+.2f}%\n"
            if analysis['fidelity_difference'] is not None:
                report += f"   • Diferencia de Fidelidad: {analysis['fidelity_difference']:+.4f}\n"
            
            # Testes estatísticos
            report += f"\n📊 Pruebas de Significancia Estadística:\n"
            
            # McNemar
            if 'mcnemar_test' in analysis and analysis['mcnemar_test']:
                mcnemar = analysis['mcnemar_test']
                if 'error' not in mcnemar and mcnemar.get('p_value') is not None:
                    sig_symbol = "***" if mcnemar['significant'] else ""
                    report += f"   • McNemar Test: p-value = {mcnemar['p_value']:.6f} {sig_symbol}\n"
                    report += f"     → {mcnemar.get('interpretation', 'N/A')}\n"
                    if 'contingency_table' in mcnemar:
                        ct = mcnemar['contingency_table']
                        report += f"     Tabla de contingencia: [[{ct[0][0]}, {ct[0][1]}], [{ct[1][0]}, {ct[1][1]}]]\n"
                elif 'error' in mcnemar:
                    report += f"   • McNemar Test: ❌ Error - {mcnemar['error']}\n"
            
            # T-test pareado
            if 'paired_ttest' in analysis and analysis['paired_ttest']:
                ttest = analysis['paired_ttest']
                if 'error' not in ttest and ttest.get('p_value') is not None:
                    sig_symbol = "***" if ttest['significant'] else ""
                    report += f"   • T-test Pareado: p-value = {ttest['p_value']:.6f} {sig_symbol}\n"
                    report += f"     → {ttest.get('interpretation', 'N/A')}\n"
                    if 'mean_difference' in ttest:
                        report += f"     Diferencia media: {ttest['mean_difference']:+.6f}\n"
                elif 'error' in ttest:
                    report += f"   • T-test Pareado: ❌ Error - {ttest['error']}\n"
            
            # Wilcoxon
            if 'wilcoxon_test' in analysis and analysis['wilcoxon_test']:
                wilcoxon = analysis['wilcoxon_test']
                if 'error' not in wilcoxon and wilcoxon.get('p_value') is not None:
                    sig_symbol = "***" if wilcoxon['significant'] else ""
                    report += f"   • Wilcoxon Signed-Rank: p-value = {wilcoxon['p_value']:.6f} {sig_symbol}\n"
                    report += f"     → {wilcoxon.get('interpretation', 'N/A')}\n"
                    if 'mean_difference' in wilcoxon:
                        report += f"     Diferencia media: {wilcoxon['mean_difference']:+.6f}\n"
                elif 'error' in wilcoxon:
                    report += f"   • Wilcoxon Signed-Rank: ❌ Error - {wilcoxon['error']}\n"
            
            report += f"\n"
        
        return report
    
    def _format_scientific_validation_report(self):
        validation = self.comparison_results.get('scientific_validation') or {}
        if validation.get('status') != 'ok':
            return "\n🔬 VALIDACIÓN CIENTÍFICA: no disponible.\n"
        report = "\n🔬 VALIDACIÓN CIENTÍFICA PAREADA:\n"
        report += f"{'='*50}\n"
        report += (
            "   Protocolo: mismas muestras, mismo MLP Original y diferencias "
            "bootstrap pareadas.\n"
        )
        for baseline, item in validation.get('comparisons', {}).items():
            pred = item['predictive_accuracy_difference']
            fid = item['same_oracle_fidelity_difference']
            mc = item['mcnemar']
            report += (
                f"   • Reloaded vs {baseline}: veredicto={item['verdict']}; "
                f"Δaccuracy={pred['difference']:+.3f} "
                f"IC95%[{pred['ci_lower']:+.3f},{pred['ci_upper']:+.3f}]; "
                f"Δfidelidad={fid['difference']:+.3f} "
                f"IC95%[{fid['ci_lower']:+.3f},{fid['ci_upper']:+.3f}]; "
                f"McNemar p={mc['p_value']:.4f}.\n"
            )
        if validation.get('claim_guard') == 'superiority_supported':
            report += "   ✅ La evidencia de este holdout respalda la alegación fuerte.\n"
        else:
            report += (
                "   ⚠️ No afirmar superioridad fuerte con esta ejecución; "
                "el resultado es no-inferior o inconcluso.\n"
            )
        return report + "\n"

    def _format_conclusions_report(self):
        validation = self.comparison_results.get('scientific_validation') or {}
        precision = self.comparison_results.get('precision') or {}
        fidelity = self.comparison_results.get('fidelity') or {}
        report = "🎯 CONCLUSIONES BASADAS EN EVIDENCIA:\n"
        report += f"{'='*50}\n"
        rel = precision.get('trepan_reloaded') or {}
        report += (
            f"   • Trepan-Reloaded: accuracy={rel.get('accuracy', 0):.1%}, "
            f"balanced accuracy={rel.get('balanced_accuracy', 0):.1%}, "
            f"macro-F1={rel.get('f1_macro', 0):.1%}.\n"
        )
        rel_fid = fidelity.get('trepan_reloaded') or {}
        report += (
            f"   • Fidelidad al oráculo activo={rel_fid.get('active_oracle_fidelity', 0):.1%}; "
            f"fidelidad controlada al MLP Original="
            f"{rel_fid.get('fidelity_to_mlp_original', 0):.1%}.\n"
        )
        if validation.get('claim_guard') == 'superiority_supported':
            report += (
                "   ✅ Esta partición respalda superioridad pareada frente a los "
                "baselines evaluados; confirmar con repeticiones y validación externa.\n"
            )
        else:
            report += (
                "   ⚠️ Superioridad fuerte no demostrada. Reportar el veredicto "
                "pareado y ampliar la evaluación; no seleccionar otro seed por resultado.\n"
            )
        report += (
            "   • Para alegaciones clínicas: usar split por paciente/centro, "
            "calibración, sensibilidade/especificidade, AUROC/AUPRC, análise de "
            "subgrupos e validação externa.\n\n"
        )
        return report

        # Implementacao legada mantida abaixo apenas para compatibilidade de diff;
        # o retorno acima impede conclusoes com ranking predefinido.
        report = f"🎯 CONCLUSIONES Y RECOMENDACIONES:\n"
        report += f"{'='*50}\n\n"
        
        precision_metrics = self.comparison_results['precision']
        fidelity_metrics = self.comparison_results['fidelity']
        statistical_analysis = self.comparison_results['statistical_analysis']
        
        # Análise de fidelidade
        if fidelity_metrics['trepan_original'] is not None:
            trepan_orig_fidelity = fidelity_metrics['trepan_original']['overall_fidelity']
            if trepan_orig_fidelity > 0.9:
                report += f"✅ Trepan-Original: Excelente fidelidad ({trepan_orig_fidelity:.1%})\n"
            elif trepan_orig_fidelity > 0.8:
                report += f"👍 Trepan-Original: Buena fidelidad ({trepan_orig_fidelity:.1%})\n"
            else:
                report += f"⚠️ Trepan-Original: Fidelidad moderada ({trepan_orig_fidelity:.1%})\n"
        
        if fidelity_metrics['trepan_reloaded'] is not None:
            trepan_reloaded_fidelity = fidelity_metrics['trepan_reloaded']['overall_fidelity']
            if trepan_reloaded_fidelity > 0.9:
                report += f"✅ Trepan-Reloaded: Excelente fidelidad ({trepan_reloaded_fidelity:.1%})\n"
            elif trepan_reloaded_fidelity > 0.8:
                report += f"👍 Trepan-Reloaded: Buena fidelidad ({trepan_reloaded_fidelity:.1%})\n"
            else:
                report += f"⚠️ Trepan-Reloaded: Fidelidad moderada ({trepan_reloaded_fidelity:.1%})\n"

        if (
            fidelity_metrics.get('trepan_original') is not None
            and fidelity_metrics.get('trepan_reloaded') is not None
        ):
            fid_orig = fidelity_metrics['trepan_original']['overall_fidelity']
            fid_reloaded = fidelity_metrics['trepan_reloaded']['overall_fidelity']
            fid_delta = fid_reloaded - fid_orig
            if fid_delta > 0.01:
                report += (
                    f"🏆 Ventaja de fidelidad (overall_fidelity): Trepan-Reloaded "
                    f"+{fid_delta:.1%} vs Trepan-Original ({fid_reloaded:.1%} vs {fid_orig:.1%})\n"
                )
                if fidelity_metrics.get('c45_j48') is not None:
                    fid_c45 = fidelity_metrics['c45_j48']['overall_fidelity']
                    if fid_reloaded > fid_c45 + 0.01:
                        report += (
                            f"🏆 Trepan-Reloaded también supera C4.5-Nativo en fidelidad "
                            f"({fid_reloaded:.1%} vs {fid_c45:.1%})\n"
                        )
            elif fid_delta < -0.01:
                report += (
                    f"ℹ️ Trepan-Original mantiene fidelidad superior en esta ejecución "
                    f"({fid_orig:.1%} vs {fid_reloaded:.1%}). Com ontologia OWL carregada, "
                    f"o modo Reloaded Full tende a recuperar vantagem.\n"
                )

        # Comparação entre algoritmos
        if 'trepan_original_vs_reloaded' in statistical_analysis:
            analysis = statistical_analysis['trepan_original_vs_reloaded']
            if analysis['accuracy_difference'] > 0.01:
                report += f"🏆 Trepan-Reloaded teve maior accuracy nesta execução (sem alegação de superioridade)\n"
            elif analysis['accuracy_difference'] < -0.01:
                report += f"🏆 Trepan-Original teve maior accuracy nesta execução (sem alegação de superioridade)\n"
            else:
                report += f"🤝 Trepan-Original e Trepan-Reloaded tiveram accuracy semelhante nesta execução\n"
            if analysis.get('fidelity_difference') is not None:
                fd = analysis['fidelity_difference']
                if fd > 0.01:
                    report += (
                        f"🏆 Trepan-Reloaded teve maior overall_fidelity nesta execução "
                        f"(Δ {fd:+.1%})\n"
                    )
                elif fd < -0.01:
                    report += (
                        f"ℹ️ Trepan-Original con mayor overall_fidelity en esta ejecución (Δ {fd:+.1%})\n"
                    )
        
        # Recomendações
        report += f"\n💡 Recomendaciones:\n"
        
        report += (
            "   • Não é produzido ranking de fidelidade quando os oráculos diferem.\n"
            "   • Use fidelity_to_mlp_original para a comparação controlada.\n"
        )

        report += f"   • Selecione o modelo apenas após validação pareada e análise de complexidade\n"
        report += f"   • Monitoree el rendimiento en datos de producción\n"
        report += f"   • Valide regularmente la fidelidad de los árboles\n"
        
        report += f"\n{'='*80}\n"
        return report
    
    def create_comparison_visualization(self, save_path=None):
        
        if not self.comparison_results:
            return "❌ Nenhuma comparação foi realizada ainda."
        
        try:
            # Configurar figura com subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Comparação de Precisão Macro e Fidelidade\nTrepan-Original vs C4.5-Nativo vs Trepan-Reloaded', 
                        fontsize=16, fontweight='bold')
            
            # 1. Gráfico de barras - Acurácia
            self._plot_accuracy_comparison(axes[0, 0])
            
            # 2. Gráfico de barras - Fidelidade
            self._plot_fidelity_comparison(axes[0, 1])
            
            # 3. Gráfico de radar - Métricas múltiplas
            self._plot_radar_comparison(axes[1, 0])
            
            # 4. Gráfico de diferenças
            self._plot_differences_comparison(axes[1, 1])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                return f"Visualização salva em: {save_path}"
            else:
                plt.show()
                return "Visualização exibida com sucesso!"
                
        except Exception as e:
            return f"❌ Erro ao criar visualização: {str(e)}"
    
    def _plot_accuracy_comparison(self, ax):
        
        precision_metrics = self.comparison_results['precision']
        
        # Apenas os três algoritmos explicativos solicitados.
        models = []
        accuracies = []
        colors = []

        if precision_metrics['trepan_original'] is not None:
            models.append('Trepan-Original')
            accuracies.append(precision_metrics['trepan_original'].get('precision_macro', precision_metrics['trepan_original']['precision']))
            colors.append('#808080')  # Cinza

        if precision_metrics.get('c45_j48') is not None:
            models.append('C4.5-Nativo')
            accuracies.append(precision_metrics['c45_j48'].get('precision_macro', precision_metrics['c45_j48']['precision']))
            colors.append('#FFFFFF')  # Branco

        if precision_metrics['trepan_reloaded'] is not None:
            models.append('Trepan-Reloaded')
            accuracies.append(precision_metrics['trepan_reloaded'].get('precision_macro', precision_metrics['trepan_reloaded']['precision']))
            colors.append('#FFFF00')  # Amarelo
        
        bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_title('Precisão Macro dos Modelos', fontweight='bold')
        ax.set_ylabel('Precisão Macro')
        ax.set_ylim(0, 1)
        ax.set_facecolor('#F5F5F5')  # Fundo cinza claro para melhor visualização
        
        # Adicionar valores nas barras
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', color='black')
    
    def _plot_fidelity_comparison(self, ax):
        
        fidelity_metrics = self.comparison_results['fidelity']
        
        models = []
        fidelities = []
        colors = []
        
        # Todos os três usam a mesma referência de fidelidade (MLP Original)
        # na comparação controlada.
        if fidelity_metrics['trepan_original'] is not None:
            models.append('Trepan-Original')
            fidelities.append(fidelity_metrics['trepan_original'].get('fidelity_to_mlp_original', fidelity_metrics['trepan_original']['overall_fidelity']))
            colors.append('#808080')  # Cinza

        if fidelity_metrics.get('c45_j48') is not None:
            models.append('C4.5-Nativo')
            fidelities.append(fidelity_metrics['c45_j48'].get('fidelity_to_mlp_original', fidelity_metrics['c45_j48']['overall_fidelity']))
            colors.append('#FFFFFF')
        
        if fidelity_metrics['trepan_reloaded'] is not None:
            models.append('Trepan-Reloaded')
            fidelities.append(fidelity_metrics['trepan_reloaded'].get('fidelity_to_mlp_original', fidelity_metrics['trepan_reloaded']['overall_fidelity']))
            colors.append('#FFFF00')  # Amarelo
        
        if models:
            bars = ax.bar(models, fidelities, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_title('Fidelidade de controlo ao MLP Original', fontweight='bold')
            ax.set_ylabel('Fidelidade ao MLP Original')
            ax.set_ylim(0, 1)
            ax.set_facecolor('#F5F5F5')  # Fundo cinza claro para melhor visualização
            
            # Adicionar valores nas barras
            for bar, fid in zip(bars, fidelities):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{fid:.3f}', ha='center', va='bottom', fontweight='bold', color='black')
        else:
            ax.text(0.5, 0.5, 'Nenhuma árvore disponível', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Fidelidade de controlo ao MLP Original', fontweight='bold')
    
    def _plot_radar_comparison(self, ax):
        
        precision_metrics = self.comparison_results['precision']
        
        # Métricas para radar
        metrics = ['Precisão Macro', 'Recall Macro', 'Macro-F1', 'Balanced Accuracy']
        
        # Trepan-Original
        trepan_orig_values = None
        if precision_metrics['trepan_original'] is not None:
            trepan_orig_values = [
                precision_metrics['trepan_original'].get('precision_macro', precision_metrics['trepan_original']['precision']),
                precision_metrics['trepan_original'].get('recall_macro', precision_metrics['trepan_original']['recall']),
                precision_metrics['trepan_original'].get('f1_macro', precision_metrics['trepan_original']['f1']),
                precision_metrics['trepan_original'].get('balanced_accuracy', precision_metrics['trepan_original']['accuracy'])
            ]
        
        # Trepan-Reloaded
        trepan_reloaded_values = None
        if precision_metrics['trepan_reloaded'] is not None:
            trepan_reloaded_values = [
                precision_metrics['trepan_reloaded'].get('precision_macro', precision_metrics['trepan_reloaded']['precision']),
                precision_metrics['trepan_reloaded'].get('recall_macro', precision_metrics['trepan_reloaded']['recall']),
                precision_metrics['trepan_reloaded'].get('f1_macro', precision_metrics['trepan_reloaded']['f1']),
                precision_metrics['trepan_reloaded'].get('balanced_accuracy', precision_metrics['trepan_reloaded']['accuracy'])
            ]

        c45_values = None
        if precision_metrics.get('c45_j48') is not None:
            c45_values = [
                precision_metrics['c45_j48'].get('precision_macro', precision_metrics['c45_j48']['precision']),
                precision_metrics['c45_j48'].get('recall_macro', precision_metrics['c45_j48']['recall']),
                precision_metrics['c45_j48'].get('f1_macro', precision_metrics['c45_j48']['f1']),
                precision_metrics['c45_j48'].get('balanced_accuracy', precision_metrics['c45_j48']['accuracy']),
            ]
        
        # Plotar radar
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Fechar o círculo
        
        if trepan_orig_values is not None:
            trepan_orig_values += trepan_orig_values[:1]
            ax.plot(angles, trepan_orig_values, 'o-', linewidth=2, label='Trepan-Original', color='#ff7f0e')
            ax.fill(angles, trepan_orig_values, alpha=0.25, color='#ff7f0e')

        if c45_values is not None:
            c45_values += c45_values[:1]
            ax.plot(angles, c45_values, 'o-', linewidth=2, label='C4.5-Nativo', color='#7f7f7f')
            ax.fill(angles, c45_values, alpha=0.18, color='#7f7f7f')
        
        if trepan_reloaded_values is not None:
            trepan_reloaded_values += trepan_reloaded_values[:1]
            ax.plot(angles, trepan_reloaded_values, 'o-', linewidth=2, label='Trepan-Reloaded', color='#2ca02c')
            ax.fill(angles, trepan_reloaded_values, alpha=0.25, color='#2ca02c')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Comparação Radar de Métricas', fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    def _plot_differences_comparison(self, ax):
        """Diferenças da métrica principal apresentada ao utilizador: Precisão Macro."""
        precision_metrics = self.comparison_results['precision']
        def pm(key):
            block = precision_metrics.get(key) or {}
            return float(block.get('precision_macro', block.get('precision', 0.0)))
        comparisons = []
        differences = []
        if precision_metrics.get('trepan_original') and precision_metrics.get('trepan_reloaded'):
            comparisons.append('T-Reload − T-Orig')
            differences.append(pm('trepan_reloaded') - pm('trepan_original'))
        if precision_metrics.get('trepan_original') and precision_metrics.get('c45_j48'):
            comparisons.append('C4.5 − T-Orig')
            differences.append(pm('c45_j48') - pm('trepan_original'))
        if precision_metrics.get('c45_j48') and precision_metrics.get('trepan_reloaded'):
            comparisons.append('T-Reload − C4.5')
            differences.append(pm('trepan_reloaded') - pm('c45_j48'))
        if comparisons:
            colors = ['red' if d < 0 else 'green' for d in differences]
            bars = ax.bar(comparisons, differences, color=colors, alpha=0.7)
            ax.set_title('Diferenças de Precisão Macro', fontweight='bold')
            ax.set_ylabel('Diferença de Precisão Macro')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            for bar, diff in zip(bars, differences):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.001 if diff >= 0 else -0.003),
                        f'{diff:+.4f}', ha='center',
                        va='bottom' if diff >= 0 else 'top', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Nenhuma comparação disponível', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Diferenças de Precisão Macro', fontweight='bold')

    def plot_performance_heatmap(self, figsize=(10, 6)):

        if not self.comparison_results:
            print("❌ Nenhuma comparação foi realizada ainda.")
            return None
        
        precision_metrics = self.comparison_results['precision']
        metrics_names = ['Precisão Macro', 'Recall Macro', 'Macro-F1', 'Balanced Accuracy']
        
        models = []
        if precision_metrics.get('trepan_original'):
            models.append('T-Orig')
        if precision_metrics.get('c45_j48'):
            models.append('C4.5')
        if precision_metrics.get('trepan_reloaded'):
            models.append('T-Reload')
        
        if len(models) < 2:
            print("⚠️ Precisa de pelo menos 2 modelos para comparação")
            return None
        
        # Matriz de valores absolutos das três árvores.
        values = np.zeros((len(metrics_names), len(models)))
        metric_keys = ['precision_macro', 'recall_macro', 'f1_macro', 'balanced_accuracy']
        
        for i, metric_key in enumerate(metric_keys):
            for j, model in enumerate(models):
                model_key = {
                    'T-Orig': 'trepan_original',
                    'T-Reload': 'trepan_reloaded',
                    'C4.5': 'c45_j48'
                }[model]
                values[i, j] = precision_metrics[model_key].get(metric_key, np.nan)
        
        # Criar heatmap
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(values, cmap='RdYlGn', aspect='auto', vmin=0.0, vmax=1.0)
        
        # Configurar ticks
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(metrics_names)))
        ax.set_xticklabels(models)
        ax.set_yticklabels(metrics_names)
        
        # Rotacionar labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Adicionar valores nas células
        for i in range(len(metrics_names)):
            for j in range(len(models)):
                if not np.isnan(values[i, j]):
                    ax.text(j, i, f'{values[i, j]:.3f}', ha="center", va="center", 
                                  color="black" if values[i, j] > 0.65 else "white",
                                  fontweight='bold')
        
        ax.set_title("Heatmap de desempenho das três árvores", fontweight='bold', pad=20)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Valor da métrica', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_violin(self, figsize=(12, 6)):

        if not self.comparison_results:
            print("❌ Nenhuma comparação foi realizada ainda.")
            return None
        
        precision_metrics = self.comparison_results['precision']
        
        # Coletar dados de validação cruzada se disponível
        cv_data = {}
        if 'cross_validation' in self.comparison_results:
            cv = self.comparison_results['cross_validation']
            if cv.get('trepan_original') and 'scores' in cv['trepan_original']:
                cv_data['T-Orig'] = cv['trepan_original']['scores']
            if cv.get('trepan_reloaded') and 'scores' in cv['trepan_reloaded']:
                cv_data['T-Reload'] = cv['trepan_reloaded']['scores']
            if cv.get('c45_j48') and 'scores' in cv['c45_j48']:
                cv_data['C4.5'] = cv['c45_j48']['scores']
        
        if not cv_data:
            print("⚠️ Dados de validação cruzada não disponíveis")
            return None
        
        # Preparar dados para violino
        data_for_violin = []
        labels = []
        for model_name, scores in cv_data.items():
            data_for_violin.append(scores)
            labels.append(model_name)
        
        # Criar gráfico de violino
        fig, ax = plt.subplots(figsize=figsize)
        parts = ax.violinplot(data_for_violin, positions=range(len(labels)), showmeans=True, showmedians=True)
        
        # Customizar violinos
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Acurácia (Validação Cruzada)')
        ax.set_title('Distribuição de Métricas (Gráfico de Violino)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrices_side_by_side(self, figsize=(16, 4), class_names=None):

        if not self.comparison_results:
            print("❌ Nenhuma comparação foi realizada ainda.")
            return None
        
        precision_metrics = self.comparison_results['precision']
        models_to_plot = []
        
        if precision_metrics.get('mlp') and precision_metrics['mlp'].get('additional_metrics'):
            cm_mlp = precision_metrics['mlp']['additional_metrics'].get('confusion_matrix')
            if cm_mlp is not None:
                models_to_plot.append(('MLP', cm_mlp, precision_metrics['mlp']['additional_metrics'].get('confusion_matrix_classes')))
        
        if precision_metrics.get('trepan_original') and precision_metrics['trepan_original'].get('additional_metrics'):
            cm_to = precision_metrics['trepan_original']['additional_metrics'].get('confusion_matrix')
            if cm_to is not None:
                models_to_plot.append(('Trepan-Original', cm_to, 
                                      precision_metrics['trepan_original']['additional_metrics'].get('confusion_matrix_classes')))
        
        if precision_metrics.get('trepan_reloaded') and precision_metrics['trepan_reloaded'].get('additional_metrics'):
            cm_tr = precision_metrics['trepan_reloaded']['additional_metrics'].get('confusion_matrix')
            if cm_tr is not None:
                models_to_plot.append(('Trepan-Reloaded', cm_tr,
                                      precision_metrics['trepan_reloaded']['additional_metrics'].get('confusion_matrix_classes')))
        
        if precision_metrics.get('c45_j48') and precision_metrics['c45_j48'].get('additional_metrics'):
            cm_c45 = precision_metrics['c45_j48']['additional_metrics'].get('confusion_matrix')
            if cm_c45 is not None:
                models_to_plot.append(('C4.5-Nativo', cm_c45,
                                      precision_metrics['c45_j48']['additional_metrics'].get('confusion_matrix_classes')))
        
        if not models_to_plot:
            print("⚠️ Nenhuma matriz de confusão disponível")
            return None
        
        n_models = len(models_to_plot)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, cm, cm_classes) in enumerate(models_to_plot):
            cm_array = np.array(cm)
            
            # Normalizar para percentuais
            cm_normalized = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            # Usar nomes de classes se disponível
            if class_names and cm_classes is not None:
                tick_labels = [class_names[int(cls)] if int(cls) < len(class_names) else f'C{int(cls)}' 
                             for cls in cm_classes]
            else:
                tick_labels = [f'C{int(cls)}' for cls in cm_classes] if cm_classes is not None else None
            
            # Plotar heatmap
            im = axes[idx].imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            axes[idx].set_title(f'{model_name}\nMatriz de Confusão', fontweight='bold')
            
            # Configurar ticks
            if tick_labels:
                axes[idx].set_xticks(np.arange(len(tick_labels)))
                axes[idx].set_yticks(np.arange(len(tick_labels)))
                axes[idx].set_xticklabels(tick_labels, rotation=45, ha='right')
                axes[idx].set_yticklabels(tick_labels)
            
            # Adicionar valores
            thresh = cm_normalized.max() / 2.
            for i in range(cm_normalized.shape[0]):
                for j in range(cm_normalized.shape[1]):
                    axes[idx].text(j, i, f'{cm_array[i, j]}\n({cm_normalized[i, j]*100:.1f}%)',
                                 ha="center", va="center",
                                 color="white" if cm_normalized[i, j] > thresh else "black",
                                 fontsize=9, fontweight='bold')
            
            axes[idx].set_ylabel('Verdadeiro')
            axes[idx].set_xlabel('Predito')
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves_comparative(self, models_dict, X_test, y_test, figsize=(10, 8), class_names=None):

        if not models_dict:
            print("❌ Nenhum modelo fornecido")
            return None
        
        unique_classes = np.unique(y_test)
        n_classes = len(unique_classes)
        
        if n_classes == 2:
            # Binário: curvas ROC simples
            fig, ax = plt.subplots(figsize=figsize)
            
            for model_name, model in models_dict.items():
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_test)
                        if y_proba.shape[1] == 2:
                            from sklearn.metrics import roc_curve, auc
                            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                            roc_auc = auc(fpr, tpr)
                            ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
                    except Exception as e:
                        print(f"⚠️ Erro ao calcular ROC para {model_name}: {e}")
            
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Taxa de Falsos Positivos')
            ax.set_ylabel('Taxa de Verdadeiros Positivos')
            ax.set_title('Curvas ROC Comparativas (Binário)', fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
        else:
            # Multiclasse: curvas ROC one-vs-rest
            fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(figsize[0] * 2, figsize[1]))
            if n_classes == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for class_idx, cls in enumerate(unique_classes):
                ax = axes[class_idx]
                
                # Binarizar y_test para esta classe
                y_test_binary = (y_test == cls).astype(int)
                
                for model_name, model in models_dict.items():
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_proba = model.predict_proba(X_test)
                            if class_idx < y_proba.shape[1]:
                                from sklearn.metrics import roc_curve, auc
                                fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, class_idx])
                                roc_auc = auc(fpr, tpr)
                                
                                cls_label = class_names[class_idx] if class_names and class_idx < len(class_names) else f'Classe {int(cls)}'
                                ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
                        except Exception as e:
                            print(f"⚠️ Erro ao calcular ROC para {model_name}, classe {cls}: {e}")
                
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Taxa de Falsos Positivos')
                ax.set_ylabel('Taxa de Verdadeiros Positivos')
                cls_label = class_names[class_idx] if class_names and class_idx < len(class_names) else f'Classe {int(cls)}'
                ax.set_title(f'ROC: {cls_label}', fontweight='bold')
                ax.legend(loc="lower right", fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Remover eixos vazios
            for idx in range(n_classes, len(axes)):
                fig.delaxes(axes[idx])
        
        plt.tight_layout()
        return fig
    
    # ==================== ANÁLISE TEMPORAL/ESTABILIDADE ====================
    
    def _record_execution(self, precision_metrics, fidelity_metrics, cross_val_metrics):

        execution_record = {
            'timestamp': datetime.now().isoformat(),
            'precision': {},
            'fidelity': {},
            'cross_validation': {}
        }
        
        # Registrar métricas de precisão
        if precision_metrics.get('mlp'):
            execution_record['precision']['mlp'] = {
                'accuracy': precision_metrics['mlp'].get('accuracy'),
                'precision': precision_metrics['mlp'].get('precision'),
                'recall': precision_metrics['mlp'].get('recall'),
                'f1': precision_metrics['mlp'].get('f1')
            }
        
        if precision_metrics.get('trepan_original'):
            execution_record['precision']['trepan_original'] = {
                'accuracy': precision_metrics['trepan_original'].get('accuracy'),
                'precision': precision_metrics['trepan_original'].get('precision'),
                'recall': precision_metrics['trepan_original'].get('recall'),
                'f1': precision_metrics['trepan_original'].get('f1')
            }
        
        if precision_metrics.get('trepan_reloaded'):
            execution_record['precision']['trepan_reloaded'] = {
                'accuracy': precision_metrics['trepan_reloaded'].get('accuracy'),
                'precision': precision_metrics['trepan_reloaded'].get('precision'),
                'recall': precision_metrics['trepan_reloaded'].get('recall'),
                'f1': precision_metrics['trepan_reloaded'].get('f1')
            }
        
        # Registrar métricas de fidelidade
        if fidelity_metrics.get('trepan_original'):
            execution_record['fidelity']['trepan_original'] = {
                'overall_fidelity': fidelity_metrics['trepan_original'].get('overall_fidelity')
            }
        
        if fidelity_metrics.get('trepan_reloaded'):
            execution_record['fidelity']['trepan_reloaded'] = {
                'overall_fidelity': fidelity_metrics['trepan_reloaded'].get('overall_fidelity')
            }
        
        # Registrar métricas de validação cruzada (incluindo variância entre folds)
        if cross_val_metrics.get('mlp'):
            mlp_cv = cross_val_metrics['mlp']
            if 'scores' in mlp_cv:
                execution_record['cross_validation']['mlp'] = {
                    'mean_accuracy': mlp_cv.get('mean_accuracy'),
                    'std_accuracy': mlp_cv.get('std_accuracy'),
                    'scores': mlp_cv.get('scores', []),
                    'variance': float(np.var(mlp_cv.get('scores', []))) if 'scores' in mlp_cv else None
                }
        
        if cross_val_metrics.get('trepan_original'):
            to_cv = cross_val_metrics['trepan_original']
            if 'scores' in to_cv:
                execution_record['cross_validation']['trepan_original'] = {
                    'mean_accuracy': to_cv.get('mean_accuracy'),
                    'std_accuracy': to_cv.get('std_accuracy'),
                    'scores': to_cv.get('scores', []),
                    'variance': float(np.var(to_cv.get('scores', []))) if 'scores' in to_cv else None
                }
        
        if cross_val_metrics.get('trepan_reloaded'):
            tr_cv = cross_val_metrics['trepan_reloaded']
            if 'scores' in tr_cv:
                execution_record['cross_validation']['trepan_reloaded'] = {
                    'mean_accuracy': tr_cv.get('mean_accuracy'),
                    'std_accuracy': tr_cv.get('std_accuracy'),
                    'scores': tr_cv.get('scores', []),
                    'variance': float(np.var(tr_cv.get('scores', []))) if 'scores' in tr_cv else None
                }
        
        self._execution_history.append(execution_record)
    
    def _calculate_stability_metrics(self):

        if len(self._execution_history) < 2:
            return
        
        stability = {}
        
        # Análise de estabilidade de precisão
        for model_name in ['mlp', 'trepan_original', 'trepan_reloaded', 'c45_j48']:
            model_key = 'trepan_original' if model_name == 'trepan_original' else \
                       'trepan_reloaded' if model_name == 'trepan_reloaded' else \
                       'c45_j48' if model_name == 'c45_j48' else 'mlp'
            
            # Extrair métricas ao longo das execuções
            accuracies = []
            precisions = []
            recalls = []
            f1s = []
            
            for execution in self._execution_history:
                if (execution['precision'].get(model_key) and 
                    execution['precision'][model_key].get('accuracy') is not None):
                    accuracies.append(execution['precision'][model_key]['accuracy'])
                    precisions.append(execution['precision'][model_key].get('precision'))
                    recalls.append(execution['precision'][model_key].get('recall'))
                    f1s.append(execution['precision'][model_key].get('f1'))
            
            if len(accuracies) > 1:
                stability[f'{model_name}_precision'] = {
                    'accuracy': {
                        'mean': float(np.mean(accuracies)),
                        'std': float(np.std(accuracies)),
                        'cv': float(np.std(accuracies) / np.mean(accuracies)) if np.mean(accuracies) > 0 else None,
                        'min': float(np.min(accuracies)),
                        'max': float(np.max(accuracies)),
                        'range': float(np.max(accuracies) - np.min(accuracies))
                    },
                    'precision': {
                        'mean': float(np.mean(precisions)) if precisions else None,
                        'std': float(np.std(precisions)) if precisions else None,
                        'cv': float(np.std(precisions) / np.mean(precisions)) if precisions and np.mean(precisions) > 0 else None
                    },
                    'recall': {
                        'mean': float(np.mean(recalls)) if recalls else None,
                        'std': float(np.std(recalls)) if recalls else None,
                        'cv': float(np.std(recalls) / np.mean(recalls)) if recalls and np.mean(recalls) > 0 else None
                    },
                    'f1': {
                        'mean': float(np.mean(f1s)) if f1s else None,
                        'std': float(np.std(f1s)) if f1s else None,
                        'cv': float(np.std(f1s) / np.mean(f1s)) if f1s and np.mean(f1s) > 0 else None
                    },
                    'n_executions': len(accuracies)
                }
        
        # Análise de estabilidade de fidelidade
        for model_name in ['trepan_original', 'trepan_reloaded']:
            model_key = 'trepan_original' if model_name == 'trepan_original' else 'trepan_reloaded'
            
            fidelities = []
            for execution in self._execution_history:
                if (execution['fidelity'].get(model_key) and 
                    execution['fidelity'][model_key].get('overall_fidelity') is not None):
                    fidelities.append(execution['fidelity'][model_key]['overall_fidelity'])
            
            if len(fidelities) > 1:
                stability[f'{model_name}_fidelity'] = {
                    'overall_fidelity': {
                        'mean': float(np.mean(fidelities)),
                        'std': float(np.std(fidelities)),
                        'cv': float(np.std(fidelities) / np.mean(fidelities)) if np.mean(fidelities) > 0 else None,
                        'min': float(np.min(fidelities)),
                        'max': float(np.max(fidelities)),
                        'range': float(np.max(fidelities) - np.min(fidelities))
                    },
                    'n_executions': len(fidelities)
                }
        
        # Análise de variância entre folds (validação cruzada)
        for model_name in ['mlp', 'trepan_original', 'trepan_reloaded']:
            model_key = 'trepan_original' if model_name == 'trepan_original' else \
                       'trepan_reloaded' if model_name == 'trepan_reloaded' else 'mlp'
            
            fold_variances = []
            fold_stds = []
            
            for execution in self._execution_history:
                if (execution['cross_validation'].get(model_key) and 
                    execution['cross_validation'][model_key].get('variance') is not None):
                    fold_variances.append(execution['cross_validation'][model_key]['variance'])
                    fold_stds.append(execution['cross_validation'][model_key].get('std_accuracy'))
            
            if len(fold_variances) > 0:
                stability[f'{model_name}_cv_variance'] = {
                    'mean_variance': float(np.mean(fold_variances)) if fold_variances else None,
                    'std_variance': float(np.std(fold_variances)) if len(fold_variances) > 1 else None,
                    'mean_std_across_folds': float(np.mean(fold_stds)) if fold_stds else None,
                    'cv_stability': {
                        'low': len([v for v in fold_stds if v < 0.01]) if fold_stds else 0,
                        'medium': len([v for v in fold_stds if 0.01 <= v < 0.05]) if fold_stds else 0,
                        'high': len([v for v in fold_stds if v >= 0.05]) if fold_stds else 0
                    },
                    'n_executions': len(fold_variances)
                }
        
        self._stability_metrics = stability
        self.comparison_results['stability'] = stability
    
    def get_stability_report(self):

        if not self._stability_metrics:
            return "⚠️ Nenhuma métrica de estabilidade disponível. Execute compare_all_models() múltiplas vezes."
        
        report = f"\n{'='*80}\n"
        report += f"📊 RELATÓRIO DE ESTABILIDADE TEMPORAL\n"
        report += f"{'='*80}\n\n"
        report += f"Total de execuções analisadas: {len(self._execution_history)}\n\n"
        
        for metric_key, metrics in self._stability_metrics.items():
            model_name = metric_key.split('_')[0]
            metric_type = '_'.join(metric_key.split('_')[1:])
            
            report += f"🔹 {model_name.upper()} - {metric_type.replace('_', ' ').title()}:\n"
            
            if 'accuracy' in metrics:
                acc = metrics['accuracy']
                report += f"   Acurácia: {acc['mean']:.4f} ± {acc['std']:.4f} "
                if acc['cv']:
                    report += f"(CV: {acc['cv']:.4f})\n"
                else:
                    report += "\n"
                report += f"   Range: [{acc['min']:.4f}, {acc['max']:.4f}] (diferença: {acc['range']:.4f})\n"
            
            if 'overall_fidelity' in metrics:
                fid = metrics['overall_fidelity']
                report += f"   Fidelidade: {fid['mean']:.4f} ± {fid['std']:.4f} "
                if fid['cv']:
                    report += f"(CV: {fid['cv']:.4f})\n"
                else:
                    report += "\n"
                report += f"   Range: [{fid['min']:.4f}, {fid['max']:.4f}] (diferença: {fid['range']:.4f})\n"
            
            if 'cv_stability' in metrics:
                cv_stab = metrics['cv_stability']
                report += f"   Estabilidade CV: Baixa variação: {cv_stab['low']}, Média: {cv_stab['medium']}, Alta: {cv_stab['high']}\n"
            
            report += f"   Número de execuções: {metrics.get('n_executions', 'N/A')}\n\n"
        
        return report
    
    # ==================== EXPORTAÇÃO ESTRUTURADA ====================
    
    def export_to_json(self, filepath=None):

        if not self.comparison_results:
            raise ValueError("Nenhum resultado disponível para exportar. Execute compare_all_models() primeiro.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"metrics_comparison_{timestamp}.json"
        
        # Converter numpy arrays para listas e outros tipos não-serializáveis
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            else:
                return obj
        
        export_data = convert_to_serializable(self.comparison_results)
        export_data['export_metadata'] = {
            'export_timestamp': datetime.now().isoformat(),
            'n_executions': len(self._execution_history),
            'stability_available': len(self._stability_metrics) > 0
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def export_to_csv(self, filepath=None, include_stability=False):

        if not self.comparison_results:
            raise ValueError("Nenhum resultado disponível para exportar. Execute compare_all_models() primeiro.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"metrics_comparison_{timestamp}.csv"
        
        rows = []
        
        # Métricas de precisão
        precision_metrics = self.comparison_results.get('precision', {})
        for model_name in ['mlp', 'trepan_original', 'trepan_reloaded', 'c45_j48']:
            model_key = 'trepan_original' if model_name == 'trepan_original' else \
                       'trepan_reloaded' if model_name == 'trepan_reloaded' else \
                       'c45_j48' if model_name == 'c45_j48' else 'mlp'
            
            if precision_metrics.get(model_key):
                metrics = precision_metrics[model_key]
                rows.append({
                    'Model': model_name,
                    'Metric_Type': 'Precision',
                    'Accuracy': metrics.get('accuracy'),
                    'Precision': metrics.get('precision'),
                    'Recall': metrics.get('recall'),
                    'F1_Score': metrics.get('f1')
                })
        
        # Métricas de fidelidade
        fidelity_metrics = self.comparison_results.get('fidelity', {})
        for model_name in ['trepan_original', 'trepan_reloaded']:
            model_key = 'trepan_original' if model_name == 'trepan_original' else 'trepan_reloaded'
            
            if fidelity_metrics.get(model_key):
                metrics = fidelity_metrics[model_key]
                rows.append({
                    'Model': model_name,
                    'Metric_Type': 'Fidelity',
                    'Overall_Fidelity': metrics.get('overall_fidelity'),
                    'Agreement_Rate': metrics.get('agreement_rate')
                })
        
        # Métricas de estabilidade (se solicitado)
        if include_stability and self._stability_metrics:
            for metric_key, metrics in self._stability_metrics.items():
                if 'accuracy' in metrics:
                    acc = metrics['accuracy']
                    rows.append({
                        'Model': metric_key.split('_')[0],
                        'Metric_Type': 'Stability_Accuracy',
                        'Mean': acc.get('mean'),
                        'Std': acc.get('std'),
                        'CV': acc.get('cv'),
                        'Range': acc.get('range')
                    })
        
        # Escrever CSV
        if rows:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                fieldnames = set()
                for row in rows:
                    fieldnames.update(row.keys())
                fieldnames = sorted(fieldnames)
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        return filepath

    def export_row_diagnostics(self, filepath=None):
        """Exporta o diagnóstico linha-a-linha do mesmo holdout bloqueado."""
        rows = self.comparison_results.get('row_diagnostics') or []
        if not rows:
            raise ValueError("Não existem diagnósticos linha-a-linha para exportar.")
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"locked_test_row_diagnostics_{timestamp}.csv"
        flat_rows = []
        for row in rows:
            flat = dict(row)
            flat['onto_feature_values'] = json.dumps(
                flat.get('onto_feature_values') or {}, ensure_ascii=False, default=str,
            )
            flat['owl_rule_or_concept'] = json.dumps(
                flat.get('owl_rule_or_concept') or {}, ensure_ascii=False, default=str,
            )
            flat_rows.append(flat)
        with open(filepath, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
        return filepath
    
    def export_to_latex(self, filepath=None):

        if not self.comparison_results:
            raise ValueError("Nenhum resultado disponível para exportar. Execute compare_all_models() primeiro.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"metrics_report_{timestamp}.tex"
        
        latex_content = []
        latex_content.append("\\documentclass[11pt,a4paper]{article}")
        latex_content.append("\\usepackage[utf8]{inputenc}")
        latex_content.append("\\usepackage[T1]{fontenc}")
        latex_content.append("\\usepackage{amsmath}")
        latex_content.append("\\usepackage{booktabs}")
        latex_content.append("\\usepackage{geometry}")
        latex_content.append("\\geometry{margin=2.5cm}")
        latex_content.append("\\title{Métricas Comparativas de Modelos}")
        latex_content.append(f"\\date{{{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}}")
        latex_content.append("\\begin{document}")
        latex_content.append("\\maketitle")
        latex_content.append("")
        latex_content.append("\\section{Métricas de Precisão}")
        latex_content.append("")
        
        # Tabela de precisão
        latex_content.append("\\begin{table}[h]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Métricas de Precisão por Modelo}")
        latex_content.append("\\begin{tabular}{lcccc}")
        latex_content.append("\\toprule")
        latex_content.append("Modelo & Acurácia & Precisão & Recall & F1-Score \\\\")
        latex_content.append("\\midrule")
        
        precision_metrics = self.comparison_results.get('precision', {})
        for model_name in ['Trepan-Original', 'C4.5-Nativo', 'Trepan-Reloaded']:
            model_key = 'trepan_original' if 'Original' in model_name else \
                       'trepan_reloaded' if 'Reloaded' in model_name else \
                       'c45_j48' if 'C4.5' in model_name else 'mlp'
            
            if precision_metrics.get(model_key):
                m = precision_metrics[model_key]
                latex_content.append(
                    f"{model_name} & {m.get('accuracy', 0):.4f} & "
                    f"{m.get('precision', 0):.4f} & {m.get('recall', 0):.4f} & "
                    f"{m.get('f1', 0):.4f} \\\\"
                )
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        latex_content.append("")
        
        # Métricas de fidelidade
        latex_content.append("\\section{Métricas de Fidelidade}")
        latex_content.append("")
        latex_content.append("\\begin{table}[h]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Fidelidade das Árvores ao MLP}")
        latex_content.append("\\begin{tabular}{lc}")
        latex_content.append("\\toprule")
        latex_content.append("Modelo & Fidelidade \\\\")
        latex_content.append("\\midrule")
        
        fidelity_metrics = self.comparison_results.get('fidelity', {})
        for model_name in ['Trepan-Original', 'Trepan-Reloaded']:
            model_key = 'trepan_original' if 'Original' in model_name else 'trepan_reloaded'
            
            if fidelity_metrics.get(model_key):
                fid = fidelity_metrics[model_key].get('overall_fidelity', 0)
                latex_content.append(f"{model_name} & {fid:.4f} \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        latex_content.append("")
        
        latex_content.append("\\end{document}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_content))
        
        return filepath
    
    def export_to_mlflow(self, experiment_name="trepan_comparison", run_name=None):

        try:
            import mlflow
            import mlflow.sklearn
        except ImportError:
            print("⚠️ MLflow não está instalado. Instale com: pip install mlflow")
            return None
        
        if not self.comparison_results:
            raise ValueError("Nenhum resultado disponível para exportar. Execute compare_all_models() primeiro.")
        
        # Criar ou obter experimento
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"⚠️ Erro ao criar/obter experimento: {e}")
            return None
        
        # Criar run
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            # Logar parâmetros
            mlflow.log_param("n_bootstrap", self.n_bootstrap)
            mlflow.log_param("random_state", self.random_state)
            
            # Logar métricas de precisão
            precision_metrics = self.comparison_results.get('precision', {})
            for model_name, metrics in precision_metrics.items():
                if metrics:
                    mlflow.log_metrics({
                        f"{model_name}_accuracy": metrics.get('accuracy', 0),
                        f"{model_name}_precision": metrics.get('precision', 0),
                        f"{model_name}_recall": metrics.get('recall', 0),
                        f"{model_name}_f1": metrics.get('f1', 0)
                    }, step=0)
            
            # Logar métricas de fidelidade
            fidelity_metrics = self.comparison_results.get('fidelity', {})
            for model_name, metrics in fidelity_metrics.items():
                if metrics:
                    mlflow.log_metric(f"{model_name}_fidelity", metrics.get('overall_fidelity', 0), step=0)
            
            # Logar métricas de validação cruzada
            cv_metrics = self.comparison_results.get('cross_validation', {})
            for model_name, metrics in cv_metrics.items():
                if metrics and 'mean_accuracy' in metrics:
                    mlflow.log_metrics({
                        f"{model_name}_cv_mean": metrics.get('mean_accuracy', 0),
                        f"{model_name}_cv_std": metrics.get('std_accuracy', 0)
                    }, step=0)
            
            # Logar métricas de estabilidade se disponíveis
            if self._stability_metrics:
                for metric_key, metrics in self._stability_metrics.items():
                    if 'accuracy' in metrics:
                        acc = metrics['accuracy']
                        mlflow.log_metrics({
                            f"stability_{metric_key}_mean": acc.get('mean', 0),
                            f"stability_{metric_key}_std": acc.get('std', 0),
                            f"stability_{metric_key}_cv": acc.get('cv', 0) if acc.get('cv') else 0
                        }, step=0)
            
            # Salvar resultados em arquivo JSON e logar como artefato
            json_path = self.export_to_json()
            mlflow.log_artifact(json_path)
            
            print(f"✅ Resultados exportados para MLflow: {run.info.run_id}")
            return run
        
        return None
