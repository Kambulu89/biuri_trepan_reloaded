from core.mlp_trainer import MLPTrainer
from core.training_config import enforce_scientific_preset
from core.trepan_reloaded_extractor import TrepanReloadedExtractor
from core.residual_ontological_oracle import (
    ResidualOntologicalOracle,
    build_residual_feature_matrix,
    build_oof_residual_feature_matrices,
    original_column_indices,
    onto_column_indices,
)
from core.ontology_acceptance import (
    compute_oracle_test_metrics,
    evaluate_selected_teacher_candidate,
    select_oracle_model,
)
from core.semantic_utility_gate import SemanticUtilityConfig, SemanticUtilityGate
from core.ontology_stage_status import build_ontology_stage_status
from core.hybrid_oracle import (
    FeatureProjectedOracle, HybridOracleConfig, WeightedHybridOracle,
    cross_fitted_probabilities, fit_validation_weighted_hybrid,
)
from pathlib import Path
import numpy as np
from sklearn.base import clone

from core.c45_j48_tree import C45Classifier
from core.oracle_optimization import (
    OracleGateConfig, compare_ontological_oracle_to_c45,
    optimize_and_calibrate_mlp,
)


class TrepanReloaded:
    """Orquestrador Biuri: ontologia carregada na GUI determina o modo do extrator."""

    def __init__(self, ontology=None, use_default_ontology=False):
        self.ontology = ontology

        if self.ontology is None and use_default_ontology:
            self.ontology = self._load_default_ontology()

        self.mlp_trainer = MLPTrainer()
        self.mlp_trainer_onto = MLPTrainer()
        self.mlp_trainer_residual = MLPTrainer()
        self.extractor = TrepanReloadedExtractor(self.ontology)
        self.arff_meta = None
        self.mlp_model = None
        self.mlp_model_onto = None
        self.mlp_model_residual = None
        self.mlp_model_onto_direct = None
        self.mlp_model_hybrid = None
        self.selected_oracle = None
        self.selected_oracle_label = None
        self.ontology_acceptance = None
        self.augmented_eval_split = None

    @property
    def label_encoder(self):
        """Sempre o encoder actual do MLPTrainer (evita referência obsoleta após reset/train)."""
        return self.mlp_trainer.label_encoder

    @property
    def has_active_ontology(self):
        return self.ontology is not None

    @staticmethod
    def _attach_ontology_stage_status(acceptance, quality_report):
        """Separa validade estrutural OWL, feature gate e professor ontológico."""
        block = dict(acceptance or {})
        block.update(build_ontology_stage_status(quality_report, block))
        block['ontology_structural_label'] = (
            'Sim' if block.get('ontology_structural_available') else 'Não'
        )
        return block

    def set_ontology(self, ontology, onto_feature_bias_weight=None):
        """Ativa modo semântico completo quando ontology não é None."""
        prev_bias = getattr(self.extractor, 'onto_feature_bias_weight', None)
        self.ontology = ontology
        bias = onto_feature_bias_weight if onto_feature_bias_weight is not None else prev_bias
        if hasattr(self.extractor, 'set_ontology'):
            self.extractor.set_ontology(ontology)
            if bias is not None:
                self.extractor.set_onto_feature_bias_weight(bias)
        else:
            self.extractor = TrepanReloadedExtractor(
                ontology, onto_feature_bias_weight=bias
            )

    def clear_ontology(self):
        self.set_ontology(None)

    def mirror_original_tree(self, tree, feature_names=None, original_audit=None):
        """Sem ontologia, espelha árvore e auditoria do Trepan-Original.

        Devolve sempre uma auditoria válida para que consumidores da GUI não
        dependam de um atributo criado apenas no ramo com OWL.
        """
        if hasattr(self.extractor, 'adopt_original_tree'):
            self.extractor.adopt_original_tree(tree, feature_names=feature_names)
        else:
            self.extractor.explainer_tree = tree
        audit = dict(original_audit or getattr(self.extractor, 'last_audit', {}) or {})
        audit.update({
            'model': 'Trepan-Reloaded',
            'mode': 'mirrored_no_ontology',
            'ontology_active': False,
            'oracle': audit.get('oracle') or 'MLP Original',
            'feature_space': 'original',
            'mirrored_from': 'Trepan-Original',
        })
        self.extractor.last_audit = dict(audit)
        return audit

    def _load_default_ontology(self):
        try:
            current_dir = Path(__file__).parent.parent
            default_ontology_path = current_dir / "data" / "sample_ontology.owl"

            if default_ontology_path.exists():
                from core.owl_runtime import load_ontology_isolated
                onto = load_ontology_isolated(default_ontology_path)
                print(f"[INFO] Ontologia padrao carregada: {default_ontology_path}")
                return onto
            print(f"[WARN] Ontologia padrao nao encontrada em: {default_ontology_path}")
            return None
        except Exception as e:
            print(f"[ERROR] Erro ao carregar ontologia padrao: {e}")
            return None

    def train_mlp(
        self,
        X,
        y,
        dataset_name='unknown',
        optimize=True,
        run_grid=True,
        run_optuna=True,
        preset=None,
        cancel_fn=None,
        progress_fn=None,
    ):
        preset = enforce_scientific_preset(preset)
        try:
            model, X_encoded, y_encoded = self.mlp_trainer.train(
                X,
                y,
                optimize=optimize,
                dataset_name=dataset_name,
                run_grid=run_grid,
                run_optuna=run_optuna,
                preset=preset,
                cancel_fn=cancel_fn,
                progress_fn=progress_fn,
            )
            self.mlp_model = model
            return "MLP treinado com sucesso!", model, X_encoded, y_encoded
        except InterruptedError:
            return "Treino cancelado pelo utilizador.", None, None, None
        except Exception as e:
            return f"Erro no treinamento: {str(e)}", None, None, None

    def train_mlp_onto(
        self,
        X_encoded_aug,
        y_encoded_aug,
        augmented_feature_names=None,
        dataset_name='unknown',
        optimize=True,
        run_grid=True,
        run_optuna=True,
        preset=None,
        cancel_fn=None,
        progress_fn=None,
    ):
        """Oráculo enriquecido (MLP_Onto) treinado na matriz augmentada."""
        preset = enforce_scientific_preset(preset)
        try:
            self.mlp_trainer_onto.label_encoder = self.mlp_trainer.label_encoder
            self.mlp_trainer_onto.feature_encoders = dict(self.mlp_trainer.feature_encoders)
            self.mlp_trainer_onto._extra_column_encoders = dict(
                self.mlp_trainer._extra_column_encoders
            )
            if self.mlp_trainer.arff_meta is not None:
                meta = dict(self.mlp_trainer.arff_meta)
                if augmented_feature_names is not None:
                    meta['features'] = list(augmented_feature_names)
                    meta['augmented_features'] = list(augmented_feature_names)
                meta['oracle_space'] = 'augmented'
                self.mlp_trainer_onto.arff_meta = meta
            model = self.mlp_trainer_onto.train_on_encoded(
                X_encoded_aug,
                y_encoded_aug,
                oracle_space='augmented',
                optimize=optimize,
                dataset_name=dataset_name,
                run_grid=run_grid,
                run_optuna=run_optuna,
                preset=preset,
                cancel_fn=cancel_fn,
                progress_fn=progress_fn,
            )
            self.mlp_model_onto = model
            n_onto = getattr(model, 'n_features_in_', X_encoded_aug.shape[1])
            print(
                f"[INFO] MLP_Onto treinado: {n_onto} features "
                f"(matriz augmentada: {X_encoded_aug.shape[1]})."
            )
            return "MLP_Onto (oráculo enriquecido) treinado com sucesso!", model
        except InterruptedError:
            self.mlp_model_onto = None
            return "Treino MLP_Onto cancelado pelo utilizador.", None
        except Exception as e:
            self.mlp_model_onto = None
            return f"Erro no treinamento MLP_Onto: {str(e)}", None

    def train_mlp_residual_onto(
        self,
        X_encoded_aug,
        y_encoded_aug,
        mlp_original,
        original_feature_names,
        augmented_feature_names,
        dataset_name='unknown',
        optimize=True,
        run_grid=True,
        run_optuna=True,
        preset=None,
        cancel_fn=None,
        progress_fn=None,
        acceptance_tolerance=0.01,
        top_k_onto_features=None,
        X_train_aug=None,
        X_test_aug=None,
        X_train_raw=None,
        y_train=None,
        y_test=None,
        ontology_quality_report=None,
    ):
        """
        Treina MLP Residual Ontológico: [base_probs(MlP Original), onto_features].
        Avalia critério de aceitação e selecciona oráculo final.
        """
        preset = enforce_scientific_preset(preset)
        try:
            from sklearn.model_selection import train_test_split
            from core.onto_feature_selector import apply_ontology_feature_selection

            X_arr = np.asarray(X_encoded_aug, dtype=float)
            y_arr = np.asarray(y_encoded_aug)
            if X_train_aug is None or X_test_aug is None or y_train is None or y_test is None:
                try:
                    X_outer_train, X_outer_test, y_outer_train, y_outer_test = train_test_split(
                        X_arr, y_arr, test_size=0.3, random_state=42, stratify=y_arr,
                    )
                except ValueError:
                    X_outer_train, X_outer_test, y_outer_train, y_outer_test = train_test_split(
                        X_arr, y_arr, test_size=0.3, random_state=42,
                    )
            else:
                X_outer_train = np.asarray(X_train_aug, dtype=float)
                X_outer_test = np.asarray(X_test_aug, dtype=float)
                y_outer_train = np.asarray(y_train)
                y_outer_test = np.asarray(y_test)

            processor = getattr(self.extractor, 'ontology_processor', None)
            # Para pipelines programáticos/CLI numéricos, reconstruir o frame
            # original a partir da matriz enriquecida quando não existem specs
            # categóricas. A GUI fornece X_train_raw explicitamente.
            if X_train_raw is None and processor is not None:
                specs = list(getattr(processor, 'feature_specs_', []) or [])
                has_categorical_semantics = any(
                    spec.get('kind') == 'categorical_group' for spec in specs
                )
                if not has_categorical_semantics:
                    try:
                        aug_names_for_raw = list(map(str, augmented_feature_names))
                        raw_indices = [
                            aug_names_for_raw.index(str(name))
                            for name in original_feature_names
                        ]
                        import pandas as pd
                        X_train_raw = pd.DataFrame(
                            X_outer_train[:, raw_indices],
                            columns=list(map(str, original_feature_names)),
                        )
                    except Exception:
                        X_train_raw = None

            # Validação interna para seleção/aceitação. O teste externo fica bloqueado.
            # Os índices são preservados para que o feature engineering semântico
            # possa ser recalibrado dentro de cada fold OOF sobre os dados brutos.
            inner_indices = np.arange(len(y_outer_train), dtype=int)
            try:
                fit_indices, validation_indices = train_test_split(
                    inner_indices, test_size=0.25,
                    random_state=1042, stratify=y_outer_train,
                )
            except ValueError:
                fit_indices, validation_indices = train_test_split(
                    inner_indices, test_size=0.25, random_state=1042,
                )
            fit_indices = np.asarray(fit_indices, dtype=int)
            validation_indices = np.asarray(validation_indices, dtype=int)
            X_fit_aug = X_outer_train[fit_indices]
            X_validation_aug = X_outer_train[validation_indices]
            y_fit = y_outer_train[fit_indices]
            y_validation = y_outer_train[validation_indices]
            raw_fit_frame = None
            if X_train_raw is not None:
                import pandas as pd
                raw_outer = pd.DataFrame(X_train_raw).copy()
                if len(raw_outer) != len(X_outer_train):
                    raise ValueError(
                        "X_train_raw deve estar alinhado às linhas de X_train_aug."
                    )
                if list(map(str, raw_outer.columns)) != list(map(str, original_feature_names)):
                    raise ValueError(
                        "X_train_raw deve conter exactamente as features originais na mesma ordem."
                    )
                raw_fit_frame = raw_outer.iloc[fit_indices].reset_index(drop=True)

            # A seleção OWL é comandada por ganho OOF contra rótulos reais.
            # ``top_k`` permanece como limite de capacidade, não como aceitação.
            gate_limit = int(top_k_onto_features) if top_k_onto_features else 40
            semantic_gate = SemanticUtilityGate(SemanticUtilityConfig(
                cv_folds=getattr(preset, 'mlp_cv_folds', 5) if preset else 5,
                max_semantic_features=max(1, gate_limit),
                random_state=42,
            ))
            semantic_report = semantic_gate.evaluate(
                X_fit_aug,
                y_fit,
                list(augmented_feature_names),
                list(original_feature_names),
                quality_report=ontology_quality_report,
                reference_estimator=mlp_original,
                semantic_processor=processor,
                raw_base_frame=raw_fit_frame,
            )
            if processor is not None and hasattr(processor, 'merge_semantic_validation_audit'):
                processor.merge_semantic_validation_audit(semantic_report)
            selected_indices = list(semantic_report['selected_indices'])
            selected_names = list(semantic_report['selected_feature_names'])
            selection_info = semantic_report
            X_fit_aug = apply_ontology_feature_selection(
                X_fit_aug, selected_indices
            )
            X_validation_aug = apply_ontology_feature_selection(
                X_validation_aug, selected_indices
            )
            X_outer_train_selected = apply_ontology_feature_selection(
                X_outer_train, selected_indices
            )
            X_outer_test_selected = apply_ontology_feature_selection(
                X_outer_test, selected_indices
            )
            augmented_feature_names = selected_names

            if not semantic_report.get('accepted'):
                acceptance = {
                    'accepted': False,
                    'ontology_feature_gate_accepted': False,
                    'oracle_gate_accepted': False,
                    'surrogate_gate_accepted': None,
                    'semantic_utility_status': semantic_report.get('status'),
                    'ontology_accepted_label': 'Não',
                    'ontology_impact': 'no_validated_predictive_utility',
                    'feature_selection': semantic_report,
                    'selected_feature_names': list(selected_names),
                    'selected_feature_indices': list(selected_indices),
                    'selection_role': 'development_oof_only',
                    'final_test_locked': True,
                    'reason': 'ONTOLOGY_VALID_BUT_NO_PREDICTIVE_UTILITY',
                }
                acceptance = self._attach_ontology_stage_status(
                    acceptance, ontology_quality_report
                )
                self.ontology_acceptance = acceptance
                self.selected_oracle = mlp_original
                self.selected_oracle_label = 'MLP Original'
                self.mlp_model_residual = None
                self.mlp_model_onto_direct = None
                self.mlp_model_hybrid = None
                self.augmented_eval_split = {
                    'X_train': X_outer_train_selected,
                    'X_test': X_outer_test_selected,
                    'y_train': y_outer_train,
                    'y_test': y_outer_test,
                    'test_role': 'locked_final_test_not_accessed_for_selection',
                }
                return (
                    'OWL válida, mas sem ganho preditivo OOF; professor original mantido.',
                    mlp_original, acceptance, mlp_original,
                )

            orig_idx = original_column_indices(
                augmented_feature_names, original_feature_names
            )
            onto_idx = onto_column_indices(
                augmented_feature_names, original_feature_names
            )
            if not onto_idx:
                acceptance = {
                    'accepted': False,
                    'ontology_feature_gate_accepted': bool(semantic_report.get('accepted')),
                    'oracle_gate_accepted': False,
                    'surrogate_gate_accepted': None,
                    'ontology_accepted_label': 'Não',
                    'teacher_has_ontology': False,
                    'ontology_impact': 'no_independent_semantic_features',
                    'feature_selection': semantic_report,
                    'selected_feature_names': list(selected_names),
                    'selected_feature_indices': list(selected_indices),
                    'selection_role': 'development_oof_only',
                    'final_test_locked': True,
                    'reason': 'NO_INDEPENDENT_ONTOLOGY_FEATURES',
                }
                acceptance = self._attach_ontology_stage_status(
                    acceptance, ontology_quality_report
                )
                self.ontology_acceptance = acceptance
                self.selected_oracle = mlp_original
                self.selected_oracle_label = 'MLP Original'
                self.mlp_model_residual = None
                self.mlp_model_onto_direct = None
                self.mlp_model_hybrid = None
                self.augmented_eval_split = {
                    'X_train': X_outer_train_selected,
                    'X_test': X_outer_test_selected,
                    'y_train': y_outer_train,
                    'y_test': y_outer_test,
                    'test_role': 'locked_final_test_not_accessed_for_selection',
                }
                return (
                    "Sem features onto_* informativas — MLP Residual Ontológico não aplicável.",
                    mlp_original,
                    acceptance,
                    mlp_original,
                )

            # Modelo original de gate reajustado apenas no subtreino interno;
            # o MLP original final pode ter visto toda a partição de desenvolvimento.
            original_gate_model = clone(mlp_original).fit(
                X_fit_aug[:, orig_idx], y_fit
            )

            X_residual_train, X_residual_test, oof_info = (
                build_oof_residual_feature_matrices(
                    X_fit_aug,
                    y_fit,
                    X_validation_aug,
                    original_gate_model,
                    orig_idx,
                    onto_idx,
                    n_splits=getattr(preset, 'mlp_cv_folds', 5) if preset else 5,
                )
            )

            from core.model_bundle import assert_train_test_schema, log_model_input_check, ModelBundle
            assert_train_test_schema(
                X_residual_train, X_residual_test, "MLP Residual Ontológico"
            )
            log_model_input_check(
                ModelBundle(
                    name="MLP Residual Ontológico (residual matrix)",
                    pipeline=None,
                    feature_names=[],
                    n_features=int(X_residual_train.shape[1]),
                    feature_space="residual",
                ),
                X_residual_train,
                operation="fit",
            )

            self.mlp_trainer_residual.label_encoder = self.mlp_trainer.label_encoder
            if self.mlp_trainer.arff_meta is not None:
                meta = dict(self.mlp_trainer.arff_meta)
                meta['residual_input'] = {
                    'n_base_probs': X_residual_train.shape[1] - len(onto_idx),
                    'n_onto_features': len(onto_idx),
                    **oof_info,
                }
                self.mlp_trainer_residual.arff_meta = meta

            self.mlp_trainer_residual.train_mlp_residual(
                X_residual_train,
                y_fit,
                X_train=X_residual_train,
                X_test=X_residual_test,
                y_train=y_fit,
                y_test=y_validation,
                optimize=optimize,
                dataset_name=dataset_name,
                run_grid=run_grid,
                run_optuna=run_optuna,
                preset=preset,
                cancel_fn=cancel_fn,
                progress_fn=progress_fn,
            )
            mlp_residual = self.mlp_trainer_residual.model
            self.mlp_trainer_residual.eval_split = {
                'X_train': X_residual_train,
                'X_test': X_residual_test,
                'y_train': y_fit,
                'y_test': y_validation,
            }
            self.augmented_eval_split = {
                'X_train': X_outer_train_selected,
                'X_test': X_outer_test_selected,
                'y_train': y_outer_train,
                'y_test': y_outer_test,
                'test_role': 'locked_final_test',
            }

            residual_oracle_gate = ResidualOntologicalOracle(
                mlp_original=original_gate_model,
                mlp_residual=mlp_residual,
                original_column_indices=orig_idx,
                onto_column_indices=onto_idx,
                n_enriched_features=X_fit_aug.shape[1],
            )

            # MLP Ontológico direto: segundo professor independente, otimizado
            # por BA, macro-F1, calibração e estabilidade no subtreino.
            search_iterations = max(
                2, min(8, int(getattr(preset, 'optuna_trials', 30) // 10 or 2))
            )
            onto_direct_gate, onto_direct_audit = optimize_and_calibrate_mlp(
                X_fit_aug, y_fit, random_state=2051,
                n_iter=search_iterations,
                cv_folds=getattr(preset, 'mlp_cv_folds', 5) if preset else 5,
            )
            projected_original_gate = FeatureProjectedOracle(
                original_gate_model, orig_idx, X_fit_aug.shape[1]
            )
            gate_components = {
                'original': projected_original_gate,
                'ontological': onto_direct_gate,
                'residual': residual_oracle_gate,
            }
            hybrid_weights, hybrid_gate = fit_validation_weighted_hybrid(
                gate_components, X_validation_aug, y_validation,
                original_component='original',
                config=HybridOracleConfig(
                    minimum_gain_over_original=max(0.0, acceptance_tolerance / 2.0)
                ),
            )

            # Comparação justa com C4.5 usando o mesmo subtreino/validação.
            c45_gate = C45Classifier(
                confidence_factor=0.25, min_samples_leaf=2, random_state=42,
            ).fit(X_fit_aug[:, orig_idx], y_fit)
            c45_metrics = compute_oracle_test_metrics(
                c45_gate, X_validation_aug[:, orig_idx], y_validation
            )
            original_metrics = compute_oracle_test_metrics(
                projected_original_gate, X_validation_aug, y_validation
            )
            direct_metrics = compute_oracle_test_metrics(
                onto_direct_gate, X_validation_aug, y_validation
            )
            residual_metrics = compute_oracle_test_metrics(
                residual_oracle_gate, X_validation_aug, y_validation
            )
            oracle_vs_c45_oof = compare_ontological_oracle_to_c45(
                onto_direct_gate,
                C45Classifier(
                    confidence_factor=0.25, min_samples_leaf=2, random_state=42,
                ),
                X_fit_aug,
                X_fit_aug[:, orig_idx],
                y_fit,
                config=OracleGateConfig(
                    folds=getattr(preset, 'mlp_cv_folds', 5) if preset else 5,
                    random_state=5071,
                    noninferiority_margin=acceptance_tolerance,
                ),
            )

            # A utility híbrida é apenas um critério de descoberta. Antes de
            # qualquer reajuste final, aplica-se o contrato científico estrito:
            # accuracy + macro-F1 + balanced accuracy não podem degradar além
            # da margem previamente definida. O teste externo continua intocado.
            selected_validation_metrics = dict(hybrid_gate['selected'])
            strict_oracle_gate = evaluate_selected_teacher_candidate(
                original_metrics=original_metrics,
                selected_metrics=selected_validation_metrics,
                hybrid_weights=hybrid_weights,
                tolerance=acceptance_tolerance,
                ontology_feature_gate_accepted=bool(semantic_report.get('accepted')),
            )
            candidate_teacher_has_ontology = bool(
                strict_oracle_gate.get('candidate_teacher_has_ontology')
            )
            oracle_gate_accepted = bool(strict_oracle_gate.get('oracle_gate_accepted'))
            frozen_hybrid_weights = dict(hybrid_weights)
            if not oracle_gate_accepted:
                # Fallback real: a árvore verá o MLP Original projetado no mesmo
                # espaço, em vez de manter accepted=True com massa OWL zero.
                frozen_hybrid_weights = {
                    'original': 1.0, 'ontological': 0.0, 'residual': 0.0,
                }

            # Reajuste final em toda a partição de desenvolvimento, depois de
            # congelar features, componentes e pesos. O teste permanece intocado.
            onto_direct_final = clone(onto_direct_gate).fit(
                X_outer_train_selected, y_outer_train
            )
            X_residual_outer, _, residual_outer_oof_audit = (
                build_oof_residual_feature_matrices(
                    X_outer_train_selected, y_outer_train,
                    X_outer_train_selected, mlp_original,
                    orig_idx, onto_idx,
                    n_splits=getattr(preset, 'mlp_cv_folds', 5) if preset else 5,
                    random_state=3049,
                )
            )
            residual_final_pipeline = clone(mlp_residual).fit(
                X_residual_outer, y_outer_train
            )
            residual_oracle_final = ResidualOntologicalOracle(
                mlp_original=mlp_original,
                mlp_residual=residual_final_pipeline,
                original_column_indices=orig_idx,
                onto_column_indices=onto_idx,
                n_enriched_features=X_outer_train_selected.shape[1],
            )
            projected_original_final = FeatureProjectedOracle(
                mlp_original, orig_idx, X_outer_train_selected.shape[1]
            )
            final_components = {
                'original': projected_original_final,
                'ontological': onto_direct_final,
                'residual': residual_oracle_final,
            }

            # Cada linha real recebe probabilidades previstas por um modelo que
            # não a treinou. Estas probabilidades alimentam a destilação posterior.
            classes = np.asarray(mlp_original.classes_)
            original_oof, original_oof_audit = cross_fitted_probabilities(
                mlp_original, X_outer_train_selected[:, orig_idx], y_outer_train,
                classes=classes,
                cv_folds=getattr(preset, 'mlp_cv_folds', 5) if preset else 5,
                random_state=4019,
            )
            direct_oof, direct_oof_audit = cross_fitted_probabilities(
                onto_direct_final, X_outer_train_selected, y_outer_train,
                classes=classes,
                cv_folds=getattr(preset, 'mlp_cv_folds', 5) if preset else 5,
                random_state=4019,
            )
            residual_oof, residual_oof_audit = cross_fitted_probabilities(
                residual_final_pipeline, X_residual_outer, y_outer_train,
                classes=classes,
                cv_folds=getattr(preset, 'mlp_cv_folds', 5) if preset else 5,
                random_state=4019,
            )
            combined_oof = (
                frozen_hybrid_weights['original'] * original_oof
                + frozen_hybrid_weights['ontological'] * direct_oof
                + frozen_hybrid_weights['residual'] * residual_oof
            )
            hybrid_audit = {
                **hybrid_gate,
                'candidate_weights_before_strict_gate': dict(hybrid_weights),
                'final_weights_after_strict_gate': dict(frozen_hybrid_weights),
                'strict_oracle_gate': dict(strict_oracle_gate),
                'oracle_gate_accepted': oracle_gate_accepted,
                'component_oof': {
                    'original': original_oof_audit,
                    'ontological': direct_oof_audit,
                    'residual': residual_oof_audit,
                },
                'residual_outer_oof': residual_outer_oof_audit,
            }
            hybrid_oracle = WeightedHybridOracle(
                final_components, frozen_hybrid_weights, classes,
                X_outer_train_selected.shape[1],
                training_oof_probabilities=combined_oof,
                audit=hybrid_audit,
            )
            # Contrato final do gate: um candidato rejeitado não permanece
            # semanticamente como "híbrido" só porque os seus pesos viraram
            # (1, 0, 0). O professor activo passa a ser explicitamente o MLP
            # Original projectado; o híbrido candidato fica apenas na auditoria.
            selected_oracle = (
                hybrid_oracle if oracle_gate_accepted else projected_original_final
            )
            self.mlp_model_onto_direct = onto_direct_final
            self.mlp_model_residual = residual_oracle_final
            self.mlp_model_hybrid = selected_oracle

            teacher_has_ontology = bool(oracle_gate_accepted)
            acceptance = {
                'accepted': oracle_gate_accepted,
                'ontology_feature_gate_accepted': bool(semantic_report.get('accepted')),
                'oracle_gate_accepted': oracle_gate_accepted,
                'surrogate_gate_accepted': None,
                'semantic_utility_status': semantic_report.get('status'),
                'teacher_status': hybrid_gate.get('status'),
                'teacher_has_ontology': teacher_has_ontology,
                'ontology_accepted_label': 'Sim' if oracle_gate_accepted else 'Não',
                'ontology_impact': (
                    'validated_noninferior_ontological_teacher'
                    if oracle_gate_accepted else 'rejected_by_strict_oracle_quality_gate'
                ),
                'accuracy_original': original_metrics['accuracy'],
                'accuracy_onto': selected_validation_metrics['accuracy'],
                'f1_original': original_metrics['macro_f1'],
                'f1_onto': selected_validation_metrics['macro_f1'],
                'balanced_accuracy_original': original_metrics['balanced_accuracy'],
                'balanced_accuracy_onto': selected_validation_metrics['balanced_accuracy'],
                'balanced_accuracy_gain': (
                    selected_validation_metrics['balanced_accuracy']
                    - original_metrics['balanced_accuracy']
                ),
                'original_metrics': original_metrics,
                'direct_ontological_metrics': direct_metrics,
                'residual_metrics': residual_metrics,
                'c45_internal_validation_metrics': c45_metrics,
                'ontological_oracle_vs_c45_oof': oracle_vs_c45_oof,
                'selected_teacher_beats_c45_on_internal_validation': bool(
                    selected_validation_metrics['balanced_accuracy']
                    > c45_metrics['balanced_accuracy']
                    and selected_validation_metrics['macro_f1']
                    > c45_metrics['f1']
                ),
                'hybrid_gate': hybrid_gate,
                'hybrid_candidate_oracle_type': getattr(hybrid_oracle, 'ORACLE_TYPE', type(hybrid_oracle).__name__),
                'final_active_oracle_type': getattr(selected_oracle, 'ORACLE_TYPE', type(selected_oracle).__name__),
                'hybrid_weights_candidate': dict(hybrid_weights),
                'hybrid_weights': dict(frozen_hybrid_weights),
                'strict_oracle_gate': strict_oracle_gate,
                'ontological_mlp_optimization': onto_direct_audit,
                'reason': strict_oracle_gate.get('reason'),
            }
            acceptance['feature_selection'] = selection_info
            acceptance['selected_feature_names'] = list(selected_names)
            acceptance['selected_feature_indices'] = list(selected_indices)
            acceptance['residual_training_protocol'] = oof_info
            acceptance['selection_role'] = 'development_oof_and_internal_validation_only'
            acceptance['final_test_locked'] = True
            acceptance = self._attach_ontology_stage_status(
                acceptance, ontology_quality_report
            )
            self.ontology_acceptance = acceptance

            label = (
                'Oráculo Híbrido Ontológico OOF'
                if teacher_has_ontology else 'MLP Original projetado no espaço OWL (fallback do gate estrito)'
            )
            self.selected_oracle = selected_oracle
            self.selected_oracle_label = label

            print(
                f"[INFO] Gate semântico={semantic_report['status']} | "
                f"pesos híbridos candidato={hybrid_weights} | "
                f"pesos finais={frozen_hybrid_weights} | "
                f"gate estrito={oracle_gate_accepted} | "
                f"BA professor={selected_validation_metrics['balanced_accuracy']:.3f} | "
                f"BA C4.5={c45_metrics['balanced_accuracy']:.3f} | Oráculo={label}"
            )
            return (
                (
                    f"Oráculo ontológico validado por não-inferioridade. Professor seleccionado: {label}."
                    if oracle_gate_accepted else
                    f"Oráculo ontológico rejeitado pelo gate estrito; fallback seleccionado: {label}."
                ),
                selected_oracle,
                acceptance,
                selected_oracle,
            )
        except InterruptedError:
            self.mlp_model_residual = None
            self.ontology_acceptance = None
            return "Treino MLP Residual cancelado pelo utilizador.", None, None, None
        except Exception as e:
            self.mlp_model_residual = None
            self.ontology_acceptance = None
            return f"Erro no treinamento MLP Residual: {str(e)}", None, None, None

    def extract_interpretable_tree(self, X_encoded, y_encoded, feature_names=None, class_names=None):
        if not self.mlp_model:
            return "Nenhum modelo MLP treinado"

        if self.has_active_ontology and feature_names is not None and class_names is not None:
            return self.extractor.extract_tree_with_ontology(
                self.mlp_model, X_encoded, y_encoded, feature_names, class_names
            )

        return self.extractor.extract_tree(
            self.mlp_model,
            X_encoded,
            y_encoded,
            feature_names=feature_names,
            class_names=class_names,
        )

    def ensure_ontology_dominance(self, mlp_model, X_encoded, y_encoded,
                                  feature_names, class_names,
                                  trepan_original_tree, c45_tree,
                                  X_eval=None, y_eval=None, mlp_model_onto=None):
        """Avaliação informativa de dominância — não altera a árvore Trepan-Reloaded."""
        if not self.has_active_ontology:
            return self.extractor.explainer_tree
        return self.extractor.report_ontology_dominance_status(
            mlp_model, X_encoded, y_encoded,
            feature_names, class_names,
            trepan_original_tree, c45_tree,
            X_eval=X_eval, y_eval=y_eval,
            mlp_model_onto=mlp_model_onto,
        )

    def generate_semantic_explanations(self, feature_names, class_names):
        if not self.has_active_ontology:
            return "❌ Explicações semânticas disponíveis apenas com ontologia OWL carregada na GUI."
        if hasattr(self.extractor, 'generate_semantic_explanations'):
            return self.extractor.generate_semantic_explanations(feature_names, class_names)
        return "❌ Extrator sem suporte a explicações semânticas."

    def validate_semantic_coherence(self, X_encoded, y_encoded, feature_names, class_names):
        if not self.has_active_ontology:
            return "ℹ️ Validação semântica desativada: nenhuma ontologia OWL carregada."
        if hasattr(self.extractor, 'validate_semantic_coherence'):
            return self.extractor.validate_semantic_coherence(
                self.mlp_model, X_encoded, y_encoded, feature_names, class_names
            )
        return "❌ Validação semântica indisponível no extrator atual."
