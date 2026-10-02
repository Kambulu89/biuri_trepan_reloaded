from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.io import arff
import pandas as pd
import os
import numpy as np

from core.mlp_optimizer import run_full_mlp_optimization, evaluate_classifier_metrics
from core.training_config import TrainingPreset, get_training_preset
from core.model_bundle import (
    ModelBundle,
    assert_train_test_schema,
    bundle_from_pipeline,
    log_model_input_check,
)
from core.arff_schema import OrderedLabelEncoder, parse_arff_class_order
from core.oracle_optimization import OracleGateConfig, calibrate_estimator
from core.mlp_convergence import extract_mlp_convergence

class MLPTrainer:
    @staticmethod
    def hidden_layer_sizes_for(n_features):
        """Dimensões adaptativas ao número de features (agnóstico ao dataset)."""
        n = max(1, int(n_features))
        h1 = int(min(128, max(16, round(n * 1.5))))
        h2 = int(min(64, max(8, round(n * 0.75))))
        return (h1, h2)

    def __init__(self):
        self.label_encoder = OrderedLabelEncoder()
        self.feature_encoders = {}
        self.model = None
        self.arff_meta = None
        self._extra_column_encoders = {}
        self.optimization_results = []
        self.last_optimization_summary = None
        self.eval_split = None
        self.bundle = None
        self.bypass_preprocessing = False
        self.transformed_feature_names = None
        self._encoded_train_matrix = None

    def reset(self, keep_label_encoder=False):
        """Reinicia encoders e modelo (evita conflito de dimensões entre treinos)."""
        self.feature_encoders = {}
        self._extra_column_encoders = {}
        self.model = None
        if not keep_label_encoder:
            declared = (self.arff_meta or {}).get('classes') or []
            self.label_encoder = OrderedLabelEncoder(declared)
        self.bundle = None
        self.bypass_preprocessing = False
        self.transformed_feature_names = None
        self._encoded_train_matrix = None

    def _normalize_cell(self, value):
        if isinstance(value, np.ndarray):
            if value.size == 1:
                value = value.item()
            else:
                return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if value is None:
            return ""
        return value

    def _column_needs_encoder(self, X, col_idx):
        for sample in X:
            val = self._normalize_cell(sample[col_idx])
            if val == "" or (isinstance(val, float) and np.isnan(val)):
                continue
            if isinstance(val, str):
                try:
                    float(val)
                    return False
                except (TypeError, ValueError):
                    return True
            try:
                float(val)
                return False
            except (TypeError, ValueError):
                return True
        return False

    def _encode_features(self, X):
        X_arr = np.asarray(X, dtype=object)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        encoded = []
        for sample in X_arr:
            sample_encoded = []
            for i, value in enumerate(sample):
                value = self._normalize_cell(value)
                if i in self.feature_encoders:
                    sample_encoded.append(
                        self.feature_encoders[i].transform([str(value)])[0]
                    )
                else:
                    sample_encoded.append(float(value))
            encoded.append(sample_encoded)
        return np.array(encoded, dtype=float)

    def _create_encoder_for_feature(self, i, X):
        le = LabelEncoder()
        X_arr = np.asarray(X, dtype=object)
        values = [str(self._normalize_cell(row[i])) for row in X_arr]
        le.fit(values)
        self.feature_encoders[i] = le

    def _fit_encoders(self, X):
        if X is None:
            return
        X_arr = np.asarray(X, dtype=object)
        if X_arr.size == 0:
            return
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        n_cols = X_arr.shape[1]
        for i in range(n_cols):
            if self._column_needs_encoder(X_arr, i):
                self._create_encoder_for_feature(i, X_arr)

    def _stratified_split(self, X_encoded, y_encoded, test_size=0.3, random_state=42):
        try:
            return train_test_split(
                X_encoded,
                y_encoded,
                test_size=test_size,
                random_state=random_state,
                stratify=y_encoded,
            )
        except ValueError:
            return train_test_split(
                X_encoded,
                y_encoded,
                test_size=test_size,
                random_state=random_state,
            )

    def _feature_names_for_space(self, n_features: int, oracle_space: str) -> list:
        meta = self.arff_meta or {}
        if oracle_space in ("augmented", "enriched"):
            names = meta.get("augmented_features") or meta.get("features")
        elif oracle_space == "residual_ontological":
            residual = meta.get("residual_input") or {}
            n_base = int(residual.get("n_base_probs", 0))
            n_onto = int(residual.get("n_onto_features", 0))
            if n_base + n_onto == n_features:
                return [f"base_prob_{i}" for i in range(n_base)] + [
                    f"onto_residual_{i}" for i in range(n_onto)
                ]
            names = meta.get("features")
        else:
            names = meta.get("original_features") or meta.get("features")
        if names and len(names) == n_features:
            return list(names)
        return [f"feature_{i}" for i in range(n_features)]

    def _attach_bundle(
        self,
        model,
        X_train,
        model_name: str,
        oracle_space: str,
        X_test=None,
    ) -> ModelBundle:
        X_train = np.asarray(X_train)
        if X_test is not None:
            assert_train_test_schema(X_train, np.asarray(X_test), model_name)
        n_features = int(X_train.shape[1])
        feature_names = self._feature_names_for_space(n_features, oracle_space)
        space_map = {
            "original": "original",
            "augmented": "enriched",
            "enriched": "enriched",
            "residual_ontological": "residual",
        }
        feature_space = space_map.get(oracle_space, oracle_space)
        self.bundle = bundle_from_pipeline(
            model_name, model, feature_names, feature_space
        )
        log_model_input_check(self.bundle, X_train, operation="fit")
        return self.bundle

    def train(
        self,
        X,
        y,
        optimize=True,
        dataset_name='unknown',
        run_grid=True,
        run_optuna=True,
        preset=None,
        cancel_fn=None,
        progress_fn=None,
        optuna_trials=None,
    ):
        self.reset(keep_label_encoder=False)
        self.optimization_results = []
        self.last_optimization_summary = None

        # Encoders de features são ajustados apenas no split de treino.
        y_for_split = np.asarray(y)
        indices = np.arange(len(y_for_split))
        train_indices, _, _, _ = self._stratified_split(indices, y_for_split)
        X_object = np.asarray(X, dtype=object)
        self._fit_encoders(X_object[np.asarray(train_indices, dtype=int)])
        X_encoded = self._encode_features(X)
        declared = (self.arff_meta or {}).get('classes') or []
        self.label_encoder = OrderedLabelEncoder(declared)
        y_encoded = self.label_encoder.fit_transform(y)

        if self.arff_meta is not None:
            self.arff_meta['oracle_space'] = 'original'

        if optimize:
            selected_preset = preset or get_training_preset('scientific')
            return self.train_optimized_from_encoded(
                X_encoded,
                y_encoded,
                model_name='MLP Original',
                dataset_name=dataset_name,
                oracle_space='original',
                run_grid=selected_preset.run_grid if preset is not None else run_grid,
                run_optuna=selected_preset.run_optuna if preset is not None else run_optuna,
                preset=selected_preset,
                cancel_fn=cancel_fn,
                progress_fn=progress_fn,
                optuna_trials=optuna_trials,
            )

        from dataclasses import replace
        preset = replace(
            preset or get_training_preset('scientific'),
            run_grid=False, run_optuna=False, run_baseline=True,
        )
        return self.train_optimized_from_encoded(
            X_encoded,
            y_encoded,
            model_name='MLP Original',
            dataset_name=dataset_name,
            oracle_space='original',
            run_grid=False,
            run_optuna=False,
            preset=preset,
            cancel_fn=cancel_fn,
            progress_fn=progress_fn,
            optuna_trials=optuna_trials,
        )

    def train_optimized_from_encoded(
        self,
        X_encoded,
        y_encoded,
        model_name='MLP Original',
        dataset_name='unknown',
        oracle_space='original',
        run_grid=True,
        run_optuna=True,
        optuna_trials=50,
        preset=None,
        cancel_fn=None,
        progress_fn=None,
    ):
        """Treino otimizado com split estratificado, StandardScaler e seleção do melhor método."""
        preset = preset or get_training_preset('scientific')
        if optuna_trials is None:
            optuna_trials = preset.optuna_trials

        X_arr = np.asarray(X_encoded, dtype=float)
        y_arr = np.asarray(y_encoded)

        indices = np.arange(len(y_arr))
        train_indices, test_indices, _, _ = self._stratified_split(indices, y_arr)
        train_indices = np.asarray(train_indices, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)
        X_train, X_test = X_arr[train_indices], X_arr[test_indices]
        y_train, y_test = y_arr[train_indices], y_arr[test_indices]
        assert_train_test_schema(X_train, X_test, model_name)
        self.eval_split = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_indices': train_indices,
            'test_indices': test_indices,
        }

        best, all_results = run_full_mlp_optimization(
            X_train,
            y_train,
            X_test,
            y_test,
            model_name=model_name,
            dataset_name=dataset_name,
            run_grid=run_grid,
            run_optuna=run_optuna,
            optuna_trials=optuna_trials,
            preset=preset,
            cancel_fn=cancel_fn,
            progress_fn=progress_fn,
            evaluate_selected_on_test=False,
        )

        self.model = best['model']
        calibration_audit = None
        if getattr(preset, 'calibrate_probabilities', True):
            try:
                self.model, calibration_audit = calibrate_estimator(
                    self.model,
                    X_train,
                    y_train,
                    config=OracleGateConfig(
                        folds=getattr(preset, 'calibration_cv_folds', 5),
                        calibration_method=getattr(
                            preset, 'calibration_method', 'sigmoid'
                        ),
                    ),
                )
            except ValueError as exc:
                calibration_audit = {
                    'scope': 'outer_training_only', 'test_used': False,
                    'skipped': True, 'reason': str(exc),
                }
        final_test_metrics = evaluate_classifier_metrics(self.model, X_test, y_test)
        best['test_metrics'] = final_test_metrics
        best['final_model_metrics'] = final_test_metrics
        best['pre_calibration_model_metrics'] = None
        self.optimization_results = all_results
        self.last_optimization_summary = best
        self.last_optimization_summary['calibration'] = calibration_audit
        self._attach_bundle(self.model, X_train, model_name, oracle_space, X_test=X_test)

        if self.arff_meta is None:
            self.arff_meta = {}
        self.arff_meta['oracle_space'] = oracle_space
        self.arff_meta['mlp_optimization'] = {
            'method': best.get('optimization_method'),
            'best_params': best.get('best_params'),
            'test_accuracy': best.get('test_metrics', {}).get('accuracy'),
            'convergence': best.get('convergence') or best.get('test_metrics', {}).get('convergence'),
            'calibration': calibration_audit,
        }

        self._encoded_train_matrix = X_arr
        if self.bundle is not None and getattr(self.bundle, 'feature_names', None):
            self.transformed_feature_names = list(self.bundle.feature_names)
        elif self.arff_meta and self.arff_meta.get('features'):
            self.transformed_feature_names = list(self.arff_meta['features'])

        return self.model, X_arr, y_arr

    def train_on_encoded(
        self,
        X_encoded,
        y_encoded,
        oracle_space='augmented',
        optimize=True,
        dataset_name='unknown',
        run_grid=True,
        run_optuna=True,
        optuna_trials=50,
        preset=None,
        cancel_fn=None,
        progress_fn=None,
    ):
        """
        Treina MLP sobre matriz já codificada (ex.: X_encoded_aug com colunas onto_*).
        Não reinicia label_encoder — partilha o estado do treino ARFF original.
        """
        if optimize:
            selected_preset = preset or get_training_preset('scientific')
            _, _, _ = self.train_optimized_from_encoded(
                X_encoded,
                y_encoded,
                model_name='MLP_Onto',
                dataset_name=dataset_name,
                oracle_space=oracle_space,
                run_grid=selected_preset.run_grid if preset is not None else run_grid,
                run_optuna=selected_preset.run_optuna if preset is not None else run_optuna,
                optuna_trials=optuna_trials,
                preset=selected_preset,
                cancel_fn=cancel_fn,
                progress_fn=progress_fn,
            )
            return self.model

        from dataclasses import replace
        preset = replace(
            preset or get_training_preset('scientific'),
            run_grid=False, run_optuna=False, run_baseline=True,
        )
        return self.train_optimized_from_encoded(
            X_encoded,
            y_encoded,
            model_name='MLP_Onto',
            dataset_name=dataset_name,
            oracle_space=oracle_space,
            run_grid=False,
            run_optuna=False,
            preset=preset,
            cancel_fn=cancel_fn,
            progress_fn=progress_fn,
            optuna_trials=optuna_trials,
        )[0]

    def train_mlp_residual(
        self,
        X_residual,
        y_encoded,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        optimize=True,
        dataset_name='unknown',
        run_grid=True,
        run_optuna=True,
        optuna_trials=50,
        preset=None,
        cancel_fn=None,
        progress_fn=None,
    ):
        """
        Treina MLP Residual Ontológico sobre [base_probs, onto_features].
        Partilha label_encoder com o MLP Original (não reinicia).
        """
        preset = preset or get_training_preset('scientific')
        if optuna_trials is None and preset is not None:
            optuna_trials = preset.optuna_trials

        if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
            assert_train_test_schema(X_train, X_test, "MLP Residual Ontológico")
            self.eval_split = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
            }
            log_model_input_check(
                ModelBundle(
                    name="MLP Residual Ontológico",
                    pipeline=None,
                    feature_names=self._feature_names_for_space(
                        int(np.asarray(X_train).shape[1]), "residual_ontological"
                    ),
                    n_features=int(np.asarray(X_train).shape[1]),
                    feature_space="residual",
                ),
                X_train,
                operation="fit",
            )
            if len(X_train) < 30 or not optimize:
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                self.model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(
                        hidden_layer_sizes=self.hidden_layer_sizes_for(np.asarray(X_train).shape[1]),
                        # Em amostras pequenas o solver quasi-Newton é mais estável
                        # que Adam sem early stopping e evita selecionar um modelo
                        # apenas porque atingiu o teto de iterações.
                        solver='lbfgs',
                        max_iter=2000,
                        tol=1e-5,
                        early_stopping=False,
                        random_state=42,
                    )),
                ]).fit(X_train, y_train)
                convergence_audit = extract_mlp_convergence(self.model)
                calibration_audit = None
                if getattr(preset, 'calibrate_probabilities', True):
                    self.model, calibration_audit = calibrate_estimator(
                        self.model, X_train, y_train,
                        config=OracleGateConfig(
                            folds=getattr(preset, 'calibration_cv_folds', 5),
                            calibration_method=getattr(
                                preset, 'calibration_method', 'sigmoid'
                            ),
                        ),
                    )
                self.optimization_results = []
                self.last_optimization_summary = {
                    'model': self.model,
                    'optimization_method': 'small_sample_safe_fit',
                    'best_params': {
                        'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-5,
                        'early_stopping': False, 'random_state': 42,
                    },
                    'convergence': convergence_audit,
                    'calibration': calibration_audit,
                }
                self._attach_bundle(
                    self.model, X_train, "MLP Residual Ontológico",
                    "residual_ontological", X_test=X_test,
                )
                return self.model
            best, all_results = run_full_mlp_optimization(
                X_train,
                y_train,
                X_test,
                y_test,
                model_name='MLP Residual Ontológico',
                dataset_name=dataset_name,
                run_grid=run_grid if preset is None else preset.run_grid,
                run_optuna=run_optuna if preset is None else preset.run_optuna,
                optuna_trials=optuna_trials,
                preset=preset,
                cancel_fn=cancel_fn,
                progress_fn=progress_fn,
            )
            self.model = best['model']
            calibration_audit = None
            if getattr(preset, 'calibrate_probabilities', True):
                try:
                    self.model, calibration_audit = calibrate_estimator(
                        self.model, X_train, y_train,
                        config=OracleGateConfig(
                            folds=getattr(preset, 'calibration_cv_folds', 5),
                            calibration_method=getattr(
                                preset, 'calibration_method', 'sigmoid'
                            ),
                        ),
                    )
                except ValueError as exc:
                    calibration_audit = {
                        'scope': 'outer_training_only', 'test_used': False,
                        'skipped': True, 'reason': str(exc),
                    }
            self.optimization_results = all_results
            self.last_optimization_summary = best
            self.last_optimization_summary['calibration'] = calibration_audit
            self._attach_bundle(
                self.model, X_train, "MLP Residual Ontológico", "residual_ontological",
                X_test=X_test,
            )
            if self.arff_meta is None:
                self.arff_meta = {}
            self.arff_meta['oracle_space'] = 'residual_ontological'
            self.arff_meta['mlp_optimization'] = {
                'method': best.get('optimization_method'),
                'best_params': best.get('best_params'),
                'test_accuracy': best.get('test_metrics', {}).get('accuracy'),
                'calibration': calibration_audit,
            }
            return self.model

        if optimize:
            return self.train_optimized_from_encoded(
                X_residual,
                y_encoded,
                model_name='MLP Residual Ontológico',
                dataset_name=dataset_name,
                oracle_space='residual_ontological',
                run_grid=run_grid,
                run_optuna=run_optuna,
                optuna_trials=optuna_trials,
                preset=preset,
                cancel_fn=cancel_fn,
                progress_fn=progress_fn,
            )[0]

        preset = preset or get_training_preset('scientific')
        X_arr = np.asarray(X_residual, dtype=float)
        y_arr = np.asarray(y_encoded)
        return self.train_optimized_from_encoded(
            X_arr,
            y_arr,
            model_name='MLP Residual Ontológico',
            dataset_name=dataset_name,
            oracle_space='residual_ontological',
            run_grid=False,
            run_optuna=False,
            preset=preset,
            cancel_fn=cancel_fn,
            progress_fn=progress_fn,
            optuna_trials=optuna_trials,
        )[0]

    def encode_dataset(self, X, y=None, transform_labels=True):
        """Codifica features (e opcionalmente rótulos) sem treinar novo MLP."""
        X_encoded = self._encode_features(X)
        if y is None:
            return X_encoded, None
        if transform_labels:
            y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = np.asarray(y)
        return X_encoded, y_encoded

    def encode_columns_by_name(
        self, X, column_names, base_column_names, extra_encoders=None
    ):
        """
        Codifica matriz alinhada a column_names.
        Colunas em base_column_names usam feature_encoders do treino ARFF;
        demais colunas (onto_*) usam encoders auxiliares por nome de coluna.
        """
        if extra_encoders is None:
            extra_encoders = self._extra_column_encoders
        base_set = set(base_column_names)
        name_to_base_idx = {n: i for i, n in enumerate(base_column_names)}

        for col_name in column_names:
            if col_name in base_set:
                continue
            col_idx = column_names.index(col_name)
            values = [row[col_idx] for row in X]
            if any(isinstance(v, str) for v in values):
                le = LabelEncoder()
                le.fit([str(v) for v in values])
                extra_encoders[col_name] = le

        encoded = []
        for sample in X:
            row = []
            for col_idx, col_name in enumerate(column_names):
                value = sample[col_idx]
                if col_name in base_set:
                    base_i = name_to_base_idx[col_name]
                    if base_i in self.feature_encoders:
                        row.append(self.feature_encoders[base_i].transform([value])[0])
                    else:
                        row.append(float(value))
                elif col_name in extra_encoders:
                    row.append(extra_encoders[col_name].transform([str(value)])[0])
                else:
                    row.append(float(value))
            encoded.append(row)
        self._extra_column_encoders = extra_encoders
        return np.array(encoded, dtype=float)

    def set_metadata(self, file_name, features, target, classes, sample_count):
        
        self.arff_meta = {
            'file_name': file_name,
            'features': features,
            'target': target,
            'classes': classes,
            'sample_count': sample_count
        }

    def load_arff(self, file_path):
        
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)

        # Decodificação segura de strings
        str_cols = df.select_dtypes([object]).columns
        for col in str_cols:
            df[col] = df[col].str.decode('utf-8')

        features = df.columns[:-1].tolist()
        target = df.columns[-1]
        classes = parse_arff_class_order(file_path, str(target))
        if not classes:
            classes = list(meta[target][1]) if target in meta else []

        self.arff_meta = {
            'file_name': os.path.basename(file_path),
            'features': features,
            'target': target,
            'classes': classes,
            'sample_count': len(df)
        }

        return df.iloc[:, :-1].values, df.iloc[:, -1].values

    def transform(self, X):
        """Codifica features brutas para o espaço do MLP."""
        if self.bypass_preprocessing:
            return np.asarray(X, dtype=float)
        return self._encode_features(X)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        X_enc = self.transform(X)
        return self.model.predict(X_enc)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        X_enc = self.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_enc)
        raise AttributeError("Modelo não suporta predict_proba")

    def get_encoded_training_matrix(self):
        """Devolve (X_train, y_train) codificados após treino."""
        if self.eval_split is not None:
            return self.eval_split['X_train'], self.eval_split['y_train']
        if self._encoded_train_matrix is not None:
            return self._encoded_train_matrix, None
        return None, None

    def get_transformed_feature_names(self):
        if self.transformed_feature_names is not None:
            return list(self.transformed_feature_names)
        if self.bundle is not None and getattr(self.bundle, 'feature_names', None):
            return list(self.bundle.feature_names)
        if self.arff_meta and self.arff_meta.get('features'):
            return list(self.arff_meta['features'])
        n = self._encoded_train_matrix.shape[1] if self._encoded_train_matrix is not None else 0
        return [f'f_{i}' for i in range(n)]

    def get_feature_intervals(self, X_encoded=None):
        X_use = np.asarray(
            X_encoded if X_encoded is not None else self._encoded_train_matrix,
            dtype=float,
        )
        if X_use is None:
            raise ValueError("Matriz codificada indisponível.")
        intervals = []
        for col in range(X_use.shape[1]):
            intervals.append((float(np.min(X_use[:, col])), float(np.max(X_use[:, col]))))
        return np.array(intervals, dtype=object)

    def export_cf_state(self):
        """Metadados para pipeline de contrafactuais na GUI."""
        X_enc, y_enc = self.get_encoded_training_matrix()
        return {
            'X_train_enc': X_enc,
            'y_train_enc': y_enc,
            'transformed_feature_names': self.get_transformed_feature_names(),
            'feature_intervals': self.get_feature_intervals(X_enc) if X_enc is not None else None,
            'class_labels': {
                i: str(c) for i, c in enumerate(getattr(self.label_encoder, 'classes_', []))
            },
            'is_multiclass': len(getattr(self.label_encoder, 'classes_', [])) > 2,
        }
