import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

class MLPTrainer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.pipeline = None
        self.preprocessor = None
        self.model = None
        self.feature_names = None
        self.categorical_features_original = None
        self.transformed_feature_names = None
        # Generador aleatorio con semilla fija para reproducibilidad
        self.rng = np.random.RandomState(42)
        # Umbral calibrado (solo binario)
        self.optimal_threshold = 0.5
        # ═══════════════════════════════════════════════════════════
        self.bypass_preprocessing = False   # ← NUEVO ATRIBUTO
        # ═══════════════════════════════════════════════════════════

    def _balance_data(self, X, y):
        """Sobremuestrea la clase minoritaria para igualar el tamaño de la mayoritaria."""
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='target')
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            return X, y
        max_count = class_counts.max()
        min_class = class_counts.idxmin()
        min_count = class_counts.min()
        if min_count == max_count:
            return X, y

        X_min = X[y == min_class]
        y_min = y[y == min_class]
        X_maj = X[y != min_class]
        y_maj = y[y != min_class]

        n_oversample = max_count - min_count
        indices_extra = self.rng.choice(len(X_min), size=n_oversample, replace=True)
        X_extra = X_min.iloc[indices_extra].copy()
        y_extra = y_min.iloc[indices_extra].copy()

        X_bal = pd.concat([X_maj, X_min, X_extra], axis=0, ignore_index=True)
        y_bal = pd.concat([y_maj, y_min, y_extra], axis=0, ignore_index=True)

        shuffle_idx = self.rng.permutation(len(X_bal))
        X_bal = X_bal.iloc[shuffle_idx].reset_index(drop=True)
        y_bal = y_bal.iloc[shuffle_idx].reset_index(drop=True)
        return X_bal, y_bal

    def _undersample_data(self, X, y):
        """Reduce la clase mayoritaria al tamaño de la minoritaria."""
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='target')
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            return X, y
        min_class = class_counts.idxmin()
        min_count = class_counts.min()
        maj_class = class_counts.idxmax()

        X_min = X[y == min_class]
        y_min = y[y == min_class]
        X_maj = X[y == maj_class]
        y_maj = y[y == maj_class]

        indices = self.rng.choice(len(X_maj), size=min_count, replace=False)
        X_bal = pd.concat([X_min, X_maj.iloc[indices]], axis=0, ignore_index=True)
        y_bal = pd.concat([y_min, y_maj.iloc[indices]], axis=0, ignore_index=True)

        shuffle_idx = self.rng.permutation(len(X_bal))
        return X_bal.iloc[shuffle_idx].reset_index(drop=True), y_bal.iloc[shuffle_idx].reset_index(drop=True)

    def train(self, X, y, categorical_features=None, mlp_params=None,
              tune=False, param_grid=None, balance_method='oversample',
              use_minmax=False, calibrate_threshold=True):
        """
        Entrena el MLP con preprocesamiento y balanceo.
        Devuelve (modelo, X_transformado_original, y_codificada_original).

        Parámetros nuevos:
        - use_minmax: si True, usa MinMaxScaler en lugar de StandardScaler.
        - calibrate_threshold: si True, calcula el umbral óptimo para clasificación binaria.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame de pandas.")

        self.feature_names = X.columns.tolist()
        self.categorical_features_original = categorical_features if categorical_features is not None else []
        numeric_features = [col for col in self.feature_names if col not in self.categorical_features_original]

        # 1. Ajustar LabelEncoder con las etiquetas originales
        self.label_encoder.fit(y)
        y_original_enc = self.label_encoder.transform(y)

        # 2. Balancear datos de entrenamiento (ahora con opción 'none')
        if balance_method == 'undersample':
            X_bal, y_bal = self._undersample_data(X, y)
        elif balance_method == 'oversample':
            X_bal, y_bal = self._balance_data(X, y)
        else:   # 'none' o cualquier otro valor -> sin balanceo manual
            X_bal, y_bal = X, y
        y_bal_enc = self.label_encoder.transform(y_bal)

        # 3. Construir preprocesador (con opción MinMaxScaler)
        transformers = []
        if numeric_features:
            scaler = MinMaxScaler if use_minmax else StandardScaler
            transformers.append(('num', scaler(), numeric_features))
        if self.categorical_features_original:
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                 self.categorical_features_original)
            )
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

        # 4. Parámetros por defecto del MLP
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'alpha': 0.001,
            'learning_rate_init': 0.001,
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 20,
            'random_state': 42
        }
        if mlp_params:
            default_params.update(mlp_params)

        # 5. Búsqueda de hiperparámetros o entrenamiento directo
        if tune:
            if param_grid is None:
                param_grid = {
                    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'classifier__alpha': [0.0001, 0.001, 0.01],
                    'classifier__learning_rate_init': [0.001, 0.01]
                }
            base_model = MLPClassifier(
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20, random_state=42, max_iter=2000
            )
            pipe = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', base_model)
            ])
            gs = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=0)
            gs.fit(X_bal, y_bal_enc)
            best_params = gs.best_params_
            for key, value in best_params.items():
                if key.startswith('classifier__'):
                    default_params[key.split('__')[1]] = value
            self.model = MLPClassifier(**default_params)
            self.preprocessor.fit(X_bal)
            X_bal_trans = self.preprocessor.transform(X_bal)
            self.model.fit(X_bal_trans, y_bal_enc)
            self.pipeline = None
        else:
            self.model = MLPClassifier(**default_params)
            self.pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', self.model)
            ])
            self.pipeline.fit(X_bal, y_bal_enc)
            self.preprocessor = self.pipeline.named_steps['preprocessor']
            self.model = self.pipeline.named_steps['classifier']

        # 6. Nombres de características transformadas
        try:
            self.transformed_feature_names = self.preprocessor.get_feature_names_out()
        except AttributeError:
            cat_names = []
            if self.categorical_features_original:
                cat_encoder = self.preprocessor.named_transformers_['cat']
                cat_names = cat_encoder.get_feature_names_out(self.categorical_features_original)
            self.transformed_feature_names = np.array(numeric_features + list(cat_names))

        # 7. Transformar el conjunto original (tamaño original)
        X_transformed = self.preprocessor.transform(X)

        # 8. Calibración del umbral (solo binario)
        if calibrate_threshold and len(self.model.classes_) == 2:
            probas = self.model.predict_proba(X_transformed)
            thresholds = np.linspace(0.01, 0.99, 99)
            best_acc = 0.0
            best_thresh = 0.5
            base_acc = accuracy_score(y_original_enc, (probas[:, 1] >= 0.5).astype(int))
            for th in thresholds:
                y_pred = (probas[:, 1] >= th).astype(int)
                acc = accuracy_score(y_original_enc, y_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = th
            # Si no mejora, dejamos 0.5
            if best_acc > base_acc:
                self.optimal_threshold = best_thresh
            else:
                self.optimal_threshold = 0.5
        else:
            self.optimal_threshold = 0.5

        # 9. Devolver modelo, X original transformado, y etiquetas originales codificadas
        return self.model, X_transformed, y_original_enc

    def predict(self, X):
        """Predice usando el umbral calibrado si es binario, si no, el estándar."""
        if self.preprocessor is None:
            raise ValueError("Preprocesador no entrenado.")
        # ═══════════════════════════════════════════════════════════
        if self.bypass_preprocessing:
            X_trans = X
        else:
            X_trans = self.transform(X)
        # ═══════════════════════════════════════════════════════════
        probas = self.model.predict_proba(X_trans)
        if probas.shape[1] == 2 and hasattr(self, 'optimal_threshold'):
            return (probas[:, 1] >= self.optimal_threshold).astype(int)
        else:
            return self.model.predict(X_trans)

    def predict_proba(self, X):
        """Devuelve las probabilidades del modelo sin aplicar umbral."""
        if self.preprocessor is None:
            raise ValueError("Preprocesador no entrenado.")
        # ═══════════════════════════════════════════════════════════
        if self.bypass_preprocessing:
            X_trans = X
        else:
            X_trans = self.transform(X)
        # ═══════════════════════════════════════════════════════════
        return self.model.predict_proba(X_trans)

    def transform(self, X):
        if self.preprocessor is None:
            raise ValueError("Preprocesador no entrenado.")
        if not isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                X = pd.DataFrame(X, columns=self.feature_names)
            else:
                raise ValueError("No se conocen nombres de características.")
        return self.preprocessor.transform(X)

    def get_transformed_feature_names(self):
        return self.transformed_feature_names

    def save(self, filepath):
        data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'categorical_features_original': self.categorical_features_original,
            'transformed_feature_names': self.transformed_feature_names,
            'preprocessor': self.preprocessor,
            'model': self.model,
            'optimal_threshold': self.optimal_threshold,
            'bypass_preprocessing': self.bypass_preprocessing   # persistencia
        }
        joblib.dump(data, filepath)

    def load(self, filepath):
        data = joblib.load(filepath)
        self.pipeline = data['pipeline']
        self.label_encoder = data['label_encoder']
        self.feature_names = data['feature_names']
        self.categorical_features_original = data['categorical_features_original']
        self.transformed_feature_names = data['transformed_feature_names']
        self.preprocessor = data['preprocessor']
        self.model = data['model']
        self.optimal_threshold = data.get('optimal_threshold', 0.5)
        self.bypass_preprocessing = data.get('bypass_preprocessing', False)