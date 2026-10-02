# Documentación de MLPTrainer (`counterfactuals/cf_mlp_trainer.py`)

> **Nota sobre nombres.** En este repositorio existen dos entrenadores de MLP distintos:
>
> | Módulo | Uso |
> |--------|-----|
> | `counterfactuals/cf_mlp_trainer.py` | Entrenador de los experimentos de contrafactuales (el documentado aquí). Se importa como `from cf_mlp_trainer import MLPTrainer`. |
> | `core/mlp_trainer.py` | Entrenador de BIURI/TREPAN Reloaded, con optimización por Grid Search y Optuna y presets de entrenamiento. |
>
> Ambos exponen una clase `MLPTrainer`, pero no son intercambiables. Los pipelines de
> `counterfactuals/pipelines/` usan `cf_mlp_trainer`.

## 1. Introducción

`MLPTrainer` es una clase diseñada para entrenar y gestionar modelos de perceptrón multicapa
(MLP) en el contexto de explicabilidad de modelos (XAI). Su objetivo es proporcionar un flujo de
trabajo unificado y robusto que incluya:

- Preprocesamiento automático de datos (variables numéricas y categóricas).
- Balanceo de clases (submuestreo o sobremuestreo).
- Entrenamiento del MLP con búsqueda de hiperparámetros opcional.
- Calibración del umbral de decisión para clasificación binaria.
- Almacenamiento y carga del modelo completo (incluyendo preprocesadores y codificadores).
- Un mecanismo de `bypass_preprocessing` que permite usar datos ya transformados (útil en
  pipelines de mejora).

Esta clase se utiliza tanto para entrenar el modelo caja negra (MLP) que se va a explicar, como
para integrarse en el flujo de generación de contrafactuales y mejora de árboles sustitutos.

## 2. Estructura y atributos principales

| Atributo | Descripción |
|----------|-------------|
| `label_encoder` | `LabelEncoder` de sklearn para codificar las etiquetas de clase. |
| `pipeline` | Pipeline completo (preprocesador + MLP) cuando se entrena sin búsqueda de hiperparámetros. |
| `preprocessor` | `ColumnTransformer` que aplica escalado a numéricas y one-hot a categóricas. |
| `model` | El modelo MLP entrenado (instancia de `MLPClassifier`). |
| `feature_names` | Lista de nombres de las características originales. |
| `categorical_features_original` | Lista de nombres de características categóricas. |
| `transformed_feature_names` | Nombres de las características después del preprocesamiento (útil para interpretación). |
| `rng` | Generador de números aleatorios con semilla fija para reproducibilidad. |
| `optimal_threshold` | Umbral de decisión calibrado (para clasificación binaria). |
| `bypass_preprocessing` | Flag que, si es `True`, omite el preprocesamiento en `predict` y `predict_proba`. |

## 3. Funcionalidades principales

### 3.1. Balanceo de datos

La clase ofrece dos métodos de balanceo para manejar conjuntos desbalanceados:

- `_balance_data` (sobremuestreo): duplica muestras de la clase minoritaria (con reemplazo) hasta
  igualar el tamaño de la mayoritaria.
- `_undersample_data` (submuestreo): reduce la clase mayoritaria al tamaño de la minoritaria.

Ambos métodos preservan el orden y aplican un barajado final para evitar sesgos.

### 3.2. Preprocesamiento

El preprocesador se construye mediante un `ColumnTransformer` que aplica:

- Escalado a las variables numéricas: por defecto `StandardScaler`, pero se puede cambiar a
  `MinMaxScaler` mediante el parámetro `use_minmax=True`.
- Codificación one-hot a las variables categóricas (con `OneHotEncoder`), manejando categorías
  desconocidas.

Los nombres de las características transformadas se almacenan en `transformed_feature_names` para
facilitar la interpretación posterior. Estos nombres son los que consumen CLEAR (para separar
`num__*` de `cat__*`) y los extractores de árboles.

### 3.3. Entrenamiento del MLP

El método `train` es el núcleo de la clase. Sus parámetros clave son:

| Parámetro | Descripción |
|-----------|-------------|
| `X` | DataFrame de pandas con las características. |
| `y` | Serie o array con las etiquetas. |
| `categorical_features` | Lista de nombres de columnas categóricas (opcional). |
| `mlp_params` | Diccionario con parámetros para `MLPClassifier`. |
| `tune` | Booleano; si `True` realiza búsqueda de hiperparámetros con `GridSearchCV`. |
| `param_grid` | Diccionario con la rejilla de búsqueda (si `tune=True`). |
| `balance_method` | `'oversample'`, `'undersample'` o `'none'`. |
| `use_minmax` | Si `True` usa `MinMaxScaler` en lugar de `StandardScaler`. |
| `calibrate_threshold` | Si `True` y es binario, calcula el umbral óptimo que maximiza la accuracy en el conjunto de entrenamiento. |

El entrenamiento sigue estos pasos:

1. Codifica las etiquetas con `LabelEncoder`.
2. Balancea los datos según el método elegido.
3. Construye el `ColumnTransformer` (preprocesador).
4. Si `tune=True`, realiza una búsqueda con validación cruzada sobre los hiperparámetros definidos
   (por defecto: tamaño de capas, `alpha`, `learning_rate_init`).
5. Entrena el MLP con los parámetros finales (los mejores si se hizo búsqueda).
6. Almacena los nombres de características transformadas.
7. Si `calibrate_threshold=True` y es binario, prueba 99 umbrales entre 0.01 y 0.99 y selecciona el
   que maximiza la accuracy en el conjunto de entrenamiento (si mejora el umbral por defecto 0.5).
8. Devuelve el modelo entrenado, los datos de entrenamiento transformados (sin balancear) y las
   etiquetas codificadas originales.

### 3.4. Predicción y probabilidades

Los métodos `predict` y `predict_proba` aplican el preprocesamiento antes de pasar los datos al
MLP. Si `bypass_preprocessing=True`, se salta el preprocesamiento, asumiendo que los datos ya están
en el formato adecuado. Esto es esencial en el pipeline de mejora, donde los contrafactuales ya
están codificados y no se deben reescalar.

`predict` utiliza el umbral calibrado (`optimal_threshold`) si el problema es binario; en caso
contrario, usa el umbral estándar (0.5).

### 3.5. Transformación manual

El método `transform` permite aplicar el preprocesador a nuevos datos (por ejemplo, para obtener
representaciones codificadas de los contrafactuales). Requiere que los datos sean un DataFrame con
las mismas columnas que el entrenamiento.

### 3.6. Guardado y carga

Los métodos `save` y `load` serializan y deserializan el estado completo del entrenador,
incluyendo el pipeline (si existe), el preprocesador, el modelo, el `LabelEncoder`, los nombres de
características (originales y transformadas), el umbral calibrado y el flag
`bypass_preprocessing`. Esto permite reutilizar el modelo sin necesidad de reentrenar.

## 4. Integración en el pipeline de explicabilidad

1. **Entrenamiento del MLP (modelo caja negra).** Se instancia un `MLPTrainer` y se llama a `train`
   con los datos de entrenamiento. El resultado es un modelo MLP listo para ser explicado.
2. **Generación de contrafactuales (CLEAR, COGS).** Los contrafactuales se generan utilizando el
   modelo MLP y los datos transformados. `transform` proporciona las versiones codificadas de los
   datos originales, que son las que usan los generadores y los extractores.
3. **Extracción de árboles sustitutos (TREPAN, TREPAN Reloaded).** Los extractores reciben el MLP
   (con `bypass_preprocessing=True` para evitar doble transformación) y los datos ya codificados.
   Los nombres de las características transformadas se usan para generar reglas legibles.
4. **Mejora de árboles con contrafactuales.** En `counterfactuals/improve_surrogate.py`, el MLP se
   usa para evaluar los contrafactuales y calcular pesos. `bypass_preprocessing=True` asegura que
   los contrafactuales (que ya están en el espacio transformado) no se preprocesen de nuevo.
5. **Evaluación y persistencia.** Tras la mejora, el nuevo árbol se guarda junto con el MLP
   original, permitiendo reproducir experimentos.

## 5. Ejemplo de uso básico

```python
from counterfactuals._bootstrap import setup

setup()

import pandas as pd
from cf_mlp_trainer import MLPTrainer

X = pd.read_csv('datos.csv')
y = pd.read_csv('etiquetas.csv').squeeze()

trainer = MLPTrainer()
model, X_trans, y_enc = trainer.train(
    X, y,
    categorical_features=['sexo', 'educacion'],
    balance_method='oversample',
    calibrate_threshold=True,
)

# Predecir sobre datos sin transformar (bypass_preprocessing=False por defecto)
predicciones = trainer.predict(nuevos)

trainer.save('mlp_model.pkl')

# Cargar en otro script
trainer2 = MLPTrainer()
trainer2.load('mlp_model.pkl')
```

## 6. Notas sobre `bypass_preprocessing`

Este atributo, `False` por defecto, se activa manualmente cuando se van a usar datos que ya han
pasado por el preprocesador (por ejemplo, en el pipeline de mejora). Su utilidad es:

- Evitar el reescalado y one-hot repetido, que podría alterar los contrafactuales.
- Acelerar las evaluaciones en bucles de optimización.
- Mantener la coherencia en el espacio de características transformado.

Siempre que se cargue un modelo guardado que vaya a usarse en el pipeline de contrafactuales o de
mejora, se debe establecer `trainer.bypass_preprocessing = True`. Los pipelines
(`pipeline_counterfactuals.py`, `pipeline_improve.py`) y `counterfactuals/service.py` ya lo hacen.

## 7. Dependencias y compatibilidad

- **scikit-learn**: `MLPClassifier`, `ColumnTransformer`, `Pipeline`, `GridSearchCV`.
- **pandas** y **numpy**: manejo de datos.
- **joblib**: serialización.

El entrenador está diseñado para funcionar con cualquier conjunto de datos tabular, tanto binario
como multiclase, y se ha probado con los datasets incluidos en los experimentos (Iris, Wine,
German Credit, WDBC, Sonar, Hepatitis).
