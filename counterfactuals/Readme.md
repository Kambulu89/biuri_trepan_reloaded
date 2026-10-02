README – Mejora de Árboles Sustitutos mediante Contrafactuales

## Opção BIURI: Melhorar árvore substituta (V9)

A interface integra agora a melhoria pós-treino do TREPAN Original e do TREPAN
Reloaded com contrafactuais ponderados. Cada árvore usa o seu próprio professor
e feature space. C4.5-Nativo não participa porque é um baseline supervisionado,
e não um substituto do MLP.

O protocolo divide o desenvolvimento em ajuste, seleção e holdout de aceitação,
compara uma reextração sem CF com outra reextração com CF usando a mesma
semente, e só instala a candidata quando o quality gate confirma ganho real sem
degradação relevante. O teste final permanece bloqueado. Consulte
`docs/MELHORIA_ARVORE_SUBSTITUTA_V9.md` para o mapa completo entre a tese, o
código de referência e esta implementação.

## Motor contrafactual interactivo no dataset carregado

Além do fluxo legado CLEAR/COGS usado para melhorar árvores, o projecto inclui
agora uma camada pós-treino isolada:

- `engine.py`: DiCE com fallback interno, CLEAR, CoGS, LORE-Local e
  LORE-Global; restrições, OWL opcional, filtro RST e métricas de qualidade.
- `transfer.py`: portão de acordo P5, categorias formais de transferência,
  robustez local/conjunta e agregação por método no dataset activo.
- `gui/counterfactual_panel.py`: aba para geração, transferência e exportação.

A aplicação nunca escolhe ou combina automaticamente datasets de referência.
Depois de o utilizador carregar e treinar um ARFF, a geração e a avaliação de
transferência P1-P8 usam exclusivamente essa sessão activa.

Os seis datasets incluídos no projecto servem apenas para testes experimentais
P9. O respectivo runner está isolado em `tests/support/` e não é importado pela
GUI nem pelo serviço operacional. Para executar voluntariamente esse teste:

```bash
python run_cf_transfer_tests.py
```

Tanto na aplicação como no teste, o padrão usa 33% das instâncias, semente 42 e
os métodos LORE-Local, CLEAR e CoGS. A biblioteca `dice-ml` é opcional; quando
não está disponível, o motor usa uma busca interna controlada. A análise de
cobertura e as limitações estão em
`docs/COUNTERFACTUAL_IMPLEMENTATION_AUDIT.md`.

1. Introducción
Este proyecto implementa un sistema completo y robusto para la explicabilidad de modelos de caja negra (perceptrones multicapa – MLP) mediante árboles de decisión sustitutos. El sistema aborda dos problemas fundamentales en el campo de la Inteligencia Artificial Explicable (XAI):

Generación de explicaciones interpretables: utilizando los algoritmos Trepan y Trepan Reloaded, extraemos árboles de decisión que aproximan el comportamiento del MLP, proporcionando reglas lógicas comprensibles por humanos.

Mejora de la fidelidad de las explicaciones: mediante la incorporación de contrafactuales (ejemplos sintéticos que cambian la predicción del MLP al modificar mínimamente las entradas) en el proceso de reentrenamiento de los árboles, logramos que estos representen al MLP con mayor precisión, especialmente en regiones de decisión críticas y cerca de las fronteras entre clases.

El sistema está diseñado para ser modular, reproducible y escalable, y viene preconfigurado para seis conjuntos de datos ampliamente utilizados en la literatura (Iris, Wine, German Credit, WDBC, Sonar, Hepatitis), con la posibilidad de añadir nuevos datasets fácilmente.

2. Requisitos e instalación
2.1. Dependencias de Python
Este subsistema forma parte del proyecto TREPAN Reloaded (BIURI) y comparte su entorno. Las dependencias están especificadas en requirements-win.txt en la raíz del proyecto; counterfactuals/requirements-counterfactuals.txt simplemente lo referencia. Para instalarlas, ejecuta desde la raíz:

bash
pip install -r requirements-win.txt
Los paquetes principales son:

Científicos: numpy, pandas, scipy

Machine Learning: scikit-learn (>=1.0)

Deep Learning: tensorflow (>=2.10) – necesario para CLEAR

Modelado Estadístico: statsmodels, jinja2, sympy – necesarios para CLEAR

Ontologías: owlready2 – para Trepan Reloaded

Visualización: graphviz, dtreeviz, matplotlib

Persistencia: joblib

2.2. Dependencia externa: Graphviz
Para la exportación de árboles como imágenes, se necesita el binario de Graphviz instalado en el sistema:

Windows: descarga e instala desde graphviz.org y asegúrate de añadirlo al PATH.

Linux (Ubuntu/Debian): sudo apt-get install graphviz

macOS: brew install graphviz

3. Estructura del proyecto
El subsistema vive en counterfactuals/, dentro del proyecto TREPAN Reloaded. Los extractores de
árboles sustitutos NO están duplicados aquí: se reutilizan los canónicos de core/.

text
trepa_reloaded/
├── run_cf_train.py                        # Fase 1 (envoltorio de línea de comandos)
├── run_cf_counterfactuals.py              # Fase 2 (envoltorio de línea de comandos)
├── run_cf_improve.py                      # Fase 3 (envoltorio de línea de comandos)
├── run_cf_all.py                          # Ejecuta las tres fases + consolidación
├── requirements-win.txt                   # Dependencias del proyecto completo
├── core/                                  # Motor BIURI compartido
│   ├── trepan_extractor.py                # Trepan original (soporta extra_X/extra_y/extra_weights)
│   ├── trepan_reloaded_extractor.py       # Trepan con integración de ontologías
│   ├── mlp_trainer.py                     # Entrenador MLP de BIURI (Grid Search / Optuna)
│   └── improve_surrogate.py               # Reexporta counterfactuals/improve_surrogate.py
├── gui/
│   └── counterfactual_worker.py           # Ejecución asíncrona de las fases desde la GUI
└── counterfactuals/
    ├── service.py                         # API de integración (GUI + línea de comandos)
    ├── dataset_config.py                  # DATASET_CONFIGS, get_dataset_config, MLP_ACCURACY
    ├── _paths.py                          # Resolución central de rutas de artefactos
    ├── _bootstrap.py                      # setup(): sys.path, directorio de trabajo, stdio UTF-8
    ├── cf_mlp_trainer.py                  # Entrenador del MLP de los experimentos de CFs
    ├── improve_surrogate.py               # Lógica central de mejora de árboles
    ├── clear/                             # CLEAR (CFs por regresiones locales)
    │   ├── CLEAR_settings.py
    │   ├── CLEAR_regression.py
    │   ├── CLEAR_perturbations.py
    │   ├── CLEAR_sensitivity_files.py
    │   ├── _pandas_compat.py              # df_append: sustituye DataFrame.append (pandas >= 2)
    │   └── CLEAR.py
    ├── cogs/                              # COGS (CFs por algoritmo evolutivo)
    │   ├── evolution.py
    │   ├── fitness.py
    │   ├── population.py
    │   ├── selection.py
    │   ├── variation.py
    │   ├── distance.py
    │   └── util.py
    ├── pipelines/
    │   ├── pipeline_train.py              # Fase 1: MLP + extracción inicial de árboles
    │   ├── pipeline_counterfactuals.py    # Fase 2: CFs + indicadores de consistencia
    │   └── pipeline_improve.py            # Fase 3: mejora de árboles y evaluación
    ├── analisis/                          # Análisis y consolidación de resultados
    │   ├── evaluar_arboles_originales.py
    │   ├── evaluar_arboles_mejorados.py
    │   └── consolidar_resultados.py
    ├── datasets/                          # Datasets en formato CSV
    │   ├── iris.csv, wine.csv, german_credit.csv, wdbc.csv, sonar.csv, hepatitis.csv
    │   ├── preparar_datasets.py            # Regenera los CSV a partir de los originales UCI
    │   └── originales/                     # Archivos UCI sin procesar (.data)
    ├── experimentos/                      # Resultados organizados por dataset
    │   └── <dataset>/
    │       ├── modelos/                   # Modelos (MLP, árboles originales, estado)
    │       ├── resultados/                # CFs generados e indicadores de consistencia
    │       ├── mejora/                    # Árboles mejorados, estadísticas y métricas
    │       └── training_summary.json
    ├── clear_output/                      # Archivos temporales generados por CLEAR
    ├── resultados_consolidados/           # Resúmenes globales de la Fase 2
    ├── resultados_mejora/                 # Resúmenes globales de la Fase 3
    ├── Readme.md                          # Este documento
    ├── Readme_mlp_trainer.md              # Documentación de cf_mlp_trainer.MLPTrainer
    └── Readme_modif.md                    # Diseño de la mejora y cambios en los extractores
4. Flujo de trabajo general (Arquitectura del sistema)
El sistema se ejecuta en tres fases independientes pero secuenciales. Cada fase genera artefactos que son consumidos por la siguiente, permitiendo una ejecución por etapas y una fácil depuración.

text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ FASE 1: pipeline_train.py                                                                   │
│                                                                                             │
│ 1. Carga el dataset (CSV) y lo divide en entrenamiento y prueba (70/30).                   │
│ 2. Entrena un MLP con MLPTrainer, aplicando:                                                │
│    - Preprocesamiento (escalado de numéricas, one-hot de categóricas).                     │
│    - Balanceo de clases (sobremuestreo o submuestreo).                                     │
│    - Calibración del umbral de decisión (para clasificación binaria).                      │
│ 3. Extrae dos árboles sustitutos usando los extractores Trepan y Trepan Reloaded,          │
│    entrenados sobre datos sintéticos generados por el propio algoritmo.                    │
│ 4. Selecciona aleatoriamente el 33% de las instancias del conjunto de entrenamiento        │
│    (estratificado por clase) para la generación posterior de contrafactuales.              │
│ 5. Guarda todos los modelos, datos codificados, árboles y el estado de la selección.      │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ FASE 2: pipeline_counterfactuals.py                                                          │
│                                                                                             │
│ 1. Carga los modelos y el estado generados en la Fase 1.                                   │
│ 2. Genera contrafactuales (CFs) para las instancias seleccionadas usando dos métodos:      │
│    - CLEAR: basado en regresiones locales (logística) que modelan el comportamiento        │
│      del MLP en el vecindario de la instancia.                                             │
│    - COGS: basado en un algoritmo evolutivo (genético) que busca el CF más cercano         │
│      que cambia la predicción.                                                             │
│ 3. Para cada CF, evalúa su comportamiento en el MLP y en cada uno de los árboles           │
│    originales (Trepan y Trepan Reloaded), clasificándolos en tres categorías:              │
│    - Coincidentes (A): el árbol cambia de clase y coincide con el MLP.                     │
│    - Divergentes (B): el árbol cambia de clase pero NO coincide con el MLP.                │
│    - Brecha (C): el árbol NO cambia de clase (falla completamente).                        │
│ 4. Calcula los indicadores de consistencia (proporciones de A, B y C) para cada            │
│    combinación (método generador × tipo de árbol).                                         │
│ 5. Guarda los CFs (con su vector codificado completo), los indicadores y resúmenes.       │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ FASE 3: pipeline_improve.py                                                                 │
│                                                                                             │
│ 1. Carga los modelos, árboles originales y los CFs generados en las fases anteriores.      │
│ 2. Para cada combinación (método de CF × tipo de sustituto):                               │
│    a. Realiza una búsqueda de hiperparámetros (grid search reducido) sobre 4 parámetros    │
│       clave (max_cfs, cf_types, include_type_C, use_frontier_weight), usando validación    │
│       cruzada estratificada de 2 folds para seleccionar la configuración que maximiza      │
│       la fidelidad en validación.                                                          │
│    b. Con la mejor configuración, reentrena el árbol utilizando improve_surrogate(),       │
│       que incorpora los CFs seleccionados con un sistema de pesos dinámicos:               │
│       - Peso base según tipo (Coincidente, Divergente, Brecha).                            │
│       - Boost adaptativo para los Divergentes (más valiosos para la mejora).              │
│       - Factor de frontera (mayor peso si caen en hojas con alta incertidumbre).           │
│       - Factor de densidad (penalización si están en regiones de baja densidad).           │
│       - Selección de los mejores CFs y combinación con anclas reales (muestras originales).│
│    c. Evalúa el árbol mejorado frente al original en el conjunto de prueba, calculando     │
│       fidelidad (frente al MLP) y precisión (frente a etiquetas reales).                   │
│    d. Guarda el árbol mejorado, estadísticas detalladas y las métricas de evaluación.      │
│ 3. Genera consolidados globales con todos los resultados.                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
5. Configuración de datasets
Los datasets se configuran mediante diccionarios en los archivos pipeline_train.py, pipeline_counterfactuals.py y pipeline_improve.py. La estructura es la misma en los tres, aunque algunos campos son específicos de cada fase.

A continuación se muestra el ejemplo de iris:

python
'iris': {
    'file': 'datasets/iris.csv',          # Ruta al archivo CSV
    'target': 'class',                    # Columna objetivo
    'numeric_features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    'categorical_features': [],           # Columnas categóricas (vacío si no hay)
    'class_labels': {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'},
    'is_multiclass': True,                # Indica si es multiclase (>2 clases)
    'mlp_params': {                       # Parámetros del MLP
        'hidden_layer_sizes': (50, 30),
        'max_iter': 1000,
        'learning_rate_init': 0.01,
        'early_stopping': False,
        'random_state': 42
    },
    'tune_mlp': False                     # Si True, realiza búsqueda de hiperparámetros
}
Campos obligatorios:

file: ruta al archivo CSV.

target: nombre de la columna que contiene las etiquetas.

numeric_features: lista de nombres de columnas numéricas.

categorical_features: lista de nombres de columnas categóricas (puede estar vacía).

class_labels: diccionario que mapea el índice de clase a su nombre (ej. {0: 'pay', 1: 'default'}).

is_multiclass: booleano que indica si el problema tiene más de dos clases.

mlp_params: diccionario con parámetros para MLPClassifier de scikit-learn.

tune_mlp: booleano; si es True, se realiza una búsqueda de hiperparámetros con GridSearchCV.

Campos opcionales (pueden aparecer según el dataset):

balance_method: 'oversample', 'undersample' o 'none' (para manejo de desbalanceo).

param_grid: diccionario para la búsqueda de hiperparámetros del MLP.

tree_sample_size: tamaño de la muestra sintética que usan los extractores de árboles.

clear_num_samples, clear_regression_sample_size, clear_max_predictors: parámetros específicos de CLEAR.

use_minmax_scaler: usar MinMaxScaler en lugar de StandardScaler.

calibrate_threshold: calcular el umbral óptimo de decisión (para clasificación binaria).

6. Ejecución paso a paso
Todos los comandos se ejecutan desde la raíz del proyecto (trepa_reloaded/). Los envoltorios
run_cf_*.py llaman a counterfactuals/service.py, que a su vez invoca los pipelines. Para ejecutar
las tres fases seguidas más la consolidación: python run_cf_all.py [dataset].

6.1. Fase 1: Entrenamiento del MLP y extracción de árboles iniciales
bash
python run_cf_train.py [dataset]
Si no se especifica un dataset, se ejecutan todos (iris, wine, german_credit, wdbc, sonar, hepatitis).

Qué hace:

Carga y preprocesa el dataset.

Entrena el MLP con los parámetros configurados.

Extrae los árboles sustitutos (Trepan y Trepan Reloaded).

Selecciona el 33% de las instancias de entrenamiento (estratificado por clase) para la generación de contrafactuales.

Salidas (en experimentos/<dataset>/modelos/):

mlp_<dataset>.pkl: modelo MLP.

mlp_trainer_<dataset>.pkl: objeto entrenador completo (contiene preprocesador, encoder, umbral).

trepan_<dataset>.pkl y trepan_reloaded_<dataset>.pkl: árboles originales.

X_train_enc_*.pkl, y_train_enc_*.pkl, X_test_enc_*.pkl, y_test_enc_*.pkl: datos codificados.

pipeline_state_<dataset>.pkl: estado de la selección, intervalos de características, etc.

6.2. Fase 2: Generación de contrafactuales e indicadores de consistencia
bash
python run_cf_counterfactuals.py [dataset]
Qué hace:

Carga los artefactos de la Fase 1.

Genera contrafactuales con CLEAR y COGS.

Clasifica los CFs en tres categorías: Coincidentes (A), Divergentes (B) y Brecha (C).

Calcula los indicadores de consistencia (proporciones de A, B y C) para cada combinación (método × árbol).

Guarda los CFs y los resultados.

Salidas (en experimentos/<dataset>/resultados/):

clear_cfs_full.csv y cogs_cfs_full.csv: contienen los CFs con su vector codificado completo (cf_vector), clase original y clase deseada.

consistency_indicators.csv: tabla con los indicadores A, B, C.

summary.json: resumen con cantidades de CFs generados y tiempos de ejecución.

6.3. Fase 3: Mejora de árboles sustitutos
bash
python run_cf_improve.py [dataset]
Qué hace:

Carga modelos, CFs y estado.

Realiza un grid search reducido con validación cruzada para seleccionar la mejor configuración.

Reentrena los árboles utilizando improve_surrogate() con los CFs seleccionados y sus pesos dinámicos.

Evalúa el rendimiento de los árboles mejorados frente a los originales.

Guarda los árboles mejorados, estadísticas y métricas.

Salidas (en experimentos/<dataset>/mejora/):

modelos/: árboles mejorados (.pkl).

stats/: estadísticas de la mejora (.json) – número de CFs usados, pesos promedio, etc.

metricas/: métricas de fidelidad y precisión (.csv) con detalle y resumen.

Adicionalmente, se generan consolidados globales en resultados_mejora/ y resultados_consolidados/.

7. Explicación detallada de los módulos clave
7.1. mlp_trainer.py – Entrenador del MLP
Este módulo proporciona una clase MLPTrainer que encapsula todo el proceso de entrenamiento del modelo de caja negra.

Funcionalidades principales:

Preprocesamiento automático: construye un ColumnTransformer que aplica:

Escalado a características numéricas (por defecto StandardScaler, configurable a MinMaxScaler).

Codificación one-hot a características categóricas (con OneHotEncoder, manejando categorías desconocidas).

Almacena los nombres de las características transformadas (transformed_feature_names).

Balanceo de clases: ofrece dos métodos:

_balance_data (sobremuestreo): duplica muestras de la clase minoritaria hasta igualar el tamaño de la mayoritaria.

_undersample_data (submuestreo): reduce la clase mayoritaria al tamaño de la minoritaria.

Entrenamiento del MLP: el método train() permite:

Especificar parámetros del MLP.

Realizar búsqueda de hiperparámetros con GridSearchCV (si tune=True).

Aplicar balanceo antes del entrenamiento.

Calibrar el umbral de decisión para problemas binarios (optimizando la precisión en el conjunto de entrenamiento).

Predicción y probabilidades: los métodos predict() y predict_proba() aplican el preprocesamiento antes de llamar al MLP. Atributo clave: bypass_preprocessing. Si se establece a True, estos métodos saltan el preprocesamiento, asumiendo que los datos de entrada ya están codificados. Esto es fundamental en las fases 2 y 3, donde los contrafactuales ya se encuentran en el espacio transformado y no deben ser reescalados ni re-codificados.

Persistencia: los métodos save() y load() permiten guardar y restaurar el estado completo del entrenador (modelo, preprocesador, encoder, umbral, etc.).

Integración en el pipeline:

En pipeline_train.py se usa para entrenar el MLP.

En pipeline_counterfactuals.py y pipeline_improve.py, se carga el entrenador guardado y se activa bypass_preprocessing=True para trabajar directamente con los datos codificados.

7.2. improve_surrogate.py – Lógica central de mejora de árboles
Este módulo implementa la función improve_surrogate(), que constituye el núcleo algorítmico de la mejora. Su objetivo es reentrenar un árbol sustituto para que aumente su fidelidad al MLP, utilizando los contrafactuales generados como ejemplos de entrenamiento adicionales con pesos dinámicos.

7.2.1. ¿Por qué es necesario mejorar el árbol?
Los árboles sustitutos extraídos inicialmente (con Trepan) suelen tener una fidelidad aceptable (típicamente >80%), pero fallan en regiones específicas del espacio de características, especialmente cerca de las fronteras de decisión. Los contrafactuales, al ser ejemplos que cruzan esas fronteras, proporcionan información valiosa para corregir estas deficiencias. Sin embargo, no todos los contrafactuales son igualmente útiles, y algunos pueden ser atípicos o ruidosos. Por ello, es crucial filtrarlos, clasificarlos y asignarles pesos adecuados.

7.2.2. Proceso paso a paso de improve_surrogate()
Paso 1: Evaluación y clasificación de los CFs en tres categorías

Para cada contrafactual, se evalúa su predicción con el MLP y con el árbol original. Se considera que un CF es válido si el MLP cambia su clase con respecto a la original y la confianza (probabilidad de la nueva clase) supera un umbral (confidence_threshold). Los CFs válidos se clasifican en:

Coincidentes (A): el árbol original también cambia de clase y la clase predicha por el árbol coincide con la del MLP. Esto indica que el árbol ya explica bien ese CF.

Divergentes (B): el árbol original cambia de clase pero la clase predicha por el árbol no coincide con la del MLP. Son los CFs más valiosos para la mejora, porque revelan regiones donde el árbol se equivoca y, por tanto, donde hay que reforzar el aprendizaje.

Brecha (C): el árbol original no cambia de clase (se queda en la clase original), aunque el MLP sí lo hace. Son CFs que el árbol falla completamente y suelen estar muy cerca de la frontera.

Esta clasificación es fundamental porque cada tipo tiene una utilidad diferente y, por tanto, merece un tratamiento distinto en el cálculo de pesos.

Paso 2: Cálculo de pesos dinámicos para los CFs

Los pesos se calculan combinando varios factores, cada uno con una justificación específica:

Peso base (base_weight_raw): depende del tipo de CF.

Coincidentes (A): 1.0

Divergentes (B): 1.0 (luego se multiplica por boost_B)

Brecha (C): weight_C (por defecto 0.3, ya que son menos fiables)

Boost adaptativo para Divergentes (boost_B): los Divergentes son los más críticos. Si hay pocos (por ejemplo, menos de 10), se les aplica un multiplicador mayor (hasta 3.0) para darles más peso y forzar al árbol a aprender esas regiones. Si hay muchos, el boost se reduce para no distorsionar el resto de los datos. La fórmula es:

python
boost_B = 1.0 + min(3.0, 10.0 / n_B)
Esto asegura que incluso unos pocos Divergentes tengan un impacto significativo en el reentrenamiento.

Factor de frontera (frontier_factor): se calcula a partir de la probabilidad máxima en la hoja del árbol original donde cae el CF (max_prob_hoja). Si el CF cae en una hoja con alta incertidumbre (probabilidad baja), significa que esa región es poco conocida por el árbol. Se le asigna un factor >1 (hasta 1.5) para darle más peso. Si la hoja es muy segura (probabilidad alta), el factor es cercano a 0.5. Esto incentiva al árbol a aprender mejor en las zonas de frontera, donde la fidelidad suele ser más baja. La fórmula es:

python
frontier_factor = 0.5 + (1.0 - max_prob_hoja)
Factor de densidad (density_factor): este factor penaliza a los CFs que se encuentran en regiones de baja densidad de datos reales. La intuición es que un CF aislado, lejos de cualquier muestra real, puede ser un artefacto del generador y no representar un caso realista, por lo que su influencia en el entrenamiento debe reducirse. Para cada clase, se precalculan los vecinos más cercanos (k=2) y se calcula el percentil 95 de las distancias al primer vecino más cercano distinto de sí mismo (dists[:,1]). Si la distancia del CF a su vecino más cercano en su clase original supera ese umbral, se reduce su peso linealmente hasta un mínimo de 0.1.

El peso final de cada CF es:

python
final_weight = base_weight * boost_B (si es B) * frontier_factor * density_factor
Paso 3: Selección de los mejores CFs

Se ordenan los CFs según probabilidad (del MLP) × peso final y se seleccionan los mejores, limitando por:

max_cfs: número máximo de CFs.

max_cf_ratio: proporción máxima respecto al total de muestras de entrenamiento.

Esto evita que el conjunto de mejora esté sobresaturado con CFs de baja calidad o en exceso.

Paso 4: Construcción del conjunto de entrenamiento enriquecido

Se construye un conjunto extra_X, extra_y que combina:

Los CFs seleccionados, con sus pesos individuales.

Un conjunto de anclas (muestras reales tomadas aleatoriamente del entrenamiento original). El número de anclas se determina mediante anchor_ratio (por defecto 20% del total de entrenamiento, con un mínimo de 50). Las anclas tienen peso 1.0.

La inclusión de anclas reales es crucial para mantener la precisión del árbol sobre la distribución real de los datos. Sin ellas, el árbol podría sobreajustarse a los CFs y perder capacidad predictiva sobre los datos originales.

El peso total deseado para los CFs se calcula como:

python
desired_total_cf_weight = cf_weight_ratio * sample_size
donde sample_size es el tamaño de la muestra sintética que usará el extractor (limitado a 5000 para datasets grandes). Este ratio se ajusta dinámicamente según la precisión del MLP (mlp_accuracy): si el MLP es más preciso, se reduce ligeramente el peso de los CFs para evitar sobresaturar.

Los pesos individuales de los CFs se escalan para que su suma sea exactamente desired_total_cf_weight, y luego se recortan a un máximo (max_cf_weight) para evitar que un solo CF domine.

Paso 5: Reentrenamiento del árbol

Finalmente, se llama al método de extracción del extractor (extract_tree o extract_tree_with_ontology) pasando los datos extra y los pesos. Los extractores fueron modificados para aceptar estos parámetros y añadirlos al conjunto de entrenamiento sintético. El árbol se reentrena, y esta vez, los CFs con mayor peso influyen más en la construcción de las particiones y las hojas, lo que resulta en una mayor fidelidad al MLP en esas regiones críticas.

7.2.3. Justificación global del diseño de improve_surrogate()
Enfoque selectivo: no todos los CFs son útiles. Filtrar por confianza y tipo asegura que solo los ejemplos más informativos se utilicen.

Ponderación dinámica: los pesos reflejan la importancia de cada CF para la mejora. Los Divergentes son los más valiosos, y el boost adaptativo compensa su escasez. El factor de frontera y el de densidad añaden un conocimiento adicional sobre la región del espacio donde se encuentra el CF.

Preservación de la precisión: las anclas reales evitan la deriva del árbol hacia regiones artificiales, manteniendo su rendimiento sobre datos reales.

Robustez: el método es estable incluso con grandes volúmenes de datos gracias a la limitación de sample_size y al uso de factores de densidad correctamente calculados.

7.3. pipeline_train.py – Fase 1 (Entrenamiento y extracción inicial)
Responsabilidad:

Cargar el dataset y particionarlo.

Entrenar el MLP con la configuración especificada.

Extraer los árboles sustitutos (Trepan y Trepan Reloaded).

Seleccionar el 33% de las instancias de entrenamiento (estratificado por clase) sin filtrar por coincidencia entre el MLP y los árboles.

Decisión de diseño clave: la selección aleatoria (y no filtrar por coincidencia) aumenta la diversidad de los CFs generados posteriormente. Al incluir tanto instancias donde los árboles aciertan como donde fallan, se obtienen CFs que cubren un espectro más amplio del espacio de características, lo que lleva a una mejora más generalizada.

7.4. pipeline_counterfactuals.py – Fase 2 (Generación de CFs e indicadores)
Responsabilidad:

Cargar los artefactos de la Fase 1.

Generar CFs con CLEAR y COGS.

Clasificar los CFs en Coincidentes, Divergentes y Brecha, calculando los indicadores de consistencia.

Generación de CFs con CLEAR:
CLEAR utiliza regresiones locales (logísticas) para modelar el comportamiento del MLP en el vecindario de cada instancia. Se ejecuta por cada clase objetivo (en multiclase) y utiliza los archivos de sensibilidad generados previamente. Los parámetros de CLEAR (como max_predictors, num_samples, regression_sample_size) están ajustados por dataset para equilibrar calidad y tiempo de ejecución.

Generación de CFs con COGS:
COGS utiliza un algoritmo genético (evolución diferencial) con función de fitness basada en la distancia de Gower y la penalización por no alcanzar la clase deseada. Se ejecuta en paralelo (ThreadPoolExecutor) para cada instancia, obteniendo hasta 3 CFs por instancia.

Indicadores de consistencia (A, B, C):
Estos indicadores miden la capacidad del árbol para explicar los CFs:

A (Coincidentes): proporción de CFs donde árbol y MLP coinciden.

B (Divergentes): proporción de CFs donde el árbol cambia de clase pero difiere del MLP.

C (Brecha): proporción de CFs donde el árbol no cambia de clase.

Un árbol con alta fidelidad tendrá un indicador A alto y B y C bajos.

7.5. pipeline_improve.py – Fase 3 (Mejora sistemática y evaluación)
Responsabilidad:

Cargar los artefactos de las fases anteriores.

Realizar un grid search reducido (16 combinaciones) sobre los parámetros clave de improve_surrogate().

Seleccionar la mejor configuración mediante validación cruzada (2 folds) que maximice la fidelidad en validación.

Reentrenar el árbol sobre todo el entrenamiento con la mejor configuración.

Evaluar el árbol mejorado frente al original en el conjunto de prueba.

Guardar los árboles mejorados, estadísticas y métricas.

Grid search reducido:
Los parámetros que se varían son:

max_cfs: 5 o 10.

cf_types: 'A+B' (solo coincidentes y divergentes) o 'all' (incluye brecha).

include_type_C: True o False.

use_frontier_weight: True o False.

Otros parámetros (como confidence_threshold, cf_weight_ratio, max_cf_weight, etc.) se mantienen fijos con valores estables y bien justificados. Esto reduce el coste computacional y evita el sobreajuste en la selección de hiperparámetros.

8. Resultados y artefactos generados
8.1. Indicadores de consistencia (Fase 2)
En experimentos/<dataset>/resultados/consistency_indicators.csv se encuentran:

indicador_a: proporción de CFs Coincidentes (A).

indicador_b: proporción de CFs Divergentes (B).

indicador_c: proporción de CFs en Brecha (C).

total_CFs_validos_MLP: número de CFs que cambian la clase en el MLP.

total_CFs_validos_sustituto: número de CFs que cambian la clase en el árbol.

Estos indicadores permiten evaluar la calidad inicial de los árboles sustitutos y sirven como referencia para medir la mejora posterior.

8.2. Métricas de mejora (Fase 3)
En experimentos/<dataset>/mejora/metricas/ se encuentran:

fidelity_original y fidelity_improved: precisión de las predicciones del árbol frente al MLP en el conjunto de prueba.

accuracy_original y accuracy_improved: precisión del árbol frente a las etiquetas reales.

relative_improvement: mejora relativa de la precisión ((acc_imp - acc_orig) / acc_orig).

used: número de CFs realmente utilizados en la mejora.

type_C_used: número de CFs de tipo Brecha utilizados (si se incluyeron).

density_discarded: número de CFs penalizados por baja densidad.

frontier_factor_mean: factor de frontera promedio de los CFs usados.

8.3. Consolidados globales
resultados_consolidados/consistency_results_all.csv: todos los indicadores de consistencia de todos los datasets.

resultados_mejora/detalle_mejora.csv: todas las métricas de mejora de todos los experimentos.

resultados_mejora/mejora_resumen.csv: resumen por combinación de parámetros (medias y desviaciones estándar).

9. Análisis y consolidación de resultados (carpeta analisis/)
La carpeta analisis/ contiene scripts auxiliares diseñados para evaluar, consolidar y visualizar los resultados experimentales. Estos scripts no son necesarios para ejecutar el flujo de trabajo principal, pero resultan esenciales para el análisis posterior y la generación de tablas.

9.1. evaluar_arboles_originales.py
Propósito:
Calcula la fidelidad (coincidencia con las predicciones del MLP) y la precisión (coincidencia con las etiquetas reales) de los árboles sustitutos originales (Trepan y Trepan Reloaded) sobre el conjunto de prueba.

Entradas:

Artefactos guardados en experimentos/<dataset>/modelos/:

X_test_enc_*.pkl, y_test_enc_*.pkl

mlp_*.pkl (con bypass_preprocessing=True)

trepan_*.pkl y trepan_reloaded_*.pkl

Salidas:

Archivo analisis/evaluacion_arboles_originales.json con un diccionario por dataset que incluye fidelidad y precisión de cada sustituto.

Tabla impresa en consola.

Uso:

bash
python analisis/evaluar_arboles_originales.py
9.2. evaluar_arboles_mejorados.py
Propósito:
Similar al anterior, pero aplicado a los árboles mejorados generados en la Fase 3. Lee todos los .pkl de experimentos/<dataset>/mejora/modelos/, extrae el árbol (ya sea directamente o desde un extractor) y calcula fidelidad y precisión en el conjunto de prueba.

Entradas:

Los mismos datos de prueba que el script anterior.

Todos los árboles mejorados guardados en experimentos/<dataset>/mejora/modelos/.

Salidas:

Archivo analisis/evaluacion_arboles_mejorados.json con una entrada por cada archivo de árbol mejorado, indicando su fidelidad y precisión.

Tabla impresa en consola con los resultados por archivo.

Uso:

bash
python analisis/evaluar_arboles_mejorados.py
9.3. consolidar_resultados.py
Propósito:
Es la herramienta central para reunir y estructurar todos los resultados experimentales en un solo lugar. Genera archivos CSV limpios que contienen:

Indicadores de consistencia (coincidencia, divergencia y brecha) consolidados de todos los datasets.

Fidelidad y precisión de los árboles originales.

Fidelidad y precisión de los árboles mejorados (agregados por combinación de método CF y sustituto).

Mejora relativa (en puntos porcentuales) de fidelidad y precisión.

Una tabla maestra con las métricas más relevantes (precisión del MLP, fidelidades originales, número de coincidencias, etc.).

Entradas:

Archivos de consistencia: experimentos/<dataset>/resultados/consistency_indicators.csv.

Modelos y árboles de experimentos/<dataset>/modelos/.

Árboles mejorados de experimentos/<dataset>/mejora/modelos/.

Salidas:

Todos los CSV se guardan en una carpeta analisis/consolidar_resultados/:

consistencia_completa.csv

arboles_originales.csv

arboles_mejorados.csv

mejora_completa.csv

tabla_resumen.csv

consistencia_promedio.csv

Uso:

bash
python analisis/consolidar_resultados.py
9.4. Nota sobre la ejecución
Estos scripts deben ejecutarse desde la raíz del proyecto (donde se encuentra la carpeta experimentos/). De esta forma, las rutas relativas a los datos de entrada son correctas. Los scripts ya están configurados para guardar sus salidas dentro de la carpeta analisis/. Si se mueven a otra ubicación, será necesario ajustar las rutas de lectura/escritura.

10. Decisiones de diseño clave y justificación
Decisión	Justificación
Selección del 33% de instancias aleatoriamente (no por coincidencia)	Aumenta la diversidad de los CFs, cubriendo regiones donde los árboles ya aciertan y donde fallan, mejorando la generalización de la mejora.
Uso de bypass_preprocessing=True en el MLP cargado	Evita re-aplicar escalado/one-hot a datos ya transformados, manteniendo coherencia en el espacio de características y evitando errores numéricos.
Guardar cf_vector completo en CLEAR	Evita reconstrucciones frágiles basadas en nombres de columnas; el vector codificado es directamente utilizable en fases posteriores.
Clasificación en Coincidentes (A), Divergentes (B) y Brecha (C)	Permite un tratamiento diferenciado de los CFs según su utilidad para la mejora. Los Divergentes son los más valiosos, y los de Brecha son los menos fiables.
Factor de densidad basado en la distancia al vecino más cercano	La distancia al vecino más cercano (distinto de sí mismo) es la métrica correcta para estimar densidad local. Esto permite penalizar CFs atípicos y mejorar la robustez.
boost_B adaptativo para Divergentes	Los Divergentes son críticos para la mejora. Si hay pocos, se les da más peso para forzar su aprendizaje; si hay muchos, se reduce el boost para no distorsionar el resto de los datos.
Factor de frontera (incertidumbre en la hoja)	Las hojas con baja confianza indican regiones de decisión inciertas. Dar más peso a CFs en esas zonas mejora la fidelidad global del árbol.
Inclusión de anclas reales	Mantiene la precisión en datos reales, evitando que el árbol se desvíe excesivamente hacia los CFs y pierda capacidad predictiva.
Grid search reducido y validación cruzada	Equilibrio entre coste computacional y calidad de la configuración; la validación cruzada evita sobreajuste y selecciona parámetros que generalizan bien.
Limitación de sample_size a 5000	Para datasets grandes, un tamaño sintético muy grande ralentiza el entrenamiento sin mejorar significativamente la calidad. 5000 es un compromiso razonable.
Parámetros de CLEAR ajustados por dataset	Datasets más grandes o complejos (ej. German Credit) necesitan menos muestras y regresiones más rápidas. Esto optimiza el tiempo de ejecución sin sacrificar la calidad de los CFs.
11. Personalización y extensión
11.1. Añadir un nuevo dataset
Coloca el archivo CSV en la carpeta datasets/.

Añade una entrada al diccionario DATASET_CONFIGS en pipeline_train.py, pipeline_counterfactuals.py y pipeline_improve.py (al menos en los dos primeros para que el flujo completo funcione).

Define correctamente los campos: características numéricas, categóricas, etiquetas, y parámetros del MLP.

Ejecuta las tres fases en orden.

11.2. Ajustar parámetros de CLEAR
En la configuración del dataset, puedes añadir o modificar:

clear_num_samples: número de muestras sintéticas para CLEAR.

clear_regression_sample_size: tamaño de la vecindad para la regresión local.

clear_max_predictors: número máximo de predictores en la regresión.

Estos parámetros afectan directamente a la calidad y velocidad de generación de CFs.

11.3. Modificar la lógica de mejora (improve_surrogate.py)
Puedes ajustar:

cf_weight_ratio: peso total deseado para los CFs (por defecto 0.05).

max_cf_weight: peso máximo individual de un CF (por defecto 20.0).

weight_C: peso base para CFs de Brecha (por defecto 0.3).

anchor_ratio: proporción de anclas reales (por defecto 0.2).

class_density_percentile: percentil para el umbral de densidad (por defecto 95).

También puedes añadir nuevos factores de peso o modificar los existentes (ej. añadir un factor basado en la distancia al centroide de la clase).

11.4. Cambiar el extractor de árboles
Los extractores se encuentran en core/ (trepan_extractor.py y trepan_reloaded_extractor.py). Puedes añadir nuevos extractores siempre que implementen los métodos extract_tree y/o extract_tree_with_ontology y soporten los parámetros extra_X, extra_y, extra_weights. Luego, modifica counterfactuals/pipelines/pipeline_improve.py para que reconozca el nuevo extractor.

12. Referencias
CLEAR: “Counterfactual Local Explanations via Regression” – implementación adaptada.

COGS: Método basado en evolución diferencial para generación de contrafactuales.

Trepan: “Extracting Decision Trees from Neural Networks” – algoritmo original para extracción de árboles sustitutos.

Trepan Reloaded: Extensión de Trepan con integración de ontologías para enriquecer las explicaciones semánticas (basado en owlready2).

13. Licencia
Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

14. Contacto
Para preguntas, sugerencias o reportes de errores, abre un issue en el repositorio o contacta al autor principal.

¡Gracias por utilizar este sistema de mejora de árboles sustitutos mediante contrafactuales! Con esta documentación, esperamos que puedas comprender, ejecutar y extender el proyecto para tus propias investigaciones en el campo de la Inteligencia Artificial Explicable.
