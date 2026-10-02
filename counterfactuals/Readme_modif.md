> **DOCUMENTO HISTÓRICO/OBSOLETO.** Não descreve a build V9.2 de produção. A produção usa exclusivamente TREPAN histórico/Reloaded histórico e C4.5-Nativo; snippets antigos de árvores sklearn abaixo são apenas registo histórico e não devem ser executados.

# Mejora de árboles sustitutos mediante contrafactuales

Este documento reúne los dos análisis de diseño del subsistema de mejora:

1. La lógica central de mejora (`counterfactuals/improve_surrogate.py` y
   `counterfactuals/pipelines/pipeline_improve.py`).
2. Las modificaciones aplicadas a los extractores TREPAN y TREPAN Reloaded para hacer posible esa
   mejora.

> **Ubicación de los extractores en este repositorio.** Los extractores que utilizan los pipelines
> de contrafactuales son los canónicos de BIURI: `core/trepan_extractor.py` y
> `core/trepan_reloaded_extractor.py`. Son ellos los que reciben `extra_X`, `extra_y` y
> `extra_weights` tal como se describe en la parte II.

---

# Parte I — Lógica central de mejora

## 1. Introducción

Este subsistema implementa un método de mejora de árboles sustitutos (*surrogate trees*) utilizando
contrafactuales (CFs) generados por otros métodos (CLEAR o COGS). El objetivo es aumentar la
fidelidad del árbol con respecto al modelo caja negra (MLP) y, al mismo tiempo, mantener o mejorar
su precisión predictiva, especialmente en regiones de decisión críticas.

Para ello se emplean dos módulos:

- `counterfactuals/improve_surrogate.py` — contiene la lógica central de mejora: filtrar CFs,
  calcular pesos, construir un conjunto de entrenamiento enriquecido y reentrenar el árbol.
- `counterfactuals/pipelines/pipeline_improve.py` — orquesta experimentos sistemáticos sobre
  múltiples datasets, realiza búsqueda de hiperparámetros (grid search) con validación cruzada, y
  almacena los árboles mejorados y sus estadísticas.

Los extractores se utilizan como base para obtener el árbol original y para el reentrenamiento,
pero su núcleo algorítmico no se modifica. La mejora se logra inyectando muestras adicionales (los
CFs) con pesos dinámicos durante la extracción del árbol.

## 2. `improve_surrogate.py`

### 2.1. Función `improve_surrogate`

Recibe el modelo MLP, los datos de entrenamiento, el extractor (que contiene el árbol original),
una lista de contrafactuales y múltiples parámetros de control. Retorna el nuevo árbol mejorado y
un diccionario de estadísticas.

#### 2.1.1. Filtrado y clasificación de contrafactuales

No todos los CFs generados son útiles. Algunos pueden ser inválidos (no cambian la clase predicha
por el MLP), tener baja confianza o no ser coherentes con el árbol original. Se distinguen tres
tipos de CFs según su comportamiento en el árbol original:

| Tipo | Comportamiento |
|------|----------------|
| **A** (coincidente) | El árbol original ya clasifica el CF con una clase diferente a la original y coincide con el MLP. Son CFs que el árbol ya explica bien. |
| **B** (divergente) | El árbol original también cambia de clase, pero no coincide con la clase del MLP. Son CFs que el árbol no explica correctamente; mejorarlos es clave para aumentar la fidelidad. |
| **C** (brecha) | El árbol original no cambia de clase, aunque el MLP sí lo hace. Son CFs que el árbol falla por completo. |

Se evalúa cada CF con el MLP y con el árbol original, se calcula la probabilidad de la clase
predicha por el MLP (`predict_proba`) y se aplica un umbral de confianza
(`confidence_threshold`). Solo los CFs con confianza suficiente y que cambien la clase en el MLP se
consideran. Luego se clasifican y se filtran según los parámetros `cf_types` (`'A'`, `'B'`,
`'A+B'`, `'all'`) e `include_type_C`.

#### 2.1.2. Cálculo de pesos y factores de ajuste

- **Factor de frontera (`frontier_factor`).** Se calcula a partir de la probabilidad máxima en la
  hoja del árbol original. Si el CF cae en una hoja con alta incertidumbre (probabilidad baja), se
  le asigna un factor > 1 para darle más peso, incentivando que el árbol aprenda mejor esas
  regiones. Si la hoja es muy segura, el factor es cercano a 0.5. Las regiones de frontera de
  decisión son las más difíciles y donde la fidelidad suele ser más baja.
- **Boost adaptativo para CFs tipo B (`boost_B`).** Los CFs tipo B son especialmente valiosos porque
  el árbol original se equivoca en ellos. Se calcula de forma adaptativa: si hay pocos CFs tipo B,
  se les da un boost mayor (hasta 3.0) para compensar su escasez; si hay muchos, el boost se reduce.
  Esto evita que un único CF tipo B tenga un peso desproporcionado.
- **Factor de densidad (`density_factor`).** Penaliza CFs situados en regiones de baja densidad de
  datos reales, bajo la premisa de que los CFs en zonas muy aisladas pueden ser artefactos del
  generador y no representar casos realistas. Se precomputan vecinos cercanos por clase con
  `NearestNeighbors` y se calcula el percentil 95 de las distancias al vecino más cercano
  (excluyendo el propio punto). Si la distancia del CF a su vecino más cercano en la clase original
  supera ese umbral, se reduce su peso (hasta un 10 % del original).

> **Corrección importante.** En versiones anteriores se usaba `dists[:, 0]` (distancia a sí mismo,
> siempre 0) en lugar de `dists[:, 1]`. Se corrigió para usar la distancia al vecino más cercano
> distinto de sí mismo, que es la métrica correcta para estimar densidad local.

#### 2.1.3. Limitación y selección de CFs

Se impone un límite superior al número de CFs usados (`max_cfs`) y a la proporción respecto al total
de muestras (`max_cf_ratio`). Luego se ordenan los CFs por `probabilidad × peso` y se seleccionan
los mejores. Esto garantiza que el conjunto de mejora no esté dominado por CFs de baja calidad ni
por un exceso de ellos, manteniendo el equilibrio con los datos reales.

#### 2.1.4. Construcción del conjunto de entrenamiento enriquecido

Se construye un conjunto `extra_X` / `extra_y` que combina:

- Los CFs seleccionados (con sus pesos individuales).
- Un conjunto de **anclas**: muestras reales tomadas aleatoriamente del entrenamiento original. Esto
  asegura que el árbol no se aleje demasiado de la distribución real.

El peso total deseado para los CFs se calcula como `cf_weight_ratio * sample_size`, donde
`sample_size` es el tamaño de la muestra sintética que usará el extractor (limitado a 5000 para
datasets grandes). Este ratio se ajusta dinámicamente según la precisión del MLP (`mlp_accuracy`):
si el MLP es más preciso, se reduce ligeramente el peso de los CFs para no sobresaturar, ya que el
árbol ya se acerca al MLP.

Los pesos individuales se escalan para que su suma sea exactamente el peso total deseado, y luego se
recortan a un máximo (`max_cf_weight`) para evitar que un solo CF domine.

#### 2.1.5. Reentrenamiento del árbol

Finalmente se llama al método de extracción del extractor (`extract_tree` o
`extract_tree_with_ontology`) pasando los datos adicionales y los pesos. El extractor genera nuevos
datos sintéticos (como hace normalmente) pero añade las muestras extra con los pesos especificados,
lo que influye en el entrenamiento del nuevo árbol.

### 2.2. Función `evaluate_improvement`

Calcula métricas de fidelidad (accuracy entre MLP y árbol) y precisión (contra las etiquetas reales)
para el árbol original y el mejorado. Se usa para cuantificar el impacto de la mejora.

## 3. `pipeline_improve.py` — Orquestación de experimentos

### 3.1. Carga de artefactos

`cargar_artefactos` recupera el `MLPTrainer` (con `bypass_preprocessing=True`), los datos
codificados (entrenamiento y prueba) y los árboles originales (Trepan y Trepan Reloaded) guardados
en `counterfactuals/experimentos/<dataset>/modelos/`. También obtiene los nombres de características
y clases.

### 3.2. Carga de contrafactuales

`cargar_cfs` lee los CFs desde el CSV generado por CLEAR o COGS. Si existe la columna `cf_vector`,
la usa directamente (evitando parseos de strings); en caso contrario hace un fallback al parseo de
la columna `cf`. Esto hace que la carga sea más robusta y eficiente.

### 3.3. Precomputación de vecinos por clase

Para el cálculo de densidad se precomputan los vecinos más cercanos y el umbral del percentil 95
para cada clase del conjunto de entrenamiento. Se hace una sola vez por dataset, evitando
recálculos en cada iteración del grid search.

### 3.4. Grid search reducido

Se definen 16 combinaciones de hiperparámetros, fijando los valores más estables
(`confidence_threshold=0.6`, `cf_weight_ratio=0.05`, `max_cf_weight=20.0`, etc.). La búsqueda se
realiza sobre:

| Parámetro | Valores | Sufijo en los nombres de archivo |
|-----------|---------|----------------------------------|
| `max_cfs` | 5, 10 | `max5` / `max10` |
| `cf_types` | `'A+B'`, `'all'` | `cf_A+B` / `cf_all` |
| `include_type_C` | `False`, `True` | `CFalse` / `CTrue` |
| `use_frontier_weight` | `False`, `True` | `FFalse` / `FTrue` |

Un grid search completo sería computacionalmente costoso. Se han seleccionado estos parámetros
porque son los que más afectan la calidad de la mejora, mientras que otros (`boost_B`, `weight_C`,
`anchor_ratio`) se dejan fijos con valores razonables.

### 3.5. Validación cruzada para seleccionar la mejor configuración

Para cada combinación se realiza una validación cruzada estratificada de 2 folds sobre el conjunto
de entrenamiento. En cada fold se entrena un árbol temporal y se mejora con los CFs
correspondientes, midiendo la fidelidad en el fold de validación. La combinación con mayor fidelidad
promedio se selecciona como la mejor para ese dataset, método de CF y tipo de sustituto. La
validación cruzada evita el sobreajuste a una partición concreta.

### 3.6. Entrenamiento final y guardado de resultados

Con la mejor configuración se entrena el árbol mejorado sobre todo el conjunto de entrenamiento y se
evalúa en el conjunto de prueba. Se guardan:

- El árbol mejorado (`.pkl`) en `experimentos/<dataset>/mejora/modelos/`.
- Las estadísticas de la mejora (`.json`) en `experimentos/<dataset>/mejora/stats/`.
- Las métricas detalladas y su resumen agrupado en `experimentos/<dataset>/mejora/metricas/`.

Además se genera un consolidado global en `counterfactuals/resultados_mejora/detalle_mejora.csv` y
`counterfactuals/resultados_mejora/mejora_resumen.csv`.

## 4. Justificación de las decisiones clave

| Decisión | Razón |
|----------|-------|
| `dists[:, 1]` en el cálculo de densidad | `dists[:, 0]` es la distancia de cada punto a sí mismo (siempre 0), lo que hacía que el factor de densidad fuera siempre 1 y nunca se penalizaran CFs atípicos. |
| Límite de `sample_size` a 5000 | Para datasets grandes el tamaño sintético por defecto puede ser muy alto, lo que ralentiza el entrenamiento y puede provocar desequilibrios. Se mantiene un mínimo de 3000 para datasets pequeños. |
| `logging` en lugar de `print` | Permite un control más fino de la salida (niveles de severidad) y es más fácil de redirigir o deshabilitar. |
| Anotaciones de tipo | Mejora la legibilidad y facilita la detección de errores en IDEs y análisis estático. |
| Boost adaptativo para tipo B | Si hay pocos CFs tipo B se les da más peso para que el nuevo árbol los aprenda; si hay muchos se reduce el boost para no distorsionar el resto de los datos. |
| Factor de frontera | Las regiones donde el árbol tiene baja confianza son las más difíciles y donde la fidelidad suele fallar. |
| Anclas reales | Evitan que el árbol se desvíe de la distribución original, manteniendo la precisión sobre datos reales. Se controla con `anchor_ratio`. |
| Grid search reducido + CV | Encuentra una buena configuración sin coste computacional excesivo, garantizando que generalice. |
| Almacenamiento estructurado | Facilita el análisis posterior y la comparación entre métodos, datasets y configuraciones. |

---

# Parte II — Modificaciones en TREPAN y TREPAN Reloaded

## 1. Contexto y motivación

Los extractores `TREPANExtractor` y `TrepanReloadedExtractor` se modificaron para integrarse en el
pipeline de mejora de árboles sustitutos mediante contrafactuales. Dicho pipeline necesita inyectar
muestras adicionales (los CFs) con pesos individualizados durante el entrenamiento del árbol, para
así aumentar la fidelidad al MLP sin perder precisión en los datos reales.

Las modificaciones son mínimas pero esenciales, y mantienen total compatibilidad con el código
existente que no utiliza estas funcionalidades.

## 2. Cambios en `trepan_extractor.py` (TREPAN original)

### 2.1. Firma del método `extract_tree`

Antes:

```python
def extract_tree(self, mlp_model, X_encoded, y_encoded, sample_size=2000,
                 feature_names=None, class_names=None):
```

Después:

```python
def extract_tree(self, mlp_model, X_encoded, y_encoded, sample_size=2000,
                 feature_names=None, class_names=None,
                 extra_X=None, extra_y=None, extra_weights=None):
```

Se añaden tres parámetros opcionales:

- `extra_X`: array de muestras adicionales (por ejemplo, contrafactuales).
- `extra_y`: etiquetas correspondientes a esas muestras.
- `extra_weights`: pesos individuales (flotantes) para cada muestra extra.

Esto permite al pipeline pasar los CFs seleccionados y sus pesos calculados dinámicamente (según
tipo, densidad, frontera, etc.) sin modificar la lógica interna del extractor más allá de incorporar
estos datos al conjunto de entrenamiento.

### 2.2. Incorporación de muestras adicionales

Bloque insertado justo después de la generación de datos sintéticos:

```python
# Fase 2: Añadir muestras adicionales si se proporcionan
n_extra = 0
if extra_X is not None and extra_y is not None:
    extra_X = np.asarray(extra_X, dtype=X_synthetic.dtype)
    extra_y = np.asarray(extra_y, dtype=y_synthetic.dtype)
    n_extra = extra_X.shape[0]
    if n_extra > 0:
        X_synthetic = np.vstack([X_synthetic, extra_X])
        y_synthetic = np.hstack([y_synthetic, extra_y])

# Construir pesos si se proporcionan
if extra_weights is not None:
    base_weights = np.ones(len(X_synthetic) - n_extra)
    sample_weight = np.hstack([base_weights, extra_weights])
else:
    sample_weight = None
```

Las muestras sintéticas originales tienen peso 1.0 y las extra reciben los pesos proporcionados (ya
normalizados o escalados externamente).

### 2.3. Propagación de `sample_weight` al entrenamiento

```python
# Antes
def _train_trepan_tree(self, X_synthetic, y_synthetic, X_real, y_real):
    tree = DecisionTreeClassifier(...)
    tree.fit(X_synthetic, y_synthetic)

# Después
def _train_trepan_tree(self, X_synthetic, y_synthetic, X_real, y_real, sample_weight=None):
    tree = DecisionTreeClassifier(...)
    tree.fit(X_synthetic, y_synthetic, sample_weight=sample_weight)
```

Permite que el peso de cada muestra influya en la construcción del árbol (cálculo de impurezas y
particiones), de modo que los CFs con mayor peso tengan más impacto en la forma final del árbol.

## 3. Cambios en `trepan_reloaded_extractor.py` (TREPAN con ontología)

Los cambios son idénticos en estructura, aplicados a los métodos correspondientes.

### 3.1. Firmas

```python
def extract_tree(self, mlp_model, X_encoded, y_encoded, sample_size=2000,
                 feature_names=None, class_names=None,
                 extra_X=None, extra_y=None, extra_weights=None):

def extract_tree_with_ontology(self, mlp_model, X_encoded, y_encoded,
                               feature_names, class_names, sample_size=2000,
                               extra_X=None, extra_y=None, extra_weights=None):
```

### 3.2. Entrenamiento con restricciones ontológicas

```python
# Antes
def _train_ontology_constrained_tree(self, X_synthetic, y_synthetic, X_real, y_real,
                                     feature_names, class_names):
    tree.fit(X_synthetic, y_synthetic)

# Después
def _train_ontology_constrained_tree(self, X_synthetic, y_synthetic, X_real, y_real,
                                     feature_names, class_names, sample_weight=None):
    tree.fit(X_synthetic, y_synthetic, sample_weight=sample_weight)
```

## 4. Justificación global

| Cambio | Razón |
|--------|-------|
| Parámetros `extra_X`, `extra_y`, `extra_weights` | Permitir inyectar contrafactuales (u otras muestras externas) junto con sus pesos, sin modificar el flujo principal del extractor. |
| Concatenación de muestras extra | Aprovechar la infraestructura existente de generación de datos sintéticos y simplemente añadir los CFs al conjunto de entrenamiento. |
| Construcción de `sample_weight` | Asignar pesos individuales a cada muestra, para que el pipeline otorgue más importancia a CFs críticos (tipo B, frontera) y menos a CFs atípicos o de baja confianza. |
| Paso de `sample_weight` al `fit` | Hacer que los pesos influyan realmente en la construcción del árbol (ganancia de información y partición). |
| Compatibilidad hacia atrás | Todos los nuevos parámetros son opcionales (`None` por defecto). Si no se usan, el comportamiento es idéntico a la versión original. |

## 5. Impacto en el pipeline de mejora

Gracias a estas modificaciones, `improve_surrogate.py` puede seleccionar los CFs más relevantes
(según tipo, confianza, densidad y frontera), calcular un peso dinámico para cada uno, pasarlos al
extractor y obtener un árbol mejorado que aumenta la fidelidad al MLP sin perder precisión en los
datos reales.

Sin ellas sería necesario modificar directamente los datos sintéticos o reimplementar el extractor,
lo cual sería más costoso y menos mantenible.

## 6. Conclusión

Las modificaciones son mínimas, no invasivas y totalmente compatibles hacia atrás. Demuestran la
flexibilidad del diseño original de TREPAN, que con pequeñas adiciones puede adaptarse a nuevos
requisitos sin reescribir su núcleo algorítmico.
