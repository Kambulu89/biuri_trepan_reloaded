## 2 July 2021 Update

CLEAR has been updated to now include penalized logistic regression, and to also improve its handling of dummy variables. Nevertheless the recommended default for CLEAR remains to use (non-penalized) logistic regression with AIC.  The parameters specifying numeric/categorical data are now contained in CLEAR_settings.py . A second update will follow shortly in which CLEAR will switch from Tensorflow to PyTorch. Please let me know via email if there you have any requests for improved functionality. A new CLEAR paper has been published which applies CLEAR to images https://arxiv.org/abs/2106.14556 ; the code for this will be uploaded to GitHub once the paper is published.

# Counterfactual Local Explanations via Regression (CLEAR)

CLEAR explains single predictions of machine learning classifiers. It is based on the view that a satisfactory explanation of a single prediction needs to both
explain the value of that prediction and answer ’what-if-things-had-been-different’ questions. In doing this it needs to state the relative importance of the input features and show how they interact. A
satisfactory explanation must also be measurable and state how well it can explain a model. It *must know when it does not know*

Please note CLEAR is designed to explain an AI system's classification probabilities NOT classification classes

### Prerequisites

CLEAR is written in Python 3.7. It runs on Windows 10.The clear.yml file specifies CLEAR's dependencies .

### Installation

Download a copy of the CLEAR repository into a new directory on your PC. The file CLEAR_settings.py contains the parameter variables for CLEAR. Open CLEAR_settings and change the value of parameter *CLEAR_path* to the name of the directory you have created for CLEAR e.g. CLEAR_path='D:/CLEAR/'

### Running CLEAR

CLEAR's parameters for the experiment should first be set. These are all in CLEAR_settings.py. The admissible values for each parameter are shown in the comment to the right of the parameter eg for *case_study* the admissible values are 'Census','PIMA Indians Diabetes','Credit Card','BreastC'. The pdf file 'Input parameters for CLEAR' documents the input parameters.

CLEAR is then run by running CLEAR.py. The user has two options:
(a) run one of the sample models/datasets provided in CLEAR_sample_models_datasets.py .To do this CLEAR.py should include the command Run_CLEAR_with_sample_model()
(b) run CLEAR with a user created model and dataset. To do this CLEAR.py needs to have details of the user model and also include a command to run Run_CLEAR(). For example:
```python
if __name__ == "__main__":
    X_train = pd.read_pickle('D:/CLEAR/X_train_Adult')
    X_test_sample = pd.read_pickle('D:/CLEAR/X_test_sample_Adult')
    model = tf.keras.models.load_model('D:/CLEAR/CLEAR_Adult.h5')

```

CLEAR will generate a report explaining a single prediction if the parameters (in CLEAR.py) 'first_obs' and 'last_obs' are set to the same value e.g first_obs=7, last_obs=7 will generate a report explaining observation 7 in the test dataset. The report is entitled 'CLEAR_prediction_report.html'

There are two detailed csv files created for each run. The first file's name consist of the characters 'CLRreg_' and the date/time it was created eg 'CLRreg_20190522-1618.csv' This contains details of the regression for each observation e.g. adjusted R-squared score, coefficient weights and so forth. The second file's name consists of the characters 'wPerturb_' and the date time. This contains details of each b-perturbation for each observation. A error histogram is also created for each run, the name consisting of characters 'Hist_' and the date/time.
 
 # CLEAR - Counterfactual Local Explanations via Regression

Este repositorio contiene una versión actualizada y robustecida del código CLEAR original entregada, con importantes mejoras en estabilidad, portabilidad y corrección de errores.

---

## Novedades respecto a la versión original

La versión original de CLEAR (publicada en julio de 2021) fue un hito en la explicabilidad de modelos mediante regresiones locales y contrafactuales. Sin embargo, con el tiempo se identificaron ciertas limitaciones que han sido abordadas en esta nueva versión:

### 1. Eliminación de la dependencia de `tkinter`
- **Cambio**: Se ha eliminado la importación de `tkinter` en `CLEAR_settings.py`. Los errores de configuración ahora se muestran por consola en lugar de en ventanas emergentes.
- **Justificación**: `tkinter` requiere un entorno gráfico, lo que impedía ejecutar CLEAR en servidores sin GUI, notebooks Jupyter o sistemas Linux sin X11.
- **Implicación**: CLEAR ahora es completamente ejecutable en cualquier entorno Python, incluidos servidores remotos y contenedores Docker.

### 2. Conversión a paquete instalable
- **Cambio**: Todas las importaciones internas son ahora relativas (ej. `from . import CLEAR_settings`), y se ha añadido un `__init__.py` en el directorio raíz.
- **Justificación**: La estructura original asumía que todos los módulos estaban en el mismo directorio, lo que dificultaba su integración como biblioteca.
- **Implicación**: Ahora CLEAR puede instalarse con `pip install .` y usarse como cualquier otro paquete Python, facilitando su reutilización en proyectos más grandes.

### 3. Manejo robusto de archivos de sensibilidad
- **Cambio**: Se han añadido comprobaciones de existencia y validez de los archivos `numSensitivity.csv` y `catSensitivity.csv`. Si no existen, se crean DataFrames vacíos o se omite el procesamiento correspondiente.
- **Justificación**: En la versión original, si algún archivo faltaba, el programa lanzaba una excepción y se detenía.
- **Implicación**: CLEAR puede ahora ejecutarse con datasets que no tengan variables numéricas o categóricas, sin interrupciones.

### 4. Corrección de errores críticos en la lógica de regresión
- **Bug corregido 1**: En `CLEAR_regression.py`, dentro del bucle de ajuste de peso para multi‑clase, se incrementaba erróneamente `explainer.additional_weighting` en lugar de `single_regress.additional_weighting`. Ahora se usa la variable correcta.
- **Bug corregido 2**: En `avoidDummyTrap`, se usaba `selected.drop(feature_to_drop)` siendo `selected` una lista; ahora se usa `selected.remove(...)`.
- **Bug corregido 3**: En `CLEAR_perturbations.py`, se accedía directamente a `explainer.counterf_rows_df` sin verificar si existía el registro; ahora se usa un `mask` y un fallback a `model.predict()`.
- **Justificación**: Estos bugs impedían que CLEAR funcionara correctamente en ciertos escenarios (especialmente multi‑clase y datasets con categóricas).
- **Implicación**: La precisión y fiabilidad de los resultados han mejorado notablemente.

### 5. Clamping numérico para evitar desbordamientos
- **Cambio**: En el cálculo de `regProbWithActPerturbation`, se clampa el valor de `wTx` entre -500 y 500 antes de aplicar `exp`.
- **Justificación**: En regresiones con coeficientes grandes, `exp(wTx)` producía `inf` o `OverflowError`.
- **Implicación**: CLEAR ya no falla ni produce valores infinitos, garantizando que todos los contrafactuales se calculen correctamente.

### 6. Compatibilidad con scikit-learn ≥ 1.2
- **Cambio**: Se usa `get_feature_names_out` si está disponible, con fallback a la antigua `get_feature_names`.
- **Justificación**: La versión original solo usaba `get_feature_names`, que fue eliminada en scikit-learn 1.2.
- **Implicación**: CLEAR es compatible con las versiones más recientes de scikit-learn sin necesidad de modificar el código.

### 7. Reescritura de la función `generateString`
- **Cambio**: Se ha reimplementado completamente para manejar correctamente términos polinómicos (`_sqrd`) y de interacción (`_`), incluidos aquellos que involucran la característica objetivo.
- **Justificación**: La implementación anterior no cubría todos los casos posibles, lo que podía dar lugar a ecuaciones incorrectas y, por tanto, a perturbaciones erróneas.
- **Implicación**: La fidelidad de los contrafactuales estimados por CLEAR es ahora mayor, especialmente en regresiones con interacciones.

### 8. Comentarios y mensajes en español
- **Cambio**: Se han añadido comentarios y mensajes de depuración en español, además de emojis para una lectura más rápida.
- **Justificación**: Facilita el mantenimiento y la comprensión del código para equipos hispanohablantes.
- **Implicación**: El código es más accesible y la depuración es más ágil.

---

## Prerrequisitos

- Python 3.7 o superior.
- Dependencias listadas en `clear.yml` (original) o en `requirements.txt` (recomendado).
- **No se requiere** entorno gráfico; CLEAR puede ejecutarse en terminal, servidor o Jupyter.

---

## Instalación

1. Clona o descarga este repositorio en tu máquina.
2. (Opcional) Crea un entorno virtual.
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt ( está en la carpeta raiz del proyecto)