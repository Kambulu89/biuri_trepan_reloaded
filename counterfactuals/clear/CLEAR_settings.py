# CLEAR_settings.py
# Eliminada dependencia de tkinter

""" Specifes CLEAR'S user input parameters. CLEAR sets the input parameters as global variables
whose values are NOT changed in any other module (these are CLEAR's only global variables).
The file 'Input Parameters for CLEAR.pdf' on Github documents the input parameters.
"""

# ============================================================================
# Declaración de variables globales (necesarias para que otros módulos las importen)
# ============================================================================
sample_model = 'model'
max_predictors = 15
first_obs = 1
last_obs = 1
num_samples = 50000
regression_type = 'logistic'
score_type = 'aic'
logistic_regularise = False
regression_sample_size = 200
CLEAR_path = 'D:/CLEAR/'
neighbourhood_algorithm = 'Balanced'
apply_counterfactual_weights = True
counterfactual_weight = 9
generate_regression_files = True
num_iterations = 1
interactions_only = False
centering = True
no_polynomimals = False
include_all_numerics = False
include_features = False
include_features_list = []
binary_decision_boundary = 0.5
multi_class_focus = 'All'
model_name = 'model'
numeric_features = []
categorical_features = []
category_prefix = []
class_labels = {}
test_sample = None  # No usado explícitamente, pero se referencia en init()
use_prev_sensitivity = False  # No usado, pero se referencia en init()
no_intercept = False   # Si True, se fuerza la regresión sin intercepto


def init():
    """Inicializa las variables globales con los valores por defecto.
       NOTA: Este método sobrescribe cualquier cambio previo. En tu pipeline
       NO lo llamas para evitar sobrescrituras, pero se mantiene por si se usa.
    """
    global sample_model, max_predictors, first_obs, last_obs, num_samples, regression_type, \
        score_type, logistic_regularise, test_sample, regression_sample_size, CLEAR_path, \
        neighbourhood_algorithm, apply_counterfactual_weights, counterfactual_weight, \
        num_iterations, generate_regression_files, interactions_only, centering, \
        no_polynomimals, multi_class_focus, use_prev_sensitivity, binary_decision_boundary, \
        include_all_numerics, include_features, include_features_list, model_name, \
        numeric_features, categorical_features, category_prefix, class_labels

    # Asignar valores por defecto
    model_name = 'model'
    numeric_features = []
    categorical_features = []
    category_prefix = []
    class_labels = {}
    sample_model = 'model'
    max_predictors = 15
    first_obs = 1
    last_obs = 1
    num_samples = 50000
    regression_type = 'logistic'
    score_type = 'aic'
    logistic_regularise = False
    regression_sample_size = 200
    CLEAR_path = 'D:/CLEAR/'
    neighbourhood_algorithm = 'Balanced'
    apply_counterfactual_weights = True
    counterfactual_weight = 9
    generate_regression_files = True
    num_iterations = 1
    interactions_only = False
    centering = True
    no_polynomimals = False
    include_all_numerics = False
    include_features = False
    include_features_list = []
    binary_decision_boundary = 0.5
    multi_class_focus = 'All'
    test_sample = None
    use_prev_sensitivity = False
    no_intercept = False

    check_input_parameters()


def check_input_parameters():
    """Verifica consistencia de los parámetros. Si hay error, imprime y sale."""
    import sys
    error_msg = ""
    if first_obs > last_obs:
        error_msg = "last_obs must be greater or equal to first obs"
    elif regression_type == 'logistic' and (score_type != 'prsquared' and score_type != 'aic'):
        error_msg = "logistic regression and score type combination incorrectly specified"
    elif regression_type == 'multiple' and score_type == 'prsquared':
        error_msg = "McFadden Pseudo R-squared cannot be used with multiple regression"
    elif regression_type not in ['multiple', 'logistic']:
        error_msg = "Regression type misspecified"
    elif neighbourhood_algorithm not in ['Balanced', 'Unbalanced']:
        error_msg = "neighbourhood algorithm misspecified"
    elif (isinstance((interactions_only & centering & no_polynomimals & apply_counterfactual_weights & generate_regression_files), bool)) is False:
        error_msg = "A boolean variable has been incorrectly specified"

    if error_msg != "":
        raise ValueError(f"ERROR in CLEAR_settings: {error_msg}")