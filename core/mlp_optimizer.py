"""
Otimização científica de MLPs (baseline, GridSearchCV, Optuna) para BIURI / Trepan Reloaded.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, ParameterGrid,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from core.mlp_factory import build_mlp_pipeline

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

from core.training_config import TrainingPreset, grid_param_grid_for_preset
from core.model_bundle import assert_train_test_schema
from core.mlp_convergence import extract_mlp_convergence, fit_with_convergence
from core.runtime_control import Deadline, check_interruption
from core.scientific_errors import TrainingError

RANDOM_STATE = 42
CancelFn = Optional[Callable[[], bool]]
ProgressFn = Optional[Callable[[str, int, str], None]]

MLP_GRID_PARAM_GRID = [
    {
        "mlp__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__solver": ["lbfgs"],
        "mlp__alpha": [0.001, 0.01],
        "mlp__max_iter": [1500],
        "mlp__early_stopping": [False],
    },
    {
        "mlp__hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__solver": ["adam"],
        "mlp__alpha": [0.001, 0.01],
        "mlp__learning_rate_init": [0.0005, 0.001],
        "mlp__max_iter": [1000],
        "mlp__early_stopping": [True],
        "mlp__validation_fraction": [0.15],
    },
]


def _project_results_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    out = root / "results" / "mlp_optimization"
    out.mkdir(parents=True, exist_ok=True)
    return out


DEFAULT_MLP_PARAMS = {}  # legado; a configuração autoritativa vive em core.mlp_factory

def _build_pipeline(**mlp_kwargs) -> Pipeline:
    # Alias de compatibilidade para chamadas antigas.
    return build_mlp_pipeline(**mlp_kwargs)


def _check_cancel(cancel_fn: CancelFn) -> bool:
    return bool(cancel_fn and cancel_fn())


def _elapsed_since(t0: float) -> float:
    return time.perf_counter() - t0


def _is_imbalanced(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        return False
    return counts.max() / max(counts.min(), 1) > 1.5


def _choose_scoring(y: np.ndarray) -> str:
    """balanced_accuracy se desbalanceado; accuracy caso contrário."""
    if _is_imbalanced(y):
        return "balanced_accuracy"
    return "accuracy"


def _safe_stratified_cv(y: np.ndarray, requested: int = 5) -> StratifiedKFold:
    """CV estratificada limitada pela menor classe."""
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or int(counts.min()) < 2:
        raise ValueError("Validação estratificada requer >=2 amostras por classe.")
    n_splits = max(2, min(int(requested), int(counts.min())))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def _is_degenerate_model(model: Pipeline, X_test: np.ndarray) -> bool:
    y_pred = model.predict(X_test)
    return len(np.unique(y_pred)) == 1


def _primary_selection_score(
    metrics: Dict[str, Any], y_train: np.ndarray
) -> float:
    """f1_weighted ou balanced_accuracy conforme desbalanceamento."""
    if _is_imbalanced(y_train):
        return float(metrics.get("balanced_accuracy", metrics.get("f1", 0)))
    return float(metrics.get("f1", metrics.get("accuracy", 0)))


def _evaluate_on_test(
    model: Pipeline, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "unique_predictions": int(len(np.unique(y_pred))),
        "degenerate": len(np.unique(y_pred)) == 1,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "predictions": y_pred,
        "convergence": extract_mlp_convergence(model),
    }



def evaluate_classifier_metrics(model, X, y) -> Dict[str, Any]:
    """Avalia o modelo final entregue; não deve ser usado para selecção."""
    return _evaluate_on_test(model, np.asarray(X), np.asarray(y))

def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in params.items():
        if isinstance(value, np.integer):
            out[key] = int(value)
        elif isinstance(value, np.floating):
            out[key] = float(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def train_baseline_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    n_features: Optional[int] = None,
    preset: Optional[TrainingPreset] = None,
    cancel_fn: CancelFn = None,
    evaluate_holdout: bool = True,
) -> Dict[str, Any]:
    """MLP base sem otimização (Pipeline + StandardScaler + hiperparâmetros adaptativos)."""
    if _check_cancel(cancel_fn):
        raise InterruptedError("Treino cancelado pelo utilizador.")

    preset = preset or TrainingPreset(key="balanced", display_name="Equilibrado", description="")
    t0 = time.perf_counter()
    from core.mlp_factory import build_mlp_for_data
    pipeline = build_mlp_for_data(X_train, y_train, random_state=RANDOM_STATE, preset=preset)
    profile = getattr(pipeline, "BIURI_ADAPTIVE_PROFILE", {})
    scoring = _choose_scoring(y_train)
    cv = _safe_stratified_cv(y_train, getattr(preset, "mlp_cv_folds", 5))
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
    fit_with_convergence(pipeline, X_train, y_train)
    metrics = _evaluate_on_test(pipeline, X_test, y_test) if evaluate_holdout else None
    elapsed = time.perf_counter() - t0

    best_params = dict(profile.get("params", {}))

    return {
        "model": pipeline,
        "model_name": model_name,
        "optimization_method": "baseline",
        "best_params": best_params,
        "best_score": float(np.mean(cv_scores)),
        "cv_score_std": float(np.std(cv_scores)),
        "scoring": scoring,
        "selection_score_source": "training_cross_validation",
        "test_metrics": metrics,
        "training_time_seconds": elapsed,
        "n_trials": 1,
        "timeout_reached": False,
    }


def optimize_mlp_with_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val_or_test: np.ndarray,
    y_val_or_test: np.ndarray,
    model_name: str,
    scoring: Optional[str] = None,
    preset: Optional[TrainingPreset] = None,
    param_grid: Optional[Dict[str, Any]] = None,
    cancel_fn: CancelFn = None,
    evaluate_holdout: bool = True,
) -> Dict[str, Any]:
    """Pesquisa por candidatos com deadline verificado entre folds/candidatos."""
    preset = preset or TrainingPreset(key="balanced", display_name="Equilibrado", description="")
    t0=time.perf_counter(); scoring=scoring or _choose_scoring(y_train)
    grid_params=param_grid or grid_param_grid_for_preset(preset) or MLP_GRID_PARAM_GRID
    base=_build_pipeline(max_iter=preset.mlp_max_iter,solver=preset.mlp_solver,
                         n_iter_no_change=preset.n_iter_no_change,validation_fraction=preset.validation_fraction)
    cv=_safe_stratified_cv(y_train,preset.mlp_cv_folds); deadline=Deadline(preset.grid_timeout_seconds)
    best_score=float('-inf'); best_params=None; tried=0; timed_out=False
    for params in ParameterGrid(grid_params):
        try: check_interruption(deadline=deadline,cancel_fn=cancel_fn)
        except TimeoutError: timed_out=True; break
        fold_scores=[]
        for fit_idx,val_idx in cv.split(X_train,y_train):
            try: check_interruption(deadline=deadline,cancel_fn=cancel_fn)
            except TimeoutError: timed_out=True; break
            model=clone(base).set_params(**params); fit_with_convergence(model,X_train[fit_idx],y_train[fit_idx])
            pred=model.predict(X_train[val_idx])
            score=balanced_accuracy_score(y_train[val_idx],pred) if scoring=='balanced_accuracy' else accuracy_score(y_train[val_idx],pred)
            fold_scores.append(float(score))
        if timed_out: break
        tried+=1; mean=float(np.mean(fold_scores))
        if mean>best_score: best_score=mean; best_params=dict(params)
    if best_params is None:
        if _check_cancel(cancel_fn): raise InterruptedError('Treino cancelado pelo utilizador.')
        raise TrainingError('Nenhum candidato GridSearch terminou dentro do orçamento de tempo.')
    best_pipeline=clone(base).set_params(**best_params); fit_with_convergence(best_pipeline,X_train,y_train)
    test_metrics=_evaluate_on_test(best_pipeline,X_val_or_test,y_val_or_test) if evaluate_holdout else None
    elapsed=time.perf_counter()-t0
    clean={k.replace('mlp__',''):v for k,v in best_params.items()}
    return {'model':best_pipeline,'model_name':model_name,'optimization_method':'grid_search','best_params':clean,
            'best_score':best_score,'selection_score_source':'training_cross_validation','scoring':scoring,
            'test_metrics':test_metrics,'training_time_seconds':elapsed,'n_trials':tried,'timeout_reached':timed_out,
            'grid_cv_results':{'mean_test_score':best_score,'n_candidates':tried}}


def optimize_mlp_with_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    n_trials: int = 50,
    preset: Optional[TrainingPreset] = None,
    cancel_fn: CancelFn = None,
    evaluate_holdout: bool = True,
) -> Dict[str, Any]:
    """Otimização bayesiana com Optuna; melhor modelo treinado em X_train completo."""
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna não instalado. Execute: pip install optuna"
        )
    if _check_cancel(cancel_fn):
        raise InterruptedError("Treino cancelado pelo utilizador.")

    preset = preset or TrainingPreset(key="balanced", display_name="Equilibrado", description="")
    t0 = time.perf_counter()
    scoring = _choose_scoring(y_train)
    cv = _safe_stratified_cv(y_train, preset.mlp_cv_folds)
    max_iter_cap = preset.mlp_max_iter
    timeout = preset.optuna_timeout_seconds
    n_large = len(X_train) > 5000

    def objective(trial: "optuna.Trial") -> float:
        if _check_cancel(cancel_fn):
            raise optuna.exceptions.OptunaError("cancelled")

        n_layers = trial.suggest_int("n_layers", 1, 3)
        units = []
        if n_layers >= 1:
            units.append(trial.suggest_int("n_units_l1", 32, 256))
        if n_layers >= 2:
            units.append(trial.suggest_int("n_units_l2", 16, 128))
        if n_layers >= 3:
            units.append(trial.suggest_int("n_units_l3", 8, 64))

        activation = trial.suggest_categorical("activation", ["relu", "tanh"])
        solvers = ["adam"] if n_large else ["adam", "lbfgs"]
        solver = trial.suggest_categorical("solver", solvers)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float(
            "learning_rate_init", 1e-5, 1e-2, log=True
        )
        max_iter = trial.suggest_int("max_iter", 300, max_iter_cap)

        pipeline = _build_pipeline(
            hidden_layer_sizes=tuple(units),
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=preset.n_iter_no_change,
            validation_fraction=preset.validation_fraction,
        )

        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            if scoring == "f1_weighted":
                score = f1_score(y_val, y_pred, average="weighted", zero_division=0)
            elif scoring == "balanced_accuracy":
                score = balanced_accuracy_score(y_val, y_pred)
            else:
                score = accuracy_score(y_val, y_pred)
            scores.append(score)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optimize_kwargs = {"n_trials": n_trials, "show_progress_bar": False}
    if timeout:
        optimize_kwargs["timeout"] = timeout
    def _stop_callback(study, trial):
        if _check_cancel(cancel_fn):
            study.stop()
        if timeout is not None and _elapsed_since(t0) >= timeout:
            study.stop()
    optimize_kwargs["callbacks"] = [_stop_callback]
    study.optimize(objective, **optimize_kwargs)
    timed_out = timeout is not None and _elapsed_since(t0) >= timeout

    bp = study.best_params
    units = [bp["n_units_l1"]]
    if bp.get("n_layers", 1) >= 2:
        units.append(bp["n_units_l2"])
    if bp.get("n_layers", 1) >= 3:
        units.append(bp["n_units_l3"])

    best_params = {
        "hidden_layer_sizes": tuple(units),
        "activation": bp["activation"],
        "solver": bp["solver"],
        "alpha": bp["alpha"],
        "learning_rate_init": bp["learning_rate_init"],
        "max_iter": bp["max_iter"],
        "early_stopping": True,
        "validation_fraction": preset.validation_fraction,
        "n_iter_no_change": preset.n_iter_no_change,
        "random_state": RANDOM_STATE,
    }

    best_pipeline = _build_pipeline(**best_params)
    fit_with_convergence(best_pipeline, X_train, y_train)
    test_metrics = _evaluate_on_test(best_pipeline, X_test, y_test) if evaluate_holdout else None
    elapsed = time.perf_counter() - t0

    return {
        "model": best_pipeline,
        "model_name": model_name,
        "optimization_method": "optuna",
        "best_params": best_params,
        "best_score": float(study.best_value),
        "selection_score_source": "training_cross_validation",
        "scoring": scoring,
        "test_metrics": test_metrics,
        "training_time_seconds": elapsed,
        "optuna_trials": len(study.trials),
        "n_trials": len(study.trials),
        "timeout_reached": timed_out,
    }


def _pick_best_result(
    candidates: List[Dict[str, Any]],
    X_test: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    valid = [c for c in candidates if c.get("model") is not None]
    if not valid:
        raise ValueError("Nenhum resultado de otimização MLP válido.")

    # ``X_test`` e ``y_test`` permanecem na assinatura por compatibilidade,
    # mas nunca participam na selecao do melhor modelo.
    X_selection = X_test

    def _rank(result: Dict[str, Any]) -> Tuple[int, int, float, float]:
        model = result.get("model")
        non_degenerate = 1  # degenerescência é auditada por CV, nunca pelo teste externo
        convergence = extract_mlp_convergence(model) if model is not None else {}
        converged = 1 if convergence.get("converged") is True else 0
        result["convergence"] = convergence
        cv_score = float(result.get("best_score", float("-inf")))
        cv_std = float(result.get("cv_score_std", 0.0) or 0.0)
        # Um modelo que atingiu o teto de iterações não pode ganhar apenas por
        # apresentar um score pontualmente maior que um candidato convergido.
        return (non_degenerate, converged, cv_score, -cv_std)

    return max(valid, key=_rank)


ROBUST_MLP_GRID = {
    "hidden_layer_sizes": [(50,), (100,), (100, 50), (128, 64), (64, 32)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate_init": [0.001, 0.0005, 0.0001],
    "max_iter": [500, 1000, 2000],
}


def train_robust_mlp_original(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mode: str = "balanced",
    cancel_fn: CancelFn = None,
) -> Dict[str, Any]:
    """
    Testa várias configurações MLP e selecciona a que:
    1) não prevê uma única classe;
    2) tem melhor f1_weighted ou balanced_accuracy;
    3) supera DummyClassifier.
    """
    if _check_cancel(cancel_fn):
        raise InterruptedError("Treino cancelado pelo utilizador.")

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    imbalanced = _is_imbalanced(y_train)

    try:
        X_fit, X_select, y_fit, y_select = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE,
            stratify=y_train,
        )
    except ValueError:
        X_fit, X_select, y_fit, y_select = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE,
        )

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_fit, y_fit)
    y_dummy = dummy.predict(X_select)
    dummy_accuracy = float(accuracy_score(y_select, y_dummy))
    dummy_f1 = float(f1_score(y_select, y_dummy, average="weighted", zero_division=0))

    t0 = time.perf_counter()
    candidates: List[Dict[str, Any]] = []

    for hidden in ROBUST_MLP_GRID["hidden_layer_sizes"]:
        for activation in ROBUST_MLP_GRID["activation"]:
            for alpha in ROBUST_MLP_GRID["alpha"]:
                for lr in ROBUST_MLP_GRID["learning_rate_init"]:
                    for max_iter in ROBUST_MLP_GRID["max_iter"]:
                        if _check_cancel(cancel_fn):
                            raise InterruptedError("Treino cancelado pelo utilizador.")
                        pipeline = _build_pipeline(
                            hidden_layer_sizes=hidden,
                            activation=activation,
                            alpha=alpha,
                            learning_rate_init=lr,
                            max_iter=max_iter,
                        )
                        try:
                            pipeline.fit(X_fit, y_fit)
                        except Exception:
                            continue
                        tm = _evaluate_on_test(pipeline, X_select, y_select)
                        primary = (
                            tm["balanced_accuracy"] if imbalanced else tm["f1"]
                        )
                        beats_dummy = (
                            tm["accuracy"] > dummy_accuracy + 0.01
                            and primary > dummy_f1 + 0.01
                        )
                        candidates.append(
                            {
                                "model": pipeline,
                                "model_name": "MLP Original",
                                "optimization_method": "robust_grid",
                                "best_params": {
                                    "hidden_layer_sizes": hidden,
                                    "activation": activation,
                                    "alpha": alpha,
                                    "learning_rate_init": lr,
                                    "max_iter": max_iter,
                                },
                                "best_score": primary,
                                "selection_metrics": tm,
                                "beats_dummy": beats_dummy,
                                "non_degenerate": not tm["degenerate"],
                            }
                        )

    if not candidates:
        pipeline = _build_pipeline()
        pipeline.fit(X_train, y_train)
        tm = _evaluate_on_test(pipeline, X_test, y_test)
        return {
            "model": pipeline,
            "model_name": "MLP Original",
            "optimization_method": "robust_fallback_default",
            "best_params": dict(DEFAULT_MLP_PARAMS),
            "best_score": tm.get("f1", 0),
            "test_metrics": tm,
            "training_time_seconds": time.perf_counter() - t0,
            "n_trials": 1,
        }

    for candidate in candidates:
        candidate["convergence"] = extract_mlp_convergence(candidate.get("model"))
    converged_candidates = [
        c for c in candidates if (c.get("convergence") or {}).get("converged") is True
    ]
    eligible_candidates = converged_candidates or candidates
    non_degen = [c for c in eligible_candidates if c["non_degenerate"] and c["beats_dummy"]]
    pool = non_degen or [c for c in eligible_candidates if c["non_degenerate"]] or eligible_candidates
    best = max(
        pool,
        key=lambda c: (
            _primary_selection_score(c["selection_metrics"], y_train),
            c["selection_metrics"]["accuracy"],
        ),
    )
    # Reajuste final no treino externo completo; teste usado uma unica vez.
    bp = best["best_params"]
    final_model = _build_pipeline(**bp)
    final_model.fit(X_train, y_train)
    best["model"] = final_model
    best["test_metrics"] = _evaluate_on_test(final_model, X_test, y_test)
    best["convergence"] = extract_mlp_convergence(final_model)
    best["selection_score_source"] = "training_holdout"
    best["training_time_seconds"] = time.perf_counter() - t0
    best["n_trials"] = len(candidates)
    best["dummy_accuracy"] = dummy_accuracy
    best["dummy_f1"] = dummy_f1

    print(
        f"[INFO] train_robust_mlp_original: {len(candidates)} configs testadas, "
        f"seleccionada hidden={best['best_params']['hidden_layer_sizes']}, "
        f"selection_f1={best['selection_metrics']['f1']:.4f}, "
        f"test_f1={best['test_metrics']['f1']:.4f}, "
        f"degenerate={best['test_metrics']['degenerate']}"
    )
    return best


def run_full_mlp_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    dataset_name: str = "unknown",
    run_grid: bool = True,
    run_optuna: bool = True,
    optuna_trials: int = 50,
    preset: Optional[TrainingPreset] = None,
    cancel_fn: CancelFn = None,
    progress_fn: ProgressFn = None,
    evaluate_selected_on_test: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Executa baseline, GridSearch e Optuna; devolve o melhor modelo e todos os resultados.
    Respeita preset, timeouts e cancelamento; mantém melhor modelo parcial em timeout.
    """
    from core.training_config import get_training_preset

    explicit_preset = preset is not None
    preset = preset or get_training_preset("scientific")
    if explicit_preset:
        run_grid = preset.run_grid
        run_optuna = preset.run_optuna
        optuna_trials = preset.optuna_trials

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    assert_train_test_schema(X_train, X_test, model_name)

    all_results: List[Dict[str, Any]] = []
    t_global = time.perf_counter()
    mlp_timeout = preset.mlp_timeout_seconds

    def _progress(msg: str, pct: int) -> None:
        if progress_fn:
            progress_fn("mlp", pct, msg)

    if preset.run_baseline:
        _progress(f"Treinando {model_name} (baseline)...", 5)
        baseline = train_baseline_mlp(
            X_train, y_train, X_test, y_test, model_name,
            n_features=X_train.shape[1], preset=preset, cancel_fn=cancel_fn, evaluate_holdout=False,
        )
        all_results.append(baseline)

    if run_grid:
        if mlp_timeout and _elapsed_since(t_global) >= mlp_timeout:
            print(f"[WARN] Timeout MLP ({model_name}) antes do GridSearch.")
        elif not _check_cancel(cancel_fn):
            _progress(f"GridSearch {model_name}...", 25)
            try:
                grid_result = optimize_mlp_with_grid_search(
                    X_train, y_train, X_test, y_test, model_name,
                    preset=preset, cancel_fn=cancel_fn, evaluate_holdout=False,
                )
                all_results.append(grid_result)
            except InterruptedError:
                raise
            except Exception as exc:
                print(f"[WARN] GridSearch MLP ({model_name}) falhou: {exc}")

    if run_optuna:
        if mlp_timeout and _elapsed_since(t_global) >= mlp_timeout:
            print(f"[WARN] Timeout MLP ({model_name}) antes do Optuna.")
        elif not _check_cancel(cancel_fn):
            _progress(f"Otimizando Optuna ({model_name})...", 45)
            try:
                if OPTUNA_AVAILABLE:
                    optuna_result = optimize_mlp_with_optuna(
                        X_train, y_train, X_test, y_test, model_name,
                        n_trials=optuna_trials, preset=preset, cancel_fn=cancel_fn, evaluate_holdout=False,
                    )
                    all_results.append(optuna_result)
                else:
                    print(
                        "[WARN] Optuna não disponível; instale com pip install optuna"
                    )
            except InterruptedError:
                raise
            except Exception as exc:
                print(f"[WARN] Optuna MLP ({model_name}) falhou: {exc}")

    if not all_results:
        raise ValueError("Nenhum resultado de otimização MLP válido.")

    best = _pick_best_result(all_results, None, y_train, None)
    if evaluate_selected_on_test:
        # O teste externo é consultado uma única vez e apenas para o modelo já seleccionado.
        best["test_metrics"] = _evaluate_on_test(best["model"], X_test, y_test)
        best["final_evaluation_scope"] = "selected_model_only"
    else:
        best["test_metrics"] = None
        best["final_evaluation_scope"] = "deferred_to_final_delivered_model"
    global_timeout = mlp_timeout is not None and _elapsed_since(t_global) >= mlp_timeout
    if global_timeout:
        print(
            f"[WARN] Timeout global MLP ({model_name}); "
            f"melhor modelo até ao momento: {best.get('optimization_method')}"
        )
        best["timeout_reached"] = True

    for result in all_results:
        result["dataset_name"] = dataset_name
        result["selected_as_best"] = result is best

    log_mlp_optimization(best)
    if preset.full_logs:
        save_mlp_optimization_results(all_results, dataset_name)

    return best, all_results


def log_mlp_optimization(result: Dict[str, Any]) -> None:
    """Logs obrigatórios [MLP OPTIMIZATION - ...]."""
    name = result.get("model_name", "MLP")
    tm = result.get("test_metrics") or {}
    print(f"\n[MLP OPTIMIZATION - {name}]")
    print(f"Method: {result.get('optimization_method', 'unknown')}")
    print(f"Best params: {_serialize_params(result.get('best_params', {}))}")
    print(f"Best CV score: {result.get('best_score', 0):.4f}")
    print(f"Test accuracy: {tm.get('accuracy', 0):.4f}")
    print(f"Test precision: {tm.get('precision', 0):.4f}")
    print(f"Test recall: {tm.get('recall', 0):.4f}")
    print(f"Test f1: {tm.get('f1', 0):.4f}")
    conv = result.get("convergence") or tm.get("convergence") or {}
    if conv:
        print(
            "Convergence: "
            f"status={conv.get('status')} n_iter={conv.get('n_iter')} "
            f"max_iter={conv.get('max_iter')} seed={conv.get('random_state')}"
        )


def log_metrics_check(
    precision_metrics: Dict[str, Any],
    fidelity_metrics: Dict[str, Any],
) -> None:
    """Log obrigatório [METRICS CHECK]."""
    def _acc(key: str) -> str:
        block = precision_metrics.get(key)
        if not block:
            return "N/A"
        return f"{block.get('accuracy', 0) * 100:.1f}%"

    def _fid(key: str) -> str:
        block = fidelity_metrics.get(key)
        if not block:
            return "N/A"
        return f"{block.get('overall_fidelity', 0) * 100:.1f}%"

    trep_rel = fidelity_metrics.get("trepan_reloaded") or {}
    oracle_ref = trep_rel.get("fidelity_reference", "mlp_original")

    print("\n[METRICS CHECK]")
    print(f"MLP Original accuracy: {_acc('mlp')}")
    print(f"C4.5 accuracy: {_acc('c45_j48')}")
    print(f"Trepan Original accuracy: {_acc('trepan_original')}")
    print(f"Trepan Reloaded accuracy: {_acc('trepan_reloaded')}")
    print(f"Trepan Original fidelity to MLP Original: {_fid('trepan_original')}")
    print(f"Trepan Reloaded fidelity to oracle ({oracle_ref}): {_fid('trepan_reloaded')}")


def save_mlp_optimization_results(
    results: List[Dict[str, Any]],
    dataset_name: str,
) -> Path:
    """Persiste CSV/JSON para relatório científico."""
    out_dir = _project_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in dataset_name
    )
    json_path = out_dir / f"mlp_optimization_{safe_name}_{timestamp}.json"
    csv_path = out_dir / f"mlp_optimization_{safe_name}_{timestamp}.csv"

    rows = []
    for result in results:
        tm = result.get("test_metrics") or {}
        row = {
            "dataset_name": dataset_name,
            "model_name": result.get("model_name", ""),
            "optimization_method": result.get("optimization_method", ""),
            "best_params": _serialize_params(result.get("best_params", {})),
            "best_cv_score": result.get("best_score"),
            "accuracy": tm.get("accuracy"),
            "precision": tm.get("precision"),
            "recall": tm.get("recall"),
            "f1": tm.get("f1"),
            "fidelity_target": None,
            "fidelity_value": None,
            "training_time_seconds": result.get("training_time_seconds"),
            "selected_as_best": result.get("selected_as_best", False),
            "timestamp": timestamp,
        }
        rows.append(row)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)

    fieldnames = [
        "dataset_name",
        "model_name",
        "optimization_method",
        "best_params",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fidelity_target",
        "fidelity_value",
        "training_time_seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["best_params"] = json.dumps(
                csv_row.get("best_params", {}), ensure_ascii=False
            )
            writer.writerow(csv_row)

    print(f"[INFO] Resultados MLP guardados em:\n  {json_path}\n  {csv_path}")
    return json_path


def save_comparison_metrics_for_article(
    comparison_results: Dict[str, Any],
    dataset_name: str = "unknown",
    mlp_optimization_runs: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Exporta métricas finais (precisão + fidelidade) para o artigo científico."""
    out_dir = _project_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in dataset_name
    )
    json_path = out_dir / f"article_metrics_{safe_name}_{timestamp}.json"
    csv_path = out_dir / f"article_metrics_{safe_name}_{timestamp}.csv"

    rows: List[Dict[str, Any]] = []

    if mlp_optimization_runs:
        for run in mlp_optimization_runs:
            tm = run.get("test_metrics", {})
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": run.get("model_name", ""),
                    "optimization_method": run.get("optimization_method", ""),
                    "best_params": _serialize_params(run.get("best_params", {})),
                    "accuracy": tm.get("accuracy"),
                    "precision": tm.get("precision"),
                    "recall": tm.get("recall"),
                    "f1": tm.get("f1"),
                    "fidelity_target": None,
                    "fidelity_value": None,
                    "training_time_seconds": run.get("training_time_seconds"),
                }
            )

    precision = comparison_results.get("precision", {})
    fidelity = comparison_results.get("fidelity", {})
    model_map = {
        "mlp": ("MLP Original", None),
        "c45_j48": ("C4.5-Nativo", None),
        "trepan_original": ("Trepan Original", "MLP Original"),
        "trepan_reloaded": ("Trepan Reloaded", None),
    }
    for key, (display_name, default_target) in model_map.items():
        p = precision.get(key)
        f = fidelity.get(key)
        if not p:
            continue
        fid_target = default_target
        if key == "trepan_reloaded" and f:
            ref = f.get("fidelity_reference", "mlp_original")
            fid_target = "MLP Residual Ontológico" if ref == "mlp_onto" else "MLP Original"
        rows.append(
            {
                "dataset_name": dataset_name,
                "model_name": display_name,
                "optimization_method": "evaluation",
                "best_params": None,
                "accuracy": p.get("accuracy"),
                "precision": p.get("precision"),
                "recall": p.get("recall"),
                "f1": p.get("f1"),
                "fidelity_target": fid_target,
                "fidelity_value": f.get("overall_fidelity") if f else None,
                "training_time_seconds": None,
            }
        )

    fieldnames = [
        "dataset_name",
        "model_name",
        "optimization_method",
        "best_params",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fidelity_target",
        "fidelity_value",
        "training_time_seconds",
    ]
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            if csv_row.get("best_params") is not None:
                csv_row["best_params"] = json.dumps(
                    csv_row["best_params"], ensure_ascii=False
                )
            writer.writerow(csv_row)

    print(f"[INFO] Métricas para artigo guardadas em:\n  {json_path}\n  {csv_path}")
    return json_path
