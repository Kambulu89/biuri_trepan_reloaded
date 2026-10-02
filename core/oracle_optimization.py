"""Otimiza, calibra e audita oráculos sem consultar o teste externo."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from core.mlp_factory import build_mlp_for_data


@dataclass(frozen=True)
class OracleGateConfig:
    folds: int = 5
    random_state: int = 42
    noninferiority_margin: float = 0.01
    required_win_rate: float = 0.60
    calibration_method: str = "sigmoid"
    calibration_primary_tolerance: float = 0.02


def _folds(y, requested: int) -> int:
    _, counts = np.unique(np.asarray(y), return_counts=True)
    if len(counts) < 2 or int(counts.min()) < 2:
        raise ValueError("Calibração exige duas classes e duas amostras por classe.")
    return max(2, min(int(requested), int(counts.min())))


def calibrate_estimator(estimator, X_train, y_train, *, config=OracleGateConfig()):
    """Decide calibração apenas dentro do treino e devolve o modelo final ajustado.

    A decisão usa um holdout interno estratificado; o teste externo nunca entra.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
    X = np.asarray(X_train, dtype=float); y = np.asarray(y_train)
    minimum=max(40,12*len(np.unique(y)))
    if len(y) < minimum:
        fitted=clone(estimator).fit(X,y)
        return fitted,{"scope":"outer_training_only","test_used":False,"skipped":True,"reason":"insufficient_calibration_sample","minimum_required":minimum}
    folds=_folds(y,config.folds)
    idx=np.arange(len(y))
    fit_idx,val_idx=train_test_split(idx,test_size=0.20,random_state=config.random_state,stratify=y)
    base=clone(estimator).fit(X[fit_idx],y[fit_idx])
    cv_inner=StratifiedKFold(n_splits=_folds(y[fit_idx],folds),shuffle=True,random_state=config.random_state)
    calibrated_inner=CalibratedClassifierCV(estimator=clone(estimator),method=config.calibration_method,cv=cv_inner).fit(X[fit_idx],y[fit_idx])
    def primary(model):
        pred=model.predict(X[val_idx])
        return {"accuracy":float(accuracy_score(y[val_idx],pred)),"balanced_accuracy":float(balanced_accuracy_score(y[val_idx],pred)),"macro_f1":float(f1_score(y[val_idx],pred,average="macro",zero_division=0))}
    before=primary(base); after=primary(calibrated_inner)
    # balanced accuracy + macro-F1 são primárias para evitar esconder colapso por desbalanceamento.
    delta=min(after["balanced_accuracy"]-before["balanced_accuracy"],after["macro_f1"]-before["macro_f1"])
    accept=delta >= -float(config.calibration_primary_tolerance)
    if accept:
        cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=config.random_state)
        final=CalibratedClassifierCV(estimator=clone(estimator),method=config.calibration_method,cv=cv).fit(X,y)
        final.ORACLE_CALIBRATED=True; final.ORACLE_CALIBRATION_METHOD=config.calibration_method
    else:
        final=clone(estimator).fit(X,y)
    return final,{"scope":"outer_training_only","test_used":False,"method":config.calibration_method,"cv_folds":folds,
                  "accepted":bool(accept),"reason":"accepted" if accept else "primary_metric_degradation",
                  "internal_before":before,"internal_after":after,"worst_primary_delta":float(delta),
                  "tolerance":float(config.calibration_primary_tolerance)}


def _oof_metrics(estimator, X, y, cv):
    pred = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict")
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "fold_scores": [],
    }
    for fit_idx, hold_idx in cv.split(X, y):
        model = clone(estimator).fit(X[fit_idx], y[fit_idx])
        fold_pred = model.predict(X[hold_idx])
        result["fold_scores"].append({
            "balanced_accuracy": float(balanced_accuracy_score(y[hold_idx], fold_pred)),
            "macro_f1": float(f1_score(
                y[hold_idx], fold_pred, average="macro", zero_division=0,
            )),
        })
    return result


def compare_ontological_oracle_to_c45(
    ontological_oracle,
    c45_estimator,
    X_ontological,
    X_original,
    y_train,
    *,
    config: OracleGateConfig = OracleGateConfig(),
) -> Dict[str, Any]:
    """Gate OOF pareado; nunca usa a partíção externa de teste."""
    X_onto = np.asarray(X_ontological, dtype=float)
    X_orig = np.asarray(X_original, dtype=float)
    y = np.asarray(y_train)
    if len(X_onto) != len(X_orig) or len(y) != len(X_orig):
        raise ValueError("Gate oráculo/C4.5 exige linhas pareadas.")
    folds = _folds(y, config.folds)
    cv_onto = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=config.random_state,
    )
    cv_c45 = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=config.random_state,
    )
    onto = _oof_metrics(ontological_oracle, X_onto, y, cv_onto)
    c45 = _oof_metrics(c45_estimator, X_orig, y, cv_c45)
    differences = np.asarray([
        a["balanced_accuracy"] - b["balanced_accuracy"]
        for a, b in zip(onto["fold_scores"], c45["fold_scores"])
    ])
    win_rate = float(np.mean(differences > 0))
    mean_delta = float(onto["balanced_accuracy"] - c45["balanced_accuracy"])
    noninferior = mean_delta >= -float(config.noninferiority_margin)
    superior = bool(mean_delta > 0 and win_rate >= config.required_win_rate)
    return {
        "scope": "outer_training_oof_only",
        "test_used": False,
        "config": asdict(config),
        "ontological_oracle": onto,
        "c45": c45,
        "balanced_accuracy_delta": mean_delta,
        "fold_differences": differences.tolist(),
        "win_rate": win_rate,
        "noninferior": bool(noninferior),
        "superior": superior,
        "accepted_for_reloaded": bool(noninferior),
        "claim": (
            "superior_on_internal_oof" if superior
            else "noninferior_on_internal_oof" if noninferior
            else "ontological_oracle_rejected"
        ),
    }


def calibration_metrics(model, X, y) -> Dict[str, Optional[float]]:
    """Métricas auxiliares; usar apenas no papel permitido pelo protocolo."""
    probs = np.asarray(model.predict_proba(np.asarray(X, dtype=float)), dtype=float)
    classes = np.asarray(model.classes_)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    indices = np.asarray([class_to_idx[label] for label in np.asarray(y)], dtype=int)
    one_hot = np.eye(len(classes))[indices]
    confidence = np.max(probs, axis=1)
    predictions = classes[np.argmax(probs, axis=1)]
    correctness = (predictions == np.asarray(y)).astype(float)
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (
            confidence <= high if high == 1.0 else confidence < high
        )
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(correctness[mask])) - float(np.mean(confidence[mask]))
            )
    return {
        "log_loss": float(log_loss(y, probs, labels=classes)),
        "brier_multiclass": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "expected_calibration_error_10_bins": float(ece),
    }


def optimize_and_calibrate_mlp(
    X_train,
    y_train,
    *,
    random_state: int = 42,
    n_iter: int = 8,
    cv_folds: int = 3,
):
    """Busca interna reprodutível seguida de calibração no mesmo treino externo."""
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train)
    folds = _folds(y, cv_folds)
    pipeline = build_mlp_for_data(X, y, random_state=random_state)
    parameters = {
        "mlp__hidden_layer_sizes": [(32,), (64,), (64, 32), (96, 48), (128, 64)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [2e-4, 5e-4, 1e-3, 2e-3],
    }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    def multiobjective_refit(cv_results):
        ba = np.asarray(cv_results["mean_test_balanced_accuracy"], dtype=float)
        macro = np.asarray(cv_results["mean_test_macro_f1"], dtype=float)
        accuracy = np.asarray(cv_results["mean_test_accuracy"], dtype=float)
        neg_loss = np.asarray(cv_results["mean_test_neg_log_loss"], dtype=float)
        stability = 0.5 * (
            np.asarray(cv_results["std_test_balanced_accuracy"], dtype=float)
            + np.asarray(cv_results["std_test_macro_f1"], dtype=float)
        )
        calibration = 1.0 / (1.0 + np.maximum(0.0, -neg_loss))
        utility = (
            0.40 * ba + 0.35 * macro + 0.15 * accuracy
            + 0.10 * calibration - 0.05 * stability
        )
        return int(np.nanargmax(utility))

    search = RandomizedSearchCV(
        pipeline, parameters, n_iter=max(1, int(n_iter)),
        scoring={
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
            "accuracy": "accuracy",
            "neg_log_loss": "neg_log_loss",
        },
        cv=cv, refit=multiobjective_refit,
        random_state=random_state, n_jobs=1, error_score="raise",
    ).fit(X, y)
    calibrated, calibration = calibrate_estimator(
        search.best_estimator_, X, y,
        config=OracleGateConfig(
            folds=folds, random_state=random_state, calibration_method="sigmoid",
        ),
    )
    best_index = int(search.best_index_)
    best_cv = {
        "balanced_accuracy": float(
            search.cv_results_["mean_test_balanced_accuracy"][best_index]
        ),
        "macro_f1": float(search.cv_results_["mean_test_macro_f1"][best_index]),
        "accuracy": float(search.cv_results_["mean_test_accuracy"][best_index]),
        "log_loss": float(-search.cv_results_["mean_test_neg_log_loss"][best_index]),
        "stability_std": float(0.5 * (
            search.cv_results_["std_test_balanced_accuracy"][best_index]
            + search.cv_results_["std_test_macro_f1"][best_index]
        )),
    }
    return calibrated, {
        "optimization": "randomized_search_inner_cv_multiobjective",
        "selection_scope": "outer_training_only",
        "test_used": False,
        "n_candidates": int(len(search.cv_results_["params"])),
        "objective_weights": {
            "balanced_accuracy": 0.40, "macro_f1": 0.35,
            "accuracy": 0.15, "calibration": 0.10,
            "stability_penalty": 0.05,
        },
        "best_cv_metrics": best_cv,
        "best_cv_balanced_accuracy": best_cv["balanced_accuracy"],
        "best_params": dict(search.best_params_),
        "calibration": calibration,
    }


__all__ = [
    "OracleGateConfig", "calibrate_estimator",
    "compare_ontological_oracle_to_c45", "calibration_metrics",
    "optimize_and_calibrate_mlp",
]
