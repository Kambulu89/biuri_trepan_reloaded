"""Protocolos de avaliação sem vazamento para treino e comparação científica."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


class PartitionRole(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    ACCEPTANCE_HOLDOUT = "acceptance_holdout"
    TEST = "test"
    EXTERNAL_TEST = "external_test"


@dataclass
class EvaluationProtocolGuard:
    """Regista e bloqueia o uso do teste durante seleção/ajuste.

    O teste pode ser consultado apenas por ``record_final_evaluation`` e uma
    única vez por execução. Isso torna violações do protocolo erros explícitos.
    """

    run_id: str = "default"
    events: List[Dict[str, Any]] = field(default_factory=list)
    final_evaluation_recorded: bool = False

    def record_selection(self, role: PartitionRole, operation: str, **details) -> None:
        role = PartitionRole(role)
        if role in {PartitionRole.TEST, PartitionRole.EXTERNAL_TEST}:
            raise RuntimeError(
                f"Vazamento bloqueado: '{operation}' tentou usar a partição "
                f"{role.value} para seleção/ajuste. Use apenas treino/validação."
            )
        self.events.append({
            "phase": "selection", "role": role.value,
            "operation": str(operation), "details": dict(details),
        })

    def record_final_evaluation(self, role: PartitionRole, operation: str, **details) -> None:
        role = PartitionRole(role)
        if role not in {PartitionRole.TEST, PartitionRole.EXTERNAL_TEST}:
            raise RuntimeError("A avaliação final deve usar test ou external_test.")
        if self.final_evaluation_recorded:
            raise RuntimeError(
                "A partição final já foi avaliada nesta execução; nova consulta "
                "poderia induzir ajuste iterativo ao teste."
            )
        self.final_evaluation_recorded = True
        self.events.append({
            "phase": "final_evaluation", "role": role.value,
            "operation": str(operation), "details": dict(details),
        })

    def audit(self) -> Dict[str, Any]:
        violations = [
            e for e in self.events
            if e["phase"] == "selection" and e["role"] in {"test", "external_test"}
        ]
        selection_ops = [str(e.get("operation", "")).lower() for e in self.events if e.get("phase") == "selection"]
        def used(token):
            return any(token in op for op in selection_ops) and bool(violations)
        return {
            "run_id": self.run_id,
            "protocol": "train/internal-validation/acceptance-holdout/locked-external-test",
            "events": list(self.events),
            "test_used_for_selection": bool(violations),
            "test_used_for_feature_selection": used("feature"),
            "test_used_for_oracle_selection": used("oracle"),
            "test_used_for_hyperparameter_selection": used("hyper"),
            "test_used_for_counterfactual_selection": used("counterfactual"),
            "test_used_for_tree_selection": used("tree"),
            "locked_test_used": self.final_evaluation_recorded,
            "final_test_evaluated": self.final_evaluation_recorded,
            "final_evaluation_recorded": self.final_evaluation_recorded,
            "valid": not violations,
        }


def make_internal_validation_split(
    X, y, *, validation_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cria validação exclusivamente dentro do conjunto de treino."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    if X_arr.ndim != 2 or len(X_arr) != len(y_arr):
        raise ValueError("X/y inválidos para criar validação interna.")
    values, counts = np.unique(y_arr, return_counts=True)
    stratify: Optional[np.ndarray] = y_arr if len(values) > 1 and counts.min() >= 2 else None
    return train_test_split(
        X_arr, y_arr, test_size=float(validation_size),
        random_state=int(random_state), stratify=stratify,
    )

# ---------------------------------------------------------------------------
# V9.2 — estratégia adaptativa e gates estatisticamente resolúveis
# ---------------------------------------------------------------------------
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

@dataclass(frozen=True)
class EvaluationStrategy:
    mode: str
    primary_metric: str
    n_splits: int
    n_repeats: int
    imbalance_ratio: float
    reason: str


def choose_evaluation_strategy(y, *, small_n_threshold: int = 200, imbalance_ratio_threshold: float = 3.0) -> EvaluationStrategy:
    y_arr=np.asarray(y); _,counts=np.unique(y_arr,return_counts=True)
    if len(counts)<2: raise ValueError('A avaliação exige pelo menos duas classes.')
    ratio=float(counts.max()/max(1,counts.min()))
    primary='balanced_accuracy' if ratio>=imbalance_ratio_threshold else 'accuracy'
    folds=max(2,min(5,int(counts.min())))
    if len(y_arr)<small_n_threshold:
        return EvaluationStrategy('repeated_stratified_cv',primary,folds,3,ratio,'amostra_pequena')
    return EvaluationStrategy('stratified_holdout_bootstrap',primary,folds,1,ratio,'amostra_suficiente')


def minimum_statistical_resolution(n_validation:int, *, k:float=1.0)->float:
    return float(k/max(1,int(n_validation)))


def noninferiority_decision(delta:float, *, configured_tolerance:float, n_validation:int, paired_ci:Optional[Tuple[float,float]]=None)->Dict[str,Any]:
    resolution=minimum_statistical_resolution(n_validation)
    tol=max(float(configured_tolerance),resolution)
    inconclusive=abs(float(delta)) < resolution
    if paired_ci is not None and paired_ci[0] <= 0 <= paired_ci[1]: inconclusive=True
    return {'delta':float(delta),'configured_tolerance':float(configured_tolerance),'effective_tolerance':tol,
            'minimum_resolution':resolution,'decision':'inconclusivo' if inconclusive else ('aceite' if delta>=-tol else 'rejeitado'),
            'inconclusive':bool(inconclusive),'accepted':bool(delta>=-tol and not inconclusive)}


def classification_metrics(y_true,y_pred)->Dict[str,Any]:
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    return {'accuracy':float(accuracy_score(y_true,y_pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y_true,y_pred)),
            'macro_f1':float(f1_score(y_true,y_pred,average='macro',zero_division=0)),
            'precision_macro':float(precision_score(y_true,y_pred,average='macro',zero_division=0)),
            'recall_macro':float(recall_score(y_true,y_pred,average='macro',zero_division=0)),
            'confusion_matrix':confusion_matrix(y_true,y_pred).tolist()}


def predictive_capacity_gate(model,X,y,*,margin:float=0.03,random_state:int=42)->Dict[str,Any]:
    X=np.asarray(X); y=np.asarray(y); strategy=choose_evaluation_strategy(y)
    cv=RepeatedStratifiedKFold(n_splits=strategy.n_splits,n_repeats=max(1,strategy.n_repeats),random_state=random_state)
    scoring='balanced_accuracy' if strategy.primary_metric=='balanced_accuracy' else 'accuracy'
    model_scores=cross_val_score(model,X,y,cv=cv,scoring=scoring)
    dummy_scores=cross_val_score(DummyClassifier(strategy='most_frequent'),X,y,cv=cv,scoring=scoring)
    delta=float(model_scores.mean()-dummy_scores.mean())
    return {'status':'capacidade_preditiva' if delta>=margin else 'sem_capacidade_preditiva','capable':bool(delta>=margin),
            'primary_metric':strategy.primary_metric,'model_mean':float(model_scores.mean()),'model_std':float(model_scores.std()),
            'dummy_mean':float(dummy_scores.mean()),'delta':delta,'required_margin':float(margin),'strategy':strategy.__dict__}
