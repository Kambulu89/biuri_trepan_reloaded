"""Auditoria explícita de convergência para MLPClassifier e wrappers comuns."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


def _walk_mlps(model)->List[MLPClassifier]:
    found=[]; seen=set()
    def visit(obj):
        if obj is None or id(obj) in seen: return
        seen.add(id(obj))
        if isinstance(obj,MLPClassifier): found.append(obj); return
        calibrated=getattr(obj,'calibrated_classifiers_',[]) or []
        if calibrated:
            # Em CalibratedClassifierCV, ``estimator`` é apenas o template não ajustado;
            # a convergência real pertence aos estimadores dos folds calibrados.
            for cc in calibrated:
                visit(getattr(cc,'estimator',None))
            return
        steps=getattr(obj,'named_steps',None)
        if steps:
            for x in steps.values(): visit(x)
        for attr in ('estimator','base_estimator'):
            nested=getattr(obj,attr,None)
            if nested is not obj: visit(nested)
    visit(model); return found


def find_mlp_classifier(model)->Optional[MLPClassifier]:
    xs=_walk_mlps(model); return xs[0] if xs else None


def fit_with_convergence(model,X,y,**fit_kwargs):
    """Faz fit e persiste no MLP se ocorreu ``ConvergenceWarning``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(X,y,**fit_kwargs)
    had=any(issubclass(w.category,ConvergenceWarning) for w in caught)
    messages=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    for mlp in _walk_mlps(model):
        mlp._biuri_convergence_warning=had
        mlp._biuri_convergence_messages=messages
    return model


def _single(mlp:MLPClassifier)->Dict[str,Any]:
    n_iter=getattr(mlp,'n_iter_',None); max_iter=getattr(mlp,'max_iter',None)
    curve=list(getattr(mlp,'loss_curve_',[]) or [])
    validation=list(getattr(mlp,'validation_scores_',[]) or [])
    warning_seen=getattr(mlp,'_biuri_convergence_warning',None)
    cap=bool(n_iter is not None and max_iter is not None and int(n_iter)>=int(max_iter))
    early=bool(getattr(mlp,'early_stopping',False))
    if warning_seen is True or cap:
        state='max_iter_reached'; status='iteration_cap_reached_or_not_proven_converged'; converged=False; stopped='max_iter'
    elif early and n_iter is not None and max_iter is not None and int(n_iter)<int(max_iter):
        # sklearn não prova tol-convergence quando pára por n_iter_no_change.
        state='stopped_early_unverified'; status='stopped_early_unverified'; converged=False; stopped='early_stopping_no_improvement'
    elif n_iter is not None:
        state='converged'; status='converged_before_iteration_cap'; converged=True; stopped='tol'
    else:
        state='unknown'; status='unknown'; converged=None; stopped='unknown'
    return {
        'available':True,'status':status,'state':state,'converged':converged,'stopped_by':stopped,
        'convergence_warning':warning_seen,'convergence_messages':list(getattr(mlp,'_biuri_convergence_messages',[]) or []),
        'n_iter':None if n_iter is None else int(n_iter),'max_iter':None if max_iter is None else int(max_iter),
        'loss_curve':[float(v) for v in curve],'loss_curve_length':len(curve),
        'validation_scores':[float(v) for v in validation],
        'best_validation_score':float(np.max(validation)) if validation else None,
        'final_loss':float(curve[-1]) if curve else (float(getattr(mlp,'loss_')) if getattr(mlp,'loss_',None) is not None else None),
        'solver':str(getattr(mlp,'solver','unknown')),'tol':float(getattr(mlp,'tol',0.0)),
        'early_stopping':early,'validation_fraction':float(getattr(mlp,'validation_fraction',0.0)),
        'random_state':getattr(mlp,'random_state',None),
    }


def extract_mlp_convergence(model)->Dict[str,Any]:
    mlps=_walk_mlps(model)
    if not mlps: return {'available':False,'status':'not_an_mlp'}
    rows=[_single(x) for x in mlps]
    if len(rows)==1: return rows[0]
    flags=[r['converged'] is True for r in rows]
    return {
        'available':True,'status':'calibrated_ensemble','converged':bool(all(flags)),
        'estimators':rows,'estimator_count':len(rows),'converged_count':int(sum(flags)),
        'convergence_rate':float(np.mean(flags)),'min_n_iter':min(r['n_iter'] for r in rows if r['n_iter'] is not None),
        'max_n_iter':max(r['n_iter'] for r in rows if r['n_iter'] is not None),
    }

__all__=['find_mlp_classifier','fit_with_convergence','extract_mlp_convergence']
