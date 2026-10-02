"""Fábrica única e adaptativa de MLPs do BIURI/TREPAN Reloaded."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional
import warnings
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from core.mlp_convergence import extract_mlp_convergence
from core.training_config import TrainingPreset, get_training_preset

DEFAULT_SEEDS=(0,1,2,42,123)

@dataclass(frozen=True)
class AdaptiveMLPProfile:
    n_samples:int; n_features:int; n_classes:int; imbalance_ratio:float
    small_sample:bool; wide:bool; params:Dict[str,Any]
    warnings:tuple[str,...]=()
    def to_dict(self): return asdict(self)


def adaptive_mlp_profile(n_samples:int,n_features:int,n_classes:int,*,imbalance_ratio:float=1.0,
                         random_state:int=42,preset:Optional[TrainingPreset]=None)->AdaptiveMLPProfile:
    p=preset or get_training_preset('scientific')
    n=max(2,int(n_samples)); d=max(1,int(n_features)); k=max(2,int(n_classes))
    small=n < p.small_sample_threshold
    wide=d >= max(20,int(n*p.wide_feature_ratio))
    h1=int(np.clip(round(np.sqrt(max(4,d))*8),p.hidden_units_min,p.hidden_units_max))
    h2=int(np.clip(round(h1/2),p.hidden_units_min,max(p.hidden_units_min,p.hidden_units_max//2)))
    alpha=p.strong_regularization_alpha if wide else p.default_regularization_alpha
    notes=[]
    if small:
        params=dict(hidden_layer_sizes=(h1,) if wide else (h1,h2),activation='relu',solver='lbfgs',alpha=alpha,
                    max_iter=max(1500,p.mlp_max_iter),tol=1e-5,early_stopping=False,random_state=random_state)
    else:
        min_val=max(0.1,(p.min_validation_samples_per_class*k)/n)
        validation_fraction=float(min(0.30,max(p.validation_fraction,min_val)))
        params=dict(hidden_layer_sizes=(h1,h2),activation='relu',solver='adam',alpha=alpha,
                    learning_rate_init=5e-4,max_iter=max(1000,p.mlp_max_iter),early_stopping=True,
                    validation_fraction=validation_fraction,n_iter_no_change=max(20,p.n_iter_no_change),
                    tol=1e-4,random_state=random_state)
    if wide: notes.append('features_maior_ou_igual_amostras_regularizacao_forte')
    if imbalance_ratio>=3: notes.append('classes_desbalanceadas_usar_balanced_accuracy_macro_f1')
    return AdaptiveMLPProfile(n,d,k,float(imbalance_ratio),small,wide,params,tuple(notes))


def build_mlp_pipeline(*,random_state:int=42,small_dataset:Optional[bool]=None,n_samples:Optional[int]=None,
                       n_features:Optional[int]=None,n_classes:Optional[int]=None,imbalance_ratio:float=1.0,
                       preset:Optional[TrainingPreset]=None,**overrides)->Pipeline:
    # small_dataset é mantido como alias de compatibilidade.
    if n_samples is None:
        n_samples=100 if small_dataset else 1000
    if n_features is None: n_features=20
    if n_classes is None: n_classes=2
    profile=adaptive_mlp_profile(n_samples,n_features,n_classes,imbalance_ratio=imbalance_ratio,
                                 random_state=random_state,preset=preset)
    params=dict(profile.params); params.update(overrides)
    # parâmetros específicos de adam não são válidos/relevantes para lbfgs.
    if params.get('solver')=='lbfgs':
        params.pop('learning_rate_init',None); params['early_stopping']=False
    pipe=Pipeline([('scaler',StandardScaler()),('mlp',MLPClassifier(**params))])
    pipe.BIURI_ADAPTIVE_PROFILE=profile.to_dict()
    return pipe


def build_mlp_for_data(X,y,*,random_state:int=42,preset:Optional[TrainingPreset]=None,**overrides):
    X=np.asarray(X); y=np.asarray(y); _,counts=np.unique(y,return_counts=True)
    ratio=float(counts.max()/max(1,counts.min())) if len(counts) else 1.0
    return build_mlp_pipeline(random_state=random_state,n_samples=len(y),n_features=X.shape[1],
                              n_classes=len(counts),imbalance_ratio=ratio,preset=preset,**overrides)


def signal_capacity_audit(estimator,X,y,*,margin:float=0.03,cv_folds:int=5,random_state:int=42)->Dict[str,Any]:
    X=np.asarray(X,dtype=float); y=np.asarray(y); _,counts=np.unique(y,return_counts=True)
    if len(counts)<2 or counts.min()<2: return {'status':'insufficient_data','capable':False}
    folds=max(2,min(int(cv_folds),int(counts.min())))
    cv=StratifiedKFold(folds,shuffle=True,random_state=random_state)
    dummy=DummyClassifier(strategy='most_frequent')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model_scores=cross_val_score(estimator,X,y,cv=cv,scoring='balanced_accuracy')
        dummy_scores=cross_val_score(dummy,X,y,cv=cv,scoring='balanced_accuracy')
    delta=float(np.mean(model_scores)-np.mean(dummy_scores))
    return {'status':'ok' if delta>=margin else 'sem_capacidade_preditiva','capable':bool(delta>=margin),
            'model_cv_balanced_accuracy':float(np.mean(model_scores)),'dummy_cv_balanced_accuracy':float(np.mean(dummy_scores)),
            'margin':float(margin),'delta':delta,'folds':folds}


def evaluate_mlp_seeds(builder,X_train,y_train,X_eval,y_eval,metric_fn,seeds:Iterable[int]=DEFAULT_SEEDS)->Dict[str,Any]:
    rows=[]
    for seed in seeds:
        model=builder(int(seed)); model.fit(X_train,y_train); pred=model.predict(X_eval)
        rows.append({'seed':int(seed),'score':float(metric_fn(y_eval,pred)),'convergence':extract_mlp_convergence(model)})
    scores=np.asarray([r['score'] for r in rows]); conv=[r['convergence'].get('converged') is True for r in rows]
    return {'runs':rows,'mean':float(scores.mean()),'std':float(scores.std()),'min':float(scores.min()),'max':float(scores.max()),'convergence_rate':float(np.mean(conv))}

__all__=['AdaptiveMLPProfile','adaptive_mlp_profile','build_mlp_pipeline','build_mlp_for_data','signal_capacity_audit','evaluate_mlp_seeds','DEFAULT_SEEDS']
