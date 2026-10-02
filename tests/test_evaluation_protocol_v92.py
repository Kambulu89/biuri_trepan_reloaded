import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from core.evaluation_protocol import choose_evaluation_strategy, noninferiority_decision, predictive_capacity_gate


def test_small_n_uses_repeated_cv_and_resolution_can_be_inconclusive():
    y=np.tile([0,1],20); s=choose_evaluation_strategy(y); assert s.mode=='repeated_stratified_cv'
    d=noninferiority_decision(.005,configured_tolerance=.01,n_validation=20); assert d['inconclusive'] and d['effective_tolerance']>=.05

def test_imbalance_changes_primary_metric():
    y=np.array([0]*95+[1]*5); assert choose_evaluation_strategy(y).primary_metric=='balanced_accuracy'

def test_model_without_signal_is_flagged():
    rng=np.random.default_rng(0); X=rng.normal(size=(120,6)); y=np.tile([0,1],60)
    gate=predictive_capacity_gate(DummyClassifier(strategy='most_frequent'),X,y,margin=.01,random_state=1)
    assert gate['status']=='sem_capacidade_preditiva'

def test_signal_model_passes_and_seed_is_reproducible():
    X,y=make_classification(n_samples=140,n_features=8,n_informative=6,class_sep=1.8,random_state=2)
    m=LogisticRegression(max_iter=1000)
    a=predictive_capacity_gate(m,X,y,margin=.02,random_state=7); b=predictive_capacity_gate(m,X,y,margin=.02,random_state=7)
    assert a['capable'] and a['model_mean']==b['model_mean']
