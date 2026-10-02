import numpy as np
from sklearn.datasets import make_classification
from core.mlp_factory import adaptive_mlp_profile, build_mlp_for_data, signal_capacity_audit

def test_small_uses_lbfgs_without_early_stopping():
    p=adaptive_mlp_profile(40,10,2); assert p.params['solver']=='lbfgs' and p.params['early_stopping'] is False

def test_large_uses_adam_and_wide_strong_regularization():
    a=adaptive_mlp_profile(1000,20,2); assert a.params['solver']=='adam' and a.params['early_stopping']
    w=adaptive_mlp_profile(120,400,2); assert w.wide and w.params['alpha']>=.01

def test_signal_dataset_beats_dummy():
    X,y=make_classification(n_samples=120,n_features=15,n_informative=10,class_sep=2.0,random_state=4)
    m=build_mlp_for_data(X,y,random_state=4); audit=signal_capacity_audit(m,X,y,margin=.02,cv_folds=3,random_state=4)
    assert audit['capable'], audit

def test_multiclass_and_extreme_scale_train():
    X,y=make_classification(n_samples=120,n_features=12,n_informative=8,n_classes=4,n_clusters_per_class=1,random_state=2)
    X[:,0]*=1e6; m=build_mlp_for_data(X,y,random_state=2); m.fit(X,y); assert len(m.predict(X[:4]))==4
