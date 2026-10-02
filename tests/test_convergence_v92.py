from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from core.mlp_convergence import fit_with_convergence, extract_mlp_convergence


def test_max_iter_short_is_detected():
    X,y=make_classification(n_samples=100,n_features=12,random_state=3)
    m=Pipeline([('s',StandardScaler()),('mlp',MLPClassifier(max_iter=1,random_state=3))])
    fit_with_convergence(m,X,y); c=extract_mlp_convergence(m)
    assert c['state']=='max_iter_reached' and c['converged'] is False

def test_early_stopping_is_not_claimed_as_converged():
    X,y=make_classification(n_samples=300,n_features=10,random_state=4)
    m=MLPClassifier(max_iter=200,early_stopping=True,n_iter_no_change=2,random_state=4)
    fit_with_convergence(m,X,y); c=extract_mlp_convergence(m)
    if c['n_iter'] < c['max_iter']:
        assert c['state']=='stopped_early_unverified' and c['converged'] is False

def test_calibrated_classifier_aggregates_clones():
    X,y=make_classification(n_samples=120,n_features=8,random_state=5)
    base=Pipeline([('s',StandardScaler()),('mlp',MLPClassifier(solver='lbfgs',max_iter=200,random_state=5))])
    c=CalibratedClassifierCV(base,cv=3).fit(X,y)
    info=extract_mlp_convergence(c)
    assert info['estimator_count']==3 and len(info['estimators'])==3
