import numpy as np
from core.trepan_original import TrepanOriginalClassifier

class Oracle:
    def predict(self,X):
        X=np.asarray(X); return ((X[:,0]>0).astype(int)+(X[:,1]>0).astype(int)+(X[:,2]>0).astype(int)>=2).astype(int)

def test_node_limit_and_high_fidelity_on_m_of_n_rule():
    rng=np.random.default_rng(4); X=rng.normal(size=(300,5)); o=Oracle(); y=o.predict(X)
    t=TrepanOriginalClassifier(max_nodes=15,max_depth=5,max_n=3,max_queries=600,random_state=4).fit(X,y,oracle=o,feature_names=[f'x{i}' for i in range(5)])
    assert t.node_count_<=15
    assert np.mean(t.predict(X)==y)>.85
    assert any(r['n']>=2 for r in t.split_audit_)
