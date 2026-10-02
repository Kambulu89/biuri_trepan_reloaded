import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from core.pipeline_v92 import train_dataset
from core.inference import load_artifact


def frame(n=80,p=8,classes=2,weights=None,seed=1):
    X,y=make_classification(n_samples=n,n_features=p,n_informative=max(2,min(p-1,5)),n_redundant=0,n_classes=classes,
                            n_clusters_per_class=1,weights=weights,class_sep=1.8,random_state=seed)
    d=pd.DataFrame(X,columns=[f'x{i}' for i in range(p)]); d['target']=[f'c{v}' for v in y]; return d

def test_small_mixed_missing_id_constant_roundtrip(tmp_path):
    d=frame(60,6); d['cat']=np.where(np.arange(len(d))%2,'A','B'); d.loc[0,'x0']=np.nan; d.loc[1,'cat']='?'; d['id']=[f'id{i}' for i in range(len(d))]; d['const']=1
    r=train_dataset(d,target='target',seed=3,out=tmp_path/'artifact',max_trepan_nodes=15)
    assert r.trepan_original is r.trepan_reloaded
    p=load_artifact(tmp_path/'artifact'); q=d.drop(columns='target').iloc[:3][list(reversed(d.drop(columns='target').columns))].copy(); q.loc[q.index[0],'cat']='NOVA'; assert len(p.predict(q))==3

def test_wide_120x400_runs_and_reports_signal_state():
    d=frame(120,400,seed=4); r=train_dataset(d,target='target',seed=4,max_trepan_nodes=7)
    assert r.metrics['signal_gate']['status'] in {'ok','sem_capacidade_preditiva'}

def test_imbalanced_and_multiclass_run():
    a=frame(120,10,2,weights=[.9,.1],seed=5); b=frame(150,12,5,seed=6)
    assert train_dataset(a,target='target',seed=5,max_trepan_nodes=7).metrics['mlp']['balanced_accuracy']>=0
    assert train_dataset(b,target='target',seed=6,max_trepan_nodes=7).metrics['mlp']['macro_f1']>=0
