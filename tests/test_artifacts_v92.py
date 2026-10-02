import json, numpy as np, pandas as pd, pytest
from sklearn.datasets import make_classification
from core.data_contract import build_data_contract
from core.preprocessing import DataPreprocessor
from core.mlp_factory import build_mlp_for_data
from core.artifacts import save_artifact
from core.inference import load_artifact
from core.model_cache import compute_dataset_hash
from core.scientific_errors import ArtifactCompatibilityError


def trained():
    rng=np.random.default_rng(2); d=pd.DataFrame({'a':rng.normal(size=80),'cat':np.where(np.arange(80)%2,'x','y'),'target':np.tile(['n','s'],40)})
    c=build_data_contract(d,'target'); p=DataPreprocessor(c).fit(d.drop(columns='target')); X=p.transform(d.drop(columns='target')); y=d.target.to_numpy(); m=build_mlp_for_data(X,y,random_state=2); m.fit(X,y); return d,c,p,m

def test_artifact_roundtrip_and_schema_flexibility(tmp_path):
    d,c,p,m=trained(); out=save_artifact(tmp_path/'a',model=m,contract=c,preprocessor=p,classes=m.classes_)
    pred=load_artifact(out); X=d.drop(columns='target')[['cat','a']].copy(); X.loc[0,'cat']='nova'; X.loc[1,'a']=np.nan
    assert len(pred.predict(X))==len(d); assert pred.validate(X)['valid']

def test_incompatible_format_errors(tmp_path):
    d,c,p,m=trained(); out=save_artifact(tmp_path/'a',model=m,contract=c,preprocessor=p,classes=m.classes_); mf=out/'manifest.json'; data=json.loads(mf.read_text()); data['artifact_format_version']='8.0'; mf.write_text(json.dumps(data))
    with pytest.raises(ArtifactCompatibilityError,match='incompatível'): load_artifact(out)

def test_cache_hash_changes_when_one_line_changes():
    X=pd.DataFrame({'x':[1,2,3],'z':['a','b','c']}); y=pd.Series([0,1,0]); a=compute_dataset_hash(X,y); X2=X.copy(); X2.loc[1,'x']=999; b=compute_dataset_hash(X2,y); assert a!=b
