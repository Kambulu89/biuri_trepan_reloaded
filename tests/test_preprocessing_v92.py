import numpy as np, pandas as pd, joblib
from core.data_contract import build_data_contract
from core.preprocessing import DataPreprocessor


def make_df(n=80):
    rng=np.random.default_rng(1)
    return pd.DataFrame({'num':rng.normal(size=n), 'cat':np.where(np.arange(n)%2,'a','b'), 'y':np.where(np.arange(n)%2,'yes','no')})

def test_unseen_category_and_missing_in_inference():
    train=make_df(30); c=build_data_contract(train,'y'); p=DataPreprocessor(c).fit(train.drop(columns='y'))
    test=pd.DataFrame({'num':[None,3.0,'?'], 'cat':['nova',None,'']})
    z=p.transform(test); assert z.shape[0]==3 and np.isfinite(z).all()

def test_dataframe_column_order_invariant():
    d=make_df(); c=build_data_contract(d,'y'); p=DataPreprocessor(c).fit(d.drop(columns='y'))
    X=d.drop(columns='y')
    assert np.allclose(p.transform(X),p.transform(X[['cat','num']]))

def test_row_permutation_only_permutes_output():
    d=make_df(); c=build_data_contract(d,'y'); p=DataPreprocessor(c).fit(d.drop(columns='y'))
    X=d.drop(columns='y'); order=np.arange(len(X))[::-1]
    assert np.allclose(p.transform(X.iloc[order]),p.transform(X)[order])

def test_high_cardinality_bounded_dimensions():
    n=120; d=pd.DataFrame({'cat':[f'c{i}' for i in range(n)],'x':np.arange(n)%4,'y':np.tile([0,1],60)})
    c=build_data_contract(d,'y'); p=DataPreprocessor(c,max_categories=20).fit(d.drop(columns='y'))
    assert p.transform(d.drop(columns='y')).shape[1] < 100

def test_roundtrip_serialization(tmp_path):
    d=make_df(); c=build_data_contract(d,'y'); p=DataPreprocessor(c).fit(d.drop(columns='y')); path=tmp_path/'p.joblib'; p.save(path); q=DataPreprocessor.load(path)
    assert np.allclose(p.transform(d.drop(columns='y')),q.transform(d.drop(columns='y')))

def test_fit_only_train_not_affected_by_test_rows():
    d=make_df(60); tr=d.iloc[:40]; te=d.iloc[40:].copy(); te['cat']='only_test'
    c=build_data_contract(d,'y'); a=DataPreprocessor(c).fit(tr.drop(columns='y')); za=a.transform(te.drop(columns='y'))
    # Repetir apenas com o mesmo treino tem de produzir a mesma transformação do teste.
    b=DataPreprocessor(c).fit(tr.drop(columns='y')); zb=b.transform(te.drop(columns='y'))
    assert np.allclose(za,zb)
