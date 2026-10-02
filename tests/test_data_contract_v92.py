import numpy as np
import pandas as pd
import pytest
from core.data_contract import build_data_contract
from core.scientific_errors import DataContractError


def codes(c): return [w['code'] for w in c.warnings]

def test_missing_formats_and_target_missing_recorded():
    df=pd.DataFrame({'x':[1,'?',None,'',5,'NA','N/A','null'], 'cat':['a']*8, 'y':['A','A','B','B','A','A','B',None]})
    c=build_data_contract(df,'y')
    assert c.removed_target_missing_rows == [7]
    x=next(x for x in c.columns if x.name=='x')
    assert x.missing_rate > .5

def test_high_cardinality_id_constant_dates_and_names():
    n=30
    df=pd.DataFrame(np.c_[np.arange(n), np.ones(n), np.tile([0,1],15)], columns=['id','const','y'])
    df['when']=pd.date_range('2025-01-01', periods=n).astype(str)
    c=build_data_contract(df,'y')
    by={x.name:x for x in c.columns}
    assert by['id'].inferred_type=='id_like'
    assert by['const'].inferred_type=='constante'
    assert by['when'].inferred_type=='data_hora'

def test_rare_target_class_errors():
    with pytest.raises(DataContractError, match='menos de 2'):
        build_data_contract(pd.DataFrame({'x':[1,2,3], 'y':['a','a','b']}),'y')

def test_leakage_named_and_text_numeric_labels_supported():
    y=np.tile(['sim','nao'],25)
    df=pd.DataFrame({'signal':y.copy(),'x':np.arange(50)%3,'y':y})
    c=build_data_contract(df,'y')
    leak=[w for w in c.warnings if w['code']=='possible_target_leakage']
    assert any(w['column']=='signal' for w in leak)
    c2=build_data_contract(df.assign(y=(np.arange(50)%2)), 'y')
    assert c2.target_confirmed

def test_unconfirmed_target_and_duplicate_empty_column_names():
    df=pd.DataFrame([[1,2,'a'],[2,3,'b'],[3,4,'a'],[4,5,'b']], columns=['','x','x'])
    c=build_data_contract(df,2)
    assert 'empty_column_names' in codes(c)
    assert 'duplicate_column_names' in codes(c)
    c2=build_data_contract(pd.DataFrame({'a':[1,2,3,4], 'class':['x','x','y','y']}),None)
    assert c2.status=='alvo_nao_confirmado'

def test_imbalance_and_wide_warnings():
    y=['a']*19+['b']*2
    d={f'x{i}':np.arange(21) for i in range(25)}; d['y']=y
    c=build_data_contract(pd.DataFrame(d),'y')
    assert 'class_imbalance' in codes(c)
    assert 'wide_dataset' in codes(c)
