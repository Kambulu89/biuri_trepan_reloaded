import pandas as pd
from core.data_loading import load_tabular

def test_csv_and_tsv(tmp_path):
    (tmp_path/'a.csv').write_text('x;y\n1;a\n2;b\n',encoding='utf-8')
    d,m=load_tabular(tmp_path/'a.csv'); assert list(d.columns)==['x','y'] and m['separator']==';'
    (tmp_path/'b.tsv').write_text('x\ty\n1\ta\n',encoding='utf-8')
    d,m=load_tabular(tmp_path/'b.tsv'); assert m['format']=='tsv'
