from pathlib import Path
import os
import pandas as pd


def test_bootstrap_does_not_change_cwd(tmp_path):
    from counterfactuals import _bootstrap
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        before = Path.cwd()
        root = _bootstrap.setup()
        assert Path.cwd() == before
        assert root.name == 'counterfactuals'
    finally:
        os.chdir(previous)


def test_hepatitis_preparation_uses_first_column_as_target_and_preserves_nan(tmp_path, monkeypatch):
    from counterfactuals.datasets import preparar_datasets as prep
    original = tmp_path / 'originales'
    original.mkdir()
    # classe, f1, histology; inclui ? que não pode ser imputado globalmente.
    (original/'hepatitis.data').write_text('1,10,2\n2,?,1\n1,30,2\n', encoding='utf-8')
    monkeypatch.setattr(prep, 'INPUT_DIR', original)
    monkeypatch.setattr(prep, 'OUTPUT_DIR', tmp_path)
    prep.process_hepatitis()
    out = pd.read_csv(tmp_path/'hepatitis.csv')
    assert out['class'].tolist() == [1,2,1]
    assert pd.isna(out.loc[1,'feature_1'])
    assert out['feature_2'].tolist() == [2,1,2]
