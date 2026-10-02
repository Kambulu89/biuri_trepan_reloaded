"""Pré-processamento agnóstico e ajustado exclusivamente nos dados de treino."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.data_contract import DataContract
from core.scientific_errors import PreprocessingError


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        self.maps_ = []
        for col in frame.columns:
            s = frame[col].astype(object).where(frame[col].notna(), "__em_falta__").astype(str)
            self.maps_.append((s.value_counts(normalize=True)).to_dict())
        self.n_features_in_ = frame.shape[1]
        return self
    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        out=[]
        for i,col in enumerate(frame.columns):
            s = frame[col].astype(object).where(frame[col].notna(), "__em_falta__").astype(str)
            out.append(s.map(self.maps_[i]).fillna(0.0).to_numpy(dtype=float))
        return np.column_stack(out) if out else np.empty((len(frame),0))
    def get_feature_names_out(self, input_features=None):
        names=list(input_features or [f"x{i}" for i in range(getattr(self,'n_features_in_',0))])
        return np.asarray([f"{x}__frequencia" for x in names], dtype=object)


@dataclass
class PreprocessingViews:
    model: np.ndarray
    feature_names: List[str]
    feature_origins: List[Dict[str, Any]]


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Transformador DataFrame -> matriz numérica com schema estável."""
    def __init__(self, contract: DataContract, *, scale_numeric: bool=False, min_frequency: int|float=2,
                 max_categories: int=50):
        self.contract=contract
        self.scale_numeric=scale_numeric
        self.min_frequency=min_frequency
        self.max_categories=max_categories

    def _normalise(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame=X.copy()
            expected=[c.name for c in self.contract.columns if c.name != self.contract.target]
            # Colunas podem chegar fora de ordem. Renomeações só são aplicadas quando inequívocas.
            missing=[c for c in expected if c not in frame.columns]
            if missing:
                # arrays convertidos antes têm nomes normalizados; DataFrame deve respeitar schema.
                raise PreprocessingError(f"Faltam colunas exigidas pelo treino: {', '.join(missing)}.")
            frame=frame[expected]
        else:
            arr=np.asarray(X, dtype=object)
            if arr.ndim==1: arr=arr.reshape(1,-1)
            expected=[c.name for c in self.contract.columns if c.name != self.contract.target]
            if arr.shape[1]!=len(expected):
                raise PreprocessingError(f"Número de colunas incompatível: esperado {len(expected)}, recebido {arr.shape[1]}.")
            frame=pd.DataFrame(arr, columns=expected)
        tokens={str(x).strip().lower() for x in self.contract.missing_tokens}
        for c in frame.columns:
            frame[c]=frame[c].map(lambda v: np.nan if v is None or (isinstance(v,str) and v.strip().lower() in tokens) else v)
        return frame

    def fit(self, X, y=None):
        frame=self._normalise(X)
        usable=[c for c in self.contract.columns if c.name != self.contract.target and c.treatment!='descartar']
        self.original_feature_names_=[c.name for c in usable]
        numeric=[c.name for c in usable if c.treatment=='usar']
        categorical=[c.name for c in usable if c.treatment=='categorica']
        frequency=[c.name for c in usable if c.treatment=='frequency_encoding']
        # Coerção explícita e accionável para numéricas.
        for c in numeric:
            coerced=pd.to_numeric(frame[c], errors='coerce')
            bad=frame[c].notna() & coerced.isna()
            if bad.any():
                sample=frame.loc[bad,c].iloc[0]
                raise PreprocessingError(f"Valor não numérico na coluna '{c}': {sample!r}.")
            frame[c]=coerced
        for c in categorical + frequency:
            # sklearn 1.8 não permite fill_value string num array float;
            # converter explicitamente para tokens categóricos estáveis.
            frame[c]=frame[c].astype(object).where(frame[c].notna(), '__em_falta__').map(str)
        transformers=[]
        if numeric:
            steps=[('imputer',SimpleImputer(strategy='median', add_indicator=True))]
            if self.scale_numeric: steps.append(('scaler',StandardScaler()))
            transformers.append(('num',Pipeline(steps),numeric))
        if categorical:
            cat_pipe=Pipeline([
                ('imputer',SimpleImputer(strategy='constant', fill_value='__em_falta__')),
                ('onehot',OneHotEncoder(handle_unknown='ignore', min_frequency=self.min_frequency,
                                        max_categories=self.max_categories, sparse_output=False, drop=None)),
            ])
            transformers.append(('cat',cat_pipe,categorical))
        if frequency:
            freq_pipe=Pipeline([
                ('imputer',SimpleImputer(strategy='constant', fill_value='__em_falta__')),
                ('freq',FrequencyEncoder()),
            ])
            transformers.append(('freq',freq_pipe,frequency))
        if not transformers:
            raise PreprocessingError("Nenhuma coluna utilizável ficou disponível após o contrato de dados.")
        self.transformer_=ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)
        self.transformer_.fit(frame, y)
        self.feature_names_out_=list(map(str,self.transformer_.get_feature_names_out()))
        self.feature_origins_=self._build_origins(numeric,categorical,frequency)
        return self

    def _build_origins(self,numeric,categorical,frequency):
        origins=[]
        for name in self.feature_names_out_:
            origin=None
            for c in categorical:
                if name==c or name.startswith(c+'_'):
                    origin=c; break
            if origin is None:
                for c in numeric+frequency:
                    if name==c or name.startswith(c+'__') or name.startswith('missingindicator_'+c):
                        origin=c; break
            origins.append({'name':name,'origin':'original','source_column':origin or name})
        return origins

    def transform(self, X):
        if not hasattr(self,'transformer_'): raise PreprocessingError('O preprocessor ainda não foi ajustado.')
        frame=self._normalise(X)
        numeric=[x.name for x in self.contract.columns if x.treatment=='usar' and x.name!=self.contract.target]
        categorical=[x.name for x in self.contract.columns if x.treatment in {'categorica','frequency_encoding'} and x.name!=self.contract.target]
        for c in numeric:
            coerced=pd.to_numeric(frame[c], errors='coerce')
            bad=frame[c].notna() & coerced.isna()
            if bad.any():
                raise PreprocessingError(f"Valor não numérico na coluna '{c}': {frame.loc[bad,c].iloc[0]!r}.")
            frame[c]=coerced
        for c in categorical:
            frame[c]=frame[c].astype(object).where(frame[c].notna(), '__em_falta__').map(str)
        return np.asarray(self.transformer_.transform(frame), dtype=float)

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self,'feature_names_out_'): raise PreprocessingError('O preprocessor ainda não foi ajustado.')
        return np.asarray(self.feature_names_out_, dtype=object)

    @property
    def feature_origins(self): return list(getattr(self,'feature_origins_',[]))

    def transform_views(self,X):
        arr=self.transform(X)
        return PreprocessingViews(arr,list(self.get_feature_names_out()),self.feature_origins)

    def save(self,path):
        joblib.dump(self,path)
    @classmethod
    def load(cls,path):
        obj=joblib.load(path)
        if not isinstance(obj,cls): raise PreprocessingError('Artefacto não contém um DataPreprocessor válido.')
        return obj


__all__=['DataPreprocessor','PreprocessingViews','FrequencyEncoder']
