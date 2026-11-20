"""Carregamento agnóstico de dados tabulares para a V9.2."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import csv
import pandas as pd
from core.scientific_errors import DataContractError


def _detect_delimiter(path:Path,encoding:str)->str:
    sample=path.read_text(encoding=encoding,errors='replace')[:8192]
    try:return csv.Sniffer().sniff(sample,delimiters=',;\t|').delimiter
    except csv.Error:return ','

def _arff_declared_schema(arff_meta)->Dict[str,Dict[str,Any]]:
    """Esquema declarado no cabeçalho ARFF: {atributo: {'type': 'nominal'|'numeric'|..., 'values': [ordem declarada]}}."""
    out:Dict[str,Dict[str,Any]]={}
    for name in arff_meta.names():
        kind,values=arff_meta[name]
        if kind=='nominal':
            out[str(name)]={'type':'nominal','values':[v.decode('utf-8') if isinstance(v,bytes) else str(v) for v in values]}
        else:
            out[str(name)]={'type':str(kind)}
    return out

def load_tabular(path,*,encoding:Optional[str]=None,separator:Optional[str]=None)->Tuple[pd.DataFrame,Dict[str,Any]]:
    p=Path(path)
    if not p.exists(): raise DataContractError(f"Ficheiro não encontrado: {p}.")
    ext=p.suffix.lower(); enc=encoding or 'utf-8'
    try:
        if ext in {'.csv','.tsv','.txt'}:
            sep=separator or ('\t' if ext=='.tsv' else _detect_delimiter(p,enc))
            df=pd.read_csv(p,sep=sep,encoding=enc)
            meta={'format':'csv' if ext!='.tsv' else 'tsv','separator':sep,'encoding':enc}
        elif ext=='.arff':
            from scipy.io import arff
            data,arff_meta=arff.loadarff(p)
            df=pd.DataFrame(data)
            for c in df.columns:
                if df[c].dtype==object: df[c]=df[c].map(lambda v:v.decode('utf-8',errors='replace') if isinstance(v,bytes) else v)
            # O tipo DECLARADO no cabeçalho acompanha os dados: um nominal {01,1} nunca pode ser reinterpretado como número.
            schema=_arff_declared_schema(arff_meta)
            df.attrs['arff_schema']=schema
            meta={'format':'arff','arff_schema':schema}
        elif ext=='.parquet':
            try: df=pd.read_parquet(p)
            except ImportError as exc: raise DataContractError('Leitura Parquet requer pyarrow ou fastparquet.') from exc
            meta={'format':'parquet'}
        elif ext in {'.xlsx','.xls'}:
            try: df=pd.read_excel(p)
            except ImportError as exc: raise DataContractError('Leitura Excel requer openpyxl (xlsx) ou engine compatível.') from exc
            meta={'format':'excel'}
        else: raise DataContractError(f"Formato não suportado: {ext}. Use CSV/TSV, ARFF, Parquet ou Excel.")
    except UnicodeDecodeError as exc:
        raise DataContractError(f"Não foi possível ler '{p.name}' com a codificação {enc}. Indique outra codificação.") from exc
    meta.update({'path':str(p.resolve()),'rows':len(df),'columns':len(df.columns)})
    return df,meta

__all__=['load_tabular']
