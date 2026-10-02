"""Pipeline headless V9.2 agnóstico ao dataset, sem dependência obrigatória de OWL."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from core.data_contract import build_data_contract, DataContract
from core.preprocessing import DataPreprocessor
from core.mlp_factory import build_mlp_for_data, signal_capacity_audit
from core.mlp_convergence import fit_with_convergence, extract_mlp_convergence
from core.c45_j48_tree import C45Classifier
from core.trepan_original import TrepanOriginalClassifier
from core.evaluation_protocol import classification_metrics
from core.artifacts import save_artifact
from core.scientific_errors import DataContractError, InsufficientSignalError

@dataclass
class TrainingResult:
    contract:DataContract; preprocessor:DataPreprocessor; mlp:Any; c45:Any; trepan_original:Any; trepan_reloaded:Any
    metrics:Dict[str,Any]; split:Dict[str,Any]; artifact_path:Optional[str]=None


def _clean_target_missing(df,contract):
    out=df.copy(); target=contract.target; tokens={x.lower() for x in contract.missing_tokens}
    mask=out[target].map(lambda v: v is None or (isinstance(v,str) and v.strip().lower() in tokens) or pd.isna(v))
    return out.loc[~mask].reset_index(drop=True)


def train_dataset(df:pd.DataFrame,*,target,seed:int=42,out:Optional[str|Path]=None,test_size:float=0.25,
                  max_trepan_nodes:int=31,require_signal:bool=False)->TrainingResult:
    contract=build_data_contract(df,target)
    if not contract.target_confirmed: raise DataContractError('A coluna-alvo tem de ser confirmada explicitamente antes do treino.')
    clean=_clean_target_missing(df,contract)
    y=clean[contract.target].to_numpy(); X=clean.drop(columns=[contract.target])
    idx=np.arange(len(clean)); _,counts=np.unique(y,return_counts=True)
    strat=y if counts.min()>=2 else None
    tr,te=train_test_split(idx,test_size=test_size,random_state=seed,stratify=strat)
    Xtr,Xte=X.iloc[tr].copy(),X.iloc[te].copy(); ytr,yte=y[tr],y[te]
    pre=DataPreprocessor(contract,scale_numeric=False).fit(Xtr,ytr)
    Ztr,Zte=pre.transform(Xtr),pre.transform(Xte)
    mlp=build_mlp_for_data(Ztr,ytr,random_state=seed); fit_with_convergence(mlp,Ztr,ytr)
    signal=signal_capacity_audit(mlp,Ztr,ytr,margin=.02,cv_folds=3,random_state=seed)
    if require_signal and not signal['capable']: raise InsufficientSignalError('O oráculo MLP não superou o baseline maioritário na validação interna.')
    # C4.5-Nativo vê apenas atributos originais, nunca onto_*.
    c45=C45Classifier(random_state=seed).fit(Xtr.to_numpy(dtype=object),ytr)
    # Configuração adaptativa para uso normal. O algoritmo é o TREPAN histórico,
    # mas os presets exactos NIPS/Thesis são opt-in via TrepanOriginalClassifier.from_preset().
    trepan_min_sample=max(len(Ztr), min(1000, max(120, len(Ztr)*3)))
    trepan=TrepanOriginalClassifier(
        max_nodes=max_trepan_nodes,max_depth=8,max_n=3,beam_width=2,
        min_sample=trepan_min_sample,
        max_queries=max(1000, trepan_min_sample*max(2, min(max_trepan_nodes, 7))),
        random_state=seed,
    )
    trepan.fit(Ztr,oracle=mlp,feature_names=list(pre.get_feature_names_out()))
    # Sem OWL o Reloaded entra explicitamente em modo espelhado: não há componente semântica a introduzir diferença.
    reloaded=trepan
    pred_mlp=mlp.predict(Zte); pred_c45=c45.predict(Xte.to_numpy(dtype=object)); pred_t=trepan.predict(Zte)
    metrics={'mlp':classification_metrics(yte,pred_mlp),'c45_native':classification_metrics(yte,pred_c45),
             'trepan_original':classification_metrics(yte,pred_t),'trepan_reloaded_no_owl':classification_metrics(yte,pred_t),
             'trepan_fidelity':float(np.mean(pred_t==pred_mlp)),'signal_gate':signal,'convergence':extract_mlp_convergence(mlp),
             'test_consulted_after_selection_only':True}
    split={'seed':seed,'train_indices':tr.tolist(),'test_indices':te.tolist(),'test_size':test_size}
    artifact_path=None
    if out is not None:
        path=save_artifact(out,model=mlp,contract=contract,preprocessor=pre,classes=getattr(mlp,'classes_',None),metrics=metrics,
                           convergence=metrics['convergence'],seeds={'root':seed},tree=trepan,
                           metadata={'feature_origins':pre.feature_origins,'mirrored_no_ontology':True,'external_test_used_for_selection':False})
        artifact_path=str(path)
    return TrainingResult(contract,pre,mlp,c45,trepan,reloaded,metrics,split,artifact_path)

__all__=['TrainingResult','train_dataset']
