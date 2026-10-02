"""Formato versionado de artefactos BIURI V9.2."""
from __future__ import annotations
import hashlib, json, platform, subprocess
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd
import sklearn
from core.scientific_errors import ArtifactCompatibilityError

ARTIFACT_FORMAT_VERSION='9.2.0'
MODEL_FILENAME='model.joblib'
MANIFEST_FILENAME='manifest.json'


def _sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def _version_major_minor(v:str)->str:
    parts=str(v).split('.')
    return '.'.join(parts[:2])

def runtime_versions()->Dict[str,str]:
    return {'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__}

def _git_commit(root:Path)->Optional[str]:
    try: return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,stderr=subprocess.DEVNULL,text=True,timeout=2).strip()
    except Exception: return None

def save_artifact(path,*,model,contract=None,preprocessor=None,classes=None,metrics=None,convergence=None,
                  calibration=None,hyperparameters=None,seeds=None,metadata=None,tree=None)->Path:
    out=Path(path); out.mkdir(parents=True,exist_ok=True)
    payload={'model':model,'preprocessor':preprocessor,'classes':list(classes) if classes is not None else None,'tree':tree}
    model_path=out/MODEL_FILENAME; joblib.dump(payload,model_path)
    contract_dict=contract.to_dict() if hasattr(contract,'to_dict') else contract
    schema_payload=json.dumps(contract_dict or {},sort_keys=True,default=str,ensure_ascii=False).encode()
    manifest={
        'artifact_format_version':ARTIFACT_FORMAT_VERSION,'runtime_versions':runtime_versions(),
        'schema_hash':hashlib.sha256(schema_payload).hexdigest(),'data_contract':contract_dict,
        'preprocessor':type(preprocessor).__name__ if preprocessor is not None else None,
        'hyperparameters':hyperparameters or {},'seeds':seeds or {},'metrics_final':metrics or {},
        'convergence':convergence or {},'calibration':calibration or {},'classes':payload['classes'],
        'model_sha256':_sha256_file(model_path),'code_commit':_git_commit(Path(__file__).resolve().parent.parent),
        'metadata':metadata or {},
    }
    (out/MANIFEST_FILENAME).write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    return out

def load_artifact_payload(path,*,strict_versions:bool=True):
    root=Path(path); mp=root/MANIFEST_FILENAME; model_path=root/MODEL_FILENAME
    if not mp.exists() or not model_path.exists(): raise ArtifactCompatibilityError(f"Artefacto incompleto em '{root}'. São necessários manifest.json e model.joblib.")
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    version=str(manifest.get('artifact_format_version',''))
    if _version_major_minor(version)!=_version_major_minor(ARTIFACT_FORMAT_VERSION):
        raise ArtifactCompatibilityError(f"Versão de artefacto incompatível: {version}; esperado {ARTIFACT_FORMAT_VERSION}. Migre ou regenere o artefacto.")
    if _sha256_file(model_path)!=manifest.get('model_sha256'):
        raise ArtifactCompatibilityError('Hash do modelo não corresponde ao manifest; artefacto corrompido ou alterado.')
    if strict_versions:
        saved=manifest.get('runtime_versions',{}); current=runtime_versions()
        # sklearn minor mismatch é relevante para persistência joblib.
        if saved.get('scikit_learn') and _version_major_minor(saved['scikit_learn'])!=_version_major_minor(current['scikit_learn']):
            raise ArtifactCompatibilityError(f"scikit-learn incompatível: artefacto={saved['scikit_learn']}, runtime={current['scikit_learn']}. Use ambiente compatível ou regenere.")
    try: payload=joblib.load(model_path)
    except (ModuleNotFoundError,AttributeError,ImportError) as exc:
        raise ArtifactCompatibilityError(f"Não foi possível carregar o artefacto por incompatibilidade de módulos: {exc}.") from exc
    return payload,manifest

__all__=['ARTIFACT_FORMAT_VERSION','save_artifact','load_artifact_payload','runtime_versions']
