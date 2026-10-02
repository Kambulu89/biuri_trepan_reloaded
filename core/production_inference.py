"""Inferência para bundles de produção BIURI V9.2."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json
import joblib
import numpy as np
import pandas as pd
from core.scientific_errors import ArtifactCompatibilityError, PreprocessingError
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier
from core.c45_j48_tree import C45Classifier

SUPPORTED_FORMATS={"biuri-v9.2-production-2-no-cart", "biuri-v9.2-production-3-adaptive-semantic", "biuri-v9.2-production-4-error-focused-semantic", "biuri-v9.2-production-5-semantic-real-gain"}

class ProductionPredictor:
    def __init__(self,bundle:Dict[str,Any],manifest:Dict[str,Any]):
        self.bundle=bundle; self.manifest=manifest
        self.preprocessor=bundle['preprocessor']; self.mlp=bundle['mlp_original']
        self.original=bundle['trepan_original']; self.reloaded=bundle['trepan_reloaded']; self.c45=bundle.get('c45_native')
        self.semantic_teacher=bundle.get('semantic_teacher')
        self.reloaded_space=bundle.get('reloaded_feature_space','original')
    def validate(self,df:pd.DataFrame)->Dict[str,Any]:
        try:
            self.preprocessor.transform(df)
            return {'valid':True,'issues':[]}
        except Exception as exc:
            return {'valid':False,'issues':[f'{type(exc).__name__}: {exc}']}
    def _z(self,df): return self.preprocessor.transform(df)
    def _z_for(self,z,model):
        # O Reloaded em espaço aumentado precisa das features semânticas, recompostas como no treino.
        if model=='trepan_reloaded' and self.reloaded_space=='augmented':
            if self.semantic_teacher is None: raise ArtifactCompatibilityError('Reloaded aumentado sem o transformador semântico no bundle.')
            return self.semantic_teacher.augment(z)
        return z
    def predict(self,df,*,model='mlp'):
        z=self._z(df)
        mapping={'mlp':self.mlp,'trepan_original':self.original,'trepan_reloaded':self.reloaded}
        if model=='mlp_semantic':
            if self.semantic_teacher is None: raise ArtifactCompatibilityError('Bundle sem professor semântico (MLP+OWL).')
            mapping['mlp_semantic']=self.semantic_teacher
        if model=='c45_native':
            if self.c45 is None: raise ArtifactCompatibilityError('Bundle sem C4.5-Nativo.')
            return self.c45.predict(pd.DataFrame(df).to_numpy(dtype=object))
        if model not in mapping: raise ValueError(f"Modelo desconhecido: {model}.")
        return mapping[model].predict(self._z_for(z,model))
    def predict_proba(self,df,*,model='mlp'):
        z=self._z(df); estimator={'mlp':self.mlp,'trepan_original':self.original,'trepan_reloaded':self.reloaded}.get(model)
        if estimator is None or not hasattr(estimator,'predict_proba'): raise ArtifactCompatibilityError(f'{model} não disponibiliza predict_proba.')
        return estimator.predict_proba(self._z_for(z,model))
    def explain(self,row,*,model='trepan_reloaded'):
        frame=row if isinstance(row,pd.DataFrame) else pd.DataFrame([row])
        z=self._z_for(self._z(frame),model); estimator=self.reloaded if model=='trepan_reloaded' else self.original
        if hasattr(estimator,'export_rules'):
            return {'prediction':str(estimator.predict(z)[0]),'rules':estimator.export_rules(),'model':model}
        return {'prediction':str(estimator.predict(z)[0]),'model':model}

def _validate_tree_families(bundle: Dict[str, Any]) -> None:
    original=bundle.get('trepan_original')
    reloaded=bundle.get('trepan_reloaded')
    c45=bundle.get('c45_native')
    if not isinstance(original, TrepanOriginalClassifier):
        raise ArtifactCompatibilityError(
            f"Artefacto recusado: trepan_original pertence à família {type(original).__module__}.{type(original).__name__}; "
            "produção exige TrepanOriginalClassifier histórico."
        )
    if not isinstance(reloaded, TrepanReloadedClassifier):
        raise ArtifactCompatibilityError(
            f"Artefacto recusado: trepan_reloaded pertence à família {type(reloaded).__module__}.{type(reloaded).__name__}; "
            "produção exige TrepanReloadedClassifier histórico."
        )
    if c45 is not None and not isinstance(c45, C45Classifier):
        raise ArtifactCompatibilityError(
            f"Artefacto recusado: baseline C4.5 inválido ({type(c45).__module__}.{type(c45).__name__})."
        )


def load_production_bundle(path)->ProductionPredictor:
    p=Path(path)
    if p.is_dir(): bundle_path=p/'production_bundle.joblib'; manifest_path=p/'manifest.json'
    else: bundle_path=p; manifest_path=p.with_name('manifest.json')
    if not bundle_path.exists(): raise ArtifactCompatibilityError(f'Bundle não encontrado: {bundle_path}.')
    if not manifest_path.exists(): raise ArtifactCompatibilityError(f'manifest.json não encontrado junto do bundle: {manifest_path}.')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    fmt=manifest.get('format_version')
    if fmt not in SUPPORTED_FORMATS: raise ArtifactCompatibilityError(f"Formato de artefacto incompatível: {fmt!r}. Suportados: {sorted(SUPPORTED_FORMATS)}.")
    bundle=joblib.load(bundle_path)
    if bundle.get('format_version')!=fmt: raise ArtifactCompatibilityError('Manifest e bundle têm versões de formato diferentes.')
    _validate_tree_families(bundle)
    if manifest.get('tree_runtime_policy') not in (None,'historical_trepan_only_no_cart'):
        raise ArtifactCompatibilityError('Política de família de árvore incompatível com a build de produção.')
    return ProductionPredictor(bundle,manifest)

__all__=['ProductionPredictor','load_production_bundle']
