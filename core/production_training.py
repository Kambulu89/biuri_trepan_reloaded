"""Pipeline de produção V9.2: dataset-agnostic, OWL opcional e protocolo pareado.

Nenhum nome de dataset é permitido neste módulo. A OWL, quando fornecida, só
pode alterar o braço Reloaded através da estrutura semântica; o oráculo, seed,
orçamento e dados permanecem iguais aos do TREPAN Original.
"""
from __future__ import annotations
import dataclasses
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib, json, platform, sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.data_contract import build_data_contract
from core.preprocessing import DataPreprocessor
from core.mlp_factory import build_mlp_for_data
from core.scientific_experiment_contract import freeze_oracle, oracle_id_of, OracleContractViolation, ORACLE_BUILDERS
from core.mlp_convergence import fit_with_convergence, extract_mlp_convergence
from core.c45_j48_tree import C45Classifier
from core.evaluation_protocol import classification_metrics
from core.controlled_trepan_experiment import (
    fit_reloaded_arm, OriginalOracleProjection,
    ControlledTrepanConfig, fit_controlled_trepan_pair,
    evaluate_controlled_trepan_pair, oracle_health_gate,
)
from core.ontology_quality import OntologyQualityGate
from core.ontology_reasoner import run_owl_reasoner
from core.ontology_processor import OntologyProcessor
from core.ontology_stage_status import build_ontology_stage_status
from core.semantic_enrichment import EnrichmentConfig, evaluate_semantic_enrichment
from core.semantic_attribution import (ATTRIBUTED, AttributionConfig, AugmentedPredictor, evaluate_attribution,
    make_random_derived_teacher, test_attribution_report)
from core.semantic_controls import shuffle_accepted_matches
from core.semantic_metadata import apply_family_relatedness, extend_semantic_inputs
from core.semantic_teacher import build_semantic_teacher, decide_semantic_teacher, make_consistency_projector, teacher_inputs_available
from core.ontology_semantic_graph import OntologySemanticGraph
from core.semantic_contribution_gate import SemanticContributionGate
from core.trepan_scientific_tuning import ScientificTrepanSearchConfig, tune_scientific_trepan
from core.scientific_validation import build_scientific_validation_report
from core.scientific_errors import DataContractError, OntologyQualityError


def _hash_dataframe(df: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _sha256_file(path: Optional[str | Path]) -> Optional[str]:
    if not path: return None
    p=Path(path)
    if not p.exists(): return None
    h=hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def _drop_missing_target(df, target, missing_tokens):
    tokens={str(x).strip().lower() for x in missing_tokens}
    def missing(v):
        return v is None or pd.isna(v) or (isinstance(v,str) and v.strip().lower() in tokens)
    return df.loc[~df[target].map(missing)].reset_index(drop=True)


def _load_ontology(path):
    try:
        from owlready2 import get_ontology
    except ImportError as exc:
        raise OntologyQualityError("OWL requer owlready2. Instale o extra de ontologia antes do treino.") from exc
    return get_ontology(str(Path(path).resolve())).load()


def _semantic_inputs(pre: DataPreprocessor, quality: Dict[str,Any], graph: OntologySemanticGraph):
    matches={m['feature']:m for m in quality.get('matches',[]) if m.get('accepted')}
    names=list(map(str,pre.get_feature_names_out()))
    sources=[str(x.get('source_column')) for x in pre.feature_origins]
    weights=[]; groups=[]; entities=[]
    for src in sources:
        match=matches.get(src)
        score=float(match.get('score',0.0)) if match else 0.0
        # Peso absoluto moderado; mesmo quando todos são iguais, grupos/grafo
        # continuam a fornecer informação relacional ao m-of-n.
        weights.append(1.0 + 0.25*score if match else 1.0)
        groups.append(graph.primary_group(src) if match else None)
        entities.append(graph.feature_to_entity.get(src) if match else None)
    n=len(names); matrix=np.zeros((n,n),dtype=float)
    for i,a in enumerate(entities):
        matrix[i,i]=1.0
        if not a: continue
        for j,b in enumerate(entities):
            if i!=j and b: matrix[i,j]=graph.relatedness(a,b)
    return names, np.asarray(weights,float), groups, matrix


def _semantic_context(ontology, quality, pre, Xtr, original_feature_names):
    """Grafo + entradas semânticas do TREPAN (pesos, grupos, relatedness, entidades) para um conjunto de matches."""
    graph=OntologySemanticGraph.from_ontology(ontology,accepted_matches=quality['matches'])
    model_names,weights,groups,matrix=_semantic_inputs(pre,quality,graph)
    accepted_matches=[m for m in quality['matches'] if m.get('accepted')]
    sources=[str(x.get('source_column')) for x in pre.feature_origins]
    entities=[graph.feature_to_entity.get(src) for src in sources]
    # Famílias de medida declaradas na OWL (mean/error/worst da mesma grandeza)
    # tornam essas features relacionadas para o m-of-n; sem famílias nada muda.
    try:
        family_proc=OntologyProcessor(ontology).fit(
            Xtr[original_feature_names],accepted_matches=accepted_matches,log=False)
        column_family={src:spec['family'] for spec in family_proc.feature_specs_
                       if spec.get('family') for src in spec.get('sources',[])}
    except Exception:
        column_family={}
    matrix=apply_family_relatedness(matrix,sources,column_family)
    return {'graph':graph,'names':model_names,'weights':weights,'groups':groups,'matrix':matrix,'entities':entities}


def _select_reloaded_mode(acfg,augmented,teacher,oracle,Ztr,ytr,cfg,ontology,quality,pre,Xtr,original_feature_names,real_ctx,seed):
    """Escolhe o modo do Reloaded pelo gate de atribuição: augmented -> original_space -> neutral.

    Devolve ``(relatório, modo, {modo: fábrica_de_controlos})``. Tudo com dados de treino.
    """
    n=len(real_ctx['names']); candidates=[]
    ctx_cache={}

    def shuffled_ctx(sd):
        if sd not in ctx_cache:
            q=dict(quality); q['matches']=shuffle_accepted_matches(quality['matches'],seed=int(seed)*100003+int(sd))
            ctx_cache[sd]=_semantic_context(ontology,q,pre,Xtr,original_feature_names)
        return ctx_cache[sd]

    def original_space_real(Z,y):
        return fit_reloaded_arm(Z,oracle=oracle,feature_names=real_ctx['names'],config=cfg,
            semantic_feature_weights=real_ctx['weights'],semantic_feature_groups=real_ctx['groups'],
            semantic_relatedness_matrix=real_ctx['matrix'],ontology_graph=real_ctx['graph'],
            semantic_feature_entities=real_ctx['entities'])

    def original_space_control(sd):
        c=shuffled_ctx(sd)
        return lambda Z,y: fit_reloaded_arm(Z,oracle=oracle,feature_names=c['names'],config=cfg,
            semantic_feature_weights=c['weights'],semantic_feature_groups=c['groups'],
            semantic_relatedness_matrix=c['matrix'],ontology_graph=c['graph'],semantic_feature_entities=c['entities'])

    cfg_aug=dataclasses.replace(cfg,mirror_when_no_semantic_effect=False)
    def aug_fit(tch):
        names,w,g,e,r,_=extend_semantic_inputs(real_ctx['names'],real_ctx['weights'],real_ctx['groups'],
            real_ctx['entities'],real_ctx['matrix'],tch.processor,tch.selected_features)
        proj=OriginalOracleProjection(oracle,tuple(range(n)))
        return lambda Z,y: AugmentedPredictor(fit_reloaded_arm(tch.augment(Z),oracle=proj,feature_names=names,
            config=cfg_aug,semantic_feature_weights=w,semantic_feature_groups=g,semantic_relatedness_matrix=r,
            query_projector=make_consistency_projector(tch,n),ontology_graph=real_ctx['graph'],
            semantic_feature_entities=e),tch)

    def aug_control(sd):
        return aug_fit(make_random_derived_teacher(teacher,Ztr,seed=int(seed)*100003+int(sd)))

    if augmented:
        candidates.append(('augmented',aug_fit(teacher),aug_control))
    candidates.append(('original_space',original_space_real,original_space_control))
    reports=[]; chosen='neutral'; variants={}
    for mode,real_variant,control_variant in candidates:
        rep=evaluate_attribution(Ztr,ytr,oracle,real_variant,control_variant,acfg)
        rep['mode']=mode; reports.append(rep); variants[mode]=control_variant
        if rep['attributed']:
            chosen=mode; break
    return {'status':ATTRIBUTED if chosen!='neutral' else 'NOT_ATTRIBUTED','candidates':reports,
            'config':dataclasses.asdict(acfg)},chosen,variants


def train_production_dataframe(
    df: pd.DataFrame, *, target, out_dir: str | Path,
    seed: int=42, test_size: float=0.25, owl_path: Optional[str|Path]=None,
    ontology=None, reasoner_report: Optional[Dict[str,Any]]=None,
    require_reasoner: bool=True, trepan_config: Optional[ControlledTrepanConfig]=None,
    scientific_tuning: bool=True,
    trepan_search: Optional[ScientificTrepanSearchConfig]=None,
    semantic_enrichment: Optional[EnrichmentConfig]=None,
    use_semantic_teacher: bool=True,
    teacher_min_evidence: Optional[str]="strong",
    augment_reloaded_space: bool=True,
    semantic_control: Optional[str]=None,
    semantic_control_seed: int=0,
    trepan_overrides: Optional[Dict[str,Any]]=None,
    semantic_attribution: Optional[AttributionConfig]=None,
    oracle_builder: str='factory',
    return_artifacts: bool=False,
) -> Dict[str,Any]:
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    contract=build_data_contract(df,target)
    if not contract.target_confirmed:
        raise DataContractError("A coluna-alvo deve ser confirmada explicitamente.")
    clean=_drop_missing_target(df,contract.target,contract.missing_tokens)
    y=clean[contract.target].to_numpy(); X=clean.drop(columns=[contract.target])
    _,counts=np.unique(y,return_counts=True)
    if len(counts)<2 or counts.min()<2: raise DataContractError("O alvo precisa de duas classes com pelo menos duas amostras por classe.")
    idx=np.arange(len(clean)); tr,te=train_test_split(idx,test_size=test_size,random_state=seed,stratify=y)
    Xtr,Xte=X.iloc[tr].copy(),X.iloc[te].copy(); ytr,yte=y[tr],y[te]
    pre=DataPreprocessor(contract,scale_numeric=False).fit(Xtr,ytr)
    Ztr,Zte=pre.transform(Xtr),pre.transform(Xte); model_names=list(map(str,pre.get_feature_names_out()))
    model_names_original=list(model_names)
    # Construtor do MLP registado no contrato científico ('factory' = produção; 'robust' = caminho robusto usado pela GUI).
    if oracle_builder=='factory':
        mlp=build_mlp_for_data(Ztr,ytr,random_state=seed); fit_with_convergence(mlp,Ztr,ytr)
    else:
        if oracle_builder not in ORACLE_BUILDERS: raise ValueError(f"oracle_builder desconhecido: {oracle_builder!r}")
        mlp=ORACLE_BUILDERS[oracle_builder](Ztr,ytr,seed)
    health=oracle_health_gate(mlp,Ztr,ytr,random_state=seed)
    c45=C45Classifier(random_state=seed).fit(Xtr.to_numpy(dtype=object),ytr)

    cfg=trepan_config or ControlledTrepanConfig(
        max_nodes=31,max_depth=31,min_samples_leaf=2,
        min_sample=max(len(Ztr),min(1000,max(120,len(Ztr)*3))),
        max_n=3,beam_width=2,max_features_per_node=min(20,max(4,Ztr.shape[1])),
        max_queries=2000,random_state=seed,
    )

    if trepan_overrides:  # ex.: ablação de componentes do Reloaded (alpha, beta, gain_criterion, ...)
        cfg=dataclasses.replace(cfg,**trepan_overrides)

    quality=None; graph=None; semantic_weights=None; semantic_groups=None; relation_matrix=None
    semantic_entities=None; original_feature_names=[]
    if owl_path or ontology is not None:
        ontology=ontology or _load_ontology(owl_path)
        if reasoner_report is None:
            reasoner_report=run_owl_reasoner(ontology) if require_reasoner else {"executed":False,"consistent":None}
        original_feature_names=[c.name for c in contract.columns if c.name!=contract.target and c.treatment!='descartar']
        q=OntologyQualityGate().evaluate(original_feature_names,ontology,reasoner_report=reasoner_report,require_reasoner=require_reasoner)
        quality=q.to_dict()
        if semantic_control not in (None,'shuffled_semantics'):
            raise ValueError("semantic_control deve ser None ou 'shuffled_semantics'.")
        if semantic_control=='shuffled_semantics' and quality['accepted']:
            # Controlo negativo: mesmas entidades e mesmo número de features, mas a atribuição
            # feature->entidade é baralhada ANTES de construir grafo, pesos, grupos, relatedness,
            # famílias, profundidades e restrições de domínio. Tudo o que vem da OWL perde o significado.
            quality['matches']=shuffle_accepted_matches(quality['matches'],seed=semantic_control_seed)
            quality['semantic_control']='shuffled_semantics'
        if quality['accepted']:
            ctx=_semantic_context(ontology,quality,pre,Xtr,original_feature_names)
            graph=ctx['graph']; model_names=ctx['names']; semantic_weights=ctx['weights']
            semantic_groups=ctx['groups']; relation_matrix=ctx['matrix']; semantic_entities=ctx['entities']

    # Enriquecimento semântico do MLP (opt-in): só treino de desenvolvimento, nunca o teste.
    # Corre ANTES do TREPAN porque o professor escolhido alimenta a procura de capacidade e
    # o treino dos dois braços (Original e Reloaded usam sempre o MESMO professor).
    enrichment=None; enrichment_result=None
    if semantic_enrichment is not None and quality:
        try:
            enrichment_result=evaluate_semantic_enrichment(
                Xtr[original_feature_names].reset_index(drop=True),ytr,ontology,
                quality_report=quality,reasoner_report=reasoner_report,config=semantic_enrichment)
            enrichment=enrichment_result.report
        except Exception as exc:  # a avaliação nunca pode quebrar o treino de produção
            enrichment_result=None
            enrichment={'decision':'NOT_EVALUATED','decision_reason':f'{type(exc).__name__}: {exc}',
                        'semantic_mlp_accepted':False,'semantic_trepan_available':False,'test_used':False}
    oracle=mlp; teacher=None
    teacher_report={'teacher':'mlp_original','reason':'semantic_enrichment_not_requested'}
    if semantic_enrichment is not None:
        choice=decide_semantic_teacher(enrichment,teacher_min_evidence if use_semantic_teacher else None)
        teacher_report={'teacher':'mlp_original','reason':choice.reason,**choice.details,
                        'min_evidence_required':teacher_min_evidence if use_semantic_teacher else None}
        if choice.use and enrichment_result is not None:
            avail=teacher_inputs_available(enrichment_result.processor,model_names_original)
            if avail.use:
                try:
                    teacher=build_semantic_teacher(enrichment_result,Ztr,ytr,model_names_original,
                                                   config=semantic_enrichment,seed=seed)
                    oracle=teacher
                    teacher_report={'teacher':'mlp_semantic','reason':choice.reason,**choice.details,
                                    **teacher.audit_}
                except Exception as exc:  # falha ao construir: mantém o professor original, sem esconder
                    teacher_report={'teacher':'mlp_original','reason':f'teacher_build_failed:{type(exc).__name__}: {exc}'}
            else:
                teacher_report={'teacher':'mlp_original','reason':avail.reason,**avail.details}

    # Contrato científico: o oráculo é CONGELADO aqui e é exatamente este objeto que o tuning e todas as árvores
    # (Original, Reloaded e variantes) consultam; o ``oracle_id`` fica no relatório como prova.
    oracle=freeze_oracle(oracle,Ztr,builder='semantic_teacher' if teacher is not None else oracle_builder)
    tuning_report = None
    if scientific_tuning:
        try:
            with oracle.scope('tuning'):
                tuning_report = tune_scientific_trepan(
                    Ztr, ytr, oracle=oracle, feature_names=model_names, base_config=cfg,
                    semantic_feature_weights=semantic_weights,
                    semantic_feature_groups=semantic_groups,
                    semantic_relatedness_matrix=relation_matrix,
                    search=trepan_search or ScientificTrepanSearchConfig(),
                    ontology_graph=graph,
                    semantic_feature_entities=semantic_entities,
                )
            cfg = ControlledTrepanConfig(**tuning_report['selected_config'])
        except Exception as exc:  # tuning falhou: configuração base canónica, com a falha registada (não escondida)
            logging.getLogger(__name__).warning("Tuning científico do TREPAN falhou (%s: %s); configuração canónica.",
                                                type(exc).__name__, exc)
            tuning_report = {"selection_scope": "training_cv_only", "test_used_for_selection": False,
                             "failed": f"{type(exc).__name__}: {exc}", "fallback": "canonical_defaults",
                             "common_capacity": dataclasses.asdict(cfg), "selected_config": dataclasses.asdict(cfg)}

    # Espaço do Reloaded: com professor semântico, o braço Reloaded divide também sobre as
    # features onto_* selecionadas (espaço aumentado). O oráculo continua a ver só as colunas
    # originais (OriginalOracleProjection) e cada consulta sintética é recomposta para ficar
    # coerente. O braço Original fica no espaço original: a única variável é a extensão semântica.
    augmented=bool(teacher is not None and augment_reloaded_space and teacher.selected_features)
    # Gate de atribuição (opt-in, só treino): o Reloaded só usa a semântica se bater sistematicamente
    # controlos com a mesma maquinaria e sem significado (semântica baralhada / features aleatórias).
    # Caso contrário degrada para o Original e não reivindica ganho nenhum.
    attribution=None; reloaded_mode='augmented' if augmented else ('original_space' if graph is not None else 'none')
    attribution_variants={}
    if semantic_attribution is not None and graph is not None:
        attribution,reloaded_mode,attribution_variants=_select_reloaded_mode(
            semantic_attribution,augmented,teacher,oracle,Ztr,ytr,cfg,
            ontology,quality,pre,Xtr,original_feature_names,
            dict(graph=graph,names=model_names,weights=semantic_weights,groups=semantic_groups,
                 matrix=relation_matrix,entities=semantic_entities),semantic_control_seed)
        augmented=(reloaded_mode=='augmented')
    pair_kwargs={}
    pair_cfg=cfg
    Zte_rel=Zte
    reloaded_space={'space':'original','n_features':len(model_names)}
    if augmented:
        names_aug,w_aug,g_aug,e_aug,r_aug,derived=extend_semantic_inputs(
            model_names,semantic_weights,semantic_groups,semantic_entities,relation_matrix,
            teacher.processor,teacher.selected_features)
        Ztr_rel=teacher.augment(Ztr); Zte_rel=teacher.augment(Zte)
        pair_kwargs=dict(
            reloaded_X_train=Ztr_rel,reloaded_feature_names=names_aug,
            original_feature_indices=tuple(range(len(model_names))),
            query_projector=make_consistency_projector(teacher,len(model_names)))
        semantic_weights_pair,semantic_groups_pair,relation_pair,entities_pair=w_aug,g_aug,r_aug,e_aug
        # O espelho refaria um TREPAN sobre as colunas aumentadas e chamar-lhe "Original" seria falso.
        pair_cfg=dataclasses.replace(cfg,mirror_when_no_semantic_effect=False)
        reloaded_space={'space':'augmented','n_features':len(names_aug),'semantic_features':list(teacher.selected_features),
                        'derived':derived,'mirror_disabled_reason':'reloaded_arm_uses_augmented_space'}
    elif reloaded_mode=='neutral':
        # Semântica não atribuível à ontologia: o Reloaded fica sem entradas semânticas e é
        # idêntico ao Original (mesmo professor, semente e orçamento). Não há ganho a reivindicar.
        semantic_weights_pair=semantic_groups_pair=relation_pair=entities_pair=None
        reloaded_space['reason']='semantics_not_attributable_to_ontology'
    else:
        semantic_weights_pair,semantic_groups_pair,relation_pair,entities_pair=(
            semantic_weights,semantic_groups,relation_matrix,semantic_entities)
        if teacher is not None and not augment_reloaded_space:
            reloaded_space['reason']='augment_reloaded_space_disabled'
    with oracle.scope('trepan_pair'):
      pair=fit_controlled_trepan_pair(
          Ztr,ytr,oracle=oracle,feature_names=model_names,config=pair_cfg,
          semantic_feature_weights=semantic_weights_pair,
          semantic_feature_groups=semantic_groups_pair,
          semantic_relatedness_matrix=relation_pair,
          run_id=f"production_v9_2_seed_{seed}",
          ontology_graph=None if reloaded_mode=='neutral' else graph,
          semantic_feature_entities=entities_pair,
          **pair_kwargs,
      )
    oracle_ids={'trepan_original':oracle_id_of(pair.base_oracle),
                'trepan_reloaded':oracle_id_of(pair.reloaded_oracle_for_audit)}
    if len(set(oracle_ids.values()))!=1 or None in oracle_ids.values() or oracle_ids['trepan_original']!=oracle.oracle_id:
        raise OracleContractViolation(f"Original e Reloaded não consultaram o mesmo oráculo congelado: {oracle_ids}")
    evaluation=evaluate_controlled_trepan_pair(pair,Zte,yte,reloaded_X_test=Zte_rel if augmented else None)
    mlp_pred=mlp.predict(Zte); c45_pred=c45.predict(Xte.to_numpy(dtype=object))
    teacher_pred=oracle.predict(Zte)  # referência de fidelidade = professor realmente usado
    evaluation['models']['mlp_original']=classification_metrics(yte,mlp_pred)
    if teacher is not None:
        evaluation['models']['mlp_semantic']=classification_metrics(yte,teacher_pred)
    evaluation['models']['c45_native']=classification_metrics(yte,c45_pred)
    original_pred=pair.original.predict(Zte); reloaded_pred=pair.reloaded.predict(Zte_rel)
    evaluation['scientific_validation']=build_scientific_validation_report(
        yte, teacher_pred, {
            'trepan_reloaded':reloaded_pred,
            'trepan_original':original_pred,
            'c45_j48':c45_pred,
        }, random_state=seed, n_bootstrap=1000,
    )
    evaluation['oracle_health']=health
    evaluation['oracle_contract']={**oracle.report(),'tree_oracle_ids':oracle_ids,'single_oracle_for_all_trees':True}
    evaluation['trepan_scientific_tuning']=tuning_report
    evaluation['ontology_quality']=quality
    evaluation['semantic_graph']=graph.summary() if graph else None
    contribution=SemanticContributionGate().evaluate(
        ontology_quality=quality,
        semantic_audit=evaluation.get('semantic_audit'),comparison=evaluation.get('comparison'),
        original_metrics=evaluation['models']['original'],reloaded_metrics=evaluation['models']['reloaded'],
    ) if quality else {"accepted":False,"status":"NO_ONTOLOGY","reasons":["ontology_not_provided"]}
    evaluation['semantic_contribution_gate']=contribution
    evaluation['semantic_enrichment']=enrichment
    evaluation['semantic_teacher']=teacher_report
    reloaded_space['mode']=reloaded_mode
    evaluation['reloaded_feature_space']=reloaded_space
    if attribution is not None:
        attribution['selected_mode']=reloaded_mode
        if semantic_attribution.report_test_controls and reloaded_mode in attribution_variants:
            # Ganho atribuível medido UMA vez no teste final: só relatado, nunca decide nada.
            ctrl_fit=attribution_variants[reloaded_mode]
            controls=[]
            for k in range(semantic_attribution.controls):
                try:
                    controls.append(ctrl_fit(semantic_attribution.random_state*7+k)(Ztr,ytr))
                except Exception:  # um controlo que não treina é ignorado (não favorece o real)
                    continue
            real_model=AugmentedPredictor(pair.reloaded,teacher) if reloaded_mode=='augmented' else pair.reloaded
            attribution['test_attribution']=test_attribution_report(real_model,controls,Zte,yte,oracle)
        evaluation['semantic_attribution']=attribution
    status=build_ontology_stage_status(quality,None,enrichment) if quality else None
    if status is not None:
        status['semantic_teacher_used']=bool(teacher is not None)
        status['semantic_teacher_reason']=teacher_report.get('reason')
    evaluation['ontology_stage_status']=status

    bundle={
        'format_version':'biuri-v9.2-production-5-semantic-real-gain', 'contract':contract, 'preprocessor':pre,
        'reloaded_feature_space':reloaded_space['space'],
        'mlp_original':mlp,'semantic_teacher':teacher,'c45_native':c45,'trepan_original':pair.original,'trepan_reloaded':pair.reloaded,
        'classes':list(map(str,getattr(mlp,'classes_',[]))),
    }
    joblib.dump(bundle,out/'production_bundle.joblib')
    manifest={
        'format_version':'biuri-v9.2-production-5-semantic-real-gain','seed':seed,'target':contract.target,
        'dataset_sha256':_hash_dataframe(clean),'owl_sha256':_sha256_file(owl_path),
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'train_rows':len(tr),'test_rows':len(te),'test_used_for_selection':False,
        'tree_runtime_policy':'historical_trepan_only_no_cart',
        'reloaded_feature_space':reloaded_space['space'],'reloaded_mode':reloaded_mode,
        'teacher_id':teacher_report['teacher'],'teacher_reason':teacher_report['reason'],
        'ontology_used':bool(quality and quality.get('accepted')),
        'semantic_contribution_status':contribution.get('status'),
        'claim_guard':evaluation['scientific_validation'].get('claim_guard'),
        'convergence':extract_mlp_convergence(mlp),
        'scientific_tuning': bool(scientific_tuning),
        'trepan_selected_config': asdict(cfg) if hasattr(cfg, '__dataclass_fields__') else str(cfg),
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    report={'manifest':manifest,'contract':contract.to_dict(),'split':{'seed':seed,'train_indices':tr.tolist(),'test_indices':te.tolist()},'evaluation':evaluation,'semantic_graph':graph.to_dict() if graph else None}
    (out/'production_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    if return_artifacts:
        # Objetos EXATOS avaliados cientificamente (nunca reconstruídos/retreinados): árvores, oráculo congelado e espaços.
        report['artifacts']={
            'trepan_original':pair.original,'trepan_reloaded':pair.reloaded,'oracle':oracle,'mlp_original':mlp,
            'feature_names_original':list(model_names),'feature_names_reloaded':list(names_aug) if augmented else list(model_names),
            'class_names':[str(c) for c in getattr(mlp,'classes_',[])],'selected_config':cfg,
            'oracle_id':oracle.oracle_id,'preprocessor':pre,
        }
    return report

__all__=['train_production_dataframe']
