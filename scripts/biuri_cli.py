#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import pandas as pd
from core.data_loading import load_tabular
from core.data_contract import build_data_contract
from core.pipeline_v92 import train_dataset
from core.inference import load_artifact
from core.logging_config import configure_logging
from core.production_training import train_production_dataframe
from core.production_inference import load_production_bundle


def _read_input(path):
    d,_=load_tabular(path); return d

def cmd_train(a):
    df,meta=load_tabular(a.data,encoding=a.encoding,separator=a.separator)
    r=train_dataset(df,target=a.target,seed=a.seed,out=a.out,max_trepan_nodes=a.max_nodes)
    report={'data':meta,'contract':r.contract.to_dict(),'metrics':r.metrics,'split':r.split,'artifact':r.artifact_path,'owl':a.owl or None}
    out=Path(a.out); (out/'training_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    (out/'training_report.md').write_text('# Relatório de treino BIURI V9.2\n\n```json\n'+json.dumps(report,indent=2,ensure_ascii=False,default=str)+'\n```\n',encoding='utf-8')
    print(json.dumps({'status':'ok','artifact':str(out)},ensure_ascii=False))

def cmd_train_production(a):
    df,meta=load_tabular(a.data,encoding=a.encoding,separator=a.separator)
    report=train_production_dataframe(
        df,target=a.target,out_dir=a.out,seed=a.seed,owl_path=a.owl,
        require_reasoner=not a.allow_unreasoned_owl,
    )
    print(json.dumps({
        'status':'ok',
        'artifact':str(Path(a.out)/'production_bundle.joblib'),
        'semantic_contribution_status':report['evaluation']['semantic_contribution_gate']['status'],
        'ontology_quality':(report['evaluation'].get('ontology_quality') or {}).get('status'),
    },ensure_ascii=False))


def cmd_predict_production(a):
    predictor=load_production_bundle(a.artifact)
    df=_read_input(a.data)
    pred=predictor.predict(df,model=a.model)
    print(json.dumps({'predictions':[str(x) for x in pred],'model':a.model},ensure_ascii=False))

def cmd_explain_production(a):
    predictor=load_production_bundle(a.artifact)
    df=_read_input(a.data)
    print(json.dumps(predictor.explain(df.iloc[[a.row]],model=a.model),ensure_ascii=False,default=str))

def cmd_predict(a):
    p=load_artifact(a.artifact); df=_read_input(a.data); pred=p.predict(df); print(json.dumps({'predictions':[str(x) for x in pred]},ensure_ascii=False))
def cmd_evaluate(a):
    p=load_artifact(a.artifact); df=_read_input(a.data); y=df.pop(a.target).to_numpy(); from core.evaluation_protocol import classification_metrics; print(json.dumps(classification_metrics(y,p.predict(df)),ensure_ascii=False))
def cmd_explain(a):
    p=load_artifact(a.artifact); df=_read_input(a.data); print(json.dumps(p.explain(df.iloc[[a.row]]),ensure_ascii=False,default=str))

def main(argv=None):
    ap=argparse.ArgumentParser(prog='biuri',description='BIURI / TREPAN Reloaded V9.2')
    ap.add_argument('--verbose',action='store_true'); ap.add_argument('--quiet',action='store_true')
    sub=ap.add_subparsers(dest='cmd',required=True)
    t=sub.add_parser('train'); t.add_argument('--data',required=True); t.add_argument('--target',required=True); t.add_argument('--out',required=True); t.add_argument('--seed',type=int,default=42); t.add_argument('--owl'); t.add_argument('--encoding'); t.add_argument('--separator'); t.add_argument('--preset',choices=['scientific'],default='scientific'); t.add_argument('--max-nodes',type=int,default=31); t.set_defaults(func=cmd_train)
    tp=sub.add_parser('train-production'); tp.add_argument('--data',required=True); tp.add_argument('--target',required=True); tp.add_argument('--out',required=True); tp.add_argument('--seed',type=int,default=42); tp.add_argument('--owl'); tp.add_argument('--encoding'); tp.add_argument('--separator'); tp.add_argument('--allow-unreasoned-owl',action='store_true'); tp.set_defaults(func=cmd_train_production)
    e=sub.add_parser('evaluate'); e.add_argument('--artifact',required=True); e.add_argument('--data',required=True); e.add_argument('--target',required=True); e.set_defaults(func=cmd_evaluate)
    p=sub.add_parser('predict'); p.add_argument('--artifact',required=True); p.add_argument('--data',required=True); p.set_defaults(func=cmd_predict)
    pp=sub.add_parser('predict-production'); pp.add_argument('--artifact',required=True); pp.add_argument('--data',required=True); pp.add_argument('--model',choices=['mlp','trepan_original','trepan_reloaded','c45_native'],default='mlp'); pp.set_defaults(func=cmd_predict_production)
    x=sub.add_parser('explain'); x.add_argument('--artifact',required=True); x.add_argument('--data',required=True); x.add_argument('--row',type=int,default=0); x.set_defaults(func=cmd_explain)
    xp=sub.add_parser('explain-production'); xp.add_argument('--artifact',required=True); xp.add_argument('--data',required=True); xp.add_argument('--row',type=int,default=0); xp.add_argument('--model',choices=['trepan_original','trepan_reloaded'],default='trepan_reloaded'); xp.set_defaults(func=cmd_explain_production)
    a=ap.parse_args(argv); configure_logging(verbose=a.verbose,quiet=a.quiet); a.func(a)
if __name__=='__main__': main()
