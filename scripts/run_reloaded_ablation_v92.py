#!/usr/bin/env python3
"""Ablação pareada dataset-agnostic para TREPAN Original vs Reloaded V9.2."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.data_loading import load_tabular
from core.production_training import train_production_dataframe


def main(argv=None):
    ap=argparse.ArgumentParser(description='Ablação controlada TREPAN V9.2')
    ap.add_argument('--data',required=True); ap.add_argument('--target',required=True)
    ap.add_argument('--owl'); ap.add_argument('--out',required=True)
    ap.add_argument('--seeds',default='11,23,42,67,101'); ap.add_argument('--allow-unreasoned-owl',action='store_true')
    a=ap.parse_args(argv); df,_=load_tabular(a.data); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in [int(x.strip()) for x in a.seeds.split(',') if x.strip()]:
        run=out/f'seed_{seed}'
        try:
            report=train_production_dataframe(df,target=a.target,out_dir=run,seed=seed,owl_path=a.owl,require_reasoner=not a.allow_unreasoned_owl)
            ev=report['evaluation']; rows.append({'seed':seed,'status':'ok','comparison':ev['comparison'],'semantic_audit':ev.get('semantic_audit'),'semantic_gate':ev.get('semantic_contribution_gate'),'models':ev['models']})
        except Exception as exc:
            rows.append({'seed':seed,'status':'failed','error':f'{type(exc).__name__}: {exc}'})
    ok=[r for r in rows if r['status']=='ok']
    metrics=['delta_accuracy','delta_balanced_accuracy','delta_macro_f1','delta_oracle_fidelity','delta_nodes','delta_depth']
    summary={'runs':len(rows),'ok':len(ok),'failed':len(rows)-len(ok),'unit':'seed_within_one_dataset','means':{m:float(np.mean([r['comparison'][m] for r in ok])) if ok else None for m in metrics},'semantic_gate_statuses':{}}
    for r in ok:
        st=r['semantic_gate'].get('status'); summary['semantic_gate_statuses'][st]=summary['semantic_gate_statuses'].get(st,0)+1
    (out/'ablation_rows.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
