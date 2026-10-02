#!/usr/bin/env python3
"""Audita pickles legados de contrafactuais e regista incompatibilidades.

Não fabrica resultados. Se um pickle falhar, o relatório indica que deve ser
regenerado a partir do dataset/pipeline correspondente.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib


def audit(root: Path):
    rows=[]
    for p in root.rglob('*.pkl'):
        try:
            obj=joblib.load(p)
            rows.append({'path':str(p),'status':'loadable','type':f'{type(obj).__module__}.{type(obj).__name__}'})
        except Exception as exc:
            rows.append({'path':str(p),'status':'incompatible','error':f'{type(exc).__name__}: {exc}','action':'regenerar a partir do dataset com V9.2'})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='counterfactuals/experimentos'); ap.add_argument('--out',default='results/legacy_pickle_audit_v9_2.json'); a=ap.parse_args()
    rows=audit(Path(a.root)); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'total':len(rows),'loadable':sum(r['status']=='loadable' for r in rows),'incompatible':sum(r['status']=='incompatible' for r in rows),'report':str(out)},ensure_ascii=False))
if __name__=='__main__': main()
