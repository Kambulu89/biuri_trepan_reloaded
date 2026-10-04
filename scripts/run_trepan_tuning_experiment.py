"""Experiência do tuning científico da estrutura do TREPAN, para qualquer dataset ARFF (sem lógica por dataset).

Usa o pipeline de produção: split estratificado -> MLP no treino -> tuning (Repeated Stratified K-Fold, SÓ treino) ->
ajuste final -> avaliação no teste UMA vez, depois de escolhida a configuração. Guarda o relatório completo em JSON.

Uso: python scripts/run_trepan_tuning_experiment.py DATASET.arff SEED SAIDA.json [--target COLUNA] [--oracle factory|robust] [--repeats N]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")


def main() -> None:
    import pandas as pd
    from scipy.io import arff

    from core.production_training import train_production_dataframe
    from core.trepan_scientific_tuning import ScientificTrepanSearchConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("arff"); ap.add_argument("seed", type=int); ap.add_argument("out")
    ap.add_argument("--target", default=None)
    ap.add_argument("--oracle", default="factory", help="construtor do oráculo registado no contrato científico")
    ap.add_argument("--repeats", type=int, default=3, help="repetições da Repeated Stratified K-Fold (3 dobras cada)")
    args = ap.parse_args()
    data, _meta = arff.loadarff(args.arff)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8")
    target = args.target or df.columns[-1]
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        report = train_production_dataframe(df, target=target, out_dir=tmp, seed=args.seed, scientific_tuning=True,
                                            trepan_search=ScientificTrepanSearchConfig(cv_repeats=args.repeats), oracle_builder=args.oracle)
    ev = report["evaluation"]
    out = {"dataset": Path(args.arff).name, "seed": args.seed, "rows": int(len(df)), "target": target,
           "wall_seconds": time.perf_counter() - t0, "oracle_builder": args.oracle, "oracle_contract": ev.get("oracle_contract"),
           "tuning": ev.get("trepan_scientific_tuning"),
           "test_models": ev.get("models"), "manifest": {k: report["manifest"].get(k) for k in ("train_rows", "test_rows", "seed")}}
    Path(args.out).write_text(json.dumps(out, default=str, indent=1))
    print("DONE", args.seed, f"{out['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
