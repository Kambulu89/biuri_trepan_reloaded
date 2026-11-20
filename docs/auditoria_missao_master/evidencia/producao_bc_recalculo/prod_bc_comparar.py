"""Compara, por semente, a execução de produção ORIGINAL, o CONTROLO (mesmo código antigo, ambiente atual) e o RECÁLCULO (código corrigido).

Uso: python prod_bc_comparar.py PRED_DIR SAIDA.csv SAIDA.json   (PRED_DIR contém {original,controlo,corrigido}_seed{S}.json de prod_bc_predicoes.py)

Tudo é alinhado por RÓTULO ('M'/'B') e pelo índice da instância de teste, nunca pela posição da coluna ou do código.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRED, OUT_CSV, OUT_JSON = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
MODELS = ["mlp", "c45", "trepan_original", "trepan_reloaded"]


def load(v, s):
    p = PRED / f"{v}_seed{s}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def by_index(r):
    idx = r["test_indices"]
    pred = {m: dict(zip(idx, r["pred"][m])) for m in MODELS}
    proba = {lab: dict(zip(idx, r["proba_mlp"][lab])) for lab in ("M", "B")}
    return idx, pred, proba, dict(zip(idx, r["y_true"]))


rows, detail = [], {}
for s in (7, 42, 123):
    o = load("original", s)
    if o is None:
        continue
    io, po, qo, yo = by_index(o)
    for v in ("controlo", "corrigido"):
        r = load(v, s)
        if r is None:
            continue
        ir, pr, qr, yr = by_index(r)
        same_idx = sorted(io) == sorted(ir) and io == ir
        common = [i for i in io if i in set(ir)]
        changed = {m: int(sum(po[m][i] != pr[m][i] for i in common)) for m in MODELS}
        dproba = max(abs(qo[lab][i] - qr[lab][i]) for lab in ("M", "B") for i in common) if common else None
        d = dict(semente=s, versao=v, mesmas_instancias_teste=same_idx, n_teste=len(ir), instancias_comuns=len(common),
                 ordem_classes_bundle=",".join(r["classes_bundle"]), codigos_na_ordem_declarada=r["codes_follow_declared_order"],
                 colunas_proba=",".join(r["proba_column_labels"]),
                 oracle_id_antes=o["oracle"]["oracle_id"], oracle_id_depois=r["oracle"]["oracle_id"],
                 mlp_pesos_iguais=o["oracle"]["weights_hash"] == r["oracle"]["weights_hash"],
                 prev_mudadas_mlp=changed["mlp"], prev_mudadas_c45=changed["c45"],
                 prev_mudadas_original=changed["trepan_original"], prev_mudadas_reloaded=changed["trepan_reloaded"],
                 max_dif_proba_mlp=dproba,
                 acc_mlp_antes=o["metrics"]["mlp_accuracy"], acc_mlp_depois=r["metrics"]["mlp_accuracy"],
                 acc_original_antes=o["metrics"]["original_accuracy"], acc_original_depois=r["metrics"]["original_accuracy"],
                 acc_reloaded_antes=o["metrics"]["reloaded_accuracy"], acc_reloaded_depois=r["metrics"]["reloaded_accuracy"],
                 fid_original_antes=o["metrics"]["original_fidelity"], fid_original_depois=r["metrics"]["original_fidelity"],
                 fid_reloaded_antes=o["metrics"]["reloaded_fidelity"], fid_reloaded_depois=r["metrics"]["reloaded_fidelity"],
                 nos_original_antes=o["nodes"]["original"], nos_original_depois=r["nodes"]["original"],
                 nos_reloaded_antes=o["nodes"]["reloaded"], nos_reloaded_depois=r["nodes"]["reloaded"],
                 delta_fid_reloaded_menos_original_antes=o["metrics"]["reloaded_fidelity"] - o["metrics"]["original_fidelity"],
                 delta_fid_reloaded_menos_original_depois=r["metrics"]["reloaded_fidelity"] - r["metrics"]["original_fidelity"],
                 estado_gate_antes=o["semantic_contribution_status"], estado_gate_depois=r["semantic_contribution_status"],
                 recalculo_coincide_com_relatorio=r["recomputed_matches_report"] and o["recomputed_matches_report"])
        rows.append(d)
        detail[f"{v}_seed{s}"] = {m: [i for i in common if po[m][i] != pr[m][i]] for m in MODELS}
T = pd.DataFrame(rows)
T.to_csv(OUT_CSV, index=False)
Path(OUT_JSON).write_text(json.dumps({"instancias_com_previsao_mudada": detail}, indent=1), encoding="utf-8")
pd.set_option("display.width", 250)
print(T.T.to_string())
