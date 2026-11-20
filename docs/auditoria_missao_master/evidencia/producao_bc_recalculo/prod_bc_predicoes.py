"""Previsões por instância de uma execução de produção (``production_bundle.joblib`` + ``production_report.json``).

Uso (correr com a versão do código que produziu o bundle, a partir da raiz desse checkout):
    python prod_bc_predicoes.py DIR_DA_EXECUCAO DADOS.arff ALVO SAIDA.json

Escreve, para as instâncias de teste da execução (índices de ``split.test_indices``, linhas do ficheiro depois de retirar alvos ausentes):
rótulo real; previsões do MLP (oráculo), do C4.5, do TREPAN Original e do TREPAN Reloaded, sempre como RÓTULOS ('M'/'B');
probabilidades do MLP alinhadas pelo rótulo (não pela posição da coluna); número de nós das árvores; identificação do oráculo.
Verifica que accuracy e fidelidade recalculadas a partir destas previsões coincidem com as do relatório.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from core.data_loading import load_tabular  # noqa: E402

run, data, target, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
report = json.loads((run / "production_report.json").read_text(encoding="utf-8"))
bundle = joblib.load(run / "production_bundle.joblib")
df, _ = load_tabular(data)
tokens = {str(t).strip().lower() for t in report["contract"]["missing_tokens"]}
clean = df.loc[~df[target].map(lambda v: v is None or (isinstance(v, str) and v.strip().lower() in tokens))].reset_index(drop=True)
te = np.asarray(report["split"]["test_indices"], dtype=int)
Xte = clean.drop(columns=[target]).iloc[te]
y = clean[target].iloc[te].astype(str).to_numpy()

classes = [str(c) for c in bundle["classes"]]
coded = bool(bundle.get("class_codes_follow_declared_order", False))   # código corrigido: 0..k-1 na ordem declarada
label = (lambda p: np.asarray([classes[int(v)] for v in p], dtype=object)) if coded else (lambda p: np.asarray([str(v) for v in p], dtype=object))

space = report["manifest"]["reloaded_feature_space"]
if space != "original":
    raise SystemExit(f"espaço do Reloaded '{space}' não suportado por este script (só 'original').")
pre = bundle["preprocessor"]
Z = pre.transform(Xte)
mlp = bundle["mlp_original"]
pred = {"mlp": label(mlp.predict(Z)),
        "c45": label(bundle["c45_native"].predict(Xte.to_numpy(dtype=object))),
        "trepan_original": label(bundle["trepan_original"].predict(Z)),
        "trepan_reloaded": label(bundle["trepan_reloaded"].predict(Z))}
proba_raw = mlp.predict_proba(Z)
col_labels = [classes[int(c)] for c in mlp.classes_] if coded else [str(c) for c in mlp.classes_]
proba = {lab: proba_raw[:, j].tolist() for j, lab in enumerate(col_labels)}

ev = report["evaluation"]["models"]
teacher = report["manifest"]["teacher_id"]
oracle_pred = pred["mlp"] if teacher == "mlp_original" else None
if oracle_pred is None:
    raise SystemExit(f"oráculo '{teacher}' não suportado por este script.")
check = {
    "mlp_accuracy": [float(np.mean(pred["mlp"] == y)), ev["mlp_original"]["accuracy"]],
    "original_accuracy": [float(np.mean(pred["trepan_original"] == y)), ev["original"]["accuracy"]],
    "reloaded_accuracy": [float(np.mean(pred["trepan_reloaded"] == y)), ev["reloaded"]["accuracy"]],
    "original_fidelity": [float(np.mean(pred["trepan_original"] == oracle_pred)), ev["original"]["oracle_fidelity"]],
    "reloaded_fidelity": [float(np.mean(pred["trepan_reloaded"] == oracle_pred)), ev["reloaded"]["oracle_fidelity"]],
    "c45_accuracy": [float(np.mean(pred["c45"] == y)), ev["c45_native"]["accuracy"]],
}
mismatch = {k: v for k, v in check.items() if abs(v[0] - v[1]) > 1e-12}
oc = report["evaluation"].get("oracle_contract", {}).get("oracle", {})
res = {
    "run": str(run), "test_indices": te.tolist(), "y_true": y.tolist(),
    "classes_bundle": classes, "codes_follow_declared_order": coded, "mlp_classes_": [str(c) for c in mlp.classes_],
    "proba_column_labels": col_labels, "proba_mlp": proba, "pred": {k: v.tolist() for k, v in pred.items()},
    "nodes": {"original": ev["original"]["nodes"], "reloaded": ev["reloaded"]["nodes"]},
    "metrics": {k: v[1] for k, v in check.items()},
    "oracle": {"teacher_id": teacher, "oracle_id": oc.get("oracle_id"), "weights_hash": oc.get("weights_hash"),
               "decisions_hash": oc.get("decisions_hash"), "classes": oc.get("classes")},
    "semantic_contribution_status": report["manifest"].get("semantic_contribution_status"),
    "comparison": report["evaluation"].get("comparison"),
    "recomputed_matches_report": not mismatch, "mismatch": mismatch,
}
Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"recomputed_matches_report": not mismatch, "mismatch": mismatch, "nodes": res["nodes"], "oracle": res["oracle"]["oracle_id"]}, ensure_ascii=False))
