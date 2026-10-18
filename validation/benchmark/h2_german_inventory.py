"""H2 (Doc. 58): inventário FACTUAL do ficheiro local ``german.data`` e da codificação de 61 features — só o que é verificável sem documentação externa: hash do ficheiro, 20 colunas de atributos
(+ coluna de classe, NÃO usada), tipo no carregador do projeto, códigos observados e contagens, mapeamento atributo → colunas one-hot, e a prioridade por variância das colunas derivadas (O2).
NÃO afirma o significado de nenhum código (isso é documentação): ``documented_definition`` fica ``PENDING_SOURCE``. A classe (coluna 21) nunca é lida para estatísticas."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main(out: str):
    from validation.benchmark import datasets as D, datasets_categorical as DC
    from validation.benchmark.h2_axiom_mechanisms import defined_union_concepts, derived_union_literals
    from validation.benchmark.b31_facts import compile_registry
    DC.register()
    f = DC.GERMAN_FILE
    raw = pd.read_csv(f, sep=" ", header=None)
    X = raw.iloc[:, :20].copy()
    X.columns = DC.GERMAN_COLUMNS
    spec = D.REGISTRY["german_credit"]
    ds, _, _ = spec.load()
    enc_names = [str(n) for n in ds.feature_names]
    attrs = []
    for i, c in enumerate(DC.GERMAN_COLUMNS):
        col = X[c]
        nominal = col.dtype == object
        vc = col.value_counts().sort_index() if not nominal or True else None
        cols = [n for n in enc_names if n == c or n.startswith(c + "__")]
        attrs.append({"file_column": i + 1, "attribute_name_in_project": c, "assumed_attribute_number": i + 1, "number_assumption": "ordem das colunas do ficheiro = numeração do dicionário (A CONFIRMAR em german.doc)",
                      "kind_in_project_loader": "nominal (one-hot)" if nominal else "numérico (inalterado)", "project_marks_ordinal_coded": c in DC.GERMAN_ORDINAL,
                      "observed_codes_counts_all_1000_rows": {str(k): int(v) for k, v in vc.items()} if nominal or col.nunique() <= 12 else {"min": float(col.min()), "max": float(col.max()), "n_distinct": int(col.nunique())},
                      "n_encoded_columns": len(cols), "encoded_columns": cols, "documented_definition": "PENDING_SOURCE", "documented_ordinal_status": "PENDING_SOURCE", "south_german_correction": "PENDING_SOURCE"})
    # prioridade por variância das colunas derivadas (O2), no treino do holdout seed 42 (índices apenas; sem rótulos)
    from core.benchmark.splits import make_splits
    y = np.asarray(ds.y)
    sp = make_splits(y, scheme="holdout", seeds=[42])[0]
    Xtr = np.asarray(ds.X, dtype=float)[sp.train_idx]
    owl = ROOT / "validation/benchmark/ontologies/proposals/german_credit/v0_draft/german_credit_O2.owl"
    reg = compile_registry(str(owl), enc_names, run_reasoner=True)
    lits = derived_union_literals(reg, Xtr, enc_names, only_concepts=defined_union_concepts(owl))
    base_var = np.nanvar(Xtr, axis=0)
    der_var = [float(np.nanvar(np.max(Xtr[:, [enc_names.index(a) for a in l.attributes]], axis=1))) for l in lits]
    allv = np.concatenate([base_var, np.array(der_var)])
    order = np.argsort(-allv, kind="stable")
    rank = {int(j): r + 1 for r, j in enumerate(order)}
    pri = []
    for k, l in enumerate(lits):
        r = rank[len(enc_names) + k]
        pri.append({"literal": l.name, "variance": der_var[k], "rank_by_variance_among_%d_columns" % len(allv): r, "in_pool_mf12": r <= 12, "in_pool_mf20": r <= 20, "in_pool_mf30": r <= 30})
    doc = {"file": str(f.relative_to(ROOT)), "file_sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest(), "file_sha256_lf_normalized": hashlib.sha256(Path(f).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), "sha256_raw": hashlib.sha256(Path(f).read_bytes()).hexdigest(), "sha256_canonical": hashlib.sha256(Path(f).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), "canonicalization": "CRLF->LF (apenas); raw = bytes desta cópia de trabalho (proveniência)", "n_rows": int(len(raw)), "n_attribute_columns": 20, "class_column": "coluna 21 (não usada neste inventário)",
           "n_encoded_features": len(enc_names), "encoding": "numéricas inalteradas (7); nominais one-hot com os níveis OBSERVADOS por ordem lexicográfica (13 → 54 colunas)", "attributes": attrs,
           "encoded_columns_total": sum(a["n_encoded_columns"] for a in attrs), "priority_of_derived_literals": {"basis": "variância no treino (holdout seed 42) — critério de prioridade do Original para o pool de literais (`_feature_priority_scores`)", "mf12_is_Original_default": True, "rows": pri},
           "note": "a prioridade por variância depende da escala bruta das colunas numéricas (credit_amount, duration, age dominam); não assume a configuração de `max_features_per_node` do benchmark real"}
    Path(out).write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(doc["file_sha256"][:16], len(attrs), doc["encoded_columns_total"])
    for r in pri:
        print(r["literal"], r["variance"].__round__(3), r["rank_by_variance_among_%d_columns" % len(allv)], r["in_pool_mf12"], r["in_pool_mf20"])


if __name__ == "__main__":
    main(sys.argv[1])
