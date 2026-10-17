"""H2 (Doc. 60): VERIFICAÇÃO DE FUNCIONAMENTO (não é experiência): corre o adaptador T1/T2 sobre as colunas reais do German Credit (treino do holdout seed 42) com o rascunho de IA O2, só para observar o
rastreio  Axioma → Inferência → Literal → Candidato → Decisão  (candidatos avaliados por literal, nós escolhidos, podas). NÃO calcula nem regista accuracy, fidelidade ou qualquer métrica de desempenho; o teste não
é tocado; as conclusões possíveis são apenas de funcionamento (os axiomas são rascunhos de IA não validados). Um único MLP do projeto (mesmo para todos os braços)."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main(out: str):
    warnings.filterwarnings("ignore")
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from core.benchmark.splits import make_splits
    from validation.benchmark import datasets as D, datasets_categorical as DC
    from validation.benchmark.b31_facts import compile_registry
    from validation.benchmark.h2_axiom_mechanisms import defined_union_concepts, derived_union_literals
    from validation.benchmark.h2_harness import H2Steered, instance_hash
    from validation.benchmark.h2_trace import augment
    DC.register()
    ds, _, _ = D.REGISTRY["german_credit"].load()
    names = [str(n) for n in ds.feature_names]
    X, y = np.asarray(ds.X, dtype=float), np.asarray(ds.y)
    sp = make_splits(y, scheme="holdout", seeds=[42])[0]
    Xtr, ytr = X[sp.train_idx], y[sp.train_idx]                                    # só treino; o teste nunca é lido
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)).fit(Xtr, ytr)
    calls = {"n": 0, "rows": 0}

    class Orc:
        def predict(self, Z):
            calls["n"] += 1
            calls["rows"] += len(Z)
            return mlp.predict(np.asarray(Z)[:, : len(names)])
    owl = ROOT / "validation/benchmark/ontologies/proposals/german_credit/v0_draft/german_credit_O2.owl"
    reg = compile_registry(str(owl), names, run_reasoner=True)
    lits = derived_union_literals(reg, Xtr, names, only_concepts=defined_union_concepts(owl))
    params = dict(min_sample=300, max_queries=3000, max_nodes=7, max_depth=3, max_n=3, beam_width=2, min_samples_leaf=5, max_features_per_node=12, random_state=42)
    res = {"purpose": "VERIFICAÇÃO DE FUNCIONAMENTO; sem métricas de desempenho; rascunho de IA não validado", "instance_hash": instance_hash(Xtr, names), "train_rows": int(len(Xtr)), "params": params, "arms": {}}
    for arm, (derived, prune) in {"A_ORIG": ([], False), "C_ENRICHED_NO_GUIDANCE": (lits, False), "F_EXPRESSIVE_MP": (lits, True)}.items():
        Xa, na = augment(Xtr, names, derived)
        c0 = calls["n"]
        m = H2Steered(derived=derived, base_names=names, prune_redundant=prune, units=[], pool_parity=True, **params).fit(Xa, oracle=Orc(), feature_names=na)
        tr = m.semantic_trace_
        res["arms"][arm] = {"n_derived": len(derived), "n_internal_nodes": tr["n_internal_nodes"], "oracle_calls": calls["n"] - c0, "queries": int(m.membership_queries_), "pruned_total": tr["pruned_total"],
                            "literals": [{"literal": l["literal"], "candidate_evaluations": l["candidate_evaluations"], "pruned": l["pruned"], "chosen_in_nodes": l["chosen_in_nodes"]} for l in tr["literals"]]}
        print(arm, "derivados", len(derived), "nós", tr["n_internal_nodes"], "literais avaliados como candidatos:", sum(1 for l in tr["literals"] if l["candidate_evaluations"] > 0), "escolhidos:", sum(1 for l in tr["literals"] if l["chosen_in_nodes"]), "podas", tr["pruned_total"])
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
