"""Validação do pipeline cfkit em datasets SINTÉTICOS (agnóstico; sem dados reais).

Corre LORE/CLEAR/CoGS (inspired) num MLP real, binário e multiclasse, com e sem
ontologia (OWL gerada em memória). Instâncias escolhidas por protocolo
(random/estratificado, seed fixa) a partir do TESTE; constraints/densidade só do TREINO.
Resultados (incl. falhas) escritos em --out. Não afina nada para favorecer um método.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from counterfactuals.cfkit import generate_counterfactual, run_cf_benchmark
from counterfactuals.cfkit.api import CFContext, explain_text, export_results
from counterfactuals.cfkit.rules import ConstraintSet, OntologyConstraintExtractor

warnings.filterwarnings("ignore")
NAMES = ["age", "children", "income", "sex", "emp_a", "emp_b", "emp_c"]
CFG = {"categorical_groups": {"employment": {"columns": NAMES[4:], "labels": ["a", "b", "c"]}},
       "immutable_features": ["sex"], "direction": {"age": "increase_only"}}


def make_data(n, seed, n_classes):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n).astype(float)
    kids = rng.integers(0, 5, n).astype(float)
    income = np.clip(rng.normal(40, 15, n) + 0.3 * (age - 40), 0, 200)
    sex = rng.integers(0, 2, n).astype(float)
    emp = np.eye(3)[rng.integers(0, 3, n)]
    X = np.column_stack([age, kids, income, sex, emp])
    score = 0.05 * (income - 40) + 0.4 * (emp[:, 1] - emp[:, 2]) - 0.15 * kids + rng.normal(0, 0.5, n)
    if n_classes == 2:
        y = (score > 0).astype(int)
    else:
        y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
    return X, y


def build_owl():
    try:
        import owlready2 as owl
    except ImportError:
        return None
    w = owl.World()
    onto = w.get_ontology("http://example.org/cfval#")
    with onto:
        class cf_dependsOn(owl.AnnotationProperty): pass
        class hasIncome(owl.DataProperty): range = [owl.ConstrainedDatatype(float, min_inclusive=0.0, max_inclusive=200.0)]
        class hasChildren(owl.DataProperty): range = [int]
    hasIncome.cf_dependsOn = ["children"]
    return onto


def run(n_classes, with_onto, out_dir, n_instances, seed):
    X, y = make_data(1500, seed, n_classes)
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    model = MLPClassifier((24,), max_iter=500, random_state=seed).fit(Xtr, ytr)
    if with_onto:
        onto = build_owl()
        ex = OntologyConstraintExtractor(onto, NAMES).extract()
        cs = ConstraintSet.from_config(Xtr, NAMES, CFG, ontology_constraints=ex)
        ctx = CFContext.build(model, Xtr, NAMES, constraints=cs, model_name="MLP", dataset_name="synthetic", ontology_hash="synthetic-owl")
    else:
        ctx = CFContext.build(model, Xtr, NAMES, config=CFG, model_name="MLP", dataset_name="synthetic")
    bench = run_cf_benchmark(model, Xte, ctx, ["LORE", "CLEAR", "COGS"], protocol="stratified",
                             n_instances=n_instances, seed=seed, max_iterations=25, max_time=20.0)
    tag = f"{'multiclass' if n_classes > 2 else 'binary'}_{'onto' if with_onto else 'noonto'}"
    i0 = bench["selection"]["indices"][0]
    pred = int(model.predict(Xte[i0:i0 + 1])[0])
    tgt = [c for c in model.classes_ if c != pred][0]
    examples = [generate_counterfactual(Xte[i0], tgt, model, m, context=ctx, random_state=seed, max_iterations=25, max_time=20.0)
                for m in ("LORE", "CLEAR", "COGS")]
    d = Path(out_dir) / tag
    d.mkdir(parents=True, exist_ok=True)
    export_results(examples, d, extra={"benchmark_summary": bench["summary"]})
    (d / "benchmark.json").write_text(json.dumps({k: bench[k] for k in ("selection", "summary", "rows")}, indent=2, default=str))
    (d / "example_text.txt").write_text("\n\n".join(explain_text(r) for r in examples))
    return tag, bench["summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cf_validation")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    summary = {}
    for k in (2, 3):
        for onto in (False, True):
            tag, s = run(k, onto, a.out, a.n, a.seed)
            summary[tag] = s
            print(tag)
            for m, v in s.items():
                print(f"  {m}: success={v['success_rate']:.2f} prox={v['mean_proximity']} spars={v['mean_sparsity']} "
                      f"plaus={v['mean_plausibility']} sem={v['semantic_validity_rate']} t={v['mean_runtime']:.2f}s {v['status_counts']}")
    Path(a.out, "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
