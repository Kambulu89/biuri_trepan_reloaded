#!/usr/bin/env python3
"""Atribuição do ganho do TREPAN Reloaded à ontologia (controlo negativo pareado).

Para cada semente (mesma divisão treino/teste, mesmo professor, mesmo orçamento de árvore):

* Original              : TREPAN Original;
* Reloaded real         : Reloaded com a ontologia real;
* Reloaded baralhado x K: a MESMA ontologia com a atribuição feature->entidade baralhada
                          (mesmas entidades e mesmo grafo, sem significado), aplicada antes de
                          construir pesos, grupos, relatedness, famílias, profundidades e restrições.

Ganho atribuível à ontologia, por semente:
    A = (BA_real - BA_original) - média_k (BA_baralhado_k - BA_original)  =  BA_real - média_k BA_baralhado_k

Se A não for sistematicamente > 0, o ganho do Reloaded NÃO vem do conhecimento da ontologia.
O teste externo só entra nas métricas finais; nada é ajustado depois de ver os resultados.

    python scripts/run_reloaded_attribution.py --dataset breast_cancer --seeds 1,2,3,4,5,6 --out out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

LOADERS = {"iris": "load_iris", "wine": "load_wine", "breast_cancer": "load_breast_cancer", "digits": "load_digits"}


def load(name: str, max_samples: int):
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    bunch = getattr(datasets, LOADERS[name])(as_frame=True)
    X = bunch.data.copy(); X.columns = [str(c) for c in X.columns]
    y = bunch.target.to_numpy()
    if len(X) > max_samples:
        X, _, y, _ = train_test_split(X, y, train_size=max_samples, random_state=0, stratify=y)
    df = X.reset_index(drop=True)
    df["target"] = [f"c{v}" for v in y]
    return df


def one_run(df, owl, seed, control, control_seed):
    from core.production_training import train_production_dataframe
    with tempfile.TemporaryDirectory() as tmp:
        rep = train_production_dataframe(
            df, target="target", out_dir=tmp, seed=seed, owl_path=owl, require_reasoner=True,
            scientific_tuning=False, semantic_control=control, semantic_control_seed=control_seed)
    ev = rep["evaluation"]
    audit = ev.get("semantic_audit") or {}
    return {
        "original": ev["models"]["original"], "reloaded": ev["models"]["reloaded"],
        "usage_rate": audit.get("ontology_usage_rate"), "decision_impact": audit.get("semantic_decision_impact"),
        "mirror_applied": (ev.get("experiment_audit") or {}).get("semantic_effect_mirror_applied"),
        "semantic_contribution": (ev.get("semantic_contribution_gate") or {}).get("status"),
    }


def boot_ci(values, n=5000, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--controls", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=800)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    owl = ROOT / "data" / "benchmark_ontologies" / f"{args.dataset}.owl"
    df = load(args.dataset, args.max_samples)
    runs = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        real = one_run(df, owl, seed, None, 0)
        ctrls = [one_run(df, owl, seed, "shuffled_semantics", seed * 1000 + k) for k in range(args.controls)]
        # a árvore Original tem de ser idêntica nos braços (mesmo split, professor e semente)
        for c in ctrls:
            assert abs(c["original"]["balanced_accuracy"] - real["original"]["balanced_accuracy"]) < 1e-12
        ba = lambda m: m["balanced_accuracy"]
        row = {
            "seed": seed, "ba_original": ba(real["original"]), "ba_reloaded_real": ba(real["reloaded"]),
            "ba_reloaded_shuffled": [ba(c["reloaded"]) for c in ctrls],
            "fid_original": real["original"]["oracle_fidelity"], "fid_reloaded_real": real["reloaded"]["oracle_fidelity"],
            "fid_reloaded_shuffled": [c["reloaded"]["oracle_fidelity"] for c in ctrls],
            "real_mirror_applied": real["mirror_applied"], "real_usage_rate": real["usage_rate"],
            "real_decision_impact": real["decision_impact"],
            "shuffled_mirror_applied": [c["mirror_applied"] for c in ctrls],
        }
        row["gain_real"] = row["ba_reloaded_real"] - row["ba_original"]
        row["gain_shuffled_mean"] = float(np.mean(row["ba_reloaded_shuffled"])) - row["ba_original"]
        row["attributable"] = row["ba_reloaded_real"] - float(np.mean(row["ba_reloaded_shuffled"]))
        row["controls_ge_real"] = int(sum(b >= row["ba_reloaded_real"] for b in row["ba_reloaded_shuffled"]))
        runs.append(row)
        print(f"{args.dataset} seed={seed} | Orig {row['ba_original']:.3f} Real {row['ba_reloaded_real']:.3f} "
              f"Shuf {np.mean(row['ba_reloaded_shuffled']):.3f} | attributable {row['attributable']:+.3f} "
              f"| controls>=real {row['controls_ge_real']}/{args.controls} | mirror(real)={row['real_mirror_applied']}", flush=True)
    A = [r["attributable"] for r in runs]
    G = [r["gain_real"] for r in runs]
    summary = {
        "dataset": args.dataset, "n_seeds": len(runs), "controls_per_seed": args.controls,
        "mean_gain_real_vs_original": float(np.mean(G)),
        "mean_attributable_to_ontology": float(np.mean(A)),
        "attributable_ci95": boot_ci(A), "seeds_attributable_positive": int(sum(a > 0 for a in A)),
        "seeds_attributable_negative": int(sum(a < 0 for a in A)),
        "controls_ge_real_total": int(sum(r["controls_ge_real"] for r in runs)),
        "controls_total": int(len(runs) * args.controls),
    }
    out = {"summary": summary, "runs": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
