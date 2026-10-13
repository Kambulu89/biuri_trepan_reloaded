"""Fase 1b/V2 — antes/depois da reutilização exata de chamadas byte-idênticas no mirror, em datasets DEV (sem avaliação confirmatória).

Por unidade (dataset, seed): Original; Reloaded com mirror e REUSE=False (comportamento pré-1b); Reloaded com mirror e REUSE=True (pós-1b).
Ontologia sintética genérica: ``query_projector`` = limites [min, max] de cada atributo do TREINO (dataset-agnóstico). Mesmo split, MLP, oráculo,
hiperparâmetros por omissão e ambiente numérico fixado. Regista consultas (chamadas/amostras por método), redundância exata, tempo,
memória (pico tracemalloc numa passagem separada), regras/previsões, fidelidade, accuracy.
    python -m validation.benchmark.v2_before_after --datasets iris wine breast_cancer digits --seeds 42 7 123 --out out.jsonl
"""
from __future__ import annotations

import os

for _k, _v in (("OPENBLAS_CORETYPE", "Haswell"), ("OPENBLAS_NUM_THREADS", "1"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_k, _v)

import argparse, hashlib, json, sys, time, tracemalloc, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split

from core.mlp_factory import build_mlp_for_data
from core.scientific_experiment_contract import freeze_oracle
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier
from validation.benchmark.eq_perf_paired import Spy

LOADERS = {"iris": skd.load_iris, "wine": skd.load_wine, "breast_cancer": skd.load_breast_cancer, "digits": skd.load_digits}


class CountSpy(Spy):
    def __init__(self, inner):
        super().__init__(inner)
        self.by = {"predict": [0, 0], "proba": [0, 0]}

    def predict(self, X):
        self.by["predict"][0] += 1; self.by["predict"][1] += len(X)
        return super().predict(X)

    def predict_proba(self, X):
        self.by["proba"][0] += 1; self.by["proba"][1] += len(X)
        return super().predict_proba(X)


def domain_projector(Xtr):
    lo, hi = np.min(Xtr, axis=0), np.max(Xtr, axis=0)
    return lambda Z: np.clip(np.asarray(Z, dtype=float), lo, hi)


def one(kind, seed, Xtr, Xte, yte, mlp, names, reuse=None, mem=False):
    fo = freeze_oracle(mlp, Xtr, builder="v2_before_after")
    spy = CountSpy(fo)
    if mem:
        tracemalloc.start()
    t0 = time.perf_counter()
    with fo.scope("v2"):
        if kind == "original":
            tree = TrepanOriginalClassifier(random_state=seed).fit(Xtr, oracle=spy, feature_names=names)
        else:
            TrepanReloadedClassifier.REUSE_IDENTICAL_ORACLE_CALLS = bool(reuse)
            tree = TrepanReloadedClassifier(random_state=seed).fit(Xtr, oracle=spy, feature_names=names, query_projector=domain_projector(Xtr))
    wall = time.perf_counter() - t0
    peak = None
    if mem:
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
    pred = tree.predict(Xte)
    oracle_te = mlp.predict(Xte)
    acc = getattr(tree, "oracle_accounting_", None)
    return {"wall_s": wall, "peak_mb": peak, "calls": dict(zip(("predict", "proba"), (spy.by["predict"][0], spy.by["proba"][0]))),
            "samples": dict(zip(("predict", "proba"), (spy.by["predict"][1], spy.by["proba"][1]))), "stream_hash": spy.h.hexdigest(),
            "rules_hash": hashlib.sha256(json.dumps(tree.export_rules(), default=str, sort_keys=True).encode()).hexdigest(),
            "pred_hash": hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest(), "nodes": len(list(tree.iter_nodes())),
            "fidelity": float(np.mean(pred == oracle_te)), "accuracy": float(np.mean(pred == yte)), "membership_queries": int(tree.membership_queries_),
            "mirror_applied": bool(getattr(tree, "semantic_effect_mirror_applied_", False)), "accounting": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True); ap.add_argument("--seeds", nargs="+", type=int, required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = open(a.out, "a")
    for name in a.datasets:
        d = LOADERS[name](); X, y = d.data.astype(float), d.target
        names = [f"x{i}" for i in range(X.shape[1])]
        for seed in a.seeds:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
            mlp = build_mlp_for_data(Xtr, ytr, random_state=seed).fit(Xtr, ytr)
            res = {}
            for tag, kw in (("original", dict(kind="original")), ("before_1b", dict(kind="reloaded", reuse=False)), ("after_1b", dict(kind="reloaded", reuse=True))):
                one(**kw, seed=seed, Xtr=Xtr, Xte=Xte, yte=yte, mlp=mlp, names=names)             # aquecimento
                res[tag] = one(**kw, seed=seed, Xtr=Xtr, Xte=Xte, yte=yte, mlp=mlp, names=names)
                res[tag]["peak_mb"] = one(**kw, seed=seed, Xtr=Xtr, Xte=Xte, yte=yte, mlp=mlp, names=names, mem=True)["peak_mb"]
            rec = {"dataset": name, "seed": seed, "n_train": len(Xtr), **res}
            out.write(json.dumps(rec) + "\n"); out.flush()
            o, b, c = res["original"], res["before_1b"], res["after_1b"]
            print(name, seed, "mirror", c["mirror_applied"], "| amostras O/antes/depois", sum(o["samples"].values()), sum(b["samples"].values()), sum(c["samples"].values()),
                  "| regras=O", b["rules_hash"] == o["rules_hash"] == c["rules_hash"], "| tempo %.2f/%.2f/%.2f" % (o["wall_s"], b["wall_s"], c["wall_s"]), flush=True)


if __name__ == "__main__":
    main()
