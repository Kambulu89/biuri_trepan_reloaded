"""Medição emparelhada Original vs Reloaded SEM ontologia efetiva (Fase 1a, D-7/EQ): tempo + igualdade do fluxo de consultas ao oráculo.

Mesmo split, mesmo MLP/oráculo congelado, mesmos hiperparâmetros (valores por omissão das classes), ambiente numérico fixado como em ``run.py``.
Aquecimento + repetições alternadas (ABBA...). Não é resultado científico; DEV apenas.
    python -m validation.benchmark.eq_perf_paired --datasets iris wine breast_cancer digits --seeds 42 7 123 --reps 7 --out out.jsonl
"""
from __future__ import annotations

import os

for _k, _v in (("OPENBLAS_CORETYPE", "Haswell"), ("OPENBLAS_NUM_THREADS", "1"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
               ("NPY_DISABLE_CPU_FEATURES", "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_KNL AVX512_KNM")):
    os.environ.setdefault(_k, _v)

import argparse
import hashlib
import json
import statistics as st
import sys
import time
import warnings
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

LOADERS = {"iris": skd.load_iris, "wine": skd.load_wine, "breast_cancer": skd.load_breast_cancer, "digits": skd.load_digits}


class Spy:
    def __init__(self, inner):
        self.inner, self.h, self.calls, self.samples, self.seq = inner, hashlib.sha256(), 0, 0, hashlib.sha256()

    def _see(self, kind, X):
        a = np.ascontiguousarray(np.asarray(X, dtype=float))
        self.h.update(kind.encode() + str(a.shape).encode() + a.tobytes())
        self.seq.update(f"{kind}{a.shape};".encode())
        self.calls += 1
        self.samples += len(a)

    def predict(self, X):
        self._see("predict", X)
        return self.inner.predict(X)

    def predict_proba(self, X):
        self._see("proba", X)
        return self.inner.predict_proba(X)

    def __getattr__(self, n):
        return getattr(self.inner, n)


def one(cls, seed, Xtr, Xte, mlp, names):
    fo = freeze_oracle(mlp, Xtr, builder="eq_perf")
    spy = Spy(fo)
    w0, c0 = time.perf_counter(), time.process_time()
    with fo.scope("eq_perf"):
        tree = cls(random_state=seed).fit(Xtr, oracle=spy, feature_names=names)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    rules = tree.export_rules()
    return {"wall_s": wall, "cpu_s": cpu, "oracle_calls": spy.calls, "oracle_samples": spy.samples, "stream_hash": spy.h.hexdigest(),
            "sequence_hash": spy.seq.hexdigest(), "membership_queries": int(tree.membership_queries_), "nodes": len(list(tree.iter_nodes())),
            "rules_hash": hashlib.sha256(json.dumps(rules, default=str, sort_keys=True).encode()).hexdigest(),
            "pred_hash": hashlib.sha256(np.ascontiguousarray(tree.predict(Xte)).tobytes()).hexdigest(), "oracle_id": getattr(fo, "oracle_id", None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = open(a.out, "a")
    for name in a.datasets:
        d = LOADERS[name]()
        X, y = d.data.astype(float), d.target
        names = [f"x{i}" for i in range(X.shape[1])]
        for seed in a.seeds:
            Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
            mlp = build_mlp_for_data(Xtr, ytr, random_state=seed).fit(Xtr, ytr)
            one(TrepanOriginalClassifier, seed, Xtr, Xte, mlp, names); one(TrepanReloadedClassifier, seed, Xtr, Xte, mlp, names)   # aquecimento
            runs = {"O": [], "R": []}
            for r in range(a.reps):
                order = ("O", "R") if r % 2 == 0 else ("R", "O")
                for tag in order:
                    runs[tag].append(one(TrepanOriginalClassifier if tag == "O" else TrepanReloadedClassifier, seed, Xtr, Xte, mlp, names))
            same = {k: len({x[k] for t in runs for x in runs[t]}) == 1 for k in ("stream_hash", "sequence_hash", "rules_hash", "pred_hash", "oracle_calls",
                                                                                   "oracle_samples", "membership_queries", "nodes")}
            wo, wr = [x["wall_s"] for x in runs["O"]], [x["wall_s"] for x in runs["R"]]
            co, cr = [x["cpu_s"] for x in runs["O"]], [x["cpu_s"] for x in runs["R"]]
            ratios = [r / o for r, o in zip(wr, wo)]
            rec = {"dataset": name, "seed": seed, "n_train": len(Xtr), "n_features": X.shape[1], "reps": a.reps, "equal_all": same,
                   "ratio_wall_median_of_medians": st.median(wr) / st.median(wo), "ratio_cpu_median": st.median(cr) / st.median(co),
                   "ratio_wall_pairs_median": st.median(ratios), "ratio_wall_pairs_min": min(ratios), "ratio_wall_pairs_max": max(ratios),
                   "wall_original_median": st.median(wo), "wall_original_stdev": st.pstdev(wo), "wall_reloaded_median": st.median(wr),
                   "wall_reloaded_stdev": st.pstdev(wr), "oracle_calls": runs["O"][0]["oracle_calls"], "oracle_samples": runs["O"][0]["oracle_samples"],
                   "stream_hash": runs["O"][0]["stream_hash"], "sequence_hash": runs["O"][0]["sequence_hash"], "nodes": runs["O"][0]["nodes"],
                   "membership_queries": runs["O"][0]["membership_queries"], "oracle_id": runs["O"][0]["oracle_id"],
                   "loadavg_end": os.getloadavg(), "raw": runs}
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(name, seed, "ratio(wall med) %.3f cpu %.3f pairs[%.3f..%.3f] eq=%s" % (rec["ratio_wall_median_of_medians"], rec["ratio_cpu_median"],
                  rec["ratio_wall_pairs_min"], rec["ratio_wall_pairs_max"], all(same.values())), flush=True)


if __name__ == "__main__":
    main()
