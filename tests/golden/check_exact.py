"""Verificação EXATA (bit-a-bit) das âncoras num processo com ambiente numérico fixado (como validation/benchmark/run.py).

A âncora exata depende do kernel BLAS do anfitrião (o oráculo sintético usa ``@``); fixar OPENBLAS_CORETYPE torna-a reprodutível.
Imprime em JSON a lista de divergências; código de saída 0 se vazia.
"""
import os
for _k, _v in (("OPENBLAS_CORETYPE", "Haswell"), ("OPENBLAS_NUM_THREADS", "1"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1")):
    os.environ[_k] = _v
import json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from eq_support import compare_golden, fingerprint, fit_model, make_problem, scenario_fit_kwargs
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier

G = json.loads((ROOT / "tests" / "golden" / "golden_effective.json").read_text())["entries"]
FLAGS = {"default": {}, "no_mirror": {"mirror_when_no_semantic_effect": False},
         "no_efsr": {"error_focused_refinement": False, "mirror_when_no_semantic_effect": False}}
kind, p, k, seed = "continuous", 8, 3, 3
X, names, oracle = make_problem(seed, p=p, k=k, kind=kind)
bad = []
m, spy = fit_model(TrepanOriginalClassifier, {}, X[:200], names, oracle, seed)
d = compare_golden(fingerprint(m, spy, X[200:]), G[f"{kind}-p{p}-k{k}-s{seed}|original"])
if d: bad.append(["original", d])
for sc in ("depths", "weights", "groups", "relatedness", "projector", "combined"):
    for fl, kw in FLAGS.items():
        m, spy = fit_model(TrepanReloadedClassifier, kw, X[:200], names, oracle, seed, fit_kw=scenario_fit_kwargs(sc, p))
        d = compare_golden(fingerprint(m, spy, X[200:]), G[f"{kind}-p{p}-k{k}-s{seed}|{sc}|{fl}"], m)
        if d: bad.append([f"{sc}|{fl}", d])
print(json.dumps(bad))
sys.exit(1 if bad else 0)
