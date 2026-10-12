"""Gera a âncora dourada do comportamento do Reloaded (Fase 1a, passo 1) — executar ANTES de alterar core/.

Uso: ``python tests/golden/make_effective_anchors.py [--out tests/golden/golden_effective.json]``. Só dados sintéticos; sem datasets conhecidos.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np

from eq_support import fit_model, fingerprint, make_problem, scenario_fit_kwargs
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier

SEEDS = (3, 11, 29)
PROBLEMS = (("continuous", 8, 3), ("mixed", 6, 2), ("continuous", 10, 4))
SCENARIOS = ("none", "depths_zero", "groups_unique", "depths", "weights", "groups", "relatedness", "projector", "combined")
FLAGS = (("default", {}), ("no_mirror", {"mirror_when_no_semantic_effect": False}),
         ("no_efsr", {"error_focused_refinement": False, "mirror_when_no_semantic_effect": False}))


def collect():
    out = {}
    for kind, p, k in PROBLEMS:
        for seed in SEEDS:
            X, names, oracle = make_problem(seed, p=p, k=k, kind=kind)
            Xtr, Xte = X[:200], X[200:]
            key = f"{kind}-p{p}-k{k}-s{seed}"
            m, spy = fit_model(TrepanOriginalClassifier, {}, Xtr, names, oracle, seed)
            out[f"{key}|original"] = fingerprint(m, spy, Xte)
            for sc in SCENARIOS:
                for fl, kw in FLAGS:
                    m, spy = fit_model(TrepanReloadedClassifier, kw, Xtr, names, oracle, seed, fit_kw=scenario_fit_kwargs(sc, p))
                    out[f"{key}|{sc}|{fl}"] = fingerprint(m, spy, Xte)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "tests/golden/golden_effective.json"))
    a = ap.parse_args()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    core = subprocess.run(["git", "hash-object", "core/trepan_reloaded_historical.py", "core/trepan_original.py"], cwd=ROOT,
                          capture_output=True, text=True).stdout.split()
    doc = {"generator": "tests/golden/make_effective_anchors.py", "git_head": head,
           "blob_reloaded": core[0], "blob_original": core[1], "numpy": np.__version__, "entries": collect()}
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(len(doc["entries"]), "entradas ->", a.out)
