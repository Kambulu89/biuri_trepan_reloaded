"""Estudo separado «Original-det vs Reloaded-det» (D4; Doc. 19 §9). EXPERIMENTAL e ISOLADO: o desempate determinístico é aplicado a TODOS os braços do modo
``det`` por parcheamento transitório (``deterministic_policy``); o modo ``canonical`` é o código tal como está. NUNCA se usa para afirmar superioridade
face ao Original canónico. Tuning estrutural desligado (orçamento fixo): o estudo mede estabilidade e a diferença Reloaded−Original em cada modo.
    python -m validation.experimental.det_reloaded_study --unit iris 42 --out FILE.json
"""
from __future__ import annotations

from validation.benchmark import run as R      # fixa o ambiente numérico

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from core.benchmark import runner as RN
from core.trepan_original import TrepanOriginalClassifier
from validation.benchmark import datasets as ds_mod
from validation.experimental.deterministic_tiebreak import deterministic_policy

ARMS = ("trepan_original", "reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled")


class _Noise:
    """Ruído determinístico de 0–3 ulp por máscara na pontuação-base (simula libm/BLAS de outra plataforma). Aplicado à classe base: o Reloaded herda-o."""
    def __init__(self, ns):
        self.ns = ns

    def __enter__(self):
        self.orig = TrepanOriginalClassifier._split_selection_score
        base, ns = self.orig, self.ns

        def noisy(me, y, mask, test=None):
            s = base(me, y, mask, test)
            h = int.from_bytes(hashlib.blake2b(np.ascontiguousarray(mask).tobytes() + bytes([ns]), digest_size=2).digest(), "little") & 3
            for _ in range(h):
                s = float(np.nextafter(s, np.inf))
            return s
        TrepanOriginalClassifier._split_selection_score = noisy

    def __exit__(self, *a):
        TrepanOriginalClassifier._split_selection_score = self.orig


def _run(dataset: str, seed: int, mode: str, noise: int | None):
    spec = ds_mod.REGISTRY[dataset]
    ds, _t, _c = spec.load()
    cfg = dataclasses.replace(R.benchmark_config(seed), structure_tuning=False)
    arms = [a for a in RN.default_arms(cfg) if a.arm_id in ARMS + ("mlp_original",)]
    from contextlib import ExitStack
    with ExitStack() as st:
        if mode == "det":
            st.enter_context(deterministic_policy(12))
        if noise is not None:
            n = _Noise(noise); n.__enter__(); st.callback(n.__exit__)
        res = RN.BenchmarkRunner(cfg, arms).run(ds, str(spec.ontology_path) if spec.ontology_path else None)
    out = {}
    for r in res.rows:
        if r["arm"] in ARMS:
            out[r["arm"]] = {"fid": r["fidelity_to_oracle"], "acc": r["accuracy_real_labels"], "nodes": r["node_count"]}
    pred_cols = [c for c in res.predictions.columns if c.startswith("pred__")]
    for c in pred_cols:
        a = c[len("pred__"):]
        if a in out:
            out[a]["pred_hash"] = hashlib.sha256(np.ascontiguousarray(res.predictions[c].to_numpy()).tobytes()).hexdigest()[:16]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds, seed = a.unit[0], int(a.unit[1])
    res = {"dataset": ds, "seed": seed, "structure_tuning": False, "canonical": _run(ds, seed, "canonical", None), "det": _run(ds, seed, "det", None),
           "noise_canonical": [_run(ds, seed, "canonical", k) for k in (1, 2)], "noise_det": [_run(ds, seed, "det", k) for k in (1, 2)]}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
