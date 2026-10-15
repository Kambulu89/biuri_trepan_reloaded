"""Estudo isolado do desempate m-of-n (Doc. 19). Compara Original, variante(score_digits=None ≡ Original) e variante(q) em:
 (S) problemas sintéticos com oráculo de forma fechada; (D) conjuntos DEV com MLP congelado (pinned env); (N) injeção de ruído de 1–3 ulp
na pontuação (simula libm/BLAS de outra plataforma); (Q) sensibilidade à quantização. Nunca altera o Original. Uso:
  python -m validation.experimental.tiebreak_study --out DIR [--dev]
"""
from __future__ import annotations

from validation.benchmark import run as _R  # noqa: F401  (fixa o ambiente numérico antes de importar numpy)

import argparse
import hashlib
import json
import pathlib
import platform

import numpy as np

from core.trepan_original import TrepanOriginalClassifier
from tests.eq_support import BUDGET, SpyOracle, ClosedFormOracle, digest
from validation.experimental.deterministic_tiebreak import DeterministicTieBreakTrepan, canonical_signature


def _fit(cls, X, names, oracle, seed, budget=BUDGET, **kw):
    spy = SpyOracle(oracle)
    m = cls(**budget, random_state=seed, **kw)
    m.fit(X, oracle=spy, feature_names=names)
    return m, spy


def _summ(m, spy, X):
    return dict(rules=digest(m.export_rules()), pred=hashlib.sha256(np.ascontiguousarray(m.predict(X)).tobytes()).hexdigest(),
                stream=spy.stream_digest(), q=int(m.oracle_query_count_), nodes=len(list(m.iter_nodes())),
                fid=float(np.mean(m.predict(X) == spy.inner.predict(X))))


class _Noise:
    """Contexto que injeta 0–3 ulp determinísticos por máscara na pontuação de ``cls``."""
    def __init__(self, cls, ns):
        self.cls, self.ns = cls, ns

    def __enter__(self):
        self.orig = self.cls.__dict__.get("_split_selection_score")
        base = self.cls._split_selection_score
        ns = self.ns

        def noisy(me, y, mask, test=None):
            s = base(me, y, mask, test)
            h = int.from_bytes(hashlib.blake2b(mask.tobytes() + bytes([ns]), digest_size=2).digest(), "little") & 3
            for _ in range(h):
                s = float(np.nextafter(s, np.inf))
            return s
        self.cls._split_selection_score = noisy
        return self

    def __exit__(self, *a):
        if self.orig is None:
            del self.cls._split_selection_score
        else:
            self.cls._split_selection_score = self.orig


def _problems(n_seeds):
    for seed in range(n_seeds):
        for kind in ("continuous", "mixed"):
            for p, k in ((8, 3), (12, 2), (6, 4)):
                X, names, o = _make_problem(seed, p, k, kind)
                yield f"syn{kind[:3]}_p{p}k{k}_s{seed}", X, names, o, seed


def _make_problem(seed, p, k, kind):
    from tests.eq_support import make_problem
    return make_problem(seed, p=p, k=k, kind=kind)


def _dev_problems(seeds=(42, 7, 123)):
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    for name, ld in (("iris", load_iris), ("wine", load_wine), ("breast_cancer", load_breast_cancer)):
        d = ld()
        for s in seeds:
            Xtr, Xte, ytr, _ = train_test_split(d.data, d.target, test_size=0.3, random_state=s, stratify=d.target)
            sc = StandardScaler().fit(Xtr)
            mlp = MLPClassifier((32, 16), max_iter=800, random_state=s).fit(sc.transform(Xtr), ytr)

            class O:
                classes_ = mlp.classes_
                def predict(self, Z, mlp=mlp, sc=sc): return mlp.predict(sc.transform(Z))
                def predict_proba(self, Z, mlp=mlp, sc=sc): return mlp.predict_proba(sc.transform(Z))
            yield f"{name}_s{s}", Xtr, [str(n) for n in d.feature_names], O(), s


def run(out: pathlib.Path, dev: bool, n_seeds: int):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    probs = list(_problems(n_seeds)) + (list(_dev_problems()) if dev else [])
    for pid, X, names, oracle, seed in probs:
        o, so = _fit(TrepanOriginalClassifier, X, names, oracle, seed)
        v0, s0 = _fit(DeterministicTieBreakTrepan, X, names, oracle, seed, score_digits=None)
        base = _summ(o, so, X)
        row = dict(id=pid, orig=base, disabled_equals_orig=_summ(v0, s0, X) == base)
        for q in (6, 9, 12, 14):
            v, sv = _fit(DeterministicTieBreakTrepan, X, names, oracle, seed, score_digits=q)
            r = _summ(v, sv, X)
            r["rules_eq_orig"], r["pred_eq_orig"], r["stream_eq_orig"] = r["rules"] == base["rules"], r["pred"] == base["pred"], r["stream"] == base["stream"]
            row[f"var{q}"] = r
        # ruído de ulp
        for label, cls, kw in (("orig", TrepanOriginalClassifier, {}), ("var12", DeterministicTieBreakTrepan, {"score_digits": 12})):
            ch = []
            ref = None
            for ns in (0, 1, 2, 3):
                with _Noise(cls, ns):
                    m, s = _fit(cls, X, names, oracle, seed, **kw)
                r = _summ(m, s, X)
                ref = ref or r
                ch.append(dict(rules=r["rules"] == ref["rules"], pred=r["pred"] == ref["pred"], stream=r["stream"] == ref["stream"], fid=r["fid"]))
            row[f"noise_{label}"] = ch
        rows.append(row)
        print(pid, "rules_changed_by_noise orig:", sum(not c["rules"] for c in row["noise_orig"]), "var12:", sum(not c["rules"] for c in row["noise_var12"]), flush=True)
    meta = dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__)
    (out / "tiebreak_study.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1, default=str))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    run(pathlib.Path(a.out), a.dev, a.seeds)
