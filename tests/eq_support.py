"""Infraestrutura partilhada dos testes de equivalência EQ (Fase 1a): oráculo-espião, cenários sintéticos e impressão digital.

Tudo sintético e dataset-agnóstico: o oráculo é uma função fechada determinística (sem MLP), logo as impressões digitais não dependem
do ambiente numérico (BLAS) e podem ser versionadas.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import numpy as np

from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier


class ClosedFormOracle:
    """Oráculo determinístico: softmax de um modelo linear com um termo não linear; ``classes_`` = 0..k-1."""

    def __init__(self, n_features: int, n_classes: int, seed: int):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(n_features, n_classes))
        self.V = rng.normal(size=(n_features, n_classes)) * 0.5
        self.classes_ = np.arange(n_classes)

    def _scores(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.W + np.tanh(X) @ self.V

    def predict(self, X):
        return np.argmax(self._scores(X), axis=1)

    def predict_proba(self, X):
        s = self._scores(X)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)


def _round_sig(a: np.ndarray, sig: int) -> np.ndarray:
    """Arredonda a ``sig`` algarismos significativos (vetorizado); zeros e não finitos ficam como estão."""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.floor(np.log10(np.abs(a)))
        dec = np.where(np.isfinite(mag), sig - 1 - mag, 0).astype(int)
    out = np.where(np.isfinite(mag), np.round(a * 10.0 ** dec) / 10.0 ** dec, a)
    return np.ascontiguousarray(out)


class SpyOracle:
    """Regista método, forma e bytes de tudo o que recebe; opcionalmente falha em ``predict_proba``."""

    def __init__(self, inner, *, forbid_proba: bool = False):
        self.inner, self.forbid_proba = inner, forbid_proba
        self.h = hashlib.sha256()
        self.hr = hashlib.sha256()          # fluxo com entradas arredondadas (portável entre plataformas)
        self.calls = {"predict": 0, "proba": 0}
        self.samples = {"predict": 0, "proba": 0}
        self.sequence: list[tuple[str, tuple[int, ...]]] = []

    def _see(self, kind, X):
        a = np.ascontiguousarray(np.asarray(X, dtype=float))
        self.h.update(kind.encode() + str(a.shape).encode() + a.tobytes())
        self.hr.update(kind.encode() + str(a.shape).encode() + _round_sig(a, 8).tobytes())
        self.calls[kind] += 1
        self.samples[kind] += len(a)
        self.sequence.append((kind, tuple(a.shape)))

    def predict(self, X):
        self._see("predict", X)
        return self.inner.predict(X)

    def predict_proba(self, X):
        if self.forbid_proba:
            raise AssertionError("predict_proba chamado no ramo sem enriquecimento ontológico (violação de EQ)")
        self._see("proba", X)
        return self.inner.predict_proba(X)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def stream_digest(self, sig: Optional[int] = None) -> str:
        return self.h.hexdigest() if sig is None else self.hr.hexdigest()


def make_problem(seed: int, n: int = 240, p: int = 8, k: int = 3, kind: str = "continuous"):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    if kind == "mixed":      # binárias, discretas e escalas muito diferentes (dataset-agnóstico)
        X[:, 0] = (X[:, 0] > 0).astype(float)
        X[:, 1] = np.round(X[:, 1] * 2)
        X[:, 2] *= 1000.0
        X[:, 3] = 1.0 if p > 3 else X[:, 3]   # constante
    names = [f"a{i}" for i in range(p)]
    return X, names, ClosedFormOracle(p, k, seed + 1)


BUDGET = dict(min_sample=200, max_queries=1500, max_nodes=9, max_depth=4, max_n=3, beam_width=2, min_samples_leaf=2,
              max_features_per_node=8, purity_epsilon=0.05)


def _jsonable(o: Any, sig: Optional[int] = None):
    """``sig=None``: floats exatos (``repr``); ``sig=n``: n algarismos significativos (portável entre plataformas/libm)."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v, sig) for k, v in sorted(o.items(), key=lambda kv: str(kv[0])) if "time" not in str(k)}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v, sig) for v in (sorted(o, key=str) if isinstance(o, set) else o)]
    if isinstance(o, np.ndarray):
        return [_jsonable(v, sig) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return repr(float(o)) if sig is None else format(float(o), f".{sig}g")
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (int, str, bool)) or o is None:
        return o
    return repr(o)


def digest(obj: Any, sig: Optional[int] = None) -> str:
    return hashlib.sha256(json.dumps(_jsonable(obj, sig), sort_keys=True).encode()).hexdigest()


def fit_model(cls, kw: dict, X, names, oracle, seed: int, *, fit_kw: Optional[dict] = None, forbid_proba: bool = False):
    spy = SpyOracle(oracle, forbid_proba=forbid_proba)
    model = cls(**BUDGET, random_state=seed, **kw)
    model.fit(X, oracle=spy, feature_names=names, **(fit_kw or {}))
    return model, spy


def fingerprint(model, spy: SpyOracle, Xte, sig: Optional[int] = None) -> dict:
    """Saídas observáveis E1–E9 (e auditorias de relato, em separado). ``sig``: floats das auditorias com n algarismos significativos
    (impressão digital portável); as saídas discretas (regras, previsões, contagens, fluxo ao oráculo) são sempre exatas."""
    nodes = list(model.iter_nodes())
    fp = {
        "rules": digest(model.export_rules()),
        "predictions": hashlib.sha256(np.ascontiguousarray(model.predict(Xte)).tobytes()).hexdigest(),
        "membership_queries": int(model.membership_queries_),
        "oracle_query_count": int(model.oracle_query_count_),
        "n_nodes": len(nodes),
        "oracle_stream": spy.stream_digest(sig),
        "oracle_calls": dict(spy.calls),
        "oracle_samples": dict(spy.samples),
        "oracle_sequence": digest(spy.sequence),
        "split_audit": digest(getattr(model, "split_audit_", []), sig),
        "expansion_order": digest(getattr(model, "expansion_order_", [])),
    }
    rep = {}
    for name in ("semantic_split_audit_", "error_region_audit_"):
        if hasattr(model, name):
            rep[name] = digest(getattr(model, name), sig)
    for name in ("semantic_projected_query_count_", "semantic_projection_batches_", "semantic_active_query_batches_",
                 "semantic_active_query_selected_", "error_focused_interventions_attempted_",
                 "error_focused_interventions_accepted_", "error_focused_interventions_rejected_",
                 "error_focused_query_batches_", "error_focused_query_selected_", "reload_extension_active_",
                 "semantic_effect_mirror_applied_", "semantic_effect_mirror_reason_"):
        if hasattr(model, name):
            rep[name] = _jsonable(getattr(model, name))
    fp["reporting"] = rep
    return fp


def domain_projector(lo: float = -2.5, hi: float = 2.5):
    def project(Z):
        return np.clip(np.asarray(Z, dtype=float), lo, hi)
    return project


def scenario_fit_kwargs(name: str, p: int) -> dict:
    """Informação ontológica sintética por cenário (sem nomes de datasets)."""
    depths = np.linspace(0.0, 1.0, p)
    weights = np.where(np.arange(p) % 2 == 0, 1.5, 1.0)
    groups = [f"g{i // 2}" for i in range(p)]
    rel = np.eye(p)
    for i in range(0, p - 1, 2):
        rel[i, i + 1] = rel[i + 1, i] = 0.8
    table = {
        "none": {},
        "depths_zero": {"semantic_feature_depths": np.zeros(p)},
        "groups_unique": {"semantic_feature_groups": [f"u{i}" for i in range(p)]},
        "depths": {"semantic_feature_depths": depths},
        "weights": {"semantic_feature_weights": weights},
        "groups": {"semantic_feature_groups": groups},
        "relatedness": {"semantic_relatedness_matrix": rel},
        "projector": {"query_projector": domain_projector()},
        "combined": {"semantic_feature_depths": depths, "semantic_feature_weights": weights,
                     "semantic_feature_groups": groups, "semantic_relatedness_matrix": rel,
                     "query_projector": domain_projector()},
    }
    return table[name]
