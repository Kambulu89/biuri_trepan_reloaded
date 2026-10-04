"""Contrato de experimento científico: um único oráculo congelado para todas as árvores de um benchmark.

Fluxo obrigatório do modo científico/benchmark::

    dataset -> split -> treino/calibração do MLP -> CONGELAR o oráculo -> TREPAN Original, Reloaded e variantes

Todas as árvores consultam exatamente o mesmo modelo (mesmos pesos, mesma função de decisão). A prova fica registada:

- ``oracle_id``: hash estável dos pesos/parâmetros ajustados do modelo + das suas decisões num conjunto-sonda fixo;
- por árvore: ``oracle_id`` visto no momento da consulta, nº de chamadas e de queries feitas;
- verificação final de que o oráculo não mudou durante o benchmark.

O módulo não contém lógica por dataset nem usa o conjunto de teste (a sonda é o treino).
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

import numpy as np


class OracleContractViolation(RuntimeError):
    """O benchmark deixou de cumprir o contrato (oráculo diferente, alterado ou reajustado)."""


def _feed(h, value: Any) -> None:
    if isinstance(value, np.ndarray):
        h.update(str(value.dtype).encode()); h.update(str(value.shape).encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (list, tuple)) and value and all(isinstance(v, np.ndarray) for v in value):
        for v in value:
            _feed(h, v)
    elif isinstance(value, (int, float, str, bool, np.integer, np.floating)):
        h.update(repr(value).encode())


def _walk_estimators(est: Any, seen: Optional[set] = None) -> Iterator[Any]:
    seen = set() if seen is None else seen
    if id(est) in seen or est is None:
        return
    seen.add(id(est))
    yield est
    for _name, step in list(getattr(est, "steps", []) or []):
        yield from _walk_estimators(step, seen)
    for sub in list(getattr(est, "estimators_", []) or []):
        if hasattr(sub, "get_params"):
            yield from _walk_estimators(sub, seen)
    inner = getattr(est, "estimator_", None) or getattr(est, "base_estimator_", None)
    if inner is not None and hasattr(inner, "get_params"):
        yield from _walk_estimators(inner, seen)


def weights_fingerprint(model: Any) -> str:
    """Hash dos hiperparâmetros simples e de todos os arrays ajustados (atributos terminados em ``_``) do modelo.

    Percorre Pipelines/ensembles. Dois modelos com os mesmos pesos têm o mesmo hash; qualquer ``fit``/``partial_fit``
    posterior muda-o.
    """
    h = hashlib.sha256()
    for est in _walk_estimators(model):
        h.update(type(est).__name__.encode())
        try:
            params = est.get_params(deep=False)
        except Exception:  # noqa: BLE001 - estimadores não-sklearn
            params = {}
        for k in sorted(params):
            _feed(h, k); _feed(h, params[k])
        for k in sorted(vars(est)):
            if k.endswith("_") and not k.startswith("_"):
                _feed(h, k); _feed(h, vars(est)[k])
    return h.hexdigest()


def decisions_fingerprint(model: Any, X_probe) -> str:
    """Hash das decisões do modelo num conjunto-sonda fixo (identidade comportamental)."""
    X = np.asarray(X_probe, dtype=float)
    h = hashlib.sha256()
    _feed(h, np.asarray(model.predict(X)).astype(str))
    if hasattr(model, "predict_proba"):
        _feed(h, np.round(np.asarray(model.predict_proba(X), dtype=float), 10))
    return h.hexdigest()


@dataclass(frozen=True)
class OracleIdentity:
    oracle_id: str
    kind: str
    builder: str
    weights_hash: str
    decisions_hash: str
    probe_rows: int
    classes: tuple

    def to_dict(self) -> dict:
        d = asdict(self)
        d["classes"] = [str(c) for c in self.classes]
        return d


def oracle_identity(model: Any, X_probe, *, builder: str = "external") -> OracleIdentity:
    w = weights_fingerprint(model)
    d = decisions_fingerprint(model, X_probe)
    oid = hashlib.sha256((w + d).encode()).hexdigest()[:16]
    classes = tuple(getattr(model, "classes_", ()) if getattr(model, "classes_", None) is not None else ())
    return OracleIdentity(oid, type(model).__name__, builder, w, d, int(len(np.asarray(X_probe))), classes)


class FrozenOracle:
    """Oráculo imutável: expõe ``predict``/``predict_proba``, conta queries e recusa qualquer reajuste.

    O mesmo objeto é entregue a TODAS as árvores do benchmark; ``scope(label)`` regista, por árvore, o ``oracle_id``
    consultado e quantas chamadas/queries fez.
    """

    ORACLE_TYPE = "frozen_oracle"

    def __init__(self, model: Any, X_probe, *, builder: str = "external"):
        self._model = model
        self.identity = oracle_identity(model, X_probe, builder=builder)
        self.oracle_id = self.identity.oracle_id
        self.classes_ = getattr(model, "classes_", None)
        self._scope: Optional[str] = None
        self.calls: Dict[str, dict] = {}

    # -- consultas ---------------------------------------------------------------------------------------------
    def _count(self, n: int) -> None:
        label = self._scope or "(sem escopo)"
        rec = self.calls.setdefault(label, {"oracle_id": self.oracle_id, "calls": 0, "queries": 0})
        rec["calls"] += 1; rec["queries"] += int(n)

    def predict(self, X):
        self._count(len(X))
        return self._model.predict(X)

    def predict_proba(self, X):
        self._count(len(X))
        return self._model.predict_proba(X)

    def __getattr__(self, name):                       # classes_, n_features_in_, ... (só leitura)
        if name.startswith("_") or name in {"fit", "partial_fit", "set_params", "fit_transform"}:
            raise AttributeError(name)
        return getattr(self._model, name)

    def fit(self, *_a, **_k):
        raise OracleContractViolation("O oráculo está congelado: fit() é proibido durante o benchmark.")

    partial_fit = fit

    @property
    def model(self):
        return self._model

    @contextmanager
    def scope(self, label: str):
        previous = self._scope
        self._scope = str(label)
        self.calls.setdefault(self._scope, {"oracle_id": self.oracle_id, "calls": 0, "queries": 0})
        try:
            yield self
        finally:
            self._scope = previous

    def verify_unchanged(self) -> bool:
        if weights_fingerprint(self._model) != self.identity.weights_hash:
            raise OracleContractViolation("Os pesos do oráculo mudaram depois de congelado.")
        return True

    def report(self) -> dict:
        return {"oracle": self.identity.to_dict(), "trees": {k: dict(v) for k, v in self.calls.items()},
                "unchanged_after": bool(self.verify_unchanged())}


def oracle_id_of(obj: Any) -> Optional[str]:
    """``oracle_id`` do oráculo realmente consultado por uma árvore (desembrulha projeções e envoltórios)."""
    seen = 0
    while obj is not None and seen < 8:
        oid = getattr(obj, "oracle_id", None) if isinstance(obj, FrozenOracle) else None
        if oid:
            return oid
        obj = getattr(obj, "oracle", None) or getattr(obj, "base_oracle", None)
        seen += 1
    return None


def freeze_oracle(model: Any, X_probe, *, builder: str = "external") -> FrozenOracle:
    return FrozenOracle(model, X_probe, builder=builder)


# --- Construtores de oráculo registados (um por contrato; configuráveis, sem nomes de datasets) ----------------------
def _build_factory(Z, y, seed: int):
    from core.mlp_convergence import fit_with_convergence
    from core.mlp_factory import build_mlp_for_data
    mlp = build_mlp_for_data(Z, y, random_state=seed)
    fit_with_convergence(mlp, Z, y)
    return mlp


class _LabelDecoded:
    """Modelo treinado sobre rótulos inteiros que expõe os rótulos originais (a GUI codifica o alvo antes de treinar)."""

    def __init__(self, estimator, classes):
        self.estimator_ = estimator
        self.classes_ = np.asarray(classes)

    def predict(self, X):
        return self.classes_[np.asarray(self.estimator_.predict(X)).astype(int)]

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def _build_robust(Z, y, seed: int):
    """MLP robusto (busca com holdout interno só no treino), o mesmo caminho de treino usado pela GUI."""
    from core.mlp_optimizer import train_robust_mlp_original
    classes, y_int = np.unique(np.asarray(y), return_inverse=True)
    result = train_robust_mlp_original(Z, y_int, Z, y_int, mode="balanced")   # X_test=treino: só relato, nunca teste externo
    return _LabelDecoded(result["model"], classes)


ORACLE_BUILDERS: Dict[str, Callable[[Any, Any, int], Any]] = {"factory": _build_factory, "robust": _build_robust}


def build_frozen_oracle(Z_train, y_train, *, seed: int, builder: str = "factory") -> FrozenOracle:
    """Treina/calibra o MLP SÓ no treino e congela-o. ``builder`` escolhe um construtor registado."""
    if builder not in ORACLE_BUILDERS:
        raise ValueError(f"Construtor de oráculo desconhecido: {builder!r}; registados: {sorted(ORACLE_BUILDERS)}")
    model = ORACLE_BUILDERS[builder](np.asarray(Z_train, dtype=float), np.asarray(y_train), int(seed))
    return FrozenOracle(model, Z_train, builder=builder)


def run_benchmark_with_frozen_oracle(
    oracle: FrozenOracle, trees: Mapping[str, Callable[[FrozenOracle], Any]],
) -> dict:
    """Ajusta cada variante (``trees[label](oracle) -> árvore``) sob o mesmo oráculo e devolve a prova do contrato.

    Levanta ``OracleContractViolation`` se o oráculo mudar durante o benchmark. O resultado traz o ``oracle_id`` único
    e, por árvore, as chamadas/queries efetuadas e o ``oracle_id`` observado.
    """
    fitted = {}
    for label, build in trees.items():
        with oracle.scope(label):
            fitted[label] = build(oracle)
    report = oracle.report()
    ids = {rec["oracle_id"] for rec in report["trees"].values()}
    report["single_oracle"] = bool(len(ids) <= 1 and ids <= {oracle.oracle_id})
    if not report["single_oracle"]:
        raise OracleContractViolation(f"Árvores consultaram oráculos diferentes: {sorted(ids)}")
    report["trees"] = {k: {**v, "oracle_id": oracle.oracle_id} for k, v in report["trees"].items()}
    report["fitted"] = fitted
    return report


__all__ = ["FrozenOracle", "OracleContractViolation", "OracleIdentity", "ORACLE_BUILDERS", "build_frozen_oracle",
           "freeze_oracle", "oracle_id_of", "oracle_identity", "run_benchmark_with_frozen_oracle", "weights_fingerprint",
           "decisions_fingerprint"]
