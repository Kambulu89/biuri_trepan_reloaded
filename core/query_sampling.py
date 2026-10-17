"""Amostrador EXPERIMENTAL de membership queries: «empírica conjunta com perturbação controlada» (estratégia C; Doc. 24/26).

Estado: experimental, DESLIGADO por omissão. Nada neste módulo é importado pelo Original canónico nem pelo Reloaded: o Original continua a usar o
``FeatureDistributionModel`` marginal. A ativação é sempre explícita (``joint_sampler_context`` / ``with_joint_sampler``), igual para o Original e para o
Reloaded, de modo que ``Reloaded(ontologia=None) ≡ Original`` também se verifica sob o novo amostrador (mesma classe, mesmas sementes, mesmo fluxo).

Propriedades (contrato, Doc. 26 §1):
* usa SÓ as linhas passadas a ``fit`` (treino do tronco/nó; nunca rótulos de teste); não conhece datasets;
* esquema de atributos INFERIDO dos dados (binária, grupo one-hot por exclusão mútua com soma 1, discreta inteira nominal, contínua) e partilhado entre
  o modelo global e os locais da mesma árvore (o tipo de uma coluna é propriedade da coluna, não do nó);
* linha-base real + perturbação: contínuas com ruído gaussiano (``noise_scale``·DP, DP local com piso ``scale_floor``·DP global) recortado a
  [lo−m·amp, hi+m·amp] do treino global; grupos one-hot reamostrados CONJUNTAMENTE; discretas reamostradas com a frequência empírica local (valores de outras linhas-base; vocabulário preservado); binárias trocadas
  com ``p_flip``; ordinais declaradas ±1 passo. Linhas que ficariam iguais à base sofrem perturbação forçada de uma unidade categórica (sem cópias);
* determinístico: RNG = SeedSequence([random_state, replica, 0xC5]); ``replica`` isola a aleatoriedade do AMOSTRADOR sem mexer na do algoritmo;
* mantém a rejeição por restrições do caminho (``draw`` herdado) e o teste de adoção local (``differs_from`` herdado, marginal por coluna).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.trepan_original import FeatureDistributionModel

DEFAULT_PARAMS = dict(noise_scale=0.25, p_resample=0.2, p_flip=0.05, margin=0.1, scale_floor=0.1)     # fixados ANTES das experiências (Doc. 26 §1)


@dataclass
class Schema:
    n_features: int
    continuous: List[int] = field(default_factory=list)
    binary: List[int] = field(default_factory=list)            # binárias que NÃO pertencem a um grupo one-hot
    onehot_groups: List[List[int]] = field(default_factory=list)
    discrete: List[int] = field(default_factory=list)          # inteiras com poucos níveis (nominais por códigos, salvo declaração ordinal)
    ordinal: List[int] = field(default_factory=list)
    vocab: dict = field(default_factory=dict)
    lo: Optional[np.ndarray] = None
    hi: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None


def infer_schema(X, *, max_levels: int = 20, ordinal_columns: Sequence[int] = ()) -> Schema:
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    sc = Schema(p, lo=X.min(axis=0), hi=X.max(axis=0), scale=np.maximum(X.std(axis=0), 1e-12))
    is_bin = [j for j in range(p) if set(np.unique(X[:, j])).issubset({0.0, 1.0}) and len(np.unique(X[:, j])) == 2]
    # Grupos one-hot = partições EXATAS das linhas por binárias de suportes disjuntos (soma por linha == 1 em TODAS as linhas). Construção gulosa por
    # frequência decrescente: um grupo só é aceite se cobrir todas as linhas (colunas raras de outros atributos que nunca coincidem por acaso não bastam).
    supp = {j: X[:, j] == 1.0 for j in is_bin}
    order = sorted(is_bin, key=lambda j: (-int(supp[j].sum()), j))
    assigned, groups = set(), []
    for j in order:
        if j in assigned:
            continue
        members, covered = [j], supp[j].copy()
        while not covered.all():
            cand = [k for k in order if k not in assigned and k not in members and not np.any(supp[k] & covered)]
            if not cand:
                break
            k = cand[0]
            members.append(k)
            covered |= supp[k]
        if covered.all() and len(members) >= 2:
            groups.append(sorted(members))
            assigned |= set(members)
    groups.sort()
    in_group = {j for g in groups for j in g}
    sc.onehot_groups = groups
    sc.binary = [j for j in is_bin if j not in in_group]
    for j in range(p):
        if j in in_group or j in sc.binary:
            continue
        col = X[:, j]
        u = np.unique(col)
        if np.all(col == np.round(col)) and len(u) <= max_levels:
            sc.discrete.append(j)
            sc.vocab[j] = u
        else:
            sc.continuous.append(j)
    sc.ordinal = [int(j) for j in ordinal_columns if j in sc.discrete]
    return sc


class JointEmpiricalDistributionModel(FeatureDistributionModel):
    """Substituto de ``FeatureDistributionModel`` com a mesma interface (``fit``, ``draw``, ``_draw_unconstrained``, ``differs_from``, ``training_columns_``).
    Configuração por atributos de CLASSE (``_params``, ``_replica``, ``_shared``) definidos por ``joint_sampler_context`` para que ``__init__`` mantenha a
    assinatura do Original (``random_state``, ``discrete_max_unique``)."""

    _params: Dict[str, float] = DEFAULT_PARAMS
    _replica: int = 0
    _shared: Optional[dict] = None
    _ordinal: Sequence[int] = ()

    def __reduce__(self):                                    # a classe é criada dinamicamente por árvore: reconstrói-a ao carregar
        state = self.__getstate__()
        return (_rebuild, (dict(self._params), int(self._replica), tuple(self._ordinal), state))

    def fit(self, X) -> "JointEmpiricalDistributionModel":
        X = np.asarray(X, dtype=float)
        super().fit(X)                                       # features_/training_columns_/n_features_in_ idênticos ao Original (teste de adoção local inalterado)
        self._rng = np.random.default_rng(np.random.SeedSequence([int(self.random_state) % (2 ** 32), int(self._replica), 0xC5]))
        finite_rows = X[np.isfinite(X).all(axis=1)]
        self._base = finite_rows if len(finite_rows) else None
        sh = self._shared if self._shared is not None else {}
        if "schema" not in sh:                               # o primeiro modelo da árvore é o global: fixa esquema e estatísticas de referência
            sh["schema"] = infer_schema(finite_rows if self._base is not None else np.zeros((1, X.shape[1])), ordinal_columns=self._ordinal)
            sh["audit"] = {"rows": 0, "forced": 0, "models": 0}
        self._schema = sh["schema"]
        self._audit = sh["audit"]
        self._audit["models"] += 1
        sd = np.maximum(finite_rows.std(axis=0), 1e-12) if self._base is not None else self._schema.scale
        self._sd = np.maximum(sd, self._params["scale_floor"] * self._schema.scale)
        return self

    def _draw_unconstrained(self, n: int) -> np.ndarray:
        if self._base is None or n <= 0:
            return super()._draw_unconstrained(n)
        P, s, rng, B = self._params, self._schema, self._rng, self._base
        base = B[rng.integers(0, len(B), n)]
        rows = base.copy()
        amp = s.hi - s.lo
        for j in s.continuous:
            rows[:, j] = np.clip(base[:, j] + rng.normal(0.0, P["noise_scale"] * self._sd[j], n), s.lo[j] - P["margin"] * amp[j], s.hi[j] + P["margin"] * amp[j])
        units = [("g", g) for g in s.onehot_groups] + [("d", j) for j in s.discrete] + [("b", j) for j in s.binary]
        for kind, u in units:
            if kind == "g":
                m = rng.random(n) < P["p_resample"]
                if m.any():
                    rows[np.ix_(m, u)] = B[rng.integers(0, len(B), int(m.sum()))][:, u]
            elif kind == "d":
                if u in s.ordinal:
                    step = rng.choice([-1.0, 0.0, 1.0], size=n, p=[P["p_resample"] / 2, 1 - P["p_resample"], P["p_resample"] / 2])
                    rows[:, u] = np.clip(base[:, u] + step, s.lo[u], s.hi[u])
                else:
                    m = rng.random(n) < P["p_resample"]
                    if m.any():
                        rows[m, u] = B[rng.integers(0, len(B), int(m.sum())), u]          # frequência empírica local
            else:
                m = rng.random(n) < P["p_flip"]
                rows[m, u] = 1.0 - rows[m, u]
        if units and not s.continuous:                       # sem contínuas, uma linha pode sair igual à base: força a perturbação de UMA unidade
            same = np.all(rows == base, axis=1)
            if same.any():
                idx = np.flatnonzero(same)
                pick = rng.integers(0, len(units), len(idx))
                for t, k in zip(idx, pick):
                    kind, u = units[k]
                    if kind == "g":
                        rows[t, u] = B[rng.integers(0, len(B))][u]
                    elif kind == "d":
                        rows[t, u] = B[rng.integers(0, len(B)), u]
                    else:
                        rows[t, u] = 1.0 - rows[t, u]
                self._audit["forced"] += int(len(idx))
        self._audit["rows"] += int(n)
        return rows


def _rebuild(params, replica, ordinal, state):
    cls, shared = make_joint_model_class(params, replica, ordinal)
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    if "_schema" in state:
        shared["schema"], shared["audit"] = state["_schema"], state["_audit"]
    return obj


def make_joint_model_class(params: Optional[dict] = None, replica: int = 0, ordinal_columns: Sequence[int] = ()):
    """Subclasse com configuração própria e esquema partilhado NOVO (um por árvore). Devolve (classe, dicionário partilhado/auditoria)."""
    shared: dict = {}
    P = dict(DEFAULT_PARAMS, **(params or {}))
    cls = type("JointEmpiricalDistributionModel", (JointEmpiricalDistributionModel,),
               {"_params": P, "_replica": int(replica), "_shared": shared, "_ordinal": tuple(ordinal_columns)})
    return cls, shared


@contextmanager
def joint_sampler_context(params: Optional[dict] = None, replica: int = 0, ordinal_columns: Sequence[int] = ()):
    """Ativa (só dentro do ``with``) o amostrador C no Original E no Reloaded: substitui o nome ``FeatureDistributionModel`` nos módulos que o
    instanciam. Fora do contexto nada muda (flag desligada por omissão). Não é seguro em múltiplos fios simultâneos (uso de execução sequencial)."""
    import core.trepan_original as TO
    import core.trepan_reloaded_historical as TR
    cls, shared = make_joint_model_class(params, replica, ordinal_columns)
    old = (TO.FeatureDistributionModel, TR.FeatureDistributionModel)
    TO.FeatureDistributionModel = cls
    TR.FeatureDistributionModel = cls
    try:
        yield shared
    finally:
        TO.FeatureDistributionModel, TR.FeatureDistributionModel = old
