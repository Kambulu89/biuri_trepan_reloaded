"""Amostradores EXPERIMENTAIS de consultas sintéticas (Doc. 24). Vivem em ``validation/experimental`` (nunca em ``core/``): servem para medir a plausibilidade,
a cobertura de fronteiras e o custo de estratégias alternativas ANTES de qualquer integração no TREPAN. Todos expõem ``draw(n, constraints=None)`` (como o
``FeatureDistributionModel`` do Original: ``constraints.accepts(X)`` filtra por rejeição) e são determinísticos dado ``random_state``.

Estratégias: A marginal histórica · As marginal + grupos categóricos conjuntos · B coerente (marginal nas originais + transformador) · C empírica conjunta com
perturbação controlada · D vizinhanças (interpolação entre uma linha e um dos seus k vizinhos) · E restrições de domínio explícitas (envolve qualquer amostrador).
Esquema inferido DOS DADOS, sem nomes de datasets: binário (∈{0,1}), grupo one-hot (binárias mutuamente exclusivas cuja soma por linha é sempre 1),
discreta/nominal por códigos inteiros (vocabulário observado), contínua. A ordinalidade NÃO é inferível dos dados: só se perturba ±1 passo em colunas declaradas
como ordinais (``ordinal_columns``); por omissão as inteiras são tratadas como nominais (reamostradas do vocabulário)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from core.trepan_original import FeatureDistributionModel


@dataclass
class Schema:
    n_features: int
    continuous: List[int] = field(default_factory=list)
    binary: List[int] = field(default_factory=list)            # binárias que NÃO pertencem a um grupo one-hot
    onehot_groups: List[List[int]] = field(default_factory=list)
    discrete: List[int] = field(default_factory=list)          # inteiras com poucos níveis (nominal por códigos, salvo declaração ordinal)
    ordinal: List[int] = field(default_factory=list)           # declaradas pelo utilizador (subconjunto de discrete)
    vocab: dict = field(default_factory=dict)                  # coluna -> valores observados
    lo: Optional[np.ndarray] = None
    hi: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None


def infer_schema(X, *, max_levels: int = 20, ordinal_columns: Sequence[int] = ()) -> Schema:
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    sc = Schema(p, lo=X.min(axis=0), hi=X.max(axis=0), scale=np.maximum(X.std(axis=0), 1e-12))
    is_bin = [j for j in range(p) if set(np.unique(X[:, j])).issubset({0.0, 1.0}) and len(np.unique(X[:, j])) == 2]
    # grupos one-hot: componentes conexas do grafo de exclusão mútua, verificadas por soma de linha == 1
    excl = {j: {k for k in is_bin if k != j and not np.any((X[:, j] == 1) & (X[:, k] == 1))} for j in is_bin}
    seen, groups = set(), []
    for j in is_bin:
        if j in seen:
            continue
        comp, stack = set(), [j]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(excl[u] - comp)
        seen |= comp
        cols = sorted(comp)
        if len(cols) >= 2 and np.all(X[:, cols].sum(axis=1) == 1.0):
            groups.append(cols)
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


class _Base:
    def __init__(self, X, *, transform: Optional[Callable] = None, random_state: int = 0, ordinal_columns: Sequence[int] = ()):
        self.X = np.asarray(X, dtype=float)
        self.schema = infer_schema(self.X, ordinal_columns=ordinal_columns)
        self.transform = transform
        self.rng = np.random.default_rng(random_state)
        self.n_features_in_ = self.X.shape[1]
        self.tries_ = 0
        self.accepted_ = 0

    def _raw(self, n: int) -> np.ndarray:                      # linhas no espaço ORIGINAL
        raise NotImplementedError

    def _finish(self, rows):
        return rows if self.transform is None else np.asarray(self.transform(rows), dtype=float)

    def draw(self, n: int, constraints=None) -> np.ndarray:
        out, remaining = [], int(n)
        for _ in range(200):
            if remaining <= 0:
                break
            batch = self._finish(self._raw(max(64, remaining * 3)))
            keep = batch if constraints is None else batch[constraints.accepts(batch)]
            self.tries_ += len(batch)
            self.accepted_ += len(keep)
            if len(keep):
                out.append(keep[:remaining])
                remaining -= min(len(keep), remaining)
        if remaining > 0:
            raise RuntimeError("amostrador: não foi possível satisfazer as restrições")
        return np.vstack(out)[:n]


class MarginalSampler(_Base):
    """A / As / B. ``on_enriched=True`` ⇒ A histórico no espaço enriquecido (sorteia também as derivadas, incoerente); ``joint_groups=True`` ⇒ As."""
    def __init__(self, X, *, transform=None, on_enriched=False, joint_groups=False, random_state=0, **kw):
        super().__init__(X, transform=transform, random_state=random_state, **kw)
        self.on_enriched, self.joint_groups = bool(on_enriched), bool(joint_groups)
        self._fit_X = np.asarray(transform(self.X), dtype=float) if (on_enriched and transform is not None) else self.X
        self.model = FeatureDistributionModel(random_state=random_state).fit(self._fit_X)

    def _raw(self, n):
        rows = self.model._draw_unconstrained(n)
        if self.joint_groups:
            for g in self.schema.onehot_groups:
                pick = self.rng.integers(0, len(self.X), n)
                rows[:, g] = self.X[pick][:, g]
        return rows

    def _finish(self, rows):
        if self.on_enriched or self.transform is None:
            return rows                                         # A: as derivadas (se existirem) já vêm sorteadas marginalmente
        return np.asarray(self.transform(rows), dtype=float)


class JointEmpiricalSampler(_Base):
    """C: linha real de base + perturbação controlada. Contínuas: ruído gaussiano ``noise_scale``·DP (recorte a [lo−m·amp, hi+m·amp]); binárias: troca com prob.
    ``p_flip``; grupos one-hot e discretas: reamostragem CONJUNTA do grupo a partir de outra linha com prob. ``p_resample``; ordinais declaradas: ±1 passo."""
    def __init__(self, X, *, transform=None, noise_scale=0.25, p_resample=0.2, p_flip=0.05, margin=0.1, random_state=0, **kw):
        super().__init__(X, transform=transform, random_state=random_state, **kw)
        self.noise_scale, self.p_resample, self.p_flip, self.margin = noise_scale, p_resample, p_flip, margin

    def _perturb_categorical(self, rows, src):
        n, s = len(rows), self.schema
        for g in s.onehot_groups:
            m = self.rng.random(n) < self.p_resample
            other = self.X[self.rng.integers(0, len(self.X), n)]
            rows[np.ix_(m, g)] = other[np.ix_(m, g)]
        for j in s.discrete:
            if j in s.ordinal:
                step = self.rng.choice([-1.0, 0.0, 1.0], size=n, p=[self.p_resample / 2, 1 - self.p_resample, self.p_resample / 2])
                rows[:, j] = np.clip(rows[:, j] + step, s.lo[j], s.hi[j])
            else:
                m = self.rng.random(n) < self.p_resample
                rows[m, j] = self.rng.choice(s.vocab[j], size=int(m.sum()))
        for j in s.binary:
            m = self.rng.random(n) < self.p_flip
            rows[m, j] = 1.0 - rows[m, j]
        return rows

    def _raw(self, n):
        s = self.schema
        rows = self.X[self.rng.integers(0, len(self.X), n)].copy()
        for j in s.continuous:
            amp = s.hi[j] - s.lo[j]
            rows[:, j] = np.clip(rows[:, j] + self.rng.normal(0, self.noise_scale * s.scale[j], n), s.lo[j] - self.margin * amp, s.hi[j] + self.margin * amp)
        return self._perturb_categorical(rows, None)


class NeighborhoodSampler(_Base):
    """D: linha de base e um de seus k vizinhos (distância mista: contínuas padronizadas + Hamming categórico); contínuas por interpolação convexa; categorias
    (grupos, discretas, binárias) copiadas de uma das duas linhas (conjuntamente por grupo)."""
    def __init__(self, X, *, transform=None, k=5, random_state=0, **kw):
        super().__init__(X, transform=transform, random_state=random_state, **kw)
        from sklearn.neighbors import NearestNeighbors
        s = self.schema
        Z = [(self.X[:, s.continuous] - self.X[:, s.continuous].mean(0)) / s.scale[s.continuous]] if s.continuous else []
        cat = [j for g in s.onehot_groups for j in g] + s.binary + s.discrete
        if cat:
            Z.append(self.X[:, cat] / np.maximum(np.abs(self.X[:, cat]).max(0), 1.0))
        self.k = min(k, len(self.X) - 1)
        self.nn = NearestNeighbors(n_neighbors=self.k + 1).fit(np.hstack(Z))
        self.neigh = self.nn.kneighbors(np.hstack(Z), return_distance=False)[:, 1:]

    def _raw(self, n):
        s = self.schema
        i = self.rng.integers(0, len(self.X), n)
        j = self.neigh[i, self.rng.integers(0, self.k, n)]
        a, b = self.X[i], self.X[j]
        rows = a.copy()
        t = self.rng.random((n, 1))
        if s.continuous:
            rows[:, s.continuous] = a[:, s.continuous] + t * (b[:, s.continuous] - a[:, s.continuous])
        take_b = self.rng.random(n) < 0.5
        for g in s.onehot_groups:
            rows[np.ix_(take_b, g)] = b[np.ix_(take_b, g)]
        for c in s.binary + s.discrete:
            m = self.rng.random(n) < 0.5
            rows[m, c] = b[m, c]
        return rows


class DomainConstrained:
    """E: aplica restrições de domínio EXPLÍCITAS (callables linha→bool) por rejeição, sobre qualquer amostrador. Só se justifica com restrições formalmente
    declaradas (as ontologias DEV não declaram facetas numéricas nem inteiros: ``minInclusive``/``xsd:integer`` ausentes)."""
    def __init__(self, sampler, predicates: Sequence[Callable[[np.ndarray], np.ndarray]]):
        self.sampler, self.predicates = sampler, list(predicates)
        self.n_features_in_ = sampler.n_features_in_

    class _C:
        def __init__(self, preds, outer):
            self.preds, self.outer = preds, outer

        def accepts(self, X):
            m = np.ones(len(X), dtype=bool)
            for p in self.preds:
                m &= np.asarray(p(X), dtype=bool)
            if self.outer is not None:
                m &= self.outer.accepts(X)
            return m

    def draw(self, n, constraints=None):
        return self.sampler.draw(n, self._C(self.predicates, constraints))

    @property
    def acceptance_rate(self):
        t = max(1, self.sampler.tries_)
        return self.sampler.accepted_ / t
