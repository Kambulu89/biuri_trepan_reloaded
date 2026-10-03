"""Geradores contrafactuais (todos *-inspired*, com divergências face ao método canónico documentadas).

Os geradores só PROPÕEM vectores; a validade final (modelo + constraints hard) é decidida em ``api``
re-passando cada CF ao modelo explicado. Sem ``time.sleep`` nem laços sem orçamento: todos os ciclos consultam ``Budget``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from counterfactuals.cfkit.metrics import DensityModel, GowerMetric
from counterfactuals.cfkit.rules import ConstraintSet
from counterfactuals.cfkit.schema import EPS

# ----------------------------------------------------------------------------- registry
METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "LORE": {
        "label": "LORE-inspired",
        "canonical": False,
        "reference": "Guidotti et al., LORE (arXiv:1805.10820)",
        "implements": ["vizinhança por algoritmo genético (50% mesma classe / 50% classe diferente)",
                       "substituto local em árvore (C4.5 nativo ou TREPAN)",
                       "regra factual + regras contrafactuais (premissas violadas mínimas)"],
        "divergences": ["o substituto local é C4.5 nativo/TREPAN, não o DecisionTreeClassifier (CART) do LORE original",
                        "se a regra contrafactual não vira a classe no modelo, aplica-se um refinamento por passo (validado no modelo) que preserva o caminho",
                        "operadores genéticos e tamanhos de população são simplificados e configuráveis",
                        "a regra contrafactual é aplicada ao vector e RE-VALIDADA no modelo (o LORE original não o faz)"],
    },
    "CLEAR": {
        "label": "CLEAR-inspired",
        "canonical": False,
        "reference": "White & Garcez, CLEAR (arXiv:1908.03020)",
        "implements": ["vizinhança com pontos de fronteira (bissecção até instâncias da classe alvo)",
                       "regressão linear local ponderada (WLR) sobre P(classe alvo)",
                       "b-contrafactuais por feature a partir dos coeficientes da WLR"],
        "divergences": ["não replica a geração exacta de perturbações nem os 'b-perturbation' datasets do CLEAR original",
                        "multiclasse tratado como alvo-vs-resto; refinamento por bissecção no modelo",
                        "restrições de domínio/schema aplicadas por projecção"],
    },
    "COGS": {
        "label": "CoGS-inspired",
        "canonical": False,
        "reference": "pesquisa genética multiobjectivo para contrafactuais (família CoGS)",
        "implements": ["população de vectores com operadores sensíveis ao tipo (inteiro/binário/one-hot/código)",
                       "fitness = perda de predição + distância + esparsidade + penalização de plausibilidade + penalização semântica soft",
                       "constraints hard por reparação/projecção; terminação por gerações/tempo/estagnação"],
        "divergences": ["não é uma réplica verificada do CoGS publicado (sem acesso ao código/benchmarks originais)",
                        "fitness escalarizada com pesos configuráveis (não Pareto)"],
    },
    "TREE": {
        "label": "Tree-path counterfactual",
        "canonical": False,
        "reference": "regras raiz→folha de árvores (C4.5/TREPAN/sklearn), incluindo testes m-of-n",
        "implements": ["satisfaz TODAS as condições do caminho até uma folha da classe alvo",
                       "m-of-n tratado como regra m-of-n (escolhe os literais mais baratos), nunca convertido num limiar"],
        "divergences": ["não é um método da literatura: é o gerador baseado em regras usado também pelo LORE-inspired"],
    },
}
SUPPORTED_METHODS = tuple(METHOD_REGISTRY)


@dataclass
class ObjectiveWeights:
    """Pesos do objectivo ontology-aware. Fixados antes de ver resultados (registados em provenance)."""

    prediction: float = 4.0
    distance: float = 1.5
    sparsity: float = 0.6
    plausibility: float = 0.5
    semantic: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


class Budget:
    """Orçamento de busca: iterações e/ou tempo. Regista a razão de terminação."""

    def __init__(self, max_iterations: int = 40, max_time: Optional[float] = None):
        self.max_iterations = int(max_iterations)
        self.max_time = None if max_time is None else float(max_time)
        self.iterations = 0
        self.start = time.perf_counter()
        self.reason = "completed"

    def tick(self, n: int = 1) -> None:
        self.iterations += n

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def exhausted(self) -> bool:
        if self.max_time is not None and self.elapsed() >= self.max_time:
            self.reason = "max_time"
            return True
        if self.iterations >= self.max_iterations:
            self.reason = "max_iterations"
            return True
        return False


class ModelAdapter:
    """Interface mínima e robusta sobre qualquer modelo com ``predict`` (e opcionalmente ``predict_proba``)."""

    def __init__(self, model: Any, name: str = "model"):
        self.model = model
        self.name = name
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = getattr(getattr(model, "model", None), "classes_", None)
        self.classes = None if classes is None else np.asarray(classes)
        self.calls = 0

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self.calls += len(X)
        return np.asarray(self.model.predict(X)).reshape(-1)

    def proba(self, X) -> Optional[np.ndarray]:
        if not hasattr(self.model, "predict_proba"):
            return None
        try:
            X = np.atleast_2d(np.asarray(X, dtype=float))
            return np.asarray(self.model.predict_proba(X), dtype=float)
        except Exception:
            return None

    def target_probability(self, X, target) -> np.ndarray:
        """P(alvo|x); sem predict_proba usa indicador 0/1 (documentado)."""
        P = self.proba(X)
        if P is not None and self.classes is not None and P.shape[1] == len(self.classes):
            idx = int(np.where(self.classes == target)[0][0]) if np.any(self.classes == target) else None
            if idx is not None:
                return P[:, idx]
        return (self.predict(X) == target).astype(float)


@dataclass
class SearchContext:
    model: ModelAdapter
    constraints: ConstraintSet
    metric: GowerMetric
    density: Optional[DensityModel]
    X_train: np.ndarray
    original: np.ndarray
    original_class: Any
    target: Any
    rng: np.random.Generator
    budget: Budget
    weights: ObjectiveWeights
    n_wanted: int = 3
    notes: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    blocked_examples: List[np.ndarray] = field(default_factory=list)   # mudam a classe mas violam constraints hard
    _train_pred: Optional[np.ndarray] = None
    _scale: Optional[np.ndarray] = None

    @property
    def schema(self):
        return self.constraints.schema

    def train_pred(self) -> np.ndarray:
        if self._train_pred is None:
            self._train_pred = self.model.predict(self.X_train)
        return self._train_pred

    def donors(self) -> np.ndarray:
        """Instâncias de TREINO que o modelo prevê como alvo (doadoras de valores plausíveis)."""
        return self.X_train[self.train_pred() == self.target]

    def scale(self) -> np.ndarray:
        if self._scale is None:
            s = np.nanstd(self.X_train, axis=0)
            self._scale = np.where(s > EPS, s, 1.0)
        return self._scale

    def repair(self, vec) -> np.ndarray:
        return self.constraints.repair(self.original, vec)

    def loss(self, vectors: np.ndarray) -> np.ndarray:
        """Objectivo (menor é melhor): perda de predição + distância + esparsidade + plausibilidade + penalização semântica soft."""
        w = self.weights
        V = np.atleast_2d(vectors)
        p_t = self.model.target_probability(V, self.target)
        out = np.zeros(len(V))
        n_units = max(1, len(self.schema.units))
        for i, v in enumerate(V):
            hard = self.constraints.hard_violations(self.original, v)
            if hard:
                out[i] = 1e6 + len(hard)
                continue
            _, soft = self.constraints.soft_violations(self.original, v)
            plaus = 0.0
            if self.density is not None and self.density.fitted and w.plausibility > 0:
                d = self.density.knn_distance(v)
                plaus = max(0.0, d - self.density.threshold) / max(self.density.threshold, EPS)
            out[i] = (w.prediction * (1.0 - p_t[i]) + w.distance * self.metric.distance(self.original, v, weighted=True)
                      + w.sparsity * self.metric.l1_changes(self.original, v) / n_units + w.plausibility * plaus + w.semantic * soft)
        return out


# ----------------------------------------------------------------------------- helpers
def _unit_values_from_donor(ctx: SearchContext, vec: np.ndarray, donor: np.ndarray, units: Sequence[int]) -> np.ndarray:
    out = vec.copy()
    for u in units:
        idx = ctx.schema.units[u].indices
        out[idx] = donor[idx]
    return out


def _mutate_unit(ctx: SearchContext, vec: np.ndarray, u: int, rng: np.random.Generator, donors: Optional[np.ndarray], sigma: float = 0.25) -> np.ndarray:
    unit = ctx.schema.units[u]
    out = vec.copy()
    spec = ctx.schema.specs[unit.indices[0]]
    if donors is not None and len(donors) and rng.random() < 0.35:
        donor = donors[rng.integers(len(donors))]
        out[unit.indices] = donor[unit.indices]
        return out
    if unit.kind == "onehot":
        block = np.zeros(len(unit.indices))
        block[int(rng.integers(len(unit.indices)))] = 1.0
        out[unit.indices] = block
    elif unit.kind == "binary":
        out[unit.indices[0]] = 1.0 - out[unit.indices[0]]
    elif unit.kind == "code":
        allowed = spec.allowed_values or (out[unit.indices[0]],)
        out[unit.indices[0]] = float(allowed[int(rng.integers(len(allowed)))])
    elif unit.kind == "integer":
        step = int(rng.integers(1, 4)) * (1 if rng.random() < 0.5 else -1)
        out[unit.indices[0]] = out[unit.indices[0]] + step
    else:
        out[unit.indices[0]] = out[unit.indices[0]] + rng.normal(0, sigma) * ctx.scale()[unit.indices[0]]
    return out


def sparsify(ctx: SearchContext, vec: np.ndarray) -> np.ndarray:
    """Reverte alterações desnecessárias (guloso): só mantém a reversão se o CF continuar válido no modelo e nas hard constraints."""
    cur = np.asarray(vec, dtype=float).copy()
    changed = ctx.schema.changed_units(ctx.original, cur)
    order = sorted(changed, key=lambda u: -ctx.metric.distance(ctx.original, _revert(ctx, cur, u), weighted=True))
    for u in order:
        if ctx.budget.exhausted():
            break
        trial = _revert(ctx, cur, u)
        if ctx.model.predict(trial)[0] == ctx.target and not ctx.constraints.hard_violations(ctx.original, trial):
            cur = trial
    return cur


def _revert(ctx: SearchContext, vec: np.ndarray, unit_idx: int) -> np.ndarray:
    out = vec.copy()
    idx = ctx.schema.units[unit_idx].indices
    out[idx] = ctx.original[idx]
    return out


def _valid_on_model(ctx: SearchContext, V: np.ndarray) -> np.ndarray:
    return ctx.model.predict(V) == ctx.target


# ----------------------------------------------------------------------------- CoGS-inspired
def cogs_inspired(ctx: SearchContext, *, population: int = 80, elite: int = 16, patience: int = 6) -> List[Dict[str, Any]]:
    rng = ctx.rng
    units = ctx.schema.searchable_units(ctx.constraints.nonactionable_policy)
    if not units:
        return []
    donors = ctx.donors()
    pop: List[np.ndarray] = []
    for i in range(population):
        v = ctx.original.copy()
        for u in rng.choice(units, size=int(rng.integers(1, min(4, len(units)) + 1)), replace=False):
            v = _mutate_unit(ctx, v, int(u), rng, donors)
        pop.append(ctx.repair(v))
    archive: Dict[Tuple, np.ndarray] = {}
    best_seen, stale = np.inf, 0
    while not ctx.budget.exhausted():
        ctx.budget.tick()
        P = np.asarray(pop)
        fit = ctx.loss(P)
        model_valid = _valid_on_model(ctx, P)
        valid = model_valid & (fit < 1e5)
        blocked = model_valid & (fit >= 1e5)          # válidos no modelo mas barrados por constraints hard
        if blocked.any():
            ctx.stats["hard_blocked"] = ctx.stats.get("hard_blocked", 0) + int(blocked.sum())
            for v in P[blocked][: max(0, 8 - len(ctx.blocked_examples))]:
                ctx.blocked_examples.append(v.copy())
        for v, f in zip(P[valid], fit[valid]):
            key = tuple(np.round(v, 9))
            if key not in archive or f < archive[key][1]:
                archive[key] = (v, float(f))
        order = np.argsort(fit)
        top = float(fit[order[0]])
        stale = 0 if top < best_seen - 1e-9 else stale + 1
        best_seen = min(best_seen, top)
        if len(archive) >= ctx.n_wanted * 6 and stale >= patience:
            ctx.budget.reason = "found_enough"
            break
        elites = [P[i] for i in order[:elite]]
        children = [e.copy() for e in elites]
        while len(children) < population:
            i, j = (int(t) for t in rng.integers(0, len(P), size=2))
            a = P[i if fit[i] <= fit[j] else j]      # torneio binário
            b = elites[int(rng.integers(len(elites)))]
            child = a.copy()
            for u in range(len(ctx.schema.units)):
                if rng.random() < 0.5:
                    child[ctx.schema.units[u].indices] = b[ctx.schema.units[u].indices]
            for u in rng.choice(units, size=int(rng.integers(1, min(3, len(units)) + 1)), replace=False):
                child = _mutate_unit(ctx, child, int(u), rng, donors)
            children.append(ctx.repair(child))
        pop = children
    ctx.stats["archive_size"] = len(archive)
    ranked = sorted(archive.values(), key=lambda item: item[1])           # melhores (menor objectivo) primeiro
    return [{"vector": v, "method": "COGS", "metadata": {"objective": f}} for v, f in ranked]


# ----------------------------------------------------------------------------- tree rules (TREE / LORE)
def _set_literal(ctx: SearchContext, vec: np.ndarray, idx: int, op: str, thr: float, truth: bool) -> None:
    spec = ctx.schema.specs[idx]
    want_le = (op == "<=") == truth       # True => precisa de valor <= thr
    if spec.kind in ("integer", "binary", "code"):
        vec[idx] = float(np.floor(thr + 1e-12)) if want_le else float(np.floor(thr + 1e-12) + 1.0)
    else:
        margin = max(1e-6 * max(abs(thr), 1.0), 1e-9)
        vec[idx] = (thr - margin) if want_le else (thr + margin)
        # nextafter evita ficar exactamente no limiar após arredondamentos
        vec[idx] = np.nextafter(vec[idx], -np.inf if want_le else np.inf)


def _literal_cost(ctx: SearchContext, vec: np.ndarray, idx: int, op: str, thr: float, truth: bool) -> float:
    trial = vec.copy()
    _set_literal(ctx, trial, idx, op, thr, truth)
    span = max(ctx.schema.specs[idx].high - ctx.schema.specs[idx].low, 1.0) if np.isfinite(ctx.schema.specs[idx].high - ctx.schema.specs[idx].low) else 1.0
    return abs(trial[idx] - vec[idx]) / span * ctx.schema.specs[idx].cost


def _literal_ok(vec: np.ndarray, idx: int, op: str, thr: float) -> bool:
    return bool(vec[idx] > thr) if op == ">" else bool(vec[idx] <= thr)


def satisfy_condition(ctx: SearchContext, vec: np.ndarray, cond) -> int:
    """Aplica UMA condição de regra (simples ou m-of-n) a ``vec``. Devolve o nº de literais alterados.

    m-of-n é tratado como m-of-n: só são alterados os literais mais baratos necessários (nunca um único limiar)."""
    changes = 0
    if cond.kind != "m_of_n":
        if not _literal_ok(vec, int(cond.feature_index), cond.operator, float(cond.threshold)):
            _set_literal(ctx, vec, int(cond.feature_index), cond.operator, float(cond.threshold), True)
            changes += 1
        return changes
    lits = [(int(f), op, float(t)) for f, _n, op, t in cond.literals]
    truths = [_literal_ok(vec, f, op, t) for f, op, t in lits]
    votes = sum(truths)
    if not cond.negated:
        need = max(0, int(cond.m) - votes)
        pool = [i for i, ok in enumerate(truths) if not ok]
        pool.sort(key=lambda i: _literal_cost(ctx, vec, lits[i][0], lits[i][1], lits[i][2], True))
        for i in pool[:need]:
            _set_literal(ctx, vec, lits[i][0], lits[i][1], lits[i][2], True)
            changes += 1
    else:
        need = max(0, votes - (int(cond.m) - 1))
        pool = [i for i, ok in enumerate(truths) if ok]
        pool.sort(key=lambda i: _literal_cost(ctx, vec, lits[i][0], lits[i][1], lits[i][2], False))
        for i in pool[:need]:
            _set_literal(ctx, vec, lits[i][0], lits[i][1], lits[i][2], False)
            changes += 1
    return changes


def rules_to_candidates(ctx: SearchContext, rules: Sequence[Any], method: str, tree_family: str = "", refine: bool = True) -> List[Dict[str, Any]]:
    """Gera um CF por regra da classe alvo, respeitando TODAS as condições do caminho.

    Um CF só é aceite se, depois da reparação do schema (inteiros, direcção, one-hot...), continua a cumprir a regra."""
    out = []
    for rule in rules:
        if rule.predicted_class != ctx.target and str(rule.predicted_class) != str(ctx.target):
            continue
        vec = ctx.original.copy()
        n_changed = 0
        for cond in rule.conditions:
            n_changed += satisfy_condition(ctx, vec, cond)
        repaired = ctx.repair(vec)
        path_ok = all(c.matches(repaired) for c in rule.conditions)
        refine_factor = 1.0
        if refine and not (path_ok and ctx.model.predict(repaired)[0] == ctx.target):
            # O limiar do substituto não coincide exactamente com a fronteira do modelo explicado: afasta-se gradualmente
            # do ponto original ao longo da regra (mesmas direcções), mantendo o caminho e validando no MODELO.
            delta = vec - ctx.original
            for factor in (1.1, 1.25, 1.5, 2.0, 3.0, 4.0):
                if ctx.budget.exhausted():
                    break
                trial = ctx.repair(ctx.original + factor * delta)
                if all(c.matches(trial) for c in rule.conditions) and ctx.model.predict(trial)[0] == ctx.target:
                    repaired, path_ok, refine_factor = trial, True, factor
                    break
        out.append({"vector": repaired, "method": method,
                    "metadata": {"rule": rule.text(), "rule_id": rule.rule_id, "support": float(rule.support), "confidence": float(rule.confidence),
                                 "premise_changes": int(n_changed), "path_respected": bool(path_ok), "tree_family": tree_family, "refine_factor": refine_factor,
                                 "has_m_of_n": any(c.kind == "m_of_n" for c in rule.conditions)}})
    out.sort(key=lambda item: (not item["metadata"]["path_respected"], item["metadata"]["premise_changes"]))
    return out


def tree_path(ctx: SearchContext, tree_model: Any, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    from counterfactuals.global_rules import extract_global_rules
    rules = extract_global_rules(tree_model, feature_names)
    ctx.budget.tick(len(rules))
    cands = rules_to_candidates(ctx, rules, "TREE", tree_family=type(tree_model).__name__)
    ctx.stats["rules_for_target"] = len(cands)
    return cands


# ----------------------------------------------------------------------------- LORE-inspired
def _genetic_neighborhood(ctx: SearchContext, size: int, generations: int = 8) -> np.ndarray:
    """Vizinhança LORE: duas populações (mesma classe / classe diferente) evoluídas por GA, mais x."""
    rng = ctx.rng
    units = ctx.schema.searchable_units(ctx.constraints.nonactionable_policy)
    base_cls = ctx.original_class
    marginals = ctx.X_train
    half = max(10, size // 2)
    pops = {}
    for mode in ("eq", "neq"):
        pop = []
        for _ in range(half):
            v = ctx.original.copy()
            for u in rng.choice(units, size=int(rng.integers(1, min(len(units), 5) + 1)), replace=False):
                row = marginals[int(rng.integers(len(marginals)))]
                v[ctx.schema.units[int(u)].indices] = row[ctx.schema.units[int(u)].indices]
            pop.append(ctx.repair(v))
        pops[mode] = np.asarray(pop)
    for _ in range(generations):
        if ctx.budget.exhausted():
            break
        ctx.budget.tick()
        for mode in ("eq", "neq"):
            P = pops[mode]
            pred = ctx.model.predict(P)
            same = (pred == base_cls).astype(float)
            dist = np.array([ctx.metric.distance(ctx.original, v) for v in P])
            fit = (same if mode == "eq" else 1.0 - same) + (1.0 - dist)
            order = np.argsort(-fit)
            parents = P[order[: max(4, half // 2)]]
            children = [p.copy() for p in parents]
            while len(children) < half:
                a, b = parents[int(rng.integers(len(parents)))], parents[int(rng.integers(len(parents)))]
                child = a.copy()
                for u in range(len(ctx.schema.units)):
                    if rng.random() < 0.5:
                        child[ctx.schema.units[u].indices] = b[ctx.schema.units[u].indices]
                if rng.random() < 0.6:
                    u = int(rng.choice(units))
                    row = marginals[int(rng.integers(len(marginals)))]
                    child[ctx.schema.units[u].indices] = row[ctx.schema.units[u].indices]
                children.append(ctx.repair(child))
            pops[mode] = np.asarray(children)
    Z = np.vstack([ctx.original.reshape(1, -1), pops["eq"], pops["neq"]])
    return Z


def lore_inspired(ctx: SearchContext, feature_names: Sequence[str], *, neighborhood: int = 600, surrogate: str = "c45") -> List[Dict[str, Any]]:
    """LORE-inspired: vizinhança genética + árvore local + regras contrafactuais (premissas violadas mínimas)."""
    from counterfactuals.global_rules import extract_global_rules
    Z = _genetic_neighborhood(ctx, neighborhood)
    labels = ctx.model.predict(Z)
    ctx.stats["neighborhood"] = int(len(Z))
    classes, counts = np.unique(labels, return_counts=True)
    ctx.stats["neighborhood_class_counts"] = {str(c): int(n) for c, n in zip(classes, counts)}
    if len(classes) < 2 or ctx.target not in classes:
        ctx.notes.append("a vizinhança genética não contém a classe alvo: sem regras contrafactuais locais")
        return []
    if surrogate == "trepan":
        from core.trepan_original import TrepanOriginalClassifier
        tree = TrepanOriginalClassifier(max_nodes=31, max_depth=5, min_samples_leaf=3, min_sample=max(len(Z), 300), max_n=3,
                                        max_queries=max(600, len(Z) * 2), random_state=int(ctx.rng.integers(1 << 30))).fit(
            Z, oracle=ctx.model.model, feature_names=list(feature_names))
    else:
        from core.c45_j48_tree import C45Classifier
        tree = C45Classifier(min_samples_leaf=2, min_samples_split=4, max_depth=6, random_state=int(ctx.rng.integers(1 << 30))).fit(Z, labels)
    fidelity = float(np.mean(np.asarray(tree.predict(Z)) == labels))
    ctx.stats["local_fidelity"] = fidelity
    ctx.stats["surrogate"] = "TREPAN (histórico)" if surrogate == "trepan" else "C4.5 nativo"
    rules = extract_global_rules(tree, feature_names)
    factual = [r for r in rules if all(c.matches(ctx.original) for c in r.conditions)]
    ctx.stats["factual_rule"] = factual[0].text() if factual else None
    cands = rules_to_candidates(ctx, rules, "LORE", tree_family=ctx.stats["surrogate"])
    for c in cands:
        c["metadata"]["local_fidelity"] = fidelity
    ctx.budget.tick(len(rules))
    return cands


# ----------------------------------------------------------------------------- CLEAR-inspired
def _bisect_boundary(ctx: SearchContext, a: np.ndarray, b: np.ndarray, steps: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """a: classe original, b: classe alvo. Devolve (último ponto não-alvo, primeiro ponto alvo) junto da fronteira."""
    lo, hi = a.copy(), b.copy()
    for _ in range(steps):
        mid = ctx.repair(0.5 * (lo + hi))
        if ctx.model.predict(mid)[0] == ctx.target:
            hi = mid
        else:
            lo = mid
    return lo, hi


def clear_inspired(ctx: SearchContext, *, n_perturb: int = 400, n_boundary: int = 8, margin: float = 0.02) -> List[Dict[str, Any]]:
    rng = ctx.rng
    schema = ctx.schema
    units = schema.searchable_units(ctx.constraints.nonactionable_policy)
    if not units:
        return []
    # (1) vizinhança: perturbações gaussianas/discretas + pontos de fronteira
    pts = []
    for _ in range(n_perturb):
        v = ctx.original.copy()
        for u in rng.choice(units, size=int(rng.integers(1, min(len(units), 4) + 1)), replace=False):
            v = _mutate_unit(ctx, v, int(u), rng, None, sigma=0.4)
        pts.append(ctx.repair(v))
    donors = ctx.donors()
    boundary_pts = []
    if len(donors):
        d = ctx.metric.pairwise_to(ctx.original, donors)
        for z in donors[np.argsort(d)[:n_boundary]]:
            if ctx.budget.exhausted():
                break
            ctx.budget.tick()
            zr = ctx.repair(z)
            if ctx.model.predict(zr)[0] != ctx.target:
                continue
            lo, hi = _bisect_boundary(ctx, ctx.original, zr)
            boundary_pts += [lo, hi]
            for t in (0.25, 0.5, 0.75, 1.0):
                boundary_pts.append(ctx.repair(ctx.original + t * (hi - ctx.original)))
    Z = np.vstack([ctx.original.reshape(1, -1)] + [np.asarray(pts)] + ([np.asarray(boundary_pts)] if boundary_pts else []))
    y = ctx.model.target_probability(Z, ctx.target)
    ctx.stats["neighborhood"] = int(len(Z))
    if np.ptp(y) <= 1e-9:
        ctx.notes.append("a vizinhança CLEAR tem P(alvo) constante: regressão local degenerada")
        return []
    # (2) regressão linear local ponderada (ridge mínimo: robusto a singularidade)
    dist = ctx.metric.pairwise_to(ctx.original, Z)
    h = max(np.median(dist), 1e-3)
    w = np.exp(-(dist ** 2) / (2 * h ** 2))
    scale = ctx.scale()
    A = np.hstack([np.ones((len(Z), 1)), (Z - ctx.original) / scale])
    lam = 1e-3
    W = w[:, None]
    gram = A.T @ (A * W) + lam * np.eye(A.shape[1])
    try:
        beta = np.linalg.solve(gram, A.T @ (w * y))
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(gram, A.T @ (w * y), rcond=None)[0]
    if not np.all(np.isfinite(beta)):
        ctx.notes.append("regressão local singular/não finita")
        return []
    pred = A @ beta
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else 0.0
    ctx.stats["local_r2"] = r2
    ctx.stats["wlr_intercept"] = float(beta[0])
    p0 = float(beta[0])           # ŷ(x): valor da regressão no ponto
    coef = beta[1:]
    tau = 0.5 + margin
    # (3) b-contrafactuais por UNIDADE humana: o ΔP necessário é distribuído pelas unidades com maior impacto/custo.
    out: List[Dict[str, Any]] = []
    gap = tau - p0
    scale_arr = scale
    gains: Dict[int, float] = {}
    onehot_choice: Dict[int, int] = {}
    for u in units:
        unit = schema.units[u]
        cols_u = unit.indices
        cost = float(np.mean([schema.specs[i].cost for i in cols_u]))
        if unit.kind == "onehot":
            active = int(np.argmax(ctx.original[cols_u]))
            best = int(np.argmax(coef[cols_u]))
            g = float(coef[cols_u][best] - coef[cols_u][active])
            if g > 0 and best != active:
                gains[u] = g / cost
                onehot_choice[u] = best
        else:
            j = cols_u[0]
            if abs(coef[j]) > 1e-12:
                gains[u] = abs(coef[j]) * scale_arr[j] / cost
    ranked = sorted(gains, key=lambda u: -gains[u])
    for k in range(1, min(4, len(ranked)) + 1):
        if ctx.budget.exhausted():
            break
        ctx.budget.tick()
        chosen = ranked[:k]
        numeric = [u for u in chosen if u not in onehot_choice]
        onehot_gain = sum(float(coef[schema.units[u].indices][onehot_choice[u]] - coef[schema.units[u].indices][int(np.argmax(ctx.original[schema.units[u].indices]))])
                          for u in chosen if u in onehot_choice)
        remaining = gap - onehot_gain
        for factor in (1.0, 1.25, 1.6, 2.2, 3.0, 4.5):
            v = ctx.original.copy()
            for u in chosen:
                idx = schema.units[u].indices
                if u in onehot_choice:
                    block = np.zeros(len(idx))
                    block[onehot_choice[u]] = 1.0
                    v[idx] = block
            if numeric:
                share = remaining / len(numeric) if remaining > 0 else gap / len(numeric)
                for u in numeric:
                    j = schema.units[u].indices[0]
                    v[j] = ctx.original[j] + factor * share / coef[j] * scale_arr[j]
            v = ctx.repair(v)
            if ctx.model.predict(v)[0] == ctx.target:
                lo, hi = _bisect_boundary(ctx, ctx.original, v, steps=10)   # (4) refinamento junto à fronteira, validado no modelo
                meta = {"local_r2": r2, "n_units_regression": k, "step_factor": factor, "wlr_p0": p0}
                out.append({"vector": hi, "method": "CLEAR", "metadata": dict(meta)})
                if not np.allclose(hi, v):
                    out.append({"vector": v, "method": "CLEAR", "metadata": dict(meta)})
                break
    ctx.stats["b_counterfactuals"] = len(out)
    return out


GENERATORS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {"COGS": cogs_inspired, "CLEAR": clear_inspired, "LORE": lore_inspired, "TREE": tree_path}
