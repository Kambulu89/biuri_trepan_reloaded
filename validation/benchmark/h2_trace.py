"""H2 (Doc. 55): adaptador EXPERIMENTAL, só ``validation/``, que torna observável a cadeia  Axioma → Inferência → Literal → Candidato → Decisão  para o mecanismo T1/T2, sem tocar no TREPAN Original,
no C4.5, no MLP nem em ``core/``. ``DerivedLiteralTrepan`` é uma subclasse do Original que:
  1. recebe X aumentado com colunas binárias derivadas (OR das colunas da extensão de uma classe definida) — representação enriquecida (mecanismo M-R);
  2. mantém as consultas ao oráculo COERENTES: as colunas derivadas das consultas recalculam-se a partir das colunas base (``_draw_membership_queries``) e o oráculo só vê as colunas base
     (``AugmentedOracle``), pelo que o oráculo é o MESMO que o do Original;
  3. opcionalmente (mecanismo M-P, ``prune_redundant``) descarta candidatos m-of-n LOGICAMENTE redundantes ou insatisfazíveis pelas relações inferidas (subsunção entre extensões; disjunção): decisão
     independente dos dados e dos rótulos;
  4. regista ``semantic_trace_``: por literal, n.º de avaliações de candidatos que o contêm, n.º de podas por razão e nós onde o teste escolhido o contém.
Sem literais derivados e sem poda, ajusta EXATAMENTE como o Original (testado: mesmas regras, mesmo n.º de consultas ao oráculo)."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.trepan_original import MofNTest, TrepanOriginalClassifier
from validation.benchmark.h2_axiom_mechanisms import DerivedLiteral


class AugmentedOracle:
    """Oráculo do Original aplicado só às colunas base (as derivadas nunca chegam ao MLP)."""

    def __init__(self, base, n_base: int):
        self.base, self.n_base = base, int(n_base)

    def predict(self, X):
        return self.base.predict(np.asarray(X)[:, : self.n_base])

    def predict_proba(self, X):
        return self.base.predict_proba(np.asarray(X)[:, : self.n_base])


def augment(X_base: np.ndarray, base_names: Sequence[str], derived: Sequence[DerivedLiteral]) -> Tuple[np.ndarray, List[str]]:
    """Colunas derivadas = OR das colunas base da extensão. Categoria desconhecida (todas as colunas do nível a 0) ⇒ derivada 0 (nunca é atribuída a um nível)."""
    pos = {n: i for i, n in enumerate(base_names)}
    cols = [np.max(np.asarray(X_base)[:, [pos[a] for a in l.attributes]], axis=1) for l in derived]
    if not cols:
        return np.asarray(X_base, dtype=float).copy(), list(base_names)
    return np.column_stack([np.asarray(X_base, dtype=float)] + cols), list(base_names) + [l.name for l in derived]


class DerivedLiteralTrepan(TrepanOriginalClassifier):
    def __init__(self, *, derived: Sequence[DerivedLiteral] = (), base_names: Optional[Sequence[str]] = None, prune_redundant: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.derived = list(derived)
        self.base_names = list(base_names) if base_names is not None else None
        self.prune_redundant = bool(prune_redundant)

    # ------------------------------------------------------------------ preparação
    def _setup(self):
        self._n_base = len(self.base_names)
        pos = {n: i for i, n in enumerate(self.base_names)}
        self._ext: Dict[int, frozenset] = {i: frozenset([i]) for i in range(self._n_base)}
        self._dcols: Dict[int, Tuple[int, ...]] = {}
        for k, l in enumerate(self.derived):
            idx = tuple(sorted(pos[a] for a in l.attributes))
            self._ext[self._n_base + k] = frozenset(idx)
            self._dcols[self._n_base + k] = idx
        self._family = {i: n.split("__")[0] for i, n in enumerate(self.base_names)}
        self._stats = {self._n_base + k: Counter() for k in range(len(self.derived))}
        self._pruned = Counter()

    def fit(self, X, y=None, *, oracle=None, **kw):
        if self.base_names is None:
            raise ValueError("base_names obrigatório")
        self._setup()
        if oracle is not None and not isinstance(oracle, AugmentedOracle) and self.derived:
            oracle = AugmentedOracle(oracle, self._n_base)
        out = super().fit(X, y, oracle=oracle, **kw)
        self.semantic_trace_ = self._build_trace()
        return out

    # ------------------------------------------------------------------ consultas coerentes
    def _draw_membership_queries(self, model, n, constraints, node):
        qx = np.array(super()._draw_membership_queries(model, n, constraints, node), dtype=float)
        for col, idx in self._dcols.items():
            qx[:, col] = np.max(qx[:, list(idx)], axis=1)                  # derivada = função determinística das colunas base da consulta
        return qx

    # ------------------------------------------------------------------ candidatos, poda e rastreio
    def _redundancy(self, test: MofNTest) -> Optional[str]:
        pos = [l for l in test.literals if l.greater]
        n = len(test.literals)
        for i in range(len(pos)):
            for j in range(len(pos)):
                if i == j:
                    continue
                a, b = self._ext.get(pos[i].feature), self._ext.get(pos[j].feature)
                if a is None or b is None:
                    continue
                if a < b and test.m == 1 and len(pos) == n:
                    return "SUBSUMPTION_COLLAPSE_OR"
                if a < b and test.m == n and len(pos) == n:
                    return "SUBSUMPTION_COLLAPSE_AND"
                if i < j and not (a & b) and test.m == n and n >= 2 and len(pos) == n and self._family.get(next(iter(a)), "x") == self._family.get(next(iter(b)), "y"):
                    return "DISJOINT_UNSATISFIABLE"            # redundante com a codificação one-hot (não é conhecimento externo)
        return None

    def _split_selection_score(self, y, mask, test=None):
        if test is not None and getattr(self, "_stats", None) is not None:
            feats = {int(l.feature) for l in test.literals}
            hit = [c for c in self._stats if c in feats]
            reason = self._redundancy(test) if self.prune_redundant and (hit or len(test.literals) > 1) else None
            for c in hit:
                self._stats[c]["candidate_evaluations"] += 1
                if reason:
                    self._stats[c]["pruned_" + reason] += 1
            if reason:
                self._pruned[reason] += 1
                return -1e9
        return super()._split_selection_score(y, mask, test)

    def _build_trace(self) -> Dict:
        chosen: Dict[int, List[int]] = defaultdict(list)
        for node in self.iter_nodes():
            if not node.is_leaf and node.test is not None:
                for l in node.test.literals:
                    if l.feature in self._stats:
                        chosen[l.feature].append(int(node.node_id))
        lits = []
        for k, l in enumerate(self.derived):
            col = self._n_base + k
            lits.append({"literal": l.name, "concept": l.concept, "attributes": list(l.attributes), "needs_reasoner": l.needs_reasoner, "fact_ids": list(l.fact_ids),
                         "candidate_evaluations": int(self._stats[col]["candidate_evaluations"]), "pruned": {r: int(v) for r, v in self._stats[col].items() if r.startswith("pruned_")},
                         "chosen_in_nodes": sorted(set(chosen.get(col, [])))})
        return {"n_derived": len(self.derived), "prune_redundant": self.prune_redundant, "pruned_total": dict(self._pruned), "literals": lits,
                "n_internal_nodes": sum(1 for n in self.iter_nodes() if not n.is_leaf)}


def test_signature(test: Optional[MofNTest]):
    if test is None:
        return None
    return (int(test.m), tuple(sorted((int(l.feature), round(float(l.threshold), 9), bool(l.greater)) for l in test.literals)))


def node_choices(model: TrepanOriginalClassifier) -> Dict[int, Tuple]:
    return {int(n.node_id): test_signature(n.test) for n in model.iter_nodes() if not n.is_leaf and n.test is not None}


def decision_influence(X_base: np.ndarray, base_names: Sequence[str], derived: Sequence[DerivedLiteral], make_oracle, fit_kwargs: dict, *, prune_redundant: bool = False) -> Dict:
    """Atribuição CONTRAFACTUAL por literal: ajusta com todos os literais derivados e, para cada literal, sem ele (mesma semente, mesmo oráculo de forma, mesmas instâncias). Um nó diz-se
    *influenciado* pelo literal L se o teste escolhido nesse nó contém L no modelo completo (``chosen``) e/ou difere do do modelo ablacionado (``changed``). A coluna de L é removida da ablação,
    por isso as assinaturas comparam-se por (m, atributos BASE ou nome do literal), não por índice. Devolve contagens; NÃO mede desempenho."""
    def fit(lits):
        Xa, na = augment(X_base, base_names, lits)
        m = DerivedLiteralTrepan(derived=lits, base_names=base_names, prune_redundant=prune_redundant, **fit_kwargs).fit(Xa, oracle=make_oracle(), feature_names=na)
        return m, na

    def named(model, names):
        out = {}
        for n in model.iter_nodes():
            if not n.is_leaf and n.test is not None:
                out[int(n.node_id)] = (int(n.test.m), tuple(sorted((names[l.feature], round(float(l.threshold), 9), bool(l.greater)) for l in n.test.literals)))
        return out
    full, nf = fit(list(derived))
    cf = named(full, nf)
    res = {}
    for l in derived:
        abl, na = fit([x for x in derived if x.concept != l.concept])
        ca = named(abl, na)
        chosen = sorted(k for k, sig in cf.items() if any(nm == l.name for nm, _, _ in sig[1]))
        changed = sorted(k for k in set(cf) | set(ca) if cf.get(k) != ca.get(k))
        res[l.concept] = {"literal": l.name, "chosen_in_nodes": chosen, "nodes_changed_by_ablation": changed, "n_chosen": len(chosen), "n_changed": len(changed)}
    return res
