"""H2 (Doc. 59/61, revisão pós-Docs. 58-61): análise TÉCNICA da elegibilidade de literais derivados e controlo experimental do *pool* de candidatos. Só ``validation/``; ``core/`` e
``max_features_per_node`` do núcleo intocados (o parâmetro do construtor é usado apenas como entrada do protótipo). NÃO calcula accuracy nem fidelidade a rótulos reais nem lê o teste: as únicas
quantidades usadas são estruturais (contagens de candidatos/consultas/nós) e o ganho de informação sobre as etiquetas do ORÁCULO no treino, que é o próprio critério interno do TREPAN.

Código agnóstico: nada aqui conhece datasets, nomes de atributos ou códigos; os conceitos chegam como ``DerivedLiteral`` (compilados de qualquer OWL pelo registo B3.1).

Etapas por literal derivado (cada uma pode rejeitá-lo, por esta ordem): (1) *pool de features* do nó = top-``max_features_per_node`` pela prioridade (Original: variância — sem rótulos);
(2) geração de candidatos simples (limiares com folhas ≥ ``min_samples_leaf``); (3) *pool de literais* do m-of-n = top ``max(8, 3·mf)`` por ganho de informação; (4) avaliação como candidato (simples ou em m-of-n);
(5) escolha. Estados da escada: AXIOM_AVAILABLE → AXIOM_VALIDATED → FACT_COMPILED → LITERAL_ELIGIBLE → LITERAL_EVALUATED → SPLIT_DECISION_CHANGED → OBSERVED_BENEFIT (este último NÃO é avaliado aqui)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.trepan_original import TrepanOriginalClassifier
from validation.benchmark.h2_axiom_mechanisms import DerivedLiteral
from validation.benchmark.h2_trace import DerivedLiteralTrepan, augment

LADDER = ("AXIOM_AVAILABLE", "AXIOM_VALIDATED", "FACT_COMPILED", "LITERAL_ELIGIBLE", "LITERAL_EVALUATED", "SPLIT_DECISION_CHANGED", "OBSERVED_BENEFIT")
REJECTION = ("NOT_IN_FEATURE_POOL", "NO_VALID_SPLIT_CANDIDATE", "BELOW_LITERAL_POOL_CUTOFF", "IN_LITERAL_POOL")


class PoolTrepan(DerivedLiteralTrepan):
    """Original (+ colunas derivadas) com controlo explícito do *pool* de features: ``forced`` (colunas que ocupam vagas extra, sem rótulos) e ``masked`` (colunas excluídas SEM libertar a vaga).
    Sem ``forced``/``masked`` o *pool* é exatamente o do Original (variância). Regista, por pesquisa de split, o estado de cada coluna derivada em cada etapa."""

    def __init__(self, *, forced: Sequence[int] = (), masked: Sequence[int] = (), **kwargs):
        super().__init__(**kwargs)
        self.forced = tuple(int(c) for c in forced)
        self.masked = tuple(int(c) for c in masked)

    def fit(self, X, y=None, *, oracle=None, **kw):
        self._pool_log: List[Dict[str, Any]] = []
        self._search_log: List[Dict[str, Any]] = []
        out = super().fit(X, y, oracle=oracle, **kw)
        self.semantic_trace_["pool"] = self.pool_trace()
        return out

    def _feature_priority_scores(self, X):
        s = np.array(np.nanvar(X, axis=0), dtype=float)
        for c in self.forced:
            s[c] = 1e18 - c                                    # vagas reservadas, determinísticas e sem rótulos
        for c in self.masked:
            s[c] = -np.inf
        return s

    def _candidate_literals(self, X, y):
        mf0 = self.max_features_per_node
        eff = mf0 - len(self.masked)                           # a coluna mascarada NÃO liberta a vaga (o resto do pool fica idêntico)
        self.max_features_per_node = max(1, eff)
        try:
            ranked = super()._candidate_literals(X, y)
        finally:
            self.max_features_per_node = mf0
        pri = self._feature_priority_scores(X)
        order = np.argsort(-pri, kind="stable")[: max(1, eff)]
        pool_feats = {int(j) for j in order}
        cutoff = max(8, mf0 * 3)
        lp = ranked[:cutoff]
        ranked_feats = {int(l.feature) for _, l in ranked}
        lp_set = {(int(l.feature), bool(l.greater)) for _, l in lp}
        rank_of = {}
        for r, (_, l) in enumerate(ranked):
            rank_of.setdefault(int(l.feature), r)
        var_rank = {int(j): int(r) for r, j in enumerate(np.argsort(-np.nanvar(X, axis=0), kind="stable"))}
        self._pool_log.append({"pool_feats": pool_feats, "ranked_feats": ranked_feats, "lp": lp_set, "lp_feats": {f for f, _ in lp_set}, "rank_of": rank_of, "var_rank": var_rank,
                               "n_ranked": len(ranked), "cutoff": cutoff})
        return ranked

    def _best_mofn(self, X, y):
        res = super()._best_mofn(X, y)
        self._search_log.append(None if res is None else {"score": float(res[1]), "sig": self._named_sig(res[0])})
        return res

    def _named_sig(self, test) -> Tuple:
        names = getattr(self, "_feature_names_for_sig", None)
        nm = (lambda j: names[j]) if names else (lambda j: int(j))
        return (int(test.m), tuple(sorted((nm(l.feature), round(float(l.threshold), 9), bool(l.greater)) for l in test.literals)))

    # ------------------------------------------------------------------ rastreio
    def _classify(self, col: int, call: Dict[str, Any]) -> str:
        if col not in call["pool_feats"]:
            return "NOT_IN_FEATURE_POOL"
        if col not in call["ranked_feats"]:
            return "NO_VALID_SPLIT_CANDIDATE"
        if col not in call["lp_feats"]:
            return "BELOW_LITERAL_POOL_CUTOFF"
        return "IN_LITERAL_POOL"

    def pool_trace(self) -> Dict[str, Any]:
        per_lit = []
        n_base = self._n_base
        for k, l in enumerate(self.derived):
            col = n_base + k
            reasons: Dict[str, int] = {r: 0 for r in REJECTION}
            for call in self._pool_log:
                reasons[self._classify(col, call)] += 1
            root = self._pool_log[0] if self._pool_log else None
            ext = sorted(self._dcols[col])
            comp = [j for j in range(n_base) if self._family.get(j) == l.family and j not in set(ext)]
            def can_build(cols, greater, call):
                return bool(cols) and len(cols) <= self.max_n and all((j, greater) in call["lp"] for j in cols)
            pos_any = any(can_build(ext, True, c) for c in self._pool_log)
            comp_any = any(can_build(comp, False, c) for c in self._pool_log)
            per_lit.append({"literal": l.name, "column": col, "concept": l.concept, "root_var_rank": (root["var_rank"].get(col) if root else None),
                            "root_reason": (self._classify(col, root) if root else None), "reasons_by_search": reasons,
                            "original_equivalent": {"extension_cols": len(ext), "complement_cols": len(comp), "positive_form_fits_max_n": 0 < len(ext) <= self.max_n, "complement_form_fits_max_n": 0 < len(comp) <= self.max_n,
                                                    "positive_form_buildable_from_literal_pool": pos_any, "complement_form_buildable_from_literal_pool": comp_any}})
        counters = {"simple_generated": 0, "simple_evaluated": 0, "mofn_generated": 0, "mofn_evaluated": 0}
        for n in self.iter_nodes():
            c = (getattr(n, "stats", None) or {}).get("candidate_counters")
            if c:
                for key in counters:
                    counters[key] += int(c.get(key, 0))
        return {"n_searches": len(self._pool_log), "feature_slots": int(self.max_features_per_node), "forced": list(self.forced), "masked": list(self.masked), "literals": per_lit,
                "counters": counters, "search_scores": [s["score"] if s else None for s in self._search_log]}


# ----------------------------------------------------------------------------------------------- execução de um braço
def fit_pool_arm(X_base: np.ndarray, names: Sequence[str], derived: Sequence[DerivedLiteral], oracle, params: Dict[str, Any], *, extras: str = "none", mf: Optional[int] = None,
                 forced_base: Sequence[int] = (), masked_derived: Sequence[int] = (), prune: bool = False) -> Dict[str, Any]:
    """``extras``: 'none' (pool do Original); 'derived' (todas as derivadas ocupam vagas extra); 'base' (``forced_base`` colunas BASE ocupam vagas extra). ``mf`` = vagas totais (já inclui extras)."""
    Xa, na = augment(X_base, names, derived)
    n_base = len(names)
    forced: List[int] = []
    if extras == "derived":
        forced = [n_base + k for k in range(len(derived))]
    elif extras == "base":
        forced = list(forced_base)
    p = dict(params)
    if mf is not None:
        p["max_features_per_node"] = int(mf)
    m = PoolTrepan(derived=list(derived), base_names=list(names), prune_redundant=prune, forced=forced, masked=[n_base + k for k in masked_derived], **p)
    m._feature_names_for_sig = list(na)
    q0 = getattr(oracle, "calls", 0)
    m.fit(Xa, oracle=oracle, feature_names=na)
    tr = m.semantic_trace_
    internal = [n for n in m.iter_nodes() if not n.is_leaf]
    return {"model": m, "trace": tr, "names": list(na), "queries": int(m.membership_queries_), "oracle_calls": getattr(oracle, "calls", 0) - q0, "n_internal_nodes": len(internal),
            "depth": int(m.get_depth()), "feature_slots": int(m.max_features_per_node), "n_columns": Xa.shape[1], "forced": forced}


def ladder_state(lit_trace: Dict[str, Any], pool_lit: Dict[str, Any], *, validated: bool, changed_by_ablation: Optional[bool]) -> Dict[str, Any]:
    """Estado mais alto atingido, com cada nível exigindo os anteriores. ``OBSERVED_BENEFIT`` fica sempre ``NOT_ASSESSED`` (precisa de benchmark com controlos; não executado)."""
    reached = {"AXIOM_AVAILABLE": True, "AXIOM_VALIDATED": bool(validated), "FACT_COMPILED": bool(lit_trace.get("fact_ids"))}
    reached["LITERAL_ELIGIBLE"] = pool_lit["reasons_by_search"].get("IN_LITERAL_POOL", 0) > 0
    reached["LITERAL_EVALUATED"] = lit_trace.get("candidate_evaluations", 0) > 0
    reached["SPLIT_DECISION_CHANGED"] = bool(changed_by_ablation)
    top = None
    for s in LADDER[:-1]:
        if reached[s]:
            top = s
        else:
            break
    return {"highest_state": top, "reached": reached, "OBSERVED_BENEFIT": "NOT_ASSESSED"}


def ablate_literal(X_base: np.ndarray, names: Sequence[str], derived: Sequence[DerivedLiteral], make_oracle, params: Dict[str, Any], k: int, *, extras: str, mf: int,
                   forced_base: Sequence[int] = (), full: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ablação por máscara: a coluna do literal ``k`` deixa de ser elegível, SEM libertar a vaga (restante *pool* idêntico), mesmo oráculo de forma, mesma semente."""
    full = full or fit_pool_arm(X_base, names, derived, make_oracle(), params, extras=extras, mf=mf, forced_base=forced_base)
    abl = fit_pool_arm(X_base, names, derived, make_oracle(), params, extras=extras, mf=mf, forced_base=forced_base, masked_derived=[k])
    sf, sa = full["model"]._search_log, abl["model"]._search_log
    n = max(len(sf), len(sa))
    diff_nodes = [i for i in range(n) if (sf[i] if i < len(sf) else "MISSING") != (sa[i] if i < len(sa) else "MISSING") and
                  not _same_sig(sf[i] if i < len(sf) else None, sa[i] if i < len(sa) else None)]
    name = derived[k].name
    chosen = [i for i, s in enumerate(sf) if s and any(nm == name for nm, _, _ in s["sig"][1])]
    root_delta = None
    if sf and sa and sf[0] and sa[0]:
        root_delta = float(sf[0]["score"] - sa[0]["score"])
    return {"literal": name, "chosen_in_searches": chosen, "searches_changed_by_ablation": diff_nodes, "changed": bool(diff_nodes), "root_score_delta_oracle_labels": root_delta}


def _same_sig(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return a["sig"] == b["sig"]


def original_buildable(model: "PoolTrepan", derived: Sequence[DerivedLiteral], base_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Para cada literal derivado: o Original (no *pool* deste modelo, que pode nem ter colunas derivadas) consegue formar o teste equivalente? Forma positiva = 1-de-n sobre as colunas da extensão
    (todas como ``x>0.5`` no *pool* de literais, n ≤ ``max_n``); forma complementar = AND dos complementares da família (``x≤0.5``). Mede OPORTUNIDADE de pesquisa, não benefício."""
    pos = {n: i for i, n in enumerate(base_names)}
    rows = []
    for l in derived:
        ext = sorted(pos[a] for a in l.attributes)
        comp = [j for j, n in enumerate(base_names) if n.split("__")[0] == l.family and j not in set(ext)]
        calls = model._pool_log
        in_feat = [all(j in c["pool_feats"] for j in ext) for c in calls]
        pos_ok = [0 < len(ext) <= model.max_n and all((j, True) in c["lp"] for j in ext) for c in calls]
        comp_ok = [0 < len(comp) <= model.max_n and all((j, False) in c["lp"] for j in comp) for c in calls]
        rows.append({"literal": l.name, "extension_cols": len(ext), "complement_cols": len(comp), "searches": len(calls), "extension_cols_all_in_feature_pool_searches": int(sum(in_feat)),
                     "positive_form_buildable_searches": int(sum(pos_ok)), "complement_form_buildable_searches": int(sum(comp_ok)),
                     "equivalent_test_buildable_in_some_search": bool(any(pos_ok) or any(comp_ok))})
    return rows


# ----------------------------------------------------------------------------------------------- contrato de comparação
def comparability(a: Dict[str, Any], b: Dict[str, Any], *, same_instances: bool, same_oracle: bool, same_params_except_pool: bool, tol: Dict[str, float]) -> Dict[str, Any]:
    """Contrato (Doc. 61 §3): condições estruturais obrigatórias + razão relativa nas quantidades igualadas (tolerâncias são PROPOSTA, não congeladas). Qualquer falha → NOT_COMPARABLE."""
    def rel(x, y):
        return abs(x - y) / max(1.0, max(abs(x), abs(y)))
    ca, cb = a["trace"]["pool"]["counters"], b["trace"]["pool"]["counters"]
    measures = {"feature_slots": (a["feature_slots"], b["feature_slots"]), "mofn_generated": (ca["mofn_generated"], cb["mofn_generated"]), "mofn_evaluated": (ca["mofn_evaluated"], cb["mofn_evaluated"]),
                "simple_evaluated": (ca["simple_evaluated"], cb["simple_evaluated"]), "queries": (a["queries"], b["queries"]), "n_internal_nodes": (a["n_internal_nodes"], b["n_internal_nodes"]), "depth": (a["depth"], b["depth"])}
    out = {"same_instances": same_instances, "same_oracle": same_oracle, "same_params_except_pool": same_params_except_pool, "measures": {}, "violations": []}
    for k, (x, y) in measures.items():
        r = rel(x, y)
        out["measures"][k] = {"a": x, "b": y, "rel_diff": round(r, 4), "tolerance": tol.get(k)}
        if k in tol and r > tol[k]:
            out["violations"].append(k)
    for flag in ("same_instances", "same_oracle", "same_params_except_pool"):
        if not out[flag]:
            out["violations"].append(flag)
    out["status"] = "COMPARABLE" if not out["violations"] else "NOT_COMPARABLE"
    return out
