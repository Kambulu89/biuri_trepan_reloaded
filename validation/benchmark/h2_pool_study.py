"""H2 (Doc. 59/61, revisão pós-Docs. 58-61): análise TÉCNICA da elegibilidade de literais derivados e controlo experimental do *pool* de candidatos. Só ``validation/``; ``core/`` e
``max_features_per_node`` do núcleo intocados (o parâmetro do construtor é usado apenas como entrada do protótipo). NÃO calcula accuracy nem fidelidade a rótulos reais nem lê o teste: as únicas
quantidades usadas são estruturais (contagens de candidatos/consultas/nós) e o ganho de informação sobre as etiquetas do ORÁCULO no treino, que é o próprio critério interno do TREPAN.

Código agnóstico: nada aqui conhece datasets, nomes de atributos ou códigos; os conceitos chegam como ``DerivedLiteral`` (compilados de qualquer OWL pelo registo B3.1).

Etapas por literal derivado (cada uma pode rejeitá-lo, por esta ordem): (1) *pool de features* do nó = top-``max_features_per_node`` pela prioridade (Original: variância — sem rótulos);
(2) geração de candidatos simples (limiares com folhas ≥ ``min_samples_leaf``); (3) *pool de literais* do m-of-n = top ``max(8, 3·mf)`` por ganho de informação; (4) avaliação como candidato (simples ou em m-of-n);
(5) escolha. Estados da escada: AXIOM_AVAILABLE → AXIOM_VALIDATED → FACT_COMPILED → LITERAL_ELIGIBLE → LITERAL_EVALUATED → SPLIT_DECISION_CHANGED → OBSERVED_BENEFIT (este último NÃO é avaliado aqui)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import time
import tracemalloc
from collections import Counter

import numpy as np

from core.trepan_original import TrepanOriginalClassifier
from validation.benchmark.h2_axiom_mechanisms import DerivedLiteral
from validation.benchmark.h2_trace import DerivedLiteralTrepan, augment

LADDER = ("LITERAL_VALID", "LITERAL_ELIGIBLE", "LITERAL_EVALUATED", "LITERAL_SELECTED", "SPLIT_CHANGED", "BENEFIT_OBSERVED")
STATE_MEANING = {
    "LITERAL_VALID": "o literal vem de um axioma disponível, consistente no reasoner, com facto compilado e atributos mapeados (validade TÉCNICA; NÃO é verdade de domínio, que fica em domain_validation)",
    "LITERAL_ELIGIBLE": "entrou no pool de features E no pool de literais do m-of-n em pelo menos uma pesquisa de split",
    "LITERAL_EVALUATED": "foi avaliado como candidato (simples ou dentro de um m-of-n) pelo critério interno do TREPAN",
    "LITERAL_SELECTED": "faz parte do teste escolhido em pelo menos um nó da árvore completa",
    "SPLIT_CHANGED": "a ablação controlada do axioma/literal (máscara sem libertar a vaga) altera pelo menos uma decisão de split",
    "BENEFIT_OBSERVED": "ganho CONJUNTO de accuracy e fidelidade, atribuído por contrastes emparelhados e controlos válidos (NÃO avaliado neste arnês: NOT_ASSESSED)"}
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
        counters: Dict[str, int] = {"simple_generated": 0, "simple_evaluated": 0, "mofn_generated": 0, "mofn_evaluated": 0}
        for n in self.iter_nodes():
            c = (getattr(n, "stats", None) or {}).get("candidate_counters")
            if c:
                for key, v in c.items():                                         # TODOS os contadores (rejeições, duplicados, candidatos semânticos…)
                    counters[key] = counters.get(key, 0) + int(v)
        counters["duplicates_removed"] = counters["simple_generated"] - counters["simple_evaluated"]
        counters["invalid_mofn"] = counters["mofn_generated"] - counters["mofn_evaluated"]
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
    q0, r0 = getattr(oracle, "calls", 0), getattr(oracle, "rows", None)
    already = tracemalloc.is_tracing()
    if not already:
        tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.process_time()
    m.fit(Xa, oracle=oracle, feature_names=na)
    cpu = time.process_time() - t0
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    if not already:
        tracemalloc.stop()
    tests = [n.test for n in m.iter_nodes() if not n.is_leaf and n.test is not None]
    complexity = {"n_tests": len(tests), "total_literals_in_tests": int(sum(len(t.literals) for t in tests)), "max_literals_in_a_test": int(max((len(t.literals) for t in tests), default=0)),
                  "derived_literals_in_tests": int(sum(1 for t in tests for l in t.literals if l.feature >= n_base))}
    tr = m.semantic_trace_
    internal = [n for n in m.iter_nodes() if not n.is_leaf]
    stops = Counter(str(n.stop_reason) for n in m.iter_nodes() if n.is_leaf)
    return {"model": m, "trace": tr, "names": list(na), "queries": int(m.membership_queries_), "oracle_calls": getattr(oracle, "calls", 0) - q0,
            "rows_consulted": (None if r0 is None else int(getattr(oracle, "rows", 0) - r0)), "cpu_seconds": float(cpu), "peak_memory_mb": float(peak_mb), "complexity": complexity, "stop_reasons": dict(stops),
            "n_internal_nodes": len(internal), "depth": int(m.get_depth()), "feature_slots": int(m.max_features_per_node), "beam_width": int(m.beam_width), "n_columns": Xa.shape[1], "forced": forced}


def ladder_state(lit_trace: Dict[str, Any], pool_lit: Dict[str, Any], *, valid: bool, selected: bool, changed_by_ablation: Optional[bool]) -> Dict[str, Any]:
    """Estados SEPARADOS por literal (nenhum implica o seguinte): válido ≠ elegível ≠ avaliado ≠ selecionado ≠ split alterado ≠ benefício. ``reached`` guarda cada um independentemente;
    ``highest_consecutive`` é o último estado da escada sem lacunas. ``BENEFIT_OBSERVED`` fica sempre ``NOT_ASSESSED`` (precisa de benchmark com controlos; não executado)."""
    reached = {"LITERAL_VALID": bool(valid)}
    reached["LITERAL_ELIGIBLE"] = pool_lit["reasons_by_search"].get("IN_LITERAL_POOL", 0) > 0
    reached["LITERAL_EVALUATED"] = lit_trace.get("candidate_evaluations", 0) > 0
    reached["LITERAL_SELECTED"] = bool(selected)
    reached["SPLIT_CHANGED"] = bool(changed_by_ablation)
    top = None
    for st in LADDER[:-1]:
        if reached[st]:
            top = st
        else:
            break
    return {"highest_consecutive": top, "reached": reached, "BENEFIT_OBSERVED": "NOT_ASSESSED"}


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
@dataclass(frozen=True)
class Ceilings:
    """Tetos AUTORIZADOS (declarados antes de qualquer resultado; PROPOSTA, não congelados) — distintos dos recursos EFETIVAMENTE CONSUMIDOS. ``None`` = sem teto nessa dimensão.
    O CPU é uma modalidade de comparação distinta: só entra no estado se tiver teto."""
    eligible_literals: Optional[int] = None          # vagas de features do pool (colunas elegíveis por pesquisa)
    evaluated_candidates: Optional[int] = None       # candidatos m-of-n avaliados (total na árvore)
    oracle_queries: Optional[int] = None             # consultas sintéticas ao oráculo (membership queries)
    rows_consulted: Optional[int] = None             # linhas enviadas ao oráculo
    nodes: Optional[int] = None                      # nós internos
    cpu_seconds: Optional[float] = None


def consumption(arm: Dict[str, Any]) -> Dict[str, Any]:
    c = arm["trace"]["pool"]["counters"]
    return {"eligible_literals": arm["feature_slots"], "evaluated_candidates": c["mofn_evaluated"], "oracle_queries": arm["queries"], "rows_consulted": arm.get("rows_consulted"),
            "nodes": arm["n_internal_nodes"], "cpu_seconds": arm.get("cpu_seconds")}


def check_ceilings(arm: Dict[str, Any], ceilings: Ceilings) -> Dict[str, Any]:
    used, over = consumption(arm), []
    for k, lim in ceilings.__dict__.items():
        if lim is not None and used.get(k) is not None and used[k] > lim:
            over.append(k)
    return {"authorized": {k: v for k, v in ceilings.__dict__.items() if v is not None}, "consumed": used, "exceeded": over}


REALIZED_DIMENSIONS = ("mofn_generated", "mofn_evaluated", "simple_evaluated", "queries", "rows_consulted", "n_internal_nodes", "depth")


def resource_report(arm: Dict[str, Any], ceilings: Optional[Ceilings] = None, *, extra_fits: int = 0) -> Dict[str, Any]:
    """Relatório obrigatório de recursos de UM braço: tetos AUTORIZADOS vs consumos EFETIVOS (candidatos, consultas, nós, vagas), nº de literais, CPU, memória, árvores construídas, complexidade e
    diferenças de representação. ``extra_fits`` = ajustes auxiliares (calibração/ablação) imputáveis ao braço."""
    c = arm["trace"]["pool"]["counters"]
    ceil = (ceilings.__dict__ if ceilings is not None else {})
    return {"authorized": {k: v for k, v in ceil.items() if v is not None},
            "candidates": {"ceiling_evaluated": ceil.get("evaluated_candidates"), "generated": c["mofn_generated"], "evaluated": c["mofn_evaluated"], "simple_evaluated": c["simple_evaluated"], "invalid": c.get("invalid_mofn")},
            "queries": {"ceiling": ceil.get("oracle_queries"), "executed": arm["queries"], "oracle_calls": arm["oracle_calls"], "rows_consulted": arm.get("rows_consulted")},
            "literals": {"feature_slots": arm["feature_slots"], "columns": arm["n_columns"], "derived": arm["trace"]["n_derived"], "forced_columns": list(arm["forced"])},
            "cpu_seconds": arm.get("cpu_seconds"), "peak_memory_mb": arm.get("peak_memory_mb"), "trees_built": 1 + int(extra_fits), "tree": {"internal_nodes": arm["n_internal_nodes"], "depth": arm["depth"], **arm.get("complexity", {})},
            "representation": {"n_base_columns": arm["n_columns"] - arm["trace"]["n_derived"], "n_derived_columns": arm["trace"]["n_derived"], "beam_width": arm.get("beam_width")}}


def comparability(a: Dict[str, Any], b: Dict[str, Any], *, same_instances: bool, same_oracle: bool, same_params_except_pool: bool, tol: Dict[str, float],
                  ceilings: Optional[Ceilings] = None, design_equalized: Sequence[str] = ()) -> Dict[str, Any]:
    """Dois conceitos SEPARADOS (Doc. 62 §4):
    * ``configuration`` — ``CONFIGURATION_COMPARABLE`` se os braços receberam condições e limites compatíveis com o contraste: mesmas instâncias, oráculo e parâmetros (exceto o pool), mesmo feixe, ambos
      DENTRO dos tetos autorizados e as dimensões que o DESENHO do contraste declara igualar (``design_equalized``, p. ex. vagas, ou candidatos avaliados no braço F) dentro da tolerância. Consumos
      diferentes sob o mesmo teto NÃO tornam o contraste não comparável.
    * ``realized_resource_match`` — grau descritivo de correspondência do consumo EFETIVO (nunca decide a comparabilidade): ``EXACT`` | ``WITHIN_TOLERANCE`` | ``PARTIAL`` + diferenças relativas.
    Tolerâncias e tetos são PROPOSTA (não congelados). Não se igualam consumos a posteriori."""
    def rel(x, y):
        return abs(x - y) / max(1.0, max(abs(x), abs(y)))
    ca, cb = a["trace"]["pool"]["counters"], b["trace"]["pool"]["counters"]
    measures = {"feature_slots": (a["feature_slots"], b["feature_slots"]), "mofn_generated": (ca["mofn_generated"], cb["mofn_generated"]), "mofn_evaluated": (ca["mofn_evaluated"], cb["mofn_evaluated"]),
                "simple_evaluated": (ca["simple_evaluated"], cb["simple_evaluated"]), "queries": (a["queries"], b["queries"]), "n_internal_nodes": (a["n_internal_nodes"], b["n_internal_nodes"]),
                "depth": (a["depth"], b["depth"]), "beam_width": (a.get("beam_width", 0), b.get("beam_width", 0))}
    if a.get("rows_consulted") is not None and b.get("rows_consulted") is not None:
        measures["rows_consulted"] = (a["rows_consulted"], b["rows_consulted"])
    structural = {"same_instances": same_instances, "same_oracle": same_oracle, "same_params_except_pool": same_params_except_pool}
    cfg_viol = [k for k, v in structural.items() if not v]
    if measures["beam_width"][0] != measures["beam_width"][1]:
        cfg_viol.append("beam_width")
    per = {}
    for k, (x, y) in measures.items():
        per[k] = {"a": x, "b": y, "rel_diff": round(rel(x, y), 4), "tolerance": tol.get(k)}
    for k in design_equalized:
        if k in per and per[k]["rel_diff"] > (tol.get(k) or 0.0):
            cfg_viol.append(f"DESIGN_NOT_EQUALIZED:{k}")
    out = {"structural": structural, "design_equalized": list(design_equalized), "measures": per,
           "cpu_modality": {"a": a.get("cpu_seconds"), "b": b.get("cpu_seconds"), "decides_status": bool(ceilings and ceilings.cpu_seconds is not None)},
           "memory_mb": {"a": a.get("peak_memory_mb"), "b": b.get("peak_memory_mb")}}
    if ceilings is not None:
        out["ceilings"] = {"a": check_ceilings(a, ceilings), "b": check_ceilings(b, ceilings)}
        for side in ("a", "b"):
            cfg_viol += [f"CEILING_EXCEEDED[{side}]:{k}" for k in out["ceilings"][side]["exceeded"]]
    out["configuration_violations"] = cfg_viol
    out["status"] = "CONFIGURATION_COMPARABLE" if not cfg_viol else "NOT_COMPARABLE"
    outside = [k for k in REALIZED_DIMENSIONS if k in per and k in tol and per[k]["rel_diff"] > tol[k]]
    exact = all(per[k]["rel_diff"] == 0.0 for k in REALIZED_DIMENSIONS if k in per)
    out["realized_resource_match"] = {"grade": "EXACT" if exact else ("WITHIN_TOLERANCE" if not outside else "PARTIAL"), "dimensions_outside_tolerance": outside,
                                      "note": "descritivo: consumos distintos sob o mesmo teto não invalidam o contraste"}
    return out


def match_generic_by_evaluated(fit_for_mf, target_evaluated: int, mf_range: Sequence[int], *, tolerance: float) -> Dict[str, Any]:
    """Braço F: pool GENÉRICO (sem OWL) cujo ``max_features_per_node`` iguala os candidatos m-of-n AVALIADOS do braço de referência. Regra pré-declarada: percorre TODO o intervalo (sem paragem
    antecipada), escolhe o menor |Δ| relativo (empate → menor mf) usando só contagens (sem rótulos reais nem desempenho). Se o melhor ficar acima da tolerância, ``feasible=False`` (→ NOT_COMPARABLE)."""
    best = None
    for mf in mf_range:
        arm = fit_for_mf(mf)
        ev = arm["trace"]["pool"]["counters"]["mofn_evaluated"]
        gap = abs(ev - target_evaluated) / max(1.0, float(target_evaluated))
        if best is None or gap < best[0] - 1e-12:
            best = (gap, mf, arm)
    return {"arm": best[2], "mf": best[1], "relative_gap": float(best[0]), "feasible": bool(best[0] <= tolerance)}


def cards_by_concept(cards_dir) -> Dict[str, Dict[str, Any]]:
    """Liga cada ficha de axioma (JSON externo) ao conceito que define, lendo o axioma OWL funcional (``EquivalentClasses(<conceito> …)``). Genérico: não conhece nomes."""
    import json
    import re
    from pathlib import Path
    out: Dict[str, Dict[str, Any]] = {}
    for f in sorted(Path(cards_dir).glob("*.json")):
        c = json.loads(f.read_text(encoding="utf-8"))
        m = re.match(r"\s*EquivalentClasses\(\s*(\S+)", c.get("owl_axiom", ""))
        if m:
            out[m.group(1)] = {"card_id": c.get("id"), "authored_by": c.get("authored_by"), "epistemic_status": c.get("epistemic_status"), "source_verified": bool(c.get("source", {}).get("verified")),
                               "reviews": {k: v.get("status") for k, v in c.get("reviews", {}).items()}, "domain_validation": "PENDING" if c.get("epistemic_status") != "VALIDATED" else "VALIDATED"}
    return out
