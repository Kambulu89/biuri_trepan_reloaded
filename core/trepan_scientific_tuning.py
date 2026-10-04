"""Selecção científica train-only da capacidade TREPAN e força semântica.

O módulo nunca recebe o teste externo. Primeiro selecciona uma capacidade comum
para Original/Reloaded usando o TREPAN Original; depois, mantendo exactamente
essa capacidade, selecciona apenas parâmetros da extensão semântica do Reloaded.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np
from scipy import stats as _scipy_stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from core.training_config import DEFAULT_QUERY_BUDGET_CAP, non_binding_query_budget, required_query_budget
from core.controlled_trepan_experiment import ControlledTrepanConfig
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier


@dataclass(frozen=True)
class ScientificTrepanSearchConfig:
    cv_folds: int = 3
    # Repeated Stratified K-Fold no TREINO: ``cv_repeats`` repetições, cada uma com a sua seed determinística
    # (``cv_seeds`` explícitas, ou derivadas de ``base_config.random_state``). Nunca vê o conjunto de teste.
    cv_repeats: int = 5                       # configuração científica principal: 5 repetições × 3 dobras
    cv_seeds: Optional[tuple] = None
    max_capacity_candidates: int = 6
    max_semantic_candidates: int = 6
    fidelity_target: float = 0.95            # legado: já não intervém na seleção (agora lexicográfica)
    fidelity_weight: float = 0.65
    balanced_accuracy_weight: float = 0.20
    macro_f1_weight: float = 0.15
    complexity_penalty: float = 0.015
    # Grelha de estrutura (literatura do TREPAN: NIPS 1995 usa 0.05 e 31 nós; a tese de 1996 usa 63 nós).
    # Escolhida por CV interna repetida, só no treino. Grelhas vazias ou tune_structure=False desativam-na e ficam
    # os valores canónicos da configuração base. A grelha é configurável/extensível (qualquer tuplo de valores).
    tune_structure: bool = True
    purity_epsilon_grid: tuple = (0.05, 0.02, 0.01)
    max_nodes_grid: tuple = (31, 63)
    # Seleção lexicográfica (ver ``_lexicographic_select``): 1) fidelity; 2) estabilidade da fidelity entre
    # candidatos estatisticamente indistinguíveis; 3) estabilidade estrutural; 4) complexidade; 5) empate real ->
    # a mais simples.
    significance_alpha: float = 0.05          # teste t reamostrado corrigido (Nadeau & Bengio, 2003), unilateral
    stability_tolerance: float = 0.005        # diferença de desvio-padrão da fidelity tratada como empate
    structural_stability_tolerance: float = 0.05   # diferença do índice de instabilidade estrutural (CV) tratada como empate
    # Orçamento de queries da grelha de estrutura. "non_binding": todos os candidatos recebem o MESMO orçamento comum,
    # que nunca limita nenhum deles (budget_exhausted == False por construção), para que o orçamento não seja uma
    # variável de confusão. "required": orçamento mínimo por candidato (comportamento anterior, pode esgotar).
    query_budget_policy: str = "non_binding"
    # Robustez interna da seleção: block bootstrap das repetições da CV (cada repetição = 1 bloco que preserva as suas
    # dobras), com a MESMA política de seleção em cada reamostragem. Não introduz política nova: mede a política.
    bootstrap_resamples: int = 1000           # só para o Monte Carlo (quando a enumeração exata é impraticável)
    bootstrap_exact_limit: int = 20000        # enumeração exata se o nº de composições distintas (multisets) <= isto
    bootstrap_seed: Optional[int] = None      # None -> derivada da seed base (determinística)
    min_selection_probability: float = 0.6    # abaixo disto o tuning é marcado "incerto" (limiar configurável, não prova)
    min_equivalent_set_probability: float = 0.6   # P(vencedora do bootstrap ∈ conjunto equivalente) para ``stable_equivalent_set``
    behavior_instability_threshold: float = 0.5   # índice de instabilidade estrutural acima do qual o comportamento da árvore é instável
    # Expansão adaptativa da capacidade (genérica; só resultados da CV do treino): se os candidatos competitivos de maior
    # capacidade atingem o teto max_nodes numa fração >= ao limiar, a grelha de nós cresce geometricamente
    # (next = int(factor * atual) + 1, ex. 31 -> 63 -> 127 -> 255), avaliada nas MESMAS dobras, até a saturação deixar
    # de ser relevante ou se atingir o limite de segurança / nº máximo de rondas. purity_epsilon não é alterado.
    capacity_expansion: bool = True
    capacity_expansion_factor: float = 2.0
    capacity_expansion_saturation_threshold: float = 0.5
    max_capacity_expansion_rounds: int = 3
    max_nodes_safety_limit: int = 255
    # A expansão exige SATURAÇÃO + GANHO DE VALIDAÇÃO suportado (teste t reamostrado corrigido, as mesmas dobras, mesmo
    # purity_epsilon em capacidades consecutivas; ver ``capacity_gain_step``); sem ganho -> ``validation_plateau``.
    # Saturação: se a fração de árvores que param por atingir max_nodes for >= isto, a estabilidade estrutural
    # desse candidato é sinalizada como potencialmente censurada pelo teto (só diagnóstico; não altera a seleção).
    node_cap_censoring_threshold: float = 0.5


def _cv_plan(y, search: "ScientificTrepanSearchConfig", base_seed: int):
    """Plano de Repeated Stratified K-Fold no treino: lista de (repetição, seed, dobra, treino, validação).

    Determinístico: as seeds são explícitas (``search.cv_seeds``) ou derivadas da seed base. O mesmo plano é usado
    por todos os candidatos, o que permite comparações emparelhadas.
    """
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    k = max(2, min(int(search.cv_folds), int(counts.min())))
    seeds = tuple(int(v) for v in search.cv_seeds) if search.cv_seeds else \
        tuple(int(base_seed) + 1009 * i for i in range(max(1, int(search.cv_repeats))))
    X_index = np.zeros(len(y))
    plan = []
    for r, seed in enumerate(seeds):
        for f, (tr, va) in enumerate(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X_index, y)):
            plan.append((r, seed, f, tr, va))
    return plan, k, seeds


def _objective(model, X_val, y_val, oracle, cfg: ControlledTrepanConfig, search: ScientificTrepanSearchConfig):
    pred = np.asarray(model.predict(X_val))
    teacher = np.asarray(oracle.predict(X_val))
    fidelity = float(accuracy_score(teacher, pred))
    ba = float(balanced_accuracy_score(y_val, pred))
    f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
    complexity = float(getattr(model, "node_count_", 0)) / max(1.0, float(cfg.max_nodes))
    score = (
        search.fidelity_weight * fidelity
        + search.balanced_accuracy_weight * ba
        + search.macro_f1_weight * f1
        - search.complexity_penalty * complexity
    )
    return {"score": float(score), "fidelity": fidelity, "balanced_accuracy": ba, "macro_f1": f1, "complexity": complexity}


def _scaled(base: ControlledTrepanConfig, **kw) -> ControlledTrepanConfig:
    """Candidato derivado de ``base``; mais nós permitidos exigem mais queries (senão ficaria truncado pelo orçamento)."""
    c = replace(base, **kw)
    need = required_query_budget(c.max_nodes, c.min_sample, cap=max(int(base.max_queries), DEFAULT_QUERY_BUDGET_CAP))
    return replace(c, max_queries=max(int(c.max_queries), need)) if c.max_nodes != base.max_nodes else c


def _node_depth_limit(base: ControlledTrepanConfig, nodes: int) -> int:
    """A profundidade nunca limita antes do teto de nós (uma árvore binária com N nós tem profundidade <= (N-1)//2)."""
    return max(int(base.max_depth), 6, (int(nodes) - 1) // 2)


def _common_query_budget(base: ControlledTrepanConfig, search: "ScientificTrepanSearchConfig", max_nodes: int) -> int:
    """Orçamento comum e não limitante (majora o consumo possível até ao maior nº de nós que a pesquisa pode atingir)."""
    top = int(max_nodes)
    if search.capacity_expansion:
        top = max(top, int(search.max_nodes_safety_limit))
    return max(int(base.max_queries), non_binding_query_budget(top, base.min_sample))


def _structure_point(base, search, eps, nodes, common):
    c = replace(base, purity_epsilon=float(eps), max_nodes=int(nodes), max_depth=_node_depth_limit(base, nodes))
    if common is not None:
        return replace(c, max_queries=common)
    cap = max(int(base.max_queries), DEFAULT_QUERY_BUDGET_CAP)
    need = required_query_budget(c.max_nodes, c.min_sample, cap=cap)
    return replace(c, max_queries=max(int(base.max_queries), need))


def _structure_candidates(base: ControlledTrepanConfig, search: "ScientificTrepanSearchConfig"):
    """Grelha purity_epsilon x max_nodes. O primeiro candidato é sempre o ponto canónico da base.

    Equidade computacional (``query_budget_policy``):
    - "non_binding" (omissão): TODOS os candidatos recebem o mesmo orçamento comum, calculado com
      ``non_binding_query_budget`` (majorante do consumo possível, incluindo a capacidade que a expansão adaptativa
      pode atingir). O orçamento nunca limita nenhum candidato, logo não pode explicar diferenças entre eles.
    - "required": cada candidato recebe o orçamento necessário para os seus próprios nós (pode esgotar).
    """
    pairs = [(float(base.purity_epsilon), int(base.max_nodes))]
    for nodes in search.max_nodes_grid:
        for eps in search.purity_epsilon_grid:
            key = (float(eps), int(nodes))
            if key not in pairs:
                pairs.append(key)
    common = None
    if str(search.query_budget_policy) == "non_binding":
        common = _common_query_budget(base, search, max(n for _e, n in pairs))
    return [_structure_point(base, search, eps, nodes, common) for eps, nodes in pairs]


def _expansion_candidates(base, search, nodes: int, existing):
    """Candidatos da capacidade expandida: a MESMA grelha de purity_epsilon, só com o novo teto de nós."""
    common = None
    if str(search.query_budget_policy) == "non_binding":
        common = _common_query_budget(base, search, nodes)
    have = {(float(c["purity_epsilon"]), int(c["max_nodes"])) for c in existing}
    out = []
    for eps in search.purity_epsilon_grid:
        if (float(eps), int(nodes)) not in have:
            out.append(_structure_point(base, search, eps, nodes, common))
    return out


def _nb_paired(diff, k: int, alpha: float):
    """Teste t reamostrado corrigido (Nadeau & Bengio, 2003) sobre diferenças emparelhadas ``diff`` (unilateral: média > 0)."""
    d = np.asarray(diff, dtype=float)
    J = len(d)
    mean = float(d.mean()) if J else 0.0
    if J < 2:
        return {"mean": mean, "t": 0.0, "se": None, "df": 0, "t_critical": float("inf"), "p_value": 1.0, "ci95": None, "significant": False}
    var = float(d.var(ddof=1))
    se = float(np.sqrt((1.0 / J + 1.0 / max(1, k - 1)) * var))
    if se <= 0.0:
        t = 0.0 if mean <= 1e-12 else float("inf")
    else:
        t = mean / se
    df = J - 1
    t_crit = float(_scipy_stats.t.ppf(1.0 - float(alpha), df))
    p = float(_scipy_stats.t.sf(t, df)) if np.isfinite(t) else 0.0
    t_two = float(_scipy_stats.t.ppf(0.975, df))
    ci = [mean - t_two * se, mean + t_two * se] if se > 0 else [mean, mean]
    return {"mean": mean, "t": float(t), "se": se, "df": int(df), "t_critical": t_crit, "p_value": p, "ci95": ci,
            "significant": bool(t > t_crit)}


_GAIN_TEST_NAME = ("teste t emparelhado reamostrado corrigido (Nadeau & Bengio, 2003), unilateral; efeito da capacidade "
                   "agregado por partição sobre os purity_epsilon comuns")


def capacity_gain_step(history, prev_nodes: int, cand_nodes: int, search: "ScientificTrepanSearchConfig", k: int) -> dict:
    """Compara duas capacidades CONSECUTIVAS nas mesmas partições de CV (só treino) e no MESMO purity_epsilon.

    Para cada partição calcula-se a média, sobre os ``purity_epsilon`` presentes nas duas capacidades, da diferença de
    fidelity (maior - menor); uma única série emparelhada alimenta um único teste (sem multiplicidade). Devolve também a
    saturação do nível candidato (fração média de árvores que atingiram o teto) e se o ganho é suportado.
    """
    def by_eps(nodes):
        return {float(h["config"]["purity_epsilon"]): h for h in history
                if h.get("stats") and int(h["config"]["max_nodes"]) == int(nodes)}
    lo, hi = by_eps(prev_nodes), by_eps(cand_nodes)
    common = sorted(set(lo) & set(hi))
    step = {"previous_max_nodes": int(prev_nodes), "candidate_max_nodes": int(cand_nodes), "paired_purity_epsilon": common,
            "previous_fidelity": None, "candidate_fidelity": None, "fidelity_delta": None, "statistical_test": None,
            "capacity_gain_supported": False, "fraction_at_node_cap": None, "saturated": False, "per_epsilon_delta": {}}
    if hi:
        step["fraction_at_node_cap"] = float(np.mean([h["stats"]["fraction_at_node_cap"] for h in hi.values()]))
        step["saturated"] = bool(step["fraction_at_node_cap"] >= float(search.capacity_expansion_saturation_threshold))
    if not common:
        step["statistical_test"] = {"name": _GAIN_TEST_NAME, "available": False, "reason": "sem purity_epsilon comum"}
        return step
    F_lo = np.array([[r["fidelity"] for r in lo[e]["per_split"]] for e in common])
    F_hi = np.array([[r["fidelity"] for r in hi[e]["per_split"]] for e in common])
    diff = (F_hi - F_lo).mean(axis=0)
    test = _nb_paired(diff, k, float(search.significance_alpha))
    step.update({"previous_fidelity": float(F_lo.mean()), "candidate_fidelity": float(F_hi.mean()), "fidelity_delta": float(diff.mean()),
                 "per_epsilon_delta": {e: float((F_hi[i] - F_lo[i]).mean()) for i, e in enumerate(common)},
                 "statistical_test": {"name": _GAIN_TEST_NAME, "available": True, "alpha": float(search.significance_alpha),
                                      "t": test["t"], "t_critical": test["t_critical"], "p_value": test["p_value"],
                                      "ci95_delta": test["ci95"], "n_splits": int(len(diff))},
                 "capacity_gain_supported": bool(test["significant"] and test["mean"] > 0)})
    return step


def run_capacity_expansion(history, evaluate_level, search: "ScientificTrepanSearchConfig", k: int) -> dict:
    """Expansão de capacidade que exige SATURAÇÃO + GANHO DE VALIDAÇÃO suportado; só usa resultados da CV do treino.

    Um nível novo só é avaliado se o nível de topo está saturado E já mostrou ganho suportado sobre o nível anterior
    (mesmas partições, mesmo purity_epsilon). Sem ganho -> ``validation_plateau`` (nenhum nível superior é avaliado).
    Saturação sem ganho significa que o algoritmo ainda consegue crescer, mas os dados de validação não justificam mais
    capacidade. ``evaluate_level(nodes)`` devolve as novas entradas de histórico (ajustes nas mesmas dobras).
    """
    levels = sorted({int(h["config"]["max_nodes"]) for h in history if h.get("stats")})
    out = {"initial_node_grid": list(levels), "final_node_grid": list(levels), "capacity_expansion_rounds": 0,
           "expansion_triggered": False, "expansion_stop_reason": "expansion_disabled", "steps": [], "last_supported_max_nodes": None,
           "interpretation": None}
    if not search.capacity_expansion:
        return out
    reason = "insufficient_capacity_levels"
    while len(levels) >= 2:
        step = capacity_gain_step(history, levels[-2], levels[-1], search, k)
        out["steps"].append(step)
        if step["capacity_gain_supported"]:
            out["last_supported_max_nodes"] = levels[-1]
        if not step["saturated"]:
            reason = "not_saturated"; step["decision"] = "stop"; break
        if not step["capacity_gain_supported"]:
            reason = "validation_plateau"; step["decision"] = "stop"; break
        out["expansion_triggered"] = True
        nxt = int(float(search.capacity_expansion_factor) * levels[-1]) + 1
        if nxt <= levels[-1]:
            reason = "invalid_expansion_factor"; step["decision"] = "stop"; break
        if nxt > int(search.max_nodes_safety_limit):
            reason = "safety_limit_reached"; step["decision"] = "stop"; break
        if out["capacity_expansion_rounds"] >= int(search.max_capacity_expansion_rounds):
            reason = "max_rounds_reached"; step["decision"] = "stop"; break
        new_entries = evaluate_level(nxt)
        if not new_entries:
            reason = "no_new_candidates"; step["decision"] = "stop"; break
        step["decision"] = "expand"
        history += new_entries
        levels.append(nxt)
        out["capacity_expansion_rounds"] += 1
    else:
        reason = "insufficient_capacity_levels"
    out["expansion_stop_reason"] = reason
    out["final_node_grid"] = list(levels)
    if reason == "validation_plateau":
        out["interpretation"] = ("a expansão de capacidade parou porque a capacidade adicional não melhorou a fidelity de validação "
                                 "(o algoritmo ainda consegue crescer estruturalmente, mas os dados de validação não justificam mais capacidade).")
    elif reason == "not_saturated":
        out["interpretation"] = "os candidatos de maior capacidade não atingem o teto de nós: não há saturação a resolver."
    return out


def _capacity_candidates(base: ControlledTrepanConfig, p: int, limit: int, grow_nodes: bool = True):
    def cfg(**kw):
        c = replace(base, **kw)
        # Mais nós permitidos exigem mais queries: sem isto o candidato "maior" ficava truncado pelo orçamento.
        need = required_query_budget(c.max_nodes, c.min_sample, cap=max(int(base.max_queries), DEFAULT_QUERY_BUDGET_CAP))
        return replace(c, max_queries=max(int(c.max_queries), need)) if c.max_nodes != base.max_nodes else c
    q2 = max(int(base.max_queries), min(12000, max(int(base.max_queries) * 2, 1000)))
    nodes2 = max(int(base.max_nodes), min(127, max(31, int(base.max_nodes) * 2 - 1))) if grow_nodes else int(base.max_nodes)
    depth2 = max(int(base.max_depth), min(nodes2, max(int(base.max_depth), 12)))
    n2 = min(5, max(2, int(base.max_n) + 1))
    f2 = min(int(p), max(int(base.max_features_per_node), min(32, max(12, int(base.max_features_per_node) * 2))))
    candidates = [
        base,
        cfg(max_nodes=nodes2, max_depth=depth2),
        cfg(max_n=n2),
        cfg(max_features_per_node=f2),
        cfg(max_queries=q2, min_sample=max(int(base.min_sample), min(2000, max(200, int(base.min_sample) * 2)))),
        cfg(max_nodes=nodes2, max_depth=depth2, max_n=n2, max_features_per_node=f2, max_queries=q2),
    ]
    out=[]; seen=set()
    for c in candidates:
        key=(c.max_nodes,c.max_depth,c.max_n,c.max_features_per_node,c.max_queries,c.min_sample,c.beam_width,c.purity_epsilon)
        if key not in seen:
            seen.add(key); out.append(c)
    return out[:max(1,int(limit))]


def _semantic_candidates(base: ControlledTrepanConfig, limit: int):
    # (ganho_semântico, coerência, fracção_active_query, budget_candidatos,
    #  força_EFSR, min_disagreement, min_ganho_fidelidade_local)
    presets = [
        (0.50, 0.10, 0.35, 12, 0.75, 0.08, 0.004),
        (1.00, 0.15, 0.50, 24, 1.00, 0.05, 0.002),
        (1.50, 0.20, 0.65, 32, 1.25, 0.05, 0.002),
        (2.00, 0.25, 0.75, 40, 1.50, 0.03, 0.001),
        (1.25, 0.35, 0.80, 48, 1.75, 0.03, 0.001),
        (2.50, 0.35, 0.85, 48, 2.00, 0.02, 0.001),
    ]
    return [
        replace(
            base,
            semantic_gain_strength=g,
            semantic_group_strength=grp,
            semantic_active_query_fraction=active,
            semantic_candidate_budget=budget,
            error_focused_refinement=True,
            error_focus_strength=focus_strength,
            error_focus_min_disagreement=min_disagreement,
            error_focus_min_local_fidelity_gain=min_local_gain,
        )
        for g,grp,active,budget,focus_strength,min_disagreement,min_local_gain in presets[:max(1,int(limit))]
    ]


def _mean_metrics(rows):
    keys=("score","fidelity","balanced_accuracy","macro_f1","complexity")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def _label(cfg: dict) -> str:
    return f"purity_epsilon={cfg['purity_epsilon']:g}, max_nodes={cfg['max_nodes']}"


def _cv(values) -> float:
    """Coeficiente de variação amostral (sem escala: não favorece árvores maiores nem menores por si só)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or v.mean() <= 0:
        return 0.0
    return float(v.std(ddof=1) / v.mean())


def structural_instability(nodes, depth=None, leaves=None) -> float:
    """Índice de instabilidade estrutural: o pior (maior) coeficiente de variação entre nós, profundidade e folhas.

    Um candidato que produz árvores 3,3,3,3,27 tem índice alto; um que produz tamanhos próximos em todas as partições
    tem índice baixo. Usa só as árvores ajustadas no treino (CV); nunca o teste.
    """
    parts = [_cv(nodes)]
    if depth is not None:
        parts.append(_cv(depth))
    if leaves is not None:
        parts.append(_cv(leaves))
    return float(max(parts))


def _lexicographic_select(F, N, meta, k, search: "ScientificTrepanSearchConfig", valid, D=None, L=None):
    """Seleção multiobjetivo lexicográfica sobre resultados EMPARELHADOS (mesmas partições para todos).

    1. Maximizar a fidelity média. Os candidatos cuja diferença para o melhor não é significativa (teste t
       reamostrado corrigido, Nadeau & Bengio 2003, unilateral, ``significance_alpha``) ficam "indistinguíveis".
    2. Entre os indistinguíveis, privilegiar a estabilidade da fidelity (menor desvio-padrão entre partições,
       com tolerância ``stability_tolerance``).
    3. Estabilidade estrutural: menor índice de instabilidade (CV de nós, profundidade e folhas, ver
       ``structural_instability``), com tolerância ``structural_stability_tolerance``.
    4. Só depois, a complexidade (menos nós médios).
    5. Empate real: a configuração mais simples (menos ``max_nodes``) e, por fim, a ordem canónica da grelha.

    Não há preferência por capacidade maior nem menor: a complexidade só intervém depois de fidelity e estabilidades
    e favorece sempre a mais simples. Devolve o índice vencedor e um veredicto (com números) por candidato.
    """
    C, J = F.shape
    verdicts = [None] * C
    idx = [i for i in range(C) if valid[i]]
    for i in range(C):
        if not valid[i]:
            verdicts[i] = {"status": "FAILED", "stage": 0, "reason_code": "FIT_FAILED",
                           "text": "falhou durante a validação cruzada (candidato descartado)."}
    if not idx:
        return None, verdicts, {}
    means = F.mean(axis=1)
    best = max(idx, key=lambda i: (means[i], -i))
    t_crit = float(_scipy_stats.t.ppf(1.0 - float(search.significance_alpha), J - 1)) if J > 1 else float("inf")
    ratio = 1.0 / max(1, k - 1)                       # n_validação / n_treino numa K-fold
    detail = {}
    indist = []
    for i in idx:
        if i == best:
            indist.append(i); detail[i] = {"delta": 0.0, "t": 0.0}; continue
        d = F[best] - F[i]
        m = float(d.mean())
        var = float(d.var(ddof=1)) if J > 1 else 0.0
        denom = float(np.sqrt((1.0 / J + ratio) * var))
        if denom <= 0.0:
            t = 0.0 if m <= 1e-12 else float("inf")
        else:
            t = m / denom
        detail[i] = {"delta": m, "t": float(t)}
        if t <= t_crit:
            indist.append(i)
    stds = {i: (float(F[i].std(ddof=1)) if J > 1 else 0.0) for i in idx}
    min_std = min(stds[i] for i in indist)
    stable = [i for i in indist if stds[i] <= min_std + float(search.stability_tolerance)]
    struct = {i: structural_instability(N[i], None if D is None else D[i], None if L is None else L[i]) for i in idx}
    min_struct = min(struct[i] for i in stable)
    tol_s = float(search.structural_stability_tolerance)
    structural = [i for i in stable if struct[i] <= min_struct + tol_s]
    nodes = {i: float(N[i].mean()) for i in idx}
    min_nodes = min(nodes[i] for i in structural)
    simplest = [i for i in structural if nodes[i] <= min_nodes + 1e-9]
    winner = min(simplest, key=lambda i: (meta[i]["max_nodes"], i))
    wl = _label(meta[winner]["config"])
    for i in idx:
        if i == winner:
            stage = 5 if len(simplest) > 1 else 4 if len(structural) > 1 else 3 if len(stable) > 1 else \
                2 if len(indist) > 1 else 1
            verdicts[i] = {"status": "WINNER", "stage": stage, "reason_code": "WINNER",
                           "text": (f"VENCE: fidelity média {means[i]:.3f}"
                                    + (" (a melhor)" if i == best else f"; indistinguível da melhor ({_label(meta[best]['config'])}, "
                                       f"Δ={detail[i]['delta']:.3f}, t={detail[i]['t']:.2f} ≤ {t_crit:.2f})")
                                    + f". {len(indist)} indistinguível(eis) em fidelity -> {len(stable)} após estabilidade da "
                                      f"fidelity -> {len(structural)} após estabilidade estrutural (índice {struct[i]:.2f}) "
                                      f"-> {len(simplest)} após complexidade"
                                    + ("; empate real resolvido pela configuração mais simples." if len(simplest) > 1 else "."))}
        elif i not in indist:
            verdicts[i] = {"status": "LOST", "stage": 1, "reason_code": "LOST_FIDELITY",
                           "text": (f"PERDE em fidelity: média {means[i]:.3f} é significativamente inferior à melhor "
                                    f"{means[best]:.3f} ({_label(meta[best]['config'])}): Δ={detail[i]['delta']:.3f}, "
                                    f"t={detail[i]['t']:.2f} > t_crítico={t_crit:.2f} (α={search.significance_alpha}).")}
        elif i not in stable:
            verdicts[i] = {"status": "LOST", "stage": 2, "reason_code": "LOST_STABILITY",
                           "text": (f"PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ={detail[i]['delta']:.3f}, "
                                    f"t={detail[i]['t']:.2f} ≤ {t_crit:.2f}), mas desvio-padrão {stds[i]:.3f} > "
                                    f"{min_std:.3f} + tolerância {search.stability_tolerance}.")}
        elif i not in structural:
            verdicts[i] = {"status": "LOST", "stage": 3, "reason_code": "LOST_STRUCTURAL_STABILITY",
                           "text": (f"PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de "
                                    f"instabilidade estrutural {struct[i]:.2f} (CV de nós {_cv(N[i]):.2f}; nós "
                                    f"{int(N[i].min())}–{int(N[i].max())}) > {min_struct:.2f} + tolerância {tol_s}.")}
        elif i not in simplest:
            verdicts[i] = {"status": "LOST", "stage": 4, "reason_code": "LOST_COMPLEXITY",
                           "text": (f"PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com "
                                    f"{nodes[i]:.1f} nós médios vs {min_nodes:.1f} de {wl}.")}
        else:
            verdicts[i] = {"status": "LOST", "stage": 5, "reason_code": "LOST_TIE",
                           "text": (f"PERDE o desempate: empate real em fidelity, estabilidades e nós; "
                                    f"prevalece a configuração mais simples ({wl}).")}
        verdicts[i]["structural_instability"] = struct[i]
        cens = bool(meta[i].get("censored"))
        verdicts[i]["structural_stability_censored"] = cens
        if cens:
            verdicts[i]["text"] += (" ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós "
                                    f"(max_nodes={meta[i]['max_nodes']}): a baixa variância pode vir do limite, não dos dados; "
                                    "o CV de nós baixo não é evidência forte de estabilidade estrutural.")
    return winner, verdicts, {"best_fidelity_candidate": best, "t_critical": t_crit, "n_splits": J,
                              "indistinguishable": indist, "stable": stable, "structural": structural,
                              "simplest": simplest, "structural_instability": struct}


def _tree_structure(model, feature_scale) -> dict:
    """Resumo semântico/estrutural da árvore ajustada: features da raiz, features usadas e assinaturas dos splits.

    A assinatura de um split é (feature, intervalo do limiar discretizado em meios desvios-padrão da feature no
    treino): dois splits do mesmo atributo com limiares próximos contam como a mesma regra. Só diagnóstico.
    """
    root = model.root_
    root_features = [] if root.is_leaf else sorted({int(l.feature) for l in root.test.literals})
    features, splits = set(), set()
    for _node, test in model.iter_splits():
        for lit in test.literals:
            j = int(lit.feature)
            features.add(j)
            step = 0.5 * float(feature_scale[j]) if j < len(feature_scale) and feature_scale[j] > 0 else 1.0
            splits.add((j, int(np.floor(float(lit.threshold) / step))))
    return {"root_features": root_features, "features": sorted(features),
            "splits": sorted([list(sp) for sp in splits])}


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return 1.0 if not a and not b else float(len(a & b) / len(a | b))


def _structure_diagnostics(rows, feature_names) -> dict:
    """Estabilidade semântica/estrutural entre as árvores de um candidato (diagnóstico; não entra na seleção)."""
    names = list(feature_names)

    def nm(j):
        return str(names[j]) if j < len(names) else f"x{j}"

    n = len(rows)
    root_counts, use_counts = {}, {}
    for r in rows:
        for j in r.get("root_features", []):
            root_counts[nm(j)] = root_counts.get(nm(j), 0) + 1
        for j in r.get("features", []):
            use_counts[nm(j)] = use_counts.get(nm(j), 0) + 1
    fsets = [tuple(r.get("features", [])) for r in rows]
    ssets = [tuple(tuple(sp) for sp in r.get("splits", [])) for r in rows]
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    feat_j = [_jaccard(fsets[a], fsets[b]) for a, b in pairs]
    split_j = [_jaccard(ssets[a], ssets[b]) for a, b in pairs]
    same = [_jaccard(fsets[a], fsets[b]) for a, b in pairs if rows[a]["nodes"] == rows[b]["nodes"]]
    same_split = [_jaccard(ssets[a], ssets[b]) for a, b in pairs if rows[a]["nodes"] == rows[b]["nodes"]]
    root_top = sorted(root_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "n_trees": int(n),
        "root_feature_freq": {k: v / n for k, v in root_top},
        "root_feature_modal_share": float(root_top[0][1] / n) if root_top else 0.0,
        "feature_usage_freq": {k: v / n for k, v in sorted(use_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]},
        "feature_set_jaccard_mean": float(np.mean(feat_j)) if feat_j else 1.0,
        "split_signature_jaccard_mean": float(np.mean(split_j)) if split_j else 1.0,
        "same_size_pairs": int(len(same)),
        "same_size_feature_jaccard_mean": float(np.mean(same)) if same else None,
        "same_size_split_jaccard_mean": float(np.mean(same_split)) if same_split else None,
        "stump_fraction": float(np.mean([1.0 if r["nodes"] <= 3 else 0.0 for r in rows])),
    }


def _split_stats(rows, candidate, repeats: int, feature_names=(), search: Optional["ScientificTrepanSearchConfig"] = None):
    """Estatísticas de um candidato a partir das linhas por partição (todas calculadas só no treino)."""
    fid = np.array([r["fidelity"] for r in rows], dtype=float)
    nodes = np.array([r["nodes"] for r in rows], dtype=float)
    per_repeat = [float(np.mean([r["fidelity"] for r in rows if r["repeat"] == i])) for i in range(repeats)]
    within = [float(np.std([r["fidelity"] for r in rows if r["repeat"] == i], ddof=1))
              for i in range(repeats) if sum(1 for r in rows if r["repeat"] == i) > 1]
    return {
        "n_splits": int(len(rows)),
        "fidelity_mean": float(fid.mean()), "fidelity_std": float(fid.std(ddof=1)) if len(fid) > 1 else 0.0,
        "fidelity_min": float(fid.min()), "fidelity_max": float(fid.max()),
        "fidelity_per_seed_mean": per_repeat,
        "between_seed_std": float(np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else 0.0,
        "within_seed_fold_std_mean": float(np.mean(within)) if within else 0.0,
        "nodes_mean": float(nodes.mean()), "nodes_std": float(nodes.std(ddof=1)) if len(nodes) > 1 else 0.0,
        "nodes_min": float(nodes.min()), "nodes_max": float(nodes.max()),
        "nodes_cv": float(nodes.std(ddof=1) / nodes.mean()) if len(nodes) > 1 and nodes.mean() > 0 else 0.0,
        "depth_mean": float(np.mean([r["depth"] for r in rows])),
        "depth_max": float(np.max([r["depth"] for r in rows])),
        "depth_std": float(np.std([r["depth"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
        "depth_cv": _cv([r["depth"] for r in rows]),
        "leaves_mean": float(np.mean([r["leaves"] for r in rows])),
        "leaves_std": float(np.std([r["leaves"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
        "leaves_cv": _cv([r["leaves"] for r in rows]),
        "structural_instability": structural_instability([r["nodes"] for r in rows], [r["depth"] for r in rows],
                                                         [r["leaves"] for r in rows]),
        "max_nodes": int(candidate.max_nodes),
        "node_cap_reached_count": int(sum(1 for r in rows if r.get("max_nodes_reached"))),
        "fraction_at_node_cap": float(np.mean([1.0 if r.get("max_nodes_reached") else 0.0 for r in rows])),
        "structural_stability_evidence": "censored_by_node_cap" if float(np.mean([1.0 if r.get("max_nodes_reached") else 0.0 for r in rows]))
        >= float((search or ScientificTrepanSearchConfig()).node_cap_censoring_threshold) else "observed",
        "structural_stability_censored": bool(
            float(np.mean([1.0 if r.get("max_nodes_reached") else 0.0 for r in rows]))
            >= float((search or ScientificTrepanSearchConfig()).node_cap_censoring_threshold)),
        "time_mean_s": float(np.mean([r["time_s"] for r in rows])), "time_total_s": float(np.sum([r["time_s"] for r in rows])),
        "query_budget": int(candidate.max_queries),
        "queries_used_mean": float(np.mean([r["queries_used"] for r in rows])),
        "queries_used_max": int(np.max([r["queries_used"] for r in rows])),
        "budget_exhausted_count": int(sum(1 for r in rows if r.get("budget_exhausted"))),
        "budget_exhausted_fraction": float(np.mean([1.0 if r.get("budget_exhausted") else 0.0 for r in rows])),
        "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in rows])),
        "macro_f1_mean": float(np.mean([r["macro_f1"] for r in rows])),
        "structure_diagnostics": _structure_diagnostics(rows, feature_names),
    }


def _wilson(k: int, n: int, z: float = 1.959964):
    if n <= 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def _weighted_percentile(values, weights, q):
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cum = np.cumsum(w) / w.sum()
    return float(v[min(int(np.searchsorted(cum, q / 100.0)), len(v) - 1)])


def _block_bootstrap(F, N, D, L, meta, k, search, valid, plan, repeats, labels, final_label, stats):
    """Robustez interna da política de seleção por bootstrap de blocos das repetições da CV.

    Cada repetição (com as suas ``k`` dobras) é um bloco; reamostram-se ``repeats`` blocos com reposição e em cada
    reamostragem corre-se EXATAMENTE a mesma ``_lexicographic_select`` sobre resultados já calculados (nada é treinado de
    novo; o teste nunca entra). A política só depende do MULTISET de blocos (médias, desvios e testes emparelhados são
    invariantes à ordem das colunas), logo:

    - ``exact``: enumeram-se todos os multisets (C(2B-1, B)) com o seu peso multinomial, o que equivale a avaliar as
      B^B reamostragens ordenadas (3125 para B=5) com apenas C(2B-1,B) avaliações (126 para B=5);
    - ``monte_carlo``: se C(2B-1,B) > ``bootstrap_exact_limit``, ``bootstrap_resamples`` reamostragens determinísticas.

    Distingue a configuração escolhida pela CV completa da moda do bootstrap: ``selection_probability`` é a da escolhida;
    ``bootstrap_runner_up`` é o 2.º da distribuição (nunca com probabilidade superior ao 1.º).
    """
    import itertools
    import math
    B = int(repeats)
    out = {"method": None, "bootstrap_samples": 0, "bootstrap_evaluations": 0, "n_blocks": B, "assessable": False, "seed": None,
           "selected_config_full_cv": final_label, "selected_config_probability": 0.0, "selection_probability": 0.0,
           "bootstrap_modal_config": None, "bootstrap_modal_probability": None, "bootstrap_runner_up": None,
           "top1_top2_margin": None, "selected_is_modal": None, "full_cv_vs_modal_gap": None,
           "full_cv_selection_fragile": None, "selection_probability_mc_interval": None, "distribution": {},
           "winner_fidelity_ci": None, "test_used": False}
    if B < 2:
        out["reason"] = "menos de 2 repetições (blocos)"
        return out
    blocks = [[j for j, (rr, *_r) in enumerate(plan) if rr == r] for r in range(B)]
    final_idx = labels.index(final_label)
    block_fid = np.array([float(F[final_idx, blk].mean()) for blk in blocks])
    n_multisets = math.comb(2 * B - 1, B)
    exact = n_multisets <= int(search.bootstrap_exact_limit)
    cache: dict = {}

    def evaluate(multiset):
        key = tuple(multiset)
        if key not in cache:
            cols = [j for p in key for j in blocks[p]]
            w, _v, _i = _lexicographic_select(F[:, cols], N[:, cols], meta, k, search, valid, D[:, cols], L[:, cols])
            cache[key] = None if w is None else labels[w]
        return cache[key]

    wins: dict = {}
    fid_vals, fid_w = [], []
    if exact:
        total = float(B ** B)
        for ms in itertools.combinations_with_replacement(range(B), B):
            counts = np.bincount(ms, minlength=B)
            weight = math.factorial(B) / float(np.prod([math.factorial(int(c)) for c in counts]))
            lab = evaluate(ms)
            if lab is None:
                continue
            wins[lab] = wins.get(lab, 0.0) + weight
            fid_vals.append(float(counts @ block_fid / B)); fid_w.append(weight)
        n_total = sum(wins.values())
        samples = int(round(total))
        out.update({"method": "exact", "bootstrap_samples": samples})
    else:
        seed = int(search.bootstrap_seed) if search.bootstrap_seed is not None else int(plan[0][1]) + 7919
        rng = np.random.default_rng(seed)
        n = int(search.bootstrap_resamples)
        for _ in range(n):
            ms = tuple(sorted(int(x) for x in rng.integers(0, B, size=B)))
            lab = evaluate(ms)
            if lab is None:
                continue
            wins[lab] = wins.get(lab, 0.0) + 1.0
            counts = np.bincount(ms, minlength=B)
            fid_vals.append(float(counts @ block_fid / B)); fid_w.append(1.0)
        n_total = sum(wins.values())
        out.update({"method": "monte_carlo", "bootstrap_samples": int(n_total), "seed": seed})
    out["bootstrap_evaluations"] = int(len(cache))
    if not n_total:
        out["reason"] = "nenhuma reamostragem avaliável"
        return out
    order = sorted(wins.items(), key=lambda kv: (-kv[1], labels.index(kv[0])))
    dist = {lab: c / n_total for lab, c in order}
    modal, p_modal = order[0][0], order[0][1] / n_total
    runner = {"label": order[1][0], "probability": order[1][1] / n_total} if len(order) > 1 else None
    p_final = wins.get(final_label, 0.0) / n_total
    out.update({
        "assessable": True, "selected_config_probability": float(p_final), "selection_probability": float(p_final),
        "bootstrap_modal_config": modal, "bootstrap_modal_probability": float(p_modal), "bootstrap_runner_up": runner,
        "top1_top2_margin": float(p_modal - (runner["probability"] if runner else 0.0)),
        "selected_is_modal": bool(modal == final_label), "full_cv_vs_modal_gap": float(p_modal - p_final),
        "full_cv_selection_fragile": bool(modal != final_label), "distribution": dist,
        "selection_probability_mc_interval": None if exact else _wilson(int(wins.get(final_label, 0)), int(n_total)),
        "winner_fidelity_ci": [_weighted_percentile(fid_vals, fid_w, 2.5), _weighted_percentile(fid_vals, fid_w, 97.5)] if fid_vals else None,
    })
    return out


def _overall_status(selections) -> str:
    if not selections:
        return "not_assessed"
    statuses = [sel["status"] for sel in selections]
    if "tuning_uncertain" in statuses:
        return "tuning_uncertain"
    return "stable_exact" if all(st == "stable_exact" for st in statuses) else "stable_equivalent_set"


def _budget_check(history) -> dict:
    """O orçamento deixou de ser variável de confusão? Mostra orçamento comum, consumo e esgotamentos por candidato."""
    rows = [h for h in history if h.get("stats")]
    budgets = sorted({int(h["stats"]["query_budget"]) for h in rows})
    return {"common_budget": budgets[0] if len(budgets) == 1 else None, "budgets": budgets,
            "all_candidates_same_budget": len(budgets) == 1,
            "any_budget_exhausted": any(h["stats"]["budget_exhausted_count"] > 0 for h in rows),
            "max_queries_used": max((h["stats"]["queries_used_max"] for h in rows), default=0)}


def tune_scientific_trepan(
    X_train,
    y_train,
    *,
    oracle,
    feature_names: Sequence[str],
    base_config: ControlledTrepanConfig,
    semantic_feature_weights=None,
    semantic_feature_groups=None,
    semantic_relatedness_matrix=None,
    query_projector=None,
    search: ScientificTrepanSearchConfig = ScientificTrepanSearchConfig(),
    ontology_graph=None,
    semantic_feature_entities=None,
) -> dict[str, Any]:
    X=np.asarray(X_train,dtype=float); y=np.asarray(y_train)
    if X.ndim != 2 or len(X)!=len(y):
        raise ValueError("X_train/y_train inválidos para tuning científico TREPAN.")
    # Garantia de isolamento: esta função NÃO recebe teste (assinatura sem X_test/y_test) e só particiona o treino.
    plan, folds, seeds = _cv_plan(y, search, base_config.random_state)
    splits = [(tr, va) for (r, _seed, _f, tr, va) in plan if r == 0]      # etapa semântica: 1.ª repetição
    repeats = len(seeds)
    feature_scale = np.nan_to_num(X.std(axis=0), nan=0.0)

    def evaluate(candidates, label_stage):
        history = []
        for candidate in candidates:
            rows = []; failed = None
            for (r, seed, f, tr, va) in plan:
                try:
                    kwargs = candidate.common_tree_kwargs()
                    kwargs["random_state"] = int(seed)          # a aleatoriedade das queries varia com a seed da repetição
                    t0 = time.perf_counter()
                    model = TrepanOriginalClassifier(**kwargs).fit(X[tr], oracle=oracle, feature_names=feature_names)
                    elapsed = time.perf_counter() - t0
                    obj = _objective(model, X[va], y[va], oracle, candidate, search)
                    rows.append({"repeat": r, "seed": int(seed), "fold": f, **obj,
                                 "nodes": int(getattr(model, "node_count_", 0)), "depth": int(model.get_depth()),
                                 "leaves": int(model.get_n_leaves()), "time_s": float(elapsed),
                                 "queries_used": int(getattr(model, "membership_queries_", 0)),
                                 "budget_exhausted": bool(getattr(model, "query_budget_exhausted_", False)),
                                 "max_nodes_reached": bool(getattr(model, "max_nodes_reached_", False)),
                                 **_tree_structure(model, feature_scale)})
                except (ValueError, RuntimeError) as exc:
                    failed = str(exc); break
            if rows and failed is None:
                mean = _mean_metrics(rows)
                stats = _split_stats(rows, candidate, repeats, feature_names, search)
            else:
                mean = {"score": float("-inf"), "fidelity": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0, "complexity": 1.0}
                stats = None
            history.append({"config": asdict(candidate), "label": _label(asdict(candidate)), "stage": label_stage,
                            "mean": mean, "stats": stats, "per_split": rows if failed is None else [], "failed": failed})
        return history

    def select(history, fallback, with_bootstrap=True):
        """Seleção lexicográfica + estabilidade da seleção entre repetições (seeds). Escreve veredictos no histórico."""
        valid = [bool(h["stats"]) for h in history]
        if not any(valid):
            for h in history:
                h["verdict"] = {"status": "FAILED", "stage": 0, "reason_code": "FIT_FAILED", "text": "falhou na CV."}
            return fallback, {"selected": asdict(fallback), "fallback": "all_candidates_failed", "stable": False,
                              "status": "tuning_uncertain", "selection_probability": 0.0, "per_repeat_winners": [],
                              "threshold": float(search.min_selection_probability)}
        J = len(plan)
        F = np.array([[r["fidelity"] for r in h["per_split"]] if h["stats"] else [0.0] * J for h in history], dtype=float)
        Nn = np.array([[r["nodes"] for r in h["per_split"]] if h["stats"] else [0.0] * J for h in history], dtype=float)
        Dd = np.array([[r["depth"] for r in h["per_split"]] if h["stats"] else [0.0] * J for h in history], dtype=float)
        Ll = np.array([[r["leaves"] for r in h["per_split"]] if h["stats"] else [0.0] * J for h in history], dtype=float)
        meta = [{"config": h["config"], "max_nodes": int(h["config"]["max_nodes"]),
                 "censored": bool(h["stats"] and h["stats"]["structural_stability_censored"])} for h in history]
        winner, verdicts, info = _lexicographic_select(F, Nn, meta, folds, search, valid, Dd, Ll)
        for h, v in zip(history, verdicts):
            h["verdict"] = v
        final = history[winner]["label"]
        # Vencedora por repetição: só informativa (3 dobras por repetição são poucas para medir estabilidade).
        per_repeat = []
        for r in range(repeats):
            cols = [j for j, (rr, *_rest) in enumerate(plan) if rr == r]
            w, _v, _i = _lexicographic_select(F[:, cols], Nn[:, cols], meta, folds, search, valid, Dd[:, cols], Ll[:, cols])
            per_repeat.append(None if w is None else history[w]["label"])
        if with_bootstrap:
            boot = _block_bootstrap(F, Nn, Dd, Ll, meta, folds, search, valid, plan, repeats,
                                    [h["label"] for h in history], final, [h["stats"] for h in history])
        else:
            boot = {"assessable": False, "selection_probability": 0.0, "method": None, "deferred": True}
        w_stats = history[winner]["stats"]
        labels = [h["label"] for h in history]
        # --- Conjunto equivalente: candidatos estatisticamente indistinguíveis da vencedora em fidelity (nos dois sentidos).
        eq_ids = []
        for i, h in enumerate(history):
            if not valid[i]:
                continue
            if i == winner:
                eq_ids.append(h["label"]); continue
            d = F[winner] - F[i]
            a_ = _nb_paired(d, folds, search.significance_alpha); b_ = _nb_paired(-d, folds, search.significance_alpha)
            if not a_["significant"] and not b_["significant"]:
                eq_ids.append(h["label"])
        dist = boot.get("distribution") or {}
        eq_prob = float(sum(dist.get(l, 0.0) for l in eq_ids)) if boot["assessable"] else None
        # --- Comportamento real da árvore escolhida (C): instabilidade estrutural e alternância tronco/árvore.
        diag = w_stats.get("structure_diagnostics") or {}
        stump_frac = float(diag.get("stump_fraction", 0.0))
        behavior_unstable = bool(float(w_stats["structural_instability"]) > float(search.behavior_instability_threshold)
                                 or 0.0 < stump_frac < 1.0)
        tree_behavior = {"fidelity_std": w_stats["fidelity_std"], "between_seed_std": w_stats["between_seed_std"],
                         "structural_instability": w_stats["structural_instability"], "stump_fraction": stump_frac,
                         "structural_stability_evidence": w_stats["structural_stability_evidence"],
                         "behavior_unstable": behavior_unstable}
        exact_ok = bool(boot["assessable"] and boot["selected_is_modal"]
                        and boot["selection_probability"] >= float(search.min_selection_probability))
        family_ok = bool(boot["assessable"] and eq_prob is not None and eq_prob >= float(search.min_equivalent_set_probability)
                         and not behavior_unstable)
        if not boot["assessable"]:
            status, status_reason = "tuning_uncertain", "bootstrap_not_assessable"
        elif exact_ok and not behavior_unstable:
            status, status_reason = "stable_exact", "ok"
        elif family_ok:
            status = "stable_equivalent_set"
            status_reason = ("exact_hyperparameter_unstable_but_equivalent_family_stable" if not exact_ok else "ok")
        else:
            status = "tuning_uncertain"
            status_reason = ("tree_behavior_unstable" if behavior_unstable else
                             "full_cv_selection_not_bootstrap_modal" if not boot["selected_is_modal"] else
                             "equivalent_family_not_stable" if (eq_prob is not None and eq_prob < float(search.min_equivalent_set_probability)) else
                             "selection_probability_below_threshold")
        stable = status != "tuning_uncertain"
        selection = {"selected": history[winner]["config"], "selected_label": final,
                     "selected_config_full_cv": final, "selected_config_probability": boot.get("selected_config_probability", 0.0),
                     "exact_selection_probability": boot.get("selected_config_probability", 0.0),
                     "bootstrap_modal_config": boot.get("bootstrap_modal_config"),
                     "bootstrap_modal_probability": boot.get("bootstrap_modal_probability"),
                     "bootstrap_runner_up": boot.get("bootstrap_runner_up"), "top1_top2_margin": boot.get("top1_top2_margin"),
                     "full_cv_selection_fragile": boot.get("full_cv_selection_fragile"),
                     "equivalent_set_covers_all_candidates": bool(len(eq_ids) == int(sum(1 for v in valid if v))),
                     "equivalent_candidate_set": {"ids": eq_ids, "count": len(eq_ids), "probability": eq_prob},
                     "equivalent_candidate_ids": eq_ids, "equivalent_candidate_count": len(eq_ids), "equivalent_set_probability": eq_prob,
                     "tree_behavior": tree_behavior,
                     "per_repeat_winners": per_repeat, "seeds": [int(v) for v in seeds],
                     "bootstrap": boot, "selection_probability": boot["selection_probability"],
                     "stable": stable, "stable_exact": status == "stable_exact", "family_stable": status == "stable_equivalent_set",
                     "status": status, "status_reason": status_reason,
                     "distinct_winners": len({w for w in per_repeat if w}),
                     "threshold": float(search.min_selection_probability),
                     "equivalent_set_threshold": float(search.min_equivalent_set_probability),
                     "node_cap": {"max_nodes": w_stats["max_nodes"], "fraction_at_node_cap": w_stats["fraction_at_node_cap"],
                                  "node_cap_reached_count": w_stats["node_cap_reached_count"],
                                  "structural_stability_censored": w_stats["structural_stability_censored"],
                                  "structural_stability_evidence": w_stats["structural_stability_evidence"]},
                     "t_critical": info.get("t_critical"), "n_splits": info.get("n_splits")}
        return ControlledTrepanConfig(**history[winner]["config"]), selection

    # Etapa 1: estrutura (purity_epsilon x max_nodes), só no treino, CV repetida. Sem grelha ou com
    # ``tune_structure=False`` ficam os valores canónicos da base.
    structure_history = []
    structure_selection = None
    structure_base = base_config
    expansion = {"enabled": bool(search.capacity_expansion), "initial_node_grid": [], "final_node_grid": [],
                 "capacity_expansion_rounds": 0, "expansion_triggered": False, "expansion_stop_reason": "structure_not_tuned",
                 "fraction_at_node_cap": None, "safety_limit": int(search.max_nodes_safety_limit),
                 "factor": float(search.capacity_expansion_factor),
                 "saturation_threshold": float(search.capacity_expansion_saturation_threshold), "steps": [],
                 "last_supported_max_nodes": None, "interpretation": None}
    if search.tune_structure and search.purity_epsilon_grid and search.max_nodes_grid:
        structure_history = evaluate(_structure_candidates(base_config, search), "structure")
        expansion.update(run_capacity_expansion(
            structure_history,
            lambda nodes: evaluate(_expansion_candidates(base_config, search, nodes, [h["config"] for h in structure_history]), "structure"),
            search, folds))
        structure_base, structure_selection = select(structure_history, base_config)
        expansion["fraction_at_node_cap"] = (structure_selection.get("node_cap") or {}).get("fraction_at_node_cap")

    # Etapa 2: restantes eixos de capacidade à volta da estrutura escolhida. O nº de nós só varia na grelha da
    # etapa 1 (ou fica o da base, canónico, se a grelha estiver desligada): não há crescimento arbitrário de nós.
    capacity_selection = None
    if structure_history and int(search.max_capacity_candidates) <= 1:
        capacity_history = structure_history              # sem eixos adicionais: a vencedora da etapa 1 é a comum
        common = structure_base
    else:
        capacity_history = evaluate(_capacity_candidates(
            structure_base, X.shape[1], search.max_capacity_candidates, grow_nodes=False), "capacity")
        common, capacity_selection = select(capacity_history, structure_base)

    semantic_history=[]
    weights=np.ones(X.shape[1],dtype=float) if semantic_feature_weights is None else np.asarray(semantic_feature_weights,dtype=float)
    for candidate in _semantic_candidates(common, search.max_semantic_candidates):
        rows=[]; failed=None; usage=[]
        for tr,va in splits:
            try:
                model=TrepanReloadedClassifier(
                    **candidate.common_tree_kwargs(),
                    semantic_gain_strength=candidate.semantic_gain_strength,
                    semantic_group_strength=candidate.semantic_group_strength,
                    semantic_candidate_budget=candidate.semantic_candidate_budget,
                    semantic_relation_threshold=candidate.semantic_relation_threshold,
                    semantic_active_query_fraction=candidate.semantic_active_query_fraction,
                    semantic_active_pool_multiplier=candidate.semantic_active_pool_multiplier,
                    error_focused_refinement=candidate.error_focused_refinement,
                    error_focus_min_disagreement=candidate.error_focus_min_disagreement,
                    error_focus_strength=candidate.error_focus_strength,
                    error_focus_semantic_weight=candidate.error_focus_semantic_weight,
                    error_focus_uncertainty_weight=candidate.error_focus_uncertainty_weight,
                    error_focus_min_local_fidelity_gain=candidate.error_focus_min_local_fidelity_gain,
                    error_focus_min_real_fidelity_gain=candidate.error_focus_min_real_fidelity_gain,
                    error_focus_anchor_k=candidate.error_focus_anchor_k,
                    error_focus_top_k=candidate.error_focus_top_k,
                    error_focus_min_regions=candidate.error_focus_min_regions,
                    mirror_when_no_semantic_effect=candidate.mirror_when_no_semantic_effect,
                    alpha=candidate.alpha,
                    beta=candidate.beta,
                    gain_criterion=candidate.gain_criterion,
                    semantic_query_projection=candidate.semantic_query_projection,
                    error_focus_fidelity_tolerance=candidate.error_focus_fidelity_tolerance,
                ).fit(
                    X[tr], oracle=oracle, feature_names=feature_names,
                    semantic_feature_weights=weights,
                    semantic_feature_groups=semantic_feature_groups,
                    semantic_relatedness_matrix=semantic_relatedness_matrix,
                    query_projector=query_projector,
                    ontology_graph=ontology_graph,
                    semantic_feature_entities=semantic_feature_entities,
                )
                rows.append(_objective(model,X[va],y[va],oracle,candidate,search))
                usage.append(float(model.semantic_audit_summary_.get("ontology_usage_rate",0.0)))
            except (ValueError, RuntimeError) as exc:
                failed=str(exc); break
        if rows and failed is None:
            mean=_mean_metrics(rows); mean["ontology_usage_rate"]=float(np.mean(usage)) if usage else 0.0
            # pequeno desempate a favor de semântica realmente utilizada, sem
            # substituir fidelidade/performance pelo simples uso da ontologia.
            mean["selection_score"] = float(mean["score"] + 0.01 * mean["ontology_usage_rate"])
        else:
            mean={"score":float("-inf"),"selection_score":float("-inf"),"fidelity":0.0,"balanced_accuracy":0.0,"macro_f1":0.0,"complexity":1.0,"ontology_usage_rate":0.0}
        semantic_history.append({"config":asdict(candidate),"mean":mean,"failed":failed})
    valid_sem=[r for r in semantic_history if np.isfinite(r["mean"]["selection_score"])]
    selected=ControlledTrepanConfig(**max(valid_sem,key=lambda r:r["mean"]["selection_score"])["config"]) if valid_sem else common
    return {
        "selection_scope":"training_cv_only",
        "test_used_for_selection":False,
        "cv_folds":int(folds),
        "cv_repeats":int(repeats),
        "search_config":asdict(search),
        "common_capacity":asdict(common),
        "selected_config":asdict(selected),
        "cv_plan":{"scheme":"RepeatedStratifiedKFold (conjunto de treino apenas)","folds":int(folds),"repeats":int(repeats),
                   "seeds":[int(v) for v in seeds],"n_splits":len(plan),"test_set_used":False},
        "selection_rule":"lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples)",
        "stability_rule":"block bootstrap das repetições da CV (mesma política em cada reamostragem); tuning_stable se selection_probability >= min_selection_probability",
        "query_budget_policy":str(search.query_budget_policy),
        "budget_check":_budget_check(structure_history),
        "capacity_expansion":expansion,
        "initial_node_grid":expansion["initial_node_grid"], "final_node_grid":expansion["final_node_grid"],
        "capacity_expansion_rounds":expansion["capacity_expansion_rounds"], "expansion_triggered":expansion["expansion_triggered"],
        "expansion_stop_reason":expansion["expansion_stop_reason"], "fraction_at_node_cap":expansion["fraction_at_node_cap"],
        "expansion_steps":expansion["steps"], "capacity_gain_supported":bool(expansion["steps"] and expansion["steps"][-1]["capacity_gain_supported"]),
        "structure_history":structure_history,
        "structure_selection":structure_selection,
        "structure_selected":{"purity_epsilon":float(structure_base.purity_epsilon),"max_nodes":int(structure_base.max_nodes)},
        "capacity_selection":capacity_selection,
        "tuning_stable":bool(all(sel["stable"] for sel in (structure_selection, capacity_selection) if sel)) if (structure_selection or capacity_selection) else True,
        "tuning_status":_overall_status([sel for sel in (structure_selection, capacity_selection) if sel]),
        "capacity_history":capacity_history,
        "semantic_history":semantic_history,
    }


__all__=["ScientificTrepanSearchConfig","tune_scientific_trepan"]
