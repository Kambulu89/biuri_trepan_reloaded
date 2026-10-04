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
    bootstrap_resamples: int = 200
    bootstrap_seed: Optional[int] = None      # None -> derivada da seed base (determinística)
    min_selection_probability: float = 0.6    # abaixo disto o tuning é marcado "incerto" (limiar configurável, não prova)
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


def _structure_candidates(base: ControlledTrepanConfig, search: "ScientificTrepanSearchConfig"):
    """Grelha purity_epsilon x max_nodes. O primeiro candidato é sempre o ponto canónico da base.

    Equidade computacional (``query_budget_policy``):
    - "non_binding" (omissão): TODOS os candidatos recebem o mesmo orçamento comum, calculado para o maior
      ``max_nodes`` da grelha com ``non_binding_query_budget`` (majorante do consumo possível). O orçamento nunca
      limita nenhum candidato, logo não pode explicar diferenças entre eles.
    - "required": cada candidato recebe o orçamento necessário para os seus próprios nós (pode esgotar).
    """
    cap = max(int(base.max_queries), DEFAULT_QUERY_BUDGET_CAP)
    pairs = [(float(base.purity_epsilon), int(base.max_nodes))]
    for nodes in search.max_nodes_grid:
        for eps in search.purity_epsilon_grid:
            key = (float(eps), int(nodes))
            if key not in pairs:
                pairs.append(key)
    common = None
    if str(search.query_budget_policy) == "non_binding":
        common = max(int(base.max_queries), non_binding_query_budget(max(n for _e, n in pairs), base.min_sample))

    def point(eps, nodes):
        c = replace(base, purity_epsilon=float(eps), max_nodes=int(nodes), max_depth=max(int(base.max_depth), 6))
        if common is not None:
            return replace(c, max_queries=common)
        need = required_query_budget(c.max_nodes, c.min_sample, cap=cap)
        return replace(c, max_queries=max(int(base.max_queries), need))

    return [point(eps, nodes) for eps, nodes in pairs]


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
                                    f"(max_nodes={meta[i]['max_nodes']}): a baixa variância pode vir do limite, não dos dados.")
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


def _block_bootstrap(F, N, D, L, meta, k, search, valid, plan, repeats, labels, final_label, stats):
    """Robustez interna da política de seleção por block bootstrap das repetições da CV.

    Cada repetição (com as suas ``k`` dobras) é um bloco; reamostram-se ``repeats`` blocos com reposição, de forma
    determinística, e em cada reamostragem corre-se EXATAMENTE a mesma ``_lexicographic_select``. Só usa resultados
    de CV do treino (o teste nunca entra). Com menos de 2 blocos não é avaliável.
    """
    B = int(search.bootstrap_resamples)
    out = {"method": "block bootstrap das repetições da CV (dobras de cada repetição preservadas)",
           "n_blocks": int(repeats), "n_resamples": 0, "assessable": False, "seed": None,
           "selection_probability": 0.0, "selection_probability_mc_interval": [0.0, 1.0],
           "runner_up": None, "margin": None, "distribution": {}, "modal_config": None,
           "selected_is_modal": None, "winner_fidelity_ci": None, "test_used": False}
    if repeats < 2 or B < 1:
        out["reason"] = "menos de 2 repetições (blocos)" if repeats < 2 else "bootstrap_resamples < 1"
        return out
    seed = int(search.bootstrap_seed) if search.bootstrap_seed is not None else int(plan[0][1]) + 7919
    rng = np.random.default_rng(seed)
    blocks = [[j for j, (rr, *_r) in enumerate(plan) if rr == r] for r in range(repeats)]
    wins: dict = {}
    winner_fid = []
    final_idx = labels.index(final_label)
    for _ in range(B):
        picks = rng.integers(0, repeats, size=repeats)
        cols = [j for p in picks for j in blocks[p]]
        w, _v, _i = _lexicographic_select(F[:, cols], N[:, cols], meta, k, search, valid, D[:, cols], L[:, cols])
        if w is None:
            continue
        wins[labels[w]] = wins.get(labels[w], 0) + 1
        winner_fid.append(float(F[final_idx, cols].mean()))
    n = sum(wins.values())
    dist = {lab: c / n for lab, c in sorted(wins.items(), key=lambda kv: (-kv[1], labels.index(kv[0])))} if n else {}
    p_final = wins.get(final_label, 0) / n if n else 0.0
    others = [(lab, c) for lab, c in sorted(wins.items(), key=lambda kv: (-kv[1], labels.index(kv[0]))) if lab != final_label]
    runner = {"label": others[0][0], "probability": others[0][1] / n} if others and n else None
    out.update({
        "assessable": bool(n), "n_resamples": int(n), "seed": seed, "selection_probability": float(p_final),
        "selection_probability_mc_interval": _wilson(wins.get(final_label, 0), n),
        "runner_up": runner, "margin": float(p_final - (runner["probability"] if runner else 0.0)),
        "distribution": dist, "modal_config": next(iter(dist), None), "selected_is_modal": bool(next(iter(dist), None) == final_label),
        "winner_fidelity_ci": [float(np.percentile(winner_fid, 2.5)), float(np.percentile(winner_fid, 97.5))] if winner_fid else None,
    })
    return out


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

    def select(history, fallback):
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
        boot = _block_bootstrap(F, Nn, Dd, Ll, meta, folds, search, valid, plan, repeats,
                                [h["label"] for h in history], final, [h["stats"] for h in history])
        stable = bool(boot["assessable"] and boot["selection_probability"] >= float(search.min_selection_probability))
        w_stats = history[winner]["stats"]
        selection = {"selected": history[winner]["config"], "selected_label": final,
                     "per_repeat_winners": per_repeat, "seeds": [int(v) for v in seeds],
                     "bootstrap": boot, "selection_probability": boot["selection_probability"],
                     "stable": stable, "status": "tuning_stable" if stable else "tuning_uncertain",
                     "distinct_winners": len({w for w in per_repeat if w}),
                     "threshold": float(search.min_selection_probability),
                     "node_cap": {"max_nodes": w_stats["max_nodes"], "fraction_at_node_cap": w_stats["fraction_at_node_cap"],
                                  "node_cap_reached_count": w_stats["node_cap_reached_count"],
                                  "structural_stability_censored": w_stats["structural_stability_censored"]},
                     "t_critical": info.get("t_critical"), "n_splits": info.get("n_splits")}
        return ControlledTrepanConfig(**history[winner]["config"]), selection

    # Etapa 1: estrutura (purity_epsilon x max_nodes), só no treino, CV repetida. Sem grelha ou com
    # ``tune_structure=False`` ficam os valores canónicos da base.
    structure_history = []
    structure_selection = None
    structure_base = base_config
    if search.tune_structure and search.purity_epsilon_grid and search.max_nodes_grid:
        structure_history = evaluate(_structure_candidates(base_config, search), "structure")
        structure_base, structure_selection = select(structure_history, base_config)

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
        "structure_history":structure_history,
        "structure_selection":structure_selection,
        "structure_selected":{"purity_epsilon":float(structure_base.purity_epsilon),"max_nodes":int(structure_base.max_nodes)},
        "capacity_selection":capacity_selection,
        "tuning_stable":bool(all(sel["stable"] for sel in (structure_selection, capacity_selection) if sel)) if (structure_selection or capacity_selection) else True,
        "tuning_status":("tuning_stable" if all(sel["stable"] for sel in (structure_selection, capacity_selection) if sel) else "tuning_uncertain") if (structure_selection or capacity_selection) else "not_assessed",
        "capacity_history":capacity_history,
        "semantic_history":semantic_history,
    }


__all__=["ScientificTrepanSearchConfig","tune_scientific_trepan"]
