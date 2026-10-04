"""Apresentador do ``ExperimentResult`` (Partes 5-16, 23-31, 34). Qt-free e sem cálculo científico.

Transforma o modelo de dados em linhas/tabelas de texto prontas a mostrar. **Não calcula
métricas**: apenas formata valores já presentes e explica a razão quando um valor não existe.
O modo básico/científico muda só o que é mostrado.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.experiment_result import ExperimentResult, Measure, ModelCard, Reason, TreeDiagnostics
from gui.strings import term, tr

BASIC, SCIENTIFIC = "basic", "scientific"
Table = Dict[str, Any]  # {"headers": [...], "rows": [[...]]}

PREDICTIVE_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "precision_macro", "recall_macro")
COMPLEXITY_KEYS = ("nodes", "depth", "leaves", "queries")
TREE_KEYS = ("trepan_original", "trepan_reloaded")


# ---------------------------------------------------------------- valores e N/A
def na_text(reason: Optional[str]) -> str:
    """'Não calculado — professor semântico rejeitado': o N/A nunca é anónimo (Parte 9)."""
    reason = reason or Reason.NOT_REPORTED
    category = Reason.CATEGORY.get(reason, "not_available")
    head = tr(f"na.{category}")
    detail = tr(f"reason.{reason}")
    return head if detail == f"reason.{reason}" else f"{head} — {detail}"


def format_measure(m: Optional[Measure], fmt: str = "{:.3f}", *, integer: bool = False) -> str:
    if m is None:
        return na_text(Reason.NOT_REPORTED)
    if m.value is None:
        return na_text(m.reason)
    return str(int(round(m.value))) if integer else fmt.format(m.value)


def format_optional(value: Any, fmt: str = "{}") -> str:
    return tr("misc.none") if value is None else fmt.format(value)


def fidelity_label_text(kind: Optional[str], value: Optional[float]) -> str:
    """Rótulo de fidelidade para o visualizador legado (valor já em %, nunca recalculado).

    ``oracle``: fidelidade ao Oracle; ``agreement``: concordância com o MLP (C4.5, diagnóstico);
    ``None``: o modelo não tem Oracle -> "Não aplicável", nunca 0.0%.
    """
    if kind == "oracle" and value is not None:
        return f"{term('fidelity')}: {value:.1f}%"
    if kind == "agreement" and value is not None:
        return f"{tr('field.agreement')}: {value:.1f}%"
    return f"{term('fidelity')}: {na_text(Reason.NO_ORACLE)}"


def yes_no(flag: Optional[bool]) -> str:
    return tr("misc.none") if flag is None else tr("status.yes" if flag else "status.no")


def status_text(status: Optional[str]) -> str:
    if not status:
        return tr("misc.none")
    key = f"status.{status}"
    text = tr(key)
    return status if text == key else text


def _model_name(key: Optional[str]) -> str:
    return term(key) if key else tr("misc.none")


# ---------------------------------------------------------------- explicações
def explain_decision(result: ExperimentResult) -> str:
    """Explicação humana do código de decisão com utilidade base, OWL e delta (Parte 8)."""
    enr = result.enrichment
    code = enr.decision or "NOT_EVALUATED"
    key = f"decision.{code}"
    text = tr(key)
    if text == key:
        text = tr("decision.unknown", code=code)
    parts = [text]
    if enr.base_utility.available or enr.owl_utility.available:
        parts.append(
            f"{tr('field.base_utility')}: {format_measure(enr.base_utility)} · "
            f"{tr('field.owl_utility')}: {format_measure(enr.owl_utility)} · "
            f"{tr('field.delta')}: {format_measure(enr.delta_utility, '{:+.3f}')}")
    return " ".join(parts) if len(parts) == 1 else f"{parts[0]} ({parts[1]})"


def _reason_text(code: Optional[str]) -> str:
    """Texto humano para um código de decisão/razão; cai para o próprio código se desconhecido."""
    if not code:
        return tr("misc.none")
    for prefix in ("decision.", "reason."):
        text = tr(prefix + code)
        if text != prefix + code:
            return text
    return str(code)


def _oracle_text(card: Optional[ModelCard]) -> str:
    if card is None:
        return na_text(Reason.TREE_NOT_BUILT)
    if card.oracle:
        return _model_name(card.oracle)
    return na_text(Reason.NO_ORACLE)


# ---------------------------------------------------------------- secções simples
def build_label(build: Optional[Dict[str, Any]]) -> str:
    b = build or {}
    if b.get("label"):
        return str(b["label"])
    if not (b.get("version") or b.get("commit")):
        return tr("misc.none")
    return f"V{b.get('version', '?')} · {b.get('commit', '?')}" + (" (+)" if b.get("dirty") else "")


def build_rows(result: ExperimentResult) -> List[Tuple[str, str]]:
    b = result.provenance.build or {}
    rows = [(tr("field.build"), build_label(b)),
            (tr("field.commit"), b.get("commit") or tr("misc.none")),
            (tr("field.semantic_pipeline"), b.get("semantic_pipeline_version") or tr("misc.none"))]
    if b.get("dirty"):
        rows[1] = (rows[1][0], f"{rows[1][1]} (+)")
    return rows


def dataset_rows(result: ExperimentResult) -> List[Tuple[str, str]]:
    d = result.dataset
    if d is None:
        return [(tr("section.dataset"), na_text(Reason.NOT_TRAINED))]
    return [(tr("field.dataset"), d.name or tr("misc.none")), (tr("field.rows"), format_optional(d.rows)),
            (tr("field.features"), format_optional(d.features)), (tr("field.classes"), format_optional(d.classes)),
            (tr("field.train"), format_optional(d.train_rows)), (tr("field.test"), format_optional(d.test_rows)),
            (tr("field.seed"), format_optional(d.seed))]


def ontology_rows(result: ExperimentResult) -> List[Tuple[str, str]]:
    o = result.ontology
    if o is None or not o.loaded:
        return [(tr("field.file"), tr("value.not_loaded")), (tr("field.structural"), na_text(Reason.NO_ONTOLOGY))]
    reasoner = status_text(o.reasoner_status)
    if o.reasoner_engine:
        reasoner += f" ({o.reasoner_engine}" + (f", {o.reasoner_seconds:.1f}s" if o.reasoner_seconds is not None else "") + ")"
    if o.reasoner_fallback:
        reasoner += f" — {o.reasoner_fallback}"
    mapping = tr("misc.none") if o.mapped is None else f"{o.mapped}/{o.total}"
    coverage = tr("misc.none") if o.coverage is None else f"{o.coverage:.0%}"
    return [(tr("field.file"), o.path or tr("misc.none")), (tr("field.structural"), status_text(o.structural_status)),
            (tr("field.reasoner"), reasoner), (tr("field.mapping"), mapping), (tr("field.coverage"), coverage),
            (tr("field.ambiguous"), format_optional(o.ambiguous)), (tr("field.tbox"), status_text(o.tbox_status)),
            (tr("field.abox"), status_text(o.abox_status)), (tr("field.richness"), format_optional(o.richness))]


def enrichment_rows(result: ExperimentResult) -> List[Tuple[str, str]]:
    """Três eixos independentes (Parte 7): ontologia · enriquecimento do MLP · semântica no TREPAN."""
    o, e = result.ontology, result.enrichment
    validity = status_text(o.structural_status) if (o and o.loaded) else na_text(Reason.NO_ONTOLOGY)
    if e.mlp_status in ("NOT_EVALUATED", "NOT_AVAILABLE") and not e.decision:
        mlp = status_text(e.mlp_status)
    else:
        mlp = f"{status_text(e.mlp_status)} — {explain_decision(result)}"
    rows = [(tr("field.ontology_validity"), validity), (tr("field.mlp_enrichment"), mlp)]
    if e.trepan_semantics_available is None:
        trepan = na_text(Reason.NOT_TRAINED)
    elif e.trepan_semantics_available:
        trepan = f"{tr('status.AVAILABLE')} ({tr('field.reloaded_mode')}: {e.reloaded_mode or tr('misc.none')})"
    else:
        trepan = (f"{tr('status.NOT_AVAILABLE')} — "
                  f"{_reason_text(e.trepan_semantics_reason) if e.trepan_semantics_reason else na_text(Reason.TEACHER_REJECTED)}")
    rows.append((tr("field.trepan_semantics"), trepan))
    rows.append((tr("field.teacher"), _model_name(e.teacher) if e.teacher else tr("misc.none")))
    if result.stale is False and e.generated is not None:
        rows.append((f"{tr('field.generated')}/{tr('field.stable')}/{tr('field.selected')}",
                     f"{e.generated} / {format_optional(e.stable)} / {format_optional(e.selected)}"))
    return rows


# ---------------------------------------------------------------- modelos
def model_card_rows(card: Optional[ModelCard], result: ExperimentResult, mode: str = BASIC) -> List[Tuple[str, str]]:
    if card is None:
        return [(tr("field.status"), na_text(Reason.NOT_TRAINED))]
    rows = [(tr("field.status"), status_text(card.status) + (f" — {_reason_text(card.status_reason)}" if card.status_reason else ""))]
    if card.key in TREE_KEYS or card.oracle or card.key == "c45":
        rows.append((tr("field.oracle"), _oracle_text(card)))
    for k in PREDICTIVE_METRICS:
        if k in card.metrics:
            rows.append((term(k), format_measure(card.metrics[k])))
        elif card.key.startswith("mlp") or card.key == "c45":
            # nunca silencioso: o valor em falta aparece com a razão (e nunca com o valor do MLP Original)
            missing = Reason.TEACHER_REJECTED if card.key == "mlp_ontological" else Reason.NOT_TRAINED
            rows.append((term(k), na_text(missing)))
    if card.key == "c45":
        rows.append((term("fidelity"), na_text(Reason.NO_ORACLE)))
        rows.append((tr("field.agreement"), format_measure(card.agreement_with_mlp)))
    elif card.key.startswith("trepan"):
        rows.append((f"{term('fidelity')} ({tr('summary.fidelity_against')} {_oracle_text(card)})", format_measure(card.fidelity)))
    for k in COMPLEXITY_KEYS:
        if k in card.complexity:
            rows.append((tr(f"field.{k}"), format_measure(card.complexity[k], integer=True)))
    if card.key in TREE_KEYS:
        diag = result.trees.get(card.key)
        if diag is not None and diag.available:
            rows.append((tr("field.loop_end"), _loop_end_text(diag)))
            if diag.m_of_n_splits is not None:
                rows.append((tr("field.m_of_n"), str(diag.m_of_n_splits)))
    if card.key == "trepan_reloaded":
        bonuses = [s.semantic_bonus.value for s in result.semantic_splits if s.semantic_bonus.available]
        rows.append((tr("field.semantic_splits_count"), str(len(result.semantic_splits))
                     if result.semantic_splits else na_text(Reason.NO_SEMANTIC_FEATURES)))
        rows.append((tr("field.semantic_score_total"), f"{sum(bonuses):.3f}" if bonuses else na_text(Reason.NO_SEMANTIC_FEATURES)))
    if card.key == "mlp_original" or mode == SCIENTIFIC:
        rows.append((tr("field.cached"), yes_no(card.cached)))
        if card.hyperparameters and mode == SCIENTIFIC:
            rows.append((tr("field.hyperparameters"), ", ".join(f"{k}={v}" for k, v in card.hyperparameters.items())))
    if card.key == "mlp_ontological":
        base = result.models.get("mlp_original")
        if card.status == "ACCEPTED" and base is not None:
            for k in ("accuracy", "macro_f1"):
                a, b = card.metrics.get(k), base.metrics.get(k)
                if a and b and a.available and b.available:
                    rows.append((f"{tr('field.delta')} {term(k)} vs {term('mlp_original')}", f"{a.value - b.value:+.3f}"))
        else:
            rows.append((tr("field.delta"), na_text(Reason.TEACHER_REJECTED)))
    if mode == SCIENTIFIC and card.evaluation_samples is not None:
        rows.append((tr("field.eval_samples"), str(card.evaluation_samples)))
    return rows


# ---------------------------------------------------------------- tabela de métricas (Parte 15)
def metrics_tables(result: ExperimentResult) -> Dict[str, Table]:
    """Três grupos separados: desempenho preditivo, fidelidade ao Oracle e complexidade."""
    keys = [k for k in ("mlp_original", "mlp_ontological", "c45", "trepan_original", "trepan_reloaded")]
    predictive = {"headers": [tr("metrics.model")] + [term(k) for k in PREDICTIVE_METRICS], "rows": []}
    fidelity = {"headers": [tr("metrics.model"), tr("metrics.oracle_used"), term("fidelity")], "rows": []}
    complexity = {"headers": [tr("metrics.model")] + [tr(f"field.{k}") for k in COMPLEXITY_KEYS], "rows": []}
    for key in keys:
        card = result.models.get(key)
        name = _model_name(key)
        if card is None or not card.metrics:
            rejected = key == "mlp_ontological" and (card is None or card.status in ("REJECTED", "NOT_AVAILABLE"))
            na = na_text(Reason.TEACHER_REJECTED if rejected else Reason.NOT_TRAINED)
            predictive["rows"].append([name] + [na] * len(PREDICTIVE_METRICS))
            continue
        predictive["rows"].append([name] + [format_measure(card.metrics.get(k)) for k in PREDICTIVE_METRICS])
        if key == "c45":
            fidelity["rows"].append([name, na_text(Reason.NO_ORACLE), na_text(Reason.NO_ORACLE)])
        elif key.startswith("trepan"):
            fidelity["rows"].append([name, _oracle_text(card), format_measure(card.fidelity)])
        if key.startswith("trepan") or key == "c45":
            complexity["rows"].append([name] + [format_measure(card.complexity.get(k), integer=True) for k in COMPLEXITY_KEYS])
    return {"predictive": predictive, "fidelity": fidelity, "complexity": complexity}


def metric_tooltip(metric_key: str) -> str:
    return tr(f"help.{metric_key}")


def provenance_text(result: ExperimentResult, model_key: str, metric_key: Optional[str] = None) -> str:
    """Proveniência de uma métrica: modelo, dataset, seed, split, oracle, build (Parte 16)."""
    p, card = result.provenance, result.models.get(model_key)
    d = result.dataset
    lines = [f"{tr('prov.model')}: {_model_name(model_key)}"]
    if metric_key:
        lines.append(f"{tr('prov.metric')}: {term(metric_key)}")
    lines += [f"{tr('prov.dataset')}: {d.name if d else tr('misc.none')} ({format_optional(p.dataset_hash)})",
              f"{tr('prov.seed')}: {format_optional(p.seed)}", f"{tr('prov.split')}: {format_optional(p.split)}",
              f"{tr('prov.oracle')}: {_oracle_text(card) if card else tr('misc.none')}",
              f"{tr('prov.evaluated_on')}: {format_optional(card.evaluation_samples if card else None)}",
              f"{tr('prov.build')}: {build_label(p.build)}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- árvores (Parte 23-24)
def _loop_end_text(diag: TreeDiagnostics) -> str:
    code = diag.loop_end_reason
    if not code:
        return tr("misc.none")
    text = tr("stop." + code)
    return code if text == "stop." + code else text


def tree_diagnostic_rows(diag: Optional[TreeDiagnostics]) -> List[Tuple[str, str]]:
    if diag is None or not diag.available:
        return [(tr("section.diagnostics"), na_text(Reason.TREE_NOT_BUILT))]
    reasons = ", ".join(f"{tr('stop.' + k) if tr('stop.' + k) != 'stop.' + k else k}: {v}" for k, v in sorted(diag.stop_reasons.items()))
    rows = [(tr("field.logical_nodes"), format_measure(diag.logical_nodes, integer=True)),
            (tr("field.rendered_nodes"), format_measure(diag.rendered_nodes, integer=True)),
            (tr("field.depth"), format_measure(diag.depth, integer=True)),
            (tr("field.queries"), format_measure(diag.queries_used, integer=True)),
            (tr("field.query_budget"), format_measure(diag.query_budget, integer=True)),
            (tr("field.budget_exhausted"), yes_no(diag.query_budget_exhausted)),
            (tr("field.node_budget"), format_measure(diag.node_budget, integer=True)),
            (tr("field.nodes_before"), format_measure(diag.nodes_before_pruning, integer=True)),
            (tr("field.nodes_after"), format_measure(diag.nodes_after_pruning, integer=True)),
            (tr("field.loop_end"), _loop_end_text(diag)),
            (tr("field.stop_reasons"), reasons or tr("misc.none"))]
    if diag.m_of_n_splits is not None:
        rows.append((tr("field.m_of_n"), str(diag.m_of_n_splits)))
    n = diag.logical_nodes.value
    if n is not None and n <= 3:
        rows.insert(0, (tr("section.diagnostics"), tr("tree.small_diagnostic")))  # nunca "erro de árvore"
    return rows


# ---------------------------------------------------------------- semântica (Partes 25-26, 29-30, 28)
def semantic_features_table(result: ExperimentResult, selected_only: bool = False) -> Table:
    rows = [[f.name, f.type, f.source, format_measure(f.stability), yes_no(f.selected), f.reason]
            for f in result.semantic_features if f.selected or not selected_only]
    return {"headers": [tr(f"col.{c}") for c in ("name", "type", "source", "stability", "selected", "reason")], "rows": rows}


def semantic_splits_table(result: ExperimentResult) -> Table:
    rows = [[format_optional(s.node), s.feature, format_measure(s.base_score), format_measure(s.semantic_bonus, "{:+.3f}"),
             format_measure(s.final_score), s.reason] for s in result.semantic_splits]
    return {"headers": [tr(f"col.{c}") for c in ("node", "feature", "base_score", "semantic_bonus", "final_score", "ontology_reason")],
            "rows": rows}


def controls_table(result: ExperimentResult) -> Table:
    order = {"no_semantics": 0, "real_owl": 1, "shuffled_owl": 2, "random_control": 3}
    rows = [[tr(f"arm.{c.arm}") if tr(f"arm.{c.arm}") != f"arm.{c.arm}" else c.arm, format_measure(c.accuracy),
             format_measure(c.fidelity), format_measure(c.nodes, integer=True), str(c.n_runs)]
            for c in sorted(result.controls, key=lambda c: order.get(c.arm, 9))]
    return {"headers": [tr("col.arm"), term("accuracy"), term("fidelity"), tr("field.nodes"), tr("col.runs")], "rows": rows}


def ablation_table(result: ExperimentResult) -> Table:
    """Sem 'vencedor' automático: apenas os números lado a lado."""
    rows = [[a.configuration, format_measure(a.accuracy), format_measure(a.fidelity), format_measure(a.nodes, integer=True),
             format_measure(a.semantic_contribution, "{:+.3f}")] for a in result.ablation]
    return {"headers": [tr("col.configuration"), term("accuracy"), term("fidelity"), tr("field.nodes"),
                        tr("col.semantic_contribution")], "rows": rows}


def benchmark_table(result: ExperimentResult) -> Table:
    b = result.benchmark
    if b is None:
        return {"headers": [tr("col.name"), tr("col.mean_std"), tr("col.runs")], "rows": []}
    rows = [[str(r.get("name", "")), f"{r['mean']:.3f} ± {r['std']:.3f}" if r.get("mean") is not None and r.get("std") is not None
             else na_text(Reason.NOT_REPORTED), str(r.get("n", b.n_runs))] for r in b.rows]
    return {"headers": [tr("col.name"), tr("col.mean_std"), tr("col.runs")], "rows": rows}


# ---------------------------------------------------------------- experiência, cache, stale
def experiment_rows(result: ExperimentResult) -> List[Tuple[str, str]]:
    p = result.provenance
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.timestamp))
    tuning = (result.config or {}).get("trepan_tuning") or {}
    tuning_row = []
    if tuning:
        sel = tuning.get("selected") or {}
        if tuning.get("failed"):
            text = f"fallback canónico: {tuning['failed']}"
        else:
            state = tuning.get("status") or ("stable_exact" if tuning.get("stable") else "tuning_uncertain")
            text = (f"purity_epsilon={sel.get('purity_epsilon')}, max_nodes={sel.get('max_nodes')} — {state} "
                    f"(selection_probability {format_optional(tuning.get('selection_probability'), '{:.0%}')} no block bootstrap)")
        tuning_row = [(tr("field.trepan_tuning"), text)]
    mode_row = [(tr("field.execution_mode"), _mode_text(result))] if p.execution_mode else []
    return mode_row + [(tr("field.experiment_id"), p.experiment_id), (tr("field.seed"), format_optional(p.seed)),
            (tr("field.dataset_hash"), format_optional(p.dataset_hash)), (tr("field.owl_hash"), format_optional(p.owl_hash)),
            (tr("field.config_hash"), format_optional(p.config_hash))] + tuning_row + build_rows(result) + [
            (tr("field.timestamp"), stamp), (tr("field.cache"), cache_text(result))]


def _mode_text(result: ExperimentResult) -> str:
    mode = result.provenance.execution_mode
    label = tr("mode.benchmark") if mode == "SCIENTIFIC_BENCHMARK" else tr("mode.interactive")
    ok = bool(result.scientific and result.scientific.benchmark_eligible)
    return f"{label} — {tr('mode.benchmark_ok') if ok else tr('mode.not_benchmark')}"


def _capacity_step_text(d) -> str:
    st = d.last_capacity_step or {}
    if not st:
        return d.expansion_interpretation or na_text(Reason.TUNING_NOT_RUN)
    test = st.get("statistical_test") or {}
    p = test.get("p_value")
    text = (f"{st.get('previous_max_nodes')} → {st.get('candidate_max_nodes')} nós: Δfidelity={st.get('fidelity_delta'):.4f} "
            f"(p={p:.3f}); ganho suportado: {'sim' if st.get('capacity_gain_supported') else 'não'}") if st.get("fidelity_delta") is not None and p is not None else \
        f"{st.get('previous_max_nodes')} → {st.get('candidate_max_nodes')} nós"
    return text + (f" — {d.expansion_interpretation}" if d.expansion_interpretation else "")


def scientific_rows(result: ExperimentResult) -> List[Tuple[str, str]]:
    """Diagnóstico científico do tuning e do oráculo; só apresenta o que o backend já calculou."""
    d = result.scientific
    if d is None:
        return []

    def pct(m):
        return format_measure(m, "{:.1%}")

    oid = (d.oracle_id or "")[:8]
    fid = (f"{format_measure(d.fidelity_mean)} ± {format_measure(d.fidelity_std)}")
    queries = f"{format_measure(d.queries_used, integer=True)} / {format_measure(d.query_budget, integer=True)}"
    cap = pct(d.fraction_at_node_cap) + (f" — ⚠ {tr('sci.censored')}" if d.node_cap_censored else "")
    return [
        (tr("field.execution_mode"), _mode_text(result)),
        (tr("sci.oracle_id"), oid if oid else na_text(Reason.NO_ORACLE_CONTRACT)),
        (tr("sci.same_oracle"), yes_no(d.same_oracle_original_reloaded) if d.same_oracle_original_reloaded is not None
         else na_text(Reason.NO_ORACLE_CONTRACT)),
        (tr("sci.seed"), format_optional(d.seed)),
        (tr("sci.cv_plan"), d.cv_plan or na_text(Reason.TUNING_NOT_RUN)),
        (tr("sci.selected"), d.selected_config or na_text(Reason.TUNING_NOT_RUN)),
        (tr("sci.fidelity"), fid),
        (tr("sci.predictive_stability"), format_measure(d.predictive_stability)),
        (tr("sci.structural_stability"), format_measure(d.structural_stability, "{:.2f}")
         + (f" — ⚠ {tr('sci.censored_weak')}" if d.structural_stability_evidence == "censored_by_node_cap" else "")),
        (tr("sci.full_cv"), d.selected_config_full_cv or na_text(Reason.TUNING_NOT_RUN)),
        (tr("sci.selection_probability"), pct(d.selection_probability)
         + (f" ({d.selection_resamples} reamostragens)" if d.selection_resamples else "")),
        (tr("sci.modal"), (f"{d.bootstrap_modal_config} ({pct(d.bootstrap_modal_probability)})" if d.bootstrap_modal_config
                           else na_text(Reason.TUNING_NOT_RUN))),
        (tr("sci.runner_up"), d.selection_runner_up or tr("misc.none")),
        (tr("sci.margin"), pct(d.selection_margin)),
        (tr("sci.fragile"), tr("sci.fragile.yes") if d.full_cv_selection_fragile else
         (tr("sci.fragile.no") if d.full_cv_selection_fragile is not None else na_text(Reason.TUNING_NOT_RUN))),
        (tr("sci.bootstrap_method"), format_optional(d.bootstrap_method)),
        (tr("sci.expansion"), (f"{'sim' if d.expansion_triggered else 'não'}; rondas {format_optional(d.capacity_expansion_rounds)}; "
                               f"grelha {d.initial_node_grid} → {d.final_node_grid}; {format_optional(d.expansion_stop_reason)}")
         if d.initial_node_grid else na_text(Reason.TUNING_NOT_RUN)),
        (tr("sci.equivalent"), (f"{d.equivalent_candidate_count} candidatos; P(família)={pct(d.equivalent_set_probability)}"
                                if d.equivalent_candidate_count is not None else na_text(Reason.TUNING_NOT_RUN))),
        (tr("sci.behavior"), tr("sci.behavior.unstable") if d.tree_behavior_unstable else
         (tr("sci.behavior.stable") if d.tree_behavior_unstable is not None else na_text(Reason.TUNING_NOT_RUN))),
        (tr("sci.capacity_step"), _capacity_step_text(d)),
        (tr("sci.node_cap"), cap),
        (tr("sci.queries"), queries),
        (tr("sci.budget_exhausted"), yes_no(d.budget_exhausted)),
        (tr("sci.test_used"), yes_no(d.test_used_for_selection)),
        (tr("sci.tuning_status"), tr(f"sci.status.{d.tuning_status}")),
    ]


def cache_text(result: ExperimentResult) -> str:
    p = result.provenance
    if p.cache_used:
        key = (p.cache_key or "")[:10]
        return tr("cache.banner", key=key)
    return tr("cache.fresh")


def cache_banner(result: ExperimentResult) -> str:
    """Vazio quando o resultado foi calculado neste treino; nunca mostra cache como se fosse novo (Parte 17)."""
    return cache_text(result) if result.provenance.cache_used else ""


def stale_banner(result: ExperimentResult) -> str:
    if not result.stale:
        return ""
    return tr("stale.banner", reasons=", ".join(tr(k) if tr(k) != k else k for k in result.stale_reasons))


# ---------------------------------------------------------------- resumo (as 18 perguntas)
def _pair(a: str, b: str) -> str:
    return f"{a} / {b}"


def summary_rows(result: ExperimentResult) -> List[Tuple[str, str]]:
    m = result.models
    to, tre = m.get("trepan_original"), m.get("trepan_reloaded")
    d = result.dataset
    o = result.ontology
    e = result.enrichment
    sel_feats = [f.name for f in result.semantic_features if f.selected]

    def both(getter, **kw):
        return _pair(*(format_measure(getter(c), **kw) if c else na_text(Reason.TREE_NOT_BUILT) for c in (to, tre)))

    stops = []
    for k in TREE_KEYS:
        diag = result.trees.get(k)
        stops.append(diag.loop_end_reason if diag and diag.available and diag.loop_end_reason else tr("misc.none"))
    stop_text = _pair(*[(tr("stop." + s) if tr("stop." + s) != "stop." + s else s) for s in stops])
    enrich_text = status_text(e.mlp_status)
    if e.decision:
        enrich_text += f" — {explain_decision(result)}"
    n_split = sum(1 for s in result.semantic_splits if s.decision_changed)
    p = result.provenance
    mlp_name = _model_name("mlp_ontological" if (e.teacher == "mlp_ontological" or e.teacher == "mlp_semantic") else "mlp_original")
    return [
        (tr("summary.dataset"), f"{d.name} ({d.rows} × {d.features})" if d else na_text(Reason.NOT_TRAINED)),
        (tr("summary.owl"), (o.path or tr("misc.none")) if (o and o.loaded) else tr("value.not_loaded")),
        (tr("summary.build"), build_label(p.build)),
        (tr("summary.mlp"), mlp_name),
        (tr("summary.ontology_valid"), status_text(o.structural_status) if (o and o.loaded) else na_text(Reason.NO_ONTOLOGY)),
        (tr("summary.enrichment"), enrich_text),
        (tr("summary.oracle_original"), _oracle_text(to)),
        (tr("summary.oracle_reloaded"), _oracle_text(tre)),
        (tr("summary.nodes"), both(lambda c: c.complexity.get("nodes"), integer=True)),
        (tr("summary.stop"), stop_text),
        (tr("summary.queries"), both(lambda c: c.complexity.get("queries"), integer=True)),
        (tr("summary.accuracy"), both(lambda c: c.metrics.get("accuracy"))),
        (f"{tr('summary.fidelity')} ({tr('summary.fidelity_against')})", both(lambda c: c.fidelity)),
        (tr("summary.owl_features"), ", ".join(sel_feats) if sel_feats else tr("value.none_selected")),
        (tr("summary.semantic_splits"), str(n_split) if result.semantic_splits else na_text(Reason.NO_SEMANTIC_FEATURES)),
        (tr("summary.cache"), cache_text(result)),
        (tr("summary.seed_config"), f"{format_optional(p.seed)} / {format_optional(p.config_hash)}"),
    ]


def render_text(result: ExperimentResult, mode: str = BASIC) -> str:
    """Texto simples (resumo) usado no relatório exportado e como fallback sem widgets."""
    blocks = [(tr("tab.summary"), summary_rows(result))]
    if mode == SCIENTIFIC:
        blocks += [(tr("section.dataset"), dataset_rows(result)), (tr("section.ontology"), ontology_rows(result)),
                   (tr("section.enrichment"), enrichment_rows(result)),
                   (tr("section.scientific"), scientific_rows(result)), (tr("section.experiment"), experiment_rows(result))]
    out = []
    for title, rows in blocks:
        out.append(f"== {title} ==")
        out += [f"{k}: {v}" for k, v in rows]
        out.append("")
    banner = stale_banner(result)
    return (banner + "\n\n" if banner else "") + "\n".join(out)
