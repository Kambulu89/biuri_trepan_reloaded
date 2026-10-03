"""Apresentador: N/A com significado, rótulos, oracle, tabelas separadas, stale/cache, sem recálculo."""
import json

import pytest

from core.experiment_result import (AblationRow, BenchmarkSummary, ControlRow, DatasetInfo, EnrichmentInfo,
                                    ExperimentResult, Measure, ModelCard, OntologyInfo, Provenance, Reason,
                                    SemanticFeatureRow, SemanticSplitRow, TreeDiagnostics)
from gui import result_presenter as rp
from gui.strings import tr


def _tree_card(key, oracle, acc, fid, nodes):
    return ModelCard(key=key, status="AVAILABLE", oracle=oracle,
                     metrics={"accuracy": Measure.of(acc), "macro_f1": Measure.of(acc - .01)},
                     fidelity=Measure.of(fid), complexity={"nodes": Measure.of(nodes), "depth": Measure.of(3),
                                                           "leaves": Measure.of(5), "queries": Measure.of(100)})


def make_result(rejected=True, **kw):
    mlp_onto = (ModelCard(key="mlp_ontological", status="REJECTED", status_reason="REJECT_NO_INFORMATIONAL_GAIN")
                if rejected else ModelCard(key="mlp_ontological", status="ACCEPTED",
                                           metrics={"accuracy": Measure.of(.90), "macro_f1": Measure.of(.89)}))
    r = ExperimentResult(
        state="RESULTS_READY",
        provenance=Provenance(experiment_id="exp123456789", seed=42, dataset_hash="dh", owl_hash="oh", config_hash="ch",
                              build={"version": "9.2.0", "commit": "abc1234", "semantic_pipeline_version": "sp-9", "dirty": False}),
        dataset=DatasetInfo(name="iris", rows=150, features=4, classes=3, train_rows=120, test_rows=30, seed=42),
        ontology=OntologyInfo(loaded=True, path="iris.owl", structural_status="VALID", reasoner_status="CONSISTENT",
                              reasoner_engine="HermiT", reasoner_seconds=1.234, mapped=3, total=4, coverage=.75, ambiguous=0),
        enrichment=EnrichmentInfo(mlp_status="REJECTED" if rejected else "ACCEPTED",
                                  decision="REJECT_NO_INFORMATIONAL_GAIN" if rejected else "ACCEPT_SIGNIFICANT_GAIN",
                                  base_utility=Measure.of(.80), owl_utility=Measure.of(.79), delta_utility=Measure.of(-.01),
                                  trepan_semantics_available=not rejected,
                                  trepan_semantics_reason="semantic teacher rejected" if rejected else None,
                                  teacher="mlp_original" if rejected else "mlp_semantic", reloaded_mode="original_space"),
        models={
            "mlp_original": ModelCard(key="mlp_original", status="AVAILABLE", cached=True, evaluation_samples=30,
                                      metrics={"accuracy": Measure.of(.91), "balanced_accuracy": Measure.of(.9),
                                               "macro_f1": Measure.of(.9), "precision_macro": Measure.of(.9),
                                               "recall_macro": Measure.of(.9)}),
            "mlp_ontological": mlp_onto,
            "c45": ModelCard(key="c45", status="AVAILABLE", metrics={"accuracy": Measure.of(.8)},
                             fidelity=Measure.na(Reason.NO_ORACLE), agreement_with_mlp=Measure.of(.85),
                             complexity={"nodes": Measure.of(9)}),
            "trepan_original": _tree_card("trepan_original", "mlp_original", .85, .95, 11),
            "trepan_reloaded": _tree_card("trepan_reloaded", "mlp_original" if rejected else "mlp_ontological", .86, .96, 13),
        },
        trees={"trepan_original": TreeDiagnostics(tree="trepan_original", logical_nodes=Measure.of(11), depth=Measure.of(3),
                                                  loop_end_reason="query_budget_before_min_sample",
                                                  stop_reasons={"pure_node": 3, "max_depth": 1}, available=True),
               "trepan_reloaded": TreeDiagnostics(tree="trepan_reloaded", logical_nodes=Measure.of(1), available=True)},
    )
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_na_texts_have_meaning_per_category():
    assert rp.na_text(Reason.TEACHER_REJECTED) == "Não calculado — professor semântico rejeitado"
    assert rp.na_text(Reason.NO_ORACLE).startswith("Não aplicável")
    assert rp.na_text(Reason.NO_ONTOLOGY).startswith("Indisponível")
    assert rp.na_text(Reason.TREE_NOT_BUILT).startswith("Não executado")


def test_missing_measure_is_never_rendered_as_zero():
    assert rp.format_measure(Measure.na(Reason.TEACHER_REJECTED)) != "0.000"
    assert rp.format_measure(Measure.of(0.0)) == "0.000"  # zero real continua a ser zero
    assert rp.format_measure(None).startswith("Indisponível")


def test_c45_has_no_fidelity_and_agreement_is_labelled():
    rows = dict(rp.model_card_rows(make_result().models["c45"], make_result()))
    assert rows["Fidelity"].startswith("Não aplicável")
    assert "Concordância com o MLP (diagnóstico)" in rows
    t = rp.metrics_tables(make_result())
    c45_fid = [r for r in t["fidelity"]["rows"] if r[0] == "C4.5"][0]
    assert c45_fid[1].startswith("Não aplicável") and c45_fid[2].startswith("Não aplicável")


def test_trepan_oracle_is_displayed_per_tree():
    r = make_result(rejected=True)
    t = {row[0]: row for row in rp.metrics_tables(r)["fidelity"]["rows"]}
    assert t["TREPAN Original"][1] == "MLP Original" and t["TREPAN Reloaded"][1] == "MLP Original"
    r2 = make_result(rejected=False)
    t2 = {row[0]: row for row in rp.metrics_tables(r2)["fidelity"]["rows"]}
    assert t2["TREPAN Reloaded"][1] == "MLP Ontológico" and t2["TREPAN Original"][1] == "MLP Original"
    labels = dict(rp.model_card_rows(r2.models["trepan_reloaded"], r2))
    assert any("MLP Ontológico)" in k and "Fidelity" in k for k in labels)


def test_metrics_are_separated_into_three_groups():
    t = rp.metrics_tables(make_result())
    assert set(t) == {"predictive", "fidelity", "complexity"}
    assert "Fidelity" not in " ".join(t["predictive"]["headers"])
    assert "Fidelity" in " ".join(t["fidelity"]["headers"])
    assert {"Nós", "Profundidade"} <= set(t["complexity"]["headers"])


def test_rejected_ontological_mlp_never_replaces_original_values():
    r = make_result(rejected=True)
    pred = {row[0]: row for row in rp.metrics_tables(r)["predictive"]["rows"]}
    assert pred["MLP Original"][1] == "0.910"
    assert all(c.startswith("Não calculado") for c in pred["MLP Ontológico"][1:])
    onto = dict(rp.model_card_rows(r.models["mlp_ontological"], r))
    assert onto["Estado"].startswith("REJEITADO")


def test_accepted_ontological_mlp_shows_delta_vs_original():
    r = make_result(rejected=False)
    rows = dict(rp.model_card_rows(r.models["mlp_ontological"], r))
    assert rows["Δ utilidade Accuracy vs MLP Original"] == "-0.010"


def test_enrichment_three_independent_axes_and_human_reason():
    rows = dict(rp.enrichment_rows(make_result(rejected=True)))
    assert rows["Validade da ontologia"] == "VÁLIDA"
    assert rows["Enriquecimento do MLP"].startswith("REJEITADO")
    assert "Utilidade base: 0.800" in rows["Enriquecimento do MLP"] and "Δ utilidade: -0.010" in rows["Enriquecimento do MLP"]
    assert "INDISPONÍVEL" in rows["Semântica no TREPAN"]
    assert "REJECT_NO_INFORMATIONAL_GAIN" not in rp.explain_decision(make_result())  # texto humano, não só o código


def test_ontology_rows_when_not_loaded():
    r = make_result(ontology=OntologyInfo(loaded=False))
    rows = dict(rp.ontology_rows(r))
    assert rows["Ficheiro"] == "Não carregada" and rows["Estado estrutural"].startswith("Indisponível")


def test_ontology_rows_structured_and_reasoner_detail():
    rows = dict(rp.ontology_rows(make_result()))
    assert rows["Reasoner"].startswith("CONSISTENTE (HermiT, 1.2s") and rows["Mapeamento"] == "3/4"
    assert rows["Cobertura"] == "75%"


def test_tree_diagnostics_and_stump_message():
    r = make_result()
    rows = dict(rp.tree_diagnostic_rows(r.trees["trepan_original"]))
    assert rows["Nós lógicos"] == "11" and "Motivo de paragem global" in rows
    assert "orçamento de queries" in rows["Motivo de paragem global"]
    stump = rp.tree_diagnostic_rows(r.trees["trepan_reloaded"])
    assert stump[0][1] == tr("tree.small_diagnostic") and "erro" not in stump[0][1].lower()
    assert rp.tree_diagnostic_rows(None)[0][1].startswith("Não executado")


def test_semantic_tables_and_selected_only_filter():
    r = make_result(semantic_features=[
        SemanticFeatureRow(name="f1", type="class", source="owl", stability=Measure.of(.9), selected=True, reason="estável"),
        SemanticFeatureRow(name="f2", type="class", source="owl", stability=Measure.of(.2), selected=False, reason="instável")],
        semantic_splits=[SemanticSplitRow(node=1, feature="f1", base_score=Measure.of(.2), semantic_bonus=Measure.of(.05),
                                          final_score=Measure.of(.25), reason="mesma classe", decision_changed=True)])
    assert len(rp.semantic_features_table(r)["rows"]) == 2
    only = rp.semantic_features_table(r, selected_only=True)
    assert [row[0] for row in only["rows"]] == ["f1"] and only["headers"][0] == "Nome"
    sp = rp.semantic_splits_table(r)
    assert sp["rows"][0][3] == "+0.050" and sp["headers"][3] == "Bónus semântico"


def test_controls_are_ordered_and_ablation_has_no_winner():
    r = make_result(controls=[ControlRow("shuffled_owl", Measure.of(.5)), ControlRow("no_semantics", Measure.of(.6)),
                              ControlRow("real_owl", Measure.of(.7))],
                    ablation=[AblationRow("A", Measure.of(.8)), AblationRow("B", Measure.of(.7))])
    arms = [row[0] for row in rp.controls_table(r)["rows"]]
    assert arms == ["Sem semântica", "OWL real", "OWL baralhada"]
    text = json.dumps(rp.ablation_table(r)).lower()
    assert "vencedor" not in text and "winner" not in text and "melhor" not in text


def test_benchmark_summary_is_mean_std_not_hundreds_of_rows():
    r = make_result(benchmark=BenchmarkSummary(metric="accuracy", n_runs=30, rows=[
        {"name": "TREPAN Original", "mean": .85, "std": .02, "n": 30}, {"name": "TREPAN Reloaded", "mean": .87, "std": .03, "n": 30}]))
    t = rp.benchmark_table(r)
    assert len(t["rows"]) == 2 and t["rows"][0][1] == "0.850 ± 0.020"


def test_cache_and_stale_banners():
    r = make_result()
    assert rp.cache_banner(r) == "" and "Calculado neste treino" in rp.experiment_rows(r)[-1][1]
    r.provenance.cache_used, r.provenance.cache_key = True, "abcdef0123456789"
    assert "RESULTADO EM CACHE" in rp.cache_banner(r) and "abcdef0123" in rp.cache_banner(r)
    assert rp.stale_banner(r) == ""
    r.stale, r.stale_reasons = True, ["stale.owl_changed"]
    assert "DESATUALIZADOS" in rp.stale_banner(r) and "ontologia" in rp.stale_banner(r)
    assert rp.stale_banner(r) in rp.render_text(r)


def test_summary_answers_the_core_questions():
    rows = dict(rp.summary_rows(make_result()))
    assert rows["Dataset ativo"].startswith("iris")
    assert rows["Build"] == "V9.2.0 · abc1234"
    assert rows["Oracle do TREPAN Original"] == "MLP Original"
    assert rows["Nós (Original / Reloaded)"] == "11 / 13"
    assert rows["Porque parou a árvore"].startswith("orçamento de queries")
    assert rows["Features OWL selecionadas"] == "nenhuma"
    assert rows["Splits semânticos"].startswith("Indisponível")


def test_provenance_text_has_all_fields():
    text = rp.provenance_text(make_result(), "trepan_original", "fidelity")
    for needle in ("TREPAN Original", "Fidelity", "iris", "42", "MLP Original", "V9.2.0"):
        assert needle in text


def test_mode_changes_only_what_is_shown_not_values():
    r = make_result()
    basic = dict(rp.model_card_rows(r.models["trepan_original"], r, rp.BASIC))
    sci = dict(rp.model_card_rows(r.models["trepan_original"], r, rp.SCIENTIFIC))
    for k, v in basic.items():
        assert sci[k] == v
    assert len(sci) >= len(basic)
    assert rp.render_text(r, rp.SCIENTIFIC).count("==") > rp.render_text(r, rp.BASIC).count("==")


def test_presenter_does_not_mutate_result():
    r = make_result()
    before = json.dumps(r.to_dict(), sort_keys=True, default=str)
    rp.summary_rows(r); rp.metrics_tables(r); rp.enrichment_rows(r); rp.render_text(r, rp.SCIENTIFIC)
    assert json.dumps(r.to_dict(), sort_keys=True, default=str) == before
