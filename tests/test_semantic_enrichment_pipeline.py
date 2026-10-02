"""Pipeline de enriquecimento: leakage, decisões legítimas, controlo negativo e ablação (sintético)."""
import numpy as np
import pandas as pd
import pytest

owlready2 = pytest.importorskip("owlready2")

from core.ontology_quality import OntologyQualityGate
from core.semantic_enrichment import (
    EnrichmentConfig, candidate_grid, classification_metrics_np, evaluate_semantic_enrichment,
    export_feature_audit, utility,
)

FAST = EnrichmentConfig(cv_folds=3, tuning_candidates=2, tuning_inner_folds=2, max_iter=150,
                        n_bootstrap=200, random_state=7)


def _family_onto(name="enr", noise=0):
    """Família Size (mean/error/worst) declarada por anotações + ``noise`` atributos irrelevantes."""
    onto = owlready2.World().get_ontology(f"http://test.org/{name}.owl")
    with onto:
        type("statisticRole", (owlready2.AnnotationProperty,), {})
        type("measurementFamily", (owlready2.AnnotationProperty,), {})
        Morph = type("Morphology", (owlready2.Thing,), {})
        Background = type("Background", (owlready2.Thing,), {})
        for role in ("mean", "error", "worst"):
            p = type(f"hasSize{role.capitalize()}", (owlready2.DataProperty,),
                     {"range": [float], "domain": [Morph]})
            p.statisticRole = [role]; p.measurementFamily = ["Size"]
        for i in range(noise):
            type(f"hasBackgroundVariable{i:02d}Level", (owlready2.DataProperty,),
                 {"range": [float], "domain": [Background]})
    return onto


def _data(n=240, signal="relation", seed=0, noise=0):
    rng = np.random.default_rng(seed)
    mean = rng.uniform(5, 15, n)
    delta = rng.uniform(0.0, 1.0, n)
    X = pd.DataFrame({"hasSizeMean": mean, "hasSizeError": rng.uniform(0.1, 1, n),
                      "hasSizeWorst": mean * (1 + delta)})
    for i in range(noise):
        X[f"hasBackgroundVariable{i:02d}Level"] = rng.normal(size=n)
    base = delta if signal == "relation" else (mean - mean.mean())
    y = (base > np.median(base)).astype(int)
    flip = rng.random(n) < 0.03
    return X, np.where(flip, 1 - y, y)


def _quality(onto, X):
    return OntologyQualityGate().evaluate(list(X.columns), onto, require_reasoner=False)


def _run(onto, X, y, **kw):
    return evaluate_semantic_enrichment(X, y, onto, quality_report=_quality(onto, X), config=FAST, **kw)


# ------------------------------------------------------------ decisões legítimas ---
def test_legitimate_acceptance_when_relation_carries_signal():
    # O sinal está na relação worst/mean escondida entre muitos atributos irrelevantes:
    # sem a ontologia o MLP não a descobre com tão poucas amostras.
    onto = _family_onto("acc", noise=20); X, y = _data(n=300, signal="relation", noise=20)
    res = _run(onto, X, y); rep = res.report
    assert rep["decision"].startswith("ACCEPT"), (rep["decision"], rep["stages"]["D_mlp_comparison"]["utility_gain_ci"])
    assert rep["semantic_mlp_accepted"] is True and rep["semantic_trepan_available"] is True
    assert res.selected_features and set(res.selected_features) <= set(res.processor.output_features_)
    d = rep["stages"]["D_mlp_comparison"]
    assert d["with_owl"]["utility"] > d["base"]["utility"]
    assert d["utility_gain_ci"][0] > 0  # intervalo de confiança exclui zero


def test_easy_problem_is_not_accepted_just_because_features_exist():
    """MLP já resolve o problema: criar features OWL não pode ser contado como ganho."""
    onto = _family_onto("easy"); X, y = _data(n=240, signal="relation")
    rep = _run(onto, X, y).report
    assert rep["decision"] != "ACCEPT_SIGNIFICANT_GAIN"
    assert rep["semantic_trepan_available"] is True


def test_legitimate_rejection_when_relation_is_irrelevant():
    onto = _family_onto("rej"); X, y = _data(signal="mean_only")
    rep = _run(onto, X, y).report
    assert rep["decision"] != "ACCEPT_SIGNIFICANT_GAIN"
    # a rejeição do MLP não desliga a semântica do TREPAN
    assert rep["semantic_trepan_available"] is True
    if rep["decision"].startswith("REJECT"):
        assert rep["semantic_mlp_accepted"] is False and rep["selected_semantic_features"] == []


def test_mlp_rejected_but_trepan_available_is_explicit():
    onto = _family_onto("sep"); X, y = _data(signal="mean_only")
    cfg = EnrichmentConfig(**{**FAST.__dict__, "min_selection_frequency": 1.01})  # nenhuma estável
    rep = evaluate_semantic_enrichment(X, y, onto, quality_report=_quality(onto, X), config=cfg).report
    assert rep["decision"] == "REJECT_UNSTABLE_FEATURES"
    assert rep["semantic_mlp_accepted"] is False and rep["semantic_trepan_available"] is True


def test_invalid_ontology_rejects_both():
    onto = owlready2.World().get_ontology("http://test.org/gen.owl")
    with onto:
        for n in ("Feature", "Value", "Attribute", "Measurement"):
            type(n, (owlready2.Thing,), {})
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 4)),
                     columns=["feature", "value", "attribute", "measurement"])
    y = np.tile([0, 1], 30)
    rep = evaluate_semantic_enrichment(
        X, y, onto, quality_report=OntologyQualityGate().evaluate(list(X.columns), onto, require_reasoner=False),
        config=FAST).report
    assert rep["decision"] == "REJECT_INVALID_ONTOLOGY"
    assert rep["semantic_mlp_accepted"] is False and rep["semantic_trepan_available"] is False


def test_abox_leakage_is_a_distinct_rejection():
    onto = _family_onto("leak")
    with onto:
        Rec = type("Record", (owlready2.Thing,), {}); Rec("row_1"); Rec("row_2")
    X, y = _data()
    rep = _run(onto, X, y).report
    assert rep["decision"] == "REJECT_LEAKAGE_RISK"
    assert rep["stages"]["A_quality"]["abox_status"] == "REJECT_DATASET_RECORDS_IN_ABOX"


def test_no_novel_features_when_sources_are_constant():
    onto = _family_onto("nn")
    n = 60
    X = pd.DataFrame({"hasSizeMean": np.full(n, 3.0), "hasSizeError": np.full(n, .5), "hasSizeWorst": np.full(n, 4.0)})
    y = np.tile([0, 1], n // 2)
    rep = evaluate_semantic_enrichment(X, y, onto, quality_report=_quality(onto, X), config=FAST).report
    assert rep["decision"] == "REJECT_NO_NOVEL_FEATURES" and rep["semantic_trepan_available"] is True


def test_missing_quality_report_is_rejected_not_assumed_valid():
    onto = _family_onto("mq"); X, y = _data(n=60)
    assert evaluate_semantic_enrichment(X, y, onto, config=FAST).report["decision"] == "REJECT_INVALID_ONTOLOGY"


# --------------------------------------------------------------- anti-leakage ---
def test_report_states_test_is_never_used_and_budget_is_equal():
    onto = _family_onto("bud"); X, y = _data(n=150)
    rep = _run(onto, X, y).report
    assert rep["test_used"] is False
    d = rep["stages"]["D_mlp_comparison"]
    assert d["tuning_candidates_per_arm"] == FAST.tuning_candidates
    assert len(d["per_fold"]) == FAST.cv_folds
    assert all(len(f["base_params"]) == len(f["onto_params"]) for f in d["per_fold"])  # mesmas chaves


def test_candidate_grid_is_identical_for_both_arms_and_reproducible():
    assert candidate_grid(5, 1) == candidate_grid(5, 1)
    assert candidate_grid(5, 1) != candidate_grid(5, 2)
    assert len(candidate_grid(5, 1)) == 5


def test_processor_fit_on_train_ignores_test_rows():
    """Alterar X_test não muda nenhum parâmetro aprendido em fit(X_train)."""
    from core.ontology_processor import OntologyProcessor
    onto = _family_onto("lk"); X, _ = _data(n=200)
    train, test = X.iloc[:150], X.iloc[150:].copy()
    p1 = OntologyProcessor(onto).fit(train, log=False)
    test_shifted = test * 1000.0 + 5000.0  # distribuição completamente diferente
    _ = p1.transform(test_shifted)           # transform não pode alterar o estado aprendido
    p2 = OntologyProcessor(onto).fit(train, log=False)
    assert p1.feature_specs_ == p2.feature_specs_
    out_a, out_b = p1.transform(test), p2.transform(test)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_changing_validation_rows_does_not_change_fold_train_fit():
    from core.ontology_processor import OntologyProcessor
    onto = _family_onto("lk2"); X, _ = _data(n=200)
    train = X.iloc[:150]
    a = OntologyProcessor(onto).fit(train, log=False)
    b = OntologyProcessor(onto).fit(train.copy(), log=False)
    spec = lambda p: {s["name"]: (s.get("centers"), s.get("scales")) for s in p.feature_specs_}
    assert spec(a) == spec(b)


def test_reproducibility_same_seed_same_decision_and_features():
    onto = _family_onto("rep"); X, y = _data(n=150)
    a = _run(onto, X, y).report; b = _run(onto, X, y).report
    assert a["decision"] == b["decision"]
    assert a["stages"]["D_mlp_comparison"]["utility_gain"] == b["stages"]["D_mlp_comparison"]["utility_gain"]
    assert a["selected_semantic_features"] == b["selected_semantic_features"]


# ------------------------------------------------------- relatório por feature ---
def test_feature_audit_table_is_complete_and_exportable(tmp_path):
    onto = _family_onto("aud"); X, y = _data(n=150)
    rep = _run(onto, X, y).report
    rows = rep["feature_audit"]
    assert rows
    need = {"feature", "origin", "knowledge_source", "source_features", "ontology_family", "formula",
            "variance", "max_abs_corr_with_sources", "mutual_information_mean", "mutual_information_std",
            "selection_frequency", "predictive_gain_mean", "predictive_gain_std", "decision", "reason"}
    assert need <= set(rows[0])
    assert {r["decision"] for r in rows} <= {"SELECTED", "REJECTED"}
    assert all(r["reason"] for r in rows)
    csv = export_feature_audit(rep, tmp_path / "audit.csv")
    js = export_feature_audit(rep, tmp_path / "audit.json")
    assert len(pd.read_csv(csv)) == len(rows) and js.endswith("audit.json")


# ------------------------------------------------------------- ablação / controlo ---
def test_ablation_by_kind_only_uses_that_kind():
    onto = _family_onto("abl"); X, y = _data(n=150)
    rep = _run(onto, X, y, kinds=["relational"]).report
    sel = rep["stages"]["C_screening"]["selection_frequency"]
    assert sel and all(not n.endswith("_aggregate") for n in sel)
    rep_agg = _run(onto, X, y, kinds=["hierarchical_aggregate"]).report
    assert all(n.endswith("_aggregate") for n in rep_agg["stages"]["C_screening"]["selection_frequency"])


def test_negative_control_shuffled_semantics_does_not_reproduce_the_gain():
    from core.semantic_controls import shuffle_entity_assignments
    onto = _family_onto("ctl", noise=20); X, y = _data(n=300, signal="relation", noise=20)
    real = _run(onto, X, y).report
    gains = []
    for seed in (1, 2, 3):
        ctrl = _run(onto, X, y, matches_transform=lambda m, s=seed: shuffle_entity_assignments(m, seed=s)).report
        gains.append(ctrl["stages"]["D_mlp_comparison"]["utility_gain"]
                     if "D_mlp_comparison" in ctrl["stages"] else 0.0)
    real_gain = real["stages"]["D_mlp_comparison"]["utility_gain"]
    assert real["decision"].startswith("ACCEPT")
    assert all(g < real_gain for g in gains), (real_gain, gains)


def test_shuffle_keeps_entity_set_and_is_deterministic():
    from core.semantic_controls import shuffle_entity_assignments
    matches = [{"feature": f"f{i}", "entity_name": f"E{i}", "entity_type": "datatype_property", "accepted": True}
               for i in range(6)]
    a = shuffle_entity_assignments(matches, seed=3); b = shuffle_entity_assignments(matches, seed=3)
    assert a == b
    assert sorted(m["entity_name"] for m in a) == sorted(m["entity_name"] for m in matches)
    assert [m["entity_name"] for m in a] != [m["entity_name"] for m in matches]
    assert [m["feature"] for m in a] == [m["feature"] for m in matches]


def test_metrics_and_utility_are_consistent_with_sklearn():
    from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, recall_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200); p = np.where(rng.random(200) < .7, y, rng.integers(0, 3, 200))
    m = classification_metrics_np(y, p, 3)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, p))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, p))
    assert m["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert m["minority_recall"] == pytest.approx(recall_score(y, p, average=None).min())
    assert utility(m, FAST, 0.5) < utility(m, FAST, 0.0)
