"""Doc. 43: correção isolada do matching ambíguo (política P3) em ``core/ontology_quality.py``. Contrato anterior preservado: matching por nome/rótulo, score mínimo 0,72, margem
mínima 0,08, preferência de propriedade de dados sobre classe em empate exato classe/propriedade; nenhuma exceção nova de score; nenhuma dependência de datasets."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pytest.importorskip("owlready2")
from core.ontology_quality import OntologyQualityGate
from core.owl_runtime import load_ontology_isolated
from validation.benchmark.b30_axiom_probe import dprop, klass, write

ROOT_CLASS = klass("Root")


def match(tmp, body, features, name="o"):
    onto = load_ontology_isolated(str(write(tmp, name, ROOT_CLASS + body)))
    return OntologyQualityGate().match_features(list(features), onto)


def test_single_exact_candidate_is_accepted(tmp_path):
    d = match(tmp_path, dprop("thermal_mean", "Root") + dprop("optical_range", "Root"), ["thermal_mean"])[0]
    assert d.accepted and d.status == "ACCEPTED" and d.entity_name == "thermal_mean" and d.score == 1.0 and d.ambiguity_diagnostic is None


def test_two_data_properties_with_score_1_are_rejected_as_ambiguous(tmp_path):
    d = match(tmp_path, dprop("thermal_mean_a", "Root", label="thermal_mean") + dprop("thermal_mean_b", "Root", label="thermal_mean"), ["thermal_mean"])[0]
    assert not d.accepted and d.status == "REJECTED_AMBIGUOUS" and d.reason == "ambiguous_match"
    dg = d.ambiguity_diagnostic
    assert dg["kind"] == "tie_between_entities" and dg["n_tied"] == 2 and {e["name"] for e in dg["tied_entities"]} == {"thermal_mean_a", "thermal_mean_b"}


def test_class_and_data_property_tie_applies_documented_type_priority(tmp_path):
    d = match(tmp_path, klass("ThermalMean", ["Root"]) + dprop("hasThermalMean", "ThermalMean", label="ThermalMean"), ["thermal_mean"])[0]
    assert d.accepted and d.entity_type == "datatype_property" and d.entity_name == "hasThermalMean" and d.tie_break == "datatype_property_over_class"


def test_three_tied_candidates_are_never_decided_by_owl_position(tmp_path):
    body = klass("ThermalMean", ["Root"]) + dprop("hasA", "ThermalMean", label="ThermalMean") + dprop("hasB", "ThermalMean", label="ThermalMean")
    d = match(tmp_path, body, ["thermal_mean"])[0]
    assert not d.accepted and d.ambiguity_diagnostic["kind"] == "tie_between_entities" and d.ambiguity_diagnostic["n_tied"] == 3
    body3 = dprop("pa", "Root", label="thermal_mean") + dprop("pb", "Root", label="thermal_mean") + dprop("pc", "Root", label="thermal_mean")
    d3 = match(tmp_path, body3, ["thermal_mean"], "o3")[0]
    assert not d3.accepted and d3.ambiguity_diagnostic["n_tied"] == 3
    two_classes = klass("ThermalMean", ["Root"]) + klass("Thermal_Mean", ["Root"])
    assert not match(tmp_path, two_classes, ["thermal_mean"], "o4")[0].accepted                     # ≥ 2 classes empatadas sem propriedade: não há prioridade de tipo


def test_type_priority_still_requires_margin_against_remaining_competitors(tmp_path):
    base = klass("ThermalMean", ["Root"]) + dprop("hasThermalMean", "ThermalMean", label="ThermalMean")
    competitor = dprop("meanThermal", "Root")                                                         # mesmos termos, outra ordem ⇒ 0,97 (a < 0,08 do escolhido)
    d = match(tmp_path, base + competitor, ["thermal_mean"], "oc")[0]
    assert not d.accepted and d.ambiguity_diagnostic["kind"] == "competitor_after_type_priority" and d.ambiguity_diagnostic["competitor"] == "meanThermal"
    ok = match(tmp_path, base + dprop("unrelated_value", "Root"), ["thermal_mean"], "od")[0]
    assert ok.accepted and ok.tie_break == "datatype_property_over_class"                             # concorrente distante: a exceção continua a aplicar-se


def test_difference_below_min_margin_is_rejected_and_sufficient_margin_accepted(tmp_path):
    gate = OntologyQualityGate()
    onto = load_ontology_isolated(str(write(tmp_path, "m1", ROOT_CLASS + dprop("temperature_level", "Root") + dprop("temperatur_level", "Root"))))
    ents = {e.name: e for e in onto.data_properties()}
    s1, s2 = gate._candidate_score("temperature_leve", ents["temperature_level"]), gate._candidate_score("temperature_leve", ents["temperatur_level"])
    assert s1 < 0.98 and abs(s1 - s2) < 0.08                                                          # pré-condição: melhor < 0,98 e diferença < 0,08
    d = OntologyQualityGate().match_features(["temperature_leve"], onto)[0]
    assert not d.accepted and d.status == "REJECTED_AMBIGUOUS" and d.ambiguity_diagnostic["kind"] == "insufficient_margin"
    ok = match(tmp_path, dprop("temperature_level", "Root") + dprop("temp_level", "Root"), ["temperature_leve"], "m2")[0]
    assert ok.accepted and ok.score < 0.98 and ok.ambiguity_margin >= 0.08 and ok.entity_name == "temperature_level"         # margem suficiente (0,86 vs 0,60)


def test_decisions_do_not_depend_on_declaration_order(tmp_path):
    a = dprop("thermal_mean_a", "Root", label="thermal_mean") + dprop("thermal_mean_b", "Root", label="thermal_mean")
    b = dprop("thermal_mean_b", "Root", label="thermal_mean") + dprop("thermal_mean_a", "Root", label="thermal_mean")
    da, db = match(tmp_path, a, ["thermal_mean"], "a")[0], match(tmp_path, b, ["thermal_mean"], "b")[0]
    assert not da.accepted and not db.accepted and da.ambiguity_diagnostic == db.ambiguity_diagnostic
    cp1 = klass("ThermalMean", ["Root"]) + dprop("hasX", "ThermalMean", label="ThermalMean") + dprop("hasFar", "Root", label="far_away_thing")
    cp2 = dprop("hasFar", "Root", label="far_away_thing") + dprop("hasX", "Root", label="ThermalMean") + klass("ThermalMean", ["Root"])
    e1, e2 = match(tmp_path, cp1, ["thermal_mean"], "c1")[0], match(tmp_path, cp2, ["thermal_mean"], "c2")[0]
    assert e1.accepted and e2.accepted and e1.entity_name == e2.entity_name == "hasX"


def test_two_features_wanting_the_same_entity_are_rejected_as_collision(tmp_path):
    ds = match(tmp_path, dprop("thermal_mean", "Root") + dprop("optical_range", "Root"), ["thermal_mean", "ThermalMean"])
    assert [d.status for d in ds] == ["REJECTED_ENTITY_COLLISION", "REJECTED_ENTITY_COLLISION"] and not any(d.accepted for d in ds)
    ds2 = match(tmp_path, dprop("thermal_mean", "Root") + dprop("optical_range", "Root"), ["optical_range", "thermal_mean"], "k2")
    assert all(d.accepted for d in ds2)


def test_alternative_labels_match_and_label_ties_are_ambiguous(tmp_path):
    d = match(tmp_path, dprop("x1", "Root", label="mean_temperature_value"), ["mean_temperature_value"])[0]
    assert d.accepted and d.entity_name == "x1"
    tie = match(tmp_path, dprop("x1", "Root", label="mean_temperature_value") + dprop("x2", "Root", label="mean_temperature_value"), ["mean_temperature_value"], "t2")[0]
    assert not tie.accepted and tie.ambiguity_diagnostic["kind"] == "tie_between_entities"


def test_normalized_names_match_exactly(tmp_path):
    for feat in ("Thermal Mean", "THERMAL-MEAN", "thermalMean", "thermal.mean"):
        d = match(tmp_path, dprop("thermal_mean", "Root") + dprop("optical_range", "Root"), [feat], "n" + str(abs(hash(feat)) % 1000))[0]
        assert d.accepted and d.entity_name == "thermal_mean", feat


def test_unknown_ontology_and_empty_ontology_reject_without_error(tmp_path):
    d = match(tmp_path, dprop("zorvanqux", "Root") + dprop("blorptrin", "Root"), ["thermal_mean"], "u")[0]
    assert not d.accepted and d.status == "REJECTED_LOW_SCORE" and d.ambiguity_diagnostic is None
    empty = load_ontology_isolated(str(write(tmp_path, "e", "")))
    ds = OntologyQualityGate().match_features(["a", "b"], empty)
    assert len(ds) == 2 and not any(x.accepted for x in ds)


def test_historical_ontologies_keep_identical_mappings():
    ev = json.loads((ROOT / "docs/evidence/b1b4/b33/matching_p3_historical_mappings.json").read_text())
    assert {"iris", "wine", "breast_cancer", "digits", "synthetic_binary", "synthetic_3class", "german_credit"} <= set(ev)
    for did, row in ev.items():
        assert row["identical_decisions"] == row["n_features"] == row["n_accepted"] and row["n_with_diagnostic"] == 0 and row["gate_report_equal_except_matches"], did


def test_historical_ontologies_decisions_are_still_the_frozen_ones_live():
    pytest.importorskip("sklearn")
    from validation.benchmark import datasets as D
    for did in ("iris", "wine", "synthetic_binary"):
        spec = D.REGISTRY[did]
        ds, _, _ = spec.load()
        names = [str(n) for n in ds.feature_names]
        ds_ = OntologyQualityGate().match_features(names, load_ontology_isolated(str(spec.ontology_path)))
        assert all(d.accepted and d.ambiguity_diagnostic is None for d in ds_), did
