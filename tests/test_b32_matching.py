"""B3.2: diagnóstico do matching ambíguo — reprodução do defeito do gate (não alterado), propriedades das políticas candidatas e retrocompatibilidade nas ontologias congeladas."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pytest.importorskip("owlready2")
from validation.benchmark.b30_axiom_probe import dprop, klass, write
from validation.benchmark.b32_matching import decide, decisions_for, impact, reproduce


def test_current_gate_accepts_perfect_tie_and_choice_depends_on_declaration_order(tmp_path):
    r = reproduce(tmp_path)
    t1, t2 = r["tie_order1"], r["tie_order2"]
    assert t1["accepted"] and t2["accepted"] and t1["ambiguity_margin"] == 0.0 and t1["score"] == t1["runner_up_score"] == 1.0
    assert t1["entity_name"] != t2["entity_name"]                    # a mesma ontologia, com a ordem trocada, devolve OUTRA entidade: escolha arbitrária
    assert {t1["entity_name"], t1["runner_up_entity"]} == {t2["entity_name"], t2["runner_up_entity"]}


def _d(score, margin, accepted=True, tie_break=None):
    return {"score": score, "ambiguity_margin": margin, "accepted": accepted, "tie_break": tie_break}


def test_policy_p1_rejects_ties_deterministically_but_keeps_documented_type_preference():
    assert decide(_d(1.0, 0.0), "P0") is True
    assert decide(_d(1.0, 0.0), "P1") is False and decide(_d(1.0, 0.0), "P2") is False
    assert decide(_d(1.0, 0.0, tie_break="datatype_property_over_class"), "P1") is True        # regra declarada (atributo = propriedade), determinística
    assert decide(_d(1.0, 0.14), "P1") is True and decide(_d(1.0, 0.03), "P2") is True          # casamento exato único mantém-se (ex.: nomes de píxeis)
    assert decide(_d(0.99, 0.03), "P0") is True and decide(_d(0.99, 0.03), "P1") is True and decide(_d(0.99, 0.03), "P2") is False
    assert decide(_d(0.8, 0.0, accepted=False), "P2") is False                                  # nunca aceita o que o gate rejeitou


def test_policy_p1_is_order_invariant(tmp_path):
    a = klass("Root") + klass("A", ["Root"]) + klass("B", ["Root"]) + dprop("thermal_mean_a", "A", label="thermal_mean") + dprop("thermal_mean_b", "B", label="thermal_mean")
    b = klass("Root") + klass("A", ["Root"]) + klass("B", ["Root"]) + dprop("thermal_mean_b", "B", label="thermal_mean") + dprop("thermal_mean_a", "A", label="thermal_mean")
    r1 = decide(decisions_for(write(tmp_path, "o1", a), ["thermal_mean"])[0], "P1")
    r2 = decide(decisions_for(write(tmp_path, "o2", b), ["thermal_mean"])[0], "P1")
    assert r1 is False and r2 is False                                                           # mesma decisão independentemente da ordem de declaração


def test_candidate_policies_do_not_change_any_frozen_ontology_matching():
    res = impact()
    assert {"iris", "wine", "breast_cancer", "digits", "synthetic_binary", "synthetic_3class", "german_credit"} <= set(res)
    for did, row in res.items():
        assert row["changed_P1"] == 0 and row["changed_P2"] == 0 and row["n_accepted_with_exact_tie"] == 0, did


def test_three_way_tie_is_masked_by_type_preference_only_p3_rejects_it(tmp_path):
    r = reproduce(tmp_path)["three_way_tie_class_and_two_properties"]
    assert r["accepted"] and r["tie_break"] == "datatype_property_over_class" and r["tie_sets"]["n_top"] == 3 and r["tie_sets"]["n_top_datatype_property"] == 2
    assert r["P1_accepts"] is True and r["P3_accepts"] is False              # P1 não cobre empates de 3 vias; P3 sim


def test_p3_changes_nothing_in_frozen_ontologies():
    assert all(row["changed_P3"] == 0 for row in impact().values())
