"""H2 (Docs. 53–57): coerência da auditoria das fichas/inferências/literais, das fichas de revisão (campos humanos vazios) e das propostas de limiares. Só leitura de evidência; sem benchmark."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "docs/evidence/b1b4/b3q"
V0 = ROOT / "validation/benchmark/ontologies/proposals/german_credit/v0_draft"


def test_inference_and_literal_audit_is_consistent_with_cards_and_open_decisions():
    a = json.loads((EV / "h2_inference_audit.json").read_text())
    assert a["labels_used"] is False and a["test_rows_used"] is False
    r = a["reasoner"]
    assert r["n_inferred"] == 55 and r["by_kind"] == {"CONCEPT_SUBSUMPTION": 25, "CONCEPT_EQUIVALENCE": 4, "ATTRIBUTE_APPLIES_TO_CONCEPT": 26}   # 25 + 4 = 29 = axiomas inferidos pelo HermiT
    assert sum(r["inferred_by_type"].values()) == 51 and r["n_inferred"] == 51 + 4                                                           # 4 equivalências triviais (união de um nível)
    assert len(a["cards"]) == 14 and not any(c["evaluable_now"] for c in a["cards"])                                                          # sem fonte verificada: conjunto avaliável vazio
    assert all(c["documentation_status"] == "UNVERIFIED_NO_LOCAL_DOCUMENT" for c in a["cards"])
    assert sum(c["decision"].startswith("TRIVIAL_SINGLETON") for c in a["cards"]) == 4
    assert all(c["decision"].startswith("EXCLUDE_NO_DOCUMENTARY_BASIS") for c in a["cards"] if "TAX" in c["card"])
    assert a["n_literals"] == 10 and all(l["11_original_can_generate"]["in_hypothesis_space_with_max_n_3"] for l in a["literals"])           # todos no espaço do Original
    assert sorted(a["trivial_or_redundant"]["trivial_negation_literals"]) == ["onto_any[AtLeast_employment_since_A72]", "onto_any[AtLeast_job_A172]"]
    assert len(a["nested_pairs_same_variable"]) == 5


def test_reasoner_check_all_variants_consistent_and_satisfiable():
    r = json.loads((EV / "h2_german_reasoner_check.json").read_text())
    assert set(r) == {"O1", "O2", "O3", "H", "I"}
    assert all(v["consistent"] is True and v["unsatisfiable_classes"] == [] for v in r.values())
    assert r["O1"]["inferred_axioms_count"] == 0 and r["O3"]["inferred_axioms_count"] == 0 and r["O2"]["inferred_axioms_count"] == 29


def test_review_forms_exist_for_three_roles_with_blank_human_fields():
    forms = {p.name: p.read_text(encoding="utf-8") for p in (V0 / "review_forms").glob("*.md")}
    assert set(forms) == {"domain_review.md", "ontology_review.md", "methodology_review.md"}
    for name, t in forms.items():
        assert t.count("a preencher por pessoa designada — vazio") >= 4                              # nome, data, assinatura, decisão
        assert "APPROVED" not in re.sub(r"\(`APPROVED` / `REJECTED` / `CHANGES_REQUESTED`\)", "", t)    # nenhuma aprovação escrita
    assert "DOMAIN_REVIEW_PENDING" in forms["domain_review.md"] and "ONTOLOGY_REVIEW_PENDING" in forms["ontology_review.md"] and "METHODOLOGY_REVIEW_PENDING" in forms["methodology_review.md"]


def test_threshold_calibration_is_independent_and_supports_documented_ranges():
    c = json.loads((EV / "threshold_calibration.json").read_text())
    assert c["seed_base"] == 9500 and c["reps"] == 30 and "sem ontologia" in c["labels_used"]
    res = c["results"]
    assert len(res) == 16 and all(v["frac_Do_gt0"] == 1.0 for v in res.values())
    assert 0.05 <= min(v["Do_mean"] for v in res.values()) <= 0.06 and max(v["Do_mean"] for v in res.values()) < 0.34
    assert max(v["Do_sd"] for v in res.values()) < 0.026 and all(abs(v["ari_median"]) < 0.05 for v in res.values())          # indicadores independentes: ARI ≈ 0
    ad = json.loads((EV / "oracle_adequacy.json").read_text())["results"]
    assert min(v["p10"] for v in ad.values()) >= 0.44 and all(v["chance_share"] == 0.125 for v in ad.values())
    decision = json.loads((ROOT / "docs/b1b4/manifest_v3/B34_EXPERIMENT_MANIFEST.v2.DRAFT.json").read_text())
    assert decision["frozen"] is False and "R1_R7" in decision["decision_record"] and "NÃO autorizada" in decision["decision_record"]["H1_campaign_7.2_CPU_h"]


def test_german_levels_evidence_is_x_only_and_codes_are_exclusive():
    lv = json.loads((EV / "german_levels_x_only.json").read_text())
    assert "nenhum rótulo" in lv["note"]
    for var, v in lv["variables"].items():
        assert v["rows_with_no_level"] == 0 and v["rows_with_multiple_levels"] == 0, var
    assert "A47" not in lv["variables"]["purpose"]["levels_present_in_data"]
