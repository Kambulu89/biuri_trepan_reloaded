"""H2 (Doc. 62): verificação parcial do subconjunto OWL usado (com mutation tests do verificador) e invariantes da verificação de pressupostos da regra de decisão das portas."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from validation.benchmark import h2_owl_profile as OP
from validation.benchmark import h2_synthetic as S

FAM = {"alpha": 4, "beta": 3}


def _write(tmp_path, text, name="o.owl"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_subset_checker_accepts_the_generated_ontologies(tmp_path):
    X, names, _ = S.make_generic_data(40, 1, FAM, 2, 1, k_true=2)
    for v in ("TRUE", "PERM", "NEUTRAL", "FLAT"):
        r = OP.check(_write(tmp_path, S.build_generic_owl(names, FAM, 2, v), f"{v}.owl"))
        assert r["subset_check"] == "PASS", (v, r["problems"])
    assert "NÃO é validação formal OWL 2 DL" in r["scope"]


def test_subset_checker_detects_mutations(tmp_path):
    X, names, _ = S.make_generic_data(40, 1, FAM, 2, 1, k_true=2)
    good = S.build_generic_owl(names, FAM, 2, "TRUE")
    mutants = {
        "construção fora da lista": good.replace("</rdf:RDF>", '<owl:ObjectProperty rdf:about="#p"/></rdf:RDF>'),
        "união com membro não declarado": good.replace('<owl:Class rdf:about="#Lvl_alpha_L3"/>', '<owl:Class rdf:about="#Inexistente"/>', 1),
        "domínio não declarado": good.replace('<rdfs:domain rdf:resource="#G_noise"/>', '<rdfs:domain rdf:resource="#Fantasma"/>', 1),
    }
    for label, text in mutants.items():
        if text == good:
            continue
        assert OP.check(_write(tmp_path, text, "m.owl"))["subset_check"] == "FAIL", label
    assert any(OP.check(_write(tmp_path, t, f"m{i}.owl"))["problems"] for i, t in enumerate(mutants.values()) if t != good)


def test_evidence_for_the_five_draft_ontologies_passes():
    ev = json.loads((ROOT / "docs/evidence/b1b4/h2/owl_subset_check.json").read_text())
    assert len(ev["checks"]) == 5 and all(c["subset_check"] == "PASS" and c["datatype_properties"] == 61 for c in ev["checks"])


def test_threshold_rule_check_invariants():
    d = json.loads((ROOT / "docs/evidence/b1b4/b3q/threshold_rule_check.json").read_text())
    assert d["all_values_in_grid"] and d["reps"] == 20
    oc = d["operating_characteristic_normal_rule_n20"]
    assert all(oc[t]["0.0"] < 0.08 for t in oc)                                                       # nível real próximo do nominal (5 %) mas NÃO garantido
    assert all(oc[t]["0.1"] < 0.01 for t in oc)                                                      # afastada da fronteira, a regra quase nunca aprova indevidamente
    assert set(d["rules"]) == {"normal_1.645", "t_19_0.95", "bonferroni_36", "bootstrap_95"}
    assert all(0 < v < 1 for v in d["mean_pairwise_correlation_between_estimators_within_cell"].values())   # estimadores NÃO são independentes dentro da réplica
