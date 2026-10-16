"""B3.1: registo de factos semânticos (tipos, proveniência, estatutos, mundo aberto), propagação de inferências e relatedness taxonómica (validation/ apenas).
Ontologias sintéticas OWL 2 (RDF/XML) controladas; sem datasets conhecidos; núcleo não alterado."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pytest.importorskip("owlready2")

from validation.benchmark.b30_axiom_probe import dprop, klass, write
from validation.benchmark.b31_facts import (Derivation, Epistemic, Kind, PermittedUse, SemanticFact, UseState, add_documented, compile_registry, decide_permitted_use,
                                            is_eliminable, observe_order_relation, ORDER_VALIDITY_CONDITIONS)
from validation.benchmark.b31_propagation import (BLOCKED_BY_MATCHING, DISCARDED, IRRELEVANT_TO_FEATURES, REACHED_CANDIDATE, RECOGNIZED_EXPLICIT, RECOGNIZED_INFERRED, propagation_report)
from validation.benchmark.b31_relatedness import (DISTINCT_DISJOINT, NO_EVIDENCE, RELATED, build_index, legacy_matrix, pair_relatedness, relatedness_matrix, semantic_units)


def _java():
    from core.ontology_reasoner import find_java_executable
    try:
        return find_java_executable()
    except Exception:
        return None


needs_java = pytest.mark.skipif(_java() is None, reason="Java/HermiT indisponível")
SV = '<owl:Restriction><owl:onProperty rdf:resource="#hasPart"/><owl:someValuesFrom rdf:resource="#B"/></owl:Restriction>'
INF = ('<owl:ObjectProperty rdf:about="#hasPart"/>\n' + klass("Root") + klass("B", ["Root"]) + '<owl:Class rdf:about="#A"><rdfs:subClassOf rdf:resource="#Root"/><owl:equivalentClass>' + SV
       + '</owl:equivalentClass></owl:Class>\n<owl:Class rdf:about="#C"><rdfs:subClassOf rdf:resource="#Root"/><rdfs:subClassOf>' + SV + '</rdfs:subClassOf></owl:Class>\n')
F4 = ["temperature_mean", "temperature_peak", "luminosity_level", "absent_attribute"]


@pytest.fixture(scope="module")
def inferred_reg(tmp_path_factory):
    t = tmp_path_factory.mktemp("b31")
    body = INF + klass("UnusedConcept", ["Root"]) + dprop("temperature_mean", "A") + dprop("temperature_peak", "C") + dprop("luminosity_level", "B")
    return compile_registry(write(t, "inf", body), F4)


@pytest.fixture(scope="module")
def taxo(tmp_path_factory):
    t = tmp_path_factory.mktemp("b31t")
    body = (klass("Root") + klass("Mid1", ["Root"]) + klass("Mid2", ["Root"], '<owl:disjointWith rdf:resource="#Mid1"/>') + klass("L1", ["Mid1"]) + klass("L2", ["Mid1"]) + klass("L3", ["Mid2"])
            + dprop("temperature_mean", "L1") + dprop("temperature_peak", "L1") + dprop("humidity_level", "L2") + dprop("pressure_value", "L3")
            + dprop("pressure_reading", "L3", '<owl:equivalentProperty rdf:resource="#pressure_value"/>') + dprop("light_level", "L2", '<rdfs:subPropertyOf rdf:resource="#humidity_level"/>'))
    F = ["temperature_mean", "temperature_peak", "humidity_level", "pressure_value", "pressure_reading", "light_level"]
    return write(t, "taxo", body), F, compile_registry(write(t, "taxo2", body), F)


# ---------------------------------------------------------------------------------------------------- registo: campos, proveniência, estatutos
def test_every_fact_has_required_typed_fields(inferred_reg):
    assert inferred_reg.ontology_status == "VALID" and inferred_reg.facts
    for f in inferred_reg.facts:
        assert f.id.startswith("F-") and f.kind in {k.value for k in Kind}
        assert f.epistemic in {e.value for e in Epistemic} and f.derivation in {d.value for d in Derivation}
        assert f.permitted_use in {u.value for u in PermittedUse} and f.use_state in {u.value for u in UseState}
        assert f.provenance.get("ontology_sha256") and f.provenance.get("compiler")
        assert isinstance(f.scope, dict) and "level" in f.scope
        assert f.reasoner_result is not None and f.reasoner_result["executed"] is True
        assert f.source_axioms, f.kind
        if f.use_state == UseState.REJECTED.value:
            assert f.rejection_reason
    assert len({f.id for f in inferred_reg.facts}) == len(inferred_reg.facts)             # ids únicos e estáveis


def test_ids_are_deterministic(tmp_path):
    body = INF + dprop("temperature_mean", "A") + dprop("temperature_peak", "C")
    a = compile_registry(write(tmp_path, "d1", body), F4[:2])
    b = compile_registry(write(tmp_path, "d2", body), F4[:2])
    assert [f.id for f in a.facts] == [f.id for f in b.facts]


def test_logical_entailment_is_separate_from_domain_and_data_statuses(inferred_reg):
    assert {f.epistemic for f in inferred_reg.facts} <= {Epistemic.LOGICALLY_ENTAILED.value, Epistemic.UNKNOWN.value}
    assert all(f.epistemic == Epistemic.UNKNOWN.value for f in inferred_reg.by_kind(Kind.MATCH_FAILURE, Kind.UNSUPPORTED_AXIOM))


# ---------------------------------------------------------------------------------------------------- mundo aberto e contratos de aplicação
def test_no_fact_authorises_elimination_from_ontology_alone(inferred_reg):
    assert not any(is_eliminable(f) for f in inferred_reg.facts)
    assert all(f.permitted_use != PermittedUse.HARD_CONSTRAINT.value for f in inferred_reg.facts)


def test_domain_range_and_disjointness_are_not_closed_validators(tmp_path):
    body = (klass("Root") + klass("P", ["Root"], '<owl:disjointWith rdf:resource="#Q"/>') + klass("Q", ["Root"])
            + dprop("measure_alpha", "P", extra='') + dprop("measure_beta", "Q"))
    reg = compile_registry(write(tmp_path, "dj", body), ["measure_alpha", "measure_beta"])
    dj = reg.by_kind(Kind.CONCEPT_DISJOINTNESS)
    assert dj and all(f.permitted_use == PermittedUse.NONE.value and f.application_restrictions for f in dj)
    for f in reg.by_kind(Kind.ATTRIBUTE_APPLIES_TO_CONCEPT):
        assert "não é validador fechado de linhas" in " ".join(f.application_restrictions)


def test_value_bounds_need_row_contract(tmp_path):
    from validation.benchmark.b30_axiom_probe import axiom_cases
    body = axiom_cases()["datatypeFacets"]
    reg = compile_registry(write(tmp_path, "fac", body), ["f1", "f2", "f3"], run_reasoner=False)
    vb = reg.by_kind(Kind.VALUE_BOUND)
    # nomes placeholder: o gate rejeita; os factos ficam rejeitados e nunca autorizam eliminação
    assert all(not is_eliminable(f) for f in reg.facts)
    for f in vb:
        assert f.permitted_use == PermittedUse.NONE.value and "requires_row_as_individual_contract" in f.application_restrictions


def test_open_world_absence_is_not_negation(taxo):
    _, F, reg = taxo
    ix = build_index(reg, F)
    r = pair_relatedness(ix, "temperature_mean", "pressure_value")
    assert r.state in (DISTINCT_DISJOINT, NO_EVIDENCE) and r.score == 0.0
    # nenhum facto negativo é criado para pares sem relação; só a disjunção DECLARADA gera facto
    assert {f.kind for f in reg.facts if f.kind == Kind.CONCEPT_DISJOINTNESS.value} == {Kind.CONCEPT_DISJOINTNESS.value}
    assert len(reg.by_kind(Kind.CONCEPT_DISJOINTNESS)) == 1


def test_missing_domain_axiom_yields_no_evidence_not_incompatibility(tmp_path):
    body = klass("Root") + dprop("temperature_mean") + dprop("temperature_peak") + dprop("luminosity_level")
    reg = compile_registry(write(tmp_path, "nd", body), ["temperature_mean", "temperature_peak", "luminosity_level"])
    ix = build_index(reg, ["temperature_mean", "temperature_peak", "luminosity_level"])
    assert pair_relatedness(ix, "temperature_mean", "temperature_peak").state == NO_EVIDENCE
    assert not semantic_units(reg, ["temperature_mean", "temperature_peak", "luminosity_level"])


# ---------------------------------------------------------------------------------------------------- reasoner e propagação
@needs_java
def test_reasoner_inference_reaches_attribute_facts(inferred_reg):
    inf = [f for f in inferred_reg.by_kind(Kind.CONCEPT_SUBSUMPTION) if f.derivation == Derivation.INFERRED_BY_REASONER.value]
    assert [f.entities for f in inf] == [("C", "A")]
    reach = [f for f in inferred_reg.by_kind(Kind.ATTRIBUTE_APPLIES_TO_CONCEPT) if f.attributes == ("temperature_peak",) and f.entities[1] == "A"]
    assert reach and reach[0].derivation == Derivation.INFERRED_BY_REASONER.value and "C SubClassOf A [inferred]" in " ".join(reach[0].source_axioms)
    u = {x.concept: x for x in semantic_units(inferred_reg, F4)}
    assert set(u["A"].features) == {"temperature_mean", "temperature_peak"} and Derivation.INFERRED_BY_REASONER.value in u["A"].derivations      # a inferência chega ao candidato


@needs_java
def test_propagation_report_distinguishes_outcomes(inferred_reg):
    units = semantic_units(inferred_reg, F4)
    rep = propagation_report(inferred_reg, F4, units, [fid for u in units for fid in u.fact_ids][:1])
    d = rep["by_destination"]
    assert d.get(BLOCKED_BY_MATCHING) == 1                                  # absent_attribute: o matching impede
    assert d.get(RECOGNIZED_EXPLICIT, 0) > 0 and d.get(RECOGNIZED_INFERRED, 0) > 0 and d.get(REACHED_CANDIDATE, 0) == 1
    assert d.get(DISCARDED, 0) >= 1                                          # someValuesFrom: reconhecido mas não interpretado
    assert d.get(IRRELEVANT_TO_FEATURES, 0) >= 1                             # UnusedConcept não afeta nenhum atributo
    assert rep["inference_status"] == "PROPAGATED"


def test_no_new_inferences_is_not_a_propagation_failure(tmp_path):
    body = klass("Root") + klass("A", ["Root"]) + klass("B", ["Root"]) + dprop("temperature_mean", "A") + dprop("temperature_peak", "A") + dprop("luminosity_level", "B")
    F = ["temperature_mean", "temperature_peak", "luminosity_level"]
    reg = compile_registry(write(tmp_path, "ni", body), F)
    rep = propagation_report(reg, F, semantic_units(reg, F))
    assert rep["n_inferred_facts"] == 0 and rep["inference_status"] == "NO_NEW_INFERENCES"


def test_inconsistent_ontology_yields_no_usable_facts(tmp_path):
    body = (klass("Root") + klass("P", ["Root"]) + klass("Q", ["Root"], '<owl:disjointWith rdf:resource="#P"/>') + klass("Bad", ["P", "Q"])
            + dprop("temperature_mean", "P") + dprop("temperature_peak", "P") + dprop("luminosity_level", "Q"))
    F = ["temperature_mean", "temperature_peak", "luminosity_level"]
    reg = compile_registry(write(tmp_path, "inc", body), F)
    assert reg.ontology_status == "INCONSISTENT"
    assert reg.usable() == [] and all(f.use_state == UseState.REJECTED.value for f in reg.facts)
    assert semantic_units(reg, F) == []


def test_matching_failure_blocks_use(tmp_path):
    body = klass("Root") + klass("A", ["Root"]) + dprop("zorvan_alpha", "A") + dprop("zorvan_beta", "A") + dprop("temperature_mean", "A")
    reg = compile_registry(write(tmp_path, "mm", body), ["temperature_mean", "pressure_unknown"])
    mf = reg.by_kind(Kind.MATCH_FAILURE)
    assert [f.attributes for f in mf] == [("pressure_unknown",)] and mf[0].use_state == UseState.REJECTED.value
    assert not [f for f in reg.by_kind(Kind.ATTRIBUTE_APPLIES_TO_CONCEPT) if f.attributes == ("pressure_unknown",)]


# ---------------------------------------------------------------------------------------------------- relações numéricas documentadas / empíricas
def _doc(**over):
    d = {"kind": Kind.ORDER_RELATION.value, "attributes": ["max_signal", "mean_signal"], "relation_class": "FORMAL_DOMAIN_CONSTRAINT",
         "source": {"citation": "Definição estatística documentada (máximo ≥ média do mesmo conjunto)", "version": "1"},
         "validity_conditions": {c: True for c in ORDER_VALIDITY_CONDITIONS}}
    d.update(over)
    return d


def test_documented_order_relation_requires_source_conditions_and_human_review(taxo):
    _, F, reg = taxo
    prov = {"ontology_sha256": reg.ontology_hash}
    pending = add_documented(reg, _doc(), prov)
    assert pending.epistemic == Epistemic.DOMAIN_DOCUMENTED.value and pending.review["domain_expert_status"] == "DOMAIN_REVIEW_PENDING"
    assert pending.permitted_use == PermittedUse.PREFERENCE.value and not is_eliminable(pending)          # sem revisão humana não é restrição obrigatória
    approved = add_documented(reg, _doc(attributes=["max_b", "mean_b"], review={"domain_expert_status": "APPROVED", "methodology_status": "APPROVED"}), prov)
    assert approved.permitted_use == PermittedUse.HARD_CONSTRAINT.value and is_eliminable(approved)       # só com revisões humanas fornecidas
    assert not any(c in approved.review.values() for c in ("claude",))
    missing = add_documented(reg, _doc(attributes=["max_c", "mean_c"], validity_conditions={"same_quantity": True}), prov)
    assert missing.epistemic == Epistemic.HYPOTHESIS.value and not is_eliminable(missing)
    nosrc = add_documented(reg, _doc(attributes=["max_d", "mean_d"], source={}), prov)
    assert nosrc.epistemic == Epistemic.HYPOTHESIS.value and nosrc.permitted_use == PermittedUse.NONE.value
    empirical = add_documented(reg, _doc(attributes=["max_e", "mean_e"], relation_class="EMPIRICAL_REGULARITY", review={"domain_expert_status": "APPROVED", "methodology_status": "APPROVED"}), prov)
    assert empirical.permitted_use == PermittedUse.NONE.value or empirical.permitted_use == PermittedUse.PREFERENCE.value
    assert empirical.permitted_use != PermittedUse.HARD_CONSTRAINT.value


def test_empirical_observation_is_never_a_universal_constraint():
    rng = np.random.default_rng(0)
    m = rng.normal(size=500)
    X = np.column_stack([m, m + np.abs(rng.normal(size=500))])
    f = observe_order_relation(X, ["mean_x", "max_x"], "mean_x", "max_x")
    assert f.scope["violation_rate"] == 0.0 and f.epistemic == Epistemic.EMPIRICALLY_OBSERVED.value          # 100 % no treino
    assert f.permitted_use == PermittedUse.NONE.value and not is_eliminable(f)
    g = observe_order_relation(np.column_stack([m, m + rng.normal(size=500)]), ["a", "b"], "a", "b")
    assert 0.2 < g.scope["violation_rate"] < 0.8 and not is_eliminable(g)                                    # regularidade que falha (cf. 64 %): nunca axioma


def test_permitted_use_policy_never_elevates_weak_statuses():
    base = dict(id="F-x", kind=Kind.ORDER_RELATION.value, entities=(), attributes=("a", "b"), source_axioms=("s",), provenance={}, derivation=Derivation.DOCUMENT.value,
                reasoner_result=None, matching_confidence=None, scope={"level": "row_values"}, application_restrictions=(), permitted_use="NONE", use_state=UseState.AVAILABLE.value)
    for ep, rc in itertools.product((Epistemic.HYPOTHESIS, Epistemic.UNKNOWN, Epistemic.EMPIRICALLY_OBSERVED), ("MATHEMATICAL_IDENTITY", "FORMAL_DOMAIN_CONSTRAINT")):
        f = SemanticFact(epistemic=ep.value, relation_class=rc, validity_conditions={c: True for c in ORDER_VALIDITY_CONDITIONS},
                         review={"domain_expert_status": "APPROVED", "methodology_status": "APPROVED"}, **base)
        assert decide_permitted_use(f) == PermittedUse.NONE.value and not is_eliminable(f)


# ---------------------------------------------------------------------------------------------------- relatedness taxonómica
def test_relatedness_properties_bounds_symmetry_reflexivity(taxo):
    _, F, reg = taxo
    for m in ("lin", "wup"):
        M, S = relatedness_matrix(reg, F, m)
        assert np.allclose(M, M.T) and np.allclose(np.diag(M), 1.0) and M.min() >= 0.0 and M.max() <= 1.0
        assert all(S[i][j] == S[j][i] for i in range(len(F)) for j in range(len(F)))


def test_relatedness_orders_same_concept_sibling_and_unrelated(taxo):
    _, F, reg = taxo
    M, S = relatedness_matrix(reg, F, "lin")
    i = {f: k for k, f in enumerate(F)}
    same = M[i["temperature_mean"], i["temperature_peak"]]
    sib = M[i["temperature_mean"], i["humidity_level"]]
    far = M[i["temperature_mean"], i["pressure_value"]]
    assert same > sib > far == 0.0
    assert S[i["temperature_mean"]][i["pressure_value"]] == DISTINCT_DISJOINT          # Mid1 ⊥ Mid2 declarado: conceitos distintos (não incompatibilidade de combinação)
    assert S[i["temperature_mean"]][i["humidity_level"]] == RELATED


def test_relatedness_subproperty_and_equivalence(taxo):
    _, F, reg = taxo
    M, _ = relatedness_matrix(reg, F, "lin")
    i = {f: k for k, f in enumerate(F)}
    assert M[i["pressure_value"], i["pressure_reading"]] == pytest.approx(1.0)           # propriedades equivalentes
    assert M[i["humidity_level"], i["light_level"]] > M[i["humidity_level"], i["temperature_mean"]]        # sub-propriedade + mesmo conceito > só ancestral comum


def test_relatedness_ignores_generic_ancestors(taxo, tmp_path):
    path, F, reg = taxo
    M0, _ = relatedness_matrix(reg, F, "lin")
    # envolve toda a ontologia num novo ancestral genérico: a matriz não pode mudar
    txt = path.read_text().replace('<owl:Class rdf:about="#Root">', '<owl:Class rdf:about="#Root"><rdfs:subClassOf rdf:resource="#UniversalGeneric"/>')
    txt = txt.replace("</rdf:RDF>", '<owl:Class rdf:about="#UniversalGeneric"/>\n</rdf:RDF>')
    p2 = tmp_path / "generic.owl"
    p2.write_text(txt)
    M1, _ = relatedness_matrix(compile_registry(p2, F), F, "lin")
    assert np.allclose(M0, M1)


def test_relatedness_invariant_to_renaming_and_feature_order(taxo, tmp_path):
    path, F, reg = taxo
    M0, _ = relatedness_matrix(reg, F, "lin")
    ren = {f: f.replace("temperature", "heat").replace("pressure", "force") for f in F}
    txt = path.read_text()
    for a, b in ren.items():
        txt = txt.replace(a, b)
    for a, b in (("Mid1", "BranchOne"), ("L1", "LeafOne"), ("Mid2", "BranchTwo")):
        txt = txt.replace(a, b)
    p2 = tmp_path / "ren.owl"
    p2.write_text(txt)
    F2 = [ren[f] for f in F]
    M1, _ = relatedness_matrix(compile_registry(p2, F2), F2, "lin")
    assert np.allclose(M0, M1)
    perm = [3, 0, 5, 1, 4, 2]
    M2, _ = relatedness_matrix(reg, [F[k] for k in perm], "lin")
    assert np.allclose(M2, M0[np.ix_(perm, perm)])


def test_relatedness_has_no_free_constants_and_beats_legacy(taxo):
    path, F, reg = taxo
    M, _ = relatedness_matrix(reg, F, "lin")
    L = legacy_matrix(str(path), F)
    iu = np.triu_indices(len(F), 1)
    assert len(set(np.round(L[iu], 4))) <= 3 and 0.7124 in set(np.round(L[iu], 4))          # histórica: hub 0,7124 para quase todos os pares
    assert len(set(np.round(M[iu], 4))) >= 4                                                  # taxonómica: valores distintos conforme a relação
    # mesma estrutura ⇒ mesmos números (sem parâmetros ajustáveis)
    M2, _ = relatedness_matrix(compile_registry(path, F), F, "lin")
    assert np.allclose(M, M2)


@needs_java
def test_relatedness_uses_inferred_relations(inferred_reg):
    ix = build_index(inferred_reg, F4)
    r = pair_relatedness(ix, "temperature_mean", "temperature_peak")
    assert r.state == RELATED and r.score > 0 and "A" in r.common_concepts          # f1∈A, f2∈C só relacionados pela subsunção inferida C ⊑ A


def test_documented_family_relation_increases_relatedness(tmp_path):
    body = klass("Root") + dprop("alpha_signal") + dprop("beta_signal") + dprop("gamma_signal")
    F = ["alpha_signal", "beta_signal", "gamma_signal"]
    reg = compile_registry(write(tmp_path, "df", body), F)
    M0, _ = relatedness_matrix(reg, F)
    assert np.allclose(M0, np.eye(3))
    add_documented(reg, {"kind": Kind.SAME_MEASUREMENT_FAMILY.value, "attributes": ["alpha_signal", "beta_signal"], "relation_class": "FORMAL_DOMAIN_CONSTRAINT",
                         "source": {"citation": "documentação do atributo"}, "level": "tabular_attribute"}, {"ontology_sha256": reg.ontology_hash})
    M1, _ = relatedness_matrix(reg, F)
    assert M1[0, 1] > 0 and M1[0, 2] == 0 == M1[1, 2]
