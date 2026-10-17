"""B3.0 (Docs. 31/32): testes sintéticos de AUDITORIA do percurso OWL (axiomas, matching, reasoner, relatedness, identificabilidade).

Convenção: os testes ``test_current_*`` CARACTERIZAM o comportamento atual do núcleo (limitações medidas e documentadas no Doc. 31/32); se o núcleo for corrigido
sob aprovação futura, devem ser revistos de forma explícita (não são requisitos de qualidade). Os testes ``test_alternative_*`` mostram que as relações semânticas
distintas PODEM produzir valores/decisões distintas com medidas alternativas (só em ``validation/``). Nenhum dataset conhecido; sem núcleo alterado."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("owlready2")
from validation.benchmark.b30_axiom_probe import axiom_cases, dprop, klass, probe, write
from validation.benchmark.b30_relatedness_alternatives import Taxonomy, relatedness_matrix, taxonomy_from_graph, wu_palmer

HUB = 0.88 * 0.92 * 0.88          # dois saltos «subsumption» pelo nó genérico DatatypeProperty (caminho máx-produto)


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    t = tmp_path_factory.mktemp("b30")
    return {k: probe(write(t, k, body), ["f1", "f2", "f3"]) for k, body in axiom_cases().items()}


# ---------------------------------------------------------------- expressividade: o que é reconhecido / ignorado -----------------------------------------
def test_current_axioms_without_effect_on_features_or_graph(cases):
    base = cases["subClassOf"]
    for ignored in ("disjointWith", "someValuesFrom", "allValuesFrom", "cardinality", "objectPropertyInverse", "propertyChain"):
        r = cases[ignored]
        assert r["edge_type_set"] == base["edge_type_set"], ignored
        assert r["groups"] == base["groups"], ignored
        assert r["derived"] == base["derived"], ignored
        assert r["relatedness"] == base["relatedness"], ignored
        assert r["depth"] == base["depth"], ignored


def test_current_equivalent_class_creates_edge_but_does_not_merge_groups(cases):
    r = cases["equivalentClass"]
    assert "equivalent" in r["edge_type_set"]
    assert r["groups"]["f1"] != r["groups"]["f2"]            # f1 e f2 em classes EQUIVALENTES continuam em grupos distintos
    assert r["derived"] == []                                 # e não há agregado ontológico para o conceito equivalente


def test_current_domain_defines_the_only_partition_signal(cases):
    assert cases["subClassOf"]["groups"] == {"f1": ["A"], "f2": ["A"], "f3": ["B"]}
    assert cases["no_domain"]["groups"] == {"f1": ["f1"], "f2": ["f2"], "f3": ["f3"]}        # sem domínio: grupos singulares (sem estrutura)
    assert cases["no_domain"]["derived"] == []
    assert [d[1] for d in cases["subClassOf"]["derived"]] == ["onto_A_aggregate"]


def test_current_datatype_facets_become_constraint_features(cases):
    names = [d[1] for d in cases["datatypeFacets"]["derived"]]
    assert "onto_f1_within_min" in names and "onto_f1_within_max" in names                   # xsd:min/maxInclusive são lidos (conhecimento verificável)


def test_current_data_subproperty_adds_group_candidate(cases):
    assert cases["dataSubPropertyOf"]["groups"]["f1"] == ["A", "fAll"]


# ---------------------------------------------------------------- relatedness ------------------------------------------------------------------------
def test_current_relatedness_is_partition_blind(cases):
    """TODAS as ontologias, com ou sem domínio, devolvem 0.7124 para qualquer par de propriedades de dados (nó genérico DatatypeProperty)."""
    for name, r in cases.items():
        vals = {v for v in r["relatedness"].values() if v is not None}
        assert vals == {round(HUB, 4)}, (name, vals)


def test_current_relatedness_same_for_different_partitions(tmp_path):
    names = ["pa", "pb", "pc", "pd"]
    o1 = klass("R") + klass("X", ["R"]) + klass("Y", ["R"]) + dprop("pa", "X") + dprop("pb", "X") + dprop("pc", "Y") + dprop("pd", "Y")
    o2 = klass("R") + klass("X", ["R"]) + klass("Y", ["R"]) + dprop("pa", "X") + dprop("pc", "X") + dprop("pb", "Y") + dprop("pd", "Y")
    r1 = probe(write(tmp_path, "o1", o1), names)["relatedness"]
    r2 = probe(write(tmp_path, "o2", o2), names)["relatedness"]
    assert r1 == r2


def _graph_for(path, names):
    from core.owl_runtime import load_ontology_isolated
    from core.ontology_semantic_graph import OntologySemanticGraph
    g = OntologySemanticGraph.from_ontology(load_ontology_isolated(str(path)), accepted_matches=[{"feature": n, "entity_name": n} for n in names])
    return g, taxonomy_from_graph(g, {n: n for n in names})


def test_alternative_relatedness_distinguishes_partitions(tmp_path):
    names = ["pa", "pb", "pc", "pd"]
    o1 = klass("R") + klass("X", ["R"]) + klass("Y", ["R"]) + dprop("pa", "X") + dprop("pb", "X") + dprop("pc", "Y") + dprop("pd", "Y")
    o2 = klass("R") + klass("X", ["R"]) + klass("Y", ["R"]) + dprop("pa", "X") + dprop("pc", "X") + dprop("pb", "Y") + dprop("pd", "Y")
    _, t1 = _graph_for(write(tmp_path, "o1", o1), names)
    _, t2 = _graph_for(write(tmp_path, "o2", o2), names)
    for measure in ("wu_palmer", "lin"):
        m1, m2 = relatedness_matrix(t1, names, measure), relatedness_matrix(t2, names, measure)
        assert m1[0, 1] == pytest.approx(1.0) and m1[0, 2] < m1[0, 1]          # mesmo conceito > conceitos irmãos
        assert m2[0, 2] == pytest.approx(1.0) and m2[0, 1] < m2[0, 2]
        assert not np.allclose(m1, m2)                                             # partições distintas ⇒ matrizes distintas


def test_alternative_relatedness_responds_to_hierarchy_depth(tmp_path):
    body = (klass("Root") + klass("Mid1", ["Root"]) + klass("Mid2", ["Root"]) + klass("L1", ["Mid1"]) + klass("L2", ["Mid1"]) + klass("L3", ["Mid2"])
            + dprop("fa", "L1") + dprop("fb", "L1") + dprop("fc", "L2") + dprop("fd", "L3"))
    names = ["fa", "fb", "fc", "fd"]
    g, t = _graph_for(write(tmp_path, "h", body), names)
    M = relatedness_matrix(t, names, "wu_palmer")
    assert M[0, 1] == pytest.approx(1.0) and M[0, 2] == pytest.approx(2 * 2 / 6) and M[0, 3] == pytest.approx(2 * 1 / 6)
    assert M[0, 1] > M[0, 2] > M[0, 3] > 0
    cur = {round(g.relatedness(g.entity_for_feature(a), g.entity_for_feature(b)), 4) for i, a in enumerate(names) for b in names[i + 1:]}
    assert cur == {round(HUB, 4)}                                                   # a medida atual não vê a hierarquia


def test_alternative_relatedness_is_zero_without_shared_nongeneric_concept():
    t = Taxonomy()
    t.parents["A"].add("RA"); t.parents["B"].add("RB")
    t.feature_concepts = {"x": {"A"}, "y": {"B"}}
    assert wu_palmer(t, "A", "B") == 0.0
    assert relatedness_matrix(t, ["x", "y"], "lin")[0, 1] == 0.0


# ---------------------------------------------------------------- matching --------------------------------------------------------------------------
def test_matching_is_lexical_not_semantic(tmp_path):
    from core.owl_runtime import load_ontology_isolated
    from core.ontology_quality import OntologyQualityGate
    body = (klass("Measurement") + klass("Thermal", ["Measurement"]) + klass("Optical", ["Measurement"]) + dprop("hasTemperatureMean", "Thermal", label="temperature mean")
            + dprop("hasTemperatureMax", "Thermal", label="temperature max") + dprop("hasLuminosity", "Optical", label="luminosity"))
    onto = load_ontology_isolated(str(write(tmp_path, "m", body)))
    gate = OntologyQualityGate()
    dec = {f: gate.match_features([f], onto)[0] for f in ("temperature mean", "Temperature_Mean", "hasTemperature", "temp_mean", "temperature_average", "lumens", "x1")}
    assert dec["temperature mean"].accepted and dec["temperature mean"].entity_name == "hasTemperatureMean" and dec["Temperature_Mean"].accepted
    assert dec["hasTemperature"].status == "REJECTED_AMBIGUOUS"                                       # ambiguidade → rejeitado
    for syn in ("temp_mean", "temperature_average", "lumens", "x1"):                                    # sinónimos/abreviaturas: NÃO mapeados (sem semântica)
        assert not dec[syn].accepted, syn


def test_matching_rejects_placeholder_ontology(tmp_path):
    from core.owl_runtime import load_ontology_isolated
    from core.ontology_quality import OntologyQualityGate
    body = klass("Group1") + klass("Group2") + dprop("f1", "Group1") + dprop("f2", "Group1") + dprop("f3", "Group2")
    onto = load_ontology_isolated(str(write(tmp_path, "g", body)))
    rep = OntologyQualityGate().evaluate(["f1", "f2", "f3"], onto, reasoner_report=None, require_reasoner=False)
    assert not rep.accepted                                                                            # nomes placeholder ⇒ ontologia genérica rejeitada


# ---------------------------------------------------------------- reasoner --------------------------------------------------------------------------
def _java():
    from core.ontology_reasoner import find_java_executable
    try:
        return find_java_executable()
    except Exception:
        return None


@pytest.mark.skipif(_java() is None, reason="Java não disponível para o HermiT")
def test_reasoner_infers_subclass_but_groups_ignore_it(tmp_path):
    sv = '<owl:Restriction><owl:onProperty rdf:resource="#hasPart"/><owl:someValuesFrom rdf:resource="#B"/></owl:Restriction>'
    body = ('<owl:ObjectProperty rdf:about="#hasPart"/>\n' + klass("Root") + klass("B", ["Root"])
            + '<owl:Class rdf:about="#A"><rdfs:subClassOf rdf:resource="#Root"/><owl:equivalentClass>' + sv + '</owl:equivalentClass></owl:Class>\n'
            + '<owl:Class rdf:about="#C"><rdfs:subClassOf rdf:resource="#Root"/><rdfs:subClassOf>' + sv + '</rdfs:subClassOf></owl:Class>\n'
            + dprop("f1", "A") + dprop("f2", "C") + dprop("f3", "B"))
    r = probe(write(tmp_path, "inf", body), ["f1", "f2", "f3"], reasoner=True)
    assert r["reasoner"]["executed"] and r["reasoner"]["consistent"]
    assert ["C", "A"] in [list(p) for p in r["reasoner"]["inferred_subclass_relations"]]               # HermiT infere C ⊑ A (não declarado)
    assert r["groups"]["f1"] == ["A"] and r["groups"]["f2"] == ["C"]                                  # mas f1 e f2 NÃO passam ao mesmo grupo/agregado
    assert r["derived"] == []


# ---------------------------------------------------------------- identificabilidade estrutural ----------------------------------------------------------
def test_current_owl_path_semantic_components_are_group_blind(tmp_path):
    """Com o fornecedor OWL real, a relatedness constante (≥ limiar 0.35) une TODAS as features originais num único componente semântico, qualquer que seja a
    partição (verdadeira ou permutada); com grupos explícitos e relatedness discriminativa os componentes seguem a partição."""
    from core.benchmark.semantic import GroupSemanticProvider, OwlSemanticProvider
    from validation.benchmark.b30_identifiability import (CLASSES, TRUE_PARTITION, WRONG_PARTITION, candidate_components, make_data, owl_for)
    X, names = make_data(0)
    n = len(names)
    for part in (TRUE_PARTITION, WRONG_PARTITION):
        ctx = OwlSemanticProvider(str(owl_for(tmp_path, "o", part))).build(X, names)
        assert ctx.available
        assert candidate_components(list(ctx.groups), ctx.relatedness, list(range(n))) == [list(range(n))]
        gctx = GroupSemanticProvider({CLASSES[i]: g for i, g in enumerate(part)}).build(X, names)
        comps = candidate_components(list(gctx.groups), gctx.relatedness, list(range(n)))
        assert len(comps) == len(part)
        assert sorted(sorted(names.index(f) for f in g) for g in part) == sorted(comps)
    t = OwlSemanticProvider(str(owl_for(tmp_path, "t", TRUE_PARTITION))).build(X, names)
    w = OwlSemanticProvider(str(owl_for(tmp_path, "w", WRONG_PARTITION))).build(X, names)
    assert t.groups != w.groups and np.allclose(np.asarray(t.relatedness)[:n, :n], np.asarray(w.relatedness)[:n, :n])      # a diferença está nos grupos, não na relatedness


def test_current_derived_aggregates_have_no_entity_hence_zero_relatedness(tmp_path):
    from core.benchmark.semantic import OwlSemanticProvider
    from validation.benchmark.b30_identifiability import TRUE_PARTITION, make_data, owl_for
    X, names = make_data(0)
    ctx = OwlSemanticProvider(str(owl_for(tmp_path, "o", TRUE_PARTITION))).build(X, names)
    rel = np.asarray(ctx.relatedness)
    n = len(names)
    assert np.allclose(rel[n:, :n], 0.0) and np.allclose(rel[:n, n:], 0.0)                              # agregados derivados: relatedness 0 com tudo
