"""H2 (Docs. 50–52): testes pequenos de CORREÇÃO SEMÂNTICA do contrato axioma→mecanismo, das fichas e do protótipo T1/T2. Sem benchmark, sem rótulos."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pytest.importorskip("owlready2")
from validation.benchmark import h2_axiom_mechanisms as M
from validation.benchmark import h2_contract as C
from validation.benchmark.b30_axiom_probe import dprop, klass, write
from validation.benchmark.b31_facts import Kind, compile_registry

LEVELS = ["A61", "A62", "A63", "A64"]
NAMES = [f"sav__{l}" for l in LEVELS]


def _union(name, members, parent):
    mem = "".join(f'<owl:Class rdf:about="#{m}"/>' for m in members)
    return f'<owl:Class rdf:about="#{name}"><rdfs:subClassOf rdf:resource="#{parent}"/><owl:equivalentClass><owl:Class><owl:unionOf rdf:parseType="Collection">{mem}</owl:unionOf></owl:Class></owl:equivalentClass></owl:Class>\n'


def _owl(tmp, name, defined=True, order=LEVELS):
    body = klass("Root") + klass("Sav", ["Root"]) + "".join(klass(f"L{l}", ["Sav"]) for l in LEVELS)
    for k in range(1, len(order)):
        body += _union(f"AtLeast{order[k]}", [f"L{x}" for x in order[k:]], "Sav") if defined else klass(f"AtLeast{order[k]}", ["Sav"])
    body += "".join(dprop(f"has_sav_{l}", f"L{l}", label=f"sav__{l}") for l in LEVELS)
    return write(tmp, name, body)


def _X(n=60, seed=0):
    rng = np.random.default_rng(seed)
    lv = rng.integers(0, 4, size=n)
    return np.eye(4)[lv]


def test_contract_registry_is_complete_and_open_world_rules_declared():
    assert C.validate_contracts() == []
    assert set(C.CONTRACTS) >= {"T1_DEFINED_CLASS_UNION", "T2_ORDINAL_ORDER", "T3_DISJOINTNESS", "T4_VALUE_BOUND", "T5_ORDER_BETWEEN_ATTRIBUTES", "T8_CARDINALITY"}
    assert "NÃO converter" in C.CONTRACTS["T3_DISJOINTNESS"].open_world_rule and "NÃO converter" in C.CONTRACTS["T8_CARDINALITY"].open_world_rule
    assert C.CONTRACTS["T1_DEFINED_CLASS_UNION"].status == C.CONTRACTS["T2_ORDINAL_ORDER"].status == "IMPLEMENTED_PROTOTYPE_VALIDATION"


def _card(**kw):
    base = dict(id="AX1", statement_nl="x", owl_axiom="y", entities=["a"], axiom_type="T2_ORDINAL_ORDER")
    base.update(kw)
    return C.AxiomCard(**base)


def test_axiom_card_rules_block_ai_validation_data_derivation_and_unsigned_approvals():
    assert C.validate_card(_card()) == []                                                         # rascunho HYPOTHESIS com revisões pendentes
    assert any("correlações" in p for p in C.validate_card(_card(derived_from_data=True)))
    assert any("sem fonte" in p for p in C.validate_card(_card(epistemic_status="DOMAIN_DOCUMENTED")))
    assert any("verificação humana" in p for p in C.validate_card(_card(epistemic_status="DOMAIN_DOCUMENTED", source={"citation": "c", "locator": "l", "verified": False})))
    c = _card()
    c.reviews["domain"] = {"status": "APPROVED"}                                                  # aprovação sem nome/data/assinatura
    assert any("sem revisor" in p for p in C.validate_card(c))
    hard = _card(permitted_use="HARD_CONSTRAINT")
    ps = C.validate_card(hard)
    assert any("restrição dura" in p for p in ps)
    assert any("três revisões" in p for p in ps) and any("UNIVERSAL" in p for p in ps)


def test_german_axiom_cards_are_ai_drafts_unverified_and_pending(tmp_path):
    from validation.benchmark.h2_german_proposal import cards
    cs = cards()
    assert len(cs) == 14 and all(C.validate_card(c) == [] for c in cs)
    assert all(c.authored_by == "AI_DRAFT" and c.epistemic_status == "HYPOTHESIS" and c.source["verified"] is False for c in cs)
    assert all(r["status"].endswith("REVIEW_PENDING") for c in cs for r in c.reviews.values())
    for f in (ROOT / "validation/benchmark/ontologies/proposals/german_credit/v0_draft/axioms").glob("*.json"):
        assert json.loads(f.read_text())["reviews"]["domain"]["status"] == "DOMAIN_REVIEW_PENDING"


def test_defined_classes_give_exact_nested_extensions_and_depend_on_the_reasoner(tmp_path):
    X = _X()
    reg = compile_registry(str(_owl(tmp_path, "o2")), NAMES, run_reasoner=True)
    lits = {l.concept: l for l in M.derived_union_literals(reg, X, NAMES)}
    assert lits["AtLeastA62"].attributes == ("sav__A62", "sav__A63", "sav__A64") and lits["AtLeastA63"].attributes == ("sav__A63", "sav__A64")
    assert all(l.needs_reasoner for l in lits.values())                                           # a extensão só existe por inferência
    Xa, na = M.apply_literals([lits["AtLeastA62"], lits["AtLeastA63"]], X, NAMES)
    c62, c63 = Xa[:, na.index("onto_any[AtLeastA62]")], Xa[:, na.index("onto_any[AtLeastA63]")]
    assert np.array_equal(c62, X[:, 1:].max(axis=1)) and np.array_equal(c63, X[:, 2:].max(axis=1)) and np.all(c63 <= c62)
    assert Xa.shape[0] == X.shape[0]                                                              # nenhuma linha removida (mundo aberto)
    nor = compile_registry(str(_owl(tmp_path, "o2b")), NAMES, run_reasoner=False)
    assert not any(l.concept.startswith("AtLeast") for l in M.derived_union_literals(nor, X, NAMES))   # sem reasoner, sem literais


def test_neutralized_and_permuted_controls_and_counterfactual(tmp_path):
    X = _X()
    full = M.derived_union_literals(compile_registry(str(_owl(tmp_path, "full")), NAMES), X, NAMES)
    prim = M.derived_union_literals(compile_registry(str(_owl(tmp_path, "prim", defined=False)), NAMES), X, NAMES)
    assert full and not [l for l in prim if l.concept.startswith("AtLeast")]                      # O3: classes primitivas ⇒ nenhuma extensão
    perm = M.derived_union_literals(compile_registry(str(_owl(tmp_path, "perm", order=["A63", "A61", "A64", "A62"])), NAMES), X, NAMES)
    assert sorted(len(l.attributes) for l in perm if l.concept.startswith("AtLeast")) == sorted(len(l.attributes) for l in full if l.concept.startswith("AtLeast"))   # mesmo perfil de tamanhos
    assert {l.attributes for l in perm} != {l.attributes for l in full}                          # conteúdo diferente
    d = M.counterfactual_diff(full, M.control_neutralize(full, ["AtLeastA63"]))
    assert d["removed"] == ["AtLeastA63"] and not d["added"]
    rnd = M.control_random_unions(full, NAMES, seed=1)
    assert [len(l.attributes) for l in rnd] == [len(l.attributes) for l in full] and all(l.family == "sav" for l in rnd)


def test_disjointness_is_not_used_as_closed_row_validation(tmp_path):
    body = klass("Root") + klass("Sav", ["Root"]) + "".join(klass(f"L{l}", ["Sav"]) for l in LEVELS)
    body += '<owl:AllDisjointClasses><owl:members rdf:parseType="Collection">' + "".join(f'<rdf:Description rdf:about="#L{l}"/>' for l in LEVELS) + '</owl:members></owl:AllDisjointClasses>\n'
    body += "".join(dprop(f"has_sav_{l}", f"L{l}", label=f"sav__{l}") for l in LEVELS)
    reg = compile_registry(str(write(tmp_path, "dj", body)), NAMES, run_reasoner=False)
    dj = reg.by_kind(Kind.CONCEPT_DISJOINTNESS)
    assert dj and all(f.permitted_use == "NONE" for f in dj)                                      # nunca consumido: não é validação de linhas
    X = _X()
    Xa, _ = M.apply_literals(M.derived_union_literals(reg, X, NAMES), X, NAMES)
    assert Xa.shape[0] == X.shape[0]


def test_german_proposal_variants_semantic_correctness():
    from validation.benchmark import datasets as D, datasets_categorical as DC
    from validation.benchmark.h2_german_proposal import build
    DC.register()
    spec = D.REGISTRY["german_credit"]
    ds, _, _ = spec.load()
    names = [str(n) for n in ds.feature_names]
    X = np.asarray(ds.X, dtype=float)
    groups = spec.groups(names)
    res = {}
    with tempfile.TemporaryDirectory() as td:
        for v in ("O2", "O3"):
            p = Path(td) / f"{v}.owl"
            p.write_text(build(v, names, groups), encoding="utf-8")
            reg = compile_registry(str(p), names, run_reasoner=True)
            assert reg.ontology_status == "VALID" and all(reg.feature_entity.get(n) for n in names)     # cobertura 100 %, ontologia válida
            res[v] = M.derived_union_literals(reg, X, names)
    assert not res["O3"] and len(res["O2"]) == 10 and all(l.needs_reasoner for l in res["O2"])
    sav = next(l for l in res["O2"] if l.concept == "AtLeast_savings_account_A62")
    assert [a.split("__")[1] for a in sav.attributes] == ["A62", "A63", "A64"]                        # A65 (desconhecido) fora da ordem
    assert all("A14" not in a and "A65" not in a for l in res["O2"] for a in l.attributes)
