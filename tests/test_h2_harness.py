"""H2 (Doc. 60): testes técnicos de PONTA A PONTA do adaptador T1/T2 integrado no arnês de ``validation/`` com ontologias sintéticas de validade conhecida. Sem métricas de desempenho."""
import hashlib
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
from core.trepan_original import TrepanOriginalClassifier
from validation.benchmark import h2_harness as H
from validation.benchmark import h2_synthetic as S
from validation.benchmark.b31_facts import compile_registry
from validation.benchmark.b31_relatedness import semantic_units
from validation.benchmark.b31_steering import SemanticSteeredTrepan
from validation.benchmark.h2_axiom_mechanisms import defined_union_concepts, derived_union_literals
from validation.benchmark.h2_trace import augment

PARAMS = dict(min_sample=150, max_queries=900, max_nodes=5, max_depth=3, max_n=1, beam_width=2, min_samples_leaf=3, max_features_per_node=10, random_state=5)


@pytest.fixture(scope="module")
def world():
    X, names, lv = S.make_data(160, seed=1)
    regs, defined = {}, {}
    with tempfile.TemporaryDirectory() as td:
        for v in ("TRUE", "PERM", "NEUTRAL", "FLAT"):
            p = S.write_owl(Path(td), v)
            regs[v] = compile_registry(str(p), names, run_reasoner=True)
            defined[v] = defined_union_concepts(p)
    return X, names, regs, defined


def ctx(world, params=None):
    X, names, regs, defined = world
    orc = S.TruthOracle()
    group = [u for u in semantic_units(regs["FLAT"], names)]
    return H.Context(X=X, names=names, oracle=orc, params=params or PARAMS, registries=regs, group_units=group, defined=defined)


def test_synthetic_ontologies_are_valid_and_axiom_validity_is_known(world):
    X, names, regs, defined = world
    assert all(r.ontology_status == "VALID" for r in regs.values())
    assert all(all(r.feature_entity.get(n) for n in names) for r in regs.values())                # matching 100 % em todas as variantes
    true = {l.concept: set(l.attributes) for l in derived_union_literals(regs["TRUE"], X, names, only_concepts=defined["TRUE"])}
    truth = {"sav__A3", "sav__A4"}                                                                 # a regra verdadeira: nível ≥ A3 (A5 = desconhecido, fora)
    assert true["AtLeast_A3"] == truth and "sav__A5" not in set().union(*true.values())
    perm = [set(l.attributes) for l in derived_union_literals(regs["PERM"], X, names, only_concepts=defined["PERM"])]
    assert truth not in perm                                                                       # a ordem trocada é logicamente válida mas não exprime a verdade
    assert derived_union_literals(regs["NEUTRAL"], X, names, only_concepts=defined["NEUTRAL"]) == [] and derived_union_literals(regs["FLAT"], X, names, only_concepts=defined["FLAT"]) == []
    # um agrupamento organizacional (classe «Noise» com 3 binárias) NÃO é um axioma de definição: sem a restrição às classes definidas geraria um literal indevido
    assert any(l.concept == "Noise" for l in derived_union_literals(regs["FLAT"], X, names)) and defined["FLAT"] == set() and "Noise" not in defined["TRUE"]
    assert all(l.needs_reasoner for l in derived_union_literals(regs["TRUE"], X, names, only_concepts=defined["TRUE"]))


def test_identity_of_instances_and_oracle_across_arms_and_resource_limits(world):
    c = ctx(world)
    res = {a: H.run_arm(a, c) for a in H.ARMS}
    assert len({r["instance_hash"] for r in res.values()}) == 1 and len({r["oracle_id"] for r in res.values()}) == 1       # mesmas instâncias, mesmo oráculo
    for a, r in res.items():
        assert r["queries"] <= PARAMS["max_queries"] and r["model"].semantic_trace_["n_internal_nodes"] <= PARAMS["max_nodes"], a   # limites de consultas e de árvore
        assert r["oracle_calls"] > 0 or r["queries"] == 0
    assert all(r.shape[1] == 10 for r in c.oracle.seen)                                            # o oráculo nunca viu colunas derivadas


def test_equivalence_when_semantics_are_disabled(world):
    X, names, regs, defined = world
    p = dict(PARAMS, max_n=3)
    o1, o2 = S.TruthOracle(), S.TruthOracle()
    a = TrepanOriginalClassifier(**p).fit(X, oracle=o1, feature_names=names)
    b = H.H2Steered(derived=[], base_names=names, prune_redundant=False, units=[], pool_parity=True, **p).fit(X, oracle=o2, feature_names=names)
    sha = lambda m: hashlib.sha256(json.dumps(m.export_rules(), default=str).encode()).hexdigest()
    assert sha(a) == sha(b) and a.membership_queries_ == b.membership_queries_ and (o1.calls, o1.rows) == (o2.calls, o2.rows)
    assert all(np.array_equal(x, y) for x, y in zip(o1.seen, o2.seen))
    # unidades apenas (E) ≡ arnês B3.1 direto
    units = semantic_units(regs["FLAT"], names)
    o3, o4 = S.TruthOracle(), S.TruthOracle()
    d = SemanticSteeredTrepan(units=units, pool_parity=True, **p).fit(X, oracle=o3, feature_names=names)
    e = H.H2Steered(derived=[], base_names=names, units=units, pool_parity=True, **p).fit(X, oracle=o4, feature_names=names)
    assert sha(d) == sha(e) and (o3.calls, o3.rows) == (o4.calls, o4.rows)


def test_chain_axiom_reasoner_registry_literal_candidate_decision_with_known_truth(world):
    c = ctx(world)
    f = H.run_arm("F_EXPRESSIVE", c)
    lit = {l["concept"]: l for l in f["trace"]["literals"]}
    t = lit["AtLeast_A3"]
    assert t["needs_reasoner"] and t["fact_ids"] and t["candidate_evaluations"] > 0                # inferência → literal → candidato
    assert 0 in t["chosen_in_nodes"]                                                               # decisão: com max_n=1 só o literal derivado exprime a regra verdadeira, e é escolhido na raiz
    h = H.run_arm("H_PERMUTED", c)
    assert all(0 not in l["chosen_in_nodes"] or l["concept"] != "AtLeast_A3" for l in h["trace"]["literals"])   # com ordem errada não há literal equivalente à verdade
    g = H.run_arm("G_NEUTRALIZED", c)
    assert g["n_derived"] == 0                                                                     # neutralizada: classes primitivas ⇒ sem literais
    a = H.run_arm("A_ORIG", c)
    assert a["trace"]["n_derived"] == 0 and a["rules_sha"] != f["rules_sha"]
    i = H.run_arm("I_CONVENTIONAL", c)
    assert {frozenset(l["attributes"]) for l in i["trace"]["literals"]} >= {frozenset({"sav__A3", "sav__A4"})}   # a engenharia convencional também recupera a ordem a partir dos dados: controlo válido


def test_counterfactual_attribution_on_the_true_axiom(world):
    from validation.benchmark.h2_trace import decision_influence
    X, names, regs, defined = world
    lits = derived_union_literals(regs["TRUE"], X, names, only_concepts=defined["TRUE"])
    res = decision_influence(X, names, lits, S.TruthOracle, dict(PARAMS))
    assert res["AtLeast_A3"]["n_chosen"] >= 1 and 0 in res["AtLeast_A3"]["nodes_changed_by_ablation"]   # sem o axioma verdadeiro a decisão na raiz muda


def test_no_leakage_train_only_and_unknown_category_handling(world):
    X, names, regs, defined = world
    Xt, _, _ = S.make_data(80, seed=99)                                                            # «teste» gerado à parte
    c = ctx(world)
    H.run_arm("F_EXPRESSIVE", c)
    seen = np.vstack(c.oracle.seen)
    keys = {tuple(r) for r in Xt}
    assert not any(tuple(r) in keys for r in seen)                                                 # nenhuma linha de teste foi consultada
    lits = derived_union_literals(regs["TRUE"], X, names, only_concepts=defined["TRUE"])
    unk = np.zeros((1, X.shape[1])); unk[0, names.index("sav__A5")] = 1.0                           # categoria «desconhecido»
    Xa, _ = augment(unk, names, lits)
    assert Xa[0, len(names):].sum() == 0                                                           # nunca mapeada a um nível ordinal
