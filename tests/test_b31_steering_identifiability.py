"""B3.1: arnês de orientação semântica (EQ sem unidades, rastreabilidade) e integridade/estrutura dos problemas de identificabilidade congelados (validation/ apenas).
Não avalia desempenho dos mecanismos (isso é feito pelo avaliador congelado, Doc. 36)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
pytest.importorskip("owlready2")

from eq_support import BUDGET, SpyOracle, fingerprint, make_problem
from core.trepan_original import TrepanOriginalClassifier
from validation.benchmark.b31_facts import Kind, compile_registry
from validation.benchmark.b31_identifiability import (CAL_SEEDS, EVAL_SEEDS, FROZEN, ProblemSpec, VARIANTS_ONTOLOGY, build_ontology, fit_variant, make_instance)
from validation.benchmark.b31_relatedness import SemanticUnit, semantic_units
from validation.benchmark.b31_steering import SemanticSteeredTrepan


def test_steering_without_units_is_exactly_the_original():
    X, names, orc = make_problem(3, p=8, k=3)
    res = []
    for cls in (TrepanOriginalClassifier, SemanticSteeredTrepan):
        spy = SpyOracle(orc)
        m = cls(**BUDGET, random_state=3)
        m.fit(X[:200], oracle=spy, feature_names=names)
        res.append(fingerprint(m, spy, X[200:]))
    assert res[0] == res[1]                                           # regras, previsões, consultas, fluxo ao oráculo, auditorias: idênticos


def test_steering_never_modifies_scores_or_uses_hard_constraints():
    X, names, orc = make_problem(4, p=8, k=3)
    u = SemanticUnit("c", tuple(names[:4]), 1.0, ("ASSERTED",), ("fact",), 1.0)
    m = SemanticSteeredTrepan(units=[u], unit_budget=2, **BUDGET, random_state=4)
    m.fit(X[:200], oracle=orc, feature_names=names)
    rep = m.trace_report()
    assert rep["n_calls"] > 0
    for e in rep["entries"]:
        assert e["scoring_modified"] is False and e["score_changed"] is False and e["hard_constraints_applied"] == 0
        assert "preferência" in e["logical_admissibility"]
        for k in ("original_choice", "final_choice", "choice_changed", "units_searched", "winning_unit", "in_final_tree", "feature_names"):
            assert k in e
        for us in e["units_searched"]:
            assert {"unit_concept", "unit_features", "fact_ids", "derivations", "mean_matching_confidence", "candidate_exists_without_owl", "unit_features_all_in_original_pool"} <= set(us)


def _frozen():
    return json.loads(FROZEN.read_text())


def test_frozen_problems_are_preregistered_and_calibrated_only_with_original():
    fz = _frozen()
    assert fz["frozen"] and fz["calibration_used_only_original"]
    assert set(CAL_SEEDS).isdisjoint(EVAL_SEEDS)
    for pname, cal in fz["calibration"].items():
        chosen = cal["chosen_max_features_per_node"]
        assert 0.2 <= cal["rates"][str(chosen)] <= 0.8                       # dificuldade intermédia: o Original NÃO recupera a regra em todas as execuções
        assert fz["problems"][pname]["max_features_per_node"] == chosen
    assert {"P1_within_group", "P2_cross_group"} <= set(fz["criteria"])
    assert set(fz["variants"]["ontology"]) == set(VARIANTS_ONTOLOGY)
    assert set(fz["code_sha256"]) >= {"validation/benchmark/b31_facts.py", "validation/benchmark/b31_steering.py"}


@pytest.fixture(scope="module")
def specs():
    fz = _frozen()
    return {k: ProblemSpec(**v) for k, v in fz["problems"].items()}


def test_variant_ontologies_have_the_designed_structure(specs, tmp_path):
    sp = specs["P1_within_group"]
    for seed in (7001, 7002, 7003):
        inst = make_instance(sp, seed)
        rn = set(inst["rule_names"])
        F = inst["names"]
        regs = {v: compile_registry(build_ontology(v, inst, seed, tmp_path), F) for v in ("TRUE", "PERM", "RAND", "IRRELEVANT", "INCOMPLETE", "AMBIGUOUS", "INCONSISTENT", "BADMATCH", "EMPTY")}
        units = {v: semantic_units(r, F) for v, r in regs.items()}
        has_rule = lambda us: any(rn <= set(u.features) for u in us)
        assert has_rule(units["TRUE"])                                    # a relação útil está disponível na ontologia verdadeira
        assert not has_rule(units["PERM"]) and not has_rule(units["RAND"])         # permutada/aleatória destroem-na
        assert not has_rule(units["IRRELEVANT"]) and not has_rule(units["INCOMPLETE"])
        assert units["EMPTY"] == [] and units["INCONSISTENT"] == [] and units["BADMATCH"] == []     # abstenção: sem unidades utilizáveis
        assert regs["INCONSISTENT"].ontology_status == "INCONSISTENT" and regs["BADMATCH"].ontology_status.startswith("REJECTED_BY_GATE")
        rejected = {f.attributes[0] for f in regs["AMBIGUOUS"].by_kind(Kind.MATCH_FAILURE)}
        assert rn <= rejected and all(f.rejection_reason == "REJECTED_AMBIGUOUS" for f in regs["AMBIGUOUS"].by_kind(Kind.MATCH_FAILURE) if f.attributes[0] in rn)
        sizes = lambda us: sorted(len(u.features) for u in us)
        assert sizes(units["PERM"]) == sizes(units["TRUE"]) == sizes(units["RAND"])           # mesma forma (tamanhos): só a relação difere


def test_abstention_variants_reproduce_the_original_tree(specs):
    sp = specs["P1_within_group"]
    seed = 7101
    inst = make_instance(sp, seed)
    with tempfile.TemporaryDirectory() as td:
        base = fit_variant(sp, inst, "ORIG", seed, Path(td))
        for v in ("EMPTY", "INCONSISTENT", "BADMATCH"):
            r = fit_variant(sp, inst, v, seed, Path(td))
            assert r["rules_sha"] == base["rules_sha"] and r["queries"] == base["queries"] and r["n_choice_changed"] == 0, v


def test_traceability_chain_for_a_semantically_influenced_candidate(specs):
    sp = specs["P1_within_group"]
    with tempfile.TemporaryDirectory() as td:
        found = 0
        for seed in range(7200, 7216):
            inst = make_instance(sp, seed)
            owl = build_ontology("TRUE", inst, seed, Path(td))
            reg = compile_registry(owl, inst["names"])
            units = semantic_units(reg, inst["names"])
            m = SemanticSteeredTrepan(units=units, unit_budget=3, registry=reg, min_sample=sp.min_sample, max_queries=sp.max_queries, max_nodes=sp.max_nodes, max_depth=sp.max_depth, max_n=sp.max_n,
                                      beam_width=sp.beam_width, min_samples_leaf=sp.min_samples_leaf, max_features_per_node=sp.max_features_per_node, random_state=seed)
            m.fit(inst["X"], oracle=inst["oracle"], feature_names=inst["names"])
            rep = m.trace_report()
            for e in rep["entries"]:
                if e["semantically_influenced"]:
                    found += 1
                    assert e["semantic_facts"], "decisão influenciada sem cadeia de factos"
                    f = e["semantic_facts"][0]
                    assert f["axioms"] and f["derivation"] in ("ASSERTED", "DERIVED_CLOSURE", "INFERRED_BY_REASONER") and f["epistemic"] and f["attribute_to_entity"]
                    assert e["winning_unit"] is not None and e["score_changed"] is False
                    assert e["units_searched"] and "candidate_exists_without_owl" in e["units_searched"][0]
        assert found > 0, "nenhuma decisão influenciada nas 16 sementes de verificação da cadeia (o teste exige exemplos para validar a rastreabilidade)"
