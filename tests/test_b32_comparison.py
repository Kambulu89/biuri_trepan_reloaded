"""B3.2: instrumentação de recursos sem efeito nas decisões, equivalência do braço eficiente ao arnês de paridade, integridade do pré-registo e contrato de recursos."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pytest.importorskip("owlready2")
from core.trepan_original import TrepanOriginalClassifier
from validation.benchmark import b31_identifiability as B31
from validation.benchmark.b31_facts import compile_registry
from validation.benchmark.b31_relatedness import semantic_units
from validation.benchmark.b31_steering import SemanticSteeredTrepan
from validation.benchmark.b32_comparison import CAL32, EVAL32, FROZEN32, MEM32, STAB32, ProblemSpec, code32, fit_arm, make_instance_mlp

FZ = json.loads(FROZEN32.read_text())
SPEC = ProblemSpec(**FZ["problems"]["P1prime_within_group"])
SEED = 7777


def _sha(m):
    return hashlib.sha256(json.dumps(m.export_rules(), default=str).encode()).hexdigest()[:16]


@pytest.fixture(scope="module")
def inst():
    return make_instance_mlp(SPEC, SEED)


def _kw():
    return dict(min_sample=SPEC.min_sample, max_queries=SPEC.max_queries, max_nodes=SPEC.max_nodes, max_depth=SPEC.max_depth, max_n=SPEC.max_n, beam_width=SPEC.beam_width,
                min_samples_leaf=SPEC.min_samples_leaf, max_features_per_node=SPEC.max_features_per_node, random_state=SEED)


def test_resource_counting_does_not_change_the_original_tree(inst):
    plain = TrepanOriginalClassifier(**_kw()).fit(inst["X"], oracle=inst["mlp"], feature_names=inst["names"])
    with tempfile.TemporaryDirectory() as td:
        r = fit_arm(SPEC, inst, "A_ORIG", SEED, Path(td))
    assert r["rules_sha"] == _sha(plain)
    assert r["eval_calls"] > 0 and r["literals_listed"] > 0 and r["queries"] > 0 and r["oracle_samples"] >= r["queries"] and r["cpu_fit_s"] > 0


def test_efficient_steering_is_the_same_tree_as_the_parity_harness(inst):
    with tempfile.TemporaryDirectory() as td:
        owl = B31.build_ontology("TRUE", inst, SEED, Path(td))
        reg = compile_registry(owl, inst["names"])
        units = semantic_units(reg, inst["names"])
        ref = SemanticSteeredTrepan(units=units, unit_budget=3, registry=reg, pool_parity=True, **_kw()).fit(inst["X"], oracle=inst["mlp"], feature_names=inst["names"])
        r = fit_arm(SPEC, inst, "C_OWL", SEED, Path(td))
    assert r["rules_sha"] == _sha(ref) and r["unit_searches"] > 0


def test_equal_candidate_budget_controls_make_the_same_number_of_unit_searches(inst):
    with tempfile.TemporaryDirectory() as td:
        rows = {a: fit_arm(SPEC, inst, a, SEED, Path(td)) for a in ("C_OWL", "D_PERM", "E_RAND", "F1_CONV", "F2_SCREEN")}
    assert len({r["unit_searches"] for r in rows.values()}) == 1 and len({r["n_units"] for r in rows.values()}) <= 2
    base = rows["C_OWL"]["eval_calls"]
    assert all(0.75 <= r["eval_calls"] / base <= 1.25 for r in rows.values())                      # candidatos avaliados comparáveis (contrato da Comparação 1)


def test_frozen_preregistration_integrity():
    assert FZ["frozen"] and set(CAL32).isdisjoint(EVAL32) and set(CAL32).isdisjoint(STAB32) and set(EVAL32).isdisjoint(STAB32) and set(MEM32) <= set(EVAL32)
    assert "Original" in FZ["calibration_used_only"] and "nunca desempenho" in FZ["calibration_used_only"]
    for k, v in FZ["calibration"].items():
        assert v["mf"]["within_target_band"] and v["beam"]["bw_h1"] in v["beam"]["table"] or str(v["beam"]["bw_h1"]) in v["beam"]["table"]
    assert set(FZ["criteria"]["ontological_advantage"]["verdicts"]) == {"ONTOLOGICAL_ADVANTAGE_OVER_GENERIC", "RELATION_SPECIFIC_ONLY", "NO_ONTOLOGICAL_ADVANTAGE"}
    # o código avaliado é o congelado (hash com fins de linha normalizados)
    assert code32() == FZ["code_sha256"]
