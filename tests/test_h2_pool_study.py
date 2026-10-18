"""H2 (Doc. 61): testes do estudo do pool (elegibilidade de literais, vagas reservadas, ablação por máscara, contrato de comparação, escada de estados). Ontologias sintéticas de validade conhecida;
sem desempenho; a maquinaria é agnóstica (dois conjuntos sintéticos de dimensões diferentes)."""
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
from validation.benchmark import h2_synthetic as S
from validation.benchmark.b31_facts import compile_registry
from validation.benchmark.h2_axiom_mechanisms import defined_union_concepts, derived_union_literals
from validation.benchmark.h2_pool_study import (LADDER, Ceilings, PoolTrepan, ablate_literal, cards_by_concept, comparability, consumption, fit_pool_arm, ladder_state,
                                                match_generic_by_evaluated, original_buildable)

P = dict(min_sample=150, max_queries=600, max_nodes=5, max_depth=3, max_n=1, beam_width=2, min_samples_leaf=3, max_features_per_node=3, random_state=5)


@pytest.fixture(scope="module")
def world():
    X, names, _ = S.make_data(160, seed=1)
    with tempfile.TemporaryDirectory() as td:
        p = S.write_owl(Path(td), "TRUE")
        reg = compile_registry(str(p), names, run_reasoner=True)
        lits = derived_union_literals(reg, X, names, only_concepts=defined_union_concepts(p))
    return X, names, lits


def _sha(m):
    return hashlib.sha256(json.dumps(m.export_rules(), default=str).encode()).hexdigest()


def test_pool_trepan_without_extras_is_the_original(world):
    X, names, _ = world
    o1, o2 = S.TruthOracle(), S.TruthOracle()
    a = TrepanOriginalClassifier(**dict(P, max_n=3)).fit(X, oracle=o1, feature_names=names)
    r = fit_pool_arm(X, names, [], o2, dict(P, max_n=3), extras="none", mf=P["max_features_per_node"])
    assert _sha(a) == _sha(r["model"]) and a.membership_queries_ == r["queries"] and (o1.calls, o1.rows) == (o2.calls, o2.rows)


def test_reserved_slots_make_the_derived_literal_eligible_and_without_labels(world):
    X, names, lits = world
    true = [k for k, l in enumerate(lits) if l.concept == "AtLeast_A3"][0]
    plain = fit_pool_arm(X, names, lits, S.TruthOracle(), P, extras="none", mf=3)
    resv = fit_pool_arm(X, names, lits, S.TruthOracle(), P, extras="derived", mf=3 + len(lits))
    r0 = plain["trace"]["pool"]["literals"][true]
    r1 = resv["trace"]["pool"]["literals"][true]
    assert r0["root_reason"] == "NOT_IN_FEATURE_POOL" and r1["root_reason"] == "IN_LITERAL_POOL"       # a rejeição pré-estatística é pela prioridade (variância), não pela lógica
    assert resv["feature_slots"] == plain["feature_slots"] + len(lits) and resv["n_columns"] == plain["n_columns"]
    assert resv["trace"]["pool"]["forced"] == [len(names) + k for k in range(len(lits))]


def test_ablation_by_mask_does_not_free_the_slot_and_is_reproducible(world):
    X, names, lits = world
    k = [i for i, l in enumerate(lits) if l.concept == "AtLeast_A3"][0]
    mf = 3 + len(lits)
    full = fit_pool_arm(X, names, lits, S.TruthOracle(), P, extras="derived", mf=mf)
    ab = ablate_literal(X, names, lits, S.TruthOracle, P, k, extras="derived", mf=mf, full=full)
    again = ablate_literal(X, names, lits, S.TruthOracle, P, k, extras="derived", mf=mf)
    assert ab == again
    m = fit_pool_arm(X, names, lits, S.TruthOracle(), P, extras="derived", mf=mf, masked_derived=[k])["model"]
    assert all(len(c["pool_feats"]) == mf - 1 and (len(names) + k) not in c["pool_feats"] for c in m._pool_log)     # vaga NÃO refilada
    assert ab["chosen_in_searches"] and ab["changed"]                                              # com max_n=1 a regra verdadeira só existe via literal derivado


def test_original_equivalent_buildability_is_measured_per_pool(world):
    X, names, lits = world
    small = fit_pool_arm(X, names, [], S.TruthOracle(), dict(P, max_n=3), extras="none", mf=3)["model"]
    wide = fit_pool_arm(X, names, [], S.TruthOracle(), dict(P, max_n=3), extras="none", mf=len(names))["model"]
    t = lambda m: {r["literal"]: r for r in original_buildable(m, lits, names)}["onto_any[AtLeast_A3]"]
    assert not t(small)["equivalent_test_buildable_in_some_search"]                                # pool pequeno: o Original não forma o teste equivalente
    assert t(wide)["equivalent_test_buildable_in_some_search"] or t(wide)["extension_cols_all_in_feature_pool_searches"] > 0


def test_comparability_contract_flags_every_violation(world):
    X, names, lits = world
    a = fit_pool_arm(X, names, lits, S.TruthOracle(), P, extras="derived", mf=3 + len(lits))
    b = fit_pool_arm(X, names, [], S.TruthOracle(), P, extras="none", mf=3 + len(lits))
    tol = {"feature_slots": 0.0, "mofn_generated": 0.0}
    ok = comparability(a, a, same_instances=True, same_oracle=True, same_params_except_pool=True, tol=tol)
    assert ok["status"] == "COMPARABLE"
    bad = comparability(a, b, same_instances=True, same_oracle=False, same_params_except_pool=True, tol=tol)
    assert bad["status"] == "NOT_COMPARABLE" and "same_oracle" in bad["violations"]
    assert comparability(a, a, same_instances=False, same_oracle=True, same_params_except_pool=True, tol=tol)["status"] == "NOT_COMPARABLE"


def test_states_are_separate_and_benefit_is_never_assessed():
    lit = {"fact_ids": ["f1"], "candidate_evaluations": 3}
    pool = {"reasons_by_search": {"IN_LITERAL_POOL": 1}}
    s = ladder_state(lit, pool, valid=True, selected=False, changed_by_ablation=False)
    assert s["reached"] == {"LITERAL_VALID": True, "LITERAL_ELIGIBLE": True, "LITERAL_EVALUATED": True, "LITERAL_SELECTED": False, "SPLIT_CHANGED": False}
    assert s["highest_consecutive"] == "LITERAL_EVALUATED" and s["BENEFIT_OBSERVED"] == "NOT_ASSESSED"               # avaliado ≠ selecionado ≠ split alterado
    s2 = ladder_state(lit, pool, valid=True, selected=True, changed_by_ablation=True)
    assert s2["highest_consecutive"] == "SPLIT_CHANGED" and s2["BENEFIT_OBSERVED"] == "NOT_ASSESSED"               # split alterado ≠ benefício
    s3 = ladder_state(lit, {"reasons_by_search": {"BELOW_LITERAL_POOL_CUTOFF": 2}}, valid=True, selected=False, changed_by_ablation=False)
    assert s3["reached"]["LITERAL_EVALUATED"] and not s3["reached"]["LITERAL_ELIGIBLE"] and s3["highest_consecutive"] == "LITERAL_VALID"   # avaliado como candidato simples sem elegibilidade m-of-n
    assert ladder_state(lit, pool, valid=False, selected=True, changed_by_ablation=True)["highest_consecutive"] is None       # sem validade técnica não há escada
    assert LADDER == ("LITERAL_VALID", "LITERAL_ELIGIBLE", "LITERAL_EVALUATED", "LITERAL_SELECTED", "SPLIT_CHANGED", "BENEFIT_OBSERVED")


def test_ceilings_are_authorizations_not_consumption_and_decide_comparability(world):
    X, names, lits = world
    a = fit_pool_arm(X, names, lits, S.TruthOracle(), P, extras="derived", mf=3 + len(lits))
    b = fit_pool_arm(X, names, [], S.TruthOracle(), P, extras="none", mf=3 + len(lits))
    cons = consumption(a)
    assert cons["eligible_literals"] == a["feature_slots"] and cons["nodes"] == a["n_internal_nodes"] and cons["cpu_seconds"] is not None and cons["rows_consulted"] is not None
    ok = comparability(a, b, same_instances=True, same_oracle=True, same_params_except_pool=True, tol={}, ceilings=Ceilings(nodes=P["max_nodes"], oracle_queries=P["max_queries"]))
    assert ok["status"] == "COMPARABLE" and ok["ceilings"]["a"]["authorized"] and ok["cpu_modality"]["decides_status"] is False
    tight = comparability(a, b, same_instances=True, same_oracle=True, same_params_except_pool=True, tol={}, ceilings=Ceilings(eligible_literals=1))
    assert tight["status"] == "NOT_COMPARABLE" and any(v.startswith("CEILING_EXCEEDED") for v in tight["violations"])
    cpu = comparability(a, b, same_instances=True, same_oracle=True, same_params_except_pool=True, tol={}, ceilings=Ceilings(cpu_seconds=-1.0))
    assert cpu["cpu_modality"]["decides_status"] is True and cpu["status"] == "NOT_COMPARABLE"                       # CPU só decide se tiver teto
    strict = comparability(a, b, same_instances=True, same_oracle=True, same_params_except_pool=True, tol={"queries": 0.0, "mofn_evaluated": 10.0})
    assert strict["status_candidates_and_ceilings"] == "COMPARABLE" or "mofn_evaluated" in strict["violations_excluding_outcomes"]


def test_generic_matching_scans_the_whole_range_and_reports_feasibility(world):
    X, names, lits = world
    ref = fit_pool_arm(X, names, lits, S.TruthOracle(), dict(P, max_n=2), extras="derived", mf=3 + len(lits))
    target = ref["trace"]["pool"]["counters"]["mofn_evaluated"]
    calls = []
    def fit(mf):
        calls.append(mf)
        return fit_pool_arm(X, names, [], S.TruthOracle(), dict(P, max_n=2), extras="none", mf=mf)
    m = match_generic_by_evaluated(fit, target, range(3, len(names) + 1), tolerance=0.5)
    assert calls == list(range(3, len(names) + 1)) and 3 <= m["mf"] <= len(names)                                    # sem paragem antecipada
    assert m["feasible"] == (m["relative_gap"] <= 0.5)
    impossible = match_generic_by_evaluated(fit, 10 ** 9, range(3, 6), tolerance=0.1)
    assert impossible["feasible"] is False                                                                          # inviável ⇒ NOT_COMPARABLE a jusante


def test_cards_are_linked_generically_and_never_validated_by_default(tmp_path):
    (tmp_path / "c1.json").write_text(json.dumps({"id": "X1", "owl_axiom": "EquivalentClasses(Foo ObjectUnionOf(A B))", "authored_by": "AI_DRAFT", "epistemic_status": "HYPOTHESIS",
                                                  "source": {"verified": False}, "reviews": {"domain": {"status": "DOMAIN_REVIEW_PENDING"}}}))
    c = cards_by_concept(tmp_path)["Foo"]
    assert c["card_id"] == "X1" and c["domain_validation"] == "PENDING" and c["source_verified"] is False and c["reviews"]["domain"] == "DOMAIN_REVIEW_PENDING"


def test_machinery_is_dimension_agnostic():
    X, names, _ = S.make_data(120, seed=3)
    X2 = X[:, :8]                                                                                      # outra dimensionalidade, sem tocar no código
    n2 = names[:8]
    class O:
        calls = 0
        def predict(self, Z):
            return (np.asarray(Z)[:, 2:4].max(axis=1) > 0.5).astype(int)
    r = fit_pool_arm(X2, n2, [], O(), dict(P, max_n=2), extras="none", mf=4)
    assert r["n_columns"] == 8 and r["feature_slots"] == 4


MACHINERY = ("h2_pool_study.py", "h2_trace.py", "h2_axiom_mechanisms.py", "h2_harness.py", "h2_contract.py")


def test_h2_machinery_has_no_dataset_or_code_specific_logic():
    """A maquinaria H2 (exceto os condutores/ontologias de um dataset de validação) não pode conter nomes de datasets, códigos de categorias nem formas fixas: o conhecimento vive em artefactos externos."""
    import re
    from validation.benchmark import datasets as D
    ids = {i.lower() for i in D.REGISTRY} | {"german", "iris", "wine", "digits", "adult", "sonar", "hepatitis"}
    for f in MACHINERY:
        src = (ROOT / "validation" / "benchmark" / f).read_text(encoding="utf-8").lower()
        for i in ids:
            assert not re.search(r"\b%s\b" % re.escape(i), src), (f, i)
        assert not re.search(r"\ba\d{2,3}\b", src), f                                                  # sem códigos de categoria do tipo A71/A410
