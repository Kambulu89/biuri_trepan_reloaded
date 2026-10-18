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
from validation.benchmark.h2_pool_study import LADDER, PoolTrepan, ablate_literal, comparability, fit_pool_arm, ladder_state, original_buildable

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


def test_ladder_requires_each_level_and_never_assesses_benefit():
    lit = {"fact_ids": ["f1"], "candidate_evaluations": 3}
    pool = {"reasons_by_search": {"IN_LITERAL_POOL": 1}}
    s = ladder_state(lit, pool, validated=True, changed_by_ablation=True)
    assert s["highest_state"] == "SPLIT_DECISION_CHANGED" and s["OBSERVED_BENEFIT"] == "NOT_ASSESSED"
    assert ladder_state(lit, pool, validated=False, changed_by_ablation=True)["highest_state"] == "AXIOM_AVAILABLE"       # sem validação, os níveis seguintes não contam
    assert ladder_state({"fact_ids": []}, {"reasons_by_search": {}}, validated=True, changed_by_ablation=False)["highest_state"] == "AXIOM_VALIDATED"
    assert LADDER[-1] == "OBSERVED_BENEFIT"


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
