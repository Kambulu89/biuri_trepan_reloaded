"""H2 (Doc. 55): correção e rastreabilidade do adaptador T1/T2 (``h2_trace``), em dados sintéticos pequenos com oráculo de forma fechada. Sem métricas de desempenho."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.trepan_original import TrepanOriginalClassifier
from validation.benchmark.h2_axiom_mechanisms import DerivedLiteral
from validation.benchmark.h2_trace import AugmentedOracle, DerivedLiteralTrepan, augment

LV = ["A61", "A62", "A63", "A64", "A65"]
NAMES = [f"sav__{l}" for l in LV] + ["other__a", "other__b"]
KW = dict(min_sample=200, max_queries=1200, max_nodes=5, max_depth=3, max_n=3, beam_width=2, min_samples_leaf=3, max_features_per_node=8, random_state=3)


class Oracle:
    """Regra fechada nas colunas BASE: poupança ≥ A63 (A63 ou A64). Conta chamadas/linhas e guarda as linhas vistas (anti-fuga)."""
    def __init__(self):
        self.calls, self.rows, self.seen = 0, 0, []

    def predict(self, X):
        X = np.asarray(X)
        self.calls += 1
        self.rows += len(X)
        self.seen.append(X.copy())
        assert X.shape[1] == len(NAMES)                                  # nunca recebe colunas derivadas
        return ((X[:, 2] > 0.5) | (X[:, 3] > 0.5)).astype(int)


def data(n=150, seed=0, with_unknown=True):
    rng = np.random.default_rng(seed)
    lv = rng.integers(0, 5 if with_unknown else 4, size=n)
    X = np.zeros((n, 7))
    X[np.arange(n), lv] = 1.0
    X[:, 5:] = rng.integers(0, 2, size=(n, 2))
    return X


def lit(concept, levels):
    return DerivedLiteral(concept, tuple(f"sav__{l}" for l in levels), True, ("F-x",), "sav")


D = [lit("AtLeast_A62", ["A62", "A63", "A64"]), lit("AtLeast_A63", ["A63", "A64"])]


def rules_sha(m):
    return hashlib.sha256(json.dumps(m.export_rules(), default=str).encode()).hexdigest()


def test_disabled_semantics_is_identical_to_the_original_including_oracle_stream():
    X = data()
    o1, o2 = Oracle(), Oracle()
    a = TrepanOriginalClassifier(**KW).fit(X, oracle=o1, feature_names=NAMES)
    b = DerivedLiteralTrepan(base_names=NAMES, **KW).fit(X, oracle=o2, feature_names=NAMES)
    assert rules_sha(a) == rules_sha(b) and a.membership_queries_ == b.membership_queries_
    assert (o1.calls, o1.rows) == (o2.calls, o2.rows)                                           # mesmo fluxo de consultas
    assert all(np.array_equal(x, y) for x, y in zip(o1.seen, o2.seen))                          # mesmas linhas consultadas


def test_derived_columns_are_coherent_in_training_and_queries_and_oracle_sees_only_base_columns():
    X = data()
    Xa, na = augment(X, NAMES, D)
    assert np.array_equal(Xa[:, 7], X[:, 1:4].max(axis=1)) and np.array_equal(Xa[:, 8], X[:, 2:4].max(axis=1)) and na[-1] == "onto_any[AtLeast_A63]"
    captured = []

    class Spy(DerivedLiteralTrepan):
        def _draw_membership_queries(self, model, n, constraints, node):
            q = super()._draw_membership_queries(model, n, constraints, node)
            captured.append(np.array(q))
            return q
    o = Oracle()
    Spy(derived=D, base_names=NAMES, **KW).fit(Xa, oracle=o, feature_names=na)
    assert o.rows > 0 and captured and all(r.shape[1] == len(NAMES) for r in o.seen)          # o oráculo só viu as 7 colunas base
    for q in captured:                                                                         # as colunas derivadas das consultas são OR determinístico das colunas base da mesma consulta
        assert np.array_equal(q[:, 7], q[:, 1:4].max(axis=1)) and np.array_equal(q[:, 8], q[:, 2:4].max(axis=1))


def test_unknown_category_never_maps_to_a_level():
    X = np.zeros((4, 7))                                                                        # nenhuma coluna de poupança a 1 = categoria desconhecida
    Xa, _ = augment(X, NAMES, D)
    assert Xa[:, 7:].sum() == 0
    X[0, 4] = 1.0                                                                               # A65 («desconhecido» documentado) também fica fora das uniões ordinais
    Xa, _ = augment(X, NAMES, D)
    assert Xa[0, 7] == 0 and Xa[0, 8] == 0


def test_trace_links_axiom_literal_candidates_and_decisions():
    X = data(seed=1)
    Xa, na = augment(X, NAMES, D)
    m = DerivedLiteralTrepan(derived=D, base_names=NAMES, **KW).fit(Xa, oracle=Oracle(), feature_names=na)
    t = m.semantic_trace_
    assert t["n_derived"] == 2 and {l["concept"] for l in t["literals"]} == {"AtLeast_A62", "AtLeast_A63"}
    assert all(l["fact_ids"] == ["F-x"] and l["needs_reasoner"] for l in t["literals"])         # proveniência do axioma/inferência preservada
    ev = {l["concept"]: l["candidate_evaluations"] for l in t["literals"]}
    assert ev["AtLeast_A63"] > 0                                                                 # o literal derivado foi realmente avaliado como candidato
    for l in t["literals"]:                                                                      # decisão ⊂ candidatos: um literal escolhido tem de ter sido avaliado
        assert (not l["chosen_in_nodes"]) or l["candidate_evaluations"] > 0
    assert t["n_internal_nodes"] <= KW["max_nodes"]                                              # controlo de complexidade


def test_redundancy_pruning_is_logical_and_data_independent():
    X = data(seed=2)
    Xa, na = augment(X, NAMES, D)
    m = DerivedLiteralTrepan(derived=D, base_names=NAMES, prune_redundant=True, **KW)
    m._setup()
    from core.trepan_original import Literal, MofNTest
    L = lambda f: Literal(f, 0.5, True)
    assert m._redundancy(MofNTest(1, (L(7), L(8)))) == "SUBSUMPTION_COLLAPSE_OR"             # AtLeast_A63 ⊑ AtLeast_A62: OR colapsa
    assert m._redundancy(MofNTest(2, (L(7), L(8)))) == "SUBSUMPTION_COLLAPSE_AND"
    assert m._redundancy(MofNTest(1, (L(2), L(7)))) == "SUBSUMPTION_COLLAPSE_OR"             # nível A63 ⊂ AtLeast_A62
    assert m._redundancy(MofNTest(2, (L(0), L(1)))) == "DISJOINT_UNSATISFIABLE"              # dois níveis da mesma variável: AND insatisfazível (redundante com one-hot)
    assert m._redundancy(MofNTest(1, (L(0), L(1)))) is None and m._redundancy(MofNTest(1, (L(5), L(7)))) is None
    assert m._redundancy(MofNTest(1, (Literal(7, 0.5, False), L(8)))) is None                # polaridade negativa: sem inferência de colapso
    Y1, Y2 = Oracle(), Oracle()
    a = DerivedLiteralTrepan(derived=D, base_names=NAMES, prune_redundant=True, **KW).fit(Xa, oracle=Y1, feature_names=na)
    assert a.semantic_trace_["prune_redundant"] is True                                          # ajuste completo com poda não falha e regista podas por razão
    assert isinstance(a.semantic_trace_["pruned_total"], dict)


def test_counterfactual_ablation_removes_exactly_one_literal_from_the_trace():
    X = data(seed=3)
    full_X, full_n = augment(X, NAMES, D)
    abl_X, abl_n = augment(X, NAMES, D[:1])
    f = DerivedLiteralTrepan(derived=D, base_names=NAMES, **KW).fit(full_X, oracle=Oracle(), feature_names=full_n)
    g = DerivedLiteralTrepan(derived=D[:1], base_names=NAMES, **KW).fit(abl_X, oracle=Oracle(), feature_names=abl_n)
    assert {l["concept"] for l in f.semantic_trace_["literals"]} - {l["concept"] for l in g.semantic_trace_["literals"]} == {"AtLeast_A63"}


def test_no_leakage_construction_is_label_free():
    import inspect
    from validation.benchmark import h2_axiom_mechanisms as M
    assert "y" not in inspect.signature(M.derived_union_literals).parameters and "labels" not in inspect.signature(M.derived_union_literals).parameters
    assert "y_test" not in inspect.signature(DerivedLiteralTrepan.fit).parameters


def test_counterfactual_decision_influence_structure_and_consistency():
    from validation.benchmark.h2_trace import decision_influence
    X = data(seed=4)
    res = decision_influence(X, NAMES, D, Oracle, KW)
    assert set(res) == {"AtLeast_A62", "AtLeast_A63"}
    for r in res.values():
        assert r["n_chosen"] == len(r["chosen_in_nodes"]) and r["n_changed"] == len(r["nodes_changed_by_ablation"])
        assert r["n_chosen"] <= KW["max_nodes"]                                                           # controlo de complexidade; «escolhido» não implica «mudou sob ablação»
    # reprodutibilidade: mesma chamada, mesmo resultado
    assert decision_influence(X, NAMES, D, Oracle, KW) == res
