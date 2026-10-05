"""Contrato científico: um oráculo congelado, com id/hash, consultado por todas as árvores do benchmark."""
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from core.scientific_experiment_contract import (FrozenOracle, OracleContractViolation, build_frozen_oracle,
                                                 freeze_oracle, oracle_id_of, oracle_identity,
                                                 run_benchmark_with_frozen_oracle, weights_fingerprint)
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 3))
    return X, (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)


def _mlp(X, y, seed=0):
    return MLPClassifier(hidden_layer_sizes=(6,), max_iter=300, random_state=seed).fit(X, y)


def test_oracle_id_is_deterministic_and_changes_with_weights():
    X, y = _data()
    a, b, c = _mlp(X, y, 0), _mlp(X, y, 0), _mlp(X, y, 1)
    assert oracle_identity(a, X).oracle_id == oracle_identity(b, X).oracle_id          # mesmo treino -> mesmo oráculo
    assert oracle_identity(a, X).oracle_id != oracle_identity(c, X).oracle_id          # pesos diferentes -> id diferente
    assert weights_fingerprint(a) == weights_fingerprint(b) != weights_fingerprint(c)


def test_frozen_oracle_refuses_refit_and_detects_mutation():
    X, y = _data()
    f = freeze_oracle(_mlp(X, y), X)
    with pytest.raises(OracleContractViolation):
        f.fit(X, y)
    with pytest.raises(OracleContractViolation):
        f.partial_fit(X, y)
    assert f.verify_unchanged()
    f.model.fit(X, 1 - y)                                                              # alguém reajusta por baixo
    with pytest.raises(OracleContractViolation):
        f.verify_unchanged()


def test_all_trees_query_the_same_oracle_and_the_proof_is_recorded():
    X, y = _data()
    f = freeze_oracle(_mlp(X, y), X, builder="test")
    kw = dict(max_nodes=7, min_sample=40, max_queries=2000, random_state=1)
    names = ["a", "b", "c"]
    report = run_benchmark_with_frozen_oracle(f, {
        "trepan_original": lambda o: TrepanOriginalClassifier(**kw).fit(X, oracle=o, feature_names=names),
        "trepan_reloaded": lambda o: TrepanReloadedClassifier(**kw).fit(X, oracle=o, feature_names=names),
        "trepan_variant_eps001": lambda o: TrepanOriginalClassifier(**{**kw, "purity_epsilon": 0.01}).fit(X, oracle=o, feature_names=names),
    })
    assert report["single_oracle"] and report["unchanged_after"]
    assert set(report["trees"]) == {"trepan_original", "trepan_reloaded", "trepan_variant_eps001"}
    assert {t["oracle_id"] for t in report["trees"].values()} == {f.oracle_id}
    assert all(t["queries"] > 0 for t in report["trees"].values())
    assert report["oracle"]["oracle_id"] == f.oracle_id and report["oracle"]["probe_rows"] == len(X)


def test_a_tree_built_on_a_different_oracle_breaks_the_contract():
    X, y = _data()
    f = freeze_oracle(_mlp(X, y), X)
    other = freeze_oracle(_mlp(X, y, 5), X)
    assert f.oracle_id != other.oracle_id
    assert oracle_id_of(f) == f.oracle_id and oracle_id_of(object()) is None

    class Adapter:                                      # envoltório tipo OriginalOracleProjection
        def __init__(self, oracle):
            self.oracle = oracle
    assert oracle_id_of(Adapter(f)) == f.oracle_id and oracle_id_of(Adapter(Adapter(other))) == other.oracle_id


def test_oracle_builders_are_registered_and_unknown_is_rejected():
    X, y = _data()
    with pytest.raises(ValueError):
        build_frozen_oracle(X, y, seed=1, builder="inexistente")
    f = build_frozen_oracle(X, y, seed=1, builder="factory")
    assert isinstance(f, FrozenOracle) and f.identity.builder == "factory"
    g = build_frozen_oracle(X, y, seed=1, builder="factory")
    assert f.oracle_id == g.oracle_id                     # reprodutível


def test_production_report_proves_single_oracle_for_original_and_reloaded():
    import pandas as pd
    from core.production_training import train_production_dataframe
    from core.trepan_scientific_tuning import ScientificTrepanSearchConfig
    X, y = _data(3)
    df = pd.DataFrame(X, columns=["a", "b", "c"]); df["y"] = np.where(y == 1, "pos", "neg")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rep = train_production_dataframe(df, target="y", out_dir=tmp, seed=3, scientific_tuning=True,
                                         trepan_search=ScientificTrepanSearchConfig(
                                             cv_folds=2, cv_repeats=1, max_capacity_candidates=1, max_semantic_candidates=1,
                                             purity_epsilon_grid=(0.05,), max_nodes_grid=(7,)))
    oc = rep["evaluation"]["oracle_contract"]
    ids = oc["tree_oracle_ids"]
    assert ids["trepan_original"] == ids["trepan_reloaded"] == oc["oracle"]["oracle_id"]
    assert oc["single_oracle_for_all_trees"] and oc["unchanged_after"]
    assert {"tuning", "trepan_pair"} <= set(oc["trees"])
    assert rep["evaluation"]["trepan_scientific_tuning"]["budget_check"]["any_budget_exhausted"] is False


def test_robust_builder_handles_string_labels_and_is_a_valid_frozen_oracle():
    X, y = _data(5)
    labels = np.where(y == 1, "pos", "neg")
    f = build_frozen_oracle(X, labels, seed=1, builder="robust")
    assert set(f.predict(X)) <= {"pos", "neg"} and list(f.classes_) == ["neg", "pos"]
    assert f.identity.builder == "robust" and f.verify_unchanged()


def test_robust_oracle_is_a_cloneable_estimator_usable_by_the_health_gate():
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    X, y = _data(6)
    labels = np.where(y == 1, "pos", "neg")
    f = build_frozen_oracle(X, labels, seed=1, builder="robust")
    assert len(cross_val_score(clone(f.model), X, labels, cv=2)) == 2
