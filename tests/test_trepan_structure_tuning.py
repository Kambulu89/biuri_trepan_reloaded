"""Tuning científico da estrutura do TREPAN: grelha purity_epsilon x max_nodes, só no treino, com fallback canónico."""
import numpy as np
import pytest

from core.controlled_trepan_experiment import ControlledTrepanConfig
from core.training_config import get_training_preset, resolve_trepan_structure_limits
from core.trepan_scientific_tuning import (ScientificTrepanSearchConfig, _capacity_candidates, _structure_candidates,
                                           tune_scientific_trepan)


class Oracle:
    classes_ = np.array([0, 1])

    def predict(self, X):
        X = np.asarray(X, float)
        return ((X[:, 0] + 0.6 * X[:, 1] * X[:, 2]) > 0).astype(int)

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.c_[1 - p, p]


def _base(**kw):
    values = dict(max_nodes=31, max_depth=31, min_samples_leaf=2, min_sample=300, max_n=2, beam_width=1,
                  max_features_per_node=4, max_queries=6000, random_state=3)
    values.update(kw)
    return ControlledTrepanConfig(**values)


def test_canonical_defaults_are_the_literature_values_not_dataset_tuned():
    preset = get_training_preset("scientific")
    assert preset.trepan_purity_epsilon == 0.05
    assert resolve_trepan_structure_limits(preset)["max_nodes"] == 31
    assert ControlledTrepanConfig().purity_epsilon == 0.05 and ControlledTrepanConfig().max_nodes == 31
    for key in ("fast", "balanced", "scientific"):
        assert get_training_preset(key).trepan_purity_epsilon == 0.05


def test_structure_grid_has_exactly_the_requested_candidates_with_base_first():
    search = ScientificTrepanSearchConfig()
    assert search.purity_epsilon_grid == (0.05, 0.02, 0.01) and search.max_nodes_grid == (31, 63)
    cands = _structure_candidates(_base(), search)
    assert (cands[0].purity_epsilon, cands[0].max_nodes) == (0.05, 31)           # base canónica primeiro
    assert {(c.purity_epsilon, c.max_nodes) for c in cands} == {(e, n) for e in (0.05, 0.02, 0.01) for n in (31, 63)}
    assert len(cands) == 6
    big = [c for c in cands if c.max_nodes == 63]
    assert all(c.max_queries >= 30 * 300 for c in big)                           # orçamento acompanha os nós
    assert all(c.max_depth >= 6 for c in cands)


def test_tuning_runs_the_structure_grid_on_training_data_only_and_records_it():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(180, 4))
    y = Oracle().predict(X)
    result = tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abcd"), base_config=_base(),
                                    search=ScientificTrepanSearchConfig(cv_folds=2, max_capacity_candidates=2, max_semantic_candidates=1))
    assert result["test_used_for_selection"] is False and result["selection_scope"] == "training_cv_only"
    assert len(result["structure_history"]) == 6
    assert result["structure_selected"]["purity_epsilon"] in (0.05, 0.02, 0.01)
    assert result["structure_selected"]["max_nodes"] in (31, 63)
    assert result["common_capacity"]["max_nodes"] in (31, 63)                    # a etapa seguinte não cresce além da grelha
    assert result["common_capacity"]["purity_epsilon"] == result["structure_selected"]["purity_epsilon"]


def test_without_capacity_search_or_with_tuning_off_the_canonical_base_is_kept():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 3))
    y = Oracle().predict(X)
    for search in (ScientificTrepanSearchConfig(cv_folds=2, max_capacity_candidates=1, max_semantic_candidates=1),
                   ScientificTrepanSearchConfig(cv_folds=2, max_capacity_candidates=3, max_semantic_candidates=1, tune_structure=False),
                   ScientificTrepanSearchConfig(cv_folds=2, max_capacity_candidates=3, max_semantic_candidates=1,
                                                purity_epsilon_grid=())):
        r = tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(), search=search)
        assert r["structure_history"] == []
        assert (r["common_capacity"]["purity_epsilon"], r["common_capacity"]["max_nodes"]) == (0.05, 31)


def test_capacity_stage_does_not_grow_nodes_beyond_the_grid_when_structure_was_tuned():
    base = _base(max_nodes=63, max_depth=63)
    assert max(c.max_nodes for c in _capacity_candidates(base, 5, 6, grow_nodes=False)) == 63
    assert max(c.max_nodes for c in _capacity_candidates(base, 5, 6, grow_nodes=True)) > 63


def test_all_candidates_failing_falls_back_to_the_canonical_base(monkeypatch):
    import core.trepan_scientific_tuning as mod

    class Boom:
        def __init__(self, **kw):
            raise RuntimeError("falha simulada")

    monkeypatch.setattr(mod, "TrepanOriginalClassifier", Boom)
    X = np.random.default_rng(2).normal(size=(90, 3))
    r = tune_scientific_trepan(X, Oracle().predict(X), oracle=Oracle(), feature_names=list("abc"), base_config=_base(),
                               search=ScientificTrepanSearchConfig(cv_folds=2, max_capacity_candidates=3, max_semantic_candidates=1))
    assert (r["common_capacity"]["purity_epsilon"], r["common_capacity"]["max_nodes"]) == (0.05, 31)
    assert all(h["failed"] for h in r["structure_history"])
