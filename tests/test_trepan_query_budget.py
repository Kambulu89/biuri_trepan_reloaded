"""O orçamento de queries acompanha o nº de nós permitido (árvore não fica truncada por 2000 queries)."""
import numpy as np

from core.controlled_trepan_experiment import ControlledTrepanConfig
from core.training_config import (get_training_preset, required_query_budget, resolve_trepan_min_sample,
                                  resolve_trepan_query_budget)
from core.trepan_scientific_tuning import _capacity_candidates


def test_required_budget_is_internal_nodes_times_min_sample():
    assert required_query_budget(31, 1000) == 15 * 1000
    assert required_query_budget(61, 1000) == 30 * 1000
    assert required_query_budget(1, 500) == 500          # nunca zero


def test_min_sample_formula_is_unchanged():
    assert resolve_trepan_min_sample(398) == 1000        # o que a GUI já calculava
    assert resolve_trepan_min_sample(50) == 150
    assert resolve_trepan_min_sample(5000) == 5000


def test_scientific_preset_budget_covers_the_allowed_nodes_and_never_shrinks():
    preset = get_training_preset("scientific")
    assert resolve_trepan_query_budget(preset, 398) >= preset.trepan_max_queries
    assert resolve_trepan_query_budget(preset, 398) == 31000   # 63 nós -> 31 internos x 1000
    # amostra pequena: nunca abaixo do orçamento do preset
    assert resolve_trepan_query_budget(preset, 20) >= preset.trepan_max_queries


def test_bigger_capacity_candidates_get_proportionally_more_queries():
    base = ControlledTrepanConfig(max_nodes=31, max_depth=31, min_samples_leaf=4, min_sample=1000, max_n=3, beam_width=2,
                                  max_features_per_node=12, max_queries=15000, random_state=42)
    for c in _capacity_candidates(base, 30, 6):
        assert c.max_queries >= required_query_budget(c.max_nodes, c.min_sample) or c.max_nodes == base.max_nodes
    assert any(c.max_nodes > base.max_nodes for c in _capacity_candidates(base, 30, 6))


def test_scientific_preset_uses_purity_epsilon_001_and_it_reaches_both_extractors():
    from core.trepan_original import TrepanOriginalExtractor
    from core.trepan_reloaded_extractor import TrepanReloadedExtractor
    preset = get_training_preset("scientific")
    assert preset.trepan_purity_epsilon == 0.01
    assert get_training_preset("fast").trepan_purity_epsilon == 0.05      # outros presets mantêm o canónico
    limits = TrepanOriginalExtractor._limits({"purity_epsilon": preset.trepan_purity_epsilon}, 2000, 398)
    assert limits["purity_epsilon"] == 0.01
    assert "purity_epsilon" not in TrepanOriginalExtractor._limits({}, 2000, 398)   # omissão = comportamento anterior
    ext = TrepanReloadedExtractor()
    ext.apply_training_preset(preset)
    assert ext._training_limits["historical_purity_epsilon"] == 0.01


def test_lower_purity_epsilon_grows_a_bigger_tree_with_enough_budget():
    from core.trepan_original import TrepanOriginalClassifier

    class Oracle:
        classes_ = np.array([0, 1])
        def predict(self, X):
            X = np.asarray(X, float)
            return ((X[:, 0] + 0.6 * X[:, 1] * X[:, 2] + 0.2 * np.sin(3 * X[:, 3])) > 0).astype(int)

    X = np.random.default_rng(3).normal(size=(300, 5))
    kw = dict(max_nodes=31, max_depth=8, min_sample=300, max_queries=20000, max_n=2, beam_width=2, random_state=1)
    coarse = TrepanOriginalClassifier(purity_epsilon=0.20, **kw).fit(X, oracle=Oracle(), feature_names=list("abcde"))
    fine = TrepanOriginalClassifier(purity_epsilon=0.01, **kw).fit(X, oracle=Oracle(), feature_names=list("abcde"))
    assert fine.node_count_ >= coarse.node_count_


def test_scientific_preset_allows_63_nodes_and_budget_scales_with_it():
    from core.training_config import resolve_trepan_structure_limits
    preset = get_training_preset("scientific")
    assert resolve_trepan_structure_limits(preset)["max_nodes"] == 63
    assert resolve_trepan_query_budget(preset, 398) == 31 * 1000     # 31 nós internos x min_sample
