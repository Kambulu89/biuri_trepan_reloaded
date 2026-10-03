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
    assert resolve_trepan_query_budget(preset, 398) == 15000
    # amostra pequena: nunca abaixo do orçamento do preset
    assert resolve_trepan_query_budget(preset, 20) >= preset.trepan_max_queries


def test_bigger_capacity_candidates_get_proportionally_more_queries():
    base = ControlledTrepanConfig(max_nodes=31, max_depth=31, min_samples_leaf=4, min_sample=1000, max_n=3, beam_width=2,
                                  max_features_per_node=12, max_queries=15000, random_state=42)
    for c in _capacity_candidates(base, 30, 6):
        assert c.max_queries >= required_query_budget(c.max_nodes, c.min_sample) or c.max_nodes == base.max_nodes
    assert any(c.max_nodes > base.max_nodes for c in _capacity_candidates(base, 30, 6))
