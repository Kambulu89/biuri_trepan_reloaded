from core.training_config import (
    TRAINING_PRESETS,
    PRODUCTION_TRAINING_PRESET,
    enforce_scientific_preset,
    get_training_preset,
    resolve_trepan_structure_limits,
)
from core.trepan_reloaded_extractor import TrepanReloadedExtractor


def test_default_training_preset_is_scientific():
    assert PRODUCTION_TRAINING_PRESET == 'scientific'
    assert get_training_preset().key == 'scientific'


def test_production_policy_coerces_non_scientific_preset():
    balanced = TRAINING_PRESETS['balanced']
    locked = enforce_scientific_preset(balanced)
    assert locked.key == 'scientific'
    assert locked.optuna_trials == TRAINING_PRESETS['scientific'].optuna_trials
    assert locked.mlp_max_iter == TRAINING_PRESETS['scientific'].mlp_max_iter


def test_scientific_none_depth_resolves_without_int_none():
    scientific = TRAINING_PRESETS['scientific']
    assert scientific.trepan_max_depth is None
    limits = resolve_trepan_structure_limits(scientific)
    assert isinstance(limits['max_depth'], int)
    assert isinstance(limits['max_nodes'], int)
    assert limits['max_depth'] == limits['max_nodes']


def test_reloaded_scientific_preset_has_integer_runtime_depth():
    extractor = TrepanReloadedExtractor()
    extractor.apply_training_preset(TRAINING_PRESETS['scientific'])
    assert extractor._training_limits['canonical_max_depth'] is not None
    assert isinstance(extractor._training_limits['canonical_max_depth'], int)
    assert extractor._training_limits['canonical_max_depth'] == extractor._training_limits['canonical_max_nodes']


def test_legacy_auxiliary_tree_helper_is_absent():
    import core.trepan_reloaded_extractor as module
    assert not hasattr(module, "_legacy_cart_baseline")


def test_historical_classifier_accepts_unbounded_depth_none():
    from core.trepan_original import TrepanOriginalClassifier
    model = TrepanOriginalClassifier(max_depth=None, max_nodes=5)
    assert model.max_depth is None
