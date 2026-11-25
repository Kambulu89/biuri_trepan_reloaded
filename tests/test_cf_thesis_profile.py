"""Perfil «tese» da melhoria contrafactual (Eldis Dennys González Pérez, 2026) e diagnóstico de consistência A/B/C.

O protocolo BIURI continua a ser o comportamento por omissão (equivalência bit a bit verificada contra a main com 30 casos;
aqui fixam-se os valores por omissão). O perfil da tese reproduz o método do Cap. 2 e de ``pipeline_improve.py`` com duas
correções de fuga de informação: CFs de cada partição só das linhas de ajuste dessa partição, e accuracy do oráculo só do treino.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from counterfactuals.surrogate_improvement import (
    CandidateParameters,
    ImprovementSearchConfig,
    THESIS_PROFILE_NOTES,
    _effective_sample_size,
    _parameter_grid,
    consistency_indicators,
    prepare_counterfactual_augmentation,
    select_and_refit_improved_surrogate,
)


class Threshold:
    classes_ = np.asarray([0, 1])

    def __init__(self, threshold, nodes=3, confidence=0.9):
        self.threshold = float(threshold)
        self.tree_ = SimpleNamespace(node_count=int(nodes))
        self.confidence = confidence

    def predict(self, values):
        return (np.asarray(values, dtype=float)[:, 0] > self.threshold).astype(int)

    def predict_proba(self, values):
        pred = self.predict(values)
        c = np.full(len(pred), self.confidence)
        return np.column_stack([np.where(pred == 0, c, 1 - c), np.where(pred == 1, c, 1 - c)])

    def get_depth(self):
        return 1

    def get_n_leaves(self):
        return 2


def test_biuri_protocol_stays_the_default():
    c = ImprovementSearchConfig()
    assert (c.profile, c.selection_protocol, c.selection_objective, c.divergent_boost) == ("biuri", "holdout", "multiobjective", "fixed")
    assert (c.divergent_fixed_weight, c.gap_weight, c.min_anchors, c.min_cf_weight) == (2.0, 0.30, 1, 0.0)
    assert c.use_frontier_weight_grid == (True,) and c.adjust_ratio_by_oracle_accuracy is False
    assert c.cf_weight_budget_rule == "sample_size" and _effective_sample_size(c, 999) == c.sample_size
    assert len(_parameter_grid(c)) == len(c.max_cfs_grid) * len(c.cf_types_grid) * len(c.cf_weight_ratio_grid)


def test_thesis_profile_matches_the_thesis_method():
    t = ImprovementSearchConfig.thesis_profile()
    assert t.profile == "tese" and t.selection_protocol == "cv2" and t.selection_objective == "fidelity"
    assert t.max_cfs_grid == (5, 10) and t.cf_types_grid == ("A+B", "A+B+C") and t.use_frontier_weight_grid == (False, True)
    assert len(_parameter_grid(t)) == 8                    # 16 combinações da tese, das quais 8 distintas (include_type_C redundante)
    assert (t.divergent_boost, t.gap_weight, t.min_anchors, t.min_cf_weight) == ("adaptive", 0.30, 50, 0.1)
    assert (t.confidence_threshold, t.density_percentile, t.strict_density_gate) == (0.60, 95.0, True)
    assert t.max_origins == 0 and t.origin_fraction == 0.33 and t.sample_size == 2000
    assert _effective_sample_size(t, 100) == 3000 and _effective_sample_size(t, 800) == 4000 and _effective_sample_size(t, 5000) == 5000
    via_options = ImprovementSearchConfig.from_options({"profile": "tese", "sample_size": 1500, "max_origins": 12})
    assert via_options.profile == "tese" and via_options.sample_size == 1500 and via_options.max_origins == 12
    assert "partição" in THESIS_PROFILE_NOTES and "agnóstico" in THESIS_PROFILE_NOTES


def _divergent_setup(n_divergent):
    """Oráculo com 3 classes simuladas por limiares: o CF muda de classe no oráculo e na árvore, mas para classes diferentes."""
    class Multi(Threshold):
        classes_ = np.asarray([0, 1, 2])

        def __init__(self, cuts):
            super().__init__(0)
            self.cuts = cuts

        def predict(self, values):
            return np.digitize(np.asarray(values, dtype=float)[:, 0], self.cuts)

        def predict_proba(self, values):
            pred = self.predict(values)
            out = np.full((len(pred), 3), 0.05)
            out[np.arange(len(pred)), pred] = 0.9
            return out

    X = np.linspace(-1, 3, 60).reshape(-1, 1)
    oracle, tree = Multi([0.0, 1.0]), Multi([0.0, 2.0])     # em ]1, 2]: oráculo=2, árvore=1 (divergente)
    cfs = [{"cf": np.asarray([1.2 + 0.01 * i]), "original_class": 0} for i in range(n_divergent)]
    return oracle, tree, X, oracle.predict(X), cfs


@pytest.mark.parametrize("n_b,expected", [(2, 4.0), (5, 3.0), (20, 1.5)])
def test_adaptive_divergent_boost_follows_the_thesis_formula(n_b, expected):
    oracle, tree, X, y, cfs = _divergent_setup(n_b)
    cfg = ImprovementSearchConfig.thesis_profile(strict_density_gate=False, max_cf_ratio=1.0)
    batch = prepare_counterfactual_augmentation(oracle, tree, X, y, cfs, parameters=CandidateParameters(50, "A+B", 0.05, True), config=cfg)
    assert batch.stats["eligible_by_type"]["B"] == n_b
    assert batch.stats["divergent_weight"] == pytest.approx(expected)          # 1 + min(3, 10 / n_B)
    fixed = prepare_counterfactual_augmentation(oracle, tree, X, y, cfs, parameters=CandidateParameters(50, "A+B", 0.05, True),
                                                config=ImprovementSearchConfig(strict_density_gate=False, max_cf_ratio=1.0))
    assert fixed.stats["divergent_weight"] == 2.0


def test_anchors_minimum_weight_floor_budget_and_accuracy_adjustment():
    X = np.linspace(-2, 2, 40).reshape(-1, 1)
    oracle, tree = Threshold(0.0), Threshold(1.0)
    y = oracle.predict(X)
    y_noisy = y.copy()
    y_noisy[:4] = 1 - y_noisy[:4]                                             # accuracy do oráculo no treino = 0,9
    cfs = [{"cf": np.asarray([0.1 + 0.05 * i]), "original_class": 0} for i in range(6)]
    t = ImprovementSearchConfig.thesis_profile(strict_density_gate=False, max_cf_ratio=1.0)
    batch = prepare_counterfactual_augmentation(oracle, tree, X, y_noisy, cfs, parameters=CandidateParameters(10, "A+B+C", 0.05, False), config=t)
    assert batch.stats["used_real_anchor"] == 40                             # max(50, 20 % de 40) limitado às 40 linhas
    assert batch.stats["oracle_accuracy_on_fit"] == pytest.approx(0.9)
    assert batch.stats["effective_cf_weight_ratio"] == pytest.approx(min(0.05 * 0.9 / 0.9, 0.10))
    assert batch.stats["effective_sample_size"] == 3000                     # min(5000, max(3000, 5·40))
    cf_weights = batch.weights[: batch.stats["used"]]
    assert cf_weights.min() >= 0.1 - 1e-12 and cf_weights.max() <= t.max_cf_weight + 1e-12
    default = prepare_counterfactual_augmentation(oracle, tree, X, y_noisy, cfs, parameters=CandidateParameters(10, "A+B+C", 0.05),
                                                  config=ImprovementSearchConfig(strict_density_gate=False, max_cf_ratio=1.0, sample_size=100))
    assert default.stats["used_real_anchor"] == 8 and default.stats["effective_cf_weight_ratio"] == 0.05


def test_consistency_indicators_use_the_thesis_definitions():
    oracle, tree = Threshold(0.0), Threshold(1.0)
    cfs = [
        {"cf": np.asarray([2.0]), "original_class": 0},    # oráculo 1, árvore 1 -> coincidente (A)
        {"cf": np.asarray([0.5]), "original_class": 0},    # oráculo 1, árvore 0 -> brecha (C)
        {"cf": np.asarray([0.7]), "original_class": 0},    # brecha (C)
        {"cf": np.asarray([-1.0]), "original_class": 0},   # oráculo 0 = origem -> não é válido no oráculo
    ]
    ind = consistency_indicators(oracle, tree, cfs)
    assert (ind["total_cfs"], ind["valid_oracle"], ind["valid_tree"], ind["coincident"], ind["divergent"], ind["gap"]) == (4, 3, 1, 1, 0, 2)
    assert ind["indicador_a"] == pytest.approx(1 / 3) and ind["indicador_b"] == 0.0 and ind["indicador_c"] == pytest.approx(2 / 3)
    assert np.isnan(consistency_indicators(oracle, tree, [])["indicador_c"])


def test_two_fold_cv_generates_counterfactuals_only_from_each_fold_fit_rows(monkeypatch):
    X = np.linspace(-2.0, 2.0, 80).reshape(-1, 1)
    oracle = Threshold(0.0)
    y = oracle.predict(X)
    seen = []

    def fake_pool(oracle_, X_fit, y_fit, names, **kwargs):
        seen.append(set(np.round(np.asarray(X_fit)[:, 0], 8)))
        return ([{"cf": np.asarray([0.1]), "original_class": 0}, {"cf": np.asarray([-0.1]), "original_class": 1}],
                {"data_role": "development_fit_only", "test_queried": False})

    monkeypatch.setattr("counterfactuals.surrogate_improvement.generate_counterfactual_pool", fake_pool)
    validation_sets = []

    def builder(request):
        validation_sets.append(set(np.round(np.asarray(request.X_validation)[:, 0], 8)))
        if request.extra_X is None:
            return Threshold(0.65), {}
        # a configuração com fronteira desligada fica mais fiel: tem de ser a escolhida pelo objetivo «fidelidade»
        return Threshold(0.0 if not request.parameters.use_frontier_weight else 0.3), {}

    cfg = ImprovementSearchConfig.thesis_profile(strict_density_gate=False, max_cf_ratio=1.0, max_complexity_ratio=2.0,
                                                 max_cfs_grid=(2,), cf_types_grid=("A+B+C",))
    result = select_and_refit_improved_surrogate(model_name="TREPAN Original", oracle=oracle, original_tree=Threshold(1.0),
                                                 X_development=X, y_development=y, feature_names=["x"], candidate_builder=builder,
                                                 config=cfg)
    sizes = result["partition_sizes"]
    assert sizes["configuration_protocol"] == "cv2" and len(sizes["configuration_folds"]) == 2
    fold_fit_sets = seen[:2]
    assert all(len(s) == n_fit for s, (n_fit, _) in zip(fold_fit_sets, sizes["configuration_folds"]))
    assert not (fold_fit_sets[0] & fold_fit_sets[1])                          # as duas metades não se sobrepõem
    audit_rows = set(np.round(X[:, 0], 8)) - fold_fit_sets[0] - fold_fit_sets[1]
    assert len(audit_rows) == sizes["acceptance_holdout"]                    # o bloco de aceitação nunca originou CFs na seleção
    assert result["selected_parameters"]["use_frontier_weight"] is False     # máxima fidelidade média
    assert result["consistency_diagnostic"]["selection"][0]["fold"] == 0
    assert result["profile_notes"] == THESIS_PROFILE_NOTES
    assert result["locked_test_used"] is False and result["final_test_evaluated"] is False


def test_gui_dialog_offers_the_thesis_profile(qapp=None):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from gui.surrogate_improvement_dialog import SurrogateImprovementDialog
    dlg = SurrogateImprovementDialog(ontology_active=False)
    try:
        assert "max_cfs_grid" in dlg.options() and "profile" not in dlg.options()       # BIURI por omissão
        dlg.profile_combo.setCurrentIndex(dlg.profile_combo.findData("tese"))
        opts = dlg.options()
        assert opts["profile"] == "tese" and "max_cfs_grid" not in opts and opts["sample_size"] == 2000
        assert not dlg.max_cfs_spin.isEnabled()
        assert opts["seed_weighting"] == "ignore"
        dlg.replicate_weights.setChecked(True)
        assert dlg.options()["seed_weighting"] == "replicate"
        assert ImprovementSearchConfig.from_options({k: v for k, v in opts.items() if k != "target_model"}).selection_protocol == "cv2"
    finally:
        dlg.close()
    assert app is not None


# ---- pesos dos CFs com efeito real na árvore (replicação das sementes; o TREPAN canónico não aceita pesos)
from counterfactuals.surrogate_improvement import replicate_weighted_seeds


def test_replication_turns_weights_into_multiplicities():
    X = np.arange(8, dtype=float).reshape(4, 2)
    out, audit = replicate_weighted_seeds(X, np.asarray([0.1, 1.0, 2.6, 4.0]))
    assert out.shape == (1 + 1 + 3 + 4, 2) and audit["rows_passed"] == 9 and audit["copies_min"] == 1 and audit["copies_max"] == 4
    assert np.array_equal(out[2:5], np.repeat(X[2:3], 3, axis=0))
    assert audit["quantization_abs_error"] == pytest.approx(0.9 + 0.0 + 0.4 + 0.0)
    same, audit_ignore = replicate_weighted_seeds(X, np.full(4, 9.0), mode="ignore")
    assert np.array_equal(same, X) and audit_ignore["seed_weighting"] == "ignore"
    assert replicate_weighted_seeds(None, None)[0] is None
    with pytest.raises(ValueError):
        replicate_weighted_seeds(X, np.asarray([1.0, -1.0, 1.0, 1.0]))
    assert ImprovementSearchConfig().seed_weighting == "ignore"                    # por omissão: pesos não replicados
    assert ImprovementSearchConfig.thesis_profile().seed_weighting == "ignore"
    assert ImprovementSearchConfig.from_options({"seed_weighting": "replicate"}).seed_weighting == "replicate"
    assert ImprovementSearchConfig.from_options({"profile": "tese", "seed_weighting": "replicate"}).seed_weighting == "replicate"


def test_weights_change_the_real_trepan_tree_and_ignore_mode_is_the_previous_behaviour():
    import warnings
    from core.trepan_original import TrepanOriginalExtractor

    class Oracle:
        classes_ = np.asarray([0, 1])

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            return ((X[:, 0] + 0.6 * X[:, 1] ** 2 - 0.3 * X[:, 2]) > 0.4).astype(int)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    oracle = Oracle()
    cfs = rng.normal(size=(15, 3))
    cfs[:, 0] = 0.4 - 0.6 * cfs[:, 1] ** 2 + 0.3 * cfs[:, 2] + rng.normal(scale=0.05, size=15)
    probe = rng.normal(size=(2000, 3))

    def predictions(seed_X):
        extractor = TrepanOriginalExtractor(random_state=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extractor.extract_tree(oracle, X, oracle.predict(X), sample_size=600, feature_names=["a", "b", "c"],
                                   X_train=X, y_train=oracle.predict(X), X_test=X[:20], y_test=oracle.predict(X[:20]), extra_X=seed_X)
        return extractor.explainer_tree.predict(probe)

    weight_one = predictions(replicate_weighted_seeds(cfs, np.ones(15))[0])
    weight_ten = predictions(replicate_weighted_seeds(cfs, np.full(15, 10.0))[0])
    assert not np.array_equal(weight_one, weight_ten)                               # o peso muda a árvore
    assert np.array_equal(predictions(replicate_weighted_seeds(cfs, np.full(15, 10.0), mode="ignore")[0]),
                          predictions(cfs))                                         # "ignore" = caminho anterior
