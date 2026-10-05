"""Otimizações de engenharia NÃO alteram resultados: cada atalho é comparado com a implementação de referência (exatidão bit a bit)."""
from __future__ import annotations

import json
from contextlib import contextmanager

import numpy as np
import pytest

from core import trepan_original as T
from core.benchmark.runner import BenchmarkConfig, BenchmarkRunner, TreeBudget, default_arms
from core.benchmark.semantic import GroupSemanticProvider, OwlSemanticProvider, ShuffledSemanticProvider
from core.benchmark.synthetic import make_synthetic
from core.trepan_scientific_tuning import ScientificTrepanSearchConfig
from validation.benchmark import datasets as ds_mod
from validation.benchmark import equivalence as eq


# ----------------------------------------------------------------------------------------- referência (código antes da otimização)
def _legacy_thresholds(sv, sy):
    out = []
    for i in range(len(sv) - 1):
        if not np.isfinite(sv[i]) or not np.isfinite(sv[i + 1]):
            continue
        if sv[i] == sv[i + 1] or sy[i] == sy[i + 1]:
            continue
        out.append((sv[i] + sv[i + 1]) / 2.0)
    return out


def _legacy_sample_column(self, j, n):
    spec = self.features_[j]
    if spec.discrete:
        return self._rng.choice(spec.values, size=n, replace=True, p=spec.probabilities)
    seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
    return spec.kde.sample(n_samples=n, random_state=seed).reshape(-1)


@contextmanager
def legacy_engine():
    """Repõe as implementações de referência (lentas) para comparar com o caminho otimizado."""
    saved = (T._information_gain, T.TrepanOriginalClassifier.__dict__["_boundary_thresholds"], T.FeatureDistributionModel._sample_column)
    T._information_gain = T._information_gain_legacy
    T.TrepanOriginalClassifier._boundary_thresholds = staticmethod(_legacy_thresholds)
    T.FeatureDistributionModel._sample_column = _legacy_sample_column
    try:
        yield
    finally:
        T._information_gain, T.TrepanOriginalClassifier._boundary_thresholds, T.FeatureDistributionModel._sample_column = saved


# ----------------------------------------------------------------------------------------- primitivas exatas
@pytest.mark.parametrize("n_classes", [2, 3, 7, 10, 12])
def test_information_gain_is_bit_identical_to_reference(n_classes):
    rng = np.random.default_rng(n_classes)
    for trial in range(40):
        classes = np.arange(n_classes) if trial % 2 else np.sort(rng.choice(200, n_classes, replace=False))
        y = rng.choice(classes, int(rng.integers(5, 400)))
        for _ in range(25):
            mask = rng.random(len(y)) < rng.random()
            assert T._information_gain(y, mask, classes) == T._information_gain_legacy(y, mask, classes)


def test_information_gain_falls_back_safely_and_cache_never_goes_stale():
    rng = np.random.default_rng(0)
    classes = np.array([0, 1, 2])
    y1, y2 = rng.choice(classes, 50), rng.choice(classes, 50)
    m = rng.random(50) < 0.5
    for _ in range(3):                                     # alternar objetos ``y`` do mesmo tamanho não pode reutilizar códigos antigos
        assert T._information_gain(y1, m, classes) == T._information_gain_legacy(y1, m, classes)
        assert T._information_gain(y2, m, classes) == T._information_gain_legacy(y2, m, classes)
    assert T._information_gain(y1, m.astype(int), classes) == T._information_gain_legacy(y1, m.astype(int), classes)     # máscara não booleana
    assert T._information_gain(np.array([5, 9, 5, 9, 9]), np.array([1, 0, 1, 0, 0], bool), classes) == \
        T._information_gain_legacy(np.array([5, 9, 5, 9, 9]), np.array([1, 0, 1, 0, 0], bool), classes)               # rótulo fora de ``classes``
    unsorted = np.array([2, 0, 1])
    assert T._information_gain(y1, m, unsorted) == T._information_gain_legacy(y1, m, unsorted)
    ys, cs = np.array(["b", "a", "b", "c", "a", "b"]), np.array(["a", "b", "c"])
    mm = np.array([1, 0, 1, 1, 0, 0], bool)
    assert T._information_gain(ys, mm, cs) == T._information_gain_legacy(ys, mm, cs)


def test_vectorized_thresholds_match_reference_loop():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(0, 120))
        sv = np.sort(rng.choice(np.r_[rng.normal(size=30), np.nan, np.inf], n)) if n else np.array([])
        sy = rng.integers(0, 4, n)
        assert T.TrepanOriginalClassifier._boundary_thresholds(sv, sy) == [float(v) for v in _legacy_thresholds(sv, sy)]


def test_kde_sampling_stream_is_unchanged():
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.normal(size=300), rng.integers(0, 5, 300), rng.exponential(size=300)])
    new = T.FeatureDistributionModel(random_state=7).fit(X)
    old = T.FeatureDistributionModel(random_state=7).fit(X)
    for n in (1, 17, 256, 1000):
        a = new.draw(n, None)
        with legacy_engine():
            b = old.draw(n, None)
        assert np.array_equal(a, b)


# ----------------------------------------------------------------------------------------- semântica: reutilização do contexto
def test_shuffled_from_context_equals_rebuild_and_skips_reasoning():
    spec = ds_mod.REGISTRY["wine"]
    ds, _, _ = spec.load()
    provider = OwlSemanticProvider(str(spec.ontology_path))
    real = provider.build(ds.X, ds.feature_names, 3)
    assert real.available, real.reason
    calls_after_real = provider.n_reasoner_calls
    reused = ShuffledSemanticProvider(provider).from_context(real, 3)
    assert provider.n_reasoner_calls == calls_after_real                    # sem novo reasoning nem recarregamento da OWL
    rebuilt = ShuffledSemanticProvider(provider).build(ds.X, ds.feature_names, 3)
    assert provider.n_reasoner_calls == calls_after_real + 1                # caminho antigo: repete o reasoning
    assert reused.structure_signature() == rebuilt.structure_signature()
    assert reused.info["shuffled_ontology_hash"] == rebuilt.info["shuffled_ontology_hash"]
    assert reused.enriched_names == rebuilt.enriched_names and reused.groups == rebuilt.groups
    probe = np.random.default_rng(0).normal(size=(40, ds.X.shape[1])) * ds.X.std(axis=0) + ds.X.mean(axis=0)
    assert np.array_equal(reused.enrich(probe), rebuilt.enrich(probe))
    assert real.available and real.structure_signature() != reused.structure_signature()      # o contexto real não foi mutado


# ----------------------------------------------------------------------------------------- ponta a ponta: A–F antes/depois
def _run(ds, groups, n_classes_cfg=None):
    cheap = ScientificTrepanSearchConfig(cv_folds=2, cv_repeats=1, purity_epsilon_grid=(0.05, 0.02), max_nodes_grid=(7, 15))
    cfg = BenchmarkConfig(seeds=(42,), tree=TreeBudget(min_sample=150, max_queries=2000, max_nodes=7, max_depth=5), extra_ablations=False,
                          different_oracle_experiment=False, structure_tuning=True, structure_search=cheap, n_boot=50, inner_cv_splits=2)
    arms = [a for a in default_arms(cfg) if a.group in ("A", "B", "C", "D", "E", "F")]
    return BenchmarkRunner(cfg, arms).run(ds, GroupSemanticProvider(groups))


@pytest.mark.parametrize("n_classes,n_features", [(2, 7), (10, 12)])
def test_optimized_pipeline_is_identical_to_reference_pipeline(n_classes, n_features):
    ds, groups = make_synthetic(f"eq_{n_classes}", n_samples=260, n_features=n_features, n_classes=n_classes, n_groups=3, seed=11)
    new = _run(ds, groups)
    with legacy_engine():
        old = _run(ds, groups)
    assert not new.skipped and not old.skipped, (new.skipped, old.skipped)
    diffs = eq.compare_frames(new.frame(), old.frame())
    assert not diffs, diffs[:10]
    assert new.semantic_report[0]["structural_protocol"]["selected"] == old.semantic_report[0]["structural_protocol"]["selected"]
    assert np.array_equal(new.predictions.filter(like="pred__").to_numpy(), old.predictions.filter(like="pred__").to_numpy())
    assert new.semantic_report[0]["work"]["number_cv_fits"] == old.semantic_report[0]["work"]["number_cv_fits"] > 0


# ----------------------------------------------------------------------------------------- o comparador de equivalência
def test_equivalence_comparator_detects_differences(tmp_path):
    from validation.benchmark import raw_store as rs
    ds, groups = make_synthetic("eq_cmp", n_samples=120, n_features=6, n_classes=2, n_groups=2, seed=5)
    result = _run(ds, groups)
    a, b = tmp_path / "a", tmp_path / "b"
    rs.write_unit(result, a, manifest_sha256="x", labels=[0, 1])
    rs.write_unit(result, b, manifest_sha256="x", labels=[0, 1])
    assert eq.compare_raw_roots(a, b)["identical"]
    target = b / "raw" / "eq_cmp" / "seed42" / "trepan_original.json"
    rec = json.loads(target.read_text())
    rec["predictions"]["surrogate_predictions"][0] = 1 - rec["predictions"]["surrogate_predictions"][0]
    target.chmod(0o644); target.write_text(json.dumps(rec))
    out = eq.compare_raw_roots(a, b)
    assert not out["identical"] and any("surrogate_predictions" in d for d in out["differences"])
    rec["predictions"]["surrogate_predictions"][0] = 1 - rec["predictions"]["surrogate_predictions"][0]
    rec["row"]["tree_training_time"] = 12345.0                                             # tempos não contam
    target.write_text(json.dumps(rec))
    assert eq.compare_raw_roots(a, b)["identical"]


# ----------------------------------------------------------------------------------------- tuning: memoização exata e etapa semântica
def _tune(**kw):
    from core.controlled_trepan_experiment import ControlledTrepanConfig
    from core.scientific_experiment_contract import freeze_oracle
    from core.trepan_scientific_tuning import tune_scientific_trepan
    from core.mlp_factory import build_mlp_for_data
    ds, _ = make_synthetic("tune_eq", n_samples=170, n_features=6, n_classes=3, n_groups=2, seed=21)
    mlp = build_mlp_for_data(ds.X, ds.y, random_state=3).fit(ds.X, ds.y)
    oracle = freeze_oracle(mlp, ds.X, builder="t")
    base = ControlledTrepanConfig(max_nodes=7, max_depth=4, min_samples_leaf=2, min_sample=120, max_n=2, beam_width=2, max_features_per_node=6,
                                  max_queries=1500, random_state=5)
    search = ScientificTrepanSearchConfig(cv_folds=2, cv_repeats=2, purity_epsilon_grid=(0.05, 0.02), max_nodes_grid=(7, 15))
    fits = {"orig": 0, "reloaded": 0}
    from core.trepan_original import TrepanOriginalClassifier
    from core.trepan_reloaded_historical import TrepanReloadedClassifier
    o_fit, r_fit = TrepanOriginalClassifier.fit, TrepanReloadedClassifier.fit

    def count_o(self, *a, **k):
        fits["orig"] += 1
        return o_fit(self, *a, **k)

    def count_r(self, *a, **k):
        fits["reloaded"] += 1
        return r_fit(self, *a, **k)
    TrepanOriginalClassifier.fit, TrepanReloadedClassifier.fit = count_o, count_r
    try:
        out = tune_scientific_trepan(ds.X, ds.y, oracle=oracle, feature_names=ds.feature_names, base_config=base, search=search, **kw)
    finally:
        TrepanOriginalClassifier.fit, TrepanReloadedClassifier.fit = o_fit, r_fit
    return out, fits


def _strip_times(o):
    return eq._strip_time(o)


def test_tuning_memoization_and_skipped_semantic_stage_do_not_change_the_structural_result():
    ref, ref_fits = _tune(reuse_identical_fits=False, run_semantic_stage=True)          # comportamento anterior
    memo, memo_fits = _tune(reuse_identical_fits=True, run_semantic_stage=True)
    lean, lean_fits = _tune(reuse_identical_fits=True, run_semantic_stage=False)         # o que o protocolo estrutural usa
    for out in (memo, lean):
        for key in ("common_capacity", "structure_selected", "tuning_status", "initial_node_grid", "final_node_grid", "expansion_stop_reason",
                    "fraction_at_node_cap", "tuning_stable", "non_discriminative_grid", "budget_check"):
            assert _strip_times(out[key]) == _strip_times(ref[key]), key
        assert _strip_times(out["structure_selection"]) == _strip_times(ref["structure_selection"])
        assert _strip_times(out["capacity_selection"]) == _strip_times(ref["capacity_selection"])
        assert _strip_times(out["structure_history"]) == _strip_times(ref["structure_history"])
        assert _strip_times(out["capacity_history"]) == _strip_times(ref["capacity_history"])
    assert _strip_times(memo["semantic_history"]) == _strip_times(ref["semantic_history"]) and memo["selected_config"] == ref["selected_config"]
    assert memo_fits["orig"] < ref_fits["orig"]                       # o candidato base repetido na etapa de capacidade já não é reajustado
    assert lean_fits["reloaded"] == 0 < ref_fits["reloaded"]          # sem ajustes Reloaded inúteis
    assert lean_fits["orig"] < memo_fits["orig"]


def test_gaussian_kde_fast_path_equals_sklearn_sample_and_caches_are_not_pickled():
    import pickle
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 3)) * [1.0, 5.0, 0.2]
    model = T.FeatureDistributionModel(random_state=11).fit(X)
    ref = T.FeatureDistributionModel(random_state=11).fit(X)
    for j in range(3):
        for n in (1, 50, 500):
            seed = int(rng.integers(0, 2**31 - 1))
            a_rng = np.random.default_rng(seed); b_rng = np.random.default_rng(seed)
            model._rng, ref._rng = a_rng, b_rng
            fast = model._sample_column(j, n)
            with legacy_engine():
                slow = ref._sample_column(j, n)
            assert np.array_equal(fast, slow)
    assert hasattr(model, "_gaussian_cache") and "_gaussian_cache" not in pickle.loads(pickle.dumps(model)).__dict__
