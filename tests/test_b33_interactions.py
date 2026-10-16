"""B3.3: testes de FUNCIONAMENTO do gerador, do rastreio por pares e das métricas (antes de qualquer calibração/avaliação)."""
import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from core.trepan_original import TrepanOriginalClassifier
from validation.benchmark import b33_interactions as B

FAM = B.FAMILIES["I3_XOR_WEAK_SIGNAL"]


def _uni_gain(x, y, t):
    def H(v):
        p = np.bincount(v, minlength=2) / max(len(v), 1)
        return -sum(q * np.log2(q) for q in p if q > 0)
    m = x > t
    return H(y) - (m.mean() * H(y[m]) + (1 - m.mean()) * H(y[~m])) if 0 < m.sum() < len(m) else 0.0


def test_instance_deterministic_and_rule_labels():
    sp = B.spec_of(FAM, 16)
    a, b = B.make_instance33(FAM, sp, 7), B.make_instance33(FAM, sp, 7)
    assert np.array_equal(a["X"], b["X"]) and a["rule_names"] == b["rule_names"] and np.array_equal(a["y_train"], b["y_train"])
    i, j = a["rule_idx"]
    t = a["rule_thr"]
    xr = (a["X"][:, i] > t[0]) ^ (a["X"][:, j] > t[1])
    assert np.array_equal(xr.astype(int), a["y_train"])                    # eta = 0: rótulos reais = regra


def test_no_signal_family_has_no_univariate_signal_and_and_family_has():
    g = {}
    for k in ("I1_AND_PAIR", "I2_XOR_NO_SIGNAL"):
        fam = B.FAMILIES[k]
        gains = []
        for s in range(5):
            inst = B.make_instance33(fam, B.spec_of(fam, 16), s)
            gains.append(max(_uni_gain(inst["X"][:, c], inst["y_train"], inst["rule_thr"][n]) for n, c in enumerate(inst["rule_idx"])))
        g[k] = float(np.mean(gains))
    assert g["I2_XOR_NO_SIGNAL"] < 0.02 < g["I1_AND_PAIR"]


def test_label_noise_applies_to_train_and_test():
    fam = B.FAMILIES["I4_XOR_WEAK_NOISY"]
    inst = B.make_instance33(fam, B.spec_of(fam, 16), 3)
    for X, y in ((inst["X"], inst["y_train"]), (inst["Xtest"], inst["y_test"])):
        rate = float(np.mean(inst["rule_fn"](X) != y))
        assert 0.05 < rate < 0.16


def test_pair_screening_finds_interaction_without_univariate_signal():
    fam = B.FAMILIES["I2_XOR_NO_SIGNAL"]
    hits = 0
    for s in range(6):
        inst = B.make_instance33(fam, B.spec_of(fam, 16), s)
        pred = inst["rule_fn"](inst["X"])                                  # rótulos da própria regra apenas para testar o rastreio em si
        units = B.pair_screening_units(inst["X"], inst["names"], pred, 4, 8)
        assert len({f for u in units for f in u.features}) == sum(len(u.features) for u in units)   # unidades disjuntas
        hits += any(set(inst["rule_names"]) <= set(u.features) for u in units)
    assert hits >= 4


def test_path_recovery_metric():
    class L:
        def __init__(s, f): s.feature = f
    class T:
        def __init__(s, fs): s.literals = [L(f) for f in fs]
    class N:
        def __init__(s, fs=None, tc=None, fc=None): s.test, s.true_child, s.false_child, s.is_leaf = (T(fs) if fs else None), tc, fc, fs is None
    class M:
        pass
    leaf = N()
    m = M(); m.root_ = N([1], N([2], leaf, leaf), leaf)
    assert B._path_recovery(m, [1, 2]) == (False, True)
    m.root_ = N([1], leaf, N([2], leaf, leaf))
    assert B._path_recovery(m, [1, 2]) == (False, True)
    m.root_ = N([3], N([1], leaf, leaf), N([2], leaf, leaf))                   # em ramos diferentes: não conta
    assert B._path_recovery(m, [1, 2]) == (False, False)
    m.root_ = N([1, 2], leaf, leaf)
    assert B._path_recovery(m, [1, 2]) == (True, True)


def test_original_arm_equals_canonical_original_and_has_no_ontology():
    fam = B.FAMILIES["I3_XOR_WEAK_SIGNAL"]
    sp = B.spec_of(fam, 16)
    inst = B.make_instance33(fam, sp, 11)
    with tempfile.TemporaryDirectory() as td:
        r = B.fit_arm33(fam, sp, inst, "A_ORIG", 11, Path(td))
    kw = dict(min_sample=sp.min_sample, max_queries=sp.max_queries, max_nodes=sp.max_nodes, max_depth=sp.max_depth, max_n=sp.max_n, beam_width=sp.beam_width,
              min_samples_leaf=sp.min_samples_leaf, max_features_per_node=sp.max_features_per_node, random_state=11)
    ref = TrepanOriginalClassifier(**kw).fit(inst["X"], oracle=inst["mlp"], feature_names=inst["names"])
    assert r["rules_sha"] == hashlib.sha256(json.dumps(ref.export_rules(), default=str).encode()).hexdigest()[:16]
    assert r["ontology_status"] is None and r["compile_wall_s"] == 0.0 and r["n_units"] == 0


def test_metric_patch_is_restored_and_arm_set_complete():
    from validation.benchmark import b32_comparison as B32
    before = (B32._metrics, B32.screening_units)
    fam = FAM; sp = B.spec_of(fam, 16); inst = B.make_instance33(fam, sp, 12)
    with tempfile.TemporaryDirectory() as td:
        r = B.fit_arm33(fam, sp, inst, "H_PAIR_SCREEN", 12, Path(td))
    assert (B32._metrics, B32.screening_units) == before
    assert r["screen_wall_s"] > 0 and r["screen_candidates"] > 0 and r["n_units"] == sp.n_groups
    for a in "A B C D E F G H I".split():
        assert any(x.startswith(a + "_") for x in B.ARMS33)


def test_ontology_controls_use_same_instance_and_data():
    fam = B.FAMILIES["I5_XOR_WEAK_HIERARCHY"]
    sp = B.spec_of(fam, 16)
    inst = B.make_instance33(fam, sp, 13)
    assert len(inst["rule_groups"]) == 2 and inst["rule_idx"][0] != inst["rule_idx"][1]
    with tempfile.TemporaryDirectory() as td:
        rs = {a: B.fit_arm33(fam, sp, inst, a, 13, Path(td)) for a in ("C_OWL", "D_PERM", "I_NEUTRALIZED")}
    assert all(r["ontology_status"] == "VALID" for r in rs.values())
    assert rs["C_OWL"]["n_units"] >= 1
