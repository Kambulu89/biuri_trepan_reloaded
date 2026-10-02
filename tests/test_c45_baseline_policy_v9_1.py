import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score

from core.c45_baseline_gate import (
    evaluate_c45_baseline_gate,
    evaluate_c45_baseline_pair,
)
from core.surrogate_acceptance import evaluate_surrogate_acceptance_metrics
from core.trepan_reloaded_extractor import TrepanReloadedExtractor
from core.metrics_view_model import build_metrics_rows


def _m(precision, recall=None, f1=None, bal=None, acc=None):
    recall = precision if recall is None else recall
    f1 = precision if f1 is None else f1
    bal = precision if bal is None else bal
    acc = precision if acc is None else acc
    return {
        "precision_macro": precision,
        "recall_macro": recall,
        "macro_f1": f1,
        "balanced_accuracy": bal,
        "accuracy": acc,
    }


def test_c45_baseline_gate_reports_failure_without_changing_metrics():
    trepan = _m(0.895)
    c45 = _m(0.923)
    out = evaluate_c45_baseline_gate(
        trepan, c45, model_label="TREPAN Reloaded", tolerance=0.0
    )
    assert out["accepted"] is False
    assert out["c45_is_oracle"] is False
    assert abs(out["deltas"]["precision_macro"] - (-0.028)) < 1e-12
    assert trepan["precision_macro"] == 0.895
    assert c45["precision_macro"] == 0.923


def test_pair_gate_checks_original_and_reloaded_against_same_c45():
    out = evaluate_c45_baseline_pair(_m(0.863), _m(0.895), _m(0.923))
    assert out["available"] is True
    assert out["trepan_original"]["accepted"] is False
    assert out["trepan_reloaded"]["accepted"] is False
    assert out["all_trepan_reach_c45_baseline"] is False


def test_surrogate_gate_can_make_c45_noninferiority_mandatory_but_never_oracle():
    out = evaluate_surrogate_acceptance_metrics(
        _m(0.895), _m(0.863),
        fidelity_to_active_oracle=0.96,
        candidate_complexity=8,
        baseline_complexity=8,
        tolerance=0.0,
        c45_metrics=_m(0.923),
        require_c45_noninferiority=True,
    )
    assert out["accepted"] is False
    assert out["c45_is_oracle"] is False
    assert out["c45_baseline_gate_accepted"] is False
    assert "c45_precision_macro_noninferior" in out["failed_checks"]


def test_extractor_primary_precision_is_macro_not_weighted():
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y_true = np.array([0, 0, 0, 1])
    y_constant = np.zeros(4, dtype=int)
    tree = DecisionTreeClassifier(random_state=1).fit(X, y_constant)
    oracle = DecisionTreeClassifier(random_state=2).fit(X, y_constant)
    extractor = TrepanReloadedExtractor()
    metrics = extractor._measure_tree_performance(tree, oracle, X, y_true)
    expected_macro = precision_score(y_true, y_constant, average="macro", zero_division=0)
    expected_weighted = precision_score(y_true, y_constant, average="weighted", zero_division=0)
    assert abs(metrics["precision_macro"] - expected_macro) < 1e-12
    assert abs(metrics["precision"] - expected_macro) < 1e-12
    assert abs(metrics["precision_weighted"] - expected_weighted) < 1e-12
    assert metrics["precision_macro"] != metrics["precision_weighted"]


def test_dominance_does_not_use_c45_fidelity_as_if_c45_were_oracle():
    extractor = TrepanReloadedExtractor()
    reloaded = {"fidelity": 0.95, "precision_macro": 0.93}
    original = {"fidelity": 0.90, "precision_macro": 0.88}
    # C4.5 pode ter concordância auxiliar maior; isso não é critério de oráculo.
    c45 = {"fidelity": 0.99, "precision_macro": 0.92}
    assert extractor._tree_dominates(reloaded, original, c45, margin=0.0) is True


def test_metrics_view_model_exposes_macro_precision_explicitly():
    results = {
        "precision": {
            "trepan_original": {
                "accuracy": 0.9,
                "balanced_accuracy": 0.88,
                "f1_macro": 0.87,
                "precision": 0.91,
                "precision_macro": 0.84,
            }
        },
        "fidelity": {"trepan_original": {"overall_fidelity": 0.93}},
    }
    row = build_metrics_rows(results)[0]
    assert row["precision_macro"] == 0.84
    assert row["precision_weighted"] == 0.91
