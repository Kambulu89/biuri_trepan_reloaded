from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from core.mlp_convergence import extract_mlp_convergence
from core.ontology_acceptance import evaluate_selected_teacher_candidate
from core.protocol_audit import build_protocol_audit
from core.surrogate_acceptance import evaluate_surrogate_acceptance_metrics
from core.metrics_view_model import claim_banner


def _metrics(acc=0.90, macro=0.90, bal=0.90):
    return {"accuracy": acc, "macro_f1": macro, "balanced_accuracy": bal}


def test_teacher_gate_is_conjunctive_not_utility_compensated():
    original = _metrics()
    # Accuracy/F1 sobem, mas balanced accuracy cai mais que a margem: rejeitar.
    candidate = _metrics(acc=0.95, macro=0.95, bal=0.87)
    result = evaluate_selected_teacher_candidate(
        original_metrics=original,
        selected_metrics=candidate,
        hybrid_weights={"original": 0.2, "ontological": 0.8, "residual": 0.0},
        tolerance=0.01,
        ontology_feature_gate_accepted=True,
    )
    assert result["accepted"] is False
    assert result["oracle_gate_accepted"] is False
    assert result["balanced_accepted"] is False
    assert result["reason"] == "STRICT_NONINFERIORITY_FAILED"


def test_teacher_gate_rejects_zero_ontology_mass_and_marks_accepted_false():
    result = evaluate_selected_teacher_candidate(
        original_metrics=_metrics(),
        selected_metrics=_metrics(acc=0.91, macro=0.91, bal=0.91),
        hybrid_weights={"original": 1.0, "ontological": 0.0, "residual": 0.0},
        tolerance=0.01,
        ontology_feature_gate_accepted=True,
    )
    assert result["accepted"] is False
    assert result["candidate_teacher_has_ontology"] is False
    assert result["ontology_mass"] == 0.0
    assert result["reason"] == "ZERO_ONTOLOGY_MASS_IN_SELECTED_HYBRID"
    assert result["ontology_accepted_label"] == "Não"


def test_teacher_gate_requires_feature_gate_and_passes_only_all_conditions():
    rejected = evaluate_selected_teacher_candidate(
        original_metrics=_metrics(),
        selected_metrics=_metrics(acc=0.905, macro=0.905, bal=0.905),
        hybrid_weights={"original": 0.5, "ontological": 0.5, "residual": 0.0},
        tolerance=0.01,
        ontology_feature_gate_accepted=False,
    )
    assert rejected["accepted"] is False
    assert rejected["reason"] == "ONTOLOGY_FEATURE_GATE_REJECTED"

    accepted = evaluate_selected_teacher_candidate(
        original_metrics=_metrics(),
        selected_metrics=_metrics(acc=0.895, macro=0.895, bal=0.895),
        hybrid_weights={"original": 0.5, "ontological": 0.5, "residual": 0.0},
        tolerance=0.01,
        ontology_feature_gate_accepted=True,
    )
    assert accepted["accepted"] is True
    assert accepted["reason"] == "STRICT_NONINFERIORITY_PASSED"


def test_surrogate_gate_fallback_is_only_decided_on_development_holdout():
    failed = evaluate_surrogate_acceptance_metrics(
        reloaded_metrics=_metrics(acc=0.90, macro=0.84, bal=0.90),
        original_metrics=_metrics(acc=0.90, macro=0.90, bal=0.90),
        fidelity_to_active_oracle=0.96,
        candidate_complexity=8,
        baseline_complexity=7,
        tolerance=0.01,
        fidelity_floor=0.90,
        oracle_gate_accepted=True,
        selection_role="development_acceptance_holdout",
    )
    assert failed["surrogate_gate_accepted"] is False
    assert failed["fallback_required"] is True
    assert failed["test_used_for_selection"] is False
    assert failed["development_holdout_used_for_selection"] is True
    assert "macro_f1_noninferior" in failed["failed_checks"]

    locked_test = evaluate_surrogate_acceptance_metrics(
        reloaded_metrics=_metrics(acc=0.80, macro=0.80, bal=0.80),
        original_metrics=_metrics(),
        fidelity_to_active_oracle=0.99,
        candidate_complexity=8,
        baseline_complexity=7,
        oracle_gate_accepted=True,
        selection_role="locked_test_audit_only",
    )
    assert locked_test["surrogate_gate_accepted"] is False
    assert locked_test["fallback_required"] is False
    assert locked_test["test_used_for_selection"] is False
    assert locked_test["development_holdout_used_for_selection"] is False


def test_protocol_hashes_are_deterministic_and_same_rows_are_enforced():
    X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    Xr = np.c_[X, [0.1, 0.2, 0.3]]
    y = np.array([0, 1, 0])
    kwargs = dict(
        X_test_reloaded=Xr,
        feature_names=["a", "b"],
        feature_names_reloaded=["a", "b", "onto_c"],
        class_order_original=[0, 1],
        class_order_active_oracle=[0, 1],
        test_row_ids=["r1", "r2", "r3"],
        seed=42,
        repeat_count=3,
        preprocessing_id="fit_train_transform_test",
    )
    first = build_protocol_audit(X, y, **kwargs)
    second = build_protocol_audit(X.copy(), y.copy(), **kwargs)
    assert first == second
    assert first["role"] == "locked_test_reporting_only"
    assert first["test_used_for_selection"] is False
    assert len(first["logical_test_rows_sha256"]) == 64
    assert first["missing_count_original"] == 1

    with pytest.raises(ValueError, match="mesmas linhas"):
        build_protocol_audit(X, y, X_test_reloaded=Xr[:2])


def test_mlp_convergence_audit_records_required_fields_and_cap_reached():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1, random_state=13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    audit = extract_mlp_convergence(model)
    for key in ("n_iter", "max_iter", "loss_curve", "converged", "random_state"):
        assert key in audit
    assert audit["n_iter"] == 1
    assert audit["max_iter"] == 1
    assert audit["converged"] is False
    assert audit["status"] == "iteration_cap_reached_or_not_proven_converged"
    assert audit["random_state"] == 13


def test_claim_banner_obeys_claim_guard_not_point_estimate_boolean():
    inconsistent = {
        "scientific_validation": {
            "strong_superiority_claim_supported": True,
            "claim_guard": "do_not_claim_superiority_from_point_estimates",
        }
    }
    assert claim_banner(inconsistent)["status"] == "not_supported"
    supported = {
        "scientific_validation": {"claim_guard": "superiority_supported"}
    }
    assert claim_banner(supported)["status"] == "supported"


def test_optional_owl_dependency_fails_with_clear_message(monkeypatch, tmp_path):
    import core.trepan_reloaded_extractor as module

    monkeypatch.setattr(module, "get_ontology", None)
    fake = tmp_path / "fake.owl"
    fake.write_text("<rdf></rdf>", encoding="utf-8")
    with pytest.raises(ImportError, match="owlready2"):
        module.TrepanReloadedExtractor.load_ontology_file(fake)


def test_packaging_contract_files_exist_and_define_python_311():
    root = Path(__file__).resolve().parents[1]
    required = [
        "pyproject.toml",
        "pytest.ini",
        "requirements-runtime.txt",
        "requirements-gui.txt",
        "requirements-test.txt",
        ".github/workflows/windows-ci.yml",
        "scripts/test_clean_windows.ps1",
    ]
    for rel in required:
        assert (root / rel).exists(), rel
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.13"' in pyproject
    workflow = (root / ".github/workflows/windows-ci.yml").read_text(encoding="utf-8")
    assert 'python-version: ["3.11", "3.12"]' in workflow
    assert "python -m pytest -q" in workflow


def test_semantic_feature_audit_merges_owl_provenance_with_oof_evidence():
    from core.ontology_processor import OntologyProcessor

    processor = OntologyProcessor(None)
    processor.feature_audit_ = [{
        "feature": "onto_ratio",
        "kind": "constraint",
        "owl_origin": {"source_columns": ["x"], "owl_entities_or_properties": ["Risk"], "provenance": "owl"},
        "derivation_rule": "x >= 2",
        "provenance": "owl",
        "accepted": True,
        "reason_not_duplicated": "distinct_from_original_and_prior_derived_features",
        "rejection_reason": None,
        "duplicate_of": None,
        "constant_rate": 0.6,
        "effect_on_metrics": None,
        "fold_stability": None,
    }]
    processor.last_engineering_stats = {"feature_audit": list(processor.feature_audit_)}
    merged = processor.merge_semantic_validation_audit({
        "status": "ACCEPT_PARTIAL_FEATURES",
        "scope": "development_oof_only",
        "feature_audit": [{
            "feature": "onto_ratio",
            "accepted_by_semantic_gate": True,
            "rejection_reason": None,
            "fold_stability": 0.8,
            "mean_mutual_information": 0.12,
            "effect_on_metrics": {"macro_f1_delta": 0.02},
        }],
    })
    row = merged[0]
    assert row["owl_origin"]["owl_entities_or_properties"] == ["Risk"]
    assert row["derivation_rule"] == "x >= 2"
    assert row["fold_stability"] == 0.8
    assert row["effect_on_metrics"]["macro_f1_delta"] == 0.02
    assert row["semantic_gate_accepted"] is True


def test_surrogate_gate_can_require_noninferiority_to_c45_without_using_c45_as_oracle():
    result = evaluate_surrogate_acceptance_metrics(
        reloaded_metrics=_metrics(acc=0.89, macro=0.89, bal=0.89),
        original_metrics=_metrics(acc=0.88, macro=0.88, bal=0.88),
        c45_metrics=_metrics(acc=0.92, macro=0.92, bal=0.92),
        require_c45_noninferiority=True,
        fidelity_to_active_oracle=0.96,
        candidate_complexity=8,
        baseline_complexity=8,
        tolerance=0.01,
        oracle_gate_accepted=True,
    )
    assert result["surrogate_gate_accepted"] is False
    assert result["require_c45_noninferiority"] is True
    assert "c45_accuracy_noninferior" in result["failed_checks"]


def test_onto_feature_bias_is_documented_as_internal_heuristic_not_formal_ig():
    from core.trepan_reloaded_extractor import TrepanReloadedExtractor

    extractor = TrepanReloadedExtractor(onto_feature_bias_weight=2.0)
    extractor.set_onto_feature_bias_weight(1.5)
    audit = extractor._onto_bias_calibration_audit
    assert audit["test_used"] is False
    assert audit["formal_weighted_information_gain"] is False
    assert audit["method"] == "heuristic_sample_and_feature_priority_weight"


def test_locked_confirmatory_launcher_and_windows_preflight_are_packaged():
    root = Path(__file__).resolve().parents[1]
    preflight = root / "scripts" / "environment_preflight.py"
    launcher = root / "scripts" / "run_confirmatory_locked.py"
    workflow = (root / ".github" / "workflows" / "windows-ci.yml").read_text(encoding="utf-8")
    assert preflight.exists()
    assert launcher.exists()
    launcher_text = launcher.read_text(encoding="utf-8")
    assert "--execute" in launcher_text
    assert "build_preflight" in launcher_text
    assert "BLOQUEADO" in launcher_text
    assert "execução única já foi consumida" in launcher_text
    assert "environment_preflight.py --gui" in workflow
