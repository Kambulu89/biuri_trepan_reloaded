from pathlib import Path
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from core.classification_metrics import compute_classification_metrics
from core.ontology_acceptance import evaluate_ontological_oracle_acceptance
from core.surrogate_acceptance import evaluate_surrogate_acceptance_metrics
from core.feature_space_contract import (
    build_feature_space_manifest,
    validate_feature_matrix,
    assert_original_space_has_no_ontology,
    assert_ontological_space_enriches_original,
)
from core.counterfactual_contract import (
    validate_counterfactual_batch_metadata,
    validate_counterfactual_candidate,
    assert_counterfactual_context_compatible,
)
from core.evaluation_protocol import EvaluationProtocolGuard, PartitionRole
from core.onto_feature_selector import audit_ontology_features, filter_audited_ontology_features


def test_precision_is_real_precision_not_accuracy():
    y = np.array([0, 0, 0, 1])
    pred = np.array([0, 0, 0, 0])
    m = compute_classification_metrics(y, pred)
    assert m['accuracy'] == pytest.approx(0.75)
    assert m['precision_macro'] == pytest.approx(0.375)
    assert m['precision_macro'] != m['accuracy']
    assert m['metric_semantics']['precision_macro'].startswith('precision_score')


def test_metrics_include_confusion_and_per_class_counts():
    m = compute_classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert m['confusion_matrix'] == [[1, 1], [0, 2]]
    row0 = m['per_class'][0]
    assert row0['tp'] == 1 and row0['fp'] == 0 and row0['fn'] == 1 and row0['tn'] == 2


def test_oracle_gate_rejects_precision_macro_drop_even_if_accuracy_is_ok():
    base = dict(precision_macro=.80, recall_macro=.80, macro_f1=.80, balanced_accuracy=.80, accuracy=.80)
    onto = dict(precision_macro=.70, recall_macro=.81, macro_f1=.81, balanced_accuracy=.81, accuracy=.82)
    out = evaluate_ontological_oracle_acceptance(base, onto, tolerance=.01)
    assert out['accepted'] is False
    assert out['fallback_to_original'] is True
    assert out['active_oracle'] == 'MLP Original'
    assert 'precision_macro_noninferior' in out['failed_checks']


def test_oracle_gate_accepts_only_when_all_required_metrics_pass():
    base = dict(precision_macro=.80, recall_macro=.80, macro_f1=.80, balanced_accuracy=.80, accuracy=.80)
    onto = dict(precision_macro=.795, recall_macro=.80, macro_f1=.805, balanced_accuracy=.80, accuracy=.80)
    out = evaluate_ontological_oracle_acceptance(base, onto, tolerance=.01)
    assert out['accepted'] is True
    assert out['fallback_to_original'] is False


def test_feature_space_original_rejects_ontology_origin():
    manifest = build_feature_space_manifest(
        'original_space', ['age', 'onto_risk'], transformer_id='raw',
        origins={'age': 'original', 'onto_risk': 'ontology'},
    )
    with pytest.raises(ValueError):
        assert_original_space_has_no_ontology(manifest)


def test_ontological_space_must_preserve_original_order_and_enrich():
    original = build_feature_space_manifest('original_space', ['a', 'b'], transformer_id='raw', origins={'a':'original','b':'original'})
    good = build_feature_space_manifest('ontological_space', ['a', 'b', 'onto_c'], transformer_id='owl-v1', origins={'a':'original','b':'original','onto_c':'ontology'})
    assert_ontological_space_enriches_original(original, good)
    bad = build_feature_space_manifest('ontological_space', ['b', 'a', 'onto_c'], transformer_id='owl-v1', origins={'a':'original','b':'original','onto_c':'ontology'})
    with pytest.raises(ValueError):
        assert_ontological_space_enriches_original(original, bad)


def test_schema_order_mismatch_fails_explicitly():
    manifest = build_feature_space_manifest('original_space', ['a', 'b'], transformer_id='raw', origins={'a':'original','b':'original'})
    with pytest.raises(ValueError, match='alinhamento posicional'):
        validate_feature_matrix(np.zeros((3,2)), manifest, feature_names=['b','a'])


def test_ontology_feature_audit_rejects_constant_duplicate_and_no_provenance():
    X = np.array([
        [0, 1, 5, 5, 0.1],
        [1, 2, 5, 5, 0.2],
        [2, 3, 5, 5, 0.4],
        [3, 4, 5, 5, 0.8],
    ], dtype=float)
    names = ['a','b','onto_const','onto_dup','onto_good']
    audit = audit_ontology_features(
        X, np.array([0,0,1,1]), names, ['a','b'],
        provenance={'onto_const': {'owl_concept':'C'}, 'onto_dup': {'owl_concept':'D'}, 'onto_good': {'owl_concept':'G'}},
    )
    accepted = filter_audited_ontology_features(audit)
    assert accepted == ['onto_good']


def test_counterfactual_batch_requires_teacher_space_and_hashes():
    with pytest.raises(ValueError):
        validate_counterfactual_batch_metadata({'teacher_id':'t1'})
    validate_counterfactual_batch_metadata({
        'teacher_id':'t1','teacher_label':'MLP Original','feature_space':'original_space',
        'schema_hash':'s','ontology_hash':'none','dataset_hash':'d','seed':42,'method':'CLEAR',
    })


def test_counterfactual_must_change_intended_teacher_prediction():
    X = np.array([[0.0],[1.0],[2.0],[3.0]])
    y = np.array([0,0,1,1])
    teacher = LogisticRegression().fit(X,y)
    valid = validate_counterfactual_candidate(teacher, [0.0], [3.0], expected_n_features=1, minimum_confidence=.5)
    assert valid['accepted'] is True
    invalid = validate_counterfactual_candidate(teacher, [0.0], [0.2], expected_n_features=1, minimum_confidence=.5)
    assert invalid['accepted'] is False
    assert invalid['checks']['changes_teacher_prediction'] is False


def test_counterfactual_cannot_be_reused_across_teacher_or_space():
    a = {'teacher_id':'orig','feature_space':'original_space','schema_hash':'s1','ontology_hash':'none','dataset_hash':'d'}
    b = {'teacher_id':'onto','feature_space':'ontological_space','schema_hash':'s2','ontology_hash':'o1','dataset_hash':'d'}
    with pytest.raises(ValueError, match='reutilização'):
        assert_counterfactual_context_compatible(a,b)


def test_protocol_blocks_test_for_selection_and_final_test_once():
    guard = EvaluationProtocolGuard('x')
    guard.record_selection(PartitionRole.TRAIN, 'feature_selection')
    guard.record_selection(PartitionRole.VALIDATION, 'oracle_selection')
    with pytest.raises(RuntimeError, match='Vazamento bloqueado'):
        guard.record_selection(PartitionRole.TEST, 'tree_selection')
    guard.record_final_evaluation(PartitionRole.TEST, 'final_report')
    with pytest.raises(RuntimeError):
        guard.record_final_evaluation(PartitionRole.TEST, 'again')


def test_surrogate_gate_rejects_precision_macro_degradation():
    original = dict(precision_macro=.85, recall_macro=.84, macro_f1=.84, balanced_accuracy=.84, accuracy=.85)
    reloaded = dict(precision_macro=.70, recall_macro=.90, macro_f1=.85, balanced_accuracy=.85, accuracy=.86)
    out = evaluate_surrogate_acceptance_metrics(
        reloaded, original, fidelity_to_active_oracle=.99,
        candidate_complexity=10, baseline_complexity=10, tolerance=.01,
    )
    assert out['accepted'] is False
    assert 'precision_macro_noninferior' in out['failed_checks']


def test_gui_comparison_is_precision_macro_not_accuracy():
    text = Path('gui/pyqt_metrics_visualizer.py').read_text(encoding='utf-8')
    assert "Precisão Macro / Precision Macro" in text
    assert "precision_values.append(self.models_data[key]['precision'])" in text
    # O gráfico principal já não deve usar accuracy como valor primário.
    assert "primary_accuracy = self.models_data[key].get('accuracy')" not in text
