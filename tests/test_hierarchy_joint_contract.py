"""Contrato conjunto da hierarquia (accuracy E fidelidade; Original e C4.5)."""
from core.hierarchy_contract import INFEASIBLE, INSUFFICIENT, NOT_SATISFIED, SATISFIED, evaluate_joint, is_scientific_success


def ev(aR, aO, aC, fR, fO, fC, **k):
    return evaluate_joint(aR, aO, aC, fR, fO, fC, **k)


def test_joint_gain_is_the_only_success():
    v = ev(0.95, 0.92, 0.90, 0.96, 0.93, 0.91)
    assert v["status"] == SATISFIED and is_scientific_success(v)


def test_accuracy_up_fidelity_down_is_tradeoff_not_success():
    v = ev(0.95, 0.92, 0.90, 0.90, 0.93, 0.85)
    assert v["status"] == NOT_SATISFIED and v["mode"] == "TRADEOFF_ACCURACY_UP_FIDELITY_DOWN" and not is_scientific_success(v)


def test_fidelity_up_accuracy_down_is_tradeoff_not_success():
    v = ev(0.90, 0.92, 0.88, 0.97, 0.93, 0.85)
    assert v["status"] == NOT_SATISFIED and v["mode"] == "TRADEOFF_FIDELITY_UP_ACCURACY_DOWN" and not is_scientific_success(v)


def test_must_also_beat_c45_in_accuracy_and_fidelity():
    v = ev(0.95, 0.94, 0.93, 0.96, 0.93, 0.97)                        # fidelidade do C4.5 acima da do Reloaded
    assert v["status"] == NOT_SATISFIED and not v["conditions"]["U5_fid_R_gt_C"] and v["conditions"]["U1_acc_R_gt_O"]
    v = ev(0.935, 0.94, 0.93, 0.96, 0.93, 0.91)                       # Reloaded acima do C4.5 mas abaixo do Original em accuracy
    assert v["status"] == NOT_SATISFIED and not v["conditions"]["U1_acc_R_gt_O"] and v["conditions"]["U3_acc_R_gt_C"]


def test_original_below_c45_is_infeasible_and_baselines_are_not_touched():
    v = ev(0.95, 0.90, 0.92, 0.96, 0.93, 0.91)
    assert v["status"] == INFEASIBLE and not is_scientific_success(v)


def test_ties_do_not_count_and_missing_data_is_insufficient():
    assert ev(0.9, 0.9, 0.8, 0.9, 0.9, 0.8)["status"] == NOT_SATISFIED
    assert ev(0.9, None, 0.8, 0.9, 0.9, 0.8)["status"] == INSUFFICIENT
    assert ev(float("nan"), 0.9, 0.8, 0.9, 0.9, 0.8)["status"] == INSUFFICIENT


def test_margins_are_strict():
    assert ev(0.925, 0.92, 0.90, 0.94, 0.93, 0.91, delta_acc=0.01)["status"] == NOT_SATISFIED
    assert ev(0.94, 0.92, 0.90, 0.95, 0.93, 0.91, delta_acc=0.01)["status"] == SATISFIED
