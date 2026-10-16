"""B3.3: o analisador pré-registado, com resultados sintéticos CONSTRUÍDOS, devolve os veredictos esperados e não depende de nenhum dataset."""
import numpy as np

from validation.benchmark import b33_report as R
from validation.benchmark.b33_interactions import ARMS33


def _rows(spec):
    rows = []
    for s in range(30):
        for a in ARMS33:
            rec, acc, fid, cost = spec.get(a, (0.0, 0.6, 0.6, 1.0))
            j = 0.001 * ((s * 7) % 5)
            rows.append({"arm": a, "seed": s, "recovered_path": float(rec), "pair_same_test": 0.0, "accuracy": acc + j, "fidelity": fid + j, "precision_macro": acc, "macro_f1": acc, "queries": 10,
                         "eval_calls": 100, "literals_listed": 50, "simple_generated": 5, "mofn_generated": 5, "oracle_samples": 10, "internal_nodes": 3, "depth": 2, "mofn_nodes": 1,
                         "mean_rule_literals": 2.0, "cpu_fit_s": cost, "compile_wall_s": 0.0, "screen_wall_s": 0.0, "total_cost_cpu_s": cost, "n_units": 8, "unit_searches": 3})
    refs = [{"accuracy": 0.55, "fidelity": 0.55, "mlp_accuracy": 0.8} for _ in range(30)]
    return {"family": "I3_X", "rows": rows, "refs": refs}


def test_ontological_advantage_when_C_dominates_everything():
    spec = {a: (0.0, 0.6, 0.6, 1.0) for a in ARMS33}
    spec["A_ORIG"] = (0.0, 0.6, 0.6, 1.0)
    spec["C_OWL"] = (1.0, 0.95, 0.9, 1.0)
    out = R.analyse_family(_rows(spec))
    assert out["verdict"] == "ONTOLOGICAL_ADVANTAGE" and out["ontology_attribution"] == "ATTRIBUTED" and out["joint_gain_status"] == "SUPPORTED_JOINT_GAIN"


def test_wide_equals_owl_is_not_advantage_and_cheaper_equivalent_flagged():
    spec = {a: (0.0, 0.6, 0.6, 1.0) for a in ARMS33}
    spec["C_OWL"] = (1.0, 0.95, 0.9, 5.0)
    spec["B_WIDE"] = (1.0, 0.95, 0.9, 1.0)
    out = R.analyse_family(_rows(spec))
    assert out["verdict"] != "ONTOLOGICAL_ADVANTAGE" and out["SIMPLER_EQUIVALENT"]["B_WIDE"] and not out["R2"]


def test_generic_pair_screen_equals_owl_blocks_attribution():
    spec = {a: (0.0, 0.6, 0.6, 1.0) for a in ARMS33}
    spec["C_OWL"] = (1.0, 0.95, 0.9, 2.0)
    spec["H_PAIR_SCREEN"] = (1.0, 0.95, 0.9, 1.0)
    out = R.analyse_family(_rows(spec))
    assert out["R1"] is False and out["verdict"] == "NO_ONTOLOGICAL_ADVANTAGE" and out["ontology_attribution"] == "NOT_ATTRIBUTED"


def test_uninformative_when_nobody_recovers_and_tradeoff_state():
    spec = {a: (0.0, 0.6, 0.6, 1.0) for a in ARMS33}
    out = R.analyse_family(_rows(spec))
    assert out["verdict"] == "UNINFORMATIVE_NO_ARM_RECOVERS" and out["joint_gain_status"] == "NO_JOINT_GAIN"
    spec["C_OWL"] = (0.0, 0.7, 0.5, 1.0)
    assert R.analyse_family(_rows(spec))["joint_gain_status"] == "TRADEOFF"


def test_global_verdict_counts_only_I3_to_I6():
    f = {"I1_a": {"verdict": "ONTOLOGICAL_ADVANTAGE"}, "I3_a": {"verdict": "ONTOLOGICAL_ADVANTAGE"}, "I4_a": {"verdict": "NO_ONTOLOGICAL_ADVANTAGE"}}
    assert R.global_verdict(f) == "B33_PARTIAL"
    f["I5_a"] = {"verdict": "ONTOLOGICAL_ADVANTAGE"}
    assert R.global_verdict(f) == "B33_ADVANTAGE_ESTABLISHED"
