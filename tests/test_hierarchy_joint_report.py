"""Relatório em dois níveis: comparabilidade obrigatória e nenhuma unidade excluída (Opção 1)."""
from validation.benchmark.hierarchy_joint_report import report


def row(acc, fid, **k):
    return {"accuracy_real_labels": acc, "fidelity_to_oracle": fid, "precision_macro_real_labels": acc, "macro_f1_real_labels": acc,
            "split_hash": k.get("split", "s"), "n_test": 30, "oracle_id": k.get("oracle", "m"), "fidelity_oracle_id": k.get("foracle")}


def unit(aR, aO, aC, fR, fO, fC, **k):
    return {"trepan_original": row(aO, fO, **k), "reloaded_owl_full": row(aR, fR, **k), "c45": row(aC, fC, split=k.get("split", "s"), oracle=None, foracle=k.get("oracle", "m")),
            "reloaded_owl_shuffled": row(aR - 0.01, fR - 0.01, **k)}


def test_infeasible_units_are_kept_and_counted_in_the_denominator():
    units = {"a:1": unit(0.95, 0.92, 0.90, 0.96, 0.93, 0.91),            # JOINT_HIERARCHY_SATISFIED
             "a:2": unit(0.95, 0.88, 0.90, 0.96, 0.93, 0.91),            # Original < C4.5 ⇒ INFEASIBLE (mantida)
             "b:1": unit(0.90, 0.92, 0.88, 0.95, 0.93, 0.85)}            # compromisso
    r = report(units, {}, "historical")
    assert r["n_units"] == 3 and r["exclusions"] == "nenhuma" and r["comparability"]["all_comparable"]
    c = r["level2_global_hierarchy"]["state_counts"]
    assert c["JOINT_HIERARCHY_SATISFIED"]["n"] == 1 and c["HIERARCHY_INFEASIBLE"]["n"] == 1 and c["TRADEOFF"]["n"] == 1
    assert c["HIERARCHY_INFEASIBLE"]["pct_of_all_units"] == 33.3                       # percentagem sobre TODAS as unidades
    assert r["level1_direct_ontological_gain"]["controls_missing"] == ["random_groups", "conventional_features"]
    assert r["questions"]["Q2_caused_by_ontological_knowledge"] == "NOT_ASSESSABLE"      # grupos aleatórios/convencional em falta


def test_different_split_or_oracle_blocks_any_verdict():
    u = unit(0.95, 0.92, 0.90, 0.96, 0.93, 0.91)
    u["reloaded_owl_full"] = row(0.95, 0.96, split="OTHER")
    assert not report({"a:1": u}, {}, "historical")["comparability"]["all_comparable"]
    u2 = unit(0.95, 0.92, 0.90, 0.96, 0.93, 0.91, oracle="m")
    u2["c45"] = row(0.90, 0.91, oracle=None, foracle="OTHER_MLP")
    r = report({"a:1": u2}, {}, "historical")
    assert not r["comparability"]["all_comparable"] and r["verdict"].startswith("INSUFFICIENT_EVIDENCE")
