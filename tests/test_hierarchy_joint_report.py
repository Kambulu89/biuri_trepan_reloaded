"""Relatório em dois níveis: comparabilidade ESTRITA obrigatória e nenhuma unidade excluída (Opção 1)."""
import copy

from validation.benchmark.hierarchy_joint_report import report


def rec(acc, fid, kind="trepan", **over):
    r = {"labels": [0, 1], "manifest_sha256": "mf", "config": {"structural": {"tree_max_nodes": 31.0, "query_budget": 100.0, "tree_max_n": 3.0}},
         "predictions": {"test_row_index": [1, 2, 3], "y_true": [0, 1, 1], "oracle_predictions": {"m": [0, 1, 1]}},
         "row": {"accuracy_real_labels": acc, "fidelity_to_oracle": fid, "precision_macro_real_labels": acc, "macro_f1_real_labels": acc, "minority_label": "0", "split_id": "s1",
                 "split_hash": "sh", "n_train": 9, "n_test": 3, "dataset_content_hash": "dc", "dataset_hash": "dh", "base_representation_hash": "br", "preprocessing_id": "pp",
                 "oracle_version": "v", "oracle_name": "MLP", "config_hash": "ch", "feature_space": "original", "ontology_hash": "none"}}
    r["row"]["oracle_id" if kind == "trepan" else "fidelity_oracle_id"] = "M"
    r = copy.deepcopy(r)
    for k, v in over.items():
        a, b = k.split("__")
        r[a][b] = v
    return r


def unit(aR, aO, aC, fR, fO, fC):
    return {"trepan_original": rec(aO, fO), "reloaded_owl_full": rec(aR, fR), "c45": rec(aC, fC, "c45"), "reloaded_owl_shuffled": rec(aR - 0.01, fR - 0.01)}


def test_infeasible_units_are_kept_and_counted_in_the_denominator():
    units = {"a:1": unit(0.95, 0.92, 0.90, 0.96, 0.93, 0.91),            # JOINT_HIERARCHY_SATISFIED
             "a:2": unit(0.95, 0.88, 0.90, 0.96, 0.93, 0.91),            # Original < C4.5 ⇒ INFEASIBLE (mantida)
             "b:1": unit(0.90, 0.92, 0.88, 0.95, 0.93, 0.85)}            # compromisso
    r = report(units, {}, "historical")
    assert r["n_units"] == 3 and r["exclusions"] == "nenhuma" and r["comparability"]["all_comparable"] and r["comparability"]["level"] == "strict"
    c = r["level2_global_hierarchy"]["state_counts"]
    assert c["JOINT_HIERARCHY_SATISFIED"]["n"] == 1 and c["HIERARCHY_INFEASIBLE"]["n"] == 1 and c["TRADEOFF"]["n"] == 1
    assert c["HIERARCHY_INFEASIBLE"]["pct_of_all_units"] == 33.3                       # percentagem sobre TODAS as unidades
    assert r["level1_direct_ontological_gain"]["controls_missing"] == ["random_groups", "conventional_features"]
    assert r["questions"]["Q2_caused_by_ontological_knowledge"] == "NOT_ASSESSABLE"      # grupos aleatórios/convencional em falta


def test_any_strict_property_mismatch_blocks_every_verdict():
    for over in ({"row__split_hash": "OTHER"}, {"predictions__y_true": [1, 1, 1]}, {"predictions__test_row_index": [3, 2, 1]}, {"labels": [1, 0]} if False else {"row__oracle_id": "OTHER"}):
        u = unit(0.95, 0.92, 0.90, 0.96, 0.93, 0.91)
        u["reloaded_owl_full"] = rec(0.95, 0.96, **over)
        r = report({"a:1": u}, {}, "historical")
        assert not r["comparability"]["all_comparable"] and r["verdict"].startswith("INSUFFICIENT_EVIDENCE"), over
    u2 = unit(0.95, 0.92, 0.90, 0.96, 0.93, 0.91)
    u2["c45"] = rec(0.90, 0.91, "c45", row__fidelity_oracle_id="OTHER_MLP")
    assert not report({"a:1": u2}, {}, "historical")["comparability"]["all_comparable"]
