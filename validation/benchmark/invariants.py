"""Invariantes do protocolo verificados sobre os resultados BRUTOS (smoke test do protocolo; sem interpretar desempenho)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from validation.benchmark.raw_store import verify_raw_metrics

TREES_CDEF = ("trepan_original", "reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled")
GATE_FIELDS = ("ontology_loaded", "ontology_valid", "ontology_mapped", "mapped_feature_count", "unmapped_feature_count", "mapping_rate",
               "semantic_features_generated", "semantic_features_available", "semantic_features_used", "reasoning_applied",
               "ontology_influenced_splits", "semantic_split_count", "semantic_rule_count", "ontology_usage_rate",
               "semantic_decision_impact", "mirror_applied", "ontology_effectively_used")


def _res(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"invariant": name, "passed": bool(passed), "detail": detail}


def check_unit(records: List[Dict[str, Any]], split_payload: Dict[str, Any], *, expected_arms: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rows = pd.DataFrame([r["row"] for r in records]).set_index("arm", drop=False)
    sem = split_payload["semantic_report"][0]
    present = set(rows.index)
    out.append(_res("arms_A_to_F_present", set(expected_arms) <= present and not split_payload["skipped_arms"],
                    f"presentes={sorted(present)}; omitidos={split_payload['skipped_arms']}"))
    out.append(_res("same_split", rows["split_hash"].nunique() == 1 and rows["split_id"].nunique() == 1, f"split_hash={rows['split_hash'].unique().tolist()}"))
    trees = rows.loc[[a for a in TREES_CDEF if a in rows.index]]
    ids = set(trees["oracle_id"]) | {rows.loc["mlp_original", "oracle_id"]}
    out.append(_res("same_frozen_oracle_id_for_A_C_D_E_F", len(ids) == 1, f"oracle_ids={sorted(map(str, ids))}"))
    out.append(_res("c45_fidelity_against_same_frozen_oracle",
                    pd.notna(rows.loc["c45", "fidelity_to_oracle"]) and rows.loc["c45", "fidelity_oracle_id"] == rows.loc["mlp_original", "oracle_id"]
                    and pd.notna(rows.loc["c45", "disagreement_rate_to_oracle"]),
                    f"fidelity={rows.loc['c45', 'fidelity_to_oracle']}, oracle_id={rows.loc['c45', 'fidelity_oracle_id']}"))
    sp = sem.get("structural_protocol", {})
    out.append(_res("structural_tuning_ontology_blind",
                    bool(sp.get("structural_tuning_ontology_blind")) and sp.get("semantic_inputs_given_to_tuning") is False
                    and trees["structural_tuning_ontology_blind"].astype(bool).all(),
                    f"source={sp.get('source')}, semantic_inputs_given_to_tuning={sp.get('semantic_inputs_given_to_tuning')}"))
    out.append(_res("structural_tuning_actually_ran_without_test",
                    sp.get("source") == "scientific_tuning_frozen" and sp.get("test_used") is False and not sp.get("test_used_for_selection"),
                    f"source={sp.get('source')}, status={sp.get('tuning_status')}, failed={sp.get('failed')}"))
    same_cfg = all(trees[c].nunique(dropna=False) == 1 for c in ("tree_max_nodes", "query_budget", "structural_source", "structural_tuning_status"))
    sel = sp.get("selected", {})
    out.append(_res("same_purity_epsilon_max_nodes_budget_for_C_D_E_F", same_cfg and bool(sel),
                    f"selected={sel}; max_nodes={trees['tree_max_nodes'].unique().tolist()}; query_budget={trees['query_budget'].unique().tolist()}"))
    d_e_f = rows.loc[[a for a in ("reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled") if a in rows.index]]
    out.append(_res("mirror_applied_false_in_D_E_F", not d_e_f["mirror_applied"].astype(bool).any() and not d_e_f["semantic_mirror_applied"].astype(bool).any(),
                    f"mirror_applied={d_e_f['mirror_applied'].tolist()}"))
    missing = [f for f in GATE_FIELDS if f not in d_e_f or d_e_f[f].isna().any()]
    out.append(_res("ontology_quality_gate_filled_in_D_E_F", not missing, f"campos em falta: {missing}"))
    cat = rows.get("ontology_category")
    out.append(_res("ontology_category_assigned_to_E_and_F", cat is not None and cat.loc[["reloaded_owl_full", "reloaded_owl_shuffled"]].notna().all(),
                    f"E={None if cat is None else cat.get('reloaded_owl_full')}; F={None if cat is None else cat.get('reloaded_owl_shuffled')}"))
    d = rows.loc["reloaded_core"]
    out.append(_res("arm_D_has_no_ontological_information",
                    d["semantic_split_count"] == 0 and d["ontology_usage_rate"] == 0 and d["semantic_features_used"] == 0 and not d["reasoning_applied"]
                    and not d["ontology_effectively_used"] and d["ontology_hash"] == "none", f"D gate: split_count={d['semantic_split_count']}"))
    bad_pred = [r["arm"] for r in records if r["predictions"]["surrogate_predictions"] is None
                or len(r["predictions"]["surrogate_predictions"]) != len(r["predictions"]["y_true"]) or not r["predictions"]["oracle_predictions"]]
    out.append(_res("predictions_stored", not bad_pred, f"braços sem predições completas: {bad_pred}"))
    out.append(_res("preprocessing_id_shared_and_fit_on_train_only",
                    rows["preprocessing_id"].nunique() == 1 and sem.get("preprocessing", {}).get("test_used_in_fit") is False
                    and rows["base_representation_hash"].nunique() == 1,
                    f"preprocessing_id={rows['preprocessing_id'].unique().tolist()}"))
    audit = sem.get("protocol_audit", {})
    events = audit.get("events", [])
    final = [e for e in events if e.get("phase") == "final_evaluation"]
    out.append(_res("test_not_used_before_final_evaluation",
                    not audit.get("violations") and len(final) == 1 and all(e["role"] != "test" for e in events if e.get("phase") == "selection"),
                    f"violations={audit.get('violations')}, final_evaluations={len(final)}"))
    nc = sem.get("negative_control") or {}
    cb, ca = nc.get("counts_before") or {}, nc.get("counts_after") or {}
    out.append(_res("negative_control_F_deterministic_and_structure_preserving",
                    bool(nc.get("shuffle_seed") is not None and nc.get("original_ontology_hash") and nc.get("shuffled_ontology_hash"))
                    and nc.get("original_ontology_hash") != nc.get("shuffled_ontology_hash") and cb == ca and bool(cb),
                    f"shuffle_seed={nc.get('shuffle_seed')}, before={cb}, after={ca}"))
    out.append(_res("c45_label_is_native_c45", bool(rows.loc["c45", "c45_is_native"]) and rows.loc["c45", "c45_algorithm"] == "C4.5-Nativo"
                    and bool(rows.loc["c45", "c45_canonical"]) and pd.isna(rows.loc["c45", "c45_node_cap"]), f"{rows.loc['c45', 'c45_algorithm']}"))
    return out


def check_all(records: List[Dict[str, Any]], splits: List[Dict[str, Any]], *, expected_arms: List[str]) -> Dict[str, Any]:
    by_unit: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in records:
        by_unit.setdefault((r["dataset"], r["master_seed"]), []).append(r)
    report: Dict[str, Any] = {"units": {}, "all_passed": True}
    split_by = {(s["dataset"], s["master_seed"]): s for s in splits}
    for (ds, seed), recs in sorted(by_unit.items()):
        checks = check_unit(recs, split_by[(ds, seed)], expected_arms=expected_arms)
        report["units"][f"{ds}/seed{seed}"] = checks
        report["all_passed"] &= all(c["passed"] for c in checks)
    rec = verify_raw_metrics(records)
    report["metrics_recomputable_from_raw"] = rec
    report["all_passed"] &= bool(rec["ok"])
    return report
