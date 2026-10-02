"""
Auditoria científica unificada do pipeline BIURI / Trepan Reloaded.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from core.feature_alignment import oracle_n_features


def _results_dir() -> Path:
    out = Path(__file__).resolve().parent.parent / "results" / "pipeline_audit"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fmt_metrics(block: Optional[Dict[str, Any]]) -> str:
    if not block:
        return "N/A"
    parts = []
    for key in ("accuracy", "precision", "recall", "f1", "fidelity"):
        val = block.get(key)
        if val is not None:
            parts.append(f"{key}={val * 100:.1f}%" if val <= 1 else f"{key}={val:.4f}")
    return ", ".join(parts) if parts else "N/A"


def log_mlp_original(
    n_features: Optional[int],
    metrics: Optional[Dict[str, float]] = None,
    optimization_method: Optional[str] = None,
) -> None:
    print("\n[MLP ORIGINAL]")
    print(f"features: {n_features if n_features is not None else 'N/A'}")
    if optimization_method:
        print(f"optimization: {optimization_method}")
    if metrics:
        print(f"accuracy: {metrics.get('accuracy', 0) * 100:.1f}%")
        print(f"precision: {metrics.get('precision', 0) * 100:.1f}%")
        print(f"recall: {metrics.get('recall', 0) * 100:.1f}%")
        print(f"f1: {metrics.get('f1', 0) * 100:.1f}%")


def log_mlp_onto(
    n_features: Optional[int],
    metrics: Optional[Dict[str, float]] = None,
    optimization_method: Optional[str] = None,
    available: bool = True,
    model_label: str = "MLP Residual Ontológico",
) -> None:
    print(f"\n[{model_label.upper().replace(' ', '_')}]")
    if not available:
        print("status: not available (no valid ontology / enriched features)")
        return
    print(f"features: {n_features if n_features is not None else 'N/A'}")
    if optimization_method:
        print(f"optimization: {optimization_method}")
    if metrics:
        print(f"accuracy: {metrics.get('accuracy', 0) * 100:.1f}%")
        print(f"precision: {metrics.get('precision', 0) * 100:.1f}%")
        print(f"recall: {metrics.get('recall', 0) * 100:.1f}%")
        print(f"f1: {metrics.get('f1', 0) * 100:.1f}%")


def log_ontology_acceptance(acceptance: Optional[Dict[str, Any]], selected_oracle: Optional[str] = None) -> None:
    print("\n[ONTOLOGY ACCEPTANCE]")
    if not acceptance:
        print("status: not evaluated")
        return
    print(f"ontology_quality_accepted: {acceptance.get('ontology_quality_accepted')}")
    print(f"ontology_structural_available: {acceptance.get('ontology_structural_available')}")
    print(f"ontology_feature_gate_accepted: {acceptance.get('ontology_feature_gate_accepted')}")
    print(f"oracle_gate_accepted: {acceptance.get('oracle_gate_accepted')}")
    print(f"surrogate_gate_accepted: {acceptance.get('surrogate_gate_accepted')}")
    print(f"MLP Original accuracy: {acceptance.get('accuracy_original', 0) * 100:.1f}%")
    print(f"MLP Original macro_f1: {acceptance.get('f1_original', 0) * 100:.1f}%")
    print(f"MLP Original balanced_accuracy: {acceptance.get('balanced_accuracy_original', 0) * 100:.1f}%")
    print(f"MLP Onto/Residual accuracy: {acceptance.get('accuracy_onto', 0) * 100:.1f}%")
    print(f"MLP Onto/Residual macro_f1: {acceptance.get('f1_onto', 0) * 100:.1f}%")
    print(f"MLP Onto/Residual balanced_accuracy: {acceptance.get('balanced_accuracy_onto', 0) * 100:.1f}%")
    print(f"accuracy_gain: {acceptance.get('accuracy_gain', 0) * 100:+.1f}%")
    print(f"f1_gain: {acceptance.get('f1_gain', 0) * 100:+.1f}%")
    print(f"ontology_impact: {acceptance.get('ontology_impact', 'N/A')}")
    print(f"Ontology accepted: {acceptance.get('ontology_accepted_label', 'N/A')}")
    if selected_oracle:
        print(f"Selected oracle for Trepan-Reloaded: {selected_oracle}")


def log_trepan_original(
    oracle: str,
    accuracy: Optional[float] = None,
    fidelity: Optional[float] = None,
) -> None:
    print("\n[TREPAN ORIGINAL]")
    print(f"oracle: {oracle}")
    if accuracy is not None:
        print(f"accuracy: {accuracy * 100:.1f}%")
    if fidelity is not None:
        print(f"fidelity: {fidelity * 100:.1f}%")


def log_trepan_reloaded(
    oracle: str,
    accuracy: Optional[float] = None,
    fidelity: Optional[float] = None,
) -> None:
    print("\n[TREPAN RELOADED]")
    print(f"oracle: {oracle}")
    if accuracy is not None:
        print(f"accuracy: {accuracy * 100:.1f}%")
    if fidelity is not None:
        print(f"fidelity: {fidelity * 100:.1f}%")


def log_c45(metrics: Optional[Dict[str, float]] = None) -> None:
    print("\n[C4.5]")
    if metrics:
        print(f"accuracy: {metrics.get('accuracy', 0) * 100:.1f}%")
        print(f"precision: {metrics.get('precision', 0) * 100:.1f}%")
        print(f"recall: {metrics.get('recall', 0) * 100:.1f}%")
        print(f"f1: {metrics.get('f1', 0) * 100:.1f}%")


def log_pipeline_training_audit(
    dataset_name: str,
    mlp_original_model=None,
    mlp_onto_model=None,
    mlp_original_summary: Optional[Dict[str, Any]] = None,
    mlp_onto_summary: Optional[Dict[str, Any]] = None,
    trepan_original_audit: Optional[Dict[str, Any]] = None,
    trepan_reloaded_audit: Optional[Dict[str, Any]] = None,
    c45_metrics: Optional[Dict[str, float]] = None,
    ontology_acceptance: Optional[Dict[str, Any]] = None,
    selected_oracle_label: Optional[str] = None,
) -> Path:
    """Logs obrigatórios após treino."""
    print("\n" + "=" * 60)
    print("[PIPELINE TRAINING AUDIT]")
    print(f"dataset: {dataset_name}")
    print("=" * 60)

    orig_n = oracle_n_features(mlp_original_model)
    onto_n = oracle_n_features(mlp_onto_model) if mlp_onto_model is not None else None

    orig_metrics = (mlp_original_summary or {}).get("test_metrics") or {}
    if trepan_original_audit:
        orig_metrics = {
            "accuracy": trepan_original_audit.get("mlp_accuracy"),
            "precision": None,
            "recall": None,
            "f1": None,
        }
        if mlp_original_summary and mlp_original_summary.get("test_metrics"):
            orig_metrics = dict(mlp_original_summary["test_metrics"])

    log_mlp_original(
        orig_n,
        orig_metrics,
        optimization_method=(mlp_original_summary or {}).get("optimization_method"),
    )

    onto_metrics = (mlp_onto_summary or {}).get("test_metrics") if mlp_onto_summary else None
    if ontology_acceptance:
        onto_metrics = ontology_acceptance.get("residual_metrics") or onto_metrics
    log_mlp_onto(
        onto_n,
        onto_metrics,
        optimization_method=(mlp_onto_summary or {}).get("optimization_method"),
        available=mlp_onto_model is not None or ontology_acceptance is not None,
        model_label="MLP Residual Ontológico",
    )

    log_ontology_acceptance(ontology_acceptance, selected_oracle_label)

    if trepan_original_audit:
        log_trepan_original(
            oracle="MLP Original",
            accuracy=trepan_original_audit.get("trepan_accuracy"),
            fidelity=trepan_original_audit.get("trepan_fidelity"),
        )

    if trepan_reloaded_audit:
        oracle_label = selected_oracle_label or trepan_reloaded_audit.get("oracle", "MLP Residual Ontológico")
        log_trepan_reloaded(
            oracle=oracle_label,
            accuracy=trepan_reloaded_audit.get("trepan_accuracy"),
            fidelity=trepan_reloaded_audit.get("trepan_fidelity"),
        )

    log_c45(c45_metrics)

    records = _build_training_records(
        dataset_name,
        mlp_original_summary,
        mlp_onto_summary,
        trepan_original_audit,
        trepan_reloaded_audit,
        c45_metrics,
        orig_n,
        onto_n,
        ontology_acceptance=ontology_acceptance,
        selected_oracle_label=selected_oracle_label,
    )
    if ontology_acceptance:
        records.append(_build_scientific_ontology_record(
            dataset_name, ontology_acceptance, selected_oracle_label
        ))
    return save_pipeline_audit_record(
        dataset_name,
        stage="training",
        records=records,
    )


def log_pipeline_comparison_audit(
    dataset_name: str,
    comparison_results: Dict[str, Any],
    fidelity_ref_reloaded: Optional[str] = None,
) -> Path:
    """Logs obrigatórios após Comparar Métricas."""
    precision = comparison_results.get("precision", {})
    fidelity = comparison_results.get("fidelity", {})

    print("\n" + "=" * 60)
    print("[PIPELINE COMPARISON AUDIT]")
    print(f"dataset: {dataset_name}")
    print("=" * 60)

    mlp_p = precision.get("mlp") or {}
    log_mlp_original(
        None,
        {
            "accuracy": mlp_p.get("accuracy"),
            "precision": mlp_p.get("precision"),
            "recall": mlp_p.get("recall"),
            "f1": mlp_p.get("f1"),
        },
    )

    trep_o = precision.get("trepan_original") or {}
    fid_o = fidelity.get("trepan_original") or {}
    log_trepan_original(
        oracle="MLP Original",
        accuracy=trep_o.get("accuracy"),
        fidelity=fid_o.get("overall_fidelity"),
    )

    trep_r = precision.get("trepan_reloaded") or {}
    fid_r = fidelity.get("trepan_reloaded") or {}
    oracle = fidelity_ref_reloaded or fid_r.get("fidelity_reference", "mlp_original")
    oracle_label = "MLP Residual Ontológico" if oracle == "mlp_onto" else "MLP Original"
    log_trepan_reloaded(
        oracle=oracle_label,
        accuracy=trep_r.get("accuracy"),
        fidelity=fid_r.get("overall_fidelity"),
    )

    c45_p = precision.get("c45_j48") or {}
    log_c45(
        {
            "accuracy": c45_p.get("accuracy"),
            "precision": c45_p.get("precision"),
            "recall": c45_p.get("recall"),
            "f1": c45_p.get("f1"),
        }
    )

    return save_pipeline_audit_record(
        dataset_name,
        stage="comparison",
        records=_build_comparison_records(dataset_name, comparison_results),
    )


def _build_scientific_ontology_record(
    dataset_name: str,
    acceptance: Dict[str, Any],
    selected_oracle_label: Optional[str],
) -> Dict[str, Any]:
    """Registo científico consolidado do impacto ontológico."""
    timestamp = datetime.now().isoformat()
    return {
        "dataset": dataset_name,
        "model": "Ontology Scientific Report",
        "optimization_method": "residual_ontological_acceptance",
        "n_features": None,
        "accuracy": acceptance.get("accuracy_onto"),
        "precision": None,
        "recall": None,
        "f1": acceptance.get("f1_onto"),
        "fidelity": None,
        "training_time_seconds": None,
        "stage": "training",
        "timestamp": timestamp,
        "accuracy_original": acceptance.get("accuracy_original"),
        "accuracy_ontological": acceptance.get("accuracy_onto"),
        "f1_original": acceptance.get("f1_original"),
        "f1_ontological": acceptance.get("f1_onto"),
        "accuracy_gain": acceptance.get("accuracy_gain"),
        "f1_gain": acceptance.get("f1_gain"),
        "ontology_accepted": acceptance.get("ontology_accepted_label"),
        "final_oracle": selected_oracle_label or "MLP Original",
        "ontology_impact": acceptance.get("ontology_impact"),
        "ontology_quality_accepted": acceptance.get("ontology_quality_accepted"),
        "ontology_structural_available": acceptance.get("ontology_structural_available"),
        "ontology_feature_gate_accepted": acceptance.get("ontology_feature_gate_accepted"),
        "oracle_gate_accepted": acceptance.get("oracle_gate_accepted"),
        "surrogate_gate_accepted": acceptance.get("surrogate_gate_accepted"),
        "quality_gate_reason": acceptance.get("reason"),
    }


def _build_training_records(
    dataset_name,
    mlp_original_summary,
    mlp_onto_summary,
    trepan_original_audit,
    trepan_reloaded_audit,
    c45_metrics,
    orig_n,
    onto_n,
    ontology_acceptance=None,
    selected_oracle_label=None,
):
    rows = []
    timestamp = datetime.now().isoformat()

    for summary, model_name in (
        (mlp_original_summary, "MLP Original"),
        (mlp_onto_summary, "MLP Residual Ontológico"),
    ):
        if not summary:
            continue
        tm = summary.get("test_metrics") or {}
        if model_name == "MLP Residual Ontológico" and ontology_acceptance:
            tm = dict(tm)
            tm.setdefault("accuracy", ontology_acceptance.get("accuracy_onto"))
            tm.setdefault("f1", ontology_acceptance.get("f1_onto"))
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "optimization_method": summary.get("optimization_method"),
                "n_features": orig_n if model_name == "MLP Original" else onto_n,
                "accuracy": tm.get("accuracy"),
                "precision": tm.get("precision"),
                "recall": tm.get("recall"),
                "f1": tm.get("f1"),
                "fidelity": None,
                "training_time_seconds": summary.get("training_time_seconds"),
                "stage": "training",
                "timestamp": timestamp,
            }
        )

    if trepan_original_audit:
        rows.append(
            {
                "dataset": dataset_name,
                "model": "Trepan Original",
                "optimization_method": "oracle_mlp",
                "n_features": orig_n,
                "accuracy": trepan_original_audit.get("trepan_accuracy"),
                "precision": None,
                "recall": None,
                "f1": None,
                "fidelity": trepan_original_audit.get("trepan_fidelity"),
                "training_time_seconds": None,
                "stage": "training",
                "timestamp": timestamp,
            }
        )

    if trepan_reloaded_audit:
        rows.append(
            {
                "dataset": dataset_name,
                "model": "Trepan Reloaded",
                "optimization_method": "oracle_mlp",
                "n_features": trepan_reloaded_audit.get("n_features"),
                "accuracy": trepan_reloaded_audit.get("trepan_accuracy"),
                "precision": None,
                "recall": None,
                "f1": None,
                "fidelity": trepan_reloaded_audit.get("trepan_fidelity"),
                "training_time_seconds": None,
                "stage": "training",
                "timestamp": timestamp,
            }
        )

    if c45_metrics:
        rows.append(
            {
                "dataset": dataset_name,
                "model": "C4.5-Nativo",
                "optimization_method": "real_labels",
                "n_features": orig_n,
                "accuracy": c45_metrics.get("accuracy"),
                "precision": c45_metrics.get("precision"),
                "recall": c45_metrics.get("recall"),
                "f1": c45_metrics.get("f1"),
                "fidelity": None,
                "training_time_seconds": None,
                "stage": "training",
                "timestamp": timestamp,
            }
        )

    return rows


def _build_comparison_records(dataset_name, comparison_results):
    precision = comparison_results.get("precision", {})
    fidelity = comparison_results.get("fidelity", {})
    timestamp = datetime.now().isoformat()
    rows = []

    mapping = {
        "mlp": ("MLP Original", None),
        "c45_j48": ("C4.5-Nativo", None),
        "trepan_original": ("Trepan Original", "mlp"),
        "trepan_reloaded": ("Trepan Reloaded", "trepan_reloaded"),
    }

    for key, (label, fid_key) in mapping.items():
        p = precision.get(key)
        if not p:
            continue
        f_block = fidelity.get(fid_key or key) if fid_key or key in fidelity else None
        rows.append(
            {
                "dataset": dataset_name,
                "model": label,
                "optimization_method": "evaluation",
                "n_features": None,
                "accuracy": p.get("accuracy"),
                "precision": p.get("precision"),
                "recall": p.get("recall"),
                "f1": p.get("f1"),
                "fidelity": (f_block or {}).get("overall_fidelity"),
                "training_time_seconds": None,
                "stage": "comparison",
                "timestamp": timestamp,
            }
        )

    return rows


def save_pipeline_audit_record(dataset_name: str, stage: str, records) -> Path:
    out = _results_dir()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out / f"pipeline_audit_{stage}_{safe}_{ts}.json"
    csv_path = out / f"pipeline_audit_{stage}_{safe}_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    fieldnames = [
        "dataset",
        "model",
        "optimization_method",
        "n_features",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fidelity",
        "training_time_seconds",
        "stage",
        "timestamp",
        "accuracy_original",
        "accuracy_ontological",
        "f1_original",
        "f1_ontological",
        "accuracy_gain",
        "f1_gain",
        "ontology_accepted",
        "final_oracle",
        "ontology_impact",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"[INFO] Auditoria guardada em:\n  {json_path}\n  {csv_path}")
    return json_path


def _tree_stats(tree) -> Dict[str, Any]:
    if tree is None:
        return {"n_nodes": 0, "n_leaves": 0, "is_trivial": True}
    n_leaves = int(tree.get_n_leaves())
    n_nodes = int(getattr(getattr(tree, "tree_", None), "node_count", 0) or 0)
    return {
        "n_nodes": n_nodes,
        "n_leaves": n_leaves,
        "is_trivial": n_leaves <= 1,
    }


def log_final_model_audit(
    mlp_diagnostic: Optional[Dict[str, Any]] = None,
    mlp_model=None,
    c45_metrics: Optional[Dict[str, float]] = None,
    trepan_original_audit: Optional[Dict[str, Any]] = None,
    trepan_reloaded_audit: Optional[Dict[str, Any]] = None,
    trepan_original_tree=None,
    trepan_reloaded_tree=None,
    ontology_acceptance: Optional[Dict[str, Any]] = None,
    selected_oracle_label: Optional[str] = None,
    reloaded_oracle_diagnostic: Optional[Dict[str, Any]] = None,
) -> None:
    """Bloco [FINAL MODEL AUDIT] antes de Comparar Métricas."""
    from core.mlp_diagnostic import _format_distribution

    print("\n" + "=" * 60)
    print("[FINAL MODEL AUDIT]")
    print("=" * 60)

    diag = mlp_diagnostic or {}
    print("\nMLP Original:")
    print(f"  accuracy: {diag.get('accuracy', 'N/A')}")
    print(f"  f1: {diag.get('f1_weighted', 'N/A')}")
    print(f"  balanced_accuracy: {diag.get('balanced_accuracy', 'N/A')}")
    print(f"  unique predictions: {diag.get('unique_y_pred_test', 'N/A')}")
    print(
        f"  prediction distribution: "
        f"{_format_distribution(diag.get('distribution_y_pred_test', {}))}"
    )
    print(f"  degenerate: {diag.get('degenerate', 'N/A')}")

    print("\nC4.5-Nativo:")
    if c45_metrics:
        print(f"  accuracy: {c45_metrics.get('accuracy')}")
        print(f"  f1: {c45_metrics.get('f1')}")
        print(f"  balanced_accuracy: {c45_metrics.get('balanced_accuracy', 'N/A')}")
        print(f"  unique predictions: {c45_metrics.get('unique_predictions', 'N/A')}")
    else:
        print("  (not available)")

    orig_stats = _tree_stats(trepan_original_tree)
    oracle_unique_orig = (
        trepan_original_audit.get("unique_y_oracle_train")
        if trepan_original_audit
        else diag.get("unique_y_pred_train")
    )
    print("\nTrepan Original:")
    if trepan_original_audit:
        print(f"  accuracy: {trepan_original_audit.get('trepan_accuracy')}")
        print(f"  fidelity to MLP Original: {trepan_original_audit.get('trepan_fidelity')}")
    print(f"  number of nodes: {orig_stats['n_nodes']}")
    print(f"  number of leaves: {orig_stats['n_leaves']}")
    trivial_orig = orig_stats["is_trivial"]
    if trivial_orig and oracle_unique_orig and oracle_unique_orig > 1:
        print("  tree is trivial: True (BUG: oráculo multi-classe, árvore folha única)")
    elif trivial_orig:
        print("  tree is trivial: True (oráculo constante)")
    else:
        print("  tree is trivial: False")
    print(f"  oracle unique predictions: {oracle_unique_orig}")

    reloaded_stats = _tree_stats(trepan_reloaded_tree)
    reloaded_oracle_diag = reloaded_oracle_diagnostic or {}
    oracle_unique_rel = reloaded_oracle_diag.get(
        "unique_pred_test",
        reloaded_oracle_diag.get("unique_pred_train"),
    )
    print("\nTrepan Reloaded:")
    if trepan_reloaded_audit:
        print(f"  accuracy: {trepan_reloaded_audit.get('trepan_accuracy')}")
        print(f"  fidelity to selected oracle: {trepan_reloaded_audit.get('trepan_fidelity')}")
    print(f"  selected oracle: {selected_oracle_label or trepan_reloaded_audit.get('oracle', 'N/A')}")
    print(f"  number of nodes: {reloaded_stats['n_nodes']}")
    print(f"  number of leaves: {reloaded_stats['n_leaves']}")
    trivial_rel = reloaded_stats["is_trivial"]
    if trivial_rel and oracle_unique_rel and oracle_unique_rel > 1:
        print("  tree is trivial: True (BUG: oráculo multi-classe, árvore folha única)")
    elif trivial_rel:
        print("  tree is trivial: True (oráculo constante)")
    else:
        print("  tree is trivial: False")
    print(f"  oracle unique predictions: {oracle_unique_rel}")

    print("\nMLP Onto/Residual:")
    if ontology_acceptance:
        print("  exists: True")
        print(f"  accepted: {ontology_acceptance.get('ontology_accepted_label', 'N/A')}")
        print(f"  accuracy: {ontology_acceptance.get('accuracy_onto')}")
        print(f"  f1: {ontology_acceptance.get('f1_onto')}")
        print(f"  balanced_accuracy: {ontology_acceptance.get('balanced_accuracy_onto')}")
        onto_metrics = ontology_acceptance.get("residual_metrics") or {}
        print(f"  unique predictions: {onto_metrics.get('unique_predictions', 'N/A')}")
    else:
        print("  exists: False")
        print("  accepted: N/A")
