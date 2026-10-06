"""Métricas do modo SCIENTIFIC / BENCHMARK para o separador «Métricas»: só reformata números JÁ calculados pelo pipeline científico.

A interface nunca recalcula (sem sklearn, sem predict/fit): o oráculo, o split e o teste são os do pipeline que produziu o ``report``.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def _precision_block(m: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not m:
        return None
    return {"accuracy": m.get("accuracy"), "balanced_accuracy": m.get("balanced_accuracy"), "precision_macro": m.get("precision_macro"),
            "recall_macro": m.get("recall_macro"), "f1_macro": m.get("macro_f1")}


def comparison_results_from_report(report: Mapping[str, Any], oracle_id: Optional[str] = None) -> Dict[str, Any]:
    """``report["evaluation"]["models"]`` -> estrutura ``comparison_results`` esperada pelo separador de métricas."""
    models = (report.get("evaluation") or {}).get("models") or {}
    reference = f"MLP congelado (oracle_id={str(oracle_id)[:8]})" if oracle_id else "MLP congelado"
    keys = {"mlp": "mlp_original", "mlp_ontological": "mlp_semantic", "trepan_original": "original", "trepan_reloaded": "reloaded",
            "c45_j48": "c45_native"}
    precision = {gui_key: block for gui_key, src in keys.items() if (block := _precision_block(models.get(src)))}
    fidelity: Dict[str, Any] = {}
    for gui_key, src, space in (("trepan_original", "original", "original"), ("trepan_reloaded", "reloaded", "reloaded"), ("c45_j48", "c45_native", "original")):
        m = models.get(src) or {}
        if m.get("oracle_fidelity") is not None:
            fidelity[gui_key] = {"overall_fidelity": m["oracle_fidelity"], "fidelity_reference": reference, "feature_space": space}
    return {"precision": precision, "fidelity": fidelity,
            "model_info": {"n_test_samples": (report.get("manifest") or {}).get("test_rows"), "source": "scientific_benchmark_pipeline",
                           "oracle_id": oracle_id}}
