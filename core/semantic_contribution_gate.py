"""Gate científico da contribuição ontológica observada no TREPAN Reloaded."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SemanticContributionConfig:
    min_ontology_usage_rate: float = 0.10
    min_decision_impact: float = 0.0
    min_fidelity_gain: float = 0.0
    max_balanced_accuracy_loss: float = 0.01
    max_macro_f1_loss: float = 0.01
    max_node_growth_ratio: float = 0.50


class SemanticContributionGate:
    """Separa 'OWL válida' de 'contribuição observada no modelo final'."""
    def __init__(self, config: SemanticContributionConfig = SemanticContributionConfig()):
        self.config = config

    def evaluate(
        self,
        *,
        ontology_quality: Optional[Dict[str, Any]],
        semantic_audit: Optional[Dict[str, Any]],
        comparison: Optional[Dict[str, Any]],
        original_metrics: Optional[Dict[str, Any]] = None,
        reloaded_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config; quality = ontology_quality or {}; audit = semantic_audit or {}; comp = comparison or {}
        if not quality.get("accepted", False):
            return {"accepted": False, "status": "ONTOLOGY_NOT_VALID", "config": asdict(cfg), "reasons": [quality.get("status", "quality_gate_failed")]}
        usage = float(audit.get("ontology_usage_rate", 0.0) or 0.0)
        impact = float(audit.get("semantic_decision_impact", 0.0) or 0.0)
        efsr_attempted = int(audit.get("error_focused_interventions_attempted", 0) or 0)
        efsr_accepted = int(audit.get("error_focused_interventions_accepted", 0) or 0)
        efsr_rejected = int(audit.get("error_focused_interventions_rejected", 0) or 0)
        efsr_local_gain_observed = bool(efsr_accepted > 0)
        dfid = float(comp.get("delta_oracle_fidelity", 0.0) or 0.0)
        dba = float(comp.get("delta_balanced_accuracy", 0.0) or 0.0)
        df1 = float(comp.get("delta_macro_f1", 0.0) or 0.0)
        reasons = []
        if usage < cfg.min_ontology_usage_rate: reasons.append("ontology_usage_below_threshold")
        if impact < cfg.min_decision_impact: reasons.append("semantic_decision_impact_below_threshold")
        if dfid < cfg.min_fidelity_gain: reasons.append("fidelity_gain_below_threshold")
        if dba < -cfg.max_balanced_accuracy_loss: reasons.append("balanced_accuracy_loss_exceeded")
        if df1 < -cfg.max_macro_f1_loss: reasons.append("macro_f1_loss_exceeded")
        node_growth = 0.0
        if original_metrics and reloaded_metrics:
            base = max(1, int(original_metrics.get("nodes", 0) or 0))
            node_growth = (int(reloaded_metrics.get("nodes", 0) or 0) - base) / base
            if node_growth > cfg.max_node_growth_ratio: reasons.append("tree_complexity_growth_exceeded")
        accepted = not reasons
        if accepted: status = "SEMANTIC_CONTRIBUTION_SUPPORTED"
        elif usage <= 0 and impact <= 0: status = "NO_OBSERVED_SEMANTIC_CONTRIBUTION"
        elif dfid < cfg.min_fidelity_gain: status = "SEMANTIC_EFFECT_WITHOUT_FIDELITY_GAIN"
        else: status = "SEMANTIC_CONTRIBUTION_NOT_SUPPORTED"
        return {
            "accepted": accepted, "status": status, "config": asdict(cfg), "reasons": reasons,
            "ontology_usage_rate": usage, "semantic_decision_impact": impact,
            "error_focused_interventions_attempted": efsr_attempted,
            "error_focused_interventions_accepted": efsr_accepted,
            "error_focused_interventions_rejected": efsr_rejected,
            "error_focused_local_gain_observed": efsr_local_gain_observed,
            "delta_oracle_fidelity": dfid, "delta_balanced_accuracy": dba,
            "delta_macro_f1": df1, "node_growth_ratio": float(node_growth),
        }

__all__ = ["SemanticContributionConfig", "SemanticContributionGate"]
