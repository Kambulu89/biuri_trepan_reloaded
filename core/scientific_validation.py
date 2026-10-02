"""Validacao estatistica pareada para comparacoes de modelos substitutos.

O modulo nao declara vencedores a partir de barras pontuais. Usa as mesmas
linhas de teste, o mesmo oraculo de referencia e margens predefinidas.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
from scipy.stats import binomtest


def paired_bootstrap_accuracy_difference(
    y_reference,
    pred_a,
    pred_b,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """IC bootstrap da diferenca accuracy(A)-accuracy(B), pareada por linha."""
    y = np.asarray(y_reference)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    if not (len(y) == len(a) == len(b)) or len(y) == 0:
        raise ValueError("y_reference, pred_a e pred_b devem ter o mesmo tamanho > 0.")
    delta_rows = (a == y).astype(float) - (b == y).astype(float)
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(y), size=(int(n_bootstrap), len(y)))
    samples = delta_rows[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "difference": float(delta_rows.mean()),
        "ci_lower": float(np.quantile(samples, alpha / 2)),
        "ci_upper": float(np.quantile(samples, 1 - alpha / 2)),
        "probability_a_better": float(np.mean(samples > 0)),
        "confidence": float(confidence),
        "n": int(len(y)),
        "method": "paired_bootstrap_percentile",
    }


def mcnemar_exact(y_true, pred_a, pred_b) -> Dict[str, float]:
    """Teste de McNemar exacto nas discordancias de correcao."""
    y = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    a_only = int(np.sum(a_ok & ~b_ok))
    b_only = int(np.sum(~a_ok & b_ok))
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue
    )
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
        "method": "exact_mcnemar",
    }


def assess_pairwise_superiority(
    y_true,
    oracle_predictions,
    candidate_predictions,
    baseline_predictions,
    predictive_margin: float = 0.0,
    fidelity_noninferiority_margin: float = 0.01,
    n_bootstrap: int = 5000,
    random_state: int = 42,
) -> Dict[str, object]:
    """Avalia superioridade preditiva e nao-inferioridade de fidelidade.

    Uma alegacao forte requer simultaneamente: limite inferior do IC da
    diferenca preditiva acima da margem, fidelidade nao-inferior ao mesmo
    oraculo e McNemar p<0.05. Caso contrario o resultado e inconclusivo ou
    apenas nao-inferior.
    """
    predictive = paired_bootstrap_accuracy_difference(
        y_true, candidate_predictions, baseline_predictions,
        n_bootstrap=n_bootstrap, random_state=random_state,
    )
    fidelity = paired_bootstrap_accuracy_difference(
        oracle_predictions, candidate_predictions, baseline_predictions,
        n_bootstrap=n_bootstrap, random_state=random_state + 1,
    )
    mcnemar = mcnemar_exact(y_true, candidate_predictions, baseline_predictions)
    predictive_superior = predictive["ci_lower"] > predictive_margin
    fidelity_noninferior = fidelity["ci_lower"] >= -fidelity_noninferiority_margin
    statistically_significant = mcnemar["p_value"] < 0.05
    if predictive_superior and fidelity_noninferior and statistically_significant:
        verdict = "superior"
    elif predictive["ci_lower"] >= -fidelity_noninferiority_margin and fidelity_noninferior:
        verdict = "non_inferior"
    else:
        verdict = "inconclusive_or_inferior"
    return {
        "verdict": verdict,
        "predictive_accuracy_difference": predictive,
        "same_oracle_fidelity_difference": fidelity,
        "mcnemar": mcnemar,
        "criteria": {
            "predictive_superior": predictive_superior,
            "same_oracle_fidelity_noninferior": fidelity_noninferior,
            "mcnemar_significant": statistically_significant,
            "predictive_margin": predictive_margin,
            "fidelity_noninferiority_margin": fidelity_noninferiority_margin,
        },
    }


def build_scientific_validation_report(
    y_true,
    oracle_predictions,
    predictions: Mapping[str, np.ndarray],
    candidate_key: str = "trepan_reloaded",
    n_bootstrap: int = 5000,
    random_state: int = 42,
) -> Dict[str, object]:
    candidate = predictions.get(candidate_key)
    if candidate is None:
        return {"status": "not_available", "reason": "candidate_predictions_missing"}
    comparisons = {}
    for index, baseline_key in enumerate(("trepan_original", "c45_j48")):
        baseline = predictions.get(baseline_key)
        if baseline is None:
            continue
        comparisons[baseline_key] = assess_pairwise_superiority(
            y_true,
            oracle_predictions,
            candidate,
            baseline,
            n_bootstrap=n_bootstrap,
            random_state=random_state + index * 17,
        )
    strong_claim_supported = bool(comparisons) and all(
        item["verdict"] == "superior" for item in comparisons.values()
    )
    return {
        "status": "ok",
        "protocol": "paired_holdout_same_rows_same_original_oracle",
        "candidate": candidate_key,
        "comparisons": comparisons,
        "strong_superiority_claim_supported": strong_claim_supported,
        "claim_guard": (
            "superiority_supported" if strong_claim_supported
            else "do_not_claim_superiority_from_point_estimates"
        ),
    }
