"""Métricas de coerência/plausibilidade das consultas sintéticas (Doc. 21/22). Só leitura; usa apenas treino e a transformação ajustada no treino.

* COERÊNCIA ALGÉBRICA: as colunas derivadas da consulta coincidem com ``transform(originais da consulta)``.
* DOMÍNIO: consultas dentro do intervalo observado no treino (por feature original); inteiros/categorias discretas válidas.
* PLAUSIBILIDADE ESTATÍSTICA: distância entre consultas e linhas reais (KS por feature; AUC de um classificador real-vs-sintético; ≈0,5 = indistinguível).
Algebricamente coerente ≠ estatisticamente plausível: as originais amostradas marginalmente podem ser implausíveis conjuntamente.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np


def algebraic_violation(Q: np.ndarray, transform: Callable, orig_idx: Sequence[int], onto_idx: Sequence[int], ref_enriched: np.ndarray) -> Dict[str, float]:
    if not len(onto_idx) or not len(Q):
        return {"viol_rate": 0.0, "viol_mag_mean": 0.0, "viol_mag_p90": 0.0, "n_derived": int(len(onto_idx))}
    Q = np.asarray(Q, dtype=float)
    re = np.asarray(transform(Q[:, list(orig_idx)]), dtype=float)[:, list(onto_idx)]
    scale = np.maximum(np.std(np.asarray(ref_enriched)[:, list(onto_idx)], axis=0), 1e-12)
    diff = np.abs(Q[:, list(onto_idx)] - re)
    rel = diff / scale                                                     # em unidades de desvio-padrão (treino) da derivada
    row_max = rel.max(axis=1)
    tol = 1e-8                                                             # ruído de ponto flutuante ≪ qualquer violação real
    bad = row_max > tol
    return {"viol_rate": float(bad.mean()), "viol_mag_mean": float(rel.mean()), "viol_mag_p90": float(np.quantile(row_max, 0.9)), "n_derived": int(len(onto_idx))}


def domain_report(Q: np.ndarray, X_ref: np.ndarray, orig_idx: Sequence[int]) -> Dict[str, float]:
    Qo = np.asarray(Q, dtype=float)[:, list(orig_idx)]
    Xo = np.asarray(X_ref, dtype=float)[:, list(orig_idx)]
    lo, hi, sd = Xo.min(axis=0), Xo.max(axis=0), np.maximum(Xo.std(axis=0), 1e-12)
    below, above = np.maximum(lo - Qo, 0.0), np.maximum(Qo - hi, 0.0)
    out_any = ((below > 0) | (above > 0)).any(axis=1)
    excess = (below + above) / sd
    integer_cols = [j for j in range(Xo.shape[1]) if np.allclose(Xo[:, j], np.round(Xo[:, j]), atol=1e-9)]
    discrete_cols = [j for j in integer_cols if len(np.unique(Xo[:, j])) <= 8]
    int_viol = float((np.abs(Qo[:, integer_cols] - np.round(Qo[:, integer_cols])) > 1e-9).any(axis=1).mean()) if integer_cols else 0.0
    cat_viol = 0.0
    if discrete_cols:
        bad = np.zeros(len(Qo), dtype=bool)
        for j in discrete_cols:
            bad |= ~np.isin(Qo[:, j], np.unique(Xo[:, j]))
        cat_viol = float(bad.mean())
    return {"out_of_range_rate": float(out_any.mean()), "excess_mean_sd_units": float(excess.mean()), "excess_max_sd_units": float(excess.max()) if excess.size else 0.0,
            "integer_columns": int(len(integer_cols)), "integrality_violation_rate": int_viol, "discrete_columns": int(len(discrete_cols)),
            "invalid_category_rate": cat_viol}


def distribution_gap(Q: np.ndarray, X_ref: np.ndarray, orig_idx: Sequence[int], onto_idx: Sequence[int], seed: int = 0, cap: int = 1500) -> Dict[str, Optional[float]]:
    from scipy.stats import ks_2samp
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    Q, R = np.asarray(Q, dtype=float), np.asarray(X_ref, dtype=float)
    q = Q[:: max(1, len(Q) // cap)][:cap]
    r = R[:: max(1, len(R) // cap)][:cap]
    out: Dict[str, Optional[float]] = {}
    out["ks_original_mean"] = float(np.mean([ks_2samp(q[:, j], r[:, j]).statistic for j in orig_idx]))
    out["ks_derived_mean"] = float(np.mean([ks_2samp(q[:, j], r[:, j]).statistic for j in onto_idx])) if len(onto_idx) else None

    def auc(cols):
        X = np.vstack([r[:, cols], q[:, cols]])
        y = np.r_[np.zeros(len(r)), np.ones(len(q))]
        if min(len(r), len(q)) < 30:
            return None
        clf = RandomForestClassifier(n_estimators=60, min_samples_leaf=3, random_state=seed, n_jobs=1)
        pr = cross_val_predict(clf, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
        return float(roc_auc_score(y, pr))
    out["auc_real_vs_synth_original_space"] = auc(list(orig_idx))
    out["auc_real_vs_synth_enriched_space"] = auc(list(orig_idx) + list(onto_idx)) if len(onto_idx) else out["auc_real_vs_synth_original_space"]
    return out


def label_shares(y_real: np.ndarray, y_syn: np.ndarray, classes) -> Dict[str, float]:
    pr = np.array([(y_real == c).mean() for c in classes])
    ps = np.array([(y_syn == c).mean() for c in classes]) if len(y_syn) else np.zeros(len(classes))
    return {"label_tv_distance": float(0.5 * np.abs(pr - ps).sum())}
