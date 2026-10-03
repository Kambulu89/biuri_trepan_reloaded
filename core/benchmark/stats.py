"""Estatística pareada: descrição, IC bootstrap, Wilcoxon/t pareado, effect sizes, Holm/BH, Friedman, evidência."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps

NAN = float("nan")


def describe(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": NAN, "std": NAN, "median": NAN, "min": NAN, "max": NAN}
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """IC percentil bootstrap da MÉDIA (sem assumir normalidade). n<2 -> (nan, nan)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], **kw) -> Tuple[float, float]:
    """IC bootstrap PAREADO da diferença média a-b (reamostra os pares)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("Amostras pareadas exigem o mesmo comprimento.")
    return bootstrap_ci(a - b, **kw)


def cohens_dz(diff: np.ndarray) -> float:
    d = np.asarray(diff, float)
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else (0.0 if d.mean() == 0 else float("inf") * np.sign(d.mean()))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum(float(np.sum(x > b)) for x in a)
    lt = sum(float(np.sum(x < b)) for x in a)
    return float((gt - lt) / (a.size * b.size))


def rank_biserial(diff: Sequence[float]) -> float:
    """Correlação rank-biserial pareada: (W+ - W-)/(W+ + W-), ignorando zeros."""
    d = np.asarray(diff, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(d))
    wp, wm = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def effect_label(value: float, kind: str) -> str:
    x = abs(value)
    cuts = {"cohens_dz": (0.2, 0.5, 0.8), "cliffs_delta": (0.147, 0.33, 0.474), "rank_biserial": (0.1, 0.3, 0.5)}[kind]
    return "negligible" if x < cuts[0] else "small" if x < cuts[1] else "medium" if x < cuts[2] else "large"


def paired_comparison(a: Sequence[float], b: Sequence[float], *, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> Dict:
    """Compara a vs b em pares (mesmo split). Diferença = a - b. Devolve p-values, IC e effect sizes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("Amostras pareadas exigem o mesmo comprimento.")
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = a - b
    n = int(d.size)
    out: Dict = {"n_pairs": n, "mean_a": float(a.mean()) if n else NAN, "mean_b": float(b.mean()) if n else NAN,
                 "mean_diff": float(d.mean()) if n else NAN, "median_diff": float(np.median(d)) if n else NAN,
                 "wins_a": int(np.sum(d > 0)), "wins_b": int(np.sum(d < 0)), "ties": int(np.sum(d == 0))}
    lo, hi = paired_bootstrap_ci(a, b, n_boot=n_boot, alpha=alpha, seed=seed) if n >= 2 else (NAN, NAN)
    out.update(ci_low=lo, ci_high=hi, ci_method=f"paired percentile bootstrap ({n_boot} resamples of pairs)")
    out.update(wilcoxon_p=NAN, wilcoxon_stat=NAN, ttest_p=NAN, normality_p=NAN, normality_ok=None)
    if n >= 2 and np.any(d != 0):
        try:
            w = sps.wilcoxon(d, zero_method="wilcox", method="auto")
            out.update(wilcoxon_p=float(w.pvalue), wilcoxon_stat=float(w.statistic))
        except ValueError:
            pass
    elif n >= 1:
        out["wilcoxon_p"] = 1.0  # todas as diferenças são zero: sem evidência de diferença
    if n >= 3:
        if np.ptp(d) > 0:
            norm_p = float(sps.shapiro(d).pvalue)
            out["normality_p"], out["normality_ok"] = norm_p, bool(norm_p >= 0.05)
            out["ttest_p"] = float(sps.ttest_rel(a, b).pvalue)
        else:
            out["ttest_p"] = 1.0 if d[0] == 0 else 0.0
    out["ttest_valid"] = bool(out["normality_ok"]) and n >= 8
    out["cohens_dz"] = cohens_dz(d) if n >= 2 else NAN
    out["cliffs_delta"] = cliffs_delta(a, b) if n >= 1 else NAN
    out["rank_biserial"] = rank_biserial(d) if n >= 1 else NAN
    out["effect_label"] = effect_label(out["rank_biserial"], "rank_biserial") if n >= 1 else "n/a"
    return out


def holm(pvalues: Sequence[float]) -> List[float]:
    """Correcção Holm-Bonferroni (controla FWER). NaN preservado."""
    p = np.asarray(pvalues, float)
    out = np.full(p.shape, np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out.tolist()


def benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    """Correcção BH (controla FDR). NaN preservado."""
    p = np.asarray(pvalues, float)
    out = np.full(p.shape, np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    m = len(order)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * m / (rank + 1))
        out[idx] = prev
    return out.tolist()


def friedman_posthoc(matrix: np.ndarray, names: Sequence[str], *, alpha: float = 0.05) -> Dict:
    """Friedman (blocos x métodos) + Wilcoxon pareado par-a-par com Holm."""
    M = np.asarray(matrix, float)
    if M.ndim != 2 or M.shape[1] < 3 or M.shape[0] < 3:
        return {"valid": False, "reason": "Friedman exige >=3 métodos e >=3 blocos"}
    stat, p = sps.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    ranks = np.vstack([sps.rankdata(-row) for row in M]).mean(axis=0)
    pairs, ps = [], []
    for i in range(M.shape[1]):
        for j in range(i + 1, M.shape[1]):
            d = M[:, i] - M[:, j]
            pv = float(sps.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            pairs.append((names[i], names[j])); ps.append(pv)
    adj = holm(ps)
    return {"valid": True, "friedman_stat": float(stat), "friedman_p": float(p), "mean_ranks": dict(zip(names, ranks.tolist())),
            "posthoc": [{"a": a, "b": b, "p": pv, "p_holm": ph} for (a, b), pv, ph in zip(pairs, ps, adj)],
            "n_blocks": int(M.shape[0])}


# ----------------------------------------------------------------------------- evidência
MECHANISM_ONLY, INDICATIVE, SUPPORTED = "MECHANISM_ONLY", "INDICATIVE", "STATISTICALLY_SUPPORTED"


@dataclass(frozen=True)
class EvidencePolicy:
    """Limiares fixados ANTES de olhar para os resultados."""
    min_units_supported: int = 10        # pares independentes (seeds/folds) mínimos
    min_test_samples: int = 150          # menor tamanho de teste por split
    alpha: float = 0.05
    min_effect: float = 0.3              # |rank-biserial| mínimo (medium)
    min_pairs_verdict: int = 5           # abaixo disto IC/veredictos são degenerados -> INSUFFICIENT_DATA


def evidence_level(comp: Dict, *, p_adjusted: Optional[float], min_test_samples: int,
                   needs_negative_control: bool = False, negative_control_passed: Optional[bool] = None,
                   policy: EvidencePolicy = EvidencePolicy()) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    n = comp.get("n_pairs", 0)
    if n < 3:
        return MECHANISM_ONLY, [f"apenas {n} pares: demonstra o mecanismo, não permite inferência"]
    ok = True
    if n < policy.min_units_supported:
        ok = False; reasons.append(f"{n} pares < {policy.min_units_supported}")
    if min_test_samples < policy.min_test_samples:
        ok = False; reasons.append(f"teste pequeno ({min_test_samples} < {policy.min_test_samples} amostras)")
    if p_adjusted is None or math.isnan(p_adjusted) or p_adjusted >= policy.alpha:
        ok = False; reasons.append("p ajustado (Holm) não significativo")
    lo, hi = comp.get("ci_low", NAN), comp.get("ci_high", NAN)
    if math.isnan(lo) or (lo <= 0 <= hi):
        ok = False; reasons.append("IC95% da diferença inclui 0")
    if abs(comp.get("rank_biserial", 0.0) or 0.0) < policy.min_effect:
        ok = False; reasons.append("effect size abaixo de 'medium'")
    if needs_negative_control and not negative_control_passed:
        ok = False; reasons.append("controlo negativo semântico não ultrapassado")
    return (SUPPORTED if ok else INDICATIVE), reasons
