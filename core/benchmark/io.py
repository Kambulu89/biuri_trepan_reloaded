"""Persistência de resultados (sem sobrescrever) e verificação das métricas a partir das predições."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core.benchmark import analysis as an
from core.benchmark import manifest as mf
from core.benchmark.metrics import classification_bundle, fidelity_to_oracle, OracleInfo
from core.benchmark.runner import BenchmarkResult


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def analyze(result: BenchmarkResult, *, seed: int = 0) -> Dict[str, Any]:
    df = result.frame()
    cfg = result.config
    agg = an.aggregate(df, n_boot=cfg["n_boot"], alpha=cfg["ci_alpha"], seed=seed)
    contrasts = an.run_contrasts(df, scheme=cfg["scheme"], n_boot=cfg["n_boot"], alpha=cfg["ci_alpha"], seed=seed)
    return {"aggregate": agg, "contrasts": contrasts,
            "negative_control": [dict(dataset=k[0], metric=k[1], **v) for k, v in an.negative_control_verdicts(contrasts).items()] if len(contrasts) else [],
            "attribution": an.semantic_attribution(contrasts) if len(contrasts) else []}


def save_result(result: BenchmarkResult, analysis: Dict[str, Any], root: str | Path = "results/benchmark") -> Path:
    """results/<experiment_id>/{manifest,config,metrics,tree_metrics,semantic_report}.json + csv + summary.md"""
    from core.benchmark.report import summary_markdown
    run_dir = mf.unique_run_dir(Path(root), result.experiment_id)
    manifest = dict(result.manifest)
    manifest["experiment_id"] = run_dir.name
    manifest["classes"] = [str(c) for c in result.class_counts]
    manifest["n_samples"], manifest["n_features"], manifest["n_classes"] = result.n_samples, result.n_features, result.n_classes
    manifest["class_counts"] = result.class_counts
    manifest["splits"] = result.splits
    (run_dir / "manifest.json").write_text(json.dumps(_clean(manifest), indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(_clean({"config": result.config, "arms": result.arms}), indent=2), encoding="utf-8")
    df = result.frame()
    df.to_csv(run_dir / "raw_results.csv", index=False)
    analysis["aggregate"].to_csv(run_dir / "aggregate.csv", index=False)
    analysis["contrasts"].to_csv(run_dir / "contrasts.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(_clean({
        "raw": df.to_dict(orient="records"), "aggregate": analysis["aggregate"].to_dict(orient="records"),
        "contrasts": analysis["contrasts"].to_dict(orient="records"),
        "negative_control": analysis["negative_control"], "attribution": analysis["attribution"]}), indent=2), encoding="utf-8")
    tree_cols = ["dataset", "split_id", "arm", "family", "node_count", "internal_nodes", "leaf_count", "depth", "average_leaf_depth",
                 "average_rule_length", "features_used", "m_of_n_count", "semantic_split_count", "membership_queries", "query_budget",
                 "query_budget_exhausted", "semantic_mirror_applied"]
    (run_dir / "tree_metrics.json").write_text(json.dumps(_clean(df[[c for c in tree_cols if c in df]].dropna(subset=["node_count"]).to_dict(orient="records")), indent=2), encoding="utf-8")
    (run_dir / "semantic_report.json").write_text(json.dumps(_clean({"per_split": result.semantic_report, "skipped_arms": result.skipped,
                                                                       "not_applicable_ablations": __import__("core.benchmark.runner", fromlist=["x"]).NOT_APPLICABLE_ABLATIONS}), indent=2), encoding="utf-8")
    if len(result.predictions):
        result.predictions.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "summary.md").write_text(summary_markdown(result, analysis), encoding="utf-8")
    return run_dir


def verify_metrics_from_predictions(run_dir: str | Path, tol: float = 1e-9) -> Dict[str, Any]:
    """Recomputa accuracy, balanced accuracy, macro-F1 e fidelity a partir de predictions.csv e compara com raw_results.csv."""
    run_dir = Path(run_dir)
    raw = pd.read_csv(run_dir / "raw_results.csv")
    pred = pd.read_csv(run_dir / "predictions.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg_labels = None
    worst, checked, mismatches = 0.0, 0, []
    all_labels = sorted(pred["y_real"].unique().tolist())
    for _, r in raw.iterrows():
        col = f"pred__{r['arm']}"
        sel = pred[(pred["dataset"] == r["dataset"]) & (pred["split_id"] == r["split_id"])]
        if col not in sel or sel.empty:
            continue
        yp = sel[col].to_numpy(); yt = sel["y_real"].to_numpy()
        minority = type(yt[0])(float(r["minority_label"])) if pd.notna(r.get("minority_label")) else None
        b = classification_bundle(yt, yp, labels=all_labels, minority_label=minority)
        for k, v in b.items():
            if k in r and pd.notna(r[k]) and pd.notna(v):
                diff = abs(float(r[k]) - float(v)); worst = max(worst, diff); checked += 1
                if diff > tol:
                    mismatches.append((r["arm"], r["split_id"], k, float(r[k]), float(v)))
        if pd.notna(r.get("fidelity_to_oracle")):
            ocol = f"oraclepred__{r['oracle_name']}"
            f = fidelity_to_oracle(sel[ocol].to_numpy(), yp, OracleInfo(str(r["oracle_name"]), "x", 1))
            diff = abs(float(r["fidelity_to_oracle"]) - f); worst = max(worst, diff); checked += 1
            if diff > tol:
                mismatches.append((r["arm"], r["split_id"], "fidelity_to_oracle", float(r["fidelity_to_oracle"]), f))
    return {"checked_values": checked, "max_abs_diff": worst, "mismatches": mismatches, "ok": not mismatches and checked > 0}
