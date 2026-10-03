"""Executa o benchmark científico (agnóstico a datasets) e gera os relatórios.

Exemplos:
    python scripts/run_benchmark.py --smoke --seeds 11 22 33 44 55 66 --out results/benchmark_smoke
    python scripts/run_benchmark.py --csv dados.csv --target classe --ontology onto.owl --scheme repeated_cv
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from core.benchmark import analysis as an
from core.benchmark import report as rp
from core.benchmark.io import analyze, save_result, verify_metrics_from_predictions
from core.benchmark.runner import BenchmarkConfig, BenchmarkRunner, Dataset, TreeBudget
from core.benchmark.synthetic import make_synthetic, write_group_tbox


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--target"); ap.add_argument("--ontology")
    ap.add_argument("--smoke", action="store_true", help="datasets sintéticos (2 e 3 classes) com TBox gerada")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    ap.add_argument("--scheme", default="holdout", choices=["holdout", "repeated_cv"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--mlp-trials", type=int, default=0)
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument("--reports-dir", default=".")
    ap.add_argument("--min-sample", type=int); ap.add_argument("--max-nodes", type=int, default=31)
    a = ap.parse_args()

    budget = TreeBudget(min_sample=a.min_sample, max_nodes=a.max_nodes)
    cfg = BenchmarkConfig(seeds=tuple(a.seeds), scheme=a.scheme, tree=budget, mlp_trials=a.mlp_trials, n_boot=a.n_boot)
    jobs = []
    if a.smoke:
        out = Path(a.out)
        for name, k, nf in (("synthetic_binary", 2, 9), ("synthetic_3class", 3, 12)):
            ds, groups = make_synthetic(name, n_classes=k, n_features=nf, seed=7 + k)
            jobs.append((ds, write_group_tbox(out / "_tbox" / f"{name}.owl", name, groups)))
    else:
        if not a.csv:
            ap.error("indique --csv ou --smoke")
        jobs.append((Dataset.from_file(a.csv, a.target), a.ontology))

    entries, frames = [], {}
    for ds, onto in jobs:
        print(f"[benchmark] {ds.name}: {len(ds.y)} amostras, {ds.X.shape[1]} features, ontologia={'sim' if onto else 'não'}", flush=True)
        result = BenchmarkRunner(cfg).run(ds, onto)
        analysis = analyze(result)
        run_dir = save_result(result, analysis, a.out)
        ver = verify_metrics_from_predictions(run_dir)
        print(f"[benchmark] guardado em {run_dir} · verificação de métricas: {'OK' if ver['ok'] else 'FALHOU'} ({ver['checked_values']} valores)", flush=True)
        entries.append(dict(result=result, analysis=analysis, run_dir=str(run_dir), verification=ver))
        frames[ds.name] = result.frame()
    import pandas as pd
    res_all = pd.concat([e["analysis"]["contrasts"] for e in entries], ignore_index=True)
    cross = an.cross_dataset(frames, res_all) if len(entries) > 1 else None
    verification = {"checked_values": sum(e["verification"]["checked_values"] for e in entries),
                    "max_abs_diff": max(e["verification"]["max_abs_diff"] for e in entries), "ok": all(e["verification"]["ok"] for e in entries)}
    rdir = Path(a.reports_dir)
    (rdir / "SCIENTIFIC_VALIDATION_REPORT.md").write_text(rp.scientific_validation_markdown(entries, cross, verification=verification), encoding="utf-8")
    (rdir / "NEGATIVE_CONTROL_REPORT.md").write_text(rp.negative_control_markdown(entries), encoding="utf-8")
    (rdir / "ABLATION_REPORT.md").write_text(rp.ablation_markdown(entries), encoding="utf-8")
    print("[benchmark] relatórios escritos em", rdir.resolve(), flush=True)


if __name__ == "__main__":
    main()
