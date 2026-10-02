"""Audita onde se calculam métricas (accuracy/F1/fidelity...) e classifica o alvo de cada fidelity.

Uso: python scripts/audit_metrics.py > METRICS_AUDIT.md
Heurística: numa linha de fidelity, ``accuracy_score(A, B)`` é considerada correcta quando A é um
nome de predição de oráculo/professor (mlp, oracle, teacher, active, original...) e SUSPEITA se A for um alvo real (y_test, y_true, y_real).
"""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRIC = re.compile(r"\b(accuracy_score|balanced_accuracy_score|precision_score|recall_score|f1_score|roc_auc_score)\b")
ORACLE_HINT = re.compile(r"(mlp|oracle|teacher|active|original|y_orac|ref)", re.I)
REAL_HINT = re.compile(r"\b(y_test|y_true|y_real|y_eval|y_val)\b")

MANUAL_OK = {("core/active_query_engine.py", 123): "val_labels = _oracle_probabilities(oracle, ref_val)", ("core/active_query_engine.py", 245): "val_labels = _oracle_probabilities(oracle, ref_val)"}
by_file, fid = Counter(), []
for base in ("core", "gui", "counterfactuals"):
    for p in sorted((ROOT / base).rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith("core/benchmark/"):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if METRIC.search(line) and not line.lstrip().startswith(("from ", "import ", "#")):
                by_file[rel] += 1
                if re.search(r"fidel", line, re.I):
                    m = re.search(r"\w+_score\(\s*([^,]+),", line)
                    first = m.group(1).strip() if m else ""
                    verdict = "SUSPEITA (alvo real)" if REAL_HINT.search(first) else ("ok (oráculo)" if ORACLE_HINT.search(first) else "rever")
                    verdict = f"ok (verificado manualmente: {MANUAL_OK[(rel, i)]})" if (rel, i) in MANUAL_OK else verdict
                    fid.append((rel, i, first, verdict, line.strip()[:110]))
print("# METRICS AUDIT\n")
print(f"{sum(by_file.values())} chamadas de métricas sklearn em {len(by_file)} ficheiros (excluindo `core/benchmark/`, o módulo central).\n")
print("| ficheiro | chamadas |\n|---|---|")
for f, n in by_file.most_common(15):
    print(f"| `{f}` | {n} |")
print(f"\n## Linhas de fidelity ({len(fid)})\n\n| ficheiro:linha | 1.º argumento | veredicto |\n|---|---|---|")
for rel, i, first, verdict, line in fid:
    print(f"| `{rel}:{i}` | `{first}` | {verdict} |")
print("\n**Resumo:** " + ", ".join(f"{k}={v}" for k, v in Counter(v[3].split(" (")[0] for v in fid).items()))
