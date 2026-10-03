"""Exporta exemplos de C4.5, TREPAN Original e TREPAN Reloaded + evidência de não-mutação.

Agnóstico a datasets: dados sintéticos multiclasse (ou --csv genérico). Uso:
    python scripts/export_tree_examples.py [--out docs/tree_visualization_examples] [--csv f.csv]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PyQt6.QtWidgets import QApplication
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from core.c45_j48_tree import C45Classifier
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier
from gui.tree_viz.model import build_visualization_model, source_node_count, tree_signature
from gui.tree_viz.layout import compute_layout, diagnostics
from gui.tree_viz.render import QtTextMeasure, export_tree

SEED = 42


def main(out: str, csv: str | None, classes: int) -> None:
    app = QApplication.instance() or QApplication([])
    if csv:
        import pandas as pd
        frame = pd.read_csv(csv)
        names = [str(c) for c in frame.columns[:-1]]
        X = frame.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        y = pd.factorize(frame.iloc[:, -1])[0]
    else:
        X, y = make_classification(n_samples=700, n_features=10, n_informative=6, n_redundant=2,
                                   n_classes=classes, n_clusters_per_class=1, flip_y=0.02, random_state=SEED)
        names = [f"measurement_{k}_{s}" for k, s in zip(range(10), ["mean", "worst", "std_error"] * 4)]
    class_names = [f"class_{c}" for c in range(len(set(y.tolist())))]
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    mlp = MLPClassifier((32,), max_iter=1000, random_state=SEED).fit(Xtr, ytr)
    trepan_kw = dict(min_sample=1000, max_queries=40000, max_nodes=31, max_depth=8, random_state=SEED)
    trees = {
        "C4.5-Nativo": C45Classifier(random_state=SEED).fit(Xtr, ytr),
        "Trepan-Original": TrepanOriginalClassifier(**trepan_kw).fit(Xtr, oracle=mlp, feature_names=names),
        "Trepan-Reloaded": TrepanReloadedClassifier(alpha=0.0, beta=0.0, **trepan_kw).fit(Xtr, oracle=mlp, feature_names=names),
    }
    outdir = Path(out); outdir.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for label, tree in trees.items():
        sig_before, pred_before = tree_signature(tree), tree.predict(Xte).copy()
        model = build_visualization_model(tree, names, class_names, label)
        layout = compute_layout(model, QtTextMeasure())
        diag = diagnostics(model, layout)
        base = outdir / label.replace(".", "").replace("-", "_").lower()
        files = []
        for fmt in ("png", "svg", "pdf"):
            info = export_tree(model, f"{base}.{fmt}", fmt=fmt, title=f"{label} · synthetic · seed {SEED}",
                               write_json=(fmt == "png"), metadata={"algorithm": label, "seed": SEED})
            files.append(info["path"])
        sig_after, pred_after = tree_signature(tree), tree.predict(Xte)
        evidence[label] = {
            "signature_before": sig_before, "signature_after": sig_after,
            "signature_unchanged": sig_before == sig_after,
            "predictions_unchanged": bool(np.array_equal(pred_before, pred_after)),
            "logical_node_count": diag["logical_node_count"], "rendered_node_count": diag["rendered_node_count"],
            "source_node_count": source_node_count(tree), "overlaps_detected": diag["overlaps_detected"],
            "depth": diag["depth"], "preset": diag["preset"], "verdict": diag["verdict"], "files": files,
        }
    (outdir / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    for label, ev in evidence.items():
        print(label, {k: v for k, v in ev.items() if k != "files"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/tree_visualization_examples")
    ap.add_argument("--csv")
    ap.add_argument("--classes", type=int, default=3)
    a = ap.parse_args()
    main(a.out, a.csv, a.classes)
