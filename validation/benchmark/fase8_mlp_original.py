"""MLP ORIGINAL (perfil por omissão, ``fit_mlp(trials=0)``) nas partições da fase 8, para a comparação com o MLP otimizado.
Usa só o treino de cada unidade para ajustar; avalia no teste da mesma unidade. Verifica, onde existe, que reproduz as previsões do baseline
da fase 1 (mesmas partições). Uso: python -m validation.benchmark.fase8_mlp_original DIR_FASE8 DIR_BASELINE SAIDA.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from validation.benchmark import run as R  # noqa: F401  PRIMEIRO: fixa o ambiente numérico
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from validation.benchmark import datasets as ds_mod
from validation.benchmark import datasets_adult  # noqa: F401
from core.benchmark.runner import fit_mlp


def main() -> None:
    root, base, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = []
    for unit in sorted(root.glob("raw/*/seed*")):
        name, seed = unit.parent.name, int(unit.name[4:])
        ds, _, _ = ds_mod.REGISTRY[name].load()
        j = json.loads((unit / "mlp_original.json").read_text(encoding="utf-8"))["predictions"]
        te = np.asarray(j["test_row_index"]); tr = np.setdiff1d(np.arange(len(ds.y)), te)
        model, _ = fit_mlp(ds.X[tr], ds.y[tr], seed, 0, 3)
        p = model.predict(ds.X[te]); yt = ds.y[te]
        ref = base / "raw" / name / f"seed{seed}" / "mlp_original.json"
        repro = None
        if ref.exists():
            r = json.loads(ref.read_text(encoding="utf-8"))["predictions"]
            repro = bool(np.array_equal(np.asarray(r["test_row_index"]), te) and np.array_equal(np.asarray(r["surrogate_predictions"]), p))
        rows.append(dict(dataset=name, seed=seed, acc=float(np.mean(p == yt)), f1=float(f1_score(yt, p, average="macro")),
                         bacc=float(balanced_accuracy_score(yt, p)), reproduz_baseline=repro))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
