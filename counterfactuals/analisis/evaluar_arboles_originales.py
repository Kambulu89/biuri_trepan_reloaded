#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluar_arboles_originales.py – Calcula fidelidad y precisión de los árboles
sustitutos originales (TREPAN y TREPAN Reloaded) sobre el conjunto de prueba.
"""

import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from counterfactuals._bootstrap import setup
from counterfactuals._paths import models_dir, CF_ROOT
from counterfactuals.dataset_config import ALL_DATASETS

setup()


def evaluar_arboles(dataset_name):
    model_dir = models_dir(dataset_name)

    try:
        X_test_enc = joblib.load(model_dir / f'X_test_enc_{dataset_name}.pkl')
        y_test_enc = joblib.load(model_dir / f'y_test_enc_{dataset_name}.pkl')
        mlp = joblib.load(model_dir / f'mlp_trainer_{dataset_name}.pkl')
        mlp.bypass_preprocessing = True
        tree_a = joblib.load(model_dir / f'trepan_{dataset_name}.pkl')
        tree_b = joblib.load(model_dir / f'trepan_reloaded_{dataset_name}.pkl')
    except FileNotFoundError as e:
        print(f"Faltan archivos para {dataset_name}: {e}")
        return None

    y_mlp = mlp.predict(X_test_enc)
    y_pred_a = tree_a.predict(X_test_enc)
    y_pred_b = tree_b.predict(X_test_enc)

    fid_a = accuracy_score(y_mlp, y_pred_a)
    fid_b = accuracy_score(y_mlp, y_pred_b)
    acc_a = accuracy_score(y_test_enc, y_pred_a)
    acc_b = accuracy_score(y_test_enc, y_pred_b)

    return {
        'TREPAN': {'fidelidad': fid_a, 'precision': acc_a},
        'TREPAN Reloaded': {'fidelidad': fid_b, 'precision': acc_b},
    }


def main():
    print("\n" + "=" * 80)
    print("EVALUACIÓN DE ÁRBOLES ORIGINALES (FIDELIDAD Y PRECISIÓN)")
    print("=" * 80)
    print(f"{'Dataset':<18} {'Árbol':<20} {'Fidelidad':<12} {'Precisión':<12}")
    print("-" * 62)

    resultados = {}
    for ds in ALL_DATASETS:
        res = evaluar_arboles(ds)
        if res:
            resultados[ds] = res
            for nombre, metrica in res.items():
                print(
                    f"{ds:<18} {nombre:<20} {metrica['fidelidad']:<12.4f} "
                    f"{metrica['precision']:<12.4f}"
                )

    out_path = CF_ROOT / 'analisis' / 'evaluacion_arboles_originales.json'
    with open(out_path, 'w') as f:
        json.dump(resultados, f, indent=2)
    print(f"\nResultados guardados en {out_path}")


if __name__ == "__main__":
    main()
