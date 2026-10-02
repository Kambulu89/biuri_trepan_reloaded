#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluar_arboles_mejorados.py – Calcula fidelidad y precisión de los árboles
sustitutos mejorados sobre el conjunto de prueba.
"""

import os
import json
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

from counterfactuals._bootstrap import setup
setup()

DATASETS = ['iris', 'wine', 'german_credit', 'wdbc', 'sonar', 'hepatitis']

def evaluar_mejorados(dataset_name):
    model_dir = os.path.join('experimentos', dataset_name, 'modelos')
    mejora_dir = os.path.join('experimentos', dataset_name, 'mejora', 'modelos')
    
    if not os.path.exists(mejora_dir):
        print(f"⚠️  No hay carpeta de mejora para {dataset_name}")
        return {}
    
    try:
        X_test_enc = joblib.load(os.path.join(model_dir, f'X_test_enc_{dataset_name}.pkl'))
        y_test_enc = joblib.load(os.path.join(model_dir, f'y_test_enc_{dataset_name}.pkl'))
        mlp = joblib.load(os.path.join(model_dir, f'mlp_{dataset_name}.pkl'))
        if hasattr(mlp, 'bypass_preprocessing'):
            mlp.bypass_preprocessing = True
    except FileNotFoundError as e:
        print(f"❌ Faltan archivos para {dataset_name}: {e}")
        return {}
    
    y_mlp = mlp.predict(X_test_enc)
    resultados = {}
    
    for filename in sorted(os.listdir(mejora_dir)):
        if not filename.endswith('.pkl'):
            continue
        
        filepath = os.path.join(mejora_dir, filename)
        try:
            arbol = joblib.load(filepath)
            # Intentar obtener el árbol (puede ser un extractor o directamente el árbol)
            if hasattr(arbol, 'predict'):
                tree = arbol
            elif hasattr(arbol, 'explainer_tree') and arbol.explainer_tree is not None:
                tree = arbol.explainer_tree
            else:
                print(f"   ⚠️  {filename}: no se pudo extraer un árbol válido")
                continue
            
            y_pred = tree.predict(X_test_enc)
            fid = accuracy_score(y_mlp, y_pred)
            acc = accuracy_score(y_test_enc, y_pred)
            resultados[filename] = {'fidelidad': fid, 'precision': acc}
        except Exception as e:
            print(f"   ❌ Error al cargar {filename}: {e}")
    
    return resultados

def main():
    print("\n" + "="*90)
    print("EVALUACIÓN DE ÁRBOLES MEJORADOS (FIDELIDAD Y PRECISIÓN)")
    print("="*90)
    
    todos_resultados = {}
    
    for ds in DATASETS:
        resultados = evaluar_mejorados(ds)
        if resultados:
            print(f"\n📁 {ds.upper()}")
            print(f"   {'Archivo':<60} {'Fidelidad':<12} {'Precisión':<12}")
            print(f"   {'-'*84}")
            for filename, metrica in resultados.items():
                print(f"   {filename:<60} {metrica['fidelidad']:<12.4f} {metrica['precision']:<12.4f}")
            todos_resultados[ds] = resultados
        else:
            print(f"\n📁 {ds.upper()}: No se encontraron árboles mejorados.")
    
    with open('analisis/evaluacion_arboles_mejorados.json', 'w') as f:
        json.dump(todos_resultados, f, indent=2)
    print("\n✅ Resultados guardados en 'evaluacion_arboles_mejorados.json'")

if __name__ == "__main__":
    main()