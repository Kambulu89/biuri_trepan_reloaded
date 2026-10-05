#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
consolidar_resultados.py – Consolida todos los resultados experimentales en un solo lugar.

Genera los archivos CSV necesarios:
- consistencia_completa.csv      → indicadores de coincidencia, divergencia y brecha
- mejora_completa.csv            → fidelidad y precisión de árboles mejorados
- arboles_originales.csv         → fidelidad y precisión de árboles originales
- arboles_mejorados.csv          → ídem para árboles mejorados (desde los .pkl)
- tabla_resumen.csv    → tabla maestra con todos los datos combinados
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from pathlib import Path

from counterfactuals._bootstrap import setup
from counterfactuals._paths import (
    CF_ROOT, experiment_dir, models_dir, results_dir, mejora_dir,
    RESULTADOS_MEJORA_DIR,
)
from validation.counterfactual_research.dataset_registry import ALL_DATASETS

setup()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
DATASETS = ALL_DATASETS
CONSOLIDAR_DIR = CF_ROOT / 'analisis' / 'consolidar_resultados'

CONSOLIDAR_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
def cargar_consistencia(dataset_name):
    """Carga los indicadores de consistencia desde consistency_indicators.csv"""
    path = results_dir(dataset_name) / 'consistency_indicators.csv'
    if not path.exists():
        print(f"⚠️  No se encontró {path}")
        return None
    df = pd.read_csv(path)
    df['dataset'] = dataset_name
    return df

def cargar_training_summary(dataset_name):
    """Carga el resumen de entrenamiento (fidelidad, precisión MLP, etc.)"""
    path = experiment_dir(dataset_name) / 'training_summary.json'
    if not path.exists():
        print(f"⚠️  No se encontró {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def cargar_mejora_resumen(dataset_name):
    """Carga el resumen de mejora (mejora_resumen.csv)"""
    path = RESULTADOS_MEJORA_DIR / 'mejora_resumen.csv'
    if not path.exists():
        path = mejora_dir(dataset_name) / 'metricas' / 'mejora_resumen.csv'
    if not path.exists():
        print(f"⚠️  No se encontró resumen de mejora para {dataset_name}")
        return None
    df = pd.read_csv(path)
    # Filtrar solo las filas de este dataset
    if 'dataset' in df.columns:
        df = df[df['dataset'] == dataset_name]
    return df

def evaluar_arboles_originales(dataset_name):
    """Calcula fidelidad y precisión de los árboles originales"""
    model_dir = models_dir(dataset_name)
    try:
        X_test = joblib.load(model_dir / f'X_test_enc_{dataset_name}.pkl')
        y_test = joblib.load(model_dir / f'y_test_enc_{dataset_name}.pkl')
        mlp = joblib.load(model_dir / f'mlp_trainer_{dataset_name}.pkl')
        mlp.bypass_preprocessing = True
        tree_a = joblib.load(model_dir / f'trepan_{dataset_name}.pkl')
        tree_b = joblib.load(model_dir / f'trepan_reloaded_{dataset_name}.pkl')
    except FileNotFoundError as e:
        print(f"❌ Faltan archivos para {dataset_name}: {e}")
        return None

    y_mlp = mlp.predict(X_test)
    result = {
        'dataset': dataset_name,
        'mlp_precision': accuracy_score(y_test, mlp.predict(X_test)),
        'trepan_fidelidad': accuracy_score(y_mlp, tree_a.predict(X_test)),
        'trepan_precision': accuracy_score(y_test, tree_a.predict(X_test)),
        'trepanreload_fidelidad': accuracy_score(y_mlp, tree_b.predict(X_test)),
        'trepanreload_precision': accuracy_score(y_test, tree_b.predict(X_test)),
    }
    return result

def evaluar_arboles_mejorados(dataset_name):
    """Calcula fidelidad y precisión de todos los árboles mejorados (archivos .pkl)"""
    model_dir = models_dir(dataset_name)
    mejora_modelos = mejora_dir(dataset_name) / 'modelos'
    if not mejora_modelos.exists():
        return []

    try:
        X_test = joblib.load(model_dir / f'X_test_enc_{dataset_name}.pkl')
        y_test = joblib.load(model_dir / f'y_test_enc_{dataset_name}.pkl')
        mlp = joblib.load(model_dir / f'mlp_trainer_{dataset_name}.pkl')
        mlp.bypass_preprocessing = True
        y_mlp = mlp.predict(X_test)
    except FileNotFoundError:
        return []

    resultados = []
    for fname in sorted(os.listdir(mejora_modelos)):
        if not fname.endswith('.pkl'):
            continue
        try:
            obj = joblib.load(mejora_modelos / fname)
            if hasattr(obj, 'predict'):
                tree = obj
            elif hasattr(obj, 'explainer_tree') and obj.explainer_tree is not None:
                tree = obj.explainer_tree
            else:
                continue
            y_pred = tree.predict(X_test)
            fid = accuracy_score(y_mlp, y_pred)
            acc = accuracy_score(y_test, y_pred)
            resultados.append({
                'dataset': dataset_name,
                'archivo': fname,
                'fidelidad': fid,
                'precision': acc
            })
        except Exception as e:
            print(f"   ❌ Error con {fname}: {e}")
    return resultados

# ---------------------------------------------------------------------------
# 1. Consistencia completa
# ---------------------------------------------------------------------------
print("📊 Consolidando indicadores de consistencia...")
dfs_consistencia = []
for ds in DATASETS:
    df = cargar_consistencia(ds)
    if df is not None:
        dfs_consistencia.append(df)

if dfs_consistencia:
    consistencia_df = pd.concat(dfs_consistencia, ignore_index=True)
    consistencia_df.to_csv(CONSOLIDAR_DIR / 'consistencia_completa.csv', index=False)
    print(f"   ✅ {len(consistencia_df)} filas guardadas en consistencia_completa.csv")
else:
    print("   ⚠️  No se encontraron datos de consistencia")
    consistencia_df = pd.DataFrame()

# ---------------------------------------------------------------------------
# 2. Árboles originales (fidelidad + precisión)
# ---------------------------------------------------------------------------
print("\n🌳 Evaluando árboles originales...")
originales = []
for ds in DATASETS:
    res = evaluar_arboles_originales(ds)
    if res:
        originales.append(res)

if originales:
    originales_df = pd.DataFrame(originales)
    originales_df.to_csv(CONSOLIDAR_DIR / 'arboles_originales.csv', index=False)
    print(f"   ✅ {len(originales_df)} filas guardadas en arboles_originales.csv")
else:
    print("   ⚠️  No se pudieron evaluar árboles originales")
    originales_df = pd.DataFrame()

# ---------------------------------------------------------------------------
# 3. Árboles mejorados (fidelidad + precisión)
# ---------------------------------------------------------------------------
print("\n🌳 Evaluando árboles mejorados...")
mejorados_list = []
for ds in DATASETS:
    res = evaluar_arboles_mejorados(ds)
    mejorados_list.extend(res)

if mejorados_list:
    mejorados_df = pd.DataFrame(mejorados_list)
    mejorados_df.to_csv(CONSOLIDAR_DIR / 'arboles_mejorados.csv', index=False)
    print(f"   ✅ {len(mejorados_df)} filas guardadas en arboles_mejorados.csv")
else:
    print("   ⚠️  No se pudieron evaluar árboles mejorados")
    mejorados_df = pd.DataFrame()

# ---------------------------------------------------------------------------
# 4. Mejora (fidelidad y precisión desde los resúmenes)
# ---------------------------------------------------------------------------
print("\n📈 Consolidando resultados de mejora...")
dfs_mejora = []
for ds in DATASETS:
    df = cargar_mejora_resumen(ds)
    if df is not None and not df.empty:
        dfs_mejora.append(df)

if dfs_mejora:
    mejora_df = pd.concat(dfs_mejora, ignore_index=True)
    mejora_df.to_csv(CONSOLIDAR_DIR / 'mejora_completa.csv', index=False)
    print(f"   ✅ {len(mejora_df)} filas guardadas en mejora_completa.csv")
else:
    print("   ⚠️  No se encontraron datos de mejora")
    mejora_df = pd.DataFrame()

# ---------------------------------------------------------------------------
# 5. Tabla maestra para Capítulo 3
# ---------------------------------------------------------------------------
print("\n🧩 Creando tabla maestra para Capítulo 3...")

# Partimos de los datos de entrenamiento (MLP acc, fidelidad, coincidencias)
filas_maestra = []
for ds in DATASETS:
    summary = cargar_training_summary(ds)
    if summary is None:
        continue
    # Buscar precisión de árboles originales
    orig_row = originales_df[originales_df['dataset'] == ds] if not originales_df.empty else pd.DataFrame()
    if not orig_row.empty:
        orig_row = orig_row.iloc[0]
    else:
        orig_row = {}

    fila = {
        'dataset': ds,
        'mlp_precision': summary.get('mlp_accuracy_test', None),
        'trepan_fidelidad_original': summary.get('trepan_fidelity', None),
        'trepanreload_fidelidad_original': summary.get('trepan_reloaded_fidelity', None),
        'trepan_precision_original': orig_row.get('trepan_precision', None),
        'trepanreload_precision_original': orig_row.get('trepanreload_precision', None),
        'coincidentes': summary.get('coincident_instances', None),
        'total_train': summary.get('total_train_instances', None),
        'seleccionadas': summary.get('selected_instances', None),
    }
    filas_maestra.append(fila)

maestra_df = pd.DataFrame(filas_maestra)

# Añadir indicadores de consistencia promediados por dataset y método CF
if not consistencia_df.empty:
    consistencia_prom = consistencia_df.groupby(['dataset', 'metodo_CF', 'tipo_sustituto']).agg(
        indicador_a=('indicador_a', 'mean'),
        indicador_b=('indicador_b', 'mean'),
        indicador_c=('indicador_c', 'mean'),
        cf_validos_mlp=('total_CFs_validos_MLP', 'mean'),
        cf_validos_sustituto=('total_CFs_validos_sustituto', 'mean'),
        cf_generados=('total_CFs_generados', 'mean'),
    ).reset_index()
    consistencia_prom.to_csv(CONSOLIDAR_DIR / 'consistencia_promedio.csv', index=False)

# Añadir mejores resultados de mejora por dataset, método CF y sustituto
if not mejora_df.empty and 'relative_improvement_mean' in mejora_df.columns:
    # Seleccionar las columnas relevantes
    cols_mejora = ['dataset', 'metodo_CF', 'tipo_sustituto', 'cf_types', 'include_type_C',
                   'confidence_threshold', 'cf_weight_ratio', 'max_cfs', 'use_frontier_weight',
                   'fidelity_improved_mean', 'accuracy_improved_mean', 'relative_improvement_mean']
    cols_presentes = [c for c in cols_mejora if c in mejora_df.columns]
    mejora_para_maestra = mejora_df[cols_presentes]
    mejora_para_maestra.to_csv(CONSOLIDAR_DIR / 'mejora_para_maestra.csv', index=False)

# Guardar tabla maestra
maestra_df.to_csv(CONSOLIDAR_DIR / 'tabla_resumen.csv', index=False)
print(f"   ✅ {len(maestra_df)} filas guardadas en tabla_resumen.csv")

# ---------------------------------------------------------------------------
# Resumen final
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("📁 CONSOLIDACIÓN COMPLETA")
print("="*70)
print(f"Los archivos consolidados están en la carpeta '{CONSOLIDAR_DIR}':")
for f in os.listdir(CONSOLIDAR_DIR):
    print(f"   📄 {f}")
print("\n✅ Listo para usar en el Capítulo 3.")