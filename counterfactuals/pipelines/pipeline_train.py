#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_train.py – Primera fase: entrena MLP y extrae árboles sustitutos.
Guarda todos los artefactos necesarios para la segunda fase.
selecciona las instancias

"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from counterfactuals._bootstrap import setup
setup()

from counterfactuals.cf_mlp_trainer import MLPTrainer
from core.trepan_original import TrepanOriginalExtractor
from core.trepan_reloaded_extractor import TrepanReloadedExtractor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from counterfactuals.dataset_config import get_dataset_config, ALL_DATASETS
from counterfactuals._paths import experiment_dir, models_dir, ensure_dirs

BASE_SEED = 42  # mesma semente base de pipeline_improve.py

# ============================================================================
# Configuración de Datasets — ver counterfactuals/dataset_config.py
# ============================================================================

def extract_fidelity_from_report(report):
    import re
    patterns = (
        r'Sobre dados de avalia[aç][aã]o:\s*[\d.]+\s*\((\d+\.\d+)%\)',
        r'Sobre dados reais:\s*(\d+\.\d+)%',
        r'Sobre datos reales:\s*(\d+\.\d+)%',
    )
    for pattern in patterns:
        match = re.search(pattern, report)
        if match:
            return float(match.group(1)) / 100.0
    for line in report.split('\n'):
        if 'Sobre dados reais:' in line or 'Sobre datos reales:' in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                return float(match.group(1)) / 100.0
    return None

def run_training_pipeline(dataset_name):
    timers = {}
    start_total = time.perf_counter()
    config = get_dataset_config(dataset_name)

    ensure_dirs(dataset_name)
    experiment_dir_path = experiment_dir(dataset_name)
    models_dir_path = models_dir(dataset_name)

    print(f"\n{'='*70}")
    print(f"▶️  ENTRENAMIENTO: {dataset_name.upper()}")
    print(f"{'='*70}")

    # ---------- Paso 1: Cargar datos ----------
    print("\n📂 Paso 1: Cargando dataset")
    start = time.perf_counter()
    df = pd.read_csv(config['file'])
    df.columns = df.columns.str.strip()
    target_col = config['target']
    if target_col not in df.columns:
        for col in df.columns:
            if 'class' in col.lower() or 'diagnosis' in col.lower() or 'outcome' in col.lower():
                target_col = col
                break
    X_orig = df.drop(columns=[target_col]).copy()
    y_orig = df[target_col].copy()

    # Renombrar categóricas en German Credit
    if dataset_name == 'german_credit':
        rename_map = {
            'attr_1': 'A1_attr_1', 'attr_3': 'A3_attr_3', 'attr_4': 'A4_attr_4',
            'attr_6': 'A6_attr_6', 'attr_7': 'A7_attr_7', 'attr_9': 'A9_attr_9',
            'attr_10': 'A10_attr_10', 'attr_12': 'A12_attr_12', 'attr_14': 'A14_attr_14',
            'attr_15': 'A15_attr_15', 'attr_17': 'A17_attr_17', 'attr_19': 'A19_attr_19',
            'attr_20': 'A20_attr_20'
        }
        X_orig.rename(columns=rename_map, inplace=True)
        config['categorical_features'] = list(rename_map.values())

    feature_names = X_orig.columns.tolist()
    class_names = list(config['class_labels'].values())

    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_orig, y_orig, test_size=0.3, random_state=42, stratify=y_orig
    )
    timers['load_data'] = time.perf_counter() - start
    print(f"  ✓ Cargado en {timers['load_data']:.2f}s")
    print(f"  Train: {X_train_orig.shape[0]}, Test: {X_test_orig.shape[0]}")

    # ---------- Paso 2: Entrenar MLP ----------
    print("\n🧠 Paso 2: Entrenando MLP")
    start = time.perf_counter()
    mlp_trainer = MLPTrainer()
    mlp_model, X_enc, y_enc = mlp_trainer.train(
        X_train_orig, y_train_orig,
        categorical_features=config.get('categorical_features', []),
        mlp_params=config['mlp_params'],
        tune=config.get('tune_mlp', False),
        balance_method=config.get('balance_method', 'oversample'),
        use_minmax=config.get('use_minmax_scaler', False),
        calibrate_threshold=config.get('calibrate_threshold', True)
    )

    transformed_feature_names = mlp_trainer.get_transformed_feature_names()
    if config.get('categorical_features'):
        category_prefix = [f"{feat}_" for feat in config['categorical_features']]
    else:
        category_prefix = []
    config['category_prefix'] = category_prefix

    X_test_enc = mlp_trainer.transform(X_test_orig)

    # Predicciones con umbral calibrado usando los datos originales
    y_train_pred = mlp_trainer.predict(X_train_orig)
    y_test_pred = mlp_trainer.predict(X_test_orig)
    acc_train = accuracy_score(y_enc, y_train_pred)
    acc_test = accuracy_score(mlp_trainer.label_encoder.transform(y_test_orig), y_test_pred)

    # Guardar modelos y datos codificados
    joblib.dump(mlp_model, models_dir_path / f'mlp_{dataset_name}.pkl')
    joblib.dump(mlp_trainer, models_dir_path / f'mlp_trainer_{dataset_name}.pkl')
    joblib.dump(X_enc, models_dir_path / f'X_train_enc_{dataset_name}.pkl')
    joblib.dump(y_enc, models_dir_path / f'y_train_enc_{dataset_name}.pkl')
    joblib.dump(X_test_enc, models_dir_path / f'X_test_enc_{dataset_name}.pkl')
    joblib.dump(mlp_trainer.label_encoder.transform(y_test_orig),
                models_dir_path / f'y_test_enc_{dataset_name}.pkl')

    timers['train_mlp'] = time.perf_counter() - start
    print(f"  ✓ MLP entrenado en {timers['train_mlp']:.2f}s")
    print(f"  Precisión train: {acc_train:.3f}, test: {acc_test:.3f}")

    # DataFrame codificado (para árboles y para CFs)
    X_train_enc_df = pd.DataFrame(X_enc, columns=transformed_feature_names)

    # ---------- Paso 3: Extraer Trepan ----------
    print("\n🌳 Paso 3: Extrayendo Trepan (sustituto A)")
    start = time.perf_counter()
    extractor_a = TrepanOriginalExtractor(random_state=BASE_SEED)
    report_a = extractor_a.extract_tree(
        mlp_model, X_enc, y_enc,
        sample_size=config.get('tree_sample_size', 2000),          # <-- configurable
        feature_names=transformed_feature_names,
        class_names=class_names
    )
    tree_a = extractor_a.explainer_tree
    fidelity_a = extract_fidelity_from_report(report_a)
    joblib.dump(tree_a, models_dir_path / f'trepan_{dataset_name}.pkl')
    timers['extract_trepan'] = time.perf_counter() - start
    print(f"  ✓ Trepan extraído en {timers['extract_trepan']:.2f}s, Fidelidad: {(fidelity_a or 0)*100:.1f}%")

    # ---------- Paso 4: Extraer Trepan Reloaded ----------
    print("\n🌳 Paso 4: Extrayendo Trepan Reloaded (sustituto B)")
    start = time.perf_counter()
    extractor_b = TrepanReloadedExtractor()
    report_b = extractor_b.extract_tree_with_ontology(
        mlp_model, X_enc, y_enc,
        feature_names=transformed_feature_names,
        class_names=class_names,
        sample_size=config.get('tree_sample_size', 2000)           # <-- configurable
    )
    tree_b = extractor_b.explainer_tree
    fidelity_b = extract_fidelity_from_report(report_b)
    joblib.dump(tree_b, models_dir_path / f'trepan_reloaded_{dataset_name}.pkl')
    timers['extract_trepan_reloaded'] = time.perf_counter() - start
    print(f"  ✓ Trepan Reloaded extraído en {timers['extract_trepan_reloaded']:.2f}s, Fidelidad: {(fidelity_b or 0)*100:.1f}%")

    # ---------- Paso 5: Seleccionar 33% de instancias (ALEATORIO ESTRATIFICADO POR CLASE) ----------
    print("\n🎯 Paso 5: Seleccionando 33% de instancias (aleatorio estratificado por clase)")
    start = time.perf_counter()

    total_train = X_train_orig.shape[0]
    target_instances = int(0.33 * total_train)

    # Obtener predicciones del MLP para las instancias de entrenamiento (se sigue calculando y guardando)
    y_mlp_train = mlp_trainer.predict(X_train_orig)
    y_tree_a_train = tree_a.predict(X_enc)
    y_tree_b_train = tree_b.predict(X_enc)

    # Selección estratificada por clase, sin filtrar por coincidencia
    np.random.seed(42)
    unique_classes = np.unique(y_train_orig)
    selected_indices = []
    for cls in unique_classes:
        cls_mask = (y_train_orig == cls)
        cls_indices = np.where(cls_mask)[0]
        n_select = int(0.33 * len(cls_indices))
        if n_select > 0:
            chosen = np.random.choice(cls_indices, size=n_select, replace=False)
            selected_indices.append(chosen)
    if selected_indices:
        selected_indices = np.concatenate(selected_indices)
        np.random.shuffle(selected_indices)
    else:
        selected_indices = np.array([], dtype=int)

    X_selected_enc_df = X_train_enc_df.iloc[selected_indices].copy()
    y_selected_mlp = y_mlp_train[selected_indices]

    # Estadísticas de coincidencia (solo informativas)
    coincide = (y_mlp_train == y_tree_a_train) & (y_mlp_train == y_tree_b_train)
    indices_coincidentes = np.where(coincide)[0]
    num_coincidentes = len(indices_coincidentes)
    coincidentes_en_seleccion = len(np.intersect1d(selected_indices, indices_coincidentes))

    print(f"  Total train: {total_train}, Seleccionadas (33% del total): {len(selected_indices)}")
    print(f"  De ellas, {coincidentes_en_seleccion} son coincidentes (los tres modelos coinciden) "
          f"de un total de {num_coincidentes} coincidentes en todo el train.")

    # Calcular intervalos de características (para COGS)
    feature_intervals = []
    for col in transformed_feature_names:
        min_val = X_train_enc_df[col].min()
        max_val = X_train_enc_df[col].max()
        feature_intervals.append((min_val, max_val))
    feature_intervals = np.array(feature_intervals, dtype=object)

    timers['select_instances'] = time.perf_counter() - start
    print(f"  ✓ Seleccionadas en {timers['select_instances']:.2f}s")

    # ---------- Guardar estado para la segunda fase ----------
    state = {
        'selected_indices': selected_indices,
        'X_selected_enc': X_selected_enc_df,
        'y_selected_mlp': y_selected_mlp,
        'transformed_feature_names': transformed_feature_names,
        'feature_intervals': feature_intervals,
        'category_prefix': config.get('category_prefix', []),
        'class_labels': config['class_labels'],
        'is_multiclass': config['is_multiclass'],
    }
    joblib.dump(state, models_dir_path / f'pipeline_state_{dataset_name}.pkl')
    print("  ✓ Estado guardado para la fase de contrafactuales.")

    # ---------- Resumen de entrenamiento ----------
    timers['total'] = time.perf_counter() - start_total
    summary = {
        'dataset': dataset_name,
        'mlp_accuracy_test': float(acc_test),
        'trepan_fidelity': float(fidelity_a) if fidelity_a is not None else None,
        'trepan_reloaded_fidelity': float(fidelity_b) if fidelity_b is not None else None,
        'coincident_instances': int(num_coincidentes),
        'total_train_instances': int(total_train),
        'selected_instances': int(len(selected_indices)),
        'timers': {k: float(v) for k, v in timers.items()}
    }
    with open(experiment_dir_path / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n⏱️  Tiempos de entrenamiento:")
    for key, value in timers.items():
        print(f"  {key}: {value:.2f}s")
    print(f"\n✅ Entrenamiento de {dataset_name} completado.\n")
    return summary

if __name__ == "__main__":
    if len(sys.argv) > 1:
        datasets_to_run = [sys.argv[1]]
        if sys.argv[1] not in ALL_DATASETS:
            print(f"❌ Dataset '{sys.argv[1]}' no reconocido. Opciones: {ALL_DATASETS}")
            sys.exit(1)
    else:
        datasets_to_run = ALL_DATASETS

    for ds in datasets_to_run:
        try:
            run_training_pipeline(ds)
        except Exception as e:
            print(f"❌ Error en {ds}: {e}")
            import traceback
            traceback.print_exc()