#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_improve.py – Mejora de sustitutos con contrafactuales.

- DEFAULT_PARAMS alineados con improve_surrogate.py.
- Guardado de árboles mejorados y estadísticas individuales por dataset.
- guarda métricas de mejora por dataset en su carpeta 'mejora/metricas/'.
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from counterfactuals._bootstrap import setup
setup()

from core.trepan_original import TrepanOriginalExtractor
from core.trepan_reloaded_extractor import TrepanReloadedExtractor
from improve_surrogate import improve_surrogate, evaluate_improvement
from counterfactuals.dataset_config import get_dataset_config, MLP_ACCURACY, ALL_DATASETS
from counterfactuals._paths import models_dir, results_dir, mejora_dir, RESULTADOS_MEJORA_DIR

BASE_SEED = 42
np.random.seed(BASE_SEED)

# Parámetros de respaldo alineados con improve_surrogate.py
DEFAULT_PARAMS = {
    'max_cfs': 10, 'cf_types': 'A+B',
    'include_type_C': True, 'use_frontier_weight': True,
    'confidence_threshold': 0.6, 'cf_weight_ratio': 0.05
}

def cargar_artefactos(dataset_name):
    model_dir = models_dir(dataset_name)
    state      = joblib.load(model_dir / f'pipeline_state_{dataset_name}.pkl')
    X_train_enc = joblib.load(model_dir / f'X_train_enc_{dataset_name}.pkl')
    y_train_enc = joblib.load(model_dir / f'y_train_enc_{dataset_name}.pkl')
    X_test_enc  = joblib.load(model_dir / f'X_test_enc_{dataset_name}.pkl')
    y_test_enc  = joblib.load(model_dir / f'y_test_enc_{dataset_name}.pkl')
    mlp_trainer = joblib.load(model_dir / f'mlp_trainer_{dataset_name}.pkl')
    mlp_trainer.bypass_preprocessing = True
    mlp_model = mlp_trainer
    tree_a = joblib.load(model_dir / f'trepan_{dataset_name}.pkl')
    tree_b = joblib.load(model_dir / f'trepan_reloaded_{dataset_name}.pkl')
    transformed_feature_names = state['transformed_feature_names']
    class_names = list(state['class_labels'].values())
    return (mlp_model, X_train_enc, y_train_enc, X_test_enc, y_test_enc,
            tree_a, tree_b, transformed_feature_names, class_names)

def cargar_cfs(dataset_name, method='clear'):
    """Carga CFs desde CSV. Si existe la columna 'cf_vector', la usa directamente;
    en caso contrario, hace fallback al parseo de la columna 'cf'."""
    results_dir_path = results_dir(dataset_name)
    filename = 'clear_cfs_full.csv' if method == 'clear' else 'cogs_cfs_full.csv'
    path = results_dir_path / filename
    if not path.exists():
        return []
    df = pd.read_csv(path)
    cfs = []
    if 'cf_vector' in df.columns:
        for _, row in df.iterrows():
            vec_str = row['cf_vector']
            if isinstance(vec_str, str):
                vec_str = vec_str.replace('[', '').replace(']', '').replace('\n', ' ').strip()
                cf_array = np.fromstring(vec_str, sep=' ')
            else:
                cf_array = np.array(vec_str)
            cfs.append({'cf': cf_array, 'original_class': int(row['original_class'])})
        return cfs
    # Fallback
    for _, row in df.iterrows():
        cf_str = row['cf'].replace('[', '').replace(']', '').replace('\n', ' ').strip()
        cf_array = np.fromstring(cf_str, sep=' ')
        cfs.append({'cf': cf_array, 'original_class': int(row['original_class'])})
    return cfs

def run_experiment(seed, datasets_to_run):
    np.random.seed(seed)
    all_entries = []   # Para el consolidado global
    for ds in datasets_to_run:
        logging.info(f"\n=== Dataset: {ds} (seed={seed}) ===")
        try:
            mlp, X_train, y_train, X_test, y_test, tree_a, tree_b, feat_names, class_names = cargar_artefactos(ds)
        except FileNotFoundError as e:
            logging.error(f"Faltan archivos: {e}")
            continue
        mlp_acc = MLP_ACCURACY.get(ds, 0.8)

        # Precalcular vecinos por clase (percentil 95, puede modificarse para hacerlo menos o mas robusto)
        class_nn_dict = {}
        unique_classes = np.unique(y_train)
        for cls in unique_classes:
            X_cls = X_train[y_train == cls]
            if len(X_cls) > 1:
                nn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X_cls)
                dists, _ = nn.kneighbors(X_cls)
                threshold = np.percentile(dists[:, 1], 95)
                class_nn_dict[cls] = (nn, threshold)
            else:
                class_nn_dict[cls] = (None, np.inf)

        clear_cfs = cargar_cfs(ds, 'clear')
        cogs_cfs  = cargar_cfs(ds, 'cogs')

        combinaciones = [
            ('CLEAR', clear_cfs, 'Trepan', 'extract_tree'),
            ('CLEAR', clear_cfs, 'TrepanReload', 'extract_tree_with_ontology'),
            ('COGS',  cogs_cfs,  'Trepan', 'extract_tree'),
            ('COGS',  cogs_cfs,  'TrepanReload', 'extract_tree_with_ontology')
        ]

        dataset_entries = []   #  para guardar las filas de este dataset

        for metodo_cf, lista_cfs, tipo_sustituto, metodo_extract in combinaciones:
            if not lista_cfs: continue

            # Grid reducido
            param_grid = list(ParameterGrid({
                'max_cfs': [5, 10],
                'cf_types': ['A+B', 'all'],
                'include_type_C': [False, True],
                'use_frontier_weight': [False, True]
            }))

            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            best_score = -1
            best_params = None
            for params in param_grid:
                fold_scores = []
                for train_idx, val_idx in skf.split(X_train, y_train):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]

                    fold_ext = TrepanOriginalExtractor(random_state=seed) if metodo_extract == 'extract_tree' else TrepanReloadedExtractor()
                    if metodo_extract == 'extract_tree':
                        fold_ext.extract_tree(mlp, X_tr, y_tr, sample_size=2000,
                                             feature_names=feat_names, class_names=class_names)
                    else:
                        fold_ext.extract_tree_with_ontology(mlp, X_tr, y_tr,
                                                           feature_names=feat_names, class_names=class_names,
                                                           sample_size=2000)
                    _, _ = improve_surrogate(
                        mlp_model=mlp, X_train=X_tr, y_train=y_tr,
                        tree_extractor=fold_ext, cf_list=lista_cfs,
                        feature_names=feat_names, class_names=class_names,
                        cf_types=params['cf_types'],
                        include_type_C=params['include_type_C'],
                        confidence_threshold=0.6,
                        max_cfs=params['max_cfs'],
                        max_cf_ratio=0.2,
                        extract_method=metodo_extract,
                        cf_weight_ratio=0.05,
                        mlp_accuracy=mlp_acc, max_cf_weight=20.0,
                        boost_B=None, weight_C=0.3, anchor_ratio=0.2,
                        anchor_mix_ratio=0.0, use_density_as_weight=True,
                        class_density_percentile=95, class_nn_dict=class_nn_dict,
                        sample_size=None, use_frontier_weight=params['use_frontier_weight'],
                        feature_intervals=None, seed=seed
                    )
                    temp_tree = fold_ext.explainer_tree
                    fold_fidelity = accuracy_score(mlp.predict(X_val), temp_tree.predict(X_val))
                    fold_scores.append(fold_fidelity)
                avg_fidelity = np.mean(fold_scores)
                if avg_fidelity > best_score:
                    best_score = avg_fidelity
                    best_params = params

            if best_params is None:
                logging.warning("No se halló configuración válida. Usando parámetros por defecto.")
                best_params = DEFAULT_PARAMS

            logging.info(f"   Mejor config: {best_params}")

            if tipo_sustituto == 'Trepan':
                extractor = TrepanOriginalExtractor(random_state=seed)
                extractor.explainer_tree = tree_a
            else:
                extractor = TrepanReloadedExtractor()
                extractor.explainer_tree = tree_b

            nuevo_arbol, stats = improve_surrogate(
                mlp_model=mlp, X_train=X_train, y_train=y_train,
                tree_extractor=extractor, cf_list=lista_cfs,
                feature_names=feat_names, class_names=class_names,
                cf_types=best_params['cf_types'],
                include_type_C=best_params['include_type_C'],
                confidence_threshold=0.6,
                max_cfs=best_params['max_cfs'],
                max_cf_ratio=0.2,
                extract_method=metodo_extract,
                cf_weight_ratio=0.05,
                mlp_accuracy=mlp_acc, max_cf_weight=20.0,
                boost_B=None, weight_C=0.3, anchor_ratio=0.2,
                anchor_mix_ratio=0.0, use_density_as_weight=True,
                class_density_percentile=95, class_nn_dict=class_nn_dict,
                sample_size=None, use_frontier_weight=best_params['use_frontier_weight'],
                feature_intervals=None, seed=seed
            )

            metricas = evaluate_improvement(mlp, X_test, y_test,
                                            tree_a if tipo_sustituto=='Trepan' else tree_b, nuevo_arbol)

            # Guardado de árbol y stats 
            modelo_mejora_dir = mejora_dir(ds) / 'modelos'
            stats_dir = mejora_dir(ds) / 'stats'
            modelo_mejora_dir.mkdir(parents=True, exist_ok=True)
            stats_dir.mkdir(parents=True, exist_ok=True)

            modo = f"cf_{best_params['cf_types']}_C{best_params['include_type_C']}_" \
                   f"F{best_params['use_frontier_weight']}_max{best_params['max_cfs']}"

            nombre_arbol = f"{tipo_sustituto.lower()}_{metodo_cf.lower()}_seed{seed}_improved.pkl"
            joblib.dump(nuevo_arbol, modelo_mejora_dir / nombre_arbol)

            nombre_stats = f"{tipo_sustituto.lower()}_{metodo_cf.lower()}_seed{seed}_stats.json"
            with open(stats_dir / nombre_stats, 'w') as f:
                json.dump(stats, f, indent=2)

            entrada = {
                'seed': seed, 'dataset': ds, 'metodo_CF': metodo_cf,
                'tipo_sustituto': tipo_sustituto,
                'cf_types': best_params['cf_types'],
                'include_type_C': best_params['include_type_C'],
                'confidence_threshold': 0.6,
                'cf_weight_ratio': 0.05,
                'max_cfs': best_params['max_cfs'],
                'use_frontier_weight': best_params['use_frontier_weight'],
                'fidelity_original': metricas['fidelity_original'],
                'fidelity_improved': metricas['fidelity_improved'],
                'accuracy_original': metricas['accuracy_original'],
                'accuracy_improved': metricas['accuracy_improved'],
                'relative_improvement': metricas['relative_improvement'],
                'used': stats['used'],
                'type_C_used': stats.get('type_C_used', 0),
                'density_discarded': stats.get('density_discarded', 0),
                'frontier_factor_mean': stats.get('frontier_factor_mean', 0.0),
            }
            all_entries.append(entrada)
            dataset_entries.append(entrada)   # añadir a la lista del dataset

        # ========== Guardar resultados específicos del dataset ==========
        if dataset_entries:
            df_ds = pd.DataFrame(dataset_entries)
            # Carpeta de métricas dentro de la mejora de este dataset
            metricas_dir = mejora_dir(ds) / 'metricas'
            metricas_dir.mkdir(parents=True, exist_ok=True)
            df_ds.to_csv(metricas_dir / 'mejora_metrics.csv', index=False)
            # Guardar resumen agrupado por combinación (media)
            group_cols = ['metodo_CF', 'tipo_sustituto', 'cf_types', 'include_type_C',
                          'confidence_threshold', 'cf_weight_ratio', 'max_cfs', 'use_frontier_weight']
            # Asegurar que todas las columnas existen
            group_cols = [c for c in group_cols if c in df_ds.columns]
            agg_funcs = {
                'fidelity_original': ['mean', 'std'],
                'fidelity_improved': ['mean', 'std'],
                'accuracy_original': ['mean', 'std'],
                'accuracy_improved': ['mean', 'std'],
                'relative_improvement': ['mean', 'std'],
                'used': 'mean',
                'type_C_used': 'mean',
                'density_discarded': 'mean',
                'frontier_factor_mean': 'mean',
            }
            agg_funcs = {k: v for k, v in agg_funcs.items() if k in df_ds.columns}
            df_resumen = df_ds.groupby(group_cols).agg(agg_funcs).reset_index()
            df_resumen.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_resumen.columns]
            df_resumen.to_csv(metricas_dir / 'mejora_resumen.csv', index=False)
            logging.info(f"   ✅ Guardadas métricas de mejora para {ds} en {metricas_dir}")
        # ======================================================================

    return all_entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', default=None)
    args = parser.parse_args()
    if args.dataset:
        datasets_to_run = [args.dataset]
    else:
        datasets_to_run = ALL_DATASETS

    seeds = [42] #en estudios posteriores comprobar resultados para mas semillas
    all_entries = []
    for seed in seeds:
        logging.info(f"\n🚀 Iniciando experimento con semilla {seed}")
        entries = run_experiment(seed, datasets_to_run)
        all_entries.extend(entries)

    if all_entries:
        df = pd.DataFrame(all_entries)
        # Consolidado global 
        RESULTADOS_MEJORA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTADOS_MEJORA_DIR / 'detalle_mejora.csv', index=False)

        group_cols = ['dataset', 'metodo_CF', 'tipo_sustituto', 'cf_types', 'include_type_C',
                      'confidence_threshold', 'cf_weight_ratio', 'max_cfs', 'use_frontier_weight']
        group_cols = [c for c in group_cols if c in df.columns]
        agg_funcs = {
            'fidelity_original': ['mean', 'std'], 'fidelity_improved': ['mean', 'std'],
            'accuracy_original': ['mean', 'std'], 'accuracy_improved': ['mean', 'std'],
            'relative_improvement': ['mean', 'std'], 'used': 'mean',
            'type_C_used': 'mean', 'density_discarded': 'mean', 'frontier_factor_mean': 'mean',
        }
        agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
        stats_df = df.groupby(group_cols).agg(agg_funcs)
        stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns]
        stats_df = stats_df.reset_index()
        stats_df.to_csv(RESULTADOS_MEJORA_DIR / 'mejora_resumen.csv', index=False)
        logging.info(f"\n📊 Resumen de mejora (global) guardado en {RESULTADOS_MEJORA_DIR}/")
    else:
        logging.warning("No se generaron resultados.")

if __name__ == "__main__":
    main()