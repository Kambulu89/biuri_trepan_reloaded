#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_counterfactuals.py – Segunda fase: genera contrafactuales con CLEAR y COGS,
calcula indicadores de consistencia coincidentes, divergentes y brecha (los que no son cf en el sustituto), guarda CFs enriquecidos
y consolida resultados.


- COGS devuelve hasta 3 CFs usando genes y fitnesses de la población final.
- CLEAR guarda el vector codificado completo (cf_vector) para evitar reconstrucción frágil.
- Manejo de excepciones en COGS.
- Logging estructurado.
- Parámetros de CLEAR optimizados por dataset (max_predictors, neighbourhood_algorithm='Unbalanced').
- German Credit con clear_num_samples=600, clear_regression_sample_size=60 para velocidad.(la alta dimensionalidad produce tiempos extremadamente amplios en generacion de cf con clear)
"""

import sys, os, time, json
import numpy as np
import pandas as pd
import joblib
import shutil
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

from counterfactuals._bootstrap import setup
setup()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

from clear import CLEAR_settings
try:
    from clear import CLEAR_regression, CLEAR_perturbations, CLEAR_sensitivity_files
    _CLEAR_IMPORT_ERROR = None
except ImportError as exc:
    # COGS e o motor interno continuam disponíveis sem TensorFlow/CLEAR.
    # A falha é apresentada apenas quando CLEAR é efetivamente solicitado.
    CLEAR_regression = CLEAR_perturbations = CLEAR_sensitivity_files = None
    _CLEAR_IMPORT_ERROR = exc
from cogs.evolution import Evolution
from cogs.fitness import gower_fitness_function

from counterfactuals.dataset_config import get_dataset_config, ALL_DATASETS
from counterfactuals._paths import (
    models_dir, results_dir, clear_output_dir,
    RESULTADOS_CONSOLIDADOS_DIR, ensure_dirs,
)

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
def run_cogs_for_single_instance(args):
    """
    Ejecuta COGS 
    Extrae los mejores individuos de la población final usando genes y fitnesses.
    """
    idx, x_encoded, original_class, mlp_model, feature_intervals, cat_indices, class_labels, is_multiclass = args
    try:
        probas = mlp_model.predict_proba(x_encoded.reshape(1, -1))[0]
        sorted_classes = np.argsort(probas)[::-1]
        desired_class = None
        for cls in sorted_classes:
            if cls != original_class:
                desired_class = cls
                break
        if desired_class is None:
            return None

        fitness_kwargs = {
            'blackbox': mlp_model,
            'desired_class': desired_class,
            'apply_fixes': False
        }

        evolution = Evolution(
            x=x_encoded,
            fitness_function=gower_fitness_function,
            fitness_function_kwargs=fitness_kwargs,
            feature_intervals=feature_intervals,
            indices_categorical_features=cat_indices if cat_indices else [],
            plausibility_constraints=None,
            evolution_type='classic',
            population_size=300,
            n_generations=120,
            mutation_probability='inv_mutable_genotype_length',
            num_features_mutation_strength=0.25,
            init_temperature=0.8,
            selection_name='tournament_2',
            noisy_evaluations=False,
            verbose=False
        )
        evolution.run()

        # Obtener los mejores individuos de la población final
        pop = evolution.population
        best_indices = np.argsort(pop.fitnesses)[::-1]
        valid_cfs = []
        for i in range(min(3, len(best_indices))):
            gene = pop.genes[best_indices[i]]
            cf_arr = np.clip(gene, feature_intervals[:,0], feature_intervals[:,1])
            pred_cf = mlp_model.predict(cf_arr.reshape(1, -1))[0]
            if pred_cf != original_class:
                valid_cfs.append((pop.fitnesses[best_indices[i]], cf_arr, pred_cf))

        if not valid_cfs:
            return None

        valid_cfs.sort(key=lambda x: x[0], reverse=True)
        resultados = []
        for fit, cf_arr, pred_cf in valid_cfs:
            resultados.append({
                'idx': idx,
                'cf': cf_arr,
                'original_class': original_class,
                'desired_class': desired_class
            })
        return resultados if len(resultados) > 1 else resultados[0]
    except Exception as e:
        logging.warning(f"COGS falló para instancia {idx}: {e}")
        return None


def run_clear_cfs(X_train_enc_df, X_selected_enc_df, mlp_model, config,
                  transformed_feature_names, clear_output_dir, feature_intervals=None,
                  optimal_threshold=0.5):
    """
    Genera CFs con CLEAR usando los datos codificados.
guarda el vector codificado completo (cf_vector) en el DataFrame de salida.
    """
    if _CLEAR_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CLEAR indisponível: instale/valide TensorFlow e statsmodels "
            f"compatíveis. Causa: {_CLEAR_IMPORT_ERROR}"
        ) from _CLEAR_IMPORT_ERROR
    from pathlib import Path

    clear_path = Path(str(clear_output_dir).rstrip('/\\'))
    clear_path.mkdir(parents=True, exist_ok=True)
    clear_output_dir = str(clear_path.resolve())

    X_instances_df = X_selected_enc_df.reset_index(drop=True)
    num_obs = len(X_instances_df)
    CLEAR_settings.first_obs = 0
    CLEAR_settings.last_obs = num_obs - 1

    numeric_features = [c for c in transformed_feature_names if c.startswith('num__')]
    # GUI/core: features já codificadas sem prefixo ColumnTransformer
    if not numeric_features:
        numeric_features = [
            c for c in transformed_feature_names
            if not str(c).startswith('cat__') and not str(c).startswith('onto_')
        ]
    CLEAR_settings.numeric_features = numeric_features

    cat_prefixes = []
    for c in transformed_feature_names:
        if c.startswith('cat__'):
            prefix = '_'.join(c.split('_')[:-1])
            if prefix not in cat_prefixes:
                cat_prefixes.append(prefix)
    CLEAR_settings.category_prefix = cat_prefixes

    CLEAR_settings.sample_model = config.get('dataset_name', 'UNKNOWN').upper()
    CLEAR_settings.class_labels = config['class_labels']
    CLEAR_settings.CLEAR_path = clear_output_dir + os.sep
    CLEAR_settings.num_samples = config.get('clear_num_samples', 800)
    CLEAR_settings.regression_sample_size = config.get('clear_regression_sample_size', 50)
    CLEAR_settings.max_predictors = config.get('clear_max_predictors', 1)
    CLEAR_settings.regression_type = 'logistic'
    CLEAR_settings.score_type = 'aic'
    CLEAR_settings.neighbourhood_algorithm = 'Unbalanced'
    CLEAR_settings.apply_counterfactual_weights = True
    CLEAR_settings.counterfactual_weight = 9
    CLEAR_settings.generate_regression_files = False
    CLEAR_settings.centering = True
    CLEAR_settings.no_polynomimals = False
    CLEAR_settings.interactions_only = False
    CLEAR_settings.include_all_numerics = False
    CLEAR_settings.include_features = False
    CLEAR_settings.binary_decision_boundary = optimal_threshold

    if config['is_multiclass']:
        class_indices = range(len(config['class_labels']))
    else:
        class_indices = [1]

    all_cfs = []
    for multi_index in class_indices:
        class_name = list(config['class_labels'].values())[multi_index]
        logging.info(f"   → Clase objetivo: {class_name} (índice {multi_index})")
        CLEAR_settings.multi_class_focus = class_name

        CLEAR_sensitivity_files.Create_sensitivity(X_train_enc_df, X_instances_df, mlp_model)

        expected_file = os.path.join(clear_output_dir, f'numSensitivity_m{multi_index}.csv')
        if os.path.exists(expected_file):
            shutil.copyfile(expected_file, os.path.join(clear_output_dir, 'numSensitivity.csv'))
            logging.info("     ✅ Archivo de sensibilidad copiado")

        explainer = CLEAR_regression.Create_Synthetic_Data(X_train_enc_df, mlp_model, neighbour_seed=1)
        results_df, _, _, boundary_df = CLEAR_regression.Run_Regressions(
            X_instances_df, explainer, multi_index=multi_index
        )
        nncomp_df, _ = CLEAR_perturbations.Calculate_Perturbations(
            explainer, results_df, boundary_df, multi_index=multi_index
        )
        if not nncomp_df.empty:
            nncomp_df['desired_class'] = multi_index
            # Construir la columna con el vector codificado completo
            cf_vectors = []
            for i in range(len(nncomp_df)):
                obs = int(nncomp_df.iloc[i]['observation'])
                feat = nncomp_df.iloc[i]['feature']
                if feat not in transformed_feature_names:
                    cf_vectors.append(np.nan * np.zeros(len(transformed_feature_names)))
                    continue
                original_vec = X_instances_df.iloc[obs].values.copy()
                feat_idx = list(transformed_feature_names).index(feat)
                if feat.startswith('cat__'):
                    group_prefix = '_'.join(feat.split('_')[:-1])
                    group_cols = [c for c in transformed_feature_names if c.startswith(group_prefix)]
                    for col in group_cols:
                        original_vec[list(transformed_feature_names).index(col)] = 0
                    original_vec[feat_idx] = 1
                else:
                    new_val = nncomp_df.iloc[i].get('estPerturbedFeatValue', np.nan)
                    if not pd.isna(new_val):
                        low, high = feature_intervals[feat_idx] if feature_intervals is not None else (-np.inf, np.inf)
                        original_vec[feat_idx] = np.clip(new_val, low, high)
                cf_vectors.append(original_vec)
            nncomp_df['cf_vector'] = cf_vectors
            all_cfs.append(nncomp_df)
            logging.info(f"     CFs generados: {len(nncomp_df)}")
    return pd.concat(all_cfs, ignore_index=True) if all_cfs else pd.DataFrame()


def run_cf_pipeline(dataset_name):
    timers = {}
    start_total = time.perf_counter()
    config = get_dataset_config(dataset_name)

    ensure_dirs(dataset_name)
    models_dir_path = models_dir(dataset_name)
    results_dir_path = results_dir(dataset_name)
    clear_output_dir_path = clear_output_dir(dataset_name)
    clear_output_dir_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n{'='*70}")
    logging.info(f"▶️  CONTRAPACTUALES: {dataset_name.upper()}")
    logging.info(f"{'='*70}")

    logging.info("📦 Cargando modelos y estado...")
    try:
        mlp_trainer = joblib.load(models_dir_path / f'mlp_trainer_{dataset_name}.pkl')
        mlp_model = mlp_trainer
        mlp_model.bypass_preprocessing = True
        if not config.get('is_multiclass', False) and hasattr(mlp_model, 'optimal_threshold'):
            optimal_threshold = mlp_model.optimal_threshold
        else:
            optimal_threshold = 0.5
        tree_a = joblib.load(models_dir_path / f'trepan_{dataset_name}.pkl')
        tree_b = joblib.load(models_dir_path / f'trepan_reloaded_{dataset_name}.pkl')
        X_enc = joblib.load(models_dir_path / f'X_train_enc_{dataset_name}.pkl')
        state = joblib.load(models_dir_path / f'pipeline_state_{dataset_name}.pkl')
    except FileNotFoundError as e:
        logging.error(f"No se encontraron los archivos necesarios: {e}")
        return pd.DataFrame(), {}

    selected_indices = state['selected_indices']
    X_selected_enc_df = state['X_selected_enc']
    y_selected_mlp = state['y_selected_mlp']
    transformed_feature_names = state['transformed_feature_names']
    feature_intervals = state['feature_intervals']
    category_prefix = state['category_prefix']
    config['class_labels'] = state['class_labels']
    config['is_multiclass'] = state['is_multiclass']
    config['category_prefix'] = category_prefix
    config['dataset_name'] = dataset_name

    X_train_enc_df = pd.DataFrame(X_enc, columns=transformed_feature_names)

    if len(selected_indices) == 0:
        logging.warning("No hay instancias seleccionadas. Se omite la generación de CFs.")
        return pd.DataFrame(), {}

    cat_indices = []
    if category_prefix:
        for i, col in enumerate(transformed_feature_names):
            if any(col.startswith(p) for p in category_prefix):
                cat_indices.append(i)

    # ---------- CLEAR ----------
    logging.info("💡 Paso 6.1: Generando CFs con CLEAR")
    start = time.perf_counter()
    clear_cfs_df = run_clear_cfs(
        X_train_enc_df, X_selected_enc_df, mlp_model, config,
        transformed_feature_names, str(clear_output_dir_path) + os.sep,
        feature_intervals=feature_intervals,
        optimal_threshold=optimal_threshold
    )
    timers['clear_cfs'] = time.perf_counter() - start
    logging.info(f"  ✓ CLEAR completado en {timers['clear_cfs']:.2f}s, Total CFs: {len(clear_cfs_df)}")

    # ---------- COGS ----------
    logging.info("💡 Paso 6.2: Generando CFs con COGS (paralelizado)")
    start = time.perf_counter()

    args_list = []
    for i, idx in enumerate(selected_indices):
        x_encoded = X_train_enc_df.iloc[idx].values
        original_class = y_selected_mlp[i]
        args_list.append((
            i, x_encoded, original_class, mlp_model,
            feature_intervals, cat_indices,
            config['class_labels'], config['is_multiclass']
        ))

    cogs_cfs = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_cogs_for_single_instance, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                if isinstance(result, list):
                    cogs_cfs.extend(result)
                else:
                    cogs_cfs.append(result)

    timers['cogs_cfs'] = time.perf_counter() - start
    logging.info(f"  ✓ COGS completado en {timers['cogs_cfs']:.2f}s, Total CFs: {len(cogs_cfs)}")

    # ---------- Indicadores de consistencia ----------
    logging.info("📊 Paso 7: Calculando indicadores de consistencia")
    start = time.perf_counter()

    def compute_indicators(cf_list, cf_method_name, mlp_model, tree_a, tree_b):
        resultados = []
        if not cf_list:
            for tree_name in ['Trepan', 'TrepanReload']:
                resultados.append({
                    'dataset': dataset_name, 'metodo_CF': cf_method_name,
                    'tipo_sustituto': tree_name,
                    'indicador_a': np.nan, 'indicador_b': np.nan, 'indicador_c': np.nan,
                    'total_CFs_validos_MLP': 0, 'total_CFs_validos_sustituto': 0, 'total_CFs_generados': 0
                })
            return pd.DataFrame(resultados)

        valid_cfs_mlp = []
        for item in cf_list:
            cf = np.array(item['cf']).reshape(1, -1)
            cf_class_mlp = mlp_model.predict(cf)[0]
            if cf_class_mlp != item['original_class']:
                valid_cfs_mlp.append({'cf': cf, 'cf_class_mlp': cf_class_mlp,
                                      'original_class': item['original_class']})
        total_valid_mlp = len(valid_cfs_mlp)

        for tree, tree_name in [(tree_a, 'Trepan'), (tree_b, 'TrepanReload')]:
            valid_in_tree = []
            for item in valid_cfs_mlp:
                cf_class_tree = tree.predict(item['cf'])[0]
                if cf_class_tree != item['original_class']:
                    valid_in_tree.append({'cf_class_mlp': item['cf_class_mlp'],
                                          'cf_class_tree': cf_class_tree})
            total_valid_tree = len(valid_in_tree)
            same = sum(1 for v in valid_in_tree if v['cf_class_tree'] == v['cf_class_mlp']) if total_valid_tree > 0 else 0

            if total_valid_mlp > 0:
                indicador_a = same / total_valid_mlp
                indicador_b = (total_valid_tree - same) / total_valid_mlp
                indicador_c = (total_valid_mlp - total_valid_tree) / total_valid_mlp
            else:
                indicador_a = indicador_b = indicador_c = np.nan

            resultados.append({
                'dataset': dataset_name, 'metodo_CF': cf_method_name,
                'tipo_sustituto': tree_name,
                'indicador_a': indicador_a, 'indicador_b': indicador_b, 'indicador_c': indicador_c,
                'total_CFs_validos_MLP': total_valid_mlp,
                'total_CFs_validos_sustituto': total_valid_tree,
                'total_CFs_generados': len(cf_list)
            })
        return pd.DataFrame(resultados)

    # Reconstrucción de CFs de CLEAR
    clear_items = []
    if not clear_cfs_df.empty and 'cf_vector' in clear_cfs_df.columns:
        for _, row in clear_cfs_df.iterrows():
            clear_items.append({
                'cf': np.array(row['cf_vector']),
                'original_class': y_selected_mlp[int(row['observation'])],
                'desired_class': row.get('desired_class', 1)
            })
    else:
        # Fallback a reconstrucción manual si no existe la columna (compatibilidad)
        for _, row in clear_cfs_df.iterrows():
            selected_idx = int(row['observation'])
            if selected_idx >= len(selected_indices): continue
            feature_name = row['feature']
            if feature_name not in transformed_feature_names: continue
            orig_encoded = X_train_enc_df.iloc[selected_indices[selected_idx]].values.copy()
            feat_idx = list(transformed_feature_names).index(feature_name)
            if feature_name.startswith('cat__'):
                group_prefix = '_'.join(feature_name.split('_')[:-1])
                group_cols = [c for c in transformed_feature_names if c.startswith(group_prefix)]
                for col in group_cols:
                    orig_encoded[list(transformed_feature_names).index(col)] = 0
                orig_encoded[feat_idx] = 1
            else:
                new_value = row.get('estPerturbedFeatValue')
                if pd.isna(new_value): continue
                low, high = feature_intervals[feat_idx]
                orig_encoded[feat_idx] = np.clip(new_value, low, high)
            clear_items.append({
                'cf': orig_encoded,
                'original_class': y_selected_mlp[selected_idx],
                'desired_class': row.get('desired_class', 1)
            })

    df_clear = compute_indicators(clear_items, 'CLEAR', mlp_model, tree_a, tree_b)

    # COGS items
    cogs_items = [{'cf': item['cf'], 'original_class': item['original_class'],
                   'desired_class': item['desired_class']} for item in cogs_cfs]
    df_cogs = compute_indicators(cogs_items, 'COGS', mlp_model, tree_a, tree_b)

    df_consistency = pd.concat([df_clear, df_cogs], ignore_index=True)
    timers['compute_indicators'] = time.perf_counter() - start

    logging.info("\n📈 Resultados de consistencia:")
    logging.info(df_consistency.to_string(index=False))

    # Guardar resultados
    df_consistency.to_csv(results_dir_path / 'consistency_indicators.csv', index=False)
    if clear_items:
        pd.DataFrame(clear_items).to_csv(results_dir_path / 'clear_cfs_full.csv', index=False)
    if cogs_cfs:
        pd.DataFrame(cogs_cfs).to_csv(results_dir_path / 'cogs_cfs_full.csv', index=False)
    if not clear_cfs_df.empty:
        clear_cfs_df.to_csv(results_dir_path / 'clear_cfs.csv', index=False)
    if cogs_cfs:
        pd.DataFrame(cogs_cfs).to_csv(results_dir_path / 'cogs_cfs.csv', index=False)

    timers['total'] = time.perf_counter() - start_total
    summary = {
        'dataset': dataset_name,
        'clear_cfs_generated': int(len(clear_cfs_df)),
        'cogs_cfs_generated': int(len(cogs_cfs)),
        'timers': {k: float(v) for k, v in timers.items()}
    }
    for _, row in df_clear.iterrows():
        key = f"clear_{row['tipo_sustituto'].lower()}"
        summary[f"{key}_indicador_a"] = float(row['indicador_a']) if not pd.isna(row['indicador_a']) else None
        summary[f"{key}_indicador_b"] = float(row['indicador_b']) if not pd.isna(row['indicador_b']) else None
    for _, row in df_cogs.iterrows():
        key = f"cogs_{row['tipo_sustituto'].lower()}"
        summary[f"{key}_indicador_a"] = float(row['indicador_a']) if not pd.isna(row['indicador_a']) else None
        summary[f"{key}_indicador_b"] = float(row['indicador_b']) if not pd.isna(row['indicador_b']) else None

    with open(results_dir_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"\n⏱️  Tiempos de CFs:")
    for key, value in timers.items():
        logging.info(f"  {key}: {value:.2f}s")
    logging.info(f"\n✅ CFs de {dataset_name} completados.\n")
    return df_consistency, summary


# ---------------------------------------------------------------------------
# Ejecución principal y consolidación
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        datasets_to_run = [sys.argv[1]]
        if sys.argv[1] not in ALL_DATASETS:
            logging.error(f"Dataset '{sys.argv[1]}' no reconocido. Opciones: {ALL_DATASETS}")
            sys.exit(1)
    else:
        datasets_to_run = ALL_DATASETS

    all_results = []
    all_summaries = []
    for ds in datasets_to_run:
        try:
            df, summary = run_cf_pipeline(ds)
            all_results.append(df)
            all_summaries.append(summary)
        except Exception as e:
            logging.error(f"Error en {ds}: {e}")
            import traceback
            traceback.print_exc()
            all_summaries.append({'dataset': ds, 'error': str(e)})

    RESULTADOS_CONSOLIDADOS_DIR.mkdir(parents=True, exist_ok=True)

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(RESULTADOS_CONSOLIDADOS_DIR / 'consistency_results_all.csv', index=False)
        logging.info("\n" + "="*70)
        logging.info("📊 RESULTADOS CONSOLIDADOS (TODOS LOS DATASETS)")
        logging.info("="*70)
        logging.info(df_all.to_string(index=False))
        logging.info(f"\n✅ Archivo guardado: {RESULTADOS_CONSOLIDADOS_DIR / 'consistency_results_all.csv'}")

    with open(RESULTADOS_CONSOLIDADOS_DIR / 'summary_all.json', 'w') as f:
        json.dump(all_summaries, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.int64, np.float64)) else x)
    logging.info(f"✅ Resumen guardado: {RESULTADOS_CONSOLIDADOS_DIR / 'summary_all.json'}")

    if all_results:
        logging.info("\n" + "="*70)
        logging.info("📋 TABLA DE RESULTADOS (copiar a la tesis)")
        logging.info("="*70)
        table = df_all[['dataset', 'metodo_CF', 'tipo_sustituto', 'indicador_a', 'indicador_b',
                        'total_CFs_validos_MLP', 'total_CFs_validos_sustituto']].copy()
        table['indicador_a'] = table['indicador_a'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "NaN")
        table['indicador_b'] = table['indicador_b'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "NaN")
        logging.info(table.to_string(index=False))
        table.to_csv(RESULTADOS_CONSOLIDADOS_DIR / 'table_for_thesis.csv', index=False)

    logging.info("\n🏁 Fin del pipeline de contrafactuales.")
