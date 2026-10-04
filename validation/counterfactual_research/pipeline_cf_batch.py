"""Execução em lote do pipeline de contrafactuais sobre datasets nomeados (validação)."""
import logging
import sys

from counterfactuals.pipelines.pipeline_counterfactuals import *  # noqa: F401,F403
from counterfactuals.pipelines.pipeline_counterfactuals import run_cf_pipeline
from validation.counterfactual_research.dataset_registry import ALL_DATASETS, get_dataset_config

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
            df, summary = run_cf_pipeline(ds, get_dataset_config(ds))
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
