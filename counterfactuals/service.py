"""
API de serviço para contrafactuais — reutilizável por CLI e GUI.

Não altera o diretório de trabalho global; os pipelines usam caminhos absolutos
via counterfactuals._paths quando invocados daqui.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from counterfactuals._paths import CF_ROOT, TREPA_ROOT
from counterfactuals.dataset_config import ALL_DATASETS, MLP_ACCURACY, get_dataset_config

__all__ = [
    'ALL_DATASETS',
    'MLP_ACCURACY',
    'train_oracle_and_surrogates',
    'generate_counterfactuals',
    'improve_surrogates',
    'run_full_pipeline',
    'evaluate_consistency',
    'load_counterfactuals',
    'generate_explanation_from_session',
    'generate_global_counterfactuals_from_session',
    'build_counterfactual_tree_from_session',
    'evaluate_transfer_from_session',
    'export_counterfactual_result',
]


def _ensure_paths() -> None:
    for path in (str(TREPA_ROOT), str(CF_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def train_oracle_and_surrogates(dataset_name: str) -> Dict[str, Any]:
    """Fase 1: MLP + árvores TREPAN/TREPAN-Reloaded + seleção 33%."""
    _ensure_paths()
    from counterfactuals.pipelines.pipeline_train import run_training_pipeline
    return run_training_pipeline(dataset_name)


def generate_counterfactuals(dataset_name: str) -> Dict[str, Any]:
    """Fase 2: CLEAR + COGS + indicadores A/B/C."""
    _ensure_paths()
    from counterfactuals.pipelines.pipeline_counterfactuals import run_cf_pipeline
    df, summary = run_cf_pipeline(dataset_name)
    return {'dataframe': df, 'summary': summary}


def improve_surrogates(
    dataset_name: Optional[str] = None,
    datasets: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Fase 3: grid search + improve_surrogate."""
    _ensure_paths()
    from counterfactuals.pipelines.pipeline_improve import run_experiment
    targets = datasets or ([dataset_name] if dataset_name else ALL_DATASETS)
    return run_experiment(seed, targets)


def run_full_pipeline(dataset_name: str, seed: int = 42) -> Dict[str, Any]:
    """Executa as três fases sequencialmente."""
    train_summary = train_oracle_and_surrogates(dataset_name)
    cf_result = generate_counterfactuals(dataset_name)
    improve_entries = improve_surrogates(dataset_name=dataset_name, seed=seed)
    return {
        'train': train_summary,
        'counterfactuals': cf_result,
        'improve': improve_entries,
    }


def evaluate_consistency(dataset_name: str) -> Optional[Any]:
    """Carrega indicadores de consistência guardados."""
    import pandas as pd
    from counterfactuals._paths import results_dir
    path = results_dir(dataset_name) / 'consistency_indicators.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_counterfactuals(dataset_name: str, method: str = 'cogs') -> List[Dict[str, Any]]:
    """Carrega CFs de disco (clear ou cogs)."""
    _ensure_paths()
    from counterfactuals.pipelines.pipeline_improve import cargar_cfs
    return cargar_cfs(dataset_name, method)


def _normalise_target_name(value: Any) -> str:
    return ''.join(ch.lower() for ch in str(value or '') if ch.isalnum())


def _resolve_interactive_context(
    session: Dict[str, Any], options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Selecciona oráculo/espaço conforme modelo-alvo e aceitação ontológica."""
    options = dict(options or {})
    target = options.get('target_model') or session.get('target_model') or 'Trepan Reloaded'
    token = _normalise_target_name(target)
    accepted = bool((session.get('ontology_acceptance') or {}).get('accepted'))
    original_oracle = session.get('mlp_original') or session.get('mlp_oracle')
    original_X = np.asarray(
        session.get('X_train_original', session.get('X_train_enc')), dtype=float,
    )
    original_y = np.asarray(
        session.get('y_train_original', session.get('y_train_enc')),
    )
    original_names = list(
        session.get('feature_names_original') or session.get('transformed_feature_names') or []
    )
    augmented_X = session.get('X_train_augmented')
    augmented_names = list(session.get('feature_names_augmented') or [])
    onto_oracle = session.get('mlp_onto')

    use_onto = (
        accepted and onto_oracle is not None and augmented_X is not None and augmented_names
        and token in {'mlpontologica', 'mlponto', 'trepanreloaded'}
    )
    if use_onto:
        X = np.asarray(augmented_X, dtype=float)
        y = np.asarray(session.get('y_train_augmented', original_y))
        names = augmented_names
        oracle = onto_oracle
        oracle_label = 'MLP Ontológica'
    else:
        X, y, names = original_X, original_y, original_names
        oracle = original_oracle
        oracle_label = 'MLP Original'

    if token in {'trepanoriginal', 'trepan'}:
        global_tree = session.get('tree_a')
        model_type = 'tree'
    elif token in {'trepanreloaded', 'reloaded'}:
        global_tree = session.get('tree_b')
        model_type = 'tree'
    elif token in {'c45', 'c45nativo'}:
        global_tree = session.get('c45_tree')
        model_type = 'tree'
    else:
        global_tree = None
        model_type = 'mlp'

    # Cada modelo-alvo é o seu próprio oráculo contrafactual. O comportamento
    # anterior rotulava C4.5 e ambos os TREPANs pelo mesmo MLP e podia produzir
    # três árvores CF idênticas apesar de os modelos globais serem diferentes.
    if model_type == 'tree' and global_tree is not None:
        oracle = global_tree
        oracle_label = f"{target} (modelo-alvo directo)"

    if oracle is None or X is None or not names:
        raise ValueError(f"Modelo/espaço indisponível para o alvo '{target}'.")
    if model_type == 'tree' and global_tree is None:
        raise ValueError(f"A árvore alvo '{target}' ainda não foi treinada.")
    return {
        'target_model': target,
        'oracle': oracle,
        'oracle_label': oracle_label,
        'X': X,
        'y': y,
        'feature_names': names,
        'global_tree': global_tree,
        'model_type': model_type,
        'ontology_active': use_onto,
        'oracle_is_target_model': model_type == 'tree',
    }


def generate_explanation_from_session(
    session: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
    progress_fn=None,
    cancel_fn=None,
) -> Dict[str, Any]:
    """Geração interactiva completa para uma instância escolhida na GUI/CLI."""
    from counterfactuals.engine import (
        CounterfactualConstraints,
        CounterfactualEngine,
        OntologyCFValidator,
    )

    options = dict(options or {})
    context = _resolve_interactive_context(session, options)

    def progress(stage, pct, msg):
        if progress_fn:
            progress_fn(stage, pct, msg)

    if cancel_fn and cancel_fn():
        raise InterruptedError('Cancelado')
    index = int(options.get('instance_index', 0))
    if index < 0 or index >= len(context['X']):
        raise IndexError(
            f"Índice de instância {index} fora do intervalo 0..{len(context['X']) - 1}."
        )

    progress('constraints', 10, 'Construindo restrições de domínio e ação...')
    constraint_config = dict(session.get('constraints') or session.get('config') or {})
    constraints = CounterfactualConstraints.from_data(
        context['X'], context['feature_names'], constraint_config,
    )
    validate_ontology = bool(options.get('validate_ontology', True))
    ontology_validator = OntologyCFValidator(
        session.get('ontology') if context['ontology_active'] and validate_ontology else None,
        session.get('ontology_feature_mapping'),
        session.get('semantic_rules'),
    )
    engine = CounterfactualEngine(
        context['oracle'],
        context['X'],
        context['feature_names'],
        y_reference=context['y'],
        constraints=constraints,
        ontology_validator=ontology_validator,
        global_tree=context['global_tree'],
        seed=int(options.get('seed', 42)),
    )
    desired = options.get('desired_class')
    if isinstance(desired, str) and desired.strip().lstrip('-').isdigit():
        desired = int(desired)
    progress('generate', 25, f"Gerando com {options.get('method', 'AUTO')}...")
    result = engine.generate(
        context['X'][index],
        desired_class=desired,
        method=options.get('method', 'AUTO'),
        model_type=context['model_type'],
        total_cfs=int(options.get('total_cfs', 5)),
        apply_rst=bool(options.get('apply_rst', False)),
        robustness_samples=int(options.get('robustness_samples', 100)),
        robustness_epsilon=float(options.get('robustness_epsilon', 0.02)),
    )
    if cancel_fn and cancel_fn():
        raise InterruptedError('Cancelado')
    result.update({
        'instance_index': index,
        'target_model': context['target_model'],
        'oracle_label': context['oracle_label'],
        'ontology_active': bool(context['ontology_active']),
        'scope': 'loaded_dataset_only',
        'dataset': session.get('dataset_name', 'dataset_carregado'),
    })
    progress('evaluate', 85, 'Calculando métricas contrafactuais formais...')
    from counterfactuals.evaluation import evaluate_generation_result
    result['formal_evaluation'] = evaluate_generation_result(
        result,
        context['oracle'],
        context['X'],
        context['y'],
        seed=int(options.get('seed', 42)),
    )
    progress('done', 100, 'Explicação contrafactual concluída.')
    return result


def generate_global_counterfactuals_from_session(
    session: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
    progress_fn=None,
    cancel_fn=None,
) -> Dict[str, Any]:
    """Extrai regras e CFs globais da árvore seleccionada no dataset activo."""
    from counterfactuals.global_rules import analyse_global_rules

    options = dict(options or {})
    context = _resolve_interactive_context(session, options)
    if context['model_type'] != 'tree' or context['global_tree'] is None:
        raise ValueError(
            "Contrafactuais globais por regras exigem Trepan Original, "
            "Trepan Reloaded ou C4.5-Nativo."
        )
    if cancel_fn and cancel_fn():
        raise InterruptedError('Cancelado')
    if progress_fn:
        progress_fn('global_rules', 20, 'Extraindo todas as regras da árvore activa...')
    result = analyse_global_rules(
        context['global_tree'],
        context['oracle'],
        context['X'],
        context['feature_names'],
        class_labels=session.get('class_labels'),
        max_per_rule=int(options.get('global_max_per_rule', 5)),
        remove_fragile=bool(options.get('global_remove_fragile', True)),
        fragile_n_changes_threshold=int(
            options.get('global_fragile_n_changes_threshold', 10)
        ),
        model_name=context['target_model'],
        dataset_name=session.get('dataset_name', 'dataset_carregado'),
    )
    result['oracle_label'] = context['oracle_label']
    result['ontology_active'] = bool(context['ontology_active'])
    if cancel_fn and cancel_fn():
        raise InterruptedError('Cancelado')
    if progress_fn:
        progress_fn('done', 100, 'Contrafactuais globais concluídos.')
    return result


def build_counterfactual_tree_from_session(
    session: Dict[str, Any],
    generation_result: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
    progress_fn=None,
    cancel_fn=None,
) -> Dict[str, Any]:
    """Constrói uma árvore CF local a partir do último resultado da sessão."""
    from counterfactuals.cf_tree import build_counterfactual_tree

    options = dict(options or {})
    if not generation_result or generation_result.get('result_type') in {
        'global_rules', 'counterfactual_tree'
    } or generation_result.get('rows') is not None:
        raise ValueError("Gere primeiro contrafactuais locais válidos para a instância activa.")
    context = _resolve_interactive_context(session, options)
    previous_target = generation_result.get('target_model')
    if previous_target and _normalise_target_name(previous_target) != _normalise_target_name(
        context['target_model']
    ):
        raise ValueError(
            "O resultado local pertence a outro modelo alvo. Gere novamente os "
            "contrafactuais antes de construir a árvore."
        )
    current_dataset = session.get('dataset_name', 'dataset_carregado')
    previous_dataset = generation_result.get('dataset')
    if previous_dataset and previous_dataset != current_dataset:
        raise ValueError("O resultado local pertence a outro dataset carregado.")
    if cancel_fn and cancel_fn():
        raise InterruptedError('Cancelado')
    if progress_fn:
        progress_fn('cf_tree', 20, 'Construindo vizinhança e árvore explicativa CF...')
    result = build_counterfactual_tree(
        context['oracle'],
        context['X'],
        generation_result,
        context['feature_names'],
        original_tree=context['global_tree'],
        class_labels=session.get('class_labels'),
        max_depth=int(options.get('cf_tree_max_depth', 5)),
        neighborhood_size=int(options.get('cf_tree_neighborhood_size', 300)),
        seed=int(options.get('seed', 42)),
        model_name=context['target_model'],
        dataset_name=current_dataset,
    )
    result['oracle_label'] = context['oracle_label']
    result['ontology_active'] = bool(context['ontology_active'])
    if cancel_fn and cancel_fn():
        raise InterruptedError('Cancelado')
    if progress_fn:
        progress_fn('done', 100, 'Árvore explicativa contrafactual concluída.')
    return result


def evaluate_transfer_from_session(
    session: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
    progress_fn=None,
    cancel_fn=None,
) -> Dict[str, Any]:
    """P1-P8 apenas para o dataset presente na sessão activa da GUI."""
    from counterfactuals.transfer import evaluate_transfer_protocol

    options = dict(options or {})
    forbidden_scope = {
        key for key in ('dataset', 'dataset_name', 'datasets', 'sessions')
        if key in options
    }
    if forbidden_scope:
        raise ValueError(
            "A aplicação analisa exclusivamente o dataset actualmente carregado; "
            "não aceite selecção externa ou múltiplos datasets nesta operação."
        )
    methods = options.get('transfer_methods') or ('LORE-LOCAL', 'CLEAR', 'COGS')
    result = evaluate_transfer_protocol(
        session,
        methods=methods,
        fraction=float(options.get('fraction', 0.33)),
        seed=int(options.get('seed', 42)),
        robustness_samples=int(options.get('robustness_samples', 100)),
        robustness_epsilon=float(options.get('robustness_epsilon', 0.02)),
        progress_fn=progress_fn,
        cancel_fn=cancel_fn,
    )
    result['scope'] = 'loaded_dataset_only'
    result['dataset'] = session.get('dataset_name', 'dataset_carregado')
    return result


def export_counterfactual_result(
    result: Dict[str, Any], directory: Any, stem: str = 'counterfactual_report',
) -> Dict[str, str]:
    from counterfactuals.export import export_counterfactual_result as _export
    return _export(result, directory, stem=stem)


def _select_instances_stratified(y_train: np.ndarray, fraction: float = 0.33, seed: int = 42):
    import numpy as np
    rng = np.random.RandomState(seed)
    unique_classes = np.unique(y_train)
    selected = []
    for cls in unique_classes:
        cls_indices = np.where(y_train == cls)[0]
        n_select = int(fraction * len(cls_indices))
        if n_select > 0:
            selected.append(rng.choice(cls_indices, size=n_select, replace=False))
    if not selected:
        return np.array([], dtype=int)
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


def generate_counterfactuals_from_session(
    session: Dict[str, Any],
    progress_fn=None,
    cancel_fn=None,
) -> Dict[str, Any]:
    """
    Gera CFs in-memory para a GUI BIURI.

    session keys: mlp_oracle, X_train_enc, y_train_enc, tree_a, tree_b,
    transformed_feature_names, class_labels, is_multiclass, config (opcional)
    """
    import pandas as pd
    import numpy as np
    from counterfactuals.pipelines.pipeline_counterfactuals import (
        run_clear_cfs,
        run_cogs_for_single_instance,
    )
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def progress(stage, pct, msg):
        if progress_fn:
            progress_fn(stage, pct, msg)

    def cancelled():
        return bool(cancel_fn and cancel_fn())

    mlp = session['mlp_oracle']
    if hasattr(mlp, 'bypass_preprocessing'):
        mlp.bypass_preprocessing = True

    X_enc = np.asarray(session['X_train_enc'], dtype=float)
    y_enc = np.asarray(session['y_train_enc'])
    feat_names = list(session['transformed_feature_names'])
    config = dict(session.get('config') or {})
    config.setdefault('class_labels', session.get('class_labels', {}))
    config.setdefault('is_multiclass', session.get('is_multiclass', len(config['class_labels']) > 2))
    config.setdefault('dataset_name', session.get('dataset_name', 'GUI'))

    X_train_df = pd.DataFrame(X_enc, columns=feat_names)
    selected_idx = _select_instances_stratified(y_enc)
    if cancelled():
        raise InterruptedError('Cancelado')

    progress('select_instances', 10, f'Seleccionadas {len(selected_idx)} instancias (33%)')
    X_sel = X_train_df.iloc[selected_idx].copy()
    y_sel = y_enc[selected_idx]

    intervals = session.get('feature_intervals')
    if intervals is None:
        intervals = np.array([(X_train_df[c].min(), X_train_df[c].max()) for c in feat_names], dtype=object)

    cat_prefix = session.get('category_prefix') or []
    cat_indices = [
        i for i, col in enumerate(feat_names)
        if any(str(col).startswith(p) for p in cat_prefix)
    ]

    clear_dir = session.get('clear_output_dir')
    if clear_dir is None:
        from counterfactuals._paths import clear_output_dir
        clear_path = clear_output_dir('gui_session')
    else:
        from pathlib import Path
        clear_path = Path(clear_dir)
    clear_path.mkdir(parents=True, exist_ok=True)

    progress('clear', 25, 'Gerando contrafactuais CLEAR...')
    if cancelled():
        raise InterruptedError('Cancelado')
    clear_df = run_clear_cfs(
        X_train_df, X_sel, mlp, config, feat_names, str(clear_path),
        feature_intervals=intervals,
        optimal_threshold=session.get('optimal_threshold', 0.5),
    )

    progress('cogs', 55, 'Gerando contrafactuais COGS...')
    cogs_items = []
    args_list = [
        (i, X_train_df.iloc[idx].values, int(y_sel[i]), mlp, intervals, cat_indices,
         config['class_labels'], config['is_multiclass'])
        for i, idx in enumerate(selected_idx)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_cogs_for_single_instance, a) for a in args_list]
        for fut in as_completed(futures):
            if cancelled():
                raise InterruptedError('Cancelado')
            res = fut.result()
            if res is None:
                continue
            if isinstance(res, list):
                cogs_items.extend(res)
            else:
                cogs_items.append(res)

    clear_items = []
    if not clear_df.empty and 'cf_vector' in clear_df.columns:
        for _, row in clear_df.iterrows():
            clear_items.append({
                'cf': np.array(row['cf_vector']),
                'original_class': int(y_sel[int(row['observation'])]),
            })

    progress('consistency', 80, 'Calculando indicadores A/B/C...')
    tree_a = session['tree_a']
    tree_b = session['tree_b']

    def _indicators(cf_list, method_name):
        rows = []
        valid_mlp = []
        for item in cf_list:
            cf = np.array(item['cf']).reshape(1, -1)
            pred = mlp.predict(cf)[0]
            if pred != item['original_class']:
                valid_mlp.append({'cf': cf, 'cf_class_mlp': pred, 'original_class': item['original_class']})
        for tree, tname in ((tree_a, 'Trepan'), (tree_b, 'TrepanReload')):
            valid_tree = []
            for item in valid_mlp:
                pt = tree.predict(item['cf'])[0]
                if pt != item['original_class']:
                    valid_tree.append({'cf_class_mlp': item['cf_class_mlp'], 'cf_class_tree': pt})
            same = sum(1 for v in valid_tree if v['cf_class_tree'] == v['cf_class_mlp'])
            n_mlp = len(valid_mlp)
            n_tree = len(valid_tree)
            rows.append({
                'metodo_CF': method_name, 'tipo_sustituto': tname,
                'indicador_a': same / n_mlp if n_mlp else float('nan'),
                'indicador_b': (n_tree - same) / n_mlp if n_mlp else float('nan'),
                'indicador_c': (n_mlp - n_tree) / n_mlp if n_mlp else float('nan'),
                'total_CFs_validos_MLP': n_mlp,
                'total_CFs_validos_sustituto': n_tree,
                'total_CFs_generados': len(cf_list),
            })
        return rows

    consistency = _indicators(clear_items, 'CLEAR') + _indicators(cogs_items, 'COGS')
    progress('done', 100, 'Contrafactuais gerados.')

    return {
        'selected_indices': selected_idx,
        'clear_items': clear_items,
        'cogs_items': cogs_items,
        'consistency': consistency,
        'y_selected': y_sel,
    }


def improve_surrogates_from_session(
    session: Dict[str, Any],
    cf_result: Optional[Dict[str, Any]] = None,
    progress_fn=None,
    cancel_fn=None,
    seed: int = 42,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Melhora TREPAN Original/Reloaded sem consultar o teste bloqueado.

    O lote gerado pela ferramenta contrafactual interactiva não é reutilizado
    automaticamente: pode conter outro modelo-alvo ou todo o dataset. Cada
    árvore recebe CFs próprios, gerados no seu espaço e no seu subtreino.
    """
    from counterfactuals.engine import OntologyCFValidator
    from counterfactuals.surrogate_improvement import (
        ImprovementSearchConfig,
        select_and_refit_improved_surrogate,
    )

    def progress(stage, pct, msg):
        if progress_fn:
            progress_fn(stage, pct, msg)

    contexts = dict(session.get('improvement_contexts') or {})
    if not contexts:
        raise ValueError(
            "A sessão não contém partições auditáveis para melhorar os substitutos. "
            "Treine novamente o projeto antes de usar esta opção."
        )
    option_values = dict(options or {})
    requested = str(option_values.pop('target_model', 'both')).strip().lower()
    aliases = {
        'both': ('trepan_original', 'trepan_reloaded'),
        'ambas': ('trepan_original', 'trepan_reloaded'),
        'trepan_original': ('trepan_original',),
        'trepan original': ('trepan_original',),
        'trepan_reloaded': ('trepan_reloaded',),
        'trepan reloaded': ('trepan_reloaded',),
    }
    if requested in {'c45', 'c4.5', 'c45-nativo'}:
        raise ValueError(
            "C4.5 é um baseline supervisionado por rótulos reais, não uma árvore "
            "substituta do MLP; a melhoria contrafactual aplica-se apenas aos TREPANs."
        )
    if requested not in aliases:
        raise ValueError(f"Árvore substituta desconhecida: {requested}.")
    target_keys = aliases[requested]
    option_values['random_state'] = int(seed)
    if 'sample_size' not in option_values:
        option_values['sample_size'] = int(session.get('improvement_sample_size', 1200))
    config = ImprovementSearchConfig.from_options(option_values)
    class_names = list(
        session.get('class_names') or session.get('class_labels', {}).values()
    )
    constraints_config = dict(session.get('constraints') or session.get('config') or {})

    def original_builder(context):
        def build(request):
            from core.trepan_original import TrepanOriginalExtractor
            extractor = TrepanOriginalExtractor(random_state=int(request.seed))
            extractor.extract_tree(
                context['oracle'],
                request.X_train,
                request.y_train,
                sample_size=config.sample_size,
                feature_names=context['feature_names'],
                class_names=class_names,
                X_train=request.X_train,
                X_test=request.X_validation,
                y_train=request.y_train,
                y_test=request.y_validation,
                training_limits=context.get('training_limits'),
                extra_X=request.extra_X,
                extra_y=request.extra_y,
                extra_weights=request.extra_weights,
            )
            return extractor.explainer_tree, dict(extractor.last_audit or {})
        return build

    def reloaded_builder(context):
        base_context = dict(context.get('reloaded_context') or {})
        if not base_context:
            return original_builder(context)

        def build(request):
            from core.trepan_reloaded_extractor import TrepanReloadedExtractor
            subcontext = dict(base_context)
            subcontext.update({
                'X_train': request.X_train,
                'X_test': request.X_validation,
                'y_train': request.y_train,
                'y_test': request.y_validation,
                'feature_names': list(context['feature_names']),
            })
            oof = base_context.get('oof_teacher_probabilities')
            if oof is not None:
                oof = np.asarray(oof)
                if len(oof) == len(context['X_development']):
                    subcontext['oof_teacher_probabilities'] = oof[request.train_indices]
                else:
                    subcontext.pop('oof_teacher_probabilities', None)

            extractor = TrepanReloadedExtractor(
                ontology=context.get('ontology'),
                onto_feature_bias_weight=context.get('onto_feature_bias_weight'),
            )
            extractor.ontology_quality_report = context.get('ontology_quality_report')
            extractor.reasoner_report = context.get('reasoner_report')
            extractor.c45_baseline = context.get('c45_baseline')
            extractor._training_limits = dict(context.get('training_limits') or {})
            # O experimento mede a contribuição dos CFs escolhidos neste botão.
            # Desliga o lote CF automático interno, tanto no controlo como nas
            # candidatas, mantendo os restantes componentes Reloaded iguais.
            extractor._training_limits['plausible_cf_budget'] = 0
            random_state = np.random.get_state()
            np.random.seed(int(request.seed))
            try:
                extractor.extract_tree_with_ontology(
                    subcontext.get('mlp_original_ref') or context['oracle'],
                    request.X_train,
                    request.y_train,
                    list(context['feature_names']),
                    class_names,
                    sample_size=config.sample_size,
                    original_feature_names=context.get('original_feature_names'),
                    mlp_model_onto=subcontext.get('mlp_model_onto'),
                    X_train=request.X_train,
                    X_test=request.X_validation,
                    y_train=request.y_train,
                    y_test=request.y_validation,
                    reloaded_context=subcontext,
                    extra_X=request.extra_X,
                    extra_y=request.extra_y,
                    extra_weights=request.extra_weights,
                )
            finally:
                np.random.set_state(random_state)
            return extractor.explainer_tree, dict(extractor.last_audit or {})
        return build

    def legacy_metrics(result):
        before = (
            result.get('matched_no_cf_metrics')
            or result.get('baseline_metrics')
            or {}
        )
        after = result.get('candidate_metrics') or before
        return {
            'fidelity_original': before.get('fidelity'),
            'fidelity_improved': after.get('fidelity'),
            'accuracy_original': before.get('accuracy'),
            'accuracy_improved': after.get('accuracy'),
            'balanced_accuracy_original': before.get('balanced_accuracy'),
            'balanced_accuracy_improved': after.get('balanced_accuracy'),
            'macro_f1_original': before.get('macro_f1'),
            'macro_f1_improved': after.get('macro_f1'),
            'evaluation_role': 'independent_internal_acceptance_holdout',
            'test_used': False,
        }

    results: Dict[str, Dict[str, Any]] = {}
    for target_position, key in enumerate(target_keys):
        context = contexts.get(key)
        if not context:
            raise ValueError(f"Contexto de melhoria indisponível: {key}.")
        if key == 'trepan_reloaded' and context.get('mirrors_original'):
            original_result = results.get('trepan_original')
            if original_result is not None:
                mirrored = dict(original_result)
                mirrored.update({
                    'model_name': 'TREPAN Reloaded',
                    'status': 'mirrored_no_ontology',
                    'mirrors_original': True,
                })
                results[key] = mirrored
                continue
        oracle = context.get('oracle')
        if oracle is None or context.get('tree') is None:
            raise ValueError(f"Oráculo/árvore em falta para {key}.")
        if hasattr(oracle, 'bypass_preprocessing'):
            oracle.bypass_preprocessing = True
        names = list(context.get('feature_names') or [])
        X_development = np.asarray(context.get('X_development'), dtype=float)
        y_development = np.asarray(context.get('y_development'))
        if len(names) != X_development.shape[1]:
            raise ValueError(
                f"Schema de melhoria incompatível para {key}: "
                f"{len(names)} nomes vs {X_development.shape[1]} colunas."
            )
        validator = None
        if key == 'trepan_reloaded' and context.get('ontology_active'):
            validator = OntologyCFValidator(
                context.get('ontology'),
                session.get('ontology_feature_mapping'),
                context.get('semantic_rules'),
            )
        builder = (
            reloaded_builder(context)
            if key == 'trepan_reloaded' and context.get('ontology_active')
            else original_builder(context)
        )
        progress(
            'improve',
            5 + target_position * 45,
            f"Experimento controlado: {context.get('display_name', key)}...",
        )
        results[key] = select_and_refit_improved_surrogate(
            model_name=context.get('display_name', key),
            oracle=oracle,
            original_tree=context['tree'],
            X_development=X_development,
            y_development=y_development,
            feature_names=names,
            candidate_builder=builder,
            config=config,
            constraints_config=constraints_config,
            ontology_validator=validator,
            progress_fn=progress_fn,
            cancel_fn=cancel_fn,
        )

    original_result = results.get('trepan_original')
    reloaded_result = results.get('trepan_reloaded')
    progress('evaluate', 95, 'Aplicando gate de melhoria sem consultar o teste...')
    output = {
        'results': results,
        'protocol': 'development-fit/internal-validation; locked test untouched',
        'test_used_for_selection': False,
        'final_test_evaluated': False,
        'prior_interactive_cf_reused': False,
        'prior_interactive_cf_present': bool(cf_result),
        'c45_eligible': False,
        'c45_reason': 'baseline supervisionado, não substituto do MLP',
        'config': dict(option_values),
    }
    if original_result is not None:
        output.update({
            'trepan_improved': (
                original_result['deployed_tree'] if original_result.get('accepted') else None
            ),
            'trepan_candidate': original_result.get('candidate_tree'),
            'trepan_stats': original_result.get('selected_augmentation') or {},
            'trepan_metrics': legacy_metrics(original_result),
            'trepan_accepted': bool(original_result.get('accepted')),
        })
    if reloaded_result is not None:
        output.update({
            'trepan_reloaded_improved': (
                reloaded_result['deployed_tree'] if reloaded_result.get('accepted') else None
            ),
            'trepan_reloaded_candidate': reloaded_result.get('candidate_tree'),
            'trepan_reloaded_stats': reloaded_result.get('selected_augmentation') or {},
            'trepan_reloaded_metrics': legacy_metrics(reloaded_result),
            'trepan_reloaded_accepted': bool(reloaded_result.get('accepted')),
        })
    progress('done', 100, 'Melhoria contrafactual concluída.')
    return output
