"""Testes de integração GUI (headless) para contrafactuais."""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel

from gui.counterfactual_worker import CounterfactualWorker
from gui.counterfactual_panel import CounterfactualPanel
from gui.biuri_app_complete import TreeVisualizationWidget


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_counterfactual_worker_generate_smoke(qapp, monkeypatch):
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.RandomState(0)
    X = rng.randn(60, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=400, random_state=0)
    mlp.fit(X, y)
    tree = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree.fit(X, y)

    class Oracle:
        def predict(self, X_in):
            return mlp.predict(np.asarray(X_in, dtype=float))

        def predict_proba(self, X_in):
            return mlp.predict_proba(np.asarray(X_in, dtype=float))

    session = {
        'mlp_oracle': Oracle(),
        'X_train_enc': X,
        'y_train_enc': y,
        'tree_a': tree,
        'tree_b': tree,
        'transformed_feature_names': [f'f{i}' for i in range(4)],
        'class_labels': {0: '0', 1: '1'},
        'is_multiclass': False,
        'config': {
            'class_labels': {0: '0', 1: '1'},
            'is_multiclass': False,
            'clear_max_predictors': 1,
        },
    }

    import pandas as pd
    monkeypatch.setattr(
        'counterfactuals.pipelines.pipeline_counterfactuals.run_clear_cfs',
        lambda *a, **k: pd.DataFrame(),
    )

    results = []
    errors = []
    worker = CounterfactualWorker(session, mode=CounterfactualWorker.STAGE_GENERATE)
    worker.finished_ok.connect(lambda r: results.append(r))
    worker.failed.connect(lambda e: errors.append(e))
    worker.run()
    assert not errors, errors
    assert results
    assert 'counterfactuals' in results[0]


def test_counterfactual_panel_options_and_results(qapp):
    panel = CounterfactualPanel()
    panel.resize(1280, 720)
    panel.configure(
        n_instances=12,
        class_labels={0: 'não', 1: 'sim'},
        available_models=['MLP Original', 'Trepan Original', 'Trepan Reloaded'],
        dataset_name='dataset_carregado.arff',
    )
    panel.instance_spin.setValue(4)
    panel.method_combo.setCurrentIndex(panel.method_combo.findData('CLEAR'))
    options = panel.options()
    assert options['instance_index'] == 4
    assert options['method'] == 'CLEAR'
    assert options['fraction'] == pytest.approx(0.33)
    assert 'dataset_carregado.arff' in panel.dataset_scope_label.text()
    assert panel.transfer_button.isEnabled()
    panel.model_combo.setCurrentText('Trepan Original')
    qapp.processEvents()
    assert panel.global_button.isEnabled()
    assert not panel.tree_button.isEnabled()

    panel.show()
    qapp.processEvents()
    for field in (
        panel.model_combo,
        panel.instance_spin,
        panel.desired_combo,
        panel.method_combo,
        panel.total_spin,
        panel.robustness_samples,
        panel.robustness_epsilon,
    ):
        assert field.isVisible()
        assert field.height() >= 30
    assert panel.controls_box.height() >= panel.controls_box.minimumSizeHint().height()
    assert panel.summary_text.width() >= 300
    assert panel.results_splitter.sizes()[0] > 0
    assert all(size > 0 for size in panel.result_tables_splitter.sizes())
    assert panel.scroll_area.isVisible()
    assert panel.scope_badge.isVisible()

    # Desktop: formulário 4 colunas, acções e indicadores numa única linha.
    panel.resize(1440, 760)
    qapp.processEvents()
    assert panel.controls_grid.getItemPosition(
        panel.controls_grid.indexOf(panel._configuration_cells[-1])
    )[0] == 1
    assert panel.actions_grid.getItemPosition(
        panel.actions_grid.indexOf(panel._action_buttons[-1])
    )[0] == 0
    assert panel.cards_grid.getItemPosition(
        panel.cards_grid.indexOf(panel._metric_card_widgets[-1])
    )[0] == 0
    assert panel.results_splitter.orientation() == Qt.Orientation.Horizontal

    # Ecrã estreito: reflow sem ocultar controlos; resultados empilham-se.
    panel.resize(700, 760)
    qapp.processEvents()
    assert panel.controls_grid.getItemPosition(
        panel.controls_grid.indexOf(panel._configuration_cells[-1])
    )[0] == 7
    assert panel.actions_grid.getItemPosition(
        panel.actions_grid.indexOf(panel._action_buttons[-1])
    )[0] == 2
    assert panel.results_splitter.orientation() == Qt.Orientation.Vertical
    panel.resize(1280, 720)
    qapp.processEvents()

    from sklearn.tree import DecisionTreeClassifier

    runtime_tree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.asarray([0, 0, 1, 1]),
    )
    panel.display_result({
        'narrative': 'Contrafactual encontrado.',
        'aggregate_metrics': {'validity': 1.0},
        'candidates': [{
            'method': 'CLEAR', 'prediction': 1,
            'changes': [{'feature': 'f0'}],
            'metrics': {'validity': True, 'proximity': 0.1, 'robustness': 0.9},   # só CFs validados no modelo habilitam a árvore CF
        }],
    })
    assert panel.candidates_table.rowCount() == 1
    assert panel.export_button.isEnabled()
    assert panel.tree_button.isEnabled()
    assert panel.candidates_table.horizontalHeaderItem(2).text() == 'Classe'

    panel.display_result({
        'dataset': 'dataset_carregado.arff',
        'rows': [{
            'instance_index': 4, 'method': 'CLEAR', 'category': 'STRONG_TRANSFER',
            'trepan_changed': True, 'reloaded_changed': True,
            'proximity': 0.1, 'joint_robustness': 0.9,
        }],
        'summary': {'valid_mlp_cf_rate': 1.0, 'mean_joint_robustness': 0.9},
    })
    assert panel.candidates_table.horizontalHeaderItem(2).text() == 'Categoria'
    assert panel.candidates_table.horizontalHeaderItem(5).text() == 'Robustez conjunta'

    panel.display_result({
        'result_type': 'global_rules',
        'narrative': 'Regras globais.',
        'aggregate_metrics': {'rule_coverage': 1.0, 'global_fidelity': 0.9},
        'counterfactual_rules': [{
            'factual_rule_id': 'R1', 'counterfactual_rule_id': 'R2',
            'target_class_name': 'sim', 'n_changes': 1,
            'target_confidence': 0.8,
            'changes': [{'feature': 'f0'}],
        }],
    })
    assert panel.candidates_table.horizontalHeaderItem(0).text() == 'Regra factual'
    assert panel.candidates_table.rowCount() == 1

    panel.display_result({
        'result_type': 'counterfactual_tree',
        'narrative': 'Árvore CF.',
        'aggregate_metrics': {'fidelity_to_oracle': 1.0},
        'tree_rules': [{
            'rule_id': 'R1', 'predicted_class_name': 'sim',
            'conditions': ['f0 > 0'], 'rule': 'SE f0 > 0 ENTÃO sim',
            'support': 0.5, 'confidence': 1.0,
        }],
        '_runtime_tree_model': runtime_tree,
    })
    assert panel.visualize_tree_button.isEnabled()
    assert panel.candidates_table.horizontalHeaderItem(0).text() == 'Regra'
    panel.close()


def test_counterfactual_tree_model_is_rendered_without_trepan_variants(qapp):
    """A árvore CF isolada deve chegar ao canvas, sem cair no placeholder."""
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(max_depth=2, random_state=7).fit(
        np.asarray([[0.0, 0.0], [0.2, 0.1], [0.8, 0.9], [1.0, 1.0]]),
        np.asarray([0, 0, 1, 1]),
    )
    visualization = TreeVisualizationWidget(
        tree_model=tree,
        feature_names=['feature_a', 'feature_b'],
        class_names=['classe 0', 'classe 1'],
        dataset_name='dataset_carregado.arff',
        primary_tree_label='Árvore contrafactual',
    )

    assert visualization.tree_widget.tree_model is tree
    assert visualization._get_current_model_name() == 'Árvore contrafactual'
    assert all(
        'Ningún árbol disponible' not in label.text()
        for label in visualization.findChildren(QLabel)
    )
    visualization.close()


def test_counterfactual_worker_transfer_smoke(qapp):
    class Oracle:
        classes_ = np.array([0, 1])

        def predict(self, values):
            values = np.asarray(values, dtype=float)
            return (values[:, 0] + values[:, 1] > 0).astype(int)

        def predict_proba(self, values):
            values = np.asarray(values, dtype=float)
            score = 1 / (1 + np.exp(-values[:, 0] - values[:, 1]))
            return np.column_stack([1 - score, score])

    rng = np.random.RandomState(9)
    X = rng.uniform(-2, 2, size=(80, 2))
    oracle = Oracle()
    y = oracle.predict(X)
    names = ['f0', 'f1']
    session = {
        'dataset_name': 'dataset_carregado.arff',
        'mlp_oracle': oracle, 'mlp_original': oracle,
        'X_train_enc': X, 'y_train_enc': y,
        'X_train_original': X, 'y_train_original': y,
        'tree_a': oracle, 'tree_b': oracle,
        'transformed_feature_names': names,
        'feature_names_original': names,
        'tree_a_feature_names': names, 'tree_b_feature_names': names,
    }
    results, errors = [], []
    worker = CounterfactualWorker(
        session,
        mode=CounterfactualWorker.STAGE_TRANSFER,
        options={
            'transfer_methods': ('DICE',), 'fraction': 0.05,
            'robustness_samples': 10,
        },
    )
    worker.finished_ok.connect(results.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors, errors
    assert results[0]['transfer']['protocol'] == 'P1-P8'
    assert results[0]['transfer']['scope'] == 'loaded_dataset_only'
    assert results[0]['transfer']['dataset'] == 'dataset_carregado.arff'


def _cf_tree_session():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.RandomState(3)
    X = rng.uniform(-2, 2, size=(300, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    oracle = RandomForestClassifier(n_estimators=15, random_state=0).fit(X, y)
    names = ['f0', 'f1', 'f2']
    return {
        'dataset_name': 'dataset_carregado.arff',
        'mlp_oracle': oracle, 'mlp_original': oracle,
        'X_train_enc': X, 'y_train_enc': y,
        'X_train_original': X, 'y_train_original': y,
        'tree_a': oracle, 'tree_b': oracle,
        'transformed_feature_names': names, 'feature_names_original': names,
        'tree_a_feature_names': names, 'tree_b_feature_names': names,
        'class_labels': {0: 'neg', 1: 'pos'},
    }


def test_construir_arvore_cf_end_to_end_worker_to_panel(qapp):
    """'Construir árvore CF': gerar CF (worker) -> árvore (worker) -> painel."""
    session = _cf_tree_session()
    outs, errors = [], []

    gen = CounterfactualWorker(
        session, mode=CounterfactualWorker.STAGE_GENERATE,
        options={'method': 'LORE-LOCAL', 'instance_index': 0, 'seed': 1, 'n_cfs': 3},
    )
    gen.finished_ok.connect(outs.append)
    gen.failed.connect(errors.append)
    gen.run()
    assert not errors, errors
    cf_result = outs[0]['counterfactuals']
    assert any(c.get('metrics', {}).get('validity') for c in cf_result['candidates'])

    tree = CounterfactualWorker(
        session, mode=CounterfactualWorker.STAGE_TREE, cf_result=cf_result,
        options={'seed': 1, 'cf_tree_neighborhood_size': 120},
    )
    tree.finished_ok.connect(outs.append)
    tree.failed.connect(errors.append)
    tree.run()
    assert not errors, errors
    tree_result = outs[1]['counterfactuals']
    assert tree_result['result_type'] == 'counterfactual_tree'
    assert tree_result['tree_kind'] == 'CF-LocalTree'
    assert tree_result['methodology']['causality'] == 'NOT_CLAIMED'
    assert 'sample_weight' not in str(tree_result['methodology']['training'])
    assert tree_result['tree_rules']

    panel = CounterfactualPanel()
    panel.display_result(tree_result)
    assert panel.candidates_table.rowCount() >= 1
    panel.close()


def test_multiclass_without_explicit_target_uses_recorded_runner_up_via_service():
    """Regressão: multiclasse sem alvo na GUI dava INVALID_TARGET e nenhum CF."""
    from sklearn.ensemble import RandomForestClassifier
    from counterfactuals.service import generate_explanation_from_session

    rng = np.random.RandomState(5)
    X = rng.uniform(-2, 2, size=(400, 3))
    y = np.digitize(X[:, 0] + 0.3 * X[:, 1], [-0.7, 0.7])
    oracle = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    names = ['f0', 'f1', 'f2']
    session = {
        'dataset_name': 'dataset_carregado.arff', 'mlp_oracle': oracle, 'mlp_original': oracle,
        'X_train_enc': X, 'y_train_enc': y, 'X_train_original': X, 'y_train_original': y,
        'tree_a': oracle, 'tree_b': oracle, 'transformed_feature_names': names,
        'feature_names_original': names, 'tree_a_feature_names': names, 'tree_b_feature_names': names,
    }
    base = {'method': 'COGS', 'instance_index': 0, 'seed': 1, 'desired_class': None, 'max_time': 10}
    strict = generate_explanation_from_session(session, dict(base))
    assert strict['status'] == 'INVALID_TARGET'              # API estrita mantém a regra
    auto = generate_explanation_from_session(session, dict(base, multiclass_auto_target=True))
    assert auto['status'] == 'SUCCESS', auto['message']
    assert auto['desired_class'] != auto['factual_prediction']
    assert any('2.ª classe mais provável' in w for w in auto['warnings'])
    assert all(c['metrics']['validity'] for c in auto['candidates'])
