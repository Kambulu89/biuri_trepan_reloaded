"""MLPTrainer (caminho interativo da GUI): categorias só vistas fora do treino e numéricos ausentes.

Reprodução (dataset clínico com categorias raras e colunas muito incompletas): os encoders eram ajustados só no split de treino e
``LabelEncoder.transform`` falhava com «y contains previously unseen labels»; um numérico ausente chegava ao MLP como NaN.
Política genérica, igual à do pipeline de produção (``DataPreprocessor``) e ajustada só no treino: categoria não vista -> categoria mais
frequente do treino; numérico ausente -> mediana do treino. Sem ausentes nem categorias novas a codificação é bit a bit a anterior.
"""
import numpy as np
import pytest

from core.mlp_trainer import MLPTrainer


def _fit(X, train_rows):
    t = MLPTrainer()
    t._fit_encoders(np.asarray(X, dtype=object)[train_rows])
    return t


def test_unseen_category_is_encoded_as_the_training_mode():
    X = np.array([["a", 1.0], ["a", 2.0], ["b", 3.0], ["zzz", 4.0]], dtype=object)
    t = _fit(X, [0, 1, 2])
    enc = t._encode_features(X)
    le = t.feature_encoders[0]
    assert enc[3, 0] == le.transform(["a"])[0]                     # 'a' é a moda do treino
    assert list(enc[:3, 0]) == list(le.transform(["a", "a", "b"]))  # categorias vistas: códigos inalterados
    assert t.encoding_events == {0: {"ausentes_imputados": 0, "categorias_nao_vistas": 1}}


def test_missing_numeric_is_imputed_with_the_training_median_only():
    X = np.array([["a", 1.0], ["b", 3.0], ["a", 100.0], ["b", np.nan], ["a", ""]], dtype=object)
    t = _fit(X, [0, 1, 3])                                          # o 100 (fora do treino) não entra na mediana
    enc = t._encode_features(X)
    assert t.numeric_fill[1] == 2.0
    assert enc[3, 1] == 2.0 and enc[4, 1] == 2.0 and enc[2, 1] == 100.0
    assert not np.isnan(enc).any()


def test_without_missing_or_unseen_values_the_encoding_is_unchanged():
    rng = np.random.default_rng(0)
    cats = rng.choice(["x", "y", "w"], size=60)
    X = np.column_stack([cats, rng.normal(size=60)]).astype(object)
    t = _fit(X, np.arange(60))
    enc = t._encode_features(X)
    assert np.array_equal(enc[:, 0], t.feature_encoders[0].transform([str(c) for c in cats]))
    assert np.array_equal(enc[:, 1], X[:, 1].astype(float))
    assert t.encoding_events == {}


def test_full_training_with_rare_categories_and_missing_values_succeeds():
    rng = np.random.default_rng(1)
    n = 80
    cat = np.array([f"raro_{i}" if i % 9 == 0 else ("p" if i % 2 else "q") for i in range(n)], dtype=object)
    num = rng.normal(size=n).astype(object)
    num[::5] = np.nan
    y = np.array(["Si" if i % 4 == 0 else "No" for i in range(n)])
    t = MLPTrainer()
    t.set_metadata("d", ["cat", "num"], "y", ["Si", "No"], n)
    model, X_enc, _ = t.train(np.column_stack([cat, num]), y, optimize=False)
    assert model is not None and not np.isnan(X_enc).any()
    assert t.encoding_events[1]["ausentes_imputados"] == 16


def test_legacy_encoder_without_mode_keeps_the_previous_error():
    """Encoders vindos de uma cache antiga (sem moda registada) mantêm o comportamento anterior: erro explícito."""
    X = np.array([["a"], ["b"], ["novo"]], dtype=object)
    t = _fit(X, [0, 1])
    del t.feature_encoders[0]._biuri_fill_code
    with pytest.raises(ValueError, match="unseen"):
        t._encode_features(X)


def test_logs_print_nd_for_metrics_that_were_not_computed(capsys):
    """Métrica ausente no registo: «N/D», nunca 0.0 % (que se lê como resultado)."""
    from core.mlp_optimizer import log_mlp_optimization
    from core.pipeline_audit import log_ontology_acceptance
    log_mlp_optimization({"model_name": "MLP", "test_metrics": {"accuracy": 0.5}})
    log_ontology_acceptance({"ontology_quality_accepted": True, "accuracy_original": 0.9})
    out = capsys.readouterr().out
    assert "Test accuracy: 0.5000" in out and "Test f1: N/D" in out
    assert "MLP Original accuracy: 90.0%" in out and "MLP Onto/Residual accuracy: N/D" in out and "accuracy_gain: N/D" in out
