"""Regressão das capacidades contrafactuais incorporadas da versão experimental."""
from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from counterfactuals.cf_tree import build_counterfactual_tree
from counterfactuals.evaluation import evaluate_generation_result
from counterfactuals.export import export_counterfactual_result
from counterfactuals.global_rules import (
    analyse_global_rules,
    extract_global_rules,
    generate_global_counterfactuals,
)
from counterfactuals.service import generate_global_counterfactuals_from_session


@pytest.fixture
def iris_models():
    X, y = load_iris(return_X_y=True)
    oracle = LogisticRegression(max_iter=500, random_state=7).fit(X, y)
    tree = DecisionTreeClassifier(max_depth=4, random_state=7).fit(X, oracle.predict(X))
    names = [f"feature_{index}" for index in range(X.shape[1])]
    return X, y, oracle, tree, names


def test_global_rules_have_real_nonzero_differences(iris_models):
    X, _, oracle, tree, names = iris_models
    rules = extract_global_rules(tree, names)
    counterfactuals = generate_global_counterfactuals(rules, max_per_rule=3)
    assert len(rules) >= 3
    assert counterfactuals
    assert all(item.n_changes > 0 for item in counterfactuals)
    result = analyse_global_rules(tree, oracle, X, names, dataset_name="iris.arff")
    assert result["scope"] == "loaded_dataset_only"
    assert result["dataset"] == "iris.arff"
    assert 0.0 <= result["aggregate_metrics"]["global_fidelity"] <= 1.0
    assert "symbolic_stability_proxy" in result["aggregate_metrics"]
    assert "robustez empírica" in result["metric_notes"]["symbolic_stability_proxy"]


def test_global_service_uses_selected_tree_and_loaded_dataset(iris_models):
    X, y, oracle, tree, names = iris_models
    session = {
        "dataset_name": "apenas_este.arff",
        "mlp_oracle": oracle,
        "mlp_original": oracle,
        "X_train_enc": X,
        "y_train_enc": y,
        "X_train_original": X,
        "y_train_original": y,
        "transformed_feature_names": names,
        "feature_names_original": names,
        "tree_a": tree,
        "tree_b": tree,
        "c45_tree": tree,
        "class_labels": {0: "setosa", 1: "versicolor", 2: "virginica"},
    }
    result = generate_global_counterfactuals_from_session(
        session, {"target_model": "Trepan Original", "global_max_per_rule": 2}
    )
    assert result["dataset"] == "apenas_este.arff"
    assert result["target_model"] == "Trepan Original"
    with pytest.raises(ValueError, match="exigem"):
        generate_global_counterfactuals_from_session(
            session, {"target_model": "MLP Original"}
        )


def _generation_result(X, oracle, names):
    original = X[0].copy()
    factual = oracle.predict(original.reshape(1, -1))[0]
    candidates = []
    for row in X:
        prediction = oracle.predict(row.reshape(1, -1))[0]
        if prediction != factual:
            candidates.append({
                "method": "DICE",
                "prediction": int(prediction),
                "vector": row.tolist(),
                "changes": [],
                "metrics": {"validity": True},
            })
        if len(candidates) == 4:
            break
    return {
        "method": "DICE",
        "target_model": "Trepan Original",
        "dataset": "iris.arff",
        "original_instance": original.tolist(),
        "feature_names": names,
        "candidates": candidates,
    }


def test_counterfactual_tree_is_evaluated_and_does_not_mutate_original(iris_models):
    X, _, oracle, tree, names = iris_models
    generation = _generation_result(X, oracle, names)
    before_features = tree.tree_.feature.copy()
    before_thresholds = tree.tree_.threshold.copy()
    result = build_counterfactual_tree(
        oracle, X, generation, names,
        original_tree=tree,
        max_depth=3,
        neighborhood_size=100,
        dataset_name="iris.arff",
    )
    assert result["result_type"] == "counterfactual_tree"
    assert result["scope"] == "loaded_dataset_only"
    assert result["aggregate_metrics"]["n_valid_counterfactuals"] == 4
    assert 0.0 <= result["aggregate_metrics"]["fidelity_to_oracle"] <= 1.0
    assert result["tree_rules"]
    assert np.array_equal(tree.tree_.feature, before_features)
    assert np.array_equal(tree.tree_.threshold, before_thresholds)
    assert result["_runtime_tree_model"] is not tree


def test_formal_metrics_use_multiclass_probability_margin(iris_models):
    X, y, oracle, _, names = iris_models
    generation = _generation_result(X, oracle, names)
    evaluation = evaluate_generation_result(generation, oracle, X, y, seed=11)
    assert evaluation["per_candidate"]
    row = evaluation["per_candidate"][0]
    vector = np.asarray(generation["candidates"][0]["vector"]).reshape(1, -1)
    probabilities = oracle.predict_proba(vector)[0]
    predicted_index = int(np.argmax(probabilities))
    expected = probabilities[predicted_index] - np.max(np.delete(probabilities, predicted_index))
    assert row["probability_margin"] == pytest.approx(expected)
    assert row["proximity_l1"] >= row["proximity_l2"]
    assert 0.0 <= row["sparsity_normalized"] <= 1.0


def test_export_supports_global_rules_and_runtime_tree(tmp_path, iris_models):
    X, _, oracle, tree, names = iris_models
    global_result = analyse_global_rules(tree, oracle, X, names)
    global_paths = export_counterfactual_result(global_result, tmp_path, stem="global")
    assert "Regra factual" in open(global_paths["markdown"], encoding="utf-8").read()
    assert open(global_paths["csv"], encoding="utf-8-sig").read().strip()

    tree_result = build_counterfactual_tree(
        oracle, X, _generation_result(X, oracle, names), names, original_tree=tree,
    )
    tree_paths = export_counterfactual_result(tree_result, tmp_path, stem="tree")
    payload = json.loads(open(tree_paths["json"], encoding="utf-8").read())
    assert "_runtime_tree_model" not in payload
    assert payload["result_type"] == "counterfactual_tree"
