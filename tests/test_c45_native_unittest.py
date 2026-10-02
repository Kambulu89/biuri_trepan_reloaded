import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from core.c45_j48_tree import C45Classifier, C45Tree


class C45NativeTests(unittest.TestCase):
    def test_numeric_multiclass_accuracy_and_compatibility_view(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        model = C45Classifier(min_samples_leaf=2, random_state=42).fit(X_train, y_train)
        self.assertGreaterEqual(accuracy_score(y_test, model.predict(X_test)), 0.85)
        self.assertEqual(model.tree_.n_features, X.shape[1])
        self.assertEqual(model.tree_.node_count, len(model.tree_.feature))
        self.assertAlmostEqual(float(model.feature_importances_.sum()), 1.0, places=7)

    def test_nominal_split_is_multiway_and_gain_ratio_avoids_id_bias(self):
        # A primeira feature é um identificador (ganho alto, SplitInfo muito alto).
        # A segunda separa perfeitamente as classes e deve vencer por Gain Ratio.
        ids = [f"id-{i}" for i in range(12)]
        signal = ["low"] * 6 + ["high"] * 6
        X = np.asarray(list(zip(ids, signal)), dtype=object)
        y = np.asarray([0] * 6 + [1] * 6)
        model = C45Classifier(
            feature_types=("categorical", "categorical"),
            min_samples_leaf=1,
            random_state=42,
        ).fit(X, y)
        self.assertEqual(model.root_.feature, 1)
        self.assertEqual(model.root_.kind, "categorical")
        self.assertEqual(len(model.root_.children), 2)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_missing_values_are_fractionally_distributed(self):
        X = np.asarray([[0.0], [0.2], [0.4], [1.0], [1.2], [1.4], [np.nan]])
        y = np.asarray([0, 0, 0, 1, 1, 1, 0])
        model = C45Classifier(min_samples_leaf=1).fit(X, y)
        self.assertEqual(len(model.predict([[np.nan]])), 1)
        self.assertTrue(np.isfinite(model.predict_proba([[np.nan]])).all())
        self.assertAlmostEqual(float(model.predict_proba([[np.nan]]).sum()), 1.0)

    def test_wrapper_identifies_real_algorithm_and_rejects_ontology(self):
        X, y = load_iris(return_X_y=True)
        wrapper = C45Tree()
        model = wrapper.train_c45_tree(
            X, y, feature_names=load_iris().feature_names,
            class_names=load_iris().target_names,
            feature_types=["numeric"] * X.shape[1],
        )
        self.assertTrue(model.is_c45_native)
        info = wrapper.get_tree_info()
        self.assertEqual(info["algorithm"], "C4.5-Nativo")
        self.assertEqual(info["split_criterion"], "gain_ratio")
        self.assertEqual(info["pruning"], "pessimistic_error_pruning")
        with self.assertRaises(ValueError):
            wrapper.train_c45_tree(X, y, ontology_mode=True)


if __name__ == "__main__":
    unittest.main()
