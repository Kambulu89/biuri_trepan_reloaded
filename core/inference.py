"""API headless de inferência para artefactos V9.2."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import pandas as pd
from core.artifacts import load_artifact_payload
from core.scientific_errors import ArtifactCompatibilityError


def _dense_2d(value):
    """Normaliza a saída do preprocessor sem assumir uma implementação concreta."""
    if hasattr(value, "toarray"):
        value = value.toarray()
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


@dataclass
class Predictor:
    model: Any
    preprocessor: Any
    classes: Any
    manifest: Dict[str, Any]
    tree: Any = None

    def _X(self, df):
        value = self.preprocessor.transform(df) if self.preprocessor is not None else df
        return _dense_2d(value)

    def validate(self, df) -> Dict[str, Any]:
        try:
            self._X(df)
            return {"valid": True, "problems": []}
        except Exception as exc:
            return {"valid": False, "problems": [str(exc)]}

    def predict(self, df):
        return self.model.predict(self._X(df))

    def predict_proba(self, df):
        if not hasattr(self.model, "predict_proba"):
            raise ArtifactCompatibilityError("O modelo não disponibiliza predict_proba().")
        return self.model.predict_proba(self._X(df))

    @staticmethod
    def _unwrap_tree(tree):
        return getattr(tree, "explainer_tree", None) or tree

    def explain(self, row):
        frame = row if isinstance(row, pd.DataFrame) else pd.DataFrame([row])
        X = self._X(frame)
        tree = self._unwrap_tree(self.tree)
        if tree is None:
            return {"available": False, "reason": "artefacto_sem_arvore"}

        # TREPAN Original/Reloaded histórico: preserva a semântica m-of-n e
        # devolve apenas o caminho realmente percorrido pela instância.
        if hasattr(tree, "root_") and hasattr(tree, "export_rules"):
            node = tree.root_
            steps = []
            row_values = X[:1]
            while not node.is_leaf:
                outcome = bool(node.test.evaluate(row_values)[0])
                steps.append({
                    "node": int(getattr(node, "node_id", -1)),
                    "condition": node.test.text(getattr(tree, "feature_names_in_", None)),
                    "outcome": outcome,
                    "branch": "sim" if outcome else "não",
                    "m": int(node.test.m),
                    "n": int(len(node.test.literals)),
                })
                node = node.true_child if outcome else node.false_child
            prediction = tree.predict(X)[0]
            if isinstance(prediction, np.generic):
                prediction = prediction.item()
            return {
                "available": True,
                "tree_family": "trepan_historical",
                "prediction": prediction,
                "path": steps,
                "leaf": {
                    "node": int(getattr(node, "node_id", -1)),
                    "prediction": node.prediction.item() if isinstance(node.prediction, np.generic) else node.prediction,
                    "probabilities": [float(v) for v in np.asarray(node.distribution, dtype=float)],
                    "support": int(len(node.real_y)),
                    "reach": float(node.reach),
                },
            }

        # Árvore sklearn: devolve o caminho com nomes transformados quando disponíveis.
        if hasattr(tree, "decision_path") and hasattr(tree, "tree_"):
            node_ids = tree.decision_path(X).indices.tolist()
            names = list(self.preprocessor.get_feature_names_out()) if self.preprocessor else []
            steps = []
            for nid in node_ids:
                f = int(tree.tree_.feature[nid])
                if f >= 0:
                    threshold = float(tree.tree_.threshold[nid])
                    value = float(X[0, f])
                    steps.append({
                        "node": int(nid),
                        "feature": names[f] if f < len(names) else f,
                        "threshold": threshold,
                        "value": value,
                        "outcome": bool(value <= threshold),
                    })
            prediction = tree.predict(X)[0]
            if isinstance(prediction, np.generic):
                prediction = prediction.item()
            return {"available": True, "tree_family": "sklearn_tree", "prediction": prediction, "path": steps}

        return {"available": False, "reason": "tipo_de_arvore_sem_explicador"}


def load_artifact(path, *, strict_versions: bool = True) -> Predictor:
    payload, manifest = load_artifact_payload(path, strict_versions=strict_versions)
    return Predictor(
        payload["model"], payload.get("preprocessor"), payload.get("classes"), manifest, payload.get("tree")
    )


__all__ = ["Predictor", "load_artifact"]
