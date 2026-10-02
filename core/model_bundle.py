"""
ModelBundle — cada modelo MLP com pipeline e schema de features isolados.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.feature_alignment import oracle_n_features


@dataclass
class ModelBundle:
    name: str
    pipeline: object
    feature_names: List[str]
    n_features: int
    feature_space: str  # original | enriched | residual
    transformer_id: str = "unspecified"
    feature_origins: Optional[List[Dict[str, Any]]] = None

    @property
    def schema_fingerprint(self) -> str:
        payload = "\x1f".join(str(name) for name in self.feature_names)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def scaler_n_features_in(self) -> Optional[int]:
        pipe = self.pipeline
        if pipe is None:
            return None
        steps = getattr(pipe, "named_steps", None)
        if steps and "scaler" in steps:
            return getattr(steps["scaler"], "n_features_in_", None)
        return oracle_n_features(pipe)


def validate_model_input(
    bundle: ModelBundle,
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    operation: str = "predict",
    current_feature_space: Optional[str] = None,
) -> None:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(
            f"{bundle.name} ({operation}): expected 2D matrix, got shape {X.shape}."
        )
    expected = bundle.n_features
    received = int(X.shape[1])
    if expected != received:
        raise ValueError(
            f"Invalid feature space for {bundle.name}: "
            f"expects {expected} features from {bundle.feature_space}, "
            f"received {received}. "
            f"Current matrix feature_space={current_feature_space or 'unknown'}."
        )
    if feature_names is not None:
        supplied = [str(name) for name in feature_names]
        expected_names = [str(name) for name in bundle.feature_names]
        if supplied != expected_names:
            raise ValueError(
                f"Schema nominal incompatível para {bundle.name}: a ordem ou os "
                "nomes dos atributos diferem do treino."
            )
    scaler_n = bundle.scaler_n_features_in()
    if scaler_n is not None and int(scaler_n) != received:
        raise ValueError(
            f"{bundle.name} scaler expects {scaler_n} features, "
            f"but received matrix with {received} columns "
            f"(feature_space={bundle.feature_space}, operation={operation})."
        )


def log_model_input_check(
    bundle: Optional[ModelBundle],
    X: np.ndarray,
    *,
    using_cache: bool = False,
    operation: str = "fit",
) -> None:
    print("\n[MODEL INPUT CHECK]")
    if bundle is None:
        print("Model: (no bundle)")
        print(f"Received X shape: {getattr(np.asarray(X), 'shape', None)}")
        print(f"Using cache: {using_cache}")
        return
    X = np.asarray(X)
    print(f"Model: {bundle.name}")
    print(f"Feature space: {bundle.feature_space}")
    print(f"Schema hash: {bundle.schema_fingerprint}")
    print(f"Transformer: {bundle.transformer_id}")
    print(f"Expected n_features: {bundle.n_features}")
    print(f"Received X shape: {X.shape}")
    names_preview = bundle.feature_names[:8]
    if len(bundle.feature_names) > 8:
        names_preview = list(names_preview) + ["..."]
    print(f"Feature names: {names_preview}")
    print(f"Using cache: {using_cache}")
    print(f"Operation: {operation}")
    print(f"Scaler n_features_in_: {bundle.scaler_n_features_in()}")


def assert_train_test_schema(
    X_train: np.ndarray,
    X_test: np.ndarray,
    model_name: str = "MLP",
) -> None:
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            f"{model_name}: X_train/X_test devem ser matrizes 2D; "
            f"recebido {X_train.shape} e {X_test.shape}."
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"{model_name}: X_train has {X_train.shape[1]} features "
            f"but X_test has {X_test.shape[1]}."
        )


def bundle_from_pipeline(
    name: str,
    pipeline: object,
    feature_names: Sequence[str],
    feature_space: str,
    *,
    transformer_id: str = "pipeline",
    feature_origins: Optional[List[Dict[str, Any]]] = None,
) -> ModelBundle:
    n = oracle_n_features(pipeline)
    if n is None:
        n = len(feature_names)
    names = [str(item) for item in feature_names]
    if len(names) != int(n):
        raise ValueError(
            f"{name}: pipeline espera {n} atributos, mas o schema contém "
            f"{len(names)} nomes."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"{name}: o schema contém nomes duplicados.")
    return ModelBundle(
        name=name,
        pipeline=pipeline,
        feature_names=names,
        n_features=int(n),
        feature_space=feature_space,
        transformer_id=transformer_id,
        feature_origins=feature_origins,
    )


def bundle_predict(
    bundle: ModelBundle,
    X: np.ndarray,
    *,
    using_cache: bool = False,
    current_feature_space: Optional[str] = None,
) -> np.ndarray:
    log_model_input_check(bundle, X, using_cache=using_cache, operation="predict")
    validate_model_input(
        bundle, X, operation="predict", current_feature_space=current_feature_space
    )
    return bundle.pipeline.predict(X)


def is_residual_oracle(oracle) -> bool:
    return getattr(oracle, "ORACLE_TYPE", None) == "residual_ontological"


def infer_oracle_feature_space(oracle, default: str = "original") -> str:
    if is_residual_oracle(oracle):
        return "residual"
    meta = getattr(oracle, "arff_meta", None)
    if isinstance(meta, dict):
        space = meta.get("oracle_space")
        if space in ("augmented", "enriched"):
            return "enriched"
        if space == "residual_ontological":
            return "residual"
        if space == "original":
            return "original"
    pipe = getattr(oracle, "pipeline", oracle)
    bundle_n = oracle_n_features(pipe)
    if bundle_n is not None and bundle_n > 12:
        return "enriched"
    return default


def select_split_for_oracle(
    oracle,
    splits: dict,
    *,
    bundle: Optional[ModelBundle] = None,
    default_space: str = "original",
) -> Optional[dict]:
    """
    Escolhe o split correcto para predict/diagnóstico do oráculo.

    splits keys: original, enriched, augmented, residual
    """
    if bundle is not None:
        space = bundle.feature_space
        if space == "enriched":
            return splits.get("enriched") or splits.get("augmented")
        if space == "residual":
            return splits.get("residual")
        return splits.get("original")

    if is_residual_oracle(oracle):
        return splits.get("enriched") or splits.get("augmented")

    space = infer_oracle_feature_space(oracle, default=default_space)
    if space == "enriched":
        return splits.get("enriched") or splits.get("augmented")
    if space == "residual":
        return splits.get("residual")
    return splits.get("original")


def safe_oracle_predict(
    oracle,
    X: np.ndarray,
    *,
    bundle: Optional[ModelBundle] = None,
    using_cache: bool = False,
    current_feature_space: Optional[str] = None,
) -> np.ndarray:
    """Predict com validação de schema quando bundle disponível."""
    X = np.asarray(X, dtype=float)
    if bundle is not None:
        return bundle_predict(
            bundle, X,
            using_cache=using_cache,
            current_feature_space=current_feature_space or bundle.feature_space,
        )
    if is_residual_oracle(oracle):
        n_enriched = getattr(oracle, "n_features_in_", None)
        if n_enriched is not None and X.shape[1] != int(n_enriched):
            raise ValueError(
                f"ResidualOntologicalOracle expects enriched matrix with "
                f"{n_enriched} features, received {X.shape[1]} "
                f"(current_feature_space={current_feature_space})."
            )
        return oracle.predict(X)
    n_expected = oracle_n_features(oracle)
    if n_expected is not None and X.shape[1] != int(n_expected):
        raise ValueError(
            f"Invalid feature space for oracle: "
            f"expected {n_expected} features from pipeline, "
            f"received {X.shape[1]}. "
            f"Current matrix feature_space={current_feature_space or 'unknown'}."
        )
    try:
        return oracle.predict(X)
    except AttributeError as exc:
        # Compatibilidade controlada com estimadores de teste/legados que não
        # implementam tags sklearn 1.6, sem contornar validações de dimensão.
        if "__sklearn_tags__" not in str(exc) or not hasattr(oracle, "steps"):
            raise
        transformed = X
        for _, step in oracle.steps[:-1]:
            transformed = step.transform(transformed)
        return oracle.steps[-1][1].predict(transformed)
