"""
Diagnóstico obrigatório do MLP Original — detecção de oráculo degenerado.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from core.feature_alignment import oracle_n_features
from core.model_bundle import (
    assert_train_test_schema,
    log_model_input_check,
    safe_oracle_predict,
    select_split_for_oracle,
)


def _distribution(y: np.ndarray) -> Dict[Any, int]:
    y_arr = np.asarray(y)
    unique, counts = np.unique(y_arr, return_counts=True)
    return {u: int(c) for u, c in zip(unique, counts)}


def _format_distribution(dist: Dict[Any, int]) -> str:
    if not dist:
        return "{}"
    parts = [f"{k}: {v}" for k, v in sorted(dist.items(), key=lambda x: str(x[0]))]
    return "{" + ", ".join(parts) + "}"


def diagnose_mlp_original(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dummy_tolerance: float = 0.01,
) -> Dict[str, Any]:
    """
    Calcula métricas completas e flags de degeneração do MLP Original.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    assert_train_test_schema(X_train, X_test, "MLP Original")
    y_pred_train = np.asarray(safe_oracle_predict(model, X_train))
    y_pred_test = np.asarray(safe_oracle_predict(model, X_test))

    unique_pred_train = np.unique(y_pred_train)
    unique_pred_test = np.unique(y_pred_test)
    single_class_train = len(unique_pred_train) == 1
    single_class_test = len(unique_pred_test) == 1
    degenerate = single_class_train or single_class_test

    accuracy = float(accuracy_score(y_test, y_pred_test))
    precision_w = float(
        precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
    )
    recall_w = float(
        recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
    )
    f1_w = float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred_test))

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)
    dummy_accuracy = float(accuracy_score(y_test, y_dummy))
    dummy_f1 = float(f1_score(y_test, y_dummy, average="weighted", zero_division=0))

    weak_vs_dummy = (
        accuracy <= dummy_accuracy + dummy_tolerance
        or degenerate
    )

    n_features = oracle_n_features(model)
    if n_features is None:
        n_features = getattr(model, "n_features_in_", X_train.shape[1])

    return {
        "X_train_shape": tuple(X_train.shape),
        "X_test_shape": tuple(X_test.shape),
        "mlp_n_features_in": n_features,
        "unique_y_train_real": int(len(np.unique(y_train))),
        "unique_y_test_real": int(len(np.unique(y_test))),
        "unique_y_pred_train": int(len(unique_pred_train)),
        "unique_y_pred_test": int(len(unique_pred_test)),
        "distribution_y_train": _distribution(y_train),
        "distribution_y_test": _distribution(y_test),
        "distribution_y_pred_train": _distribution(y_pred_train),
        "distribution_y_pred_test": _distribution(y_pred_test),
        "accuracy": accuracy,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        "classification_report": classification_report(
            y_test, y_pred_test, output_dict=True, zero_division=0
        ),
        "degenerate": degenerate,
        "single_class_train": single_class_train,
        "single_class_test": single_class_test,
        "weak_vs_dummy": weak_vs_dummy,
        "dummy_accuracy": dummy_accuracy,
        "dummy_f1_weighted": dummy_f1,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
    }


def log_mlp_original_diagnostic(diag: Dict[str, Any]) -> None:
    """Imprime bloco [MLP ORIGINAL DIAGNOSTIC]."""
    print("\n[MLP ORIGINAL DIAGNOSTIC]")
    print(f"X_train_original shape: {diag.get('X_train_shape')}")
    print(f"X_test_original shape: {diag.get('X_test_shape')}")
    print(f"MLP n_features_in_: {diag.get('mlp_n_features_in')}")
    print(f"Unique y_train real: {diag.get('unique_y_train_real')}")
    print(f"Unique y_test real: {diag.get('unique_y_test_real')}")
    print(f"Unique y_pred_mlp_train: {diag.get('unique_y_pred_train')}")
    print(f"Unique y_pred_mlp_test: {diag.get('unique_y_pred_test')}")
    print(f"Distribution y_train real: {_format_distribution(diag.get('distribution_y_train', {}))}")
    print(f"Distribution y_test real: {_format_distribution(diag.get('distribution_y_test', {}))}")
    print(
        f"Distribution y_pred_mlp_train: "
        f"{_format_distribution(diag.get('distribution_y_pred_train', {}))}"
    )
    print(
        f"Distribution y_pred_mlp_test: "
        f"{_format_distribution(diag.get('distribution_y_pred_test', {}))}"
    )
    print(f"Accuracy: {diag.get('accuracy', 0):.4f}")
    print(f"Precision weighted: {diag.get('precision_weighted', 0):.4f}")
    print(f"Recall weighted: {diag.get('recall_weighted', 0):.4f}")
    print(f"F1 weighted: {diag.get('f1_weighted', 0):.4f}")
    print(f"Balanced accuracy: {diag.get('balanced_accuracy', 0):.4f}")
    print(f"Confusion matrix:\n{np.array(diag.get('confusion_matrix', []))}")

    if diag.get("degenerate"):
        print(
            "\n[WARNING] MLP Original degenerado: o modelo está a prever apenas uma classe. "
            "Trepan resultará numa árvore folha única."
        )
    if diag.get("weak_vs_dummy") and not diag.get("degenerate"):
        print(
            "\n[WARNING] O MLP Original não superou o classificador majoritário. "
            "Rever hiperparâmetros, encoding, normalização ou distribuição das classes."
        )
    elif diag.get("weak_vs_dummy"):
        print(
            "\n[WARNING] O MLP Original não superou o classificador majoritário. "
            "Rever hiperparâmetros, encoding, normalização ou distribuição das classes."
        )


def diagnose_oracle(
    model,
    X_train: np.ndarray = None,
    X_test: np.ndarray = None,
    oracle_label: str = "Oracle",
    splits: Optional[dict] = None,
    bundle=None,
) -> Dict[str, Any]:
    """Diagnóstico rápido de oráculo (train + test predictions)."""
    if splits is not None:
        split = select_split_for_oracle(model, splits, bundle=bundle)
        if split is None:
            raise ValueError(
                f"{oracle_label}: no compatible eval split for oracle feature space."
            )
        X_train = split["X_train"]
        X_test = split["X_test"]

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    assert_train_test_schema(X_train, X_test, oracle_label)
    if bundle is not None:
        log_model_input_check(bundle, X_train, operation="predict")
    y_pred_train = np.asarray(
        safe_oracle_predict(model, X_train, bundle=bundle)
    )
    y_pred_test = np.asarray(
        safe_oracle_predict(model, X_test, bundle=bundle)
    )
    unique_train = int(len(np.unique(y_pred_train)))
    unique_test = int(len(np.unique(y_pred_test)))
    degenerate = unique_train == 1 or unique_test == 1
    return {
        "oracle_label": oracle_label,
        "unique_pred_train": unique_train,
        "unique_pred_test": unique_test,
        "distribution_train": _distribution(y_pred_train),
        "distribution_test": _distribution(y_pred_test),
        "degenerate": degenerate,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
    }


def log_oracle_diagnostic(diag: Dict[str, Any]) -> None:
    label = diag.get("oracle_label", "Oracle")
    print(f"\n[{label.upper().replace(' ', '_')} DIAGNOSTIC]")
    print(f"Unique y_pred_train: {diag.get('unique_pred_train')}")
    print(f"Unique y_pred_test: {diag.get('unique_pred_test')}")
    print(
        f"Distribution y_pred_train: "
        f"{_format_distribution(diag.get('distribution_train', {}))}"
    )
    print(
        f"Distribution y_pred_test: "
        f"{_format_distribution(diag.get('distribution_test', {}))}"
    )
    if diag.get("degenerate"):
        print(
            f"\n[WARNING] {label} degenerado: prevê apenas uma classe. "
            "Trepan resultará numa árvore folha única."
        )


def ensure_robust_mlp_original(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    existing_model=None,
    mode: str = "balanced",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Se MLP degenerado ou fraco, tenta reotimização; devolve (modelo, diagnóstico).
    """
    from core.mlp_optimizer import train_robust_mlp_original

    diag = diagnose_mlp_original(
        existing_model, X_train, y_train, X_test, y_test
    ) if existing_model is not None else None

    needs_retry = (
        diag is None
        or diag.get("degenerate")
        or diag.get("weak_vs_dummy")
    )

    if needs_retry:
        print("\n[INFO] A tentar reotimização automática do MLP Original...")
        result = train_robust_mlp_original(
            X_train, y_train, X_test, y_test, mode=mode
        )
        model = result["model"]
        diag = diagnose_mlp_original(model, X_train, y_train, X_test, y_test)
        diag["robust_retrained"] = True
        diag["robust_optimization_method"] = result.get("optimization_method")
        diag["robust_best_params"] = result.get("best_params")
        log_mlp_original_diagnostic(diag)
        if diag.get("degenerate"):
            print(
                "[WARNING] MLP Original continua degenerado após reotimização. "
                "Trepan será trivial (oráculo constante)."
            )
        return model, diag

    log_mlp_original_diagnostic(diag)
    diag["robust_retrained"] = False
    return existing_model, diag
