"""
Roteamento explícito de feature_space para Trepan-Reloaded.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from core.feature_alignment import oracle_n_features
from core.model_bundle import ModelBundle, assert_train_test_schema, is_residual_oracle


ORACLE_MLP_ORIGINAL = "MLP Original"
ORACLE_MLP_ONTO = "MLP_Onto"
ORACLE_MLP_RESIDUAL = "MLP Residual Ontológico"
ORACLE_HYBRID = "Oráculo Híbrido Ontológico OOF"
ORACLE_PROJECTED_ORIGINAL = "MLP Original projetado no espaço OWL"


def _split_or_empty(split: Optional[dict]) -> Optional[dict]:
    if not split:
        return None
    required = ("X_train", "X_test", "y_train", "y_test")
    if not all(k in split for k in required):
        return None
    return split


def resolve_reloaded_oracle_key(
    selected_oracle_label: Optional[str],
    ontology_enabled: bool,
    ontology_accepted: bool,
) -> str:
    """mlp_original | mlp_onto | mlp_residual"""
    if not ontology_enabled or not ontology_accepted:
        return "mlp_original"

    label = (selected_oracle_label or ORACLE_MLP_ORIGINAL).strip()
    if label == ORACLE_MLP_RESIDUAL:
        return "mlp_residual"
    if label == ORACLE_MLP_ONTO:
        return "mlp_onto"
    if label in {ORACLE_HYBRID, ORACLE_PROJECTED_ORIGINAL}:
        return "mlp_hybrid"
    return "mlp_original"


def get_trepan_reloaded_context(
    *,
    ontology_enabled: bool,
    ontology_acceptance: Optional[dict],
    selected_oracle_label: Optional[str],
    selected_oracle=None,
    mlp_original=None,
    mlp_onto=None,
    mlp_residual_pipeline=None,
    mlp_residual_oracle=None,
    eval_split_original: Optional[dict] = None,
    eval_split_enriched: Optional[dict] = None,
    eval_split_residual: Optional[dict] = None,
    feature_names_original: Optional[List[str]] = None,
    feature_names_enriched: Optional[List[str]] = None,
    feature_names_residual: Optional[List[str]] = None,
    bundle_original: Optional[ModelBundle] = None,
    bundle_onto: Optional[ModelBundle] = None,
    bundle_residual: Optional[ModelBundle] = None,
) -> Dict[str, Any]:
    """
    Devolve oráculo + matrizes + feature_space coerentes para Trepan-Reloaded.
    Nunca infere espaço apenas pelo número de colunas.
    """
    ontology_accepted = bool(ontology_acceptance and ontology_acceptance.get("accepted"))
    ontology_structural_available = bool(
        ontology_enabled
        and ontology_acceptance
        and ontology_acceptance.get(
            "ontology_structural_available", ontology_acceptance.get("accepted", False)
        )
    )
    oracle_key = resolve_reloaded_oracle_key(
        selected_oracle_label, ontology_enabled, ontology_accepted
    )

    split_orig = _split_or_empty(eval_split_original)
    split_enr = _split_or_empty(eval_split_enriched)
    split_res = _split_or_empty(eval_split_residual)

    if oracle_key == "mlp_hybrid":
        if selected_oracle is None:
            raise ValueError("Oráculo híbrido/projetado seleccionado, mas indisponível.")
        if split_enr is None:
            raise ValueError("Oráculo híbrido exige eval_split_enriched.")
        assert_train_test_schema(
            split_enr["X_train"], split_enr["X_test"], ORACLE_HYBRID
        )
        names = feature_names_enriched or (
            bundle_onto.feature_names if bundle_onto else None
        )
        if not names or len(names) != int(split_enr["X_train"].shape[1]):
            raise ValueError("Schema enriquecido ausente para o oráculo híbrido.")
        label = selected_oracle_label or ORACLE_HYBRID
        wrapper_bundle = ModelBundle(
            name=label, pipeline=selected_oracle,
            feature_names=list(names), n_features=len(names),
            feature_space="enriched",
        )
        return {
            "oracle": selected_oracle,
            "oracle_name": label,
            "oracle_key": "mlp_hybrid",
            "X_train": np.asarray(split_enr["X_train"], dtype=float),
            "X_test": np.asarray(split_enr["X_test"], dtype=float),
            "y_train": np.asarray(split_enr["y_train"]),
            "y_test": np.asarray(split_enr["y_test"]),
            "feature_names": list(names),
            "feature_space": "enriched",
            "oracle_bundle": wrapper_bundle,
            "mlp_model_onto": selected_oracle,
            "ontology_enabled": ontology_enabled,
            "ontology_accepted": ontology_accepted,
            "ontology_structural_available": ontology_structural_available,
            "use_ontology_semantic_pipeline": ontology_structural_available,
            "residual_oracle_wrapper": selected_oracle,
            "oof_teacher_probabilities": getattr(
                selected_oracle, "training_oof_probabilities_", None
            ),
        }

    if oracle_key == "mlp_residual":
        if mlp_residual_pipeline is None or mlp_residual_oracle is None:
            raise ValueError(
                "MLP Residual Ontológico seleccionado, mas o pipeline residual "
                "ou o wrapper enriquecido não está disponível."
            )
        if split_enr is None:
            raise ValueError(
                "MLP Residual Ontológico exige eval_split_enriched para que o "
                "TREPAN Reloaded preserve os atributos OWL."
            )
        if split_res is None:
            raise ValueError(
                "MLP Residual Ontológico seleccionado mas eval_split_residual em falta."
            )
        assert_train_test_schema(
            split_res["X_train"], split_res["X_test"], ORACLE_MLP_RESIDUAL
        )
        assert_train_test_schema(
            split_enr["X_train"], split_enr["X_test"], ORACLE_MLP_RESIDUAL
        )
        names = feature_names_enriched or (
            bundle_onto.feature_names if bundle_onto else None
        )
        if not names or len(names) != int(split_enr["X_train"].shape[1]):
            raise ValueError(
                "Schema enriquecido ausente/incompatível para o oráculo residual."
            )
        wrapper_bundle = ModelBundle(
            name=ORACLE_MLP_RESIDUAL,
            pipeline=mlp_residual_oracle,
            feature_names=list(names),
            n_features=len(names),
            feature_space="enriched",
        )
        return {
            "oracle": mlp_residual_oracle,
            "oracle_name": ORACLE_MLP_RESIDUAL,
            "oracle_key": "mlp_residual",
            "X_train": np.asarray(split_enr["X_train"], dtype=float),
            "X_test": np.asarray(split_enr["X_test"], dtype=float),
            "y_train": np.asarray(split_enr["y_train"]),
            "y_test": np.asarray(split_enr["y_test"]),
            "feature_names": list(names),
            "feature_space": "enriched",
            "oracle_bundle": wrapper_bundle,
            "residual_pipeline_bundle": bundle_residual,
            "mlp_model_onto": mlp_residual_oracle,
            "ontology_enabled": ontology_enabled,
            "ontology_accepted": ontology_accepted,
            "ontology_structural_available": ontology_structural_available,
            "use_ontology_semantic_pipeline": ontology_structural_available,
            "residual_oracle_wrapper": mlp_residual_oracle,
        }

    if oracle_key == "mlp_onto":
        if mlp_onto is None:
            raise ValueError("MLP_Onto seleccionado, mas o modelo não está disponível.")
        if split_enr is None:
            raise ValueError("MLP_Onto seleccionado mas eval_split_enriched em falta.")
        else:
            assert_train_test_schema(
                split_enr["X_train"], split_enr["X_test"], ORACLE_MLP_ONTO
            )
            names = feature_names_enriched or (
                bundle_onto.feature_names if bundle_onto else None
            )
            if not names:
                n = int(split_enr["X_train"].shape[1])
                names = [f"feature_{i}" for i in range(n)]
            return {
                "oracle": mlp_onto,
                "oracle_name": ORACLE_MLP_ONTO,
                "oracle_key": "mlp_onto",
                "X_train": np.asarray(split_enr["X_train"], dtype=float),
                "X_test": np.asarray(split_enr["X_test"], dtype=float),
                "y_train": np.asarray(split_enr["y_train"]),
                "y_test": np.asarray(split_enr["y_test"]),
                "feature_names": list(names),
                "feature_space": "enriched",
                "oracle_bundle": bundle_onto,
                "mlp_model_onto": mlp_onto,
                "ontology_enabled": ontology_enabled,
                "ontology_accepted": ontology_accepted,
                "ontology_structural_available": ontology_structural_available,
                "use_ontology_semantic_pipeline": ontology_structural_available,
                "residual_oracle_wrapper": None,
            }

    # mlp_original — ontologia rejeitada ou oráculo original seleccionado
    if split_orig is None:
        raise ValueError("eval_split_original em falta para Trepan-Reloaded.")
    assert_train_test_schema(
        split_orig["X_train"], split_orig["X_test"], ORACLE_MLP_ORIGINAL
    )
    names = feature_names_original or (
        bundle_original.feature_names if bundle_original else None
    )
    if not names:
        n = int(split_orig["X_train"].shape[1])
        names = [f"feature_{i}" for i in range(n)]

    return {
        "oracle": mlp_original,
        "oracle_name": ORACLE_MLP_ORIGINAL,
        "oracle_key": "mlp_original",
        "X_train": np.asarray(split_orig["X_train"], dtype=float),
        "X_test": np.asarray(split_orig["X_test"], dtype=float),
        "y_train": np.asarray(split_orig["y_train"]),
        "y_test": np.asarray(split_orig["y_test"]),
        "feature_names": list(names),
        "feature_space": "original",
        "oracle_bundle": bundle_original,
        "mlp_model_onto": None,
        "ontology_enabled": ontology_enabled,
        "ontology_accepted": ontology_accepted,
        "ontology_structural_available": ontology_structural_available,
        "use_ontology_semantic_pipeline": ontology_structural_available,
        "residual_oracle_wrapper": None,
    }


def oracle_expected_features(ctx: Dict[str, Any]) -> Optional[int]:
    bundle = ctx.get("oracle_bundle")
    if bundle is not None:
        return int(bundle.n_features)
    oracle = ctx.get("oracle")
    if oracle is None:
        return None
    n = oracle_n_features(oracle)
    return int(n) if n is not None else None


def log_trepan_reloaded_context(ctx: Dict[str, Any]) -> None:
    expected = oracle_expected_features(ctx)
    print("\n[TREPAN RELOADED CONTEXT]")
    print(f"Ontology enabled: {ctx.get('ontology_enabled')}")
    print(f"Ontology accepted: {ctx.get('ontology_accepted')}")
    print(f"Selected oracle: {ctx.get('oracle_name')}")
    bundle = ctx.get("oracle_bundle")
    oracle_space = bundle.feature_space if bundle else ctx.get("feature_space")
    print(f"Oracle feature_space: {oracle_space}")
    print(f"Oracle expected features: {expected}")
    print(f"Trepan feature_space: {ctx.get('feature_space')}")
    print(f"X_train shape: {ctx.get('X_train').shape if ctx.get('X_train') is not None else None}")
    print(f"X_test shape: {ctx.get('X_test').shape if ctx.get('X_test') is not None else None}")
    fn = ctx.get("feature_names") or []
    print(f"Feature names count: {len(fn)}")
