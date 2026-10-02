"""Plano de ablação pareado e bloqueado ao conjunto de desenvolvimento.

O módulo não abre nem cria um teste externo. O chamador fornece um avaliador
do pipeline e recebe exatamente os mesmos folds para todas as variantes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


PRIMARY_METRICS = ("balanced_accuracy", "macro_f1", "minority_recall", "accuracy")


@dataclass(frozen=True)
class AblationVariant:
    name: str
    feature_space: str
    owl_enabled: bool
    teacher: str
    controls: Mapping[str, object] = field(default_factory=dict)


def required_ablation_variants() -> list[AblationVariant]:
    """Variantes obrigatórias, incluindo controlos e pares de componentes."""
    variants = [
        AblationVariant("MLP Original", "original", False, "none"),
        AblationVariant("TREPAN Original", "original", False, "mlp_original"),
        AblationVariant("C4.5", "original", False, "real_labels"),
        AblationVariant("Reloaded sem OWL", "original", False, "mlp_original"),
        AblationVariant("MLP Ontológico", "enriched", True, "none"),
        AblationVariant("Reloaded com OWL", "enriched", True, "hybrid_ontological"),
        AblationVariant(
            "Controlo aleatório", "original_plus_random", False,
            "same_protocol", {"semantic_control": "seeded_random_features"},
        ),
        AblationVariant(
            "Controlo sintético", "original_plus_synthetic", False,
            "same_protocol", {"semantic_control": "non_owl_transformations"},
        ),
    ]
    component_pairs = [
        ("reasoner", False, True),
        ("structural_matching", False, True),
        ("ontology_categories", False, True),
        ("disagreement_queries", False, True),
        ("counterfactuals", False, True),
        ("multiobjective_pruning", False, True),
    ]
    for component, disabled, enabled in component_pairs:
        for state, value in (("sem", disabled), ("com", enabled)):
            variants.append(AblationVariant(
                f"Reloaded OWL — {state} {component}", "enriched", True,
                "hybrid_ontological", {component: value},
            ))
    for mode in ("hard", "soft", "hybrid"):
        variants.append(AblationVariant(
            f"Reloaded OWL — destilação {mode}", "enriched", True,
            "hybrid_ontological", {"distillation": mode},
        ))
    return variants


def validate_ablation_coverage(variants: Sequence[AblationVariant]) -> None:
    names = [item.name for item in variants]
    if len(names) != len(set(names)):
        raise ValueError("O plano de ablação contém variantes duplicadas.")
    required_names = {item.name for item in required_ablation_variants()}
    missing = sorted(required_names - set(names))
    if missing:
        raise ValueError(f"Variantes obrigatórias ausentes: {missing}")


def run_development_ablation(
    X,
    y,
    evaluator: Callable[[AblationVariant, np.ndarray, np.ndarray, int], Mapping[str, float]],
    *,
    variants: Optional[Sequence[AblationVariant]] = None,
    folds: int = 5,
    repeats: int = 3,
    random_state: int = 42,
) -> dict:
    """Executa ablação OOF pareada; ``evaluator`` nunca recebe o teste externo."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X/y incompatíveis no estudo de ablação.")
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("A ablação exige duas classes e duas amostras por classe.")
    folds = max(2, min(int(folds), int(counts.min())))
    variants = list(variants or required_ablation_variants())
    validate_ablation_coverage(variants)
    splitter = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=int(repeats), random_state=random_state,
    )
    split_plan = list(splitter.split(X, y))
    rows = []
    for split_id, (fit_idx, val_idx) in enumerate(split_plan):
        for variant in variants:
            values = dict(evaluator(variant, fit_idx.copy(), val_idx.copy(), split_id))
            forbidden = {"test", "test_accuracy", "y_test", "X_test"} & set(values)
            if forbidden:
                raise ValueError(
                    f"O avaliador tentou expor o teste externo: {sorted(forbidden)}"
                )
            missing = [metric for metric in PRIMARY_METRICS if metric not in values]
            if missing:
                raise ValueError(f"Métricas obrigatórias ausentes: {missing}")
            if not all(np.isfinite(float(values[key])) for key in PRIMARY_METRICS):
                raise ValueError("A ablação produziu métricas não finitas.")
            rows.append({
                "split_id": split_id, "variant": variant.name,
                "fit_size": int(len(fit_idx)), "validation_size": int(len(val_idx)),
                "feature_space": variant.feature_space,
                "owl_enabled": variant.owl_enabled, "teacher": variant.teacher,
                "controls": dict(variant.controls),
                **{key: float(value) for key, value in values.items()},
            })
    summary: Dict[str, dict] = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant.name]
        summary[variant.name] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in selected])),
                "std": float(np.std([row[metric] for row in selected], ddof=1))
                if len(selected) > 1 else 0.0,
            }
            for metric in PRIMARY_METRICS
        }
    return {
        "scope": "development_repeated_stratified_oof",
        "external_test_used": False,
        "same_folds_for_all_variants": True,
        "folds": folds, "repeats": int(repeats), "random_state": random_state,
        "variants": [asdict(item) for item in variants],
        "rows": rows, "summary": summary,
    }


def confirmatory_benchmark_gate(
    *, tests_passed: bool, ontology_catalog: Sequence[Mapping[str, object]],
) -> dict:
    """Bloqueia confirmação sem testes limpos e ontologias externas auditáveis."""
    usable = [
        row for row in ontology_catalog
        if row.get("independent") is True
        and row.get("license")
        and row.get("sha256")
        and row.get("quality_status") == "VALID_DOMAIN_ONTOLOGY"
    ]
    approved = bool(tests_passed and usable)
    return {
        "approved": approved,
        "status": "READY_FOR_ONE_CONFIRMATORY_RUN" if approved else "BLOCKED",
        "tests_passed": bool(tests_passed),
        "usable_independent_ontologies": len(usable),
        "reason": None if approved else (
            "tests_not_passed" if not tests_passed
            else "no_independent_versioned_domain_ontology"
        ),
    }


__all__ = [
    "AblationVariant", "PRIMARY_METRICS", "required_ablation_variants",
    "validate_ablation_coverage", "run_development_ablation",
    "confirmatory_benchmark_gate",
]
