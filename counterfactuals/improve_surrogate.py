"""Compatibilidade para a API histórica de melhoria de substitutos."""

from counterfactuals.surrogate_improvement import (
    AugmentationBatch,
    CandidateParameters,
    ImprovementSearchConfig,
    TrainingRequest,
    evaluate_improvement,
    evaluate_surrogate,
    generate_counterfactual_pool,
    improve_surrogate,
    prepare_counterfactual_augmentation,
    select_and_refit_improved_surrogate,
)

__all__ = [
    "AugmentationBatch",
    "CandidateParameters",
    "ImprovementSearchConfig",
    "TrainingRequest",
    "evaluate_improvement",
    "evaluate_surrogate",
    "generate_counterfactual_pool",
    "improve_surrogate",
    "prepare_counterfactual_augmentation",
    "select_and_refit_improved_surrogate",
]
