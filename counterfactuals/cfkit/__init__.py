"""cfkit: pipeline contrafactual unificado, agnóstico a datasets e auditável.

Princípios (ver IMPLEMENTATION_REPORT_COUNTERFACTUALS.md):
* todo CF devolvido é re-validado no modelo explicado;
* constraints hard nunca são violadas num SUCCESS; soft entram como penalização;
* ranges/densidade aprendidos SÓ no treino; sem ontologia -> semantic_validation=NOT_AVAILABLE;
* LORE/CLEAR/CoGS são rotulados *-inspired* (divergências documentadas em ``METHOD_REGISTRY``);
* constraints semânticas NÃO são causalidade.
"""
from counterfactuals.cfkit.api import (  # noqa: F401
    CounterfactualRequestError, compare_methods, generate_counterfactual, run_cf_benchmark,
)
from counterfactuals.cfkit.result import CounterfactualResult, CounterfactualStatus  # noqa: F401
from counterfactuals.cfkit.rules import ConstraintSet, OntologyConstraintExtractor  # noqa: F401
from counterfactuals.cfkit.schema import FeatureSchema, FeatureSpec  # noqa: F401
