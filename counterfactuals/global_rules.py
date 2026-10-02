"""Contrafactuais globais sobre árvores sklearn e TREPAN histórico.

Uma única implementação serve C4.5-Nativo, TREPAN Original e TREPAN Reloaded sem converter regras m-of-n para CART.
As regras são extraídas sem alterar a árvore e todas as métricas são calculadas
exclusivamente sobre o dataset presente na sessão activa.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RuleCondition:
    feature_index: int
    feature: str
    operator: str
    threshold: float
    kind: str = "literal"
    m: int = 0
    literals: Tuple[Tuple[int, str, str, float], ...] = ()
    negated: bool = False

    def text(self) -> str:
        if self.kind == 'm_of_n':
            body = '; '.join(
                f"{name} {op} {threshold:.6g}"
                for _, name, op, threshold in self.literals
            )
            prefix = 'NÃO ' if self.negated else ''
            return f"{prefix}{self.m}-of-{len(self.literals)}({body})"
        return f"{self.feature} {self.operator} {self.threshold:.6g}"

    def matches(self, row: np.ndarray) -> bool:
        if self.kind == 'm_of_n':
            votes = 0
            for index, _name, op, threshold in self.literals:
                value = row[int(index)]
                votes += int(value > threshold) if op == '>' else int(value <= threshold)
            result = votes >= int(self.m)
            return (not result) if self.negated else result
        value = row[self.feature_index]
        if self.operator == '<=':
            return bool(value <= self.threshold)
        if self.operator == '>':
            return bool(value > self.threshold)
        raise ValueError(f'Operador de regra não suportado: {self.operator}')

    def signature(self) -> tuple:
        if self.kind == 'm_of_n':
            return ('m_of_n', int(self.m), bool(self.negated), tuple(self.literals))
        return ('literal', int(self.feature_index), self.operator, round(float(self.threshold), 12))

    def group_key(self) -> tuple:
        if self.kind == 'm_of_n':
            return ('m_of_n', tuple(item[0] for item in self.literals))
        return ('literal', int(self.feature_index))


@dataclass
class GlobalRule:
    rule_id: str
    conditions: List[RuleCondition]
    predicted_class: Any
    predicted_class_name: str
    support: float
    confidence: float

    def text(self) -> str:
        premise = " E ".join(item.text() for item in self.conditions) or "VERDADEIRO"
        return f"SE {premise}, ENTÃO classe {self.predicted_class_name}"


@dataclass
class ConditionChange:
    feature: str
    factual_condition: Optional[str]
    counterfactual_condition: Optional[str]
    change_type: str


@dataclass
class GlobalRuleCounterfactual:
    factual_rule: GlobalRule
    counterfactual_rule: GlobalRule
    changes: List[ConditionChange]

    @property
    def n_changes(self) -> int:
        return len(self.changes)


def unwrap_tree(model: Any) -> Any:
    """Resolve wrappers usados pelo projecto sem acoplar a uma classe concreta."""
    candidates = (
        model,
        getattr(model, "tree_model", None),
        getattr(model, "explainer_tree", None),
        getattr(getattr(model, "extractor", None), "explainer_tree", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, 'tree_'):
            return candidate
        if hasattr(candidate, 'root_') and hasattr(candidate, 'export_rules'):
            return candidate
    raise TypeError('O modelo seleccionado não contém uma árvore treinada suportada.')


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def extract_global_rules(
    model: Any,
    feature_names: Sequence[str],
    class_labels: Optional[Mapping[Any, Any]] = None,
) -> List[GlobalRule]:
    """Extrai uma regra raiz→folha por folha de sklearn ou TREPAN histórico."""
    tree_model = unwrap_tree(model)
    names = [str(name) for name in feature_names]
    if int(getattr(tree_model, 'n_features_in_', len(names))) != len(names):
        raise ValueError('A árvore e os nomes de features pertencem a espaços diferentes.')
    labels = dict(class_labels or {})
    rules: List[GlobalRule] = []

    if hasattr(tree_model, 'root_') and hasattr(tree_model, 'iter_nodes'):
        def historical_condition(test, outcome: bool) -> RuleCondition:
            literals = tuple(
                (
                    int(lit.feature),
                    names[int(lit.feature)] if int(lit.feature) < len(names) else f'feature_{lit.feature}',
                    '>' if lit.greater else '<=',
                    float(lit.threshold),
                )
                for lit in test.literals
            )
            return RuleCondition(
                feature_index=-1, feature=f'{test.m}-of-{len(test.literals)}',
                operator='m-of-n', threshold=0.0, kind='m_of_n',
                m=int(test.m), literals=literals, negated=not bool(outcome),
            )

        def walk_h(node, conditions: List[RuleCondition]) -> None:
            if node.is_leaf:
                predicted = _scalar(node.prediction)
                distribution = np.asarray(node.distribution, dtype=float)
                confidence = float(distribution.max()) if distribution.size else 0.0
                class_name = str(labels.get(predicted, predicted))
                rules.append(GlobalRule(
                    rule_id=f'R{len(rules) + 1}',
                    conditions=list(conditions),
                    predicted_class=predicted,
                    predicted_class_name=class_name,
                    support=float(node.reach),
                    confidence=confidence,
                ))
                return
            walk_h(node.false_child, conditions + [historical_condition(node.test, False)])
            walk_h(node.true_child, conditions + [historical_condition(node.test, True)])

        walk_h(tree_model.root_, [])
        return rules

    structure = tree_model.tree_
    classes = np.asarray(getattr(tree_model, 'classes_', []))
    root_samples = max(float(structure.weighted_n_node_samples[0]), 1.0)

    def walk(node: int, conditions: List[RuleCondition]) -> None:
        left = int(structure.children_left[node])
        right = int(structure.children_right[node])
        if left == right:
            counts = np.asarray(structure.value[node]).reshape(-1)
            class_index = int(np.argmax(counts)) if counts.size else 0
            predicted = _scalar(classes[class_index]) if classes.size else class_index
            total = float(np.sum(counts))
            confidence = float(counts[class_index] / total) if total else 0.0
            class_name = str(labels.get(predicted, predicted))
            rules.append(GlobalRule(
                rule_id=f'R{len(rules) + 1}', conditions=list(conditions),
                predicted_class=predicted, predicted_class_name=class_name,
                support=float(structure.weighted_n_node_samples[node] / root_samples),
                confidence=confidence,
            ))
            return
        feature_index = int(structure.feature[node])
        if feature_index < 0 or feature_index >= len(names):
            raise ValueError('A árvore referencia uma feature inexistente no dataset activo.')
        threshold = float(structure.threshold[node])
        feature = names[feature_index]
        walk(left, conditions + [RuleCondition(feature_index, feature, '<=', threshold)])
        walk(right, conditions + [RuleCondition(feature_index, feature, '>', threshold)])

    walk(0, [])
    return rules


def _condition_differences(
    factual: GlobalRule, counterfactual: GlobalRule,
) -> List[ConditionChange]:
    """Compara caminhos incluindo condições m-of-n como unidades semânticas."""
    grouped: Dict[tuple, Tuple[List[RuleCondition], List[RuleCondition]]] = {}
    keys = {c.group_key() for c in factual.conditions + counterfactual.conditions}
    for key in sorted(keys, key=str):
        grouped[key] = (
            [c for c in factual.conditions if c.group_key() == key],
            [c for c in counterfactual.conditions if c.group_key() == key],
        )

    changes: List[ConditionChange] = []
    for left, right in grouped.values():
        unmatched_right = list(right)
        unmatched_left: List[RuleCondition] = []
        for condition in left:
            exact = next((
                idx for idx, other in enumerate(unmatched_right)
                if condition.signature() == other.signature()
            ), None)
            if exact is None:
                unmatched_left.append(condition)
            else:
                unmatched_right.pop(exact)

        pair_count = min(len(unmatched_left), len(unmatched_right))
        for index in range(pair_count):
            before, after = unmatched_left[index], unmatched_right[index]
            kind = 'inversao' if before.operator != after.operator or before.negated != after.negated else 'limiar'
            changes.append(ConditionChange(
                feature=before.feature, factual_condition=before.text(),
                counterfactual_condition=after.text(), change_type=kind,
            ))
        for before in unmatched_left[pair_count:]:
            changes.append(ConditionChange(before.feature, before.text(), None, 'remocao'))
        for after in unmatched_right[pair_count:]:
            changes.append(ConditionChange(after.feature, None, after.text(), 'adicao'))
    return changes


def generate_global_counterfactuals(
    rules: Sequence[GlobalRule], max_per_rule: int = 5,
) -> List[GlobalRuleCounterfactual]:
    """Selecciona as transições simbólicas mínimas para classes diferentes."""
    output: List[GlobalRuleCounterfactual] = []
    limit = max(1, int(max_per_rule))
    for factual in rules:
        candidates: List[GlobalRuleCounterfactual] = []
        for other in rules:
            if other.rule_id == factual.rule_id or other.predicted_class == factual.predicted_class:
                continue
            changes = _condition_differences(factual, other)
            if not changes:
                # Defesa adicional: duas folhas diferentes nunca são uma mudança nula.
                changes = [ConditionChange("caminho", factual.text(), other.text(), "caminho")]
            candidates.append(GlobalRuleCounterfactual(factual, other, changes))
        candidates.sort(key=lambda item: (
            item.n_changes,
            -item.counterfactual_rule.confidence,
            -item.counterfactual_rule.support,
            item.counterfactual_rule.rule_id,
        ))
        output.extend(candidates[:limit])
    return output


def _matches(rule: GlobalRule, row: np.ndarray) -> bool:
    return all(condition.matches(row) for condition in rule.conditions)


def _rule_dict(rule: GlobalRule) -> Dict[str, Any]:
    value = asdict(rule)
    value["text"] = rule.text()
    return value


def _cf_dict(item: GlobalRuleCounterfactual) -> Dict[str, Any]:
    return {
        "factual_rule_id": item.factual_rule.rule_id,
        "counterfactual_rule_id": item.counterfactual_rule.rule_id,
        "factual_class": item.factual_rule.predicted_class,
        "target_class": item.counterfactual_rule.predicted_class,
        "target_class_name": item.counterfactual_rule.predicted_class_name,
        "n_changes": item.n_changes,
        "changes": [asdict(change) for change in item.changes],
        "factual_rule": item.factual_rule.text(),
        "counterfactual_rule": item.counterfactual_rule.text(),
        "target_support": item.counterfactual_rule.support,
        "target_confidence": item.counterfactual_rule.confidence,
        "validity": item.counterfactual_rule.predicted_class != item.factual_rule.predicted_class,
    }


def analyse_global_rules(
    model: Any,
    oracle: Any,
    X: Any,
    feature_names: Sequence[str],
    *,
    class_labels: Optional[Mapping[Any, Any]] = None,
    max_per_rule: int = 5,
    remove_fragile: bool = True,
    fragile_n_changes_threshold: int = 10,
    model_name: str = "árvore",
    dataset_name: str = "dataset_carregado",
) -> Dict[str, Any]:
    """Executa extracção, contrafactuais e métricas sobre o dataset carregado."""
    tree_model = unwrap_tree(model)
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("Dataset, árvore e nomes de features estão desalinhados.")
    rules = extract_global_rules(tree_model, feature_names, class_labels)
    raw_counterfactuals = generate_global_counterfactuals(rules, max_per_rule=max_per_rule)
    counterfactuals = list(raw_counterfactuals)
    if remove_fragile:
        counterfactuals = [
            item for item in counterfactuals
            if item.n_changes <= int(fragile_n_changes_threshold)
            and len(item.counterfactual_rule.conditions)
            <= 2 * max(1, len(item.factual_rule.conditions))
        ]
    tree_prediction = np.asarray(tree_model.predict(values)).reshape(-1)
    oracle_prediction = np.asarray(oracle.predict(values)).reshape(-1)
    cf_rule_ids = {item.factual_rule.rule_id for item in counterfactuals}
    matched_rule_ids = []
    covered_instances = 0
    for row in values:
        matched = next((rule for rule in rules if _matches(rule, row)), None)
        matched_rule_ids.append(matched.rule_id if matched else None)
        covered_instances += int(matched is not None and matched.rule_id in cf_rule_ids)
    conditions = [len(rule.conditions) for rule in rules]
    changes = [item.n_changes for item in counterfactuals]
    signatures: Dict[Tuple[Tuple[int, str, float], ...], set] = {}
    for rule in rules:
        signature = tuple(c.signature() for c in rule.conditions)
        signatures.setdefault(signature, set()).add(str(rule.predicted_class))
    conflicts = sum(1 for classes in signatures.values() if len(classes) > 1)
    proxies = [1.0 / (1.0 + value) for value in changes]
    grouped: Dict[Tuple[str, str, int], List[GlobalRuleCounterfactual]] = {}
    for item in counterfactuals:
        grouped.setdefault((
            item.factual_rule.rule_id,
            str(item.counterfactual_rule.predicted_class),
            item.n_changes,
        ), []).append(item)
    clusters = []
    for (factual_id, target_class, n_changes), members in sorted(grouped.items()):
        representative = min(
            members,
            key=lambda item: (
                -item.counterfactual_rule.confidence,
                -item.counterfactual_rule.support,
                item.counterfactual_rule.rule_id,
            ),
        )
        clusters.append({
            "factual_rule_id": factual_id,
            "target_class": target_class,
            "n_changes": n_changes,
            "n_variants": len(members),
            "representative_rule_id": representative.counterfactual_rule.rule_id,
            "summary": (
                f"{factual_id} → classe {representative.counterfactual_rule.predicted_class_name}: "
                f"{len(members)} variante(s), {n_changes} alteração(ões)"
            ),
        })
    metrics = {
        "global_fidelity": float(np.mean(tree_prediction == oracle_prediction)),
        "rule_coverage": len(cf_rule_ids) / len(rules) if rules else 0.0,
        "instance_coverage": covered_instances / len(values) if len(values) else 0.0,
        "mean_conditions": float(np.mean(conditions)) if conditions else 0.0,
        "max_conditions": int(max(conditions)) if conditions else 0,
        "mean_symbolic_changes": float(np.mean(changes)) if changes else 0.0,
        "min_symbolic_changes": int(min(changes)) if changes else 0,
        "max_symbolic_changes": int(max(changes)) if changes else 0,
        "symbolic_stability_proxy": float(np.mean(proxies)) if proxies else 0.0,
        "rule_consistency": 1.0 - (conflicts / len(signatures)) if signatures else 1.0,
        "n_rules": len(rules),
        "n_counterfactual_rules": len(counterfactuals),
        "n_filtered_fragile": len(raw_counterfactuals) - len(counterfactuals),
        "n_clusters": len(clusters),
    }
    rules_by_class: Dict[str, List[Dict[str, Any]]] = {}
    for rule in rules:
        rules_by_class.setdefault(str(rule.predicted_class), []).append({
            "rule_id": rule.rule_id,
            "class_name": rule.predicted_class_name,
            "support": rule.support,
            "confidence": rule.confidence,
            "conditions": [asdict(condition) for condition in rule.conditions],
        })
    cfs_by_factual: Dict[str, List[Dict[str, Any]]] = {}
    for item in counterfactuals:
        cfs_by_factual.setdefault(item.factual_rule.rule_id, []).append(_cf_dict(item))
    for_audit = {
        "factual_rules_by_class": rules_by_class,
        "counterfactuals_by_factual_rule": cfs_by_factual,
        "cluster_summaries": [cluster["summary"] for cluster in clusters],
    }
    for_fairness = {
        "rules_per_class": {
            class_id: {
                "class_name": rows[0]["class_name"] if rows else class_id,
                "n_rules": len(rows),
                "total_support": float(sum(row["support"] for row in rows)),
                "rule_ids": [row["rule_id"] for row in rows],
            }
            for class_id, rows in rules_by_class.items()
        },
        "note": "Estrutura descritiva; conclusões de equidade exigem atributos protegidos e análise própria.",
    }
    for_owl = {
        "factual_rules": [{
            "id": rule.rule_id,
            "antecedent": [asdict(condition) for condition in rule.conditions],
            "consequent_class": rule.predicted_class,
            "consequent_class_name": rule.predicted_class_name,
        } for rule in rules],
        "counterfactual_rules": [{
            "factual_rule_id": item.factual_rule.rule_id,
            "counterfactual_rule_id": item.counterfactual_rule.rule_id,
            "target_class": item.counterfactual_rule.predicted_class,
            "changes": [asdict(change) for change in item.changes],
        } for item in counterfactuals],
    }
    narrative = (
        f"Foram extraídas {len(rules)} regras de {model_name} no dataset activo e "
        f"geradas {len(counterfactuals)} transições para classes alternativas. "
        f"A fidelidade global ao oráculo é {metrics['global_fidelity']:.1%} e a "
        f"cobertura das instâncias por regras com alternativa é {metrics['instance_coverage']:.1%}."
    )
    return {
        "result_type": "global_rules",
        "scope": "loaded_dataset_only",
        "dataset": dataset_name,
        "target_model": model_name,
        "feature_names": list(feature_names),
        "factual_rules": [_rule_dict(rule) for rule in rules],
        "counterfactual_rules": [_cf_dict(item) for item in counterfactuals],
        "clusters": clusters,
        "aggregate_metrics": metrics,
        "narrative": narrative,
        "metric_notes": {
            "symbolic_stability_proxy": (
                "Proxy descritivo 1/(1+n.º de alterações); não equivale a robustez empírica."
            ),
            "validity": "A regra de destino é uma folha real da mesma árvore e tem classe diferente.",
        },
        "for_audit": for_audit,
        "for_fairness": for_fairness,
        "for_owl": for_owl,
        "uses": ["auditoria", "comparação de modelos", "equidade", "exportação OWL"],
    }


__all__ = [
    "RuleCondition", "GlobalRule", "ConditionChange", "GlobalRuleCounterfactual",
    "unwrap_tree", "extract_global_rules", "generate_global_counterfactuals",
    "analyse_global_rules",
]
