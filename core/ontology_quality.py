"""Quality gate ontológico: TBox/ABox, matching e ontologias genéricas."""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from core.ontology_semantic_graph import _explicit_entity_bounds
from core.owl_annotations import annotation_values


GENERIC_ENTITY_NAMES = {
    "thing", "entity", "class", "concept", "feature", "attribute", "property",
    "instance", "record", "sample", "dataset", "observation", "value", "label",
    "target", "classification", "classificationlabel", "category", "measurement",
}
RECORD_TOKENS = {
    "record", "row", "sample", "observation", "patient", "subject", "case",
    "datasetinstance", "traininginstance", "testinstance", "personrecord",
}
IDENTIFIER_PROPERTIES = {
    "rowid", "recordid", "sampleid", "patientid", "subjectid", "instanceid",
    "split", "fold", "partition", "datasetrow", "rowindex",
}
ROLE_ANNOTATIONS = ("statisticRole", "measurementRole", "semanticRole", "statistic_role")
FAMILY_ANNOTATIONS = ("measurementFamily", "semanticFamily", "featureFamily", "measurement_family")
PLACEHOLDER_RE = re.compile(r"^(?:feature|feat|attr|attribute|column|col|x|f)[_-]?\d+$")


def normalise_token(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def identifier_tokens(value: Any) -> List[str]:
    """Tokeniza nomes de schema sem assumir convenções de um dataset.

    Reconhece snake_case, kebab-case, espaços e CamelCase. A comparação por
    conjunto de tokens permite, por exemplo, ``feature_total`` ↔
    ``TotalFeature`` sem manter aliases específicos por domínio.
    """
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [tok for tok in re.split(r"[^A-Za-z0-9]+", text.lower()) if tok]


def entity_labels(entity) -> List[str]:
    """Rótulos da entidade (rdfs:label / skos:prefLabel / skos:altLabel), lidos por propriedade/IRI."""
    return annotation_values(entity, ("label", "prefLabel", "altLabel"))


def entity_kind(entity) -> str:
    try:
        if hasattr(entity, "is_instance_of") or hasattr(entity, "INDIRECT_is_a"):
            if not isinstance(entity, type):
                return "named_individual"
    except Exception:
        pass
    text = str(type(entity)).lower()
    if "dataproperty" in text:
        return "datatype_property"
    if "objectproperty" in text:
        return "object_property"
    if "thingclass" in text or "class" in text:
        return "class"
    return "unknown"


def ontology_entities(ontology) -> Dict[str, List[Any]]:
    def safe(method):
        try:
            return list(getattr(ontology, method)())
        except Exception:
            return []
    return {
        "classes": safe("classes"),
        "datatype_properties": safe("data_properties"),
        "object_properties": safe("object_properties"),
        "individuals": safe("individuals"),
    }


def _has_annotation(entity: Any, names: Sequence[str]) -> bool:
    """Indica se a entidade declara alguma das anotações (por propriedade/IRI, sem depender da ordem de carregamento)."""
    return bool(annotation_values(entity, names))


def taxonomy_depth(classes: Sequence[Any]) -> int:
    """Maior cadeia de subclasses declarada (ciclos protegidos)."""
    class_set = set(classes)
    memo: Dict[Any, int] = {}

    def depth(cls, visiting):
        if cls in memo:
            return memo[cls]
        if cls in visiting:
            return 0
        parents = []
        try:
            parents = [p for p in (getattr(cls, "is_a", []) or []) if p in class_set]
        except Exception:
            pass
        value = 1 + max((depth(p, visiting | {cls}) for p in parents), default=0)
        memo[cls] = value
        return value

    return max((depth(c, frozenset()) for c in classes), default=0)


def mapping_entropy(entity_names: Sequence[str]) -> float:
    """Entropia normalizada [0,1] da distribuição features -> entidades.

    1.0 = cada feature numa entidade distinta; 0.0 = tudo na mesma entidade.
    """
    n = len(entity_names)
    if n <= 1:
        return 1.0 if n == 1 else 0.0
    counts: Dict[str, int] = {}
    for name in entity_names:
        counts[name] = counts.get(name, 0) + 1
    import math
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    return float(h / math.log(n))


def semantic_richness(entities: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Mede conhecimento *para além da taxonomia* que a OWL realmente declara.

    Cobertura de matching e ausência de ABox não dizem se a ontologia tem algo a
    ensinar ao modelo. As fontes contadas são as que o enriquecimento consegue
    usar: relações entre conceitos (propriedades de objeto), restrições OWL,
    limites numéricos declarados e papéis estatísticos por família de medida.
    """
    object_properties = len(entities.get("object_properties", []))
    restrictions = 0
    for cls in entities.get("classes", []):
        try:
            restrictions += sum(
                1 for parent in getattr(cls, "is_a", []) or []
                if hasattr(parent, "property") and hasattr(parent, "type")
            )
        except Exception:
            continue
    bounded, roles, families = 0, 0, 0
    for prop in entities.get("datatype_properties", []):
        try:
            low, high = _explicit_entity_bounds(prop)
        except Exception:
            low = high = None
        bounded += int(low is not None or high is not None)
        roles += int(_has_annotation(prop, ROLE_ANNOTATIONS))
        families += int(_has_annotation(prop, FAMILY_ANNOTATIONS))
    sources = {
        "object_properties": object_properties > 0,
        "restrictions": restrictions > 0,
        "declared_bounds": bounded > 0,
        "statistic_roles": roles > 0 and families > 0,
    }
    active = sum(sources.values())
    return {
        "object_property_count": object_properties,
        "restriction_count": restrictions,
        "bounded_datatype_properties": bounded,
        "role_annotated_properties": roles,
        "family_annotated_properties": families,
        "knowledge_sources": sorted(name for name, on in sources.items() if on),
        "level": "poor" if active == 0 else "limited" if active == 1 else "rich",
    }


@dataclass(frozen=True)
class MatchDecision:
    feature: str
    entity_name: str | None
    entity_type: str | None
    score: float
    runner_up_score: float
    ambiguity_margin: float
    accepted: bool
    reason: str
    runner_up_entity: str | None = None
    status: str = "ACCEPTED"
    tie_break: str | None = None


@dataclass
class OntologyQualityReport:
    accepted: bool
    status: str
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    matches: List[Dict[str, Any]] = field(default_factory=list)
    abox: Dict[str, Any] = field(default_factory=dict)
    reasoner: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OntologyQualityGate:
    """Rejeita ontologias ambíguas, genéricas ou com ABox de registos."""

    def __init__(
        self,
        *,
        min_match_score: float = 0.72,
        min_ambiguity_margin: float = 0.08,
        min_feature_coverage: float = 0.35,
        min_unique_entities: int = 2,
        max_generic_ratio: float = 0.45,
    ):
        self.min_match_score = float(min_match_score)
        self.min_ambiguity_margin = float(min_ambiguity_margin)
        self.min_feature_coverage = float(min_feature_coverage)
        self.min_unique_entities = int(min_unique_entities)
        self.max_generic_ratio = float(max_generic_ratio)

    @staticmethod
    def _candidate_score(feature: str, entity) -> float:
        probe = normalise_token(feature)
        probe_tokens = identifier_tokens(feature)
        raw_candidates = [str(getattr(entity, "name", ""))]
        raw_candidates.extend(entity_labels(entity))
        if not probe or not raw_candidates:
            return 0.0

        best = 0.0
        probe_token_set = set(probe_tokens)
        for raw_candidate in raw_candidates:
            candidate = normalise_token(raw_candidate)
            candidate_tokens = identifier_tokens(raw_candidate)
            if not candidate:
                continue
            candidate_token_set = set(candidate_tokens)
            if probe == candidate:
                score = 1.0
            elif probe_token_set and probe_token_set == candidate_token_set:
                # Mesmos termos em ordem/convenção de escrita diferente.
                score = 0.97
            elif min(len(probe), len(candidate)) >= 4 and (
                probe in candidate or candidate in probe
            ):
                score = 0.86
            else:
                seq = difflib.SequenceMatcher(None, probe, candidate).ratio()
                if probe_token_set and candidate_token_set:
                    union = probe_token_set | candidate_token_set
                    jaccard = len(probe_token_set & candidate_token_set) / max(len(union), 1)
                else:
                    jaccard = 0.0
                score = max(seq * 0.90, jaccard * 0.92)
            best = max(best, float(score))
        return best

    def match_features(self, feature_names: Sequence[str], ontology) -> List[MatchDecision]:
        """Matching ARFF -> OWL auditável.

        Cada decisão guarda o melhor candidato, o segundo candidato (com nome e
        score), a margem e um ``status``. Duas regras evitam falsos positivos:

        * desempate por tipo: se a margem é pequena entre uma classe e uma
          propriedade de dados, um atributo mapeia-se à propriedade (atributos
          são medições, não conceitos); fica registado em ``tie_break``;
        * colisão: várias features nunca ficam silenciosamente na mesma
          entidade. Só a de maior score (sem empate) mantém o match.
        """
        entities = ontology_entities(ontology)
        candidates = (
            entities["classes"]
            + entities["datatype_properties"]
            + entities["object_properties"]
        )
        decisions: List[MatchDecision] = []
        for feature in feature_names:
            ranked = sorted(
                ((self._candidate_score(feature, entity), entity) for entity in candidates),
                key=lambda pair: pair[0],
                reverse=True,
            )
            tie_break = None
            if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.min_ambiguity_margin:
                kinds = (entity_kind(ranked[0][1]), entity_kind(ranked[1][1]))
                if kinds == ("class", "datatype_property"):
                    ranked[0], ranked[1] = ranked[1], ranked[0]
                    tie_break = "datatype_property_over_class"
            best_score, best = ranked[0] if ranked else (0.0, None)
            runner_score, runner = ranked[1] if len(ranked) > 1 else (0.0, None)
            margin = best_score - runner_score
            accepted = (
                best is not None
                and best_score >= self.min_match_score
                and (best_score >= 0.98 or margin >= self.min_ambiguity_margin or tie_break is not None)
            )
            if best is None or best_score < self.min_match_score:
                reason, status = "score_below_threshold", "REJECTED_LOW_SCORE"
            elif not accepted:
                reason, status = "ambiguous_match", "REJECTED_AMBIGUOUS"
            else:
                reason, status = "accepted", "ACCEPTED"
            decisions.append(MatchDecision(
                feature=str(feature),
                entity_name=getattr(best, "name", None) if best is not None else None,
                entity_type=entity_kind(best) if best is not None else None,
                score=float(best_score),
                runner_up_score=float(runner_score),
                ambiguity_margin=float(margin),
                accepted=bool(accepted),
                reason=reason,
                runner_up_entity=getattr(runner, "name", None) if runner is not None else None,
                status=status,
                tie_break=tie_break,
            ))
        return self._resolve_entity_collisions(decisions)

    @staticmethod
    def _resolve_entity_collisions(decisions: List[MatchDecision]) -> List[MatchDecision]:
        by_entity: Dict[str, List[int]] = {}
        for index, decision in enumerate(decisions):
            if decision.accepted and decision.entity_name:
                by_entity.setdefault(decision.entity_name, []).append(index)
        for indices in by_entity.values():
            if len(indices) < 2:
                continue
            top = max(decisions[i].score for i in indices)
            winners = [i for i in indices if decisions[i].score == top]
            keep = winners[0] if len(winners) == 1 else None
            for i in indices:
                if i == keep:
                    continue
                d = decisions[i]
                decisions[i] = MatchDecision(
                    feature=d.feature, entity_name=d.entity_name, entity_type=d.entity_type,
                    score=d.score, runner_up_score=d.runner_up_score,
                    ambiguity_margin=d.ambiguity_margin, accepted=False,
                    reason="entity_collision", runner_up_entity=d.runner_up_entity,
                    status="REJECTED_ENTITY_COLLISION", tie_break=d.tie_break,
                )
        return decisions

    @staticmethod
    def _individual_type_names(individual) -> List[str]:
        types = []
        for attr in ("is_a", "is_instance_of", "INDIRECT_is_a"):
            try:
                types.extend(
                    getattr(item, "name", str(item))
                    for item in (getattr(individual, attr, []) or [])
                )
            except Exception:
                continue
        return list(dict.fromkeys(types))

    @staticmethod
    def _individual_properties(individual) -> List[Tuple[str, Any]]:
        values = []
        try:
            properties = list(individual.get_properties())
        except Exception:
            properties = []
        for prop in properties:
            name = getattr(prop, "name", str(prop))
            try:
                raw = list(prop[individual])
            except Exception:
                raw = []
            values.extend((name, value) for value in raw)
        return values

    def audit_abox(
        self,
        ontology,
        *,
        forbidden_test_identifiers: Iterable[Any] | None = None,
    ) -> Dict[str, Any]:
        forbidden = {
            normalise_token(value) for value in (forbidden_test_identifiers or [])
            if normalise_token(value)
        }
        records, leaked, controlled = [], [], []
        for individual in ontology_entities(ontology)["individuals"]:
            name = getattr(individual, "name", str(individual))
            type_names = self._individual_type_names(individual)
            property_values = self._individual_properties(individual)
            type_tokens = {normalise_token(value) for value in type_names}
            property_tokens = {normalise_token(name) for name, _ in property_values}
            record_like = bool(type_tokens & RECORD_TOKENS) or bool(
                property_tokens & IDENTIFIER_PROPERTIES
            ) or bool(re.match(r"^(?:row|record|sample|patient|subject|case)[_-]?\d+$", name, re.I))
            # Indivíduos com várias observações escalares representam linhas,
            # não termos de um vocabulário categórico controlado.
            record_like = record_like or len(property_values) >= 3
            identifiers = {normalise_token(name)} | {
                normalise_token(value) for prop, value in property_values
                if normalise_token(prop) in IDENTIFIER_PROPERTIES
            }
            if forbidden and identifiers & forbidden:
                leaked.append(name)
            if record_like:
                records.append(name)
            else:
                controlled.append(name)
        accepted = not records and not leaked
        if leaked:
            status = "REJECT_TEST_INSTANCE_LEAKAGE"
        elif records:
            status = "REJECT_DATASET_RECORDS_IN_ABOX"
        else:
            status = "SAFE"
        return {
            "accepted": accepted,
            "status": status,
            "tbox_only_for_records": not records,
            "dataset_record_individuals": records,
            "test_identifier_collisions": leaked,
            "controlled_vocabulary_individuals": controlled,
            "individual_count": len(records) + len(controlled),
        }

    def evaluate(
        self,
        feature_names: Sequence[str],
        ontology,
        *,
        forbidden_test_identifiers: Iterable[Any] | None = None,
        reasoner_report: Dict[str, Any] | None = None,
        require_reasoner: bool = True,
    ) -> OntologyQualityReport:
        entities = ontology_entities(ontology)
        all_tbox = (
            entities["classes"] + entities["datatype_properties"] + entities["object_properties"]
        )
        names = [normalise_token(getattr(entity, "name", "")) for entity in all_tbox]
        generic = [
            name for name in names
            if name in GENERIC_ENTITY_NAMES or PLACEHOLDER_RE.match(name or "")
        ]
        generic_ratio = len(generic) / max(len(names), 1)
        decisions = self.match_features(feature_names, ontology)
        accepted_matches = [decision for decision in decisions if decision.accepted]
        coverage = len(accepted_matches) / max(len(feature_names), 1)
        unique_entities = len({decision.entity_name for decision in accepted_matches})
        abox = self.audit_abox(
            ontology, forbidden_test_identifiers=forbidden_test_identifiers
        )
        issues, warnings = [], []
        if not all_tbox:
            issues.append("Ontologia sem TBox utilizável.")
        if generic_ratio > self.max_generic_ratio:
            issues.append("Ontologia excessivamente genérica/placeholder.")
        if coverage < self.min_feature_coverage:
            issues.append(
                f"Cobertura de matching insuficiente ({coverage:.1%} < {self.min_feature_coverage:.1%})."
            )
        if unique_entities < min(self.min_unique_entities, max(1, len(feature_names))):
            issues.append("Poucas entidades ontológicas distintas foram mapeadas.")
        if not abox["accepted"]:
            issues.append("ABox contém registos do dataset ou identificadores do teste.")
        reasoner_report = dict(reasoner_report or {})
        if require_reasoner and not reasoner_report.get("executed"):
            issues.append("Reasoner OWL não foi executado; consistência não demonstrada.")
        elif reasoner_report.get("consistent") is False:
            issues.append("Reasoner encontrou inconsistência lógica.")
        elif reasoner_report.get("unsatisfiable_classes"):
            issues.append("Reasoner encontrou classes insatisfazíveis.")
        ambiguous = sum(decision.reason == "ambiguous_match" for decision in decisions)
        if ambiguous:
            warnings.append(f"{ambiguous} matching(s) rejeitado(s) por ambiguidade.")
        mapped_names = [d.entity_name for d in accepted_matches if d.entity_name]
        generic_hits = [
            n for n in mapped_names
            if normalise_token(n) in GENERIC_ENTITY_NAMES or PLACEHOLDER_RE.match(normalise_token(n) or "")
        ]
        generic_match_ratio = len(generic_hits) / max(len(mapped_names), 1)
        if generic_match_ratio > self.max_generic_ratio:
            issues.append("Matching dirigido sobretudo a entidades genéricas.")
        collisions = sum(d.reason == "entity_collision" for d in decisions)
        if collisions:
            warnings.append(f"{collisions} matching(s) rejeitado(s) por colisão de entidade.")
        richness = semantic_richness(entities)
        if richness["level"] == "poor":
            warnings.append(
                "Ontologia semanticamente pobre: sem propriedades de objeto, restrições, "
                "limites declarados nem papéis estatísticos; o enriquecimento limita-se a "
                "agregados de grupo (combinações lineares das features originais)."
            )
        metrics = {
            "ambiguous_matches": int(ambiguous),
            "entity_collisions": int(collisions),
            "generic_match_ratio": float(generic_match_ratio),
            "distinct_entity_ratio": float(unique_entities / max(len(accepted_matches), 1)),
            "mapping_entropy": mapping_entropy(mapped_names),
            "semantic_specificity": float(1.0 - generic_match_ratio),
            "ontology_depth": int(taxonomy_depth(entities["classes"])),
            "reasoner_used": bool(reasoner_report.get("reasoner_used", reasoner_report.get("executed"))),
            "reasoner_consistent": reasoner_report.get("consistent"),
            "abox_safe": bool(abox["accepted"]),
            "knowledge_split": {
                "tbox": {
                    "classes": len(entities["classes"]),
                    "datatype_properties": len(entities["datatype_properties"]),
                    "object_properties": len(entities["object_properties"]),
                },
                "abox": {
                    "individuals": len(entities["individuals"]),
                    "status": abox.get("status"),
                },
            },
            "semantic_richness": richness,
            "feature_coverage": coverage,
            "mapped_features": len(accepted_matches),
            "total_features": len(feature_names),
            "unique_mapped_entities": unique_entities,
            "generic_entity_ratio": generic_ratio,
            "tbox_entity_count": len(all_tbox),
        }
        accepted = not issues
        if accepted:
            status = "VALID_DOMAIN_ONTOLOGY"
        elif not abox["accepted"]:
            status = "CONTAMINATED_ONTOLOGY"
        elif reasoner_report.get("consistent") is False or reasoner_report.get("unsatisfiable_classes"):
            status = "LOGICALLY_INCONSISTENT"
        elif require_reasoner and not reasoner_report.get("executed"):
            status = "REASONER_NOT_VALIDATED"
        elif coverage < self.min_feature_coverage or unique_entities < min(
            self.min_unique_entities, max(1, len(feature_names))
        ):
            status = "INSUFFICIENT_SCHEMA_COVERAGE"
        elif generic_ratio > self.max_generic_ratio or generic_match_ratio > self.max_generic_ratio:
            status = "GENERIC_ONTOLOGY"
        else:
            status = "INVALID_ONTOLOGY"
        metrics["mean_matching_confidence"] = float(
            sum(decision.score for decision in accepted_matches) / len(accepted_matches)
        ) if accepted_matches else 0.0
        return OntologyQualityReport(
            accepted=accepted,
            status=status,
            issues=issues,
            warnings=warnings,
            metrics=metrics,
            matches=[asdict(decision) for decision in decisions],
            abox=abox,
            reasoner=reasoner_report,
        )
