"""Quality gate ontológico: TBox/ABox, matching e ontologias genéricas."""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple


GENERIC_ENTITY_NAMES = {
    "thing", "entity", "class", "concept", "feature", "attribute", "property",
    "instance", "record", "sample", "dataset", "observation", "value", "label",
    "target", "classification", "classificationlabel", "category",
}
RECORD_TOKENS = {
    "record", "row", "sample", "observation", "patient", "subject", "case",
    "datasetinstance", "traininginstance", "testinstance", "personrecord",
}
IDENTIFIER_PROPERTIES = {
    "rowid", "recordid", "sampleid", "patientid", "subjectid", "instanceid",
    "split", "fold", "partition", "datasetrow", "rowindex",
}
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
    labels = []
    for attr in ("label", "prefLabel", "altLabel"):
        try:
            raw = getattr(entity, attr, []) or []
            if not isinstance(raw, (list, tuple, set)):
                raw = [raw]
            labels.extend(str(item) for item in raw if item is not None)
        except Exception:
            continue
    return list(dict.fromkeys(labels))


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
        entities = ontology_entities(ontology)
        candidates = (
            entities["classes"]
            + entities["datatype_properties"]
            + entities["object_properties"]
        )
        decisions = []
        for feature in feature_names:
            ranked = sorted(
                ((self._candidate_score(feature, entity), entity) for entity in candidates),
                key=lambda pair: pair[0],
                reverse=True,
            )
            best_score, best = ranked[0] if ranked else (0.0, None)
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            margin = best_score - runner_up
            accepted = (
                best is not None
                and best_score >= self.min_match_score
                and (best_score >= 0.98 or margin >= self.min_ambiguity_margin)
            )
            reason = "accepted"
            if best is None or best_score < self.min_match_score:
                reason = "score_below_threshold"
            elif not accepted:
                reason = "ambiguous_match"
            decisions.append(MatchDecision(
                feature=str(feature),
                entity_name=getattr(best, "name", None) if best is not None else None,
                entity_type=entity_kind(best) if best is not None else None,
                score=float(best_score),
                runner_up_score=float(runner_up),
                ambiguity_margin=float(margin),
                accepted=bool(accepted),
                reason=reason,
            ))
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
        return {
            "accepted": accepted,
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
        metrics = {
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
        elif generic_ratio > self.max_generic_ratio:
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
