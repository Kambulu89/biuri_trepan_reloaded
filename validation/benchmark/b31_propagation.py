"""B3.1 (Doc. 34 §5 / Doc. 36): mapa da PROPAGAÇÃO ``OWL → Reasoner → Fact Registry → relação semântica → candidato m-of-n``.
Para cada facto do registo devolve o destino: o que foi reconhecido (explícito/inferido), descartado (e porquê), bloqueado pelo matching, irrelevante para os atributos presentes,
ou chegou a um candidato (unidade pesquisada). Distingue «ausência de inferências novas» de «falha de propagação»: a inferência só conta como falha se existir facto
INFERRED_BY_REASONER que não alcança nenhum atributo mapeado relevante."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

from validation.benchmark.b31_facts import Derivation, FactRegistry, Kind, UseState
from validation.benchmark.b31_relatedness import SemanticUnit

RECOGNIZED_EXPLICIT = "RECOGNIZED_EXPLICIT"
RECOGNIZED_INFERRED = "RECOGNIZED_INFERRED"
DISCARDED = "DISCARDED"
BLOCKED_BY_MATCHING = "BLOCKED_BY_MATCHING"
IRRELEVANT_TO_FEATURES = "IRRELEVANT_TO_FEATURES"
REACHED_CANDIDATE = "REACHED_CANDIDATE"


def propagation_report(reg: FactRegistry, features: Sequence[str], units: Sequence[SemanticUnit], searched_fact_ids: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    feats = set(features)
    mapped = {f for f in feats if reg.feature_entity.get(f)}
    searched = set(searched_fact_ids or [])
    unit_fact_ids = {fid for u in units for fid in u.fact_ids}
    mapped_entities = {reg.feature_entity[f] for f in mapped}
    # conceitos a que algum atributo mapeado se aplica (para decidir irrelevância dos factos de esquema)
    relevant_concepts = {fa.entities[1] for fa in reg.by_kind(Kind.ATTRIBUTE_APPLIES_TO_CONCEPT) if fa.attributes[0] in mapped and fa.use_state != UseState.REJECTED.value}
    rows: List[Dict[str, Any]] = []
    for f in reg.facts:
        if f.use_state == UseState.REJECTED.value:
            dest, why = (BLOCKED_BY_MATCHING, f.rejection_reason) if f.kind == Kind.MATCH_FAILURE.value else (DISCARDED, f.rejection_reason)
        elif f.kind in (Kind.CONCEPT_DISJOINTNESS.value, Kind.VALUE_BOUND.value):
            dest, why = DISCARDED, "sem contrato de aplicação a linhas/combinações (permitted_use=NONE)"
        elif f.kind == Kind.ATTRIBUTE_APPLIES_TO_CONCEPT.value:
            reached = f.id in searched
            in_unit = f.id in unit_fact_ids
            if reached:
                dest, why = REACHED_CANDIDATE, "unidade pesquisada"
            elif in_unit:
                dest, why = (RECOGNIZED_INFERRED if f.derivation == Derivation.INFERRED_BY_REASONER.value else RECOGNIZED_EXPLICIT), "em unidade disponível, não pesquisada"
            else:
                dest, why = (RECOGNIZED_INFERRED if f.derivation == Derivation.INFERRED_BY_REASONER.value else RECOGNIZED_EXPLICIT), "reconhecido; conceito genérico/sem extensão partilhada (sem unidade)"
        elif f.kind in (Kind.CONCEPT_SUBSUMPTION.value, Kind.CONCEPT_EQUIVALENCE.value, Kind.PROPERTY_SUBSUMPTION.value, Kind.PROPERTY_EQUIVALENCE.value):
            ents = set(f.entities)
            subject_ok = (f.entities[0] in relevant_concepts) if f.kind == Kind.CONCEPT_SUBSUMPTION.value else bool(ents & relevant_concepts)
            if f.kind.startswith("CONCEPT") and not subject_ok:
                dest, why = IRRELEVANT_TO_FEATURES, "nenhum atributo mapeado depende destes conceitos"
            elif f.kind.startswith("PROPERTY") and not (ents & mapped_entities) and not any(e in mapped_entities for e in ents):
                dest, why = IRRELEVANT_TO_FEATURES, "propriedades não mapeadas a atributos"
            else:
                dest, why = (RECOGNIZED_INFERRED if f.derivation == Derivation.INFERRED_BY_REASONER.value else RECOGNIZED_EXPLICIT), "reconhecido"
        else:
            dest, why = DISCARDED, f"tipo {f.kind} sem consumo definido"
        rows.append({"fact_id": f.id, "kind": f.kind, "derivation": f.derivation, "destination": dest, "reason": why})
    c = Counter((r["destination"]) for r in rows)
    inferred = [r for r in rows if r["derivation"] == Derivation.INFERRED_BY_REASONER.value]
    inferred_reaching = [r for r in inferred if r["destination"] in (RECOGNIZED_INFERRED, REACHED_CANDIDATE) and r["kind"] == Kind.ATTRIBUTE_APPLIES_TO_CONCEPT.value]
    return {"rows": rows, "by_destination": dict(c), "n_inferred_facts": len(inferred), "n_inferred_attribute_facts_reaching": len(inferred_reaching),
            "inference_status": ("NO_NEW_INFERENCES" if not inferred else ("PROPAGATED" if inferred_reaching or any(r["destination"] == REACHED_CANDIDATE for r in inferred) else "PROPAGATION_FAILURE_OR_IRRELEVANT")),
            "reasoner": reg.reasoner}
