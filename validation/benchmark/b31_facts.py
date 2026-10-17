"""B3.1 (Doc. 34): registo TIPADO de factos semânticos, só em ``validation/`` (nenhum import de ``validation`` por ``core``; nenhum ``core`` alterado).

Recebe uma ontologia OWL, corre (a) extração dos axiomas EXPLÍCITOS, (b) o reasoner, (c) o matching atributo↔entidade do gate de qualidade existente, e produz ``SemanticFact``
com identificador estável, tipo, entidades, axiomas de origem, proveniência, estatuto epistémico, resultado do reasoner, confiança do matching, âmbito, restrições de
aplicação, uso permitido, estado de utilização e razão de rejeição. Princípios (pedido de B3.1, §3–4 e §10):

* NÃO converte relações OWL em regras tabulares. Um facto é conhecimento tipado; o que o consumidor pode fazer com ele é ``permitted_use`` (NONE|PREFERENCE|HARD_CONSTRAINT).
* Mundo aberto: ausência de uma relação nunca gera um facto negativo. ``CONCEPT_DISJOINTNESS`` só existe se o axioma estiver declarado e NÃO autoriza eliminar combinações
  de atributos (``permitted_use=NONE``: não há contrato que ligue disjunção de conceitos à co-ocorrência de atributos numa linha).
* ``rdfs:domain``/``rdfs:range``/cardinalidades NÃO são validadores fechados de linhas: ``VALUE_BOUND`` (facetas) fica com ``permitted_use=NONE`` e a restrição
  ``requires_row_as_individual_contract`` até existir um contrato que justifique tratar a linha como indivíduo de valores completos.
* Só ``DOMAIN_DOCUMENTED`` com fonte, condições de validade integralmente declaradas, classe de relação formal/identidade E revisão de domínio ``APPROVED`` por um humano pode ter
  ``HARD_CONSTRAINT``. O compilador nunca marca ``APPROVED``: sem revisão humana o facto fica ``DOMAIN_REVIEW_PENDING`` (``permitted_use`` ≤ PREFERENCE).
* ``EMPIRICALLY_OBSERVED`` (p. ex. 100 % das linhas de treino) e ``HYPOTHESIS``/``UNKNOWN`` nunca justificam eliminação.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class Epistemic(str, Enum):
    LOGICALLY_ENTAILED = "LOGICALLY_ENTAILED"
    DOMAIN_DOCUMENTED = "DOMAIN_DOCUMENTED"
    EMPIRICALLY_OBSERVED = "EMPIRICALLY_OBSERVED"
    HYPOTHESIS = "HYPOTHESIS"
    UNKNOWN = "UNKNOWN"


class Derivation(str, Enum):
    ASSERTED = "ASSERTED"                          # axioma explícito na ontologia
    DERIVED_CLOSURE = "DERIVED_CLOSURE"            # fecho transitivo/herança de axiomas explícitos (RDFS/OWL), calculado pelo compilador
    INFERRED_BY_REASONER = "INFERRED_BY_REASONER"  # subsunção/equivalência inferida pelo reasoner e não declarada
    DOCUMENT = "DOCUMENT"                          # fonte documental externa (sidecar)
    DATA = "DATA"                                  # observação nos dados
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Kind(str, Enum):
    CONCEPT_SUBSUMPTION = "CONCEPT_SUBSUMPTION"
    CONCEPT_EQUIVALENCE = "CONCEPT_EQUIVALENCE"
    CONCEPT_DISJOINTNESS = "CONCEPT_DISJOINTNESS"
    PROPERTY_SUBSUMPTION = "PROPERTY_SUBSUMPTION"
    PROPERTY_EQUIVALENCE = "PROPERTY_EQUIVALENCE"
    ATTRIBUTE_APPLIES_TO_CONCEPT = "ATTRIBUTE_APPLIES_TO_CONCEPT"
    VALUE_BOUND = "VALUE_BOUND"
    ORDER_RELATION = "ORDER_RELATION"
    SAME_MEASUREMENT_FAMILY = "SAME_MEASUREMENT_FAMILY"
    EMPIRICAL_ORDER_OBSERVATION = "EMPIRICAL_ORDER_OBSERVATION"
    MATCH_FAILURE = "MATCH_FAILURE"
    UNSUPPORTED_AXIOM = "UNSUPPORTED_AXIOM"


class RelationClass(str, Enum):
    MATHEMATICAL_IDENTITY = "MATHEMATICAL_IDENTITY"
    FORMAL_DOMAIN_CONSTRAINT = "FORMAL_DOMAIN_CONSTRAINT"
    EMPIRICAL_REGULARITY = "EMPIRICAL_REGULARITY"
    HYPOTHESIS = "HYPOTHESIS"
    NONE = "NONE"


class PermittedUse(str, Enum):
    NONE = "NONE"
    PREFERENCE = "PREFERENCE"
    HARD_CONSTRAINT = "HARD_CONSTRAINT"


class UseState(str, Enum):
    NOT_USED = "NOT_USED"
    AVAILABLE = "AVAILABLE"
    USED_AS_PREFERENCE = "USED_AS_PREFERENCE"
    USED_AS_CONSTRAINT = "USED_AS_CONSTRAINT"
    REJECTED = "REJECTED"


# condições exigidas para tratar uma relação numérica de ordem/identidade como restrição formal (pedido B3.1 §10)
ORDER_VALIDITY_CONDITIONS = ("same_quantity", "same_observation_set", "compatible_statistical_definitions", "preprocessing_preserves_relation", "missing_values_handled")


@dataclass
class SemanticFact:
    id: str
    kind: str
    entities: Tuple[str, ...]
    attributes: Tuple[str, ...]
    source_axioms: Tuple[str, ...]
    provenance: Dict[str, Any]
    epistemic: str
    derivation: str
    reasoner_result: Optional[Dict[str, Any]]
    matching_confidence: Optional[float]
    scope: Dict[str, Any]
    application_restrictions: Tuple[str, ...]
    permitted_use: str
    use_state: str = UseState.NOT_USED.value
    rejection_reason: Optional[str] = None
    relation_class: str = RelationClass.NONE.value
    validity_conditions: Dict[str, Any] = field(default_factory=dict)
    review: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("entities", "attributes", "source_axioms", "application_restrictions"):
            d[k] = list(d[k])
        return d


def make_id(kind: str, entities: Sequence[str], axioms: Sequence[str], extra: str = "") -> str:
    h = hashlib.sha256(json.dumps([kind, list(entities), list(axioms), extra], sort_keys=True).encode()).hexdigest()
    return f"F-{kind[:3]}-{h[:12]}"


def decide_permitted_use(fact: SemanticFact) -> str:
    """Política única (testada) do uso máximo admissível. Nunca depende do dataset; só do estatuto, do tipo e da revisão."""
    ep, kind = fact.epistemic, fact.kind
    if fact.rejection_reason or fact.use_state == UseState.REJECTED.value:
        return PermittedUse.NONE.value
    if ep in (Epistemic.HYPOTHESIS.value, Epistemic.UNKNOWN.value, Epistemic.EMPIRICALLY_OBSERVED.value):
        return PermittedUse.NONE.value
    if kind in (Kind.CONCEPT_DISJOINTNESS.value, Kind.VALUE_BOUND.value, Kind.MATCH_FAILURE.value, Kind.UNSUPPORTED_AXIOM.value, Kind.EMPIRICAL_ORDER_OBSERVATION.value):
        return PermittedUse.NONE.value
    if ep == Epistemic.DOMAIN_DOCUMENTED.value and kind == Kind.ORDER_RELATION.value:
        formal = fact.relation_class in (RelationClass.MATHEMATICAL_IDENTITY.value, RelationClass.FORMAL_DOMAIN_CONSTRAINT.value)
        conds = all(bool(fact.validity_conditions.get(c)) for c in ORDER_VALIDITY_CONDITIONS)
        approved = fact.review.get("domain_expert_status") == "APPROVED" and fact.review.get("methodology_status") == "APPROVED"
        if formal and conds and approved:
            return PermittedUse.HARD_CONSTRAINT.value
        return PermittedUse.PREFERENCE.value if formal and conds else PermittedUse.NONE.value
    return PermittedUse.PREFERENCE.value


def is_eliminable(fact: SemanticFact) -> bool:
    """Único predicado que um consumidor pode usar para eliminar candidatos/consultas: exige HARD_CONSTRAINT e estado não rejeitado."""
    return fact.permitted_use == PermittedUse.HARD_CONSTRAINT.value and fact.use_state != UseState.REJECTED.value and fact.epistemic in (
        Epistemic.DOMAIN_DOCUMENTED.value, Epistemic.LOGICALLY_ENTAILED.value)


@dataclass
class FactRegistry:
    facts: List[SemanticFact] = field(default_factory=list)
    ontology_status: str = "NO_ONTOLOGY"
    ontology_hash: Optional[str] = None
    reasoner: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    feature_entity: Dict[str, Optional[str]] = field(default_factory=dict)
    match_confidence: Dict[str, float] = field(default_factory=dict)
    unsupported: Dict[str, int] = field(default_factory=dict)
    taxonomy: Dict[str, Any] = field(default_factory=dict)

    def add(self, f: SemanticFact) -> SemanticFact:
        f.permitted_use = decide_permitted_use(f)
        self.facts.append(f)
        return f

    def by_kind(self, *kinds: Kind) -> List[SemanticFact]:
        ks = {k.value for k in kinds}
        return [f for f in self.facts if f.kind in ks]

    def get(self, fid: str) -> Optional[SemanticFact]:
        return next((f for f in self.facts if f.id == fid), None)

    def usable(self) -> List[SemanticFact]:
        return [f for f in self.facts if f.use_state != UseState.REJECTED.value and f.permitted_use != PermittedUse.NONE.value]

    def summary(self) -> Dict[str, Any]:
        from collections import Counter
        return {"ontology_status": self.ontology_status, "n_facts": len(self.facts), "by_kind": dict(Counter(f.kind for f in self.facts)), "by_epistemic": dict(Counter(f.epistemic for f in self.facts)),
                "by_permitted_use": dict(Counter(f.permitted_use for f in self.facts)), "by_use_state": dict(Counter(f.use_state for f in self.facts)), "unsupported_axioms": dict(self.unsupported)}

    def to_json(self) -> str:
        return json.dumps({"summary": self.summary(), "facts": [f.to_dict() for f in self.facts]}, indent=1, default=str)


# ------------------------------------------------------------------------------------------------------ compilador
def _name(x) -> Optional[str]:
    n = getattr(x, "name", None)
    return str(n) if n else None


def _named_classes(items: Iterable[Any], onto_classes: set) -> List[str]:
    out = []
    for it in items:
        n = _name(it)
        if n and n in onto_classes and n != "Thing":
            out.append(n)
    return out


def _restriction_kind(r) -> Optional[str]:
    t = getattr(r, "type", None)
    mapping = {24: "someValuesFrom", 25: "allValuesFrom", 26: "hasValue", 27: "cardinality", 28: "minCardinality", 29: "maxCardinality"}
    if t is None:
        return None
    return mapping.get(int(t) if isinstance(t, int) else -1, f"restriction_type_{t}")


def extract_explicit(onto) -> Dict[str, Any]:
    """Axiomas EXPLÍCITOS (antes de o reasoner materializar nada em ``is_a``)."""
    classes = {c.name: c for c in onto.classes()}
    cn = set(classes)
    sub: List[Tuple[str, str]] = []
    equiv: List[Tuple[str, str]] = []
    disj: List[Tuple[str, str]] = []
    unsupported: Dict[str, int] = {}
    for n, c in classes.items():
        for p in _named_classes(c.is_a, cn):
            sub.append((n, p))
        for it in list(c.is_a) + list(c.equivalent_to):
            if hasattr(it, "type") and not hasattr(it, "name"):
                k = _restriction_kind(it) or "class_expression"
                unsupported[k] = unsupported.get(k, 0) + 1
            elif _name(it) is None:
                unsupported["class_expression"] = unsupported.get("class_expression", 0) + 1
        for e in _named_classes(c.equivalent_to, cn):
            equiv.append((n, e))
        try:
            for d in c.disjoints():
                ents = sorted({x.name for x in d.entities if _name(x) in cn})
                for i in range(len(ents)):
                    for j in range(i + 1, len(ents)):
                        disj.append((ents[i], ents[j]))
        except Exception:
            pass
    props = {q.name: q for q in onto.data_properties()}
    dom: Dict[str, List[str]] = {}
    psub: List[Tuple[str, str]] = []
    pequiv: List[Tuple[str, str]] = []
    for n, q in props.items():
        dom[n] = [d for d in _named_classes(q.domain, cn)]
        for s in q.is_a:
            if _name(s) in props and _name(s) != n:
                psub.append((n, _name(s)))
        for e in q.equivalent_to:
            if _name(e) in props and _name(e) != n:
                pequiv.append((n, _name(e)))
    for q in list(onto.object_properties()):
        if getattr(q, "property_chain", None):
            unsupported["propertyChainAxiom"] = unsupported.get("propertyChainAxiom", 0) + 1
    return {"classes": sorted(cn), "sub": sorted(set(sub)), "equiv": sorted({tuple(sorted(p)) for p in equiv}), "disj": sorted({tuple(sorted(p)) for p in disj}), "props": sorted(props), "domain": dom,
            "psub": sorted(set(psub)), "pequiv": sorted({tuple(sorted(p)) for p in pequiv}), "unsupported": unsupported,
            "n_object_properties": len(list(onto.object_properties()))}


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def concept_closure(classes: Sequence[str], sub: Sequence[Tuple[str, str]], equiv: Sequence[Tuple[str, str]], inferred_sub: Sequence[Tuple[str, str]],
                    inferred_equiv: Sequence[Tuple[str, str]]) -> Dict[str, Dict[str, Tuple[str, Tuple[str, ...]]]]:
    """Fecho dos antepassados de cada conceito: ``anc[c][a] = (derivation, axiom_path)``. Equivalências (asseridas ou inferidas) fundem os nós (mesma classe de equivalência).
    ``derivation``: ASSERTED (aresta explícita direta), DERIVED_CLOSURE (cadeia de explícitas/equivalência), INFERRED_BY_REASONER (alguma aresta inferida no caminho)."""
    uf = _UF(classes)
    for a, b in list(equiv) + list(inferred_equiv):
        if a in uf.p and b in uf.p:
            uf.union(a, b)
    members: Dict[str, List[str]] = {}
    for c in classes:
        members.setdefault(uf.find(c), []).append(c)
    edges: Dict[str, List[Tuple[str, str, bool, str]]] = {}
    for kind, lst, inf in (("sub", sub, False), ("sub", inferred_sub, True)):
        for c, p in lst:
            if c in uf.p and p in uf.p:
                edges.setdefault(uf.find(c), []).append((uf.find(p), "inferred" if inf else "asserted", inf, f"{c} SubClassOf {p}" + (" [inferred]" if inf else "")))
    anc: Dict[str, Dict[str, Tuple[str, Tuple[str, ...]]]] = {}
    for c in classes:
        root = uf.find(c)
        best: Dict[str, Tuple[bool, int, Tuple[str, ...]]] = {}
        stack: List[Tuple[str, bool, Tuple[str, ...]]] = [(root, False, ())]
        while stack:
            node, inf_flag, path = stack.pop()
            for (nxt, _k, inf_edge, ax) in edges.get(node, []):
                f2 = inf_flag or inf_edge
                p2 = path + (ax,)
                key = (f2, len(p2))
                if nxt == root:
                    continue
                if nxt not in best or key < (best[nxt][0], best[nxt][1]):
                    best[nxt] = (f2, len(p2), p2)
                    stack.append((nxt, f2, p2))
        res: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
        for r, (f2, ln, path) in best.items():
            der = Derivation.INFERRED_BY_REASONER.value if f2 else (Derivation.ASSERTED.value if ln == 1 and c == root else Derivation.DERIVED_CLOSURE.value)
            for m in members[r]:
                res[m] = (der, path)
        # a própria classe de equivalência (classes equivalentes são antepassados "ao mesmo nível")
        for m in members[root]:
            if m != c:
                res.setdefault(m, (Derivation.DERIVED_CLOSURE.value, (f"{c} EquivalentTo {m}",)))
        anc[c] = res
    return anc


def compile_registry(owl_path: str | Path, feature_names: Sequence[str], *, documented: Optional[Sequence[Dict[str, Any]]] = None, run_reasoner: bool = True,
                     reasoner_engine: str = "hermit") -> FactRegistry:
    """OWL → (explícitos) → reasoner → matching → factos. Nunca lança por ontologia inválida: devolve registo com ``ontology_status`` e factos rejeitados."""
    from core.ontology_quality import OntologyQualityGate
    from core.ontology_semantic_graph import _explicit_entity_bounds
    from core.owl_runtime import load_ontology_isolated
    names = [str(n) for n in feature_names]
    reg = FactRegistry()
    path = Path(owl_path)
    reg.ontology_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    prov = {"ontology_file": path.name, "ontology_sha256": reg.ontology_hash, "compiler": "validation/benchmark/b31_facts.py"}
    try:
        onto = load_ontology_isolated(str(path))
        ex = extract_explicit(onto)
    except Exception as exc:                                     # OWL ilegível
        reg.ontology_status = "INVALID_OWL"
        reg.quality = {"error": f"{type(exc).__name__}: {exc}"}
        return reg
    reg.unsupported = dict(ex["unsupported"])
    # --- reasoner
    rep: Dict[str, Any] = {"executed": False, "consistent": None}
    if run_reasoner:
        from core.ontology_reasoner import run_owl_reasoner
        try:
            rep = run_owl_reasoner(onto, engine=reasoner_engine, infer_property_values=True, debug=0)
        except Exception as exc:
            rep = {"executed": False, "consistent": None, "error": f"{type(exc).__name__}: {exc}"}
    reg.reasoner = {k: rep.get(k) for k in ("engine", "executed", "consistent", "unsatisfiable_classes", "inferred_axioms_count", "error")}
    if rep.get("executed") and rep.get("consistent") is False or rep.get("unsatisfiable_classes"):
        reg.ontology_status = "INCONSISTENT"
    elif not rep.get("executed"):
        reg.ontology_status = "REASONER_NOT_EXECUTED"
    inf_sub = [tuple(p) for p in (rep.get("inferred_subclass_relations") or [])]
    inf_eq = [tuple(sorted(p)) for p in (rep.get("inferred_equivalences") or []) if len(p) == 2]
    rr = {"engine": rep.get("engine"), "executed": bool(rep.get("executed")), "consistent": rep.get("consistent")}
    # --- matching e gate
    gate = OntologyQualityGate()
    decisions = gate.match_features(names, onto)
    try:
        quality = gate.evaluate(names, onto, reasoner_report=rep, require_reasoner=run_reasoner)
        reg.quality = {"accepted": bool(quality.accepted), "status": quality.status, "issues": list(quality.issues), "warnings": list(quality.warnings)}
        if reg.ontology_status == "NO_ONTOLOGY":
            reg.ontology_status = "VALID" if quality.accepted else f"REJECTED_BY_GATE:{quality.status}"
    except Exception as exc:
        reg.quality = {"accepted": False, "error": f"{type(exc).__name__}: {exc}"}
        reg.ontology_status = reg.ontology_status if reg.ontology_status != "NO_ONTOLOGY" else "REJECTED_BY_GATE:ERROR"
    blocked = reg.ontology_status != "VALID"
    block_reason = None if not blocked else f"ontology_status={reg.ontology_status}"
    prov_r = dict(prov, reasoner=rr)
    props = set(ex["props"])
    for d in decisions:
        tie = bool(d.accepted and d.ambiguity_margin <= 1e-12)           # empate perfeito entre duas entidades: o gate do núcleo aceita-o (score ≥ 0,98); aqui é ambíguo
        usable = bool(d.accepted and not tie)
        reg.feature_entity[d.feature] = d.entity_name if usable else None
        reg.match_confidence[d.feature] = float(d.score)
        if not usable:
            status = "PERFECT_TIE_AMBIGUOUS" if tie else d.status
            ents = (str(d.entity_name),) if d.entity_name else ()
            reg.add(SemanticFact(make_id(Kind.MATCH_FAILURE.value, [d.feature], [status]), Kind.MATCH_FAILURE.value, ents, (d.feature,),
                                 (f"match {d.feature}: {status} (melhor candidato {d.entity_name or 'nenhum'}, score {d.score:.2f}, margem {d.ambiguity_margin:.2f})",),
                                 dict(prov_r, matching_step="OntologyQualityGate.match_features"), Epistemic.UNKNOWN.value, Derivation.NOT_APPLICABLE.value, rr, float(d.score),
                                 {"level": "tabular_attribute"}, ("atributo sem entidade ontológica aceite",), PermittedUse.NONE.value, UseState.REJECTED.value, status))
    # --- fecho de conceitos (SEM aplicar se bloqueado: factos ficam REJECTED com razão)
    anc = concept_closure(ex["classes"], ex["sub"], ex["equiv"], inf_sub, inf_eq)
    reg.taxonomy = {"classes": ex["classes"], "ancestors": {c: {a: v[0] for a, v in d.items()} for c, d in anc.items()}}

    def fact(kind, ents, attrs, axioms, ep, der, scope, restr, conf=None, rel_class=RelationClass.NONE.value):
        f = SemanticFact(make_id(kind, ents, axioms, der), kind, tuple(ents), tuple(attrs), tuple(axioms), dict(prov_r), ep, der, rr, conf, scope, tuple(restr), PermittedUse.NONE.value,
                         relation_class=rel_class)
        if blocked:
            f.use_state, f.rejection_reason = UseState.REJECTED.value, block_reason
        else:
            f.use_state = UseState.AVAILABLE.value
        return reg.add(f)

    ent = Epistemic.LOGICALLY_ENTAILED.value
    for (c, p) in ex["sub"]:
        fact(Kind.CONCEPT_SUBSUMPTION.value, (c, p), (), (f"{c} SubClassOf {p}",), ent, Derivation.ASSERTED.value, {"level": "ontology_schema"}, ())
    for (c, p) in inf_sub:
        if (c, p) not in set(ex["sub"]):
            fact(Kind.CONCEPT_SUBSUMPTION.value, (c, p), (), (f"{c} SubClassOf {p} [inferred]",), ent, Derivation.INFERRED_BY_REASONER.value, {"level": "ontology_schema"}, ())
    for (a, b) in ex["equiv"]:
        fact(Kind.CONCEPT_EQUIVALENCE.value, (a, b), (), (f"{a} EquivalentTo {b}",), ent, Derivation.ASSERTED.value, {"level": "ontology_schema"}, ())
    for (a, b) in inf_eq:
        if (a, b) not in set(ex["equiv"]):
            fact(Kind.CONCEPT_EQUIVALENCE.value, (a, b), (), (f"{a} EquivalentTo {b} [inferred]",), ent, Derivation.INFERRED_BY_REASONER.value, {"level": "ontology_schema"}, ())
    for (a, b) in ex["disj"]:
        fact(Kind.CONCEPT_DISJOINTNESS.value, (a, b), (), (f"{a} DisjointWith {b}",), ent, Derivation.ASSERTED.value, {"level": "ontology_schema"},
             ("open_world: disjunção de conceitos não autoriza eliminar combinações de atributos de uma linha (sem contrato)",))
    for (p, q) in ex["psub"]:
        fact(Kind.PROPERTY_SUBSUMPTION.value, (p, q), (), (f"{p} SubPropertyOf {q}",), ent, Derivation.ASSERTED.value, {"level": "ontology_schema"}, ())
    for (p, q) in ex["pequiv"]:
        fact(Kind.PROPERTY_EQUIVALENCE.value, (p, q), (), (f"{p} EquivalentProperty {q}",), ent, Derivation.ASSERTED.value, {"level": "ontology_schema"}, ())
    # propriedades: fecho de super-propriedades (sub, equivalência) e herança de domínio
    puf = _UF(sorted(props))
    for a, b in ex["pequiv"]:
        puf.union(a, b)
    psup: Dict[str, set] = {p: set() for p in props}
    for p, q in ex["psub"]:
        psup[p].add(q)
    changed = True
    while changed:
        changed = False
        for p in props:
            for q in list(psup[p]):
                for r in psup.get(q, ()):
                    if r not in psup[p]:
                        psup[p].add(r); changed = True
    for a, b in ex["pequiv"]:
        for p in (a, b):
            o = b if p == a else a
            for r in psup.get(o, ()):
                psup[p].add(r)
    # --- factos de atributo
    by_entity = {}
    for f_name, e_name in reg.feature_entity.items():
        if e_name:
            by_entity.setdefault(e_name, []).append(f_name)
    for f_name, e_name in sorted(reg.feature_entity.items()):
        if not e_name:
            continue
        conf = reg.match_confidence.get(f_name)
        if e_name not in props:
            fact(Kind.MATCH_FAILURE.value, (e_name,), (f_name,), (f"match {f_name} -> {e_name}",), Epistemic.UNKNOWN.value, Derivation.NOT_APPLICABLE.value, {"level": "tabular_attribute"},
                 ("atributo casado com um conceito (não com propriedade de dados): sem contrato de interpretação",), conf).use_state = UseState.REJECTED.value
            reg.facts[-1].rejection_reason = "matched_entity_is_not_a_data_property"
            reg.facts[-1].permitted_use = PermittedUse.NONE.value
            continue
        all_props = {e_name} | {q for q in psup.get(e_name, ())} | {m for m in props if puf.find(m) == puf.find(e_name)}
        rank = {Derivation.ASSERTED.value: 0, Derivation.DERIVED_CLOSURE.value: 1, Derivation.INFERRED_BY_REASONER.value: 2}
        cand: Dict[str, Tuple[str, Tuple[str, ...]]] = {}

        def offer(concept: str, der: str, path: Tuple[str, ...]):
            cur = cand.get(concept)
            if cur is None or (rank[der], len(path)) < (rank[cur[0]], len(cur[1])):
                cand[concept] = (der, path)

        for q in sorted(all_props):
            direct = (q == e_name)
            for d in ex["domain"].get(q, []):
                path0 = (f"{q} Domain {d}",) if direct else (f"{e_name} SubPropertyOf* {q}", f"{q} Domain {d}")
                offer(d, Derivation.ASSERTED.value if direct else Derivation.DERIVED_CLOSURE.value, path0)
                for a, (der, path) in anc.get(d, {}).items():
                    inferred = der == Derivation.INFERRED_BY_REASONER.value
                    offer(a, Derivation.INFERRED_BY_REASONER.value if inferred else Derivation.DERIVED_CLOSURE.value, path0 + tuple(path))
        for concept, (der, path) in sorted(cand.items()):
            fact(Kind.ATTRIBUTE_APPLIES_TO_CONCEPT.value, (e_name, concept), (f_name,), path, ent, der,
                 {"level": "tabular_attribute", "interpretation": "propriedade aplicável a indivíduos do conceito"}, ("não é validador fechado de linhas (rdfs:domain não classifica linhas)",), conf)
    # facetas numéricas (VALUE_BOUND) — sem contrato de linha ⇒ permitted_use NONE
    for f_name, e_name in sorted(reg.feature_entity.items()):
        if not e_name or e_name not in props:
            continue
        ent_obj = next((q for q in onto.data_properties() if q.name == e_name), None)
        lo, hi = _explicit_entity_bounds(ent_obj) if ent_obj is not None else (None, None)
        if lo is not None or hi is not None:
            fact(Kind.VALUE_BOUND.value, (e_name,), (f_name,), (f"{e_name} Range xsd:[{lo},{hi}]",), ent, Derivation.ASSERTED.value,
                 {"level": "individual_values", "bounds": [lo, hi]}, ("requires_row_as_individual_contract", "rdfs:range não é validador fechado de linhas"), reg.match_confidence.get(f_name))
    for k, n in reg.unsupported.items():
        fact(Kind.UNSUPPORTED_AXIOM.value, (k,), (), (f"{n}× {k}",), Epistemic.UNKNOWN.value, Derivation.NOT_APPLICABLE.value, {"level": "ontology_schema"},
             ("tipo de axioma sem tradução para atributos tabulares: reconhecido mas NÃO interpretado",)).use_state = UseState.REJECTED.value
        reg.facts[-1].rejection_reason = "unsupported_axiom_type_no_tabular_contract"
        reg.facts[-1].permitted_use = PermittedUse.NONE.value
    for dfact in documented or []:
        add_documented(reg, dfact, prov_r)
    return reg


def add_documented(reg: FactRegistry, d: Dict[str, Any], prov: Dict[str, Any]) -> SemanticFact:
    """Facto documentado externamente (sidecar). Exige ``source``; para ORDER_RELATION as 5 condições de validade e ``relation_class``.
    O estatuto é DOMAIN_DOCUMENTED só se a fonte e as condições estiverem completas; caso contrário HYPOTHESIS. Revisão: campos ``review`` fornecidos por HUMANOS;
    por omissão ``DOMAIN_REVIEW_PENDING``. Este compilador nunca escreve ``APPROVED``."""
    kind = d["kind"]
    attrs = tuple(d["attributes"])
    src = d.get("source") or {}
    rc = d.get("relation_class", RelationClass.NONE.value)
    conds = dict(d.get("validity_conditions", {}))
    review = {"domain_expert_status": (d.get("review") or {}).get("domain_expert_status", "DOMAIN_REVIEW_PENDING"),
              "methodology_status": (d.get("review") or {}).get("methodology_status", "REVIEW_PENDING"),
              "ontology_engineer_status": (d.get("review") or {}).get("ontology_engineer_status", "REVIEW_PENDING")}
    complete = bool(src.get("citation")) and (kind != Kind.ORDER_RELATION.value or all(c in conds for c in ORDER_VALIDITY_CONDITIONS))
    if not complete:
        ep, why = Epistemic.HYPOTHESIS.value, "fonte ou condições de validade em falta"
    elif rc == RelationClass.HYPOTHESIS.value:
        ep, why = Epistemic.HYPOTHESIS.value, "declarada como hipótese"
    elif rc == RelationClass.EMPIRICAL_REGULARITY.value:
        ep, why = Epistemic.DOMAIN_DOCUMENTED.value, None        # regularidade documentada e aproximada: no máximo preferência
    else:
        ep, why = Epistemic.DOMAIN_DOCUMENTED.value, None
    f = SemanticFact(make_id(kind, attrs, [json.dumps(src, sort_keys=True)], rc), kind, tuple(d.get("entities", ())), attrs, (src.get("citation", "SEM FONTE"),),
                     dict(prov, document_source=src), ep, Derivation.DOCUMENT.value, None, None, {"level": d.get("level", "row_values"), "validity": d.get("scope", {})},
                     tuple(d.get("application_restrictions", ())), PermittedUse.NONE.value, UseState.AVAILABLE.value if why is None else UseState.REJECTED.value, why, rc, conds, review)
    if why is None and rc in (RelationClass.MATHEMATICAL_IDENTITY.value, RelationClass.FORMAL_DOMAIN_CONSTRAINT.value) and review["domain_expert_status"] != "APPROVED":
        f.application_restrictions = f.application_restrictions + ("DOMAIN_REVIEW_PENDING: sem revisão de domínio aprovada não é restrição obrigatória",)
    return reg.add(f)


def observe_order_relation(X, names: Sequence[str], a: str, b: str, *, prov: Optional[Dict[str, Any]] = None) -> SemanticFact:
    """Observação EMPÍRICA «a ≤ b» no treino. Nunca promovida a axioma universal, mesmo com 100 % de satisfação (permitted_use=NONE)."""
    X = np.asarray(X, dtype=float)
    ia, ib = list(names).index(a), list(names).index(b)
    ok = np.isfinite(X[:, ia]) & np.isfinite(X[:, ib])
    viol = float(np.mean(X[ok, ia] > X[ok, ib])) if ok.any() else None
    return SemanticFact(make_id(Kind.EMPIRICAL_ORDER_OBSERVATION.value, [a, b], [f"{a}<={b}"], "data"), Kind.EMPIRICAL_ORDER_OBSERVATION.value, (), (a, b), (f"observed {a} <= {b}",),
                        dict(prov or {}, n_rows=int(ok.sum())), Epistemic.EMPIRICALLY_OBSERVED.value, Derivation.DATA.value, None, None,
                        {"level": "training_rows", "violation_rate": viol, "n_rows": int(ok.sum())}, ("regularidade empírica: não é universal nem restrição",), PermittedUse.NONE.value)
