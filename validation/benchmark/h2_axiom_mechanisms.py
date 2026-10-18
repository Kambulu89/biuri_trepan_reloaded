"""H2 (Doc. 51): PROTÓTIPO, só ``validation/``, do mecanismo T1/T2 «classe definida → literal derivado». Cadeia: OWL → reasoner → registo de factos → atributos mapeados → literal binário derivado
(OR das colunas da extensão do conceito) → candidato. Serve para testes de CORREÇÃO SEMÂNTICA (extensão exata, aninhamento, dependência do reasoner, controlos, contrafactual), não para desempenho.
Nada aqui altera X de forma dependente de rótulos; sem rótulos, sem teste; o núcleo não é tocado."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from validation.benchmark.b31_facts import FactRegistry, Kind


@dataclass(frozen=True)
class DerivedLiteral:
    concept: str
    attributes: Tuple[str, ...]
    needs_reasoner: bool            # a extensão só existe graças a uma inferência (nenhuma asserção direta a produz)
    fact_ids: Tuple[str, ...]
    family: str

    @property
    def name(self) -> str:
        return f"onto_any[{self.concept}]"


def family_of(attr: str) -> str:
    return attr.split("__")[0]


def concept_extensions(reg: FactRegistry) -> Dict[str, dict]:
    """concept → {attributes, needs_reasoner, fact_ids}. Usa apenas ATTRIBUTE_APPLIES_TO_CONCEPT utilizáveis (não rejeitados)."""
    ext: Dict[str, dict] = {}
    for f in reg.by_kind(Kind.ATTRIBUTE_APPLIES_TO_CONCEPT):
        if f.use_state == "REJECTED" or len(f.entities) < 2:
            continue
        c = f.entities[1]
        e = ext.setdefault(c, {"attributes": set(), "derivs": {}, "fact_ids": []})
        e["attributes"].update(f.attributes)
        for a in f.attributes:
            e["derivs"].setdefault(a, set()).add(f.derivation)
        e["fact_ids"].append(f.id)
    for c, e in ext.items():
        e["needs_reasoner"] = any("INFERRED_BY_REASONER" in d and "ASSERTED" not in d and "DERIVED_CLOSURE" not in d for d in e["derivs"].values())
    return ext


def _is_binary(col: np.ndarray) -> bool:
    return bool(np.all(np.isin(col, (0.0, 1.0))))


def defined_union_concepts(owl_path) -> set:
    """Nomes das classes DEFINIDAS por equivalência com uma união (``owl:equivalentClass`` + ``owl:unionOf``) no ficheiro OWL — os únicos conceitos de que o mecanismo T1/T2 deriva literais.
    Classes organizacionais (grupos temáticos, raízes) e classes primitivas ficam de fora: um agrupamento NÃO é um axioma de definição (evita transformar partições em literais)."""
    import xml.etree.ElementTree as ET
    ns = {"owl": "http://www.w3.org/2002/07/owl#", "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
    root = ET.parse(owl_path).getroot()
    out = set()
    for el in root.findall("owl:Class", ns):
        name = (el.get("{%s}about" % ns["rdf"]) or "").split("#")[-1]
        for eq in el.findall("owl:equivalentClass", ns):
            if eq.findall(".//owl:unionOf", ns):
                out.add(name)
    return out


def _unsafe_unmatched(reg: FactRegistry, names: Sequence[str]) -> Dict[str, str]:
    """Atributos de dados cujo matching FALHOU de forma que torna a extensão de um conceito possivelmente INCOMPLETA: empate/ambiguidade/colisão ou com um melhor candidato existente na ontologia.
    Um atributo sem qualquer entidade candidata (não modelado) NÃO é inseguro: em mundo aberto, o nível não modelado fica fora do conceito."""
    out: Dict[str, str] = {}
    pos = set(names)
    for f in reg.by_kind(Kind.MATCH_FAILURE):
        reason = str(f.rejection_reason or "")
        if any(a in pos for a in f.attributes) and (f.entities or any(k in reason for k in ("AMBIGUOUS", "TIE", "COLLISION"))):
            for a in f.attributes:
                out[a] = reason or "MATCH_FAILURE_WITH_CANDIDATE"
    return out


def derived_union_literals(reg: FactRegistry, X: np.ndarray, names: Sequence[str], *, min_size: int = 2, only_concepts: Optional[set] = None,
                           rejected: Optional[list] = None) -> List[DerivedLiteral]:
    """Literais derivados para conceitos cuja extensão tem ≥ ``min_size`` atributos BINÁRIOS da MESMA família (variável) — evita misturar variáveis. Descarta extensões constantes no treino
    e duplicados (mesmo conjunto de atributos). Se ``only_concepts`` for dado (recomendado: ``defined_union_concepts``), só se consideram classes definidas por união; sem ele consideram-se
    TODOS os conceitos (inclui grupos organizacionais — comportamento de prototipagem, não usar em arms experimentais).
    SOLIDEZ: se algum atributo da MESMA família do conceito teve um matching ambíguo/falhado (``_unsafe_unmatched``), a extensão pode estar incompleta e o OR seria um literal INCORRETO; nesse
    caso o literal NÃO é criado e a rejeição regista-se em ``rejected`` (lista opcional de dicionários)."""
    pos = {n: i for i, n in enumerate(names)}
    unsafe = _unsafe_unmatched(reg, names)
    seen: Dict[Tuple[str, ...], DerivedLiteral] = {}
    for c, e in sorted(concept_extensions(reg).items()):
        if only_concepts is not None and c not in only_concepts:
            continue
        attrs = tuple(sorted(a for a in e["attributes"] if a in pos))
        if attrs and len({family_of(a) for a in attrs}) == 1:
            bad = sorted(a for a in unsafe if family_of(a) == family_of(attrs[0]))
            if bad:                                                     # extensão possivelmente incompleta (matching ambíguo): o OR seria incorreto — avaliado ANTES do filtro de tamanho
                if rejected is not None:
                    rejected.append({"concept": c, "reason": "INCOMPLETE_EXTENSION_UNRESOLVED_MATCH", "unresolved_attributes": bad, "match_status": {a: unsafe[a] for a in bad}})
                continue
        if len(attrs) < min_size or len({family_of(a) for a in attrs}) != 1:
            continue
        if not all(_is_binary(X[:, pos[a]]) for a in attrs):
            continue
        col = np.max(X[:, [pos[a] for a in attrs]], axis=1)
        if col.min() == col.max():
            continue                                                    # tautologia/contradição no treino
        if attrs in seen:
            continue
        seen[attrs] = DerivedLiteral(c, attrs, bool(e["needs_reasoner"]), tuple(e["fact_ids"]), family_of(attrs[0]))
    return list(seen.values())


def apply_literals(lits: Sequence[DerivedLiteral], X: np.ndarray, names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    pos = {n: i for i, n in enumerate(names)}
    cols = [np.max(X[:, [pos[a] for a in l.attributes]], axis=1) for l in lits]
    if not cols:
        return X.copy(), list(names)
    return np.column_stack([X] + cols), list(names) + [l.name for l in lits]


# ------------------------------------------------------------------------------------------------------ controlos ao nível do literal (os de ontologia constroem-se no OWL)
def control_random_unions(lits: Sequence[DerivedLiteral], names: Sequence[str], seed: int) -> List[DerivedLiteral]:
    """Controlo I: mesmos nº e tamanhos de uniões, membros aleatórios dentro da MESMA família (mesmas colunas binárias disponíveis)."""
    rng = np.random.default_rng(seed)
    fam: Dict[str, List[str]] = {}
    for n in names:
        fam.setdefault(family_of(n), []).append(n)
    out = []
    for l in lits:
        pool = sorted(fam[l.family])
        k = min(len(l.attributes), len(pool))
        out.append(DerivedLiteral(l.concept + "~rand", tuple(sorted(rng.choice(pool, size=k, replace=False))), False, (), l.family))
    return out


def control_neutralize(lits: Sequence[DerivedLiteral], concepts: Sequence[str]) -> List[DerivedLiteral]:
    """Neutralização de relações específicas: remove os literais dos conceitos indicados (ablação por axioma)."""
    drop = set(concepts)
    return [l for l in lits if l.concept not in drop]


def counterfactual_diff(full: Sequence[DerivedLiteral], ablated: Sequence[DerivedLiteral]) -> Dict[str, List[str]]:
    a, b = {l.concept: l for l in full}, {l.concept: l for l in ablated}
    return {"removed": sorted(set(a) - set(b)), "added": sorted(set(b) - set(a)), "unchanged": sorted(set(a) & set(b))}
