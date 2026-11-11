"""Leitura, na própria ontologia OWL, do esquema do dataset: coluna-alvo, ordem das classes e tipos dos atributos.

Genérico e agnóstico: nenhum nome de dataset; o vocabulário (multilingue) vem de ``ontology_schema_vocabulary.json``.
A ontologia só DECLARA; quem decide e valida contra o ficheiro de dados e o modelo treinado é ``core.dataset_schema``.

Fontes lidas (por ordem de prioridade dentro da ontologia):
1. Anotações/propriedades canónicas ``biuri*`` (ou sinónimos do vocabulário) em qualquer entidade:
   ``columnName``, ``dataType``, ``isTarget``, ``classOrder``, ``classIndex``/``classValue``, ``ordinalOrder``.
2. Indivíduos de classes "descritor de atributo" e "etiqueta-alvo" (``Atributo``/``EtiquetaObjetivo``…), por vocabulário.
3. Propriedades de dados cujo rótulo/nome coincide com uma coluna do dataset: tipo pelo ``rdfs:range`` (xsd).
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import datetime as _dt

_VOCAB_PATH = Path(__file__).with_name("ontology_schema_vocabulary.json")


def norm(text: Any) -> str:
    """Minúsculas, sem acentos, só alfanuméricos."""
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", t.lower())


@lru_cache(maxsize=4)
def load_vocabulary(path: Optional[str] = None) -> Dict[str, Any]:
    data = json.loads(Path(path or _VOCAB_PATH).read_text(encoding="utf-8"))
    out = {"properties": {k: {norm(x) for x in v} for k, v in data["properties"].items()},
           "classes": {k: {norm(x) for x in v} for k, v in data["classes"].items()},
           "kinds": {}, "target_markers": {norm(x) for x in data.get("target_markers", [])}}
    for kind, words in data["kinds"].items():
        for w in words:
            out["kinds"][norm(w)] = kind
    return out


@dataclass
class OntologySchemaDeclaration:
    """O que a ontologia declara sobre o dataset (pode estar vazio)."""
    target_column: Optional[str] = None
    target_evidence: str = ""
    class_labels: List[str] = field(default_factory=list)          # ordenadas conforme a ontologia
    class_order_evidence: str = ""
    class_order_explicit: bool = False                              # True: ordem declarada (índice/lista); False: ordem de nomeação
    column_types: Dict[str, str] = field(default_factory=dict)      # coluna -> kind
    type_evidence: Dict[str, str] = field(default_factory=dict)
    ordinal_orders: Dict[str, List[str]] = field(default_factory=dict)
    declared_columns: List[str] = field(default_factory=list)       # colunas que a ontologia descreve
    notes: List[str] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return bool(self.target_column or self.class_labels or self.column_types or self.declared_columns)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def summary_lines(self) -> List[str]:
        if not self.has_any:
            return ["📚 Esquema na ontologia: nada declarado (usa-se o ficheiro de dados)."]
        lines = ["📚 Esquema declarado na ontologia:"]
        if self.target_column:
            lines.append(f"   • alvo: {self.target_column} ({self.target_evidence})")
        if self.class_labels:
            lines.append(f"   • classes: {', '.join(self.class_labels)} ({self.class_order_evidence})")
        if self.declared_columns:
            lines.append(f"   • {len(self.declared_columns)} atributos descritos; tipos declarados: {len(self.column_types)}")
        return lines


# ----------------------------------------------------------------------------- leitura
def _world(ontology):
    return getattr(ontology, "world", None) or getattr(getattr(ontology, "namespace", None), "world", None)


def _role_pairs(ontology, synonyms: set) -> List[Tuple[Any, Any]]:
    """Pares (sujeito, valor) de todas as propriedades (dados/anotação) cujo nome normalizado está em ``synonyms``."""
    world = _world(ontology)
    if world is None:
        return []
    out: List[Tuple[Any, Any]] = []
    props: List[Any] = []
    for method in ("data_properties", "annotation_properties"):
        try:
            props.extend(getattr(world, method)())
        except Exception:
            continue
    seen = set()
    for prop in props:
        if id(prop) in seen or norm(getattr(prop, "name", "")) not in synonyms:
            continue
        seen.add(id(prop))
        try:
            out.extend(list(prop.get_relations()))
        except Exception:
            continue
    return out


def _text(value: Any) -> str:
    v = getattr(value, "name", value)
    return str(v).strip()


def _natural_key(name: str):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(name))]


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "sim", "si", "sí"}


def range_kind(entity) -> Optional[str]:
    """Tipo de valor (numeric/categorical/boolean/datetime) a partir do ``rdfs:range`` de uma propriedade de dados."""
    try:
        for item in list(getattr(entity, "range", []) or []):
            base = getattr(item, "base_datatype", item)
            if base in (int, float):
                return "numeric"
            if base is str:
                return "categorical"
            if base is bool:
                return "boolean"
            if base in (_dt.datetime, _dt.date):
                return "datetime"
    except Exception:
        return None
    return None


def _kind_from_token(token: str, vocab) -> Optional[str]:
    return vocab["kinds"].get(norm(token))


def _split_list(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"[;,|]", str(text)) if p.strip()]


def read_ontology_schema(ontology, columns: Optional[Sequence[str]] = None, vocabulary: Optional[Dict[str, Any]] = None) -> OntologySchemaDeclaration:
    """Lê a declaração de esquema. ``columns`` (opcional) permite reconhecer propriedades-por-coluna pelo rótulo/nome."""
    decl = OntologySchemaDeclaration()
    if ontology is None:
        return decl
    vocab = vocabulary or load_vocabulary()
    props = vocab["properties"]
    try:
        _read(ontology, columns, vocab, props, decl)
    except Exception as exc:  # a leitura nunca impede o carregamento; a falha fica registada
        decl.notes.append(f"leitura do esquema na ontologia falhou: {exc}")
    decl.declared_columns = list(dict.fromkeys(decl.declared_columns))
    return decl


def _read(ontology, columns, vocab, props, decl: OntologySchemaDeclaration) -> None:
    # --- descritores de atributo: entidades com nome de coluna declarado
    names: Dict[Any, str] = {}
    for ent, val in _role_pairs(ontology, props["column_name"]):
        names[ent] = _text(val)
    types: Dict[Any, str] = {}
    for ent, val in _role_pairs(ontology, props["data_type"]):
        types[ent] = _text(val)
    ordinals: Dict[Any, str] = {}
    for ent, val in _role_pairs(ontology, props["ordinal_order"]):
        ordinals[ent] = _text(val)
    targets = {ent for ent, val in _role_pairs(ontology, props["is_target"]) if _bool(_text(val))}

    for ent, col in names.items():
        decl.declared_columns.append(col)
        tok = types.get(ent)
        if tok is not None:
            if norm(tok) in vocab["target_markers"]:
                targets.add(ent)
            else:
                kind = _kind_from_token(tok, vocab)
                if kind:
                    decl.column_types[col] = kind
                    decl.type_evidence[col] = f"{getattr(ent, 'name', ent)}.tipo={tok}"
        if ent in ordinals:
            decl.ordinal_orders[col] = _split_list(ordinals[ent])
    for ent in targets:
        col = names.get(ent)
        if col and not decl.target_column:
            decl.target_column = col
            decl.target_evidence = f"{getattr(ent, 'name', ent)} marcado como alvo"

    # --- propriedades de dados por coluna (rótulo/nome == coluna): tipo pelo rdfs:range
    if columns:
        by_norm = {norm(c): c for c in columns}
        try:
            data_props = list(ontology.data_properties())
        except Exception:
            data_props = []
        role_names = set().union(*props.values())          # propriedades do próprio vocabulário de esquema não descrevem colunas
        for p in data_props:
            if norm(getattr(p, "name", "")) in role_names:
                continue
            labels = [getattr(p, "name", "")] + [str(x) for x in (getattr(p, "label", []) or [])]
            for lab in labels:
                col = by_norm.get(norm(lab))
                if col is None:
                    continue
                decl.declared_columns.append(col)
                kind = range_kind(p)
                if kind and col not in decl.column_types:
                    decl.column_types[col] = kind
                    decl.type_evidence[col] = f"{getattr(p, 'name', p)}.range"
                break

    # --- etiquetas de classe: ordem explícita > índice > ordem de nomeação
    explicit = []
    for ent, val in _role_pairs(ontology, props["class_order"]):
        explicit = _split_list(_text(val))
        decl.class_order_evidence = f"anotação classOrder em {getattr(ent, 'name', ent)}"
        break
    if explicit:
        decl.class_labels, decl.class_order_explicit = explicit, True
        return
    values: Dict[Any, str] = {ent: _text(v) for ent, v in _role_pairs(ontology, props["class_value"])}
    label_inds = []
    try:
        for cls in ontology.classes():
            if norm(getattr(cls, "name", "")) in vocab["classes"]["target_label"]:
                label_inds.extend(list(cls.instances()))
    except Exception:
        pass
    label_inds = [i for i in dict.fromkeys(label_inds) if i in values or getattr(i, "label", None)]
    if not label_inds and values:
        label_inds = list(values)
    if not label_inds:
        return
    index = {ent: _text(v) for ent, v in _role_pairs(ontology, props["class_index"])}
    def val(i):
        return values.get(i) or (str((getattr(i, "label", None) or [""])[0]))
    if label_inds and all(i in index and re.fullmatch(r"-?\d+(\.\d+)?", index[i]) for i in label_inds):
        label_inds.sort(key=lambda i: float(index[i]))
        decl.class_order_explicit, decl.class_order_evidence = True, "índice declarado (classIndex)"
    else:
        label_inds.sort(key=lambda i: _natural_key(getattr(i, "name", str(i))))
        decl.class_order_explicit, decl.class_order_evidence = False, "ordem de nomeação dos indivíduos de etiqueta"
    decl.class_labels = [val(i) for i in label_inds if val(i)]


__all__ = ["OntologySchemaDeclaration", "read_ontology_schema", "load_vocabulary", "range_kind", "norm"]
