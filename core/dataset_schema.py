"""Configuração e validação agnósticas de datasets CSV/ARFF (coluna-alvo, ordem das classes, tipos).

Contrato (sem lógica por dataset; só propriedades gerais dos dados e do ficheiro):

* Coluna-alvo — precedência: escolha manual > configuração guardada do dataset > diretiva ARFF explícita
  (``% biuri:target = nome`` ou ``@relation ... -C <n>``) > última coluna.
* Ordem das classes — precedência: ordem do modelo/encoder já treinado (protegida) > ordem manual > ordem nominal
  declarada no ARFF > ordenação determinística (numérica se todos os rótulos forem numéricos; senão lexicográfica).
  O mapeamento rótulo→índice é guardado e é o MESMO usado por métricas, matrizes, probabilidades, regras e explicações
  (todos derivam do ``OrderedLabelEncoder`` construído a partir desta ordem).
* Tipos — numeric, categorical, ordinal, boolean, datetime. Prioridade: declarado no ARFF > manual > guardado >
  inferência pelos valores. Códigos numéricos de poucas categorias NÃO são classificados como numéricos sem aviso.
* Validação — relatório de incompatibilidades (críticas bloqueiam; avisos informam) entre ficheiro, esquema guardado,
  ARFF de referência, modelo/encoder treinado e ontologia.

Nada aqui altera modelos já treinados: a ordem de um modelo treinado é só consultada e protegida.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
KINDS = ("numeric", "categorical", "ordinal", "boolean", "datetime")
MISSING_TOKENS = frozenset({"", "?", "na", "n/a", "nan", "null", "none"})
BOOLEAN_TOKEN_SETS = (
    frozenset({"0", "1"}), frozenset({"true", "false"}), frozenset({"yes", "no"}),
    frozenset({"y", "n"}), frozenset({"t", "f"}), frozenset({"sim", "nao", "não"}),
)
# false-like primeiro: define o índice 0/1 dos booleanos
_FALSE_TOKENS = frozenset({"0", "false", "no", "n", "f", "nao", "não"})
CODE_MAX_UNIQUE = 10           # inteiros com <= 10 valores distintos: códigos? (ambíguo, sinalizado)
CRITICAL, WARNING, INFO = "critical", "warning", "info"


# ----------------------------------------------------------------------------- estruturas
@dataclass
class ColumnSpec:
    name: str
    kind: str = "numeric"
    source: str = "inferred"            # arff_declared | inferred | manual | saved
    declared: Optional[str] = None      # tipo declarado no ARFF (texto)
    categories: Optional[List[str]] = None
    ordinal_order: Optional[List[str]] = None
    n_missing: int = 0
    n_unique: int = 0
    ambiguous: bool = False
    note: str = ""


@dataclass
class DatasetSchema:
    dataset_id: str
    file_name: str
    source_format: str                  # arff | csv
    target: str
    target_source: str                  # default_last | arff_explicit | manual | saved
    columns: List[ColumnSpec] = field(default_factory=list)   # todas as colunas, incluindo o alvo (ordem do ficheiro)
    class_order: List[str] = field(default_factory=list)
    class_order_source: str = "sorted"  # trained_model | manual | arff_declared | sorted
    class_mapping: Dict[str, int] = field(default_factory=dict)
    drop_missing_rows: bool = False
    separator: Optional[str] = None
    encoding: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    # -- derivados
    def column(self, name: str) -> ColumnSpec:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def feature_names(self) -> List[str]:
        return [c.name for c in self.columns if c.name != self.target]

    @property
    def forced_categorical(self) -> List[str]:
        """Categóricas cujos valores parecem números: o encoder do MLP não pode tratá-las como numéricas."""
        out = []
        for c in self.columns:
            if c.name != self.target and c.kind == "categorical" and c.categories and all(_is_number(v) for v in c.categories):
                out.append(c.name)
        return out

    def is_default_arff(self) -> bool:
        """Esquema sem qualquer intervenção: o comportamento anterior do BIURI para ARFF mantém-se exatamente."""
        return (self.source_format == "arff" and self.target_source == "default_last"
                and self.class_order_source == "arff_declared" and not self.drop_missing_rows
                and all(c.source in ("arff_declared", "inferred") for c in self.columns))

    @property
    def requires_conversion(self) -> bool:
        return self.source_format != "arff" or not self.is_default_arff()

    def signature(self) -> str:
        """Hash estável do que afeta o treino (alvo, ordem das classes, tipos, categorias ordinais, descarte de ausentes)."""
        payload = {
            "target": self.target, "class_order": list(self.class_order), "drop": bool(self.drop_missing_rows),
            "columns": [[c.name, c.kind, list(c.ordinal_order or [])] for c in self.columns],
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def cache_signature(self) -> Optional[str]:
        """Assinatura para a chave da cache de modelos; None (= chave inalterada) quando o esquema é o comportamento anterior."""
        return None if self.is_default_arff() else self.signature()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["signature"] = self.signature()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetSchema":
        cols = [ColumnSpec(**{k: v for k, v in c.items() if k in ColumnSpec.__dataclass_fields__}) for c in data.get("columns", [])]
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "columns"}
        return cls(columns=cols, **fields)

    def summary_lines(self) -> List[str]:
        lines = [f"🎯 Coluna-alvo: {self.target}  ({_SOURCE_LABELS.get(self.target_source, self.target_source)})",
                 f"🏷️ Ordem das classes ({_SOURCE_LABELS.get(self.class_order_source, self.class_order_source)}): "
                 + ", ".join(f"{i}={c}" for i, c in enumerate(self.class_order))]
        lines.append("🧱 Tipos das colunas:")
        for c in self.columns:
            tag = " (alvo)" if c.name == self.target else ""
            extra = []
            if c.n_missing:
                extra.append(f"{c.n_missing} ausentes")
            if c.ambiguous:
                extra.append("a confirmar")
            lines.append(f"   • {c.name}{tag}: {c.kind} [{_SOURCE_LABELS.get(c.source, c.source)}]"
                         + (f" — {', '.join(extra)}" if extra else ""))
        return lines


_SOURCE_LABELS = {
    "default_last": "última coluna por omissão", "arff_explicit": "definida no ARFF", "manual": "escolha manual",
    "saved": "configuração guardada", "trained_model": "ordem do modelo treinado", "arff_declared": "declarada no ARFF",
    "sorted": "ordenação determinística", "arff_declared_type": "declarado no ARFF", "inferred": "inferido",
}


@dataclass
class Issue:
    code: str
    severity: str
    message: str
    column: Optional[str] = None


@dataclass
class ValidationReport:
    issues: List[Issue] = field(default_factory=list)

    def add(self, code: str, severity: str, message: str, column: Optional[str] = None) -> None:
        self.issues.append(Issue(code, severity, message, column))

    @property
    def critical(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == CRITICAL]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.critical

    def format(self) -> str:
        icon = {CRITICAL: "⛔", WARNING: "⚠️", INFO: "ℹ️"}
        return "\n".join(f"{icon.get(i.severity, '-')} [{i.code}] {i.message}" for i in self.issues)


# ----------------------------------------------------------------------------- utilidades
def _is_number(value: Any) -> bool:
    try:
        float(str(value).strip())
        return True
    except (TypeError, ValueError):
        return False


def _norm_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return _norm_text(value).lower() in MISSING_TOKENS


def file_fingerprint(path: str | os.PathLike) -> str:
    """Identidade do ficheiro: SHA-256 do conteúdo canónico (CRLF→LF) — igual em Linux e Windows."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:16]


def sort_class_labels(labels: Iterable[str]) -> List[str]:
    """Ordem determinística: numérica se todos os rótulos forem numéricos; senão lexicográfica."""
    uniq = list(dict.fromkeys(_norm_text(v) for v in labels))
    if uniq and all(_is_number(v) for v in uniq):
        return sorted(uniq, key=lambda v: (float(v), v))
    return sorted(uniq)


# ----------------------------------------------------------------------------- leitura
def read_arff_header(path: str | os.PathLike) -> Dict[str, Any]:
    """Cabeçalho ARFF: relação, atributos (nome, declaração) e diretiva explícita de alvo, se existir."""
    attr_re = re.compile(r"^\s*@attribute\s+(?:(['\"])(.*?)\1|(\S+))\s+(.+?)\s*$", re.IGNORECASE)
    relation, attributes, directive = None, [], None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("%"):
            m = re.match(r"^%\s*biuri\s*:\s*target\s*=\s*(.+?)\s*$", s, re.IGNORECASE)
            if m:
                directive = m.group(1).strip().strip("'\"")
            continue
        low = s.lower()
        if low.startswith("@relation"):
            relation = s[9:].strip().strip("'\"")
        elif low.startswith("@data"):
            break
        else:
            m = attr_re.match(s)
            if m:
                attributes.append((m.group(2) or m.group(3), m.group(4).strip()))
    if directive is None and relation:
        m = re.search(r"-C\s+(-?\d+)", relation)
        if m and attributes:
            n = int(m.group(1))
            if n < 0 and abs(n) <= len(attributes):
                directive = attributes[n][0]            # -C -1 = última
            elif n > 0 and n <= len(attributes):
                directive = attributes[n - 1][0]        # -C 1 = primeira (convenção WEKA/MEKA)
    names = [a for a, _ in attributes]
    if directive is not None and directive not in names:
        directive = None
    return {"relation": relation, "attributes": attributes, "target_directive": directive}


def _nominal_values(declaration: str) -> Optional[List[str]]:
    text = declaration.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    out, token, quote = [], [], None
    for ch in text[1:-1]:
        if ch in ("'", '"'):
            if quote is None:
                quote = ch
            elif quote == ch:
                quote = None
            else:
                token.append(ch)
        elif ch == "," and quote is None:
            out.append("".join(token).strip().strip("'\""))
            token = []
        else:
            token.append(ch)
    out.append("".join(token).strip().strip("'\""))
    return [v for v in out if v != ""]


def read_tabular_text(path: str | os.PathLike, *, encoding: str = "utf-8", separator: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Lê CSV/TSV como TEXTO (nenhuma coerção silenciosa: '007' continua '007'); valida cabeçalho duplicado."""
    p = Path(path)
    enc = "utf-8-sig" if encoding.lower() in ("utf-8", "utf8") else encoding
    raw = p.read_text(encoding=enc)
    if separator is None:
        if p.suffix.lower() == ".tsv":
            separator = "\t"
        else:
            try:
                separator = csv.Sniffer().sniff(raw[:8192], delimiters=",;\t|").delimiter
            except csv.Error:
                separator = ","
    header = next(csv.reader(io.StringIO(raw), delimiter=separator), [])
    header = [h.strip() for h in header]
    dups = sorted({h for h in header if header.count(h) > 1})
    df = pd.read_csv(io.StringIO(raw), sep=separator, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df, {"separator": separator, "encoding": encoding, "duplicate_columns": dups, "header": header}


# ----------------------------------------------------------------------------- tipos
def _parse_datetime(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(series, errors="coerce", utc=True)


def _declared_kind(declaration: str) -> Optional[Tuple[str, Optional[List[str]]]]:
    low = declaration.strip().lower()
    nominal = _nominal_values(declaration)
    if nominal is not None:
        lowered = frozenset(v.lower() for v in nominal)
        if any(lowered == s for s in BOOLEAN_TOKEN_SETS):
            return "boolean", nominal
        return "categorical", nominal
    if low.startswith(("numeric", "real", "integer")):
        return "numeric", None
    if low.startswith("date"):
        return "datetime", None
    if low.startswith("string"):
        return "categorical", None
    return None


def infer_kind(values: Sequence[Any]) -> Tuple[str, bool, str]:
    """Inferência pelos valores → (tipo, ambíguo, nota). Ausentes ignorados."""
    present = [_norm_text(v) for v in values if not is_missing(v)]
    if not present:
        return "categorical", True, "coluna sem valores"
    uniq = list(dict.fromkeys(present))
    lowered = frozenset(u.lower() for u in uniq)
    if len(uniq) <= 2 and any(lowered <= s for s in BOOLEAN_TOKEN_SETS):
        return "boolean", False, ""
    all_num = all(_is_number(u) for u in uniq)
    if all_num:
        integer_valued = all(float(u).is_integer() for u in uniq)
        if integer_valued and len(uniq) <= CODE_MAX_UNIQUE:
            return "categorical", True, "códigos numéricos (≤10 valores inteiros): confirme se são categorias ou uma escala/contagem"
        return "numeric", False, ""
    series = pd.Series(uniq)
    if _parse_datetime(series).notna().mean() >= 0.9 and not all_num:
        return "datetime", False, ""
    return "categorical", False, ""


# ----------------------------------------------------------------------------- construção do esquema
def build_schema(
    df: pd.DataFrame,
    *,
    file_path: str | os.PathLike,
    source_format: str,
    arff_header: Optional[Mapping[str, Any]] = None,
    saved: Optional[DatasetSchema] = None,
    manual_target: Optional[str] = None,
    manual_class_order: Optional[Sequence[str]] = None,
    type_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    drop_missing_rows: Optional[bool] = None,
    trained_class_order: Optional[Sequence[str]] = None,
    separator: Optional[str] = None,
    encoding: Optional[str] = None,
) -> DatasetSchema:
    """Constrói o esquema segundo as precedências do módulo. ``df`` traz os valores originais (texto ou tipos ARFF)."""
    names = [str(c) for c in df.columns]
    if not names:
        raise ValueError("Dataset sem colunas.")
    header = dict(arff_header or {})
    declared = {n: d for n, d in header.get("attributes", [])}

    # --- alvo
    if manual_target is not None:
        target, tsrc = manual_target, "manual"
    elif saved is not None and saved.target in names:
        target, tsrc = saved.target, ("saved" if saved.target_source in ("manual", "saved") else saved.target_source)
    elif header.get("target_directive") in names:
        target, tsrc = header["target_directive"], "arff_explicit"
    else:
        target, tsrc = names[-1], "default_last"
    if target not in names:
        raise ValueError(f"Coluna-alvo inexistente: {target!r}.")

    overrides = {k: dict(v) for k, v in (type_overrides or {}).items()}
    saved_cols = {c.name: c for c in (saved.columns if saved else [])}

    columns: List[ColumnSpec] = []
    for name in names:
        series = df[name]
        present = [v for v in series if not is_missing(v)]
        n_missing = int(len(series) - len(present))
        n_unique = len(set(_norm_text(v) for v in present))
        spec = ColumnSpec(name=name, n_missing=n_missing, n_unique=n_unique, declared=declared.get(name))
        decl = _declared_kind(declared[name]) if name in declared else None
        if name in overrides and overrides[name].get("kind") in KINDS:
            spec.kind, spec.source = overrides[name]["kind"], "manual"
            spec.ordinal_order = list(overrides[name].get("ordinal_order") or []) or None
        elif name in saved_cols and saved_cols[name].source in ("manual", "saved"):
            spec.kind, spec.source = saved_cols[name].kind, "saved"
            spec.ordinal_order = list(saved_cols[name].ordinal_order or []) or None
        elif decl is not None:
            spec.kind, spec.source = decl[0], "arff_declared"
        else:
            kind, amb, note = infer_kind(series)
            spec.kind, spec.source, spec.ambiguous, spec.note = kind, "inferred", amb, note
        if spec.kind in ("categorical", "boolean", "ordinal"):
            spec.categories = sorted(set(_norm_text(v) for v in present))
        if spec.kind == "ordinal" and not spec.ordinal_order:
            spec.ordinal_order = None
        columns.append(spec)

    # --- classes
    target_labels = [_norm_text(v) for v in df[target] if not is_missing(v)]
    observed = list(dict.fromkeys(target_labels))
    declared_classes: List[str] = []
    if target in declared:
        nominal = _nominal_values(declared[target])
        declared_classes = [_norm_text(v) for v in (nominal or [])]
    trained = [_norm_text(v) for v in (trained_class_order or [])]
    if manual_class_order is not None:
        order, osrc = [_norm_text(v) for v in manual_class_order], "manual"
    elif saved is not None and saved.class_order_source == "manual" and set(saved.class_order) == set(observed) and saved.target == target:
        order, osrc = list(saved.class_order), "manual"
    else:
        if declared_classes and set(declared_classes) >= set(observed):
            order, osrc = declared_classes, "arff_declared"
        else:
            order, osrc = sort_class_labels(observed), "sorted"
        # ordem do modelo/encoder já treinado tem prioridade sobre a derivação por omissão (só se for um rearranjo
        # exato dos rótulos observados); quando coincide com a derivação mantém-se a etiqueta original.
        if trained and set(trained) == set(observed) and trained != order:
            order, osrc = list(trained), "trained_model"
    drop = bool(drop_missing_rows) if drop_missing_rows is not None else bool(saved.drop_missing_rows if saved else False)
    return DatasetSchema(
        dataset_id=file_fingerprint(file_path), file_name=Path(file_path).name, source_format=source_format,
        target=target, target_source=tsrc, columns=columns, class_order=order, class_order_source=osrc,
        class_mapping={c: i for i, c in enumerate(order)}, drop_missing_rows=drop, separator=separator, encoding=encoding,
    )


# ----------------------------------------------------------------------------- aplicação ao dataframe
def apply_schema(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Devolve um dataframe pronto para o pipeline do BIURI: alvo como última coluna (texto), features convertidas.

    numeric→float; categorical→texto; boolean→0.0/1.0; datetime→segundos desde 1970 (UTC); ordinal→posto na ordem declarada.
    """
    work = df.copy()
    if schema.drop_missing_rows:
        mask = work.apply(lambda col: col.map(is_missing))
        work = work.loc[~mask.any(axis=1)].reset_index(drop=True)
    out: Dict[str, Any] = {}
    for spec in schema.columns:
        if spec.name == schema.target:
            continue
        col = work[spec.name]
        if spec.kind == "numeric":
            out[spec.name] = col.map(lambda v: np.nan if is_missing(v) else float(_norm_text(v))).astype(float)
        elif spec.kind == "boolean":
            cats = [c for c in (spec.categories or [])]
            ordered = sorted(cats, key=lambda c: (c.lower() not in _FALSE_TOKENS, c))
            idx = {c: float(i) for i, c in enumerate(ordered)}
            out[spec.name] = col.map(lambda v: np.nan if is_missing(v) else idx.get(_norm_text(v), np.nan)).astype(float)
        elif spec.kind == "datetime":
            parsed = _parse_datetime(col.map(lambda v: None if is_missing(v) else _norm_text(v)))
            secs = (parsed - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
            out[spec.name] = secs.astype(float)
        elif spec.kind == "ordinal":
            idx = {c: float(i) for i, c in enumerate(spec.ordinal_order or [])}
            out[spec.name] = col.map(lambda v: np.nan if is_missing(v) else idx.get(_norm_text(v), np.nan)).astype(float)
        else:
            out[spec.name] = col.map(lambda v: np.nan if is_missing(v) else _norm_text(v)).astype(object)
    out[schema.target] = work[schema.target].map(lambda v: np.nan if is_missing(v) else _norm_text(v)).astype(object)
    return pd.DataFrame(out, columns=schema.feature_names + [schema.target])


def move_target_last(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c != target] + [target]
    return df.loc[:, cols] if list(df.columns) != cols else df


# ----------------------------------------------------------------------------- validação
def validate_schema(
    schema: DatasetSchema,
    df: pd.DataFrame,
    *,
    trained: Optional[Mapping[str, Any]] = None,
    saved: Optional[DatasetSchema] = None,
    arff_reference: Optional[Mapping[str, Any]] = None,
    ontology_info: Optional[Mapping[str, Mapping[str, Any]]] = None,
    strict_missing: Optional[bool] = None,
    duplicate_columns: Sequence[str] = (),
) -> ValidationReport:
    """Incompatibilidades CSV/ARFF/encoder/MLP/ontologia. ``strict_missing``: ausentes bloqueiam (omissão: só se o esquema
    não for o comportamento anterior de ARFF, para não alterar fluxos existentes).

    * ``trained``: {'class_order': [...], 'target': str, 'feature_names': [...]} do modelo/encoder já treinado.
    * ``arff_reference``: cabeçalho de um ARFF equivalente (``read_arff_header``).
    * ``ontology_info``: {feature: {'entity': str, 'range': 'numeric'|'categorical'|'boolean'|'datetime'|None}}.
    """
    rep = ValidationReport()
    names = [str(c) for c in df.columns]
    strict = (not schema.is_default_arff()) if strict_missing is None else bool(strict_missing)
    miss_sev = CRITICAL if (strict and not schema.drop_missing_rows) else WARNING

    for dup in duplicate_columns:
        rep.add("DUPLICATE_COLUMN", CRITICAL, f"Coluna duplicada no cabeçalho: {dup!r}.", dup)
    if schema.target not in names:
        rep.add("TARGET_MISSING", CRITICAL, f"Coluna-alvo {schema.target!r} não existe no dataset.", schema.target)
        return rep
    feats = [c for c in names if c != schema.target]
    if not feats:
        rep.add("NO_FEATURES", CRITICAL, "O dataset não tem features além da coluna-alvo.")
    spec_names = [c.name for c in schema.columns]
    if sorted(spec_names) != sorted(names):
        rep.add("SCHEMA_COLUMNS_MISMATCH", CRITICAL, "As colunas do esquema não coincidem com as do dataset (ficheiro alterado?).")

    # --- alvo e classes
    target_series = df[schema.target]
    n_tmiss = int(sum(is_missing(v) for v in target_series))
    legacy = schema.is_default_arff()
    if n_tmiss and not schema.drop_missing_rows:
        rep.add("TARGET_HAS_MISSING", WARNING if legacy else CRITICAL, f"A coluna-alvo tem {n_tmiss} valores ausentes.", schema.target)
    observed = list(dict.fromkeys(_norm_text(v) for v in target_series if not is_missing(v)))
    if len(observed) < 2:
        rep.add("TARGET_SINGLE_CLASS", CRITICAL, f"A coluna-alvo tem {len(observed)} classe(s); são necessárias ≥ 2.", schema.target)
    if set(schema.class_order) != set(observed) or len(schema.class_order) != len(set(schema.class_order)):
        extra = sorted(set(schema.class_order) - set(observed)); lack = sorted(set(observed) - set(schema.class_order))
        only_declared_extra = legacy and not lack and len(schema.class_order) == len(set(schema.class_order))
        rep.add("CLASS_ORDER_INVALID", WARNING if only_declared_extra else CRITICAL,
                f"A ordem das classes tem de ser uma permutação exata dos rótulos observados (a mais: {extra}; em falta: {lack}).",
                schema.target)
    if schema.class_mapping != {c: i for i, c in enumerate(schema.class_order)}:
        rep.add("CLASS_MAPPING_INCONSISTENT", CRITICAL, "O mapeamento rótulo→índice não corresponde à ordem das classes.")

    # --- tipos, ausentes, categorias desconhecidas
    for spec in schema.columns:
        if spec.name == schema.target or spec.name not in names:
            continue
        col = df[spec.name]
        present = [v for v in col if not is_missing(v)]
        miss = len(col) - len(present)
        if miss:
            rep.add("MISSING_VALUES", miss_sev, f"{miss} valor(es) ausente(s) em {spec.name!r} (o pipeline do MLP não imputa).", spec.name)
        texts = [_norm_text(v) for v in present]
        if spec.kind == "numeric":
            bad = [t for t in texts if not _is_number(t)]
            if bad:
                rep.add("TYPE_NUMERIC_INVALID", CRITICAL, f"{spec.name!r} é numérica mas tem valores não numéricos (ex.: {bad[:3]}).", spec.name)
        elif spec.kind == "boolean":
            if len(set(texts)) > 2:
                rep.add("TYPE_BOOLEAN_INVALID", CRITICAL, f"{spec.name!r} é booleana mas tem mais de 2 valores distintos.", spec.name)
        elif spec.kind == "datetime":
            if present and _parse_datetime(pd.Series(texts)).isna().any():
                rep.add("TYPE_DATETIME_INVALID", CRITICAL, f"{spec.name!r} é datetime mas há valores que não são datas.", spec.name)
        elif spec.kind == "ordinal":
            if not spec.ordinal_order:
                rep.add("ORDINAL_ORDER_MISSING", CRITICAL, f"{spec.name!r} é ordinal sem ordem de categorias definida.", spec.name)
            else:
                unknown = sorted(set(texts) - set(spec.ordinal_order))
                if unknown:
                    rep.add("UNKNOWN_CATEGORIES", CRITICAL, f"{spec.name!r}: categorias fora da ordem ordinal: {unknown[:5]}.", spec.name)
        if spec.kind in ("categorical", "boolean") and saved is not None:
            try:
                known = set(saved.column(spec.name).categories or [])
            except KeyError:
                known = set()
            if known:
                unknown = sorted(set(texts) - known)
                if unknown:
                    sev = CRITICAL if trained else WARNING
                    rep.add("UNKNOWN_CATEGORIES", sev, f"{spec.name!r}: categorias desconhecidas face ao esquema guardado: {unknown[:5]}.", spec.name)
        if spec.ambiguous:
            rep.add("AMBIGUOUS_NUMERIC_CODES", WARNING, f"{spec.name!r}: {spec.note or 'tipo a confirmar'}.", spec.name)
        if spec.n_unique <= 1 and present:
            rep.add("CONSTANT_COLUMN", WARNING, f"{spec.name!r} é constante.", spec.name)

    # --- ARFF de referência
    if arff_reference:
        ref_attrs = [n for n, _ in arff_reference.get("attributes", [])]
        if ref_attrs and set(ref_attrs) != set(names):
            only_ref = sorted(set(ref_attrs) - set(names)); only_here = sorted(set(names) - set(ref_attrs))
            rep.add("ARFF_COLUMNS_MISMATCH", CRITICAL, f"Colunas diferentes do ARFF de referência (só no ARFF: {only_ref}; só aqui: {only_here}).")
        ref_decl = {n: _declared_kind(d) for n, d in arff_reference.get("attributes", [])}
        for spec in schema.columns:
            rd = ref_decl.get(spec.name)
            if rd is None or spec.name == schema.target:
                continue
            compatible = {"numeric": {"numeric", "ordinal"}, "categorical": {"categorical", "ordinal", "boolean"},
                          "boolean": {"boolean", "categorical"}, "datetime": {"datetime"}}.get(rd[0], {rd[0]})
            if spec.kind not in compatible:
                rep.add("ARFF_TYPE_MISMATCH", CRITICAL, f"{spec.name!r}: o ARFF declara {rd[0]} mas o esquema usa {spec.kind}.", spec.name)

    # --- modelo / encoder já treinado
    if trained:
        t_target = trained.get("target")
        if t_target is not None and str(t_target) != schema.target:
            rep.add("TRAINED_TARGET_MISMATCH", CRITICAL, f"O modelo treinado usa o alvo {t_target!r}, não {schema.target!r}.", schema.target)
        t_feats = [str(c) for c in (trained.get("feature_names") or [])]
        if t_feats and t_feats != schema.feature_names:
            rep.add("TRAINED_FEATURES_MISMATCH", CRITICAL, "As features do dataset diferem das do modelo treinado (nomes/ordem).")
        t_order = [_norm_text(c) for c in (trained.get("class_order") or [])]
        if t_order and t_order != list(schema.class_order):
            rep.add("CLASS_ORDER_CHANGES_TRAINED_MODEL", CRITICAL,
                    f"A ordem das classes do modelo treinado ({t_order}) difere da configurada ({list(schema.class_order)}); "
                    "alterá-la sem descartar o modelo corromperia métricas e probabilidades.", schema.target)

    # --- ontologia (tipos compatíveis com os mapeamentos)
    forced = set(schema.forced_categorical)
    for feat, info in (ontology_info or {}).items():
        if feat not in schema.feature_names:
            continue
        spec = schema.column(feat)
        rng = info.get("range")
        if feat in forced:
            rep.add("ONTOLOGY_TYPE_CONFLICT", CRITICAL,
                    f"{feat!r} é categórica (códigos numéricos) mas está mapeada a {info.get('entity')!r}: o enriquecimento OWL "
                    "tratá-la-ia como numérica. Defina-a como numérica/ordinal ou retire o mapeamento.", feat)
        elif rng and rng != spec.kind and not (rng == "numeric" and spec.kind in ("ordinal",)):
            rep.add("ONTOLOGY_RANGE_MISMATCH", WARNING,
                    f"{feat!r} ({spec.kind}) está mapeada a {info.get('entity')!r} com domínio de valores {rng}.", feat)
    return rep


# ----------------------------------------------------------------------------- persistência por dataset
def config_dir() -> Path:
    base = os.environ.get("BIURI_DATASET_CONFIG_DIR")
    return Path(base) if base else Path.home() / ".biuri" / "dataset_configs"


def save_schema(schema: DatasetSchema, directory: Optional[str | os.PathLike] = None) -> Path:
    d = Path(directory) if directory else config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{schema.dataset_id}.json"
    path.write_text(json.dumps(schema.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_saved_schema(dataset_id: str, directory: Optional[str | os.PathLike] = None) -> Optional[DatasetSchema]:
    path = (Path(directory) if directory else config_dir()) / f"{dataset_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
            return None
        return DatasetSchema.from_dict(data)
    except (ValueError, TypeError, KeyError):
        return None


__all__ = [
    "ColumnSpec", "DatasetSchema", "Issue", "ValidationReport", "KINDS", "CRITICAL", "WARNING", "INFO",
    "build_schema", "apply_schema", "validate_schema", "read_arff_header", "read_tabular_text", "move_target_last",
    "infer_kind", "sort_class_labels", "file_fingerprint", "save_schema", "load_saved_schema", "config_dir", "is_missing",
]
