"""Leitor ARFF (formato denso) com suporte a Unicode.

O leitor do scipy (``scipy.io.arff.loadarff``) guarda os valores nominais como bytes ASCII: qualquer valor nominal com acentos
(``'Cardiopatía'``) falha com ``UnicodeEncodeError``. No Windows, o ficheiro UTF-8 é ainda descodificado com a codificação do sistema
(cp1252), o que produz «ordinal not in range(128)» em posições duplicadas. Este leitor:

- lê o ficheiro em UTF-8 (com ou sem BOM) e só recorre a cp1252 se o ficheiro não for UTF-8 válido;
- devolve o MESMO resultado que ``loadarff`` + descodificação UTF-8 usada até aqui nos ficheiros ASCII: atributos numéricos como
  ``float64`` (ausente = ``NaN``), nominais como ``str`` (ausente = ``'?'``, como o scipy);
- valida cada valor nominal contra a declaração e indica a linha em caso de erro;
- recusa explicitamente ARFF esparso e atributos relacionais (o scipy também não os lê).

Genérico: não conhece nenhum dataset.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


_NUMERIC = ("numeric", "real", "integer")
_ATTR_RE = re.compile(r"^\s*@attribute\s+(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"|(\S+))\s+(.+?)\s*$", re.IGNORECASE)


class ArffMeta:
    """Subconjunto compatível com ``scipy.io.arff.MetaData``: ``names()``, ``types()``, ``meta[nome] -> (tipo, valores)``."""

    def __init__(self, relation: str, attributes: List[Tuple[str, str, Optional[Tuple[str, ...]]]], encoding: str):
        self.name = relation
        self._attributes = attributes
        self.encoding = encoding

    def names(self) -> List[str]:
        return [a[0] for a in self._attributes]

    def types(self) -> List[str]:
        return [a[1] for a in self._attributes]

    def __getitem__(self, key: str):
        for name, kind, values in self._attributes:
            if name == key:
                return kind, values
        raise KeyError(key)

    def __contains__(self, key) -> bool:
        return key in self.names()

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


def _read_text(path: Path) -> Tuple[str, str]:
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252"), "cp1252"


def _split_data_line(line: str, lineno: int) -> List[str]:
    """Divide uma linha de dados por vírgulas, respeitando aspas simples/duplas e escapes ``\\``."""
    out, token, quote, quoted, i = [], [], None, False, 0
    while i < len(line):
        ch = line[i]
        if quote is not None:
            if ch == "\\" and i + 1 < len(line):
                token.append(line[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            else:
                token.append(ch)
        elif ch in ("'", '"') and not quoted and not "".join(token).strip():
            quote, quoted, token = ch, True, []
        elif quoted and ch.isspace():
            pass                                            # espaços entre a aspa de fecho e a vírgula não fazem parte do valor
        elif ch == ",":
            out.append(("".join(token) if quoted else "".join(token).strip(), quoted))
            token, quoted = [], False
        else:
            token.append(ch)
        i += 1
    if quote is not None:
        raise ValueError(f"ARFF, linha {lineno}: aspas não fechadas.")
    out.append(("".join(token) if quoted else "".join(token).strip(), quoted))
    return out


def _attribute_kind(decl: str, name: str, lineno: int) -> Tuple[str, Optional[Tuple[str, ...]]]:
    low = decl.strip().lower()
    if decl.strip().startswith("{"):
        body = decl.strip()
        if not body.endswith("}"):
            raise ValueError(f"ARFF, linha {lineno}: declaração nominal sem '}}' no atributo '{name}'.")
        # o mesmo tokenizador das linhas de dados: aspas simples/duplas e escapes ``\'`` tratados da mesma forma nos dois sítios
        values = [v for v, _q in _split_data_line(body[1:-1], lineno) if v != ""]
        if not values:
            raise ValueError(f"ARFF, linha {lineno}: atributo nominal '{name}' sem valores.")
        return "nominal", tuple(values)
    first = low.split()[0] if low.split() else ""
    if first in _NUMERIC:
        return "numeric", None
    if first == "string":
        return "string", None
    if first == "date":
        return "date", None
    if first == "relational":
        raise ValueError(f"ARFF, linha {lineno}: atributos relacionais não são suportados ('{name}').")
    raise ValueError(f"ARFF, linha {lineno}: tipo desconhecido '{decl}' no atributo '{name}'.")


def read_arff(path) -> Tuple[pd.DataFrame, ArffMeta]:
    """Lê um ARFF denso. Devolve ``(DataFrame, ArffMeta)`` com valores nominais já em ``str``."""
    text, encoding = _read_text(Path(path))
    relation, attributes, data_rows, in_data = "", [], [], False
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if not in_data:
            low = s.lower()
            if low.startswith("@relation"):
                relation = s[9:].strip().strip("'\"")
            elif low.startswith("@attribute"):
                m = _ATTR_RE.match(s)
                if not m:
                    raise ValueError(f"ARFF, linha {lineno}: declaração de atributo inválida: {s[:80]}")
                name = m.group(1) if m.group(1) is not None else (m.group(2) if m.group(2) is not None else m.group(3))
                kind, values = _attribute_kind(m.group(4), name, lineno)
                attributes.append((name, kind, values))
            elif low.startswith("@data"):
                in_data = True
                if not attributes:
                    raise ValueError("ARFF sem atributos declarados antes de @data.")
            elif low.startswith("@end"):
                continue
            else:
                raise ValueError(f"ARFF, linha {lineno}: linha de cabeçalho não reconhecida: {s[:80]}")
            continue
        if s.startswith("{"):
            raise ValueError(f"ARFF, linha {lineno}: formato esparso não é suportado; converta para ARFF denso.")
        fields = _split_data_line(s, lineno)
        if len(fields) != len(attributes):
            raise ValueError(f"ARFF, linha {lineno}: {len(fields)} valores para {len(attributes)} atributos.")
        data_rows.append((lineno, fields))
    if not in_data:
        raise ValueError("ARFF sem secção @data.")

    columns: Dict[str, Any] = {}
    for j, (name, kind, values) in enumerate(attributes):
        raw = [(lineno, f[j]) for lineno, f in data_rows]
        if kind == "numeric":
            col = np.empty(len(raw), dtype=float)
            for i, (lineno, (v, quoted)) in enumerate(raw):
                if v == "?" and not quoted:
                    col[i] = np.nan
                    continue
                try:
                    col[i] = float(v)
                except ValueError:
                    raise ValueError(f"ARFF, linha {lineno}: valor não numérico {v!r} no atributo numérico '{name}'.") from None
            columns[name] = col
        elif kind == "nominal":
            allowed = set(values)
            col = []
            for lineno, (v, quoted) in raw:
                if v == "?" and not quoted:
                    col.append("?")                         # como o scipy: o ausente nominal fica '?'
                    continue
                if v not in allowed:
                    raise ValueError(f"ARFF, linha {lineno}: valor {v!r} não declarado no atributo nominal '{name}' {sorted(allowed)[:6]}…")
                col.append(v)
            columns[name] = np.asarray(col, dtype=object)
        elif kind == "string":
            columns[name] = np.asarray([("?" if (v == "?" and not q) else v) for _ln, (v, q) in raw], dtype=object)
        else:  # date
            vals = [None if (v == "?" and not q) else v for _ln, (v, q) in raw]
            columns[name] = pd.to_datetime(pd.Series(vals, dtype=object), errors="raise").to_numpy()
    df = pd.DataFrame(columns, columns=[a[0] for a in attributes])
    return df, ArffMeta(relation, attributes, encoding)


__all__ = ["ArffMeta", "read_arff"]
