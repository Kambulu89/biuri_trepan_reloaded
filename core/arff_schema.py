"""Schema ARFF e codificação de classes que preserva a ordem declarada."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


def _split_nominal_values(raw: str) -> List[str]:
    """Separa ``{a, 'b,c', d}`` sem reordenar os valores."""
    text = raw.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return []
    body = text[1:-1]
    values, token, quote = [], [], None
    for char in body:
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            else:
                token.append(char)
        elif char == "," and quote is None:
            values.append("".join(token).strip().strip("'\""))
            token = []
        else:
            token.append(char)
    values.append("".join(token).strip().strip("'\""))
    return [value for value in values if value != ""]


def parse_arff_class_order(path, target_name: str | None = None) -> List[str]:
    """Lê a ordem nominal do alvo diretamente do cabeçalho ARFF."""
    attribute_re = re.compile(
        r"^\s*@attribute\s+(?:(['\"])(.*?)\1|([^\s]+))\s+(.+?)\s*$",
        re.IGNORECASE,
    )
    attributes = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped.lower().startswith("@data"):
            break
        match = attribute_re.match(stripped)
        if match:
            name = match.group(2) or match.group(3)
            attributes.append((name, match.group(4).strip()))
    if not attributes:
        return []
    if target_name is None:
        _, declaration = attributes[-1]
    else:
        matches = [decl for name, decl in attributes if name == target_name]
        if not matches:
            return []
        declaration = matches[-1]
    return _split_nominal_values(declaration)


class OrderedLabelEncoder:
    """Subset compatível com ``LabelEncoder`` sem ordenação lexicográfica."""

    def __init__(self, declared_classes: Sequence | None = None):
        self.declared_classes = list(declared_classes or [])
        self.classes_ = np.asarray([], dtype=object)
        self._index = {}

    @staticmethod
    def _normalise(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            return value.item()
        return value

    def fit(self, y: Iterable):
        observed = [self._normalise(value) for value in y]
        observed_unique = list(dict.fromkeys(observed))
        declared = [self._normalise(value) for value in self.declared_classes]
        missing = [value for value in observed_unique if value not in declared]
        if declared and missing:
            raise ValueError(
                "Classes observadas não declaradas no ARFF: "
                + ", ".join(map(str, missing))
            )
        order = declared or observed_unique
        absent = [value for value in order if value not in observed_unique]
        if absent:
            raise ValueError(
                "Classes declaradas no ARFF sem amostras no dataset: "
                + ", ".join(map(str, absent))
            )
        self.classes_ = np.asarray(order, dtype=object)
        self._index = {value: index for index, value in enumerate(order)}
        return self

    def transform(self, y: Iterable) -> np.ndarray:
        if not self._index:
            raise ValueError("OrderedLabelEncoder ainda não foi ajustado.")
        values = [self._normalise(value) for value in y]
        unknown = [value for value in values if value not in self._index]
        if unknown:
            raise ValueError(
                "Rótulos fora do schema ARFF: " + ", ".join(map(str, dict.fromkeys(unknown)))
            )
        return np.asarray([self._index[value] for value in values], dtype=int)

    def fit_transform(self, y: Iterable) -> np.ndarray:
        values = list(y)
        return self.fit(values).transform(values)

    def inverse_transform(self, y: Iterable[int]) -> np.ndarray:
        indices = np.asarray(list(y), dtype=int)
        if np.any(indices < 0) or np.any(indices >= len(self.classes_)):
            raise ValueError("Índice de classe fora do schema ARFF.")
        return self.classes_[indices]
