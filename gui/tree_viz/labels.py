"""Formatação EXCLUSIVAMENTE visual. Nunca alteram a árvore: devolvem strings novas."""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Sequence

# Abreviações genéricas de vocabulário estatístico/técnico (sem referência a datasets).
_ABBREV = {
    "worst": "w", "mean": "mn", "average": "avg", "standard": "std", "deviation": "dev",
    "error": "err", "points": "pts", "point": "pt", "dimension": "dim", "number": "num",
    "minimum": "min", "maximum": "max", "relative": "rel", "difference": "diff",
    "perimeter": "perim", "concavity": "concav", "concave": "concav", "compactness": "compact",
    "smoothness": "smooth", "symmetry": "sym", "texture": "tex", "measurement": "meas",
    "fractal": "fract", "length": "len", "width": "wid", "height": "hgt", "total": "tot",
    "ratio": "rat", "count": "cnt", "value": "val", "feature": "ft", "attribute": "attr",
    "temperature": "temp", "pressure": "press", "frequency": "freq", "percentage": "pct",
}
_TOKEN_SPLIT = re.compile(r"[_\s\-\.]+|(?<=[a-z0-9])(?=[A-Z])")


def _tokens(name: str) -> List[str]:
    return [t for t in _TOKEN_SPLIT.split(name) if t]


def _ellipsis_middle(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    keep = max(2, max_len - 1)
    head = (keep + 1) // 2
    tail = keep - head
    return text[:head] + "…" + (text[-tail:] if tail else "")


def is_ontology_feature(name: str) -> bool:
    return str(name).lower().startswith(("onto_", "onto:"))


def make_display_feature_name(name: str, max_len: int = 14) -> str:
    """Nome compacto para o desenho. O nome completo permanece em tooltip/painel/export."""
    full = str(name)
    onto = is_ontology_feature(full)
    core = full[5:] if onto else full
    prefix = "onto:" if onto else ""
    budget = max(4, max_len - len(prefix))
    if len(core) <= budget:
        return prefix + core
    toks = _tokens(core)
    # 1.º: abreviar tokens conhecidos, preservando a ordem
    short = [(_ABBREV.get(t.lower(), t)) for t in toks]
    cand = "_".join(short)
    if len(cand) <= budget:
        return prefix + cand
    # 2.º: encurtar tokens longos para 4 letras, mantendo sufixos curtos (ex.: w, mn)
    short = [t if len(t) <= 4 else t[:4] for t in short]
    cand = "_".join(short)
    if len(cand) <= budget:
        return prefix + cand
    # 3.º: reticências no meio (nunca confunde o início/fim distintivos)
    return prefix + _ellipsis_middle(cand, budget)


def format_threshold(value: Optional[float], significant: int = 4) -> str:
    """Limiar para display (ex.: 16.794835219 -> 16.79). O valor real nunca é alterado."""
    if value is None:
        return "?"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return str(v)
    if v == 0:
        return "0"
    if abs(v) >= 1e6 or abs(v) < 1e-3:
        return f"{v:.{max(1, significant - 1)}e}".replace("e+0", "e").replace("e-0", "e-")
    text = f"{v:.{significant}g}"
    return text


def full_threshold(value: Optional[float]) -> str:
    return "?" if value is None else repr(float(value))


def display_class_name(raw, class_names: Optional[Sequence] = None) -> str:
    """Nome de classe legível; um rótulo inteiro indexa ``class_names`` quando possível."""
    names = list(class_names or [])
    if isinstance(raw, bool):
        return str(raw)
    try:
        as_float = float(raw)
        idx = int(as_float)
        if as_float == idx and 0 <= idx < len(names):
            return str(names[idx])
    except (TypeError, ValueError):
        pass
    return str(raw)


def short_class_name(name: str, max_len: int = 8) -> str:
    return _ellipsis_middle(str(name), max_len)


def wrap_text(text: str, width: int = 60) -> str:
    """Quebra de linha SÓ para tooltips e painéis (nunca para dentro dos círculos)."""
    out, line = [], ""
    for word in str(text).split(" "):
        if line and len(line) + 1 + len(word) > width:
            out.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return "\n".join(out)


# Paleta categórica distinguível (daltónicos: Okabe-Ito adaptada). Atribuída por POSIÇÃO
# estável do rótulo na lista de classes do dataset, nunca por hash (hash(str) varia por processo).
CLASS_PALETTE = [
    ("#56B4E9", "#0B2A3C"), ("#E69F00", "#3A2600"), ("#009E73", "#00281D"), ("#CC79A7", "#3A1229"),
    ("#F0E442", "#3A3600"), ("#D55E00", "#FFFFFF"), ("#0072B2", "#FFFFFF"), ("#999999", "#101010"),
]


def class_color_map(labels: Iterable[str]) -> dict:
    """label -> (fill, text, marker). O marcador (letra/forma) garante que a cor nunca é a única pista."""
    out = {}
    for i, label in enumerate(dict.fromkeys(str(l) for l in labels)):
        fill, text = CLASS_PALETTE[i % len(CLASS_PALETTE)]
        out[label] = (fill, text)
    return out
