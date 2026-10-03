"""Controlos negativos para o enriquecimento semântico.

Um ganho atribuído à ontologia só é credível se uma ontologia com a **mesma
estrutura mas sem o significado** não o reproduzir sistematicamente.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def shuffle_entity_assignments(matches: List[Dict[str, Any]], *, seed: int) -> List[Dict[str, Any]]:
    """Baralha quais entidades OWL correspondem a cada feature (semântica quebrada).

    Mantém exatamente o mesmo conjunto de entidades e o mesmo número de features;
    só o significado feature -> entidade é destruído. Determinístico por ``seed``.
    """
    rows = [dict(m) for m in matches]
    if len(rows) < 2:
        return rows
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    # Derangement sempre que possível: nenhuma feature mantém a sua própria entidade,
    # para o controlo não herdar parte do significado real por pontos fixos.
    for _ in range(200):
        if not np.any(order == np.arange(len(rows))):
            break
        order = rng.permutation(len(rows))
    entity_fields = [k for k in ("entity_name", "entity_type") if k in rows[0]]
    shuffled = [{k: rows[i][k] for k in entity_fields} for i in order]
    for row, new in zip(rows, shuffled):
        row.update(new)
        row["control"] = "shuffled_entity_assignment"
    return rows


def shuffle_accepted_matches(matches: List[Dict[str, Any]], *, seed: int) -> List[Dict[str, Any]]:
    """Baralha só os matches aceites; os rejeitados ficam como estão (mesmo conjunto aceite)."""
    rows = [dict(m) for m in matches]
    idx = [i for i, m in enumerate(rows) if m.get("accepted")]
    shuffled = shuffle_entity_assignments([rows[i] for i in idx], seed=seed)
    for i, new in zip(idx, shuffled):
        rows[i] = new
    return rows
