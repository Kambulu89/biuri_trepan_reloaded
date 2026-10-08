"""Leitura de anotações OWL independente da ordem de carregamento e de colisões de nome entre ontologias."""
from __future__ import annotations

import re
from typing import Any, List, Sequence


def annotation_values(entity: Any, names: Sequence[str]) -> List[str]:
    """Valores das anotações ``names`` de uma entidade, INDEPENDENTES da ordem de carregamento e de colisões de nome.

    O owlready2 resolve ``entity.statisticRole`` pelo NOME curto dentro do mundo: se duas ontologias do mesmo mundo declaram uma
    anotação com o mesmo nome (IRIs diferentes), o atributo aponta para UMA delas e as entidades da outra parecem não ter valor.
    Por isso lemos cada propriedade de anotação por objeto/IRI (``prop[entity]``) e juntamos todas as que casam com o nome
    normalizado. O acesso por atributo fica só como complemento de compatibilidade."""
    targets = {re.sub(r"[^a-z0-9]+", "", str(name).lower()) for name in names}
    values: List[str] = []

    def _add(raw: Any) -> None:
        if raw is None:
            return
        if not isinstance(raw, (list, tuple, set)):
            raw = [raw]
        for item in raw:
            value = getattr(item, "name", item)
            if value is not None and str(value).strip():
                values.append(str(value).strip())

    world = getattr(getattr(entity, "namespace", None), "world", None)
    if world is not None:
        try:
            properties = list(world.annotation_properties())
        except Exception:
            properties = []
        for prop in properties:
            if re.sub(r"[^a-z0-9]+", "", str(getattr(prop, "name", "")).lower()) not in targets:
                continue
            try:
                _add(list(prop[entity]))
            except Exception:
                continue
    for attr in names:
        try:
            _add(getattr(entity, attr, None))
        except Exception:
            continue
    return list(dict.fromkeys(values))


__all__ = ["annotation_values"]
