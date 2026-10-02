"""Compatibilidade V9.2 para imports históricos do extractor TREPAN.

A implementação antiga baseada numa árvore de destilação foi removida da build
de produção. Qualquer import legado de ``TREPANExtractor`` é redireccionado para
o TREPAN Original histórico de Craven & Shavlik implementado em
:mod:`core.trepan_original`.
"""
from __future__ import annotations

import warnings

from core.trepan_original import TrepanOriginalExtractor


class TREPANExtractor(TrepanOriginalExtractor):
    """Alias de compatibilidade para :class:`TrepanOriginalExtractor`.

    Novos módulos devem importar ``TrepanOriginalExtractor`` directamente.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TREPANExtractor é um alias legado. Use core.trepan_original.TrepanOriginalExtractor.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["TREPANExtractor"]
