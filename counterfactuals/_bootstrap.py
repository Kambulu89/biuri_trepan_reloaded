"""Bootstrap para executar pipelines de contrafactuais a partir de qualquer diretório."""
from __future__ import annotations

import sys
from pathlib import Path

CF_ROOT = Path(__file__).resolve().parent


def _configure_utf8_stdio() -> None:
    """Evita UnicodeEncodeError no Windows (cp1252) com emojis nos pipelines."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def setup() -> Path:
    """Adiciona o pacote ao ``sys.path`` sem alterar o cwd do processo."""
    _configure_utf8_stdio()
    cf_str = str(CF_ROOT)
    trepa_root = str(CF_ROOT.parent)
    for path in (trepa_root, cf_str):
        if path not in sys.path:
            sys.path.insert(0, path)
    return CF_ROOT
