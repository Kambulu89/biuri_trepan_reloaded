"""Identificação da build em execução (versão, commit, pipeline semântico).

Serve para a interface e para os manifestos de experiência responderem
"que código produziu este resultado?". Nunca falha: sem git ou sem metadados
devolve ``unknown`` em vez de lançar exceção.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from core.semantic_version import SEMANTIC_PIPELINE_VERSION

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True, timeout=3)
        return out.strip()
    except Exception:
        return None


def _project_version() -> str:
    try:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if match:
            return match.group(1)
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("biuri-trepan-reloaded")
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def get_build_info() -> Dict[str, Any]:
    """Versão do projeto, commit (curto), estado do working tree e versões do pipeline."""
    commit = os.environ.get("BIURI_COMMIT") or _git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = None
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is not None:
        dirty = bool(status)
    return {
        "version": _project_version(),
        "commit": commit,
        "dirty": dirty,
        "semantic_pipeline_version": SEMANTIC_PIPELINE_VERSION,
        "python": platform.python_version(),
        "platform": sys.platform,
    }


def build_label() -> str:
    """Linha curta para barra de estado: ``V9.2.0 · abc1234 (+local)``."""
    info = get_build_info()
    suffix = " (+alterações locais)" if info.get("dirty") else ""
    return f"V{info['version']} · {info['commit']}{suffix}"
