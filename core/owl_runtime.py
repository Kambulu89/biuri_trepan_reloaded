"""Importação diagnosticada do ``owlready2``: transforma «cannot import name 'get_ontology' from 'owlready2' (unknown location)» num erro acionável.

«(unknown location)» significa que ``owlready2`` foi importado como *namespace package*: existe uma pasta ``owlready2`` sem ``__init__.py`` e NENHUM
pacote real foi encontrado depois dela no ``sys.path`` (instalação interrompida/corrompida: p.ex. pasta vazia deixada por um ``pip uninstall``/
``install`` falhado, ou ficheiros bloqueados no Windows). Genérico: não depende de nenhum dataset.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, List

REQUIRED_NAMES = ("get_ontology", "World", "Thing", "sync_reasoner")


class OwlRuntimeError(ImportError):
    """O ``owlready2`` não está utilizável. ``diagnostic`` descreve a causa; ``user_message`` traz os passos de correção."""

    def __init__(self, user_message: str, diagnostic: Dict[str, Any]):
        super().__init__(user_message)
        self.user_message = user_message
        self.diagnostic = diagnostic


def diagnose_owlready2() -> Dict[str, Any]:
    """Estado do ``owlready2`` sem levantar exceções: ``{"ok": bool, "cause": ..., "file": ..., "paths": [...], "missing": [...]}``."""
    sys.modules.pop("owlready2", None) if getattr(sys.modules.get("owlready2"), "__file__", True) is None else None
    try:
        module = importlib.import_module("owlready2")
    except ImportError as exc:
        return {"ok": False, "cause": "not_installed", "file": None, "paths": [], "missing": list(REQUIRED_NAMES), "error": repr(exc)}
    file = getattr(module, "__file__", None)
    paths: List[str] = [str(p) for p in list(getattr(module, "__path__", []) or [])]
    missing = [n for n in REQUIRED_NAMES if not hasattr(module, n)]
    if file is None:
        cause = "namespace_package"            # pasta sem __init__.py: «(unknown location)»
    elif missing:
        cause = "incomplete_or_wrong_module"
    else:
        cause = "ok"
    return {"ok": cause == "ok", "cause": cause, "file": file, "paths": paths, "missing": missing, "version": getattr(module, "VERSION", None)}


def _message(diag: Dict[str, Any]) -> str:
    where = ", ".join(diag.get("paths") or []) or "(sem localização)"
    head = {
        "not_installed": "O pacote owlready2 não está instalado neste Python.",
        "namespace_package": ("O owlready2 foi encontrado só como uma PASTA sem __init__.py ((unknown location)); o pacote real não está instalado/está corrompido. "
                              f"Pastas encontradas: {where}."),
        "incomplete_or_wrong_module": f"Existe um módulo 'owlready2' mas sem {', '.join(diag.get('missing') or [])} (instalação incompleta ou módulo errado em {diag.get('file')}).",
    }.get(diag.get("cause"), "O owlready2 não está utilizável.")
    return (f"{head}\n\nComo corrigir (no mesmo ambiente Python que executa a BIURI):\n"
            "1. Feche a BIURI e outras janelas Python.\n"
            "2. python -m pip uninstall -y owlready2\n"
            "3. Apague a pasta residual 'owlready2' em site-packages (se existir) e qualquer pasta/ficheiro chamado 'owlready2' no diretório do projeto.\n"
            "4. python -m pip install --no-cache-dir owlready2==0.47\n"
            "5. Verifique: python -c \"import owlready2; print(owlready2.__file__)\" — tem de mostrar um caminho .../owlready2/__init__.py.\n"
            "O projeto pode continuar SEM ontologia (sem enriquecimento OWL).")


def import_owlready2():
    """Devolve o módulo ``owlready2`` ou levanta :class:`OwlRuntimeError` (subclasse de ``ImportError``) com a causa e os passos de correção."""
    diag = diagnose_owlready2()
    if not diag["ok"]:
        raise OwlRuntimeError(_message(diag), diag)
    return sys.modules["owlready2"]
