"""Preflight reproducível para o ambiente-alvo BIURI / TREPAN Reloaded V9.2.

Não instala nem altera dependências. Apenas verifica o contrato declarado antes de
executar a suíte GUI/OWL ou o benchmark confirmatório bloqueado.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

RUNTIME_IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "optuna": "optuna",
}
# Visualização opcional: a ausência (pacote ou binário ``dot``) NUNCA impede treino, benchmark, MLP, ontologia ou TREPAN.
OPTIONAL_VISUALIZATION_IMPORTS = {"graphviz": "graphviz", "dtreeviz": "dtreeviz"}
GUI_IMPORTS = {"PyQt6": "PyQt6"}
OWL_IMPORTS = {"owlready2": "owlready2"}
CLEAR_IMPORTS = {"tensorflow": "tensorflow"}


def _probe_java() -> dict:
    executable = shutil.which("java")
    if not executable:
        return {"ok": False, "path": None, "version": None, "reason": "java_not_found"}
    try:
        proc = subprocess.run(
            [executable, "-version"], capture_output=True, text=True, timeout=10,
        )
        text = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {
            "ok": proc.returncode == 0,
            "path": executable,
            "version": text[0] if text else None,
            "returncode": proc.returncode,
        }
    except Exception as exc:  # pragma: no cover - platform-specific diagnostic
        return {"ok": False, "path": executable, "version": None, "reason": repr(exc)}


def _probe_imports(mapping: dict[str, str]) -> dict:
    result = {}
    for package, module in mapping.items():
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", None)
            if module == "owlready2" and (getattr(imported, "__file__", None) is None or not hasattr(imported, "get_ontology")):
                # «(unknown location)»: pasta 'owlready2' sem __init__.py (instalação corrompida ou pasta homónima a sobrepor-se ao pacote)
                result[package] = {"ok": False, "error": "owlready2 importado sem localização/sem get_ontology: reinstale "
                                   "(pip uninstall -y owlready2; apague a pasta residual; pip install owlready2==0.47)"}
                continue
            result[package] = {"ok": True, "version": version}
        except Exception as exc:
            result[package] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def build_preflight(*, require_gui: bool = False, require_owl: bool = False, require_clear: bool = False) -> dict:
    runtime = _probe_imports(RUNTIME_IMPORTS)
    gui = _probe_imports(GUI_IMPORTS) if require_gui else {}
    owl = _probe_imports(OWL_IMPORTS) if require_owl else {}
    clear = _probe_imports(CLEAR_IMPORTS) if require_clear else {}
    python_target_ok = sys.version_info[:2] in {(3, 11), (3, 12)}
    optional_viz = _probe_imports(OPTIONAL_VISUALIZATION_IMPORTS)
    graphviz_dot = shutil.which("dot")
    java = _probe_java()
    checks = {
        "python_3_11": python_target_ok,
        "runtime_dependencies": all(row["ok"] for row in runtime.values()),
        "java": bool(java.get("ok")) if require_owl else True,
    }
    if require_gui:
        checks["gui_dependencies"] = all(row["ok"] for row in gui.values())
    if require_owl:
        checks["owl_dependencies"] = all(row["ok"] for row in owl.values())
    if require_clear:
        checks["clear_dependencies"] = all(row["ok"] for row in clear.values())
    return {
        "contract": "BIURI_TREPAN_RELOADED_V9_2_TARGET_ENVIRONMENT",
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "target_python": "3.11.x or 3.12.x",
        "runtime_imports": runtime,
        "gui_imports": gui,
        "owl_imports": owl,
        "clear_imports": clear,
        "java": java,
        "graphviz_dot": graphviz_dot,
        "optional_visualization": {
            "imports": optional_viz,
            "graphviz_dot_binary": graphviz_dot is not None,
            "available": all(row["ok"] for row in optional_viz.values()) and graphviz_dot is not None,
            "note": "Opcional: só a exportação de árvores para imagem (PNG via Graphviz) depende disto.",
        },
        "checks": checks,
        "ready": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Exigir também PyQt6.")
    parser.add_argument("--owl", action="store_true", help="Exigir também Owlready2 + Java para HermiT/Pellet.")
    parser.add_argument("--clear", action="store_true", help="Exigir também TensorFlow/CLEAR.")
    parser.add_argument("--json", type=Path, help="Guardar relatório JSON.")
    args = parser.parse_args()
    report = build_preflight(require_gui=args.gui, require_owl=args.owl, require_clear=args.clear)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
