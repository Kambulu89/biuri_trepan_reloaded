"""Execução explícita de reasoner OWL DL e relatório lógico verificável.

O Owlready2 delega HermiT/Pellet a um processo Java.  Em Windows é comum o
Java estar instalado mas não constar do ``PATH``; por isso este módulo resolve
e configura o executável antes de chamar o reasoner.  A ausência do runtime é
reportada de forma estruturada e nunca é confundida com consistência lógica.
"""
from __future__ import annotations

import glob
import importlib
import os
from pathlib import Path
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence


@dataclass
class ReasonerReport:
    executed: bool
    consistent: bool | None
    engine: str
    unsatisfiable_classes: List[str] = field(default_factory=list)
    inconsistent_classes: List[str] = field(default_factory=list)
    inferred_object_properties: bool = False
    error: str | None = None
    error_type: str | None = None
    java_executable: str | None = None
    user_message: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _class_names(values) -> List[str]:
    names = []
    for value in values or []:
        name = getattr(value, "name", None) or str(value)
        if name not in ("Nothing", "owl.Nothing"):
            names.append(name)
    return sorted(set(names))


def _valid_java_candidate(value: str | os.PathLike[str] | None) -> str | None:
    """Devolve um caminho absoluto apenas quando o candidato é um ficheiro."""
    if not value:
        return None
    try:
        candidate = Path(os.path.expandvars(os.path.expanduser(str(value))))
        if candidate.is_file():
            return str(candidate.resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def find_java_executable(
    *,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    platform_name: str | None = None,
    owlready_java_exe: str | None = None,
    extra_candidates: Sequence[str | os.PathLike[str]] = (),
) -> str | None:
    """Localiza Java de forma portável, incluindo instalações comuns Windows.

    A ordem é deliberada: configuração explícita, ``JAVA_HOME``/``JRE_HOME``,
    configuração atual do Owlready2, ``PATH`` e, por fim, diretórios usuais.
    A função é separada para permitir testes sem depender de Java instalado.
    """
    env = os.environ if environ is None else environ
    platform_id = os.name if platform_name is None else platform_name
    executable_name = "java.exe" if platform_id == "nt" else "java"

    candidates: list[str | os.PathLike[str]] = []
    for key in ("OWLREADY2_JAVA_EXE", "JAVA_EXE"):
        if env.get(key):
            candidates.append(env[key])
    for key in ("JAVA_HOME", "JRE_HOME"):
        if env.get(key):
            candidates.append(Path(env[key]) / "bin" / executable_name)
    candidates.extend(extra_candidates)

    if owlready_java_exe:
        owlready_path = _valid_java_candidate(owlready_java_exe)
        if owlready_path:
            return owlready_path
        try:
            resolved = which(str(owlready_java_exe))
        except (OSError, TypeError, ValueError):
            resolved = None
        if resolved:
            candidates.append(resolved)

    for command in (executable_name, "java"):
        try:
            resolved = which(command)
        except (OSError, TypeError, ValueError):
            resolved = None
        if resolved:
            candidates.append(resolved)

    if platform_id == "nt":
        install_roots = []
        for key in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            if env.get(key):
                install_roots.append(Path(env[key]))
        patterns = (
            ("Java", "*", "bin", "java.exe"),
            ("Eclipse Adoptium", "*", "bin", "java.exe"),
            ("Microsoft", "jdk-*", "bin", "java.exe"),
            ("Amazon Corretto", "*", "bin", "java.exe"),
            ("BellSoft", "*", "bin", "java.exe"),
        )
        for root in install_roots:
            for parts in patterns:
                candidates.extend(
                    sorted(glob.glob(str(root.joinpath(*parts))), reverse=True)
                )

    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normcase(os.path.abspath(str(candidate)))
        if normalized in seen:
            continue
        seen.add(normalized)
        valid = _valid_java_candidate(candidate)
        if valid:
            return valid
    return None


def _configure_owlready_java(owlready2_module) -> str | None:
    """Configura tanto o namespace público como o módulo interno reasoning."""
    reasoning_module = getattr(owlready2_module, "reasoning", None)
    if reasoning_module is None:
        try:
            reasoning_module = importlib.import_module(
                f"{owlready2_module.__name__}.reasoning"
            )
        except (ImportError, AttributeError):
            reasoning_module = None
    current = getattr(reasoning_module, "JAVA_EXE", None)
    if current is None:
        current = getattr(owlready2_module, "JAVA_EXE", None)
    java_executable = find_java_executable(owlready_java_exe=current)
    if java_executable:
        setattr(owlready2_module, "JAVA_EXE", java_executable)
        if reasoning_module is not None:
            setattr(reasoning_module, "JAVA_EXE", java_executable)
    return java_executable


def java_required_message() -> str:
    """Mensagem acionável para a GUI sem expor o traceback do subprocesso."""
    return (
        "Java não foi encontrado. Instale um JRE/JDK de 64 bits (recomendado: "
        "Java 17), defina JAVA_HOME ou OWLREADY2_JAVA_EXE e reinicie o BIURI. "
        "A ontologia não foi ativada porque a consistência OWL DL não pôde ser "
        "demonstrada; o ARFF pode continuar a ser usado sem enriquecimento OWL."
    )


def run_owl_reasoner(
    ontology,
    *,
    engine: str = "hermit",
    infer_property_values: bool = True,
    debug: int = 0,
) -> Dict[str, Any]:
    """Executa HermiT/Pellet fornecido pelo Owlready2.

    Falhas de Java ou do reasoner não são convertidas em sucesso estrutural: o
    relatório fica ``executed=False`` e o quality gate deve rejeitar a OWL.
    """
    if ontology is None:
        return ReasonerReport(
            False,
            None,
            engine,
            error="Ontologia ausente.",
            error_type="ontology_missing",
        ).to_dict()
    requested = str(engine or "hermit").strip().lower()
    if requested not in {"hermit", "pellet"}:
        return ReasonerReport(
            executed=False,
            consistent=None,
            engine=requested,
            error="ValueError: Reasoner deve ser 'hermit' ou 'pellet'.",
            error_type="invalid_engine",
            user_message="Selecione o reasoner HermiT ou Pellet.",
        ).to_dict()
    try:
        import owlready2

        java_executable = _configure_owlready_java(owlready2)
        if not java_executable:
            message = java_required_message()
            return ReasonerReport(
                executed=False,
                consistent=None,
                engine=requested,
                inferred_object_properties=False,
                error="FileNotFoundError: executável Java não localizado.",
                error_type="java_not_found",
                java_executable=None,
                user_message=message,
            ).to_dict()

        from owlready2 import Nothing, sync_reasoner, sync_reasoner_pellet

        if requested == "pellet":
            sync_reasoner_pellet(
                [ontology],
                infer_property_values=infer_property_values,
                infer_data_property_values=False,
                debug=debug,
            )
        elif requested == "hermit":
            sync_reasoner(
                [ontology],
                infer_property_values=infer_property_values,
                debug=debug,
            )
        inconsistent = []
        try:
            inconsistent = list(ontology.world.inconsistent_classes())
        except Exception:
            try:
                inconsistent = list(ontology.inconsistent_classes())
            except Exception:
                inconsistent = []
        unsatisfiable = []
        try:
            for cls in ontology.classes():
                if cls is Nothing:
                    continue
                if Nothing in getattr(cls, "equivalent_to", []):
                    unsatisfiable.append(cls)
                elif Nothing in getattr(cls, "ancestors", lambda: set())():
                    unsatisfiable.append(cls)
        except Exception:
            pass
        inconsistent_names = _class_names(inconsistent)
        unsat_names = _class_names(unsatisfiable)
        consistent = not inconsistent_names and not unsat_names
        return ReasonerReport(
            executed=True,
            consistent=consistent,
            engine=requested,
            unsatisfiable_classes=unsat_names,
            inconsistent_classes=inconsistent_names,
            inferred_object_properties=bool(infer_property_values),
            java_executable=java_executable,
        ).to_dict()
    except Exception as exc:
        is_missing_java = isinstance(exc, FileNotFoundError)
        return ReasonerReport(
            executed=False,
            consistent=None,
            engine=requested,
            inferred_object_properties=False,
            error=f"{type(exc).__name__}: {exc}",
            error_type="java_not_found" if is_missing_java else "reasoner_failed",
            java_executable=locals().get("java_executable"),
            user_message=java_required_message() if is_missing_java else (
                "O reasoner OWL DL falhou. A ontologia não foi ativada; "
                "verifique a sintaxe OWL, a memória disponível e a instalação Java."
            ),
        ).to_dict()
