"""«cannot import name 'get_ontology' from 'owlready2' (unknown location)»: erro diagnosticado e a aplicação continua sem OWL."""
import importlib
import sys
from pathlib import Path

import pytest

import core.owl_runtime as owl_runtime
from core.production_training import OntologyQualityError
from core.production_training import _load_ontology   # importados ANTES de esconder o pacote real
from core.owl_runtime import OwlRuntimeError, diagnose_owlready2, import_owlready2


def test_a_healthy_owlready2_is_returned_and_diagnosed_ok():
    pytest.importorskip("owlready2")
    assert diagnose_owlready2()["ok"] and import_owlready2().get_ontology


@pytest.fixture
def shadowed_owlready2(tmp_path, monkeypatch):
    """Instalação corrompida: só resta uma pasta ``owlready2`` SEM __init__.py (o pacote real deixou de ser encontrado) => namespace package."""
    (tmp_path / "owlready2").mkdir()
    saved = sys.modules.pop("owlready2", None)
    for name in [m for m in sys.modules if m.startswith("owlready2.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    visible = [p for p in sys.path if not (Path(p or ".") / "owlready2" / "__init__.py").exists()]
    monkeypatch.setattr(sys, "path", [str(tmp_path)] + visible)
    importlib.invalidate_caches()
    sys.path_importer_cache.clear()
    yield tmp_path
    sys.modules.pop("owlready2", None)
    if saved is not None:
        sys.modules["owlready2"] = saved


def test_unknown_location_namespace_package_gets_an_actionable_message(shadowed_owlready2):
    with pytest.raises(ImportError):                       # o comportamento original: ImportError «(unknown location)»
        from owlready2 import get_ontology  # noqa: F401
    sys.modules.pop("owlready2", None)
    diag = diagnose_owlready2()
    assert not diag["ok"] and diag["cause"] == "namespace_package" and diag["file"] is None
    sys.modules.pop("owlready2", None)
    with pytest.raises(OwlRuntimeError) as err:
        import_owlready2()
    msg = err.value.user_message
    assert isinstance(err.value, ImportError) and "pip uninstall -y owlready2" in msg and "pip install --no-cache-dir owlready2==0.47" in msg
    assert "unknown location" in msg and str(shadowed_owlready2) in msg and "SEM ontologia" in msg


def test_production_training_reports_the_runtime_problem_as_an_ontology_error(shadowed_owlready2, tmp_path):
    sys.modules.pop("owlready2", None)
    with pytest.raises(OntologyQualityError, match="pip uninstall"):
        _load_ontology(tmp_path / "x.owl")


def test_gui_continues_without_ontology_when_owlready2_is_broken(shadowed_owlready2, monkeypatch):
    """Sem janelas: o método do GUI devolve False (carregar dados sem OWL) em vez de abortar com «Error inesperado»."""
    src = Path(__file__).resolve().parents[1].joinpath("gui", "biuri_app_complete.py").read_text(encoding="utf-8")
    assert "except OwlRuntimeError as owl_error" in src and "return False" in src.split("except OwlRuntimeError as owl_error", 1)[1][:700]
    assert "from owlready2 import get_ontology, OwlReadyOntologyParsingError" not in src


def test_preflight_flags_a_namespace_package_owlready2(shadowed_owlready2):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import environment_preflight as pre
    sys.modules.pop("owlready2", None)
    probe = pre._probe_imports({"owlready2": "owlready2"})["owlready2"]
    assert probe["ok"] is False and "pip install owlready2" in probe["error"]
