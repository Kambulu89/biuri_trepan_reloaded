"""Regressões dos dois erros reportados na aplicação Windows."""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

from core import ontology_reasoner


ROOT = Path(__file__).parents[1]
GUI_SOURCE = ROOT / "gui" / "biuri_app_complete.py"


def _biuri_method(name: str) -> ast.FunctionDef:
    tree = ast.parse(GUI_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "BiuriApp":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"BiuriApp.{name} não encontrado")


def _self_assignments(method: ast.FunctionDef) -> set[str]:
    assignments = set()
    for node in ast.walk(method):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                assignments.add(target.attr)
    return assignments


def test_gui_initializes_both_trepan_audits():
    assignments = _self_assignments(_biuri_method("__init__"))
    assert "trepan_original_audit" in assignments
    assert "trepan_reloaded_audit" in assignments


def test_no_owl_training_assigns_mirrored_reloaded_audit_before_reading_it():
    method = _biuri_method("_execute_training_pipeline")
    source = ast.get_source_segment(GUI_SOURCE.read_text(encoding="utf-8"), method)
    assert "self.trepan_reloaded_audit = self.trepan.mirror_original_tree" in source
    assignment_at = source.index("self.trepan_reloaded_audit =")
    performance_read_at = source.index("best_score=(getattr(self, 'trepan_reloaded_audit'")
    assert assignment_at < performance_read_at


def test_java_home_is_resolved_even_when_java_is_not_on_path(tmp_path):
    java = tmp_path / "jdk-17" / "bin" / "java.exe"
    java.parent.mkdir(parents=True)
    java.write_bytes(b"stub")
    found = ontology_reasoner.find_java_executable(
        environ={"JAVA_HOME": str(java.parents[1])},
        which=lambda _command: None,
        platform_name="nt",
    )
    assert found == str(java.resolve())


def test_missing_java_returns_structured_failure_not_raw_winerror(monkeypatch):
    fake = ModuleType("owlready2")
    fake.__path__ = []
    fake.JAVA_EXE = "java"
    fake.reasoning = SimpleNamespace(JAVA_EXE="java")
    monkeypatch.setitem(sys.modules, "owlready2", fake)
    monkeypatch.setattr(ontology_reasoner, "find_java_executable", lambda **_: None)

    report = ontology_reasoner.run_owl_reasoner(object())

    assert report["executed"] is False
    assert report["consistent"] is None
    assert report["error_type"] == "java_not_found"
    assert "JAVA_HOME" in report["user_message"]


def test_reasoner_configures_internal_owlready_java_binding(monkeypatch, tmp_path):
    java = tmp_path / "java"
    java.write_bytes(b"stub")
    reasoning = SimpleNamespace(JAVA_EXE="java")
    calls = []
    nothing = object()

    fake = ModuleType("owlready2")
    fake.__path__ = []
    fake.reasoning = reasoning
    fake.JAVA_EXE = "java"
    fake.Nothing = nothing
    fake.sync_reasoner = lambda ontologies, **kwargs: calls.append((ontologies, kwargs))
    fake.sync_reasoner_pellet = lambda ontologies, **kwargs: calls.append(
        (ontologies, kwargs)
    )
    monkeypatch.setitem(sys.modules, "owlready2", fake)
    monkeypatch.setattr(
        ontology_reasoner,
        "find_java_executable",
        lambda **_: str(java.resolve()),
    )

    class World:
        @staticmethod
        def inconsistent_classes():
            return []

    ontology = SimpleNamespace(world=World(), classes=lambda: [])
    report = ontology_reasoner.run_owl_reasoner(ontology)

    assert report["executed"] is True
    assert report["consistent"] is True
    assert report["java_executable"] == str(java.resolve())
    assert fake.JAVA_EXE == str(java.resolve())
    assert reasoning.JAVA_EXE == str(java.resolve())
    assert len(calls) == 1
