"""Regression tests for explanation and ontology-report generation."""

import ast
from pathlib import Path

import numpy as np


GUI_SOURCE = Path(__file__).parents[1] / "gui" / "biuri_app_complete.py"


def _get_method_node(method_name):
    tree = ast.parse(GUI_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "BiuriApp":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"Method BiuriApp.{method_name} not found")


def test_user_friendly_explanation_handles_all_classes():
    """The per-class section must not fail because of clases/classes mismatch."""
    method = _get_method_node("generate_user_friendly_explanation")
    method.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(GUI_SOURCE), "exec"), namespace)

    explanation = namespace["generate_user_friendly_explanation"](
        object(),
        np.array(["ClassA", "ClassA", "ClassB", "ClassB"]),
        np.array(["ClassA", "ClassB", "ClassB", "ClassB"]),
        0.75,
    )

    assert "Categoría 'ClassA'" in explanation
    assert "Categoría 'ClassB'" in explanation


def test_ontology_report_uses_the_assigned_classes_variable():
    """Keep the ontology report from reintroducing the same naming defect."""
    method = _get_method_node("load_ontology_from_file")
    assigned_names = {
        node.id
        for node in ast.walk(method)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }

    assert "classes" in assigned_names
    assert "clases" not in assigned_names
