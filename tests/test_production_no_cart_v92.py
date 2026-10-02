"""Guards de produção: TREPAN Original/Reloaded não podem regressar a CART."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIRS = ("core", "gui", "counterfactuals", "scripts")


def _python_files():
    for folder in RUNTIME_DIRS:
        yield from (ROOT / folder).rglob("*.py")


def test_runtime_never_imports_decision_tree_classifier():
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "sklearn.tree":
                if any(alias.name == "DecisionTreeClassifier" for alias in node.names):
                    offenders.append(str(path.relative_to(ROOT)))
            if isinstance(node, ast.Import):
                if any(alias.name == "sklearn.tree.DecisionTreeClassifier" for alias in node.names):
                    offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_reloaded_primary_source_has_no_legacy_tree_constructor():
    source = (ROOT / "core" / "trepan_reloaded_extractor.py").read_text(encoding="utf-8")
    assert "_legacy_cart_baseline" not in source
    assert "DecisionTreeClassifier(" not in source
    assert "TrepanReloadedClassifier(" in source


def test_legacy_trepan_import_resolves_to_historical_core():
    from core.trepan_extractor import TREPANExtractor
    from core.trepan_original import TrepanOriginalExtractor
    assert issubclass(TREPANExtractor, TrepanOriginalExtractor)


def test_algorithm_registry_contains_only_supported_tree_families():
    from core.algorithm_identity import ALGORITHMS
    assert "trepan_original" in ALGORITHMS
    assert "trepan_reloaded" in ALGORITHMS
    assert "c45_j48" in ALGORITHMS
    assert all("cart" not in key.lower() for key in ALGORITHMS)


def test_production_loader_rejects_incompatible_tree_family():
    import pytest
    from core.production_inference import _validate_tree_families
    from core.scientific_errors import ArtifactCompatibilityError

    with pytest.raises(ArtifactCompatibilityError):
        _validate_tree_families({
            'trepan_original': object(),
            'trepan_reloaded': object(),
            'c45_native': None,
        })
