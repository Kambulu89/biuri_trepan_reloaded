"""Camada de strings: cobertura, fallbacks e terminologia fixa (Partes 32-33)."""
import re
from pathlib import Path

import pytest

from core.experiment_result import Reason
from gui import strings
from gui.strings import FORBIDDEN_VARIANTS, TERMS, catalog_keys, term, tr


@pytest.fixture(autouse=True)
def _reset_language():
    strings.set_language("pt")
    yield
    strings.set_language("pt")


def test_default_language_is_portuguese():
    assert strings.get_language() == "pt"
    assert tr("state.NO_DATA") == "Sem dados"


def test_unknown_key_falls_back_to_key_and_never_raises():
    assert tr("no.such.key") == "no.such.key"
    assert tr("no.such.key", lang="xx") == "no.such.key"


def test_missing_english_key_falls_back_to_portuguese():
    pt_only = sorted(catalog_keys("pt") - catalog_keys("en"))
    if pt_only:
        assert tr(pt_only[0], lang="en") == tr(pt_only[0], lang="pt")


def test_format_errors_do_not_raise():
    assert isinstance(tr("msg.small_sample", n=3), str)
    assert isinstance(tr("msg.small_sample"), str)  # placeholder em falta


def test_unsupported_language_resets_to_default():
    strings.set_language("zz")
    assert strings.get_language() == "pt"


def test_canonical_terms_fixed():
    assert term("mlp_original") == "MLP Original"
    assert term("mlp_ontological") == "MLP Ontológico"
    assert term("trepan_reloaded") == "TREPAN Reloaded"
    assert term("fidelity") == "Fidelity" and term("oracle") == "Oracle"
    assert term("ontology") == "Ontologia"


def test_every_reason_has_na_text():
    for code, category in Reason.CATEGORY.items():
        assert f"reason.{code}" in catalog_keys("pt"), code
        assert f"na.{category}" in catalog_keys("pt"), category


def test_na_meanings_present():
    for cat in ("not_calculated", "not_applicable", "not_available", "not_executed"):
        assert tr(f"na.{cat}") != f"na.{cat}"


def test_catalogs_do_not_use_forbidden_variants():
    for lang in strings.SUPPORTED_LANGUAGES:
        for key in catalog_keys(lang):
            text = tr(key, lang=lang)
            for bad in FORBIDDEN_VARIANTS:
                assert bad not in text, (lang, key, bad)


def test_new_gui_modules_use_no_forbidden_variants():
    root = Path(__file__).resolve().parents[1] / "gui"
    for name in ("messages.py", "experiment_state.py", "result_presenter.py", "export_results.py",
                 "result_builder.py", "audit_panel.py"):
        path = root / name
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        for bad in FORBIDDEN_VARIANTS:
            assert bad not in src, (name, bad)
