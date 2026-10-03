"""Painel Qt (offscreen): mostra apenas o que o apresentador devolve."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.audit_panel import AuditPanel, table_rows  # noqa: E402
from tests.test_result_presenter import make_result  # noqa: E402
from core.experiment_result import SemanticFeatureRow, Measure  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_empty_panel_shows_no_result_message(app):
    p = AuditPanel()
    assert p.table("summary") is None
    assert not p.stale_banner.isVisible() and not p.cache_banner.isVisible()


def test_panel_renders_all_sections(app):
    p = AuditPanel()
    p.set_result(make_result())
    for name in ("summary", "ontology", "enrichment", "metrics_predictive", "metrics_fidelity", "metrics_complexity",
                 "card_mlp_original", "card_c45", "tree_diagnostics", "experiment"):
        assert p.table(name) is not None, name
    rows = dict(map(tuple, table_rows(p.table("summary"))))
    assert rows["Dataset ativo"].startswith("iris") and rows["Oracle do TREPAN Reloaded"] == "MLP Original"
    assert p.build_label.text() == "V9.2.0 · abc1234"


def test_c45_card_has_no_fidelity_as_oracle(app):
    p = AuditPanel()
    p.set_result(make_result())
    card = dict(map(tuple, table_rows(p.table("card_c45"))))
    assert card["Fidelity"].startswith("Não aplicável")


def test_stale_and_cache_banners_visible_and_cleared(app):
    p = AuditPanel()
    r = make_result()
    r.stale, r.stale_reasons = True, ["stale.owl_changed"]
    r.provenance.cache_used, r.provenance.cache_key = True, "k" * 12
    p.set_result(r)
    assert not p.stale_banner.isHidden() and "DESATUALIZADOS" in p.stale_banner.text()
    assert not p.cache_banner.isHidden() and "RESULTADO EM CACHE" in p.cache_banner.text()
    r2 = make_result()
    p.set_result(r2)
    assert p.stale_banner.isHidden() and p.cache_banner.isHidden()


def test_switching_trees_changes_diagnostics_and_oracle(app):
    p = AuditPanel()
    p.set_result(make_result(rejected=False))
    d1 = table_rows(p.table("tree_diagnostics"))
    p.select_tree("trepan_reloaded")
    d2 = table_rows(p.table("tree_diagnostics"))
    assert d1 != d2
    assert "Árvore pequena" in d2[0][1]            # stump -> diagnóstico, nunca erro
    p.select_tree("trepan_original")
    assert table_rows(p.table("tree_diagnostics")) == d1


def test_selected_only_filter_and_mode_switch(app):
    p = AuditPanel()
    r = make_result(semantic_features=[SemanticFeatureRow(name="a", selected=True, stability=Measure.of(.9)),
                                       SemanticFeatureRow(name="b", selected=False, stability=Measure.of(.1))])
    p.set_result(r)
    assert len(table_rows(p.table("semantic_features"))) == 2
    p.set_selected_only(True)
    assert [row[0] for row in table_rows(p.table("semantic_features"))] == ["a"]
    assert p.tabs.isTabVisible(p.tabs.indexOf(p.tab_pages["log"])) is False
    p.mode_combo.setCurrentIndex(1)
    assert p.mode == "scientific" and p.tabs.isTabVisible(p.tabs.indexOf(p.tab_pages["log"]))
    assert p.result is r  # o modo nunca altera o resultado


def test_metric_cells_have_provenance_tooltip(app):
    p = AuditPanel()
    p.set_result(make_result())
    t = p.table("metrics_fidelity")
    row = [i for i, x in enumerate(table_rows(t)) if x[0] == "TREPAN Original"][0]
    tip = t.item(row, 2).toolTip()
    assert "TREPAN Original" in tip and "MLP Original" in tip and "42" in tip
