"""P03: o relatório de produção declara sempre se a diferença Reloaded − Original foi avaliada contra controlos.

Sem o gate de atribuição (por omissão), a alegação é «não atribuir o ganho à ontologia». A guarda só relata: não altera modelos.
"""
import json
import sys
from pathlib import Path

import pytest

owlready2 = pytest.importorskip("owlready2")

from core.production_training import semantic_attribution_guard, train_production_dataframe
from core.semantic_attribution import ATTRIBUTED, AttributionConfig
from test_production_attribution_gate import ENRICH, GATE, _family_onto, _hidden_relation, _uninformative


def test_guard_logic_is_explicit():
    assert semantic_attribution_guard(False, None, "none")["status"] == "NO_ONTOLOGY"
    g = semantic_attribution_guard(True, None, "original_space")
    assert g["status"] == "NOT_EVALUATED_AGAINST_CONTROLS" and g["evaluated_against_controls"] is False
    assert g["claim"] == "do_not_attribute_reloaded_gain_to_ontology" and "--semantic-attribution" in g["how_to_evaluate"]
    ok = semantic_attribution_guard(True, {"status": ATTRIBUTED}, "augmented")
    assert ok["status"] == ATTRIBUTED and ok["claim"] == "ontology_attributable_gain_supported_on_training_controls"
    no = semantic_attribution_guard(True, {"status": "NOT_ATTRIBUTED"}, "neutral")
    assert no["claim"] == "do_not_attribute_reloaded_gain_to_ontology" and no["evaluated_against_controls"] is True


def test_default_production_run_declares_not_evaluated(tmp_path):
    df, onto = _uninformative()
    rep = train_production_dataframe(df, target="target", out_dir=tmp_path, seed=4, ontology=onto,
                                     require_reasoner=False, scientific_tuning=False)
    g = rep["evaluation"]["semantic_attribution_guard"]
    assert g["status"] == "NOT_EVALUATED_AGAINST_CONTROLS" and g["claim"] == "do_not_attribute_reloaded_gain_to_ontology"
    assert rep["manifest"]["semantic_attribution_status"] == "NOT_EVALUATED_AGAINST_CONTROLS"
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["semantic_attribution_status"] == g["status"]


def test_gate_on_uninformative_ontology_is_not_attributed(tmp_path):
    df, onto = _uninformative()
    rep = train_production_dataframe(df, target="target", out_dir=tmp_path, seed=4, ontology=onto,
                                     require_reasoner=False, scientific_tuning=False, semantic_attribution=GATE)
    g = rep["evaluation"]["semantic_attribution_guard"]
    assert g["evaluated_against_controls"] is True and g["status"] != ATTRIBUTED
    assert g["claim"] == "do_not_attribute_reloaded_gain_to_ontology"


def test_gate_on_informative_ontology_is_attributed(tmp_path):
    rep = train_production_dataframe(_hidden_relation(), target="target", out_dir=tmp_path, seed=5, ontology=_family_onto(20, "g1"),
                                     require_reasoner=False, scientific_tuning=False, semantic_enrichment=ENRICH, semantic_attribution=GATE)
    g = rep["evaluation"]["semantic_attribution_guard"]
    assert g["status"] == rep["evaluation"]["semantic_attribution"]["status"]
    if g["status"] == ATTRIBUTED:
        assert g["claim"] == "ontology_attributable_gain_supported_on_training_controls"


def test_without_ontology_the_guard_says_no_ontology(tmp_path):
    df, _ = _uninformative()
    rep = train_production_dataframe(df, target="target", out_dir=tmp_path, seed=4, require_reasoner=False, scientific_tuning=False)
    assert rep["evaluation"]["semantic_attribution_guard"]["status"] == "NO_ONTOLOGY"


def test_cli_flag_passes_the_attribution_config(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import biuri_cli
    seen = {}

    def fake(df, **kw):
        seen.update(kw)
        return {"evaluation": {"semantic_contribution_gate": {"status": "x"}, "ontology_quality": None,
                               "semantic_attribution_guard": {"status": "NOT_EVALUATED_AGAINST_CONTROLS"}}}
    monkeypatch.setattr(biuri_cli, "train_production_dataframe", fake)
    df, _ = _uninformative(n=40)
    data = tmp_path / "d.csv"
    df.to_csv(data, index=False)
    for flag, expected in ((["--semantic-attribution"], AttributionConfig), ([], type(None))):
        seen.clear()
        biuri_cli.main(["train-production", "--data", str(data), "--target", "target", "--out", str(tmp_path / "o")] + flag)
        assert isinstance(seen["semantic_attribution"], expected)
