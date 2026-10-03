"""Opção de controlo negativo do pipeline de produção e utilitários de baralhamento."""
import numpy as np
import pandas as pd
import pytest

owlready2 = pytest.importorskip("owlready2")

from core.production_training import train_production_dataframe
from core.semantic_controls import shuffle_accepted_matches, shuffle_entity_assignments


def _matches(n=8, rejected=()):
    return [{"feature": f"f{i}", "entity_name": f"E{i}", "entity_type": "datatype_property",
             "score": 1.0, "accepted": i not in rejected} for i in range(n)]


def test_shuffle_is_a_derangement_when_possible():
    for seed in range(40):
        out = shuffle_entity_assignments(_matches(6), seed=seed)
        assert all(o["entity_name"] != f"E{i}" for i, o in enumerate(out)), seed
        assert sorted(o["entity_name"] for o in out) == [f"E{i}" for i in range(6)]


def test_shuffle_accepted_leaves_rejected_untouched():
    base = _matches(8, rejected=(2, 5))
    out = shuffle_accepted_matches(base, seed=3)
    for i in (2, 5):
        assert out[i] == base[i] and out[i]["accepted"] is False
    acc = [i for i, m in enumerate(base) if m["accepted"]]
    assert sorted(out[i]["entity_name"] for i in acc) == sorted(base[i]["entity_name"] for i in acc)
    assert all(out[i]["entity_name"] != base[i]["entity_name"] for i in acc)
    assert [m["feature"] for m in out] == [m["feature"] for m in base]


def test_shuffle_is_deterministic_per_seed():
    assert shuffle_accepted_matches(_matches(), seed=1) == shuffle_accepted_matches(_matches(), seed=1)
    assert shuffle_accepted_matches(_matches(), seed=1) != shuffle_accepted_matches(_matches(), seed=2)


def _onto():
    onto = owlready2.World().get_ontology("http://test.org/ctl_opt.owl")
    with onto:
        A = type("GroupA", (owlready2.Thing,), {}); B = type("GroupB", (owlready2.Thing,), {})
        for i in range(3):
            type(f"hasAlpha{i}", (owlready2.DataProperty,), {"range": [float], "domain": [A]})
            type(f"hasBeta{i}", (owlready2.DataProperty,), {"range": [float], "domain": [B]})
    return onto


def _df(n=100):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f"hasAlpha{i}": rng.normal(size=n) for i in range(3)} | {f"hasBeta{i}": rng.normal(size=n) for i in range(3)})
    df["target"] = np.where(df["hasAlpha0"] + df["hasBeta1"] > 0, "y", "n")
    return df


def test_production_shuffled_control_changes_semantics_but_not_the_original_arm(tmp_path):
    kw = dict(target="target", seed=3, require_reasoner=False, scientific_tuning=False)
    real = train_production_dataframe(_df(), out_dir=tmp_path / "r", ontology=_onto(), **kw)
    ctrl = train_production_dataframe(_df(), out_dir=tmp_path / "c", ontology=_onto(),
                                      semantic_control="shuffled_semantics", semantic_control_seed=7, **kw)
    assert ctrl["evaluation"]["ontology_quality"]["semantic_control"] == "shuffled_semantics"
    assert "semantic_control" not in real["evaluation"]["ontology_quality"]
    # mesma divisão, professor e semente: o braço Original é idêntico
    assert real["evaluation"]["models"]["original"] == ctrl["evaluation"]["models"]["original"]
    real_groups = real["semantic_graph"]["feature_groups"]; ctrl_groups = ctrl["semantic_graph"]["feature_groups"]
    assert real_groups != ctrl_groups   # o grafo semântico foi realmente baralhado


def test_production_rejects_unknown_control(tmp_path):
    with pytest.raises(ValueError):
        train_production_dataframe(_df(), target="target", out_dir=tmp_path, seed=3, ontology=_onto(),
                                   require_reasoner=False, scientific_tuning=False, semantic_control="nonsense")


def test_trepan_overrides_are_applied_and_recorded(tmp_path):
    rep = train_production_dataframe(_df(), target="target", out_dir=tmp_path, seed=3, ontology=_onto(),
                                     require_reasoner=False, scientific_tuning=False,
                                     trepan_overrides={"alpha": 0.0, "beta": 0.0, "gain_criterion": "information_gain"})
    cfg = rep["evaluation"]["experiment_audit"]["config"]
    assert cfg["alpha"] == 0.0 and cfg["beta"] == 0.0 and cfg["gain_criterion"] == "information_gain"
