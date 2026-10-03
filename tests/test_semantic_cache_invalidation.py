"""A cache nunca reutiliza um MLP treinado com outra versão/conteúdo/config semântica."""
import os

import numpy as np
import pytest

from core.model_cache import (
    build_cache_key, compute_dataset_hash, compute_matching_hash, compute_ontology_hash,
    compute_semantic_config_hash,
)
from core.semantic_enrichment import EnrichmentConfig
from core.training_config import get_training_preset

PRESET = get_training_preset("balanced")


def _key(**over):
    args = dict(model_role="mlp_ontological", dataset_hash="d" * 16, ontology_hash="o" * 12,
                preprocessing_hash="p", preset=PRESET, feature_space="enriched",
                n_features=8, feature_names=["a", "b"], selected_onto_features=["onto_x"])
    args.update(over)
    return build_cache_key(**args)


def test_key_is_stable_for_identical_inputs():
    assert _key() == _key() and len(_key()) == 64


@pytest.mark.parametrize("change", [
    {"dataset_hash": "e" * 16},
    {"ontology_hash": "x" * 12},
    {"preprocessing_hash": "q"},
    {"feature_names": ["a", "c"]},
    {"selected_onto_features": ["onto_y"]},
    {"semantic_pipeline_version": "0.0.1"},
    {"semantic_config_hash": "cfg1"},
    {"matching_hash": "m1"},
    {"model_role": "mlp_original"},
])
def test_every_relevant_input_changes_the_key(change):
    assert _key(**change) != _key()


def test_dataset_hash_changes_with_a_single_cell():
    X = np.random.default_rng(0).normal(size=(30, 3)); y = np.arange(30) % 2
    X2 = X.copy(); X2[7, 1] += 1e-9
    assert compute_dataset_hash(X, y, "d") != compute_dataset_hash(X2, y, "d")


def test_ontology_hash_is_content_based(tmp_path):
    a = tmp_path / "a.owl"; b = tmp_path / "copy" / "b.owl"; b.parent.mkdir()
    a.write_text("<owl>v1</owl>"); b.write_text("<owl>v1</owl>")
    os.utime(b, (1, 1))  # outro mtime e outro caminho, mesmo conteúdo
    assert compute_ontology_hash(str(a)) == compute_ontology_hash(str(b))
    stat = a.stat()
    a.write_text("<owl>v2</owl>")  # mesmo tamanho, conteúdo diferente
    os.utime(a, (stat.st_atime, stat.st_mtime))  # e mtime restaurado
    assert compute_ontology_hash(str(a)) != compute_ontology_hash(str(b))


def test_missing_ontology_has_explicit_sentinel(tmp_path):
    assert compute_ontology_hash(None) == "no_ontology"
    assert compute_ontology_hash(str(tmp_path / "nope.owl")) == "no_ontology"


owlready2 = pytest.importorskip("owlready2")


def _onto(extra_prop=False, extra_individual=False):
    onto = owlready2.World().get_ontology("http://test.org/cache.owl")
    with onto:
        A = type("A", (owlready2.Thing,), {})
        type("B", (A,), {})
        type("hasX", (owlready2.DataProperty,), {"range": [float]})
        if extra_prop:
            type("hasY", (owlready2.DataProperty,), {"range": [float]})
        if extra_individual:
            A("someone")
    return onto


def test_object_hash_detects_tbox_and_abox_changes():
    base = compute_ontology_hash(None, _onto())
    assert base == compute_ontology_hash(None, _onto())
    assert base != compute_ontology_hash(None, _onto(extra_prop=True))        # mesma lista de classes
    assert base != compute_ontology_hash(None, _onto(extra_individual=True))  # só a ABox muda


def test_matching_hash_tracks_entity_and_score_changes():
    m = [{"feature": "f", "entity_name": "E", "score": 0.9, "accepted": True}]
    assert compute_matching_hash(m) == compute_matching_hash(list(m))
    assert compute_matching_hash(m) != compute_matching_hash([{**m[0], "entity_name": "E2"}])
    assert compute_matching_hash(m) != compute_matching_hash([{**m[0], "score": 0.5}])
    assert compute_matching_hash(m) != compute_matching_hash([{**m[0], "accepted": False}])


def test_semantic_config_hash_follows_dataclass_fields():
    base = compute_semantic_config_hash(EnrichmentConfig())
    assert base == compute_semantic_config_hash(EnrichmentConfig())
    assert base != compute_semantic_config_hash(EnrichmentConfig(min_selection_frequency=0.9))
    assert base != compute_semantic_config_hash(EnrichmentConfig(), {"drop_linear_redundant": True})
