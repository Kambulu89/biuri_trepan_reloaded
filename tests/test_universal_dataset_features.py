"""
Garante que o pipeline não assume 9 features (Breast Cancer) nem corta colunas por índice fixo.
"""
import pytest
pytest.importorskip("owlready2", reason="dependência opcional não instalada")
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from owlready2 import get_ontology
from core.trepan_reloaded_extractor import TrepanReloadedExtractor
from core.ontology_processor import OntologyProcessor

NUM_FEATURES = 15


@pytest.fixture
def dummy_ontology(tmp_path):
    props = "\n".join(
        f"""  <owl:DatatypeProperty rdf:about="http://example.org/dummy#Attr_{i:02d}">
    <rdfs:domain rdf:resource="http://example.org/dummy#Record"/>
    <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#integer"/>
    <rdfs:label>Attribute {i}</rdfs:label>
  </owl:DatatypeProperty>"""
        for i in range(1, NUM_FEATURES + 1)
    )
    owl_content = f"""<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/dummy#"
     xml:base="http://example.org/dummy"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
  <owl:Ontology rdf:about="http://example.org/dummy"/>
  <owl:Class rdf:about="http://example.org/dummy#Record"/>
  <owl:Class rdf:about="http://example.org/dummy#Outcome"/>
{props}
</rdf:RDF>
"""
    path = tmp_path / "dummy_15.owl"
    path.write_text(owl_content, encoding="utf-8")
    return get_ontology(str(path)).load()


@pytest.fixture
def dummy_feature_names():
    return [f"Attr_{i:02d}" for i in range(1, NUM_FEATURES + 1)]


def test_resolve_schemas_15_features(dummy_ontology, dummy_feature_names):
    rng = np.random.default_rng(99)
    X = rng.integers(0, 10, size=(40, NUM_FEATURES)).astype(float)
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=300, random_state=42)
    mlp.fit(X, rng.integers(0, 2, size=40))

    ext = TrepanReloadedExtractor(ontology=dummy_ontology)
    matrix_names, base_names, oracle_names = ext._resolve_feature_schemas(
        X, dummy_feature_names, mlp
    )
    assert len(base_names) == NUM_FEATURES
    assert len(oracle_names) == NUM_FEATURES
    assert len(matrix_names) == NUM_FEATURES
    assert oracle_names == dummy_feature_names


def test_oracle_matrix_uses_all_15_columns_not_slice_nine(
    dummy_ontology, dummy_feature_names
):
    rng = np.random.default_rng(7)
    n_extra = 4
    base = rng.random((20, NUM_FEATURES))
    extra = rng.random((20, n_extra))
    X_wide = np.hstack([extra, base])  # originais NÃO são as primeiras colunas
    wide_names = [f"noise_{i}" for i in range(n_extra)] + dummy_feature_names

    mlp = MLPClassifier(max_iter=200, random_state=1)
    mlp.fit(base, rng.integers(0, 2, size=20))

    ext = TrepanReloadedExtractor(ontology=dummy_ontology)
    ext._resolve_feature_schemas(X_wide, wide_names, mlp, dummy_feature_names)
    ext._mlp_model_original = mlp
    X_oracle = ext._matrix_for_oracle(X_wide, wide_names, mlp)

    assert X_oracle.shape[1] == NUM_FEATURES
    assert np.allclose(X_oracle, base)
    assert not np.allclose(X_oracle, X_wide[:, :NUM_FEATURES])


def test_enrich_training_matrix_expands_beyond_15(dummy_ontology, dummy_feature_names):
    rng = np.random.default_rng(3)
    X = rng.integers(1, 8, size=(25, NUM_FEATURES)).astype(float)
    ext = TrepanReloadedExtractor(ontology=dummy_ontology)
    ext._extract_domain_knowledge(dummy_feature_names, ["A", "B"])
    proc = OntologyProcessor(dummy_ontology, matcher=ext)
    X_new, names, stats = proc.enrich_training_matrix(
        X, dummy_feature_names, ext.feature_semantics, log=False
    )
    assert stats["original_features"] == NUM_FEATURES
    assert X_new.shape[1] > NUM_FEATURES
    assert len(names) == X_new.shape[1]
    assert len(names) != 9


def test_extract_tree_with_15_features_runs(dummy_ontology, dummy_feature_names):
    rng = np.random.default_rng(11)
    X = rng.integers(0, 10, size=(80, NUM_FEATURES)).astype(float)
    y = (X[:, 0] + X[:, 1] > 10).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    mlp = MLPClassifier(hidden_layer_sizes=(24, 12), max_iter=400, random_state=42)
    mlp.fit(X_train, y_train)

    ext = TrepanReloadedExtractor(ontology=dummy_ontology)
    ext.extract_tree(
        mlp, X_train, y_train, sample_size=400,
        feature_names=dummy_feature_names,
        class_names=["0", "1"],
        original_feature_names=dummy_feature_names,
    )
    assert ext.explainer_tree is not None
    assert len(ext._base_feature_names) == NUM_FEATURES
    assert getattr(mlp, "n_features_in_", NUM_FEATURES) == NUM_FEATURES
    n_tree = ext.explainer_tree.n_features_in_
    assert n_tree > NUM_FEATURES
    X_test_aug, aug_names, _ = OntologyProcessor(
        ext.ontology, matcher=ext
    ).enrich_training_matrix(
        X_test, dummy_feature_names, ext.feature_semantics, log=False
    )
    X_pred = ext.apply_ontology_split_bias(X_test_aug)
    assert X_pred.shape[1] == n_tree
