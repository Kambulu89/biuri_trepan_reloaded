from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from core.ontology_processor import OntologyProcessor
from core.semantic_utility_gate import SemanticUtilityConfig, SemanticUtilityGate


class FakeClass:
    def __init__(self, name):
        self.name = name
        self.label = [name]
        self.is_a = []


class FakeDataProperty:
    def __init__(self, name, label, domain, family, role):
        self.name = name
        self.label = [label]
        self.domain = [domain]
        self.is_a = []
        self.measurementFamily = [family]
        self.statisticRole = [role]


class FakeOntology:
    def __init__(self, parent, properties):
        self.parent = parent
        self.properties = list(properties)

    def classes(self):
        return [self.parent]

    def data_properties(self):
        return list(self.properties)

    def object_properties(self):
        return []

    def individuals(self):
        return []


def _processor_and_frame(n=80):
    rng = np.random.default_rng(2026)
    parent = FakeClass("Morphology")
    props = [
        FakeDataProperty("hasMeanRadius", "mean radius", parent, "radius", "mean"),
        FakeDataProperty("hasRadiusError", "radius error", parent, "radius", "error"),
        FakeDataProperty("hasWorstRadius", "worst radius", parent, "radius", "worst"),
        FakeDataProperty("hasMeanArea", "mean area", parent, "area", "mean"),
        FakeDataProperty("hasAreaError", "area error", parent, "area", "error"),
        FakeDataProperty("hasWorstArea", "worst area", parent, "area", "worst"),
    ]
    ontology = FakeOntology(parent, props)
    frame = pd.DataFrame({
        "mean radius": rng.normal(12, 2, n),
        "radius error": rng.normal(0.4, 0.08, n),
        "worst radius": rng.normal(16, 3, n),
        # escala deliberadamente muito maior para verificar normalização
        "mean area": rng.normal(650, 150, n),
        "area error": rng.normal(40, 8, n),
        "worst area": rng.normal(900, 180, n),
    })
    matches = [
        {
            "feature": prop.label[0],
            "entity_name": prop.name,
            "accepted": True,
        }
        for prop in props
    ]
    proc = OntologyProcessor(ontology)
    proc.fit(frame, accepted_matches=matches, log=False)
    return proc, frame


def test_hierarchical_aggregate_is_train_standardized_and_relations_are_owl_driven():
    proc, frame = _processor_and_frame()
    stats = proc.last_engineering_stats
    assert stats["hierarchical_features"] >= 1
    assert stats["relational_features"] == 6
    assert "onto_radius_worst_minus_mean" in stats["relational_names"]
    assert "onto_radius_relative_worst_delta" in stats["relational_names"]
    assert "onto_radius_error_ratio" in stats["relational_names"]

    enriched = proc.transform(frame)
    agg = enriched["onto_Morphology_aggregate"].to_numpy()
    # Cada fonte é z-normalizada no treino; a média do agregado deve ficar ~0.
    assert abs(float(np.mean(agg))) < 1e-10

    diff = enriched["onto_radius_worst_minus_mean"].to_numpy()
    np.testing.assert_allclose(
        diff,
        (frame["worst radius"] - frame["mean radius"]).to_numpy(),
    )


def test_fold_local_transform_recalibrates_statistics_without_mutating_outer_state():
    proc, frame = _processor_and_frame()
    outer_specs = copy.deepcopy(proc.feature_specs_)
    train = frame.iloc[:50].copy()
    validation = frame.iloc[50:].copy()
    train_enriched, val_enriched, meta = proc.transform_fold_pair(train, validation)

    assert meta["scope"] == "inner_fold_training_only"
    assert meta["state_mutated"] is False
    assert list(train_enriched.columns) == proc.output_features_
    assert list(val_enriched.columns) == proc.output_features_
    assert abs(float(train_enriched["onto_Morphology_aggregate"].mean())) < 1e-10
    assert proc.feature_specs_ == outer_specs


def test_semantic_gate_uses_mlp_and_fold_local_semantic_refit():
    proc, frame = _processor_and_frame(n=100)
    enriched = proc.transform(frame)
    X = enriched.to_numpy(dtype=float)
    # Target sintético apenas para testar o protocolo do gate, não o benchmark.
    y = (
        enriched["onto_radius_relative_worst_delta"].to_numpy()
        > np.median(enriched["onto_radius_relative_worst_delta"].to_numpy())
    ).astype(int)

    report = SemanticUtilityGate(SemanticUtilityConfig(
        cv_folds=3,
        min_predictive_gain=-1.0,
        max_noninferiority_loss=1.0,
        complexity_penalty=0.0,
        min_selection_stability=0.0,
        max_single_feature_audits=2,
    )).evaluate(
        X,
        y,
        list(enriched.columns),
        list(frame.columns),
        semantic_processor=proc,
        raw_base_frame=frame,
    )

    assert report["gate_estimator"]["family"] == "MLPClassifier"
    assert report["fold_local_semantic_refit"]["enabled"] is True
    assert report["fold_local_semantic_refit"]["scope"] == "inner_fold_training_only"
    assert report["test_used"] is False


def test_breast_cancer_tbox_declares_measurement_family_and_statistic_role():
    from pathlib import Path
    import xml.etree.ElementTree as ET

    path = Path(__file__).resolve().parents[1] / "data" / "benchmark_ontologies" / "breast_cancer.owl"
    root = ET.parse(path).getroot()
    ns = {
        "owl": "http://www.w3.org/2002/07/owl#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "bc": "http://example.org/trepan-benchmark/breast-cancer#",
    }
    rows = []
    for prop in root.findall("owl:DatatypeProperty", ns):
        label = prop.find("rdfs:label", ns)
        family = prop.find("bc:measurementFamily", ns)
        role = prop.find("bc:statisticRole", ns)
        if label is not None and family is not None and role is not None:
            rows.append((label.text, family.text, role.text))
    assert len(rows) == 30
    assert ("mean radius", "radius", "mean") in rows
    assert ("radius error", "radius", "error") in rows
    assert ("worst radius", "radius", "worst") in rows
