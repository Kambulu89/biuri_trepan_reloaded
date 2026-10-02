import pytest
pytest.importorskip("owlready2", reason="dependência opcional não instalada")
from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest
from owlready2 import AllDisjoint, DataProperty, Thing, World

from core.arff_schema import OrderedLabelEncoder, parse_arff_class_order
from core.benchmark_ontologies import ensure_builtin_domain_ontologies
from core.ontology_processor import OntologyProcessor, OntologySchemaError
from core.ontology_quality import OntologyQualityGate
from core.ontology_reasoner import run_owl_reasoner


def _iris_ontology(tmp_path):
    path = ensure_builtin_domain_ontologies(tmp_path)["iris"]
    return World().get_ontology(str(Path(path).resolve())).load()


def test_fit_transform_is_training_only_and_schema_strict(tmp_path):
    ontology = _iris_ontology(tmp_path)
    names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    train = pd.DataFrame(
        [[5.0, 3.2, 1.4, 0.2], [6.0, 3.0, 4.5, 1.5], [6.8, 3.1, 5.7, 2.1]],
        columns=names,
    )
    processor = OntologyProcessor(ontology).fit(train, log=False)
    specs_before = list(processor.feature_specs_)
    test = pd.DataFrame([[1000.0, 900.0, 800.0, 700.0]], columns=names)
    transformed = processor.transform(test)
    assert processor.fit_row_count_ == 3
    assert processor.feature_specs_ == specs_before
    assert list(transformed.columns) == processor.output_features_
    with pytest.raises(OntologySchemaError):
        processor.transform(test.drop(columns=[names[0]]))


def test_duplicate_semantic_feature_is_dropped(tmp_path):
    ontology = _iris_ontology(tmp_path)
    names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    train = pd.DataFrame(np.tile(np.arange(1, 5), (6, 1)), columns=names)
    processor = OntologyProcessor(ontology).fit(train, log=False)
    # Todos os agregados são constantes; nenhuma pseudo-feature é preservada.
    assert processor.output_features_ == names
    assert processor.last_engineering_stats["dropped_duplicate_or_constant"] == 2


def test_generic_quality_gate_and_record_abox_are_rejected(tmp_path):
    world = World()
    ontology = world.get_ontology("http://example.org/generic#")
    with ontology:
        feature1 = types.new_class("Feature1", (Thing,))
        feature2 = types.new_class("Feature2", (Thing,))
        record = types.new_class("PatientRecord", (Thing,))
        p1 = types.new_class("rowId", (DataProperty,))
        p2 = types.new_class("valueA", (DataProperty,))
        p3 = types.new_class("valueB", (DataProperty,))
        item = record("Patient_001")
        p1[item] = ["001"]
        p2[item] = [1.0]
        p3[item] = [2.0]
    gate = OntologyQualityGate(min_feature_coverage=0.1, max_generic_ratio=0.30)
    report = gate.evaluate(
        ["feature1", "feature2"], ontology,
        reasoner_report={"executed": True, "consistent": True, "engine": "test"},
    )
    assert not report.accepted
    assert report.metrics["generic_entity_ratio"] > 0.30
    assert report.abox["dataset_record_individuals"] == ["Patient_001"]


def test_named_individual_matching_creates_only_collapsed_category_group():
    world = World()
    ontology = world.get_ontology("http://example.org/categories#")
    with ontology:
        fruit = types.new_class("FruitChoice", (Thing,))
        vehicle = types.new_class("VehicleChoice", (Thing,))
        AllDisjoint([fruit, vehicle])
        for name, cls, label in (
            ("RedApple", fruit, "red apple"),
            ("GreenApple", fruit, "green apple"),
            ("Sedan", vehicle, "sedan"),
            ("SportUtilityVehicle", vehicle, "suv"),
        ):
            individual = cls(name)
            individual.label = [label]
        category = types.new_class("hasCategory", (DataProperty,))
        category.label = ["category"]
    frame = pd.DataFrame({"category": ["red apple", "green apple", "sedan", "suv"]})
    processor = OntologyProcessor(ontology).fit(frame, log=False)
    assert "onto_category_semantic_group" in processor.output_features_
    transformed = processor.transform(frame)
    assert transformed["onto_category_semantic_group"].nunique() == 2
    assert processor.category_matching_["category"]["red apple"]["entity_type"] == "named_individual"


def test_real_reasoner_and_arff_class_order(tmp_path):
    ontology = _iris_ontology(tmp_path / "owl")
    reasoner = run_owl_reasoner(ontology, engine="hermit", debug=0)
    assert reasoner["executed"] is True
    assert reasoner["consistent"] is True

    arff = tmp_path / "ordered.arff"
    arff.write_text(
        "@RELATION ordered\n@ATTRIBUTE x NUMERIC\n"
        "@ATTRIBUTE class {'>50K','<=50K'}\n@DATA\n1,'<=50K'\n2,'>50K'\n",
        encoding="utf-8",
    )
    order = parse_arff_class_order(arff, "class")
    assert order == [">50K", "<=50K"]
    encoder = OrderedLabelEncoder(order)
    encoded = encoder.fit_transform(["<=50K", ">50K"])
    assert encoded.tolist() == [1, 0]
    assert encoder.inverse_transform([0, 1]).tolist() == [">50K", "<=50K"]
