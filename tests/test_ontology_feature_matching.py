"""
Testes de matching ontológico: Class, DatatypeProperty e ObjectProperty.
"""
import pytest
pytest.importorskip("owlready2", reason="dependência opcional não instalada")
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from owlready2 import get_ontology
from core.trepan_reloaded_extractor import TrepanReloadedExtractor

DATA_DIR = ROOT / "data"
BCW_OWL = DATA_DIR / "Breast_Cancer_Wisconsin.owl"
BCW_TBOX_FIXTURE = ROOT / "tests" / "fixtures" / "breast_cancer_wisconsin_tbox.owl"

BCW_FEATURES = [
    "Clump_Thickness",
    "Cell_Size_Uniformity",
    "Cell_Shape_Uniformity",
    "Marginal_Adhesion",
    "Single_Epi_Cell_Size",
    "Bare_Nuclei",
    "Bland_Chromatin",
    "Normal_Nucleoli",
    "Mitoses",
]

TARGET_EXCLUDED = ("Benign", "Malignant", "DiagnosisClass", "Class")


@pytest.fixture(scope="module")
def bcw_extractor():
    if BCW_TBOX_FIXTURE.exists():
        onto = TrepanReloadedExtractor.load_ontology_file(BCW_TBOX_FIXTURE)
    elif BCW_OWL.exists():
        try:
            onto = TrepanReloadedExtractor.load_ontology_file(BCW_OWL)
        except Exception as exc:
            pytest.skip(f"OWL Wisconsin indisponível para testes: {exc}")
    else:
        pytest.skip("Fixture Breast Cancer Wisconsin não encontrada")
    extractor = TrepanReloadedExtractor(ontology=onto)
    # Como no pipeline: os nomes das classes do alvo vêm dos metadados do dataset (não de uma lista embutida no código).
    extractor.set_target_metadata(class_names=["benign", "malignant"])
    return extractor


@pytest.fixture(scope="module")
def bcw_entities(bcw_extractor):
    pairs, stats = bcw_extractor._get_ontology_matching_entities()
    entities = TrepanReloadedExtractor._unwrap_matching_entities(pairs)
    return entities, stats, pairs


def test_matching_pairs_format(bcw_entities):
    _entities, stats, pairs = bcw_entities
    assert pairs
    for entity, etype in pairs:
        assert hasattr(entity, 'name')
        assert etype in ('class', 'datatype_property', 'object_property')


def test_matching_entities_include_datatype_properties(bcw_entities):
    entities, stats, _pairs = bcw_entities
    assert stats["datatype_properties"] == 9
    assert stats["matching_datatype_properties"] == 9
    assert stats["total"] >= 9


def test_diagnosis_classes_excluded_from_feature_pool(bcw_entities):
    entities, _, _ = bcw_entities
    names = {e.name for e in entities}
    for excluded in TARGET_EXCLUDED:
        assert excluded not in names


def test_all_bcw_features_map_to_datatype_property(bcw_extractor, bcw_entities):
    entities, _, _ = bcw_entities
    mapped = 0
    for feature in BCW_FEATURES:
        result = bcw_extractor._find_matching_concept(feature, entities)
        assert result is not None, f"Sem match para {feature}"
        assert result["score"] >= 0.95, f"Score baixo para {feature}: {result['score']}"
        assert result["entity_type"] == "datatype_property", (
            f"{feature} mapeou para {result['entity_type']}"
        )
        assert result["matched_entity"] == feature
        mapped += 1
    assert mapped == 9


def test_features_do_not_map_to_benign_or_malignant(bcw_extractor, bcw_entities):
    entities, _, _ = bcw_entities
    for feature in BCW_FEATURES:
        result = bcw_extractor._find_matching_concept(feature, entities)
        assert result["matched_entity"] not in ("Benign", "Malignant")


def test_bare_nuclei_not_mapped_to_benign(bcw_extractor, bcw_entities):
    entities, _, _ = bcw_entities
    result = bcw_extractor._find_matching_concept("Bare_Nuclei", entities)
    assert result["matched_entity"] == "Bare_Nuclei"
    assert result["matched_entity"] != "Benign"


def test_target_classes_still_map(bcw_extractor):
    classes = list(bcw_extractor.ontology.classes())
    for label in ("benign", "malignant", "Benign", "Malignant"):
        result = bcw_extractor._find_matching_concept(
            label, classes, allow_target_classes=True
        )
        assert result is not None
        assert result["matched_entity"] in ("Benign", "Malignant")


def test_domain_knowledge_maps_nine_features(bcw_extractor):
    bcw_extractor._extract_domain_knowledge(BCW_FEATURES, ["benign", "malignant"])
    stats = bcw_extractor.domain_knowledge.get("mapping_stats", {})
    assert stats.get("mapped_features") == 9
    assert stats.get("unmapped_features") == 0
    for info in bcw_extractor.feature_semantics.values():
        if info.get("ontology_derived"):
            continue
        assert info.get("ontology_mapped") is True
        assert info.get("entity_type") == "datatype_property"


def test_class_feature_still_works_with_owl_class(tmp_path):
    """Ontologias legadas com features como owl:Class continuam válidas."""
    owl_content = """<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/test#"
     xml:base="http://example.org/test"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
  <owl:Ontology rdf:about="http://example.org/test"/>
  <owl:Class rdf:about="http://example.org/test#PatientRecord">
    <rdfs:label>Patient Record</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/test#Age">
    <rdfs:label>Age</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/test#Outcome">
    <rdfs:subClassOf rdf:resource="http://www.w3.org/2002/07/owl#Thing"/>
  </owl:Class>
</rdf:RDF>
"""
    owl_path = tmp_path / "legacy.owl"
    owl_path.write_text(owl_content, encoding="utf-8")
    onto = get_ontology(str(owl_path)).load()
    extractor = TrepanReloadedExtractor(ontology=onto)
    pairs, _ = extractor._get_ontology_matching_entities()
    entities = TrepanReloadedExtractor._unwrap_matching_entities(pairs)
    result = extractor._find_matching_concept("Age", entities)
    assert result is not None
    assert result["entity_type"] == "class"
    assert result["matched_entity"] == "Age"


def test_object_property_matching(tmp_path):
    owl_content = """<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/obj#"
     xml:base="http://example.org/obj"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
  <owl:Ontology rdf:about="http://example.org/obj"/>
  <owl:Class rdf:about="http://example.org/obj#Sample"/>
  <owl:ObjectProperty rdf:about="http://example.org/obj#linkedToPartner">
    <rdfs:label>linked to partner</rdfs:label>
    <rdfs:domain rdf:resource="http://example.org/obj#Sample"/>
  </owl:ObjectProperty>
</rdf:RDF>
"""
    owl_path = tmp_path / "obj.owl"
    owl_path.write_text(owl_content, encoding="utf-8")
    onto = get_ontology(str(owl_path)).load()
    extractor = TrepanReloadedExtractor(ontology=onto)
    pairs, _ = extractor._get_ontology_matching_entities()
    entities = TrepanReloadedExtractor._unwrap_matching_entities(pairs)
    result = extractor._find_matching_concept("linkedToPartner", entities)
    assert result is not None
    assert result["entity_type"] == "object_property"
