"""A OWL enriquecida do Breast Cancer é só TBox e declara família/papel em todas as 30 propriedades."""
from pathlib import Path

import pytest

owlready2 = pytest.importorskip("owlready2")

OWL = Path(__file__).resolve().parents[1] / "data" / "breast_cancer_enriched.owl"
FAMILIES = {"radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points",
            "symmetry", "fractal_dimension"}


@pytest.fixture(scope="module")
def onto():
    return owlready2.get_ontology(str(OWL.resolve())).load()      # igual à produção: caminho local (as_uri() gera "/D:/..." inválido no Windows)


def test_tbox_only_no_instances(onto):
    assert list(onto.individuals()) == []


def test_each_leaf_property_has_family_role_and_three_parents(onto):
    leaves = [p for p in onto.data_properties() if getattr(p, "measurementFamily", None)]
    assert len(leaves) == 30
    roles = {(p.measurementFamily[0], p.statisticRole[0]) for p in leaves}
    assert {(f, r) for f in FAMILIES for r in ("mean", "error", "worst")} == roles
    for p in leaves:
        parents = {x.name for x in p.is_a if hasattr(x, "name")}
        family = p.measurementFamily[0]
        assert any(name.endswith("Measurement") and "Nuclear" not in name for name in parents), (p.name, parents)
        assert any(name in {"hasNuclearSizeMeasurement", "hasNuclearShapeMeasurement", "hasNuclearTextureMeasurement"}
                   for name in parents), (p.name, family)


def test_original_aliases_are_preserved_for_matching(onto):
    labels = {str(l) for p in onto.data_properties() for l in p.label}
    for alias in ("radius_mean", "f_1", "radius1", "fractal_dimension_worst", "f_30", "concave_points2"):
        assert alias in labels
