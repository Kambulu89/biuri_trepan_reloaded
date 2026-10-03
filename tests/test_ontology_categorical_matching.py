"""Matching semântico de categorias nominais do ARFF com classes/indivíduos da OWL."""
import numpy as np
import pandas as pd
import pytest

owlready2 = pytest.importorskip("owlready2")

from core.ontology_processor import OntologyProcessor


def _onto(name="cat", with_individuals=False):
    onto = owlready2.World().get_ontology(f"http://test.org/{name}.owl")
    with onto:
        Employment = type("EmploymentType", (owlready2.Thing,), {})
        Private = type("CommercialEmployment", (Employment,), {})
        Public = type("GovernmentalEmployment", (Employment,), {})
        Own = type("IndependentEmployment", (Employment,), {})
        if with_individuals:
            type("PrivateSectorWorker", (Private,), {})  # classe
            Public("FederalGovernmentWorker"); Public("StateGovernmentWorker"); Public("LocalGovernmentWorker")
            Own("SelfEmployedWorker")
        else:
            type("PrivateSectorWorker", (Private,), {})
            type("FederalGovernmentWorker", (Public,), {})
            type("StateGovernmentWorker", (Public,), {})
            type("LocalGovernmentWorker", (Public,), {})
            type("SelfEmployedWorker", (Own,), {})
        type("hasWorkClass", (owlready2.DataProperty,), {"range": [str]})
    return onto


def _frame(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "hasWorkClass": rng.choice(["private", "federal-gov", "state-gov", "local-gov", "self-emp"], n),
        "hasNumber": rng.normal(size=n),
    })


@pytest.mark.parametrize("with_individuals", [False, True])
def test_values_are_matched_to_subclasses_or_individuals_and_grouped(with_individuals):
    onto = _onto("c%d" % with_individuals, with_individuals)
    X = _frame()
    matches = [{"feature": "hasWorkClass", "entity_name": "hasWorkClass", "accepted": True}]
    proc = OntologyProcessor(onto).fit(X, accepted_matches=matches, log=False)
    audit = proc.category_matching_["hasWorkClass"]
    assert audit["private"]["entity"] == "PrivateSectorWorker"       # 'private' -> PrivateSectorWorker
    assert audit["federal-gov"]["entity"] == "FederalGovernmentWorker"
    spec = next(s for s in proc.feature_specs_ if s["kind"] == "categorical_group")
    assert spec["provenance"] == "owl_named_individual_type"
    mapping = spec["mapping"]
    # categorias de governo ficam no mesmo grupo; privado e por conta própria noutros
    assert mapping["federal-gov"] == mapping["state-gov"] == mapping["local-gov"]
    assert len({mapping["private"], mapping["self-emp"], mapping["federal-gov"]}) == 3
    out = proc.transform(X)
    assert out["onto_hasWorkClass_semantic_group"].nunique() == 3


def test_unmatched_value_prevents_a_partial_group_instead_of_guessing():
    onto = _onto("c_un")
    X = _frame(); X.loc[0, "hasWorkClass"] = "never-worked-xyz"
    proc = OntologyProcessor(onto).fit(
        X, accepted_matches=[{"feature": "hasWorkClass", "entity_name": "hasWorkClass", "accepted": True}], log=False)
    assert not [s for s in proc.feature_specs_ if s["kind"] == "categorical_group"]
    audit = proc.category_matching_["hasWorkClass"]["never-worked-xyz"]
    assert audit["entity"] is None and audit["group"] is None


def test_unseen_category_at_transform_fails_loudly_not_silently():
    from core.ontology_processor import OntologySchemaError
    onto = _onto("c_ns")
    X = _frame()
    proc = OntologyProcessor(onto).fit(
        X, accepted_matches=[{"feature": "hasWorkClass", "entity_name": "hasWorkClass", "accepted": True}], log=False)
    Y = X.copy(); Y.loc[0, "hasWorkClass"] = "brand-new-category"
    with pytest.raises(OntologySchemaError):
        proc.transform(Y)


def test_ambiguous_category_is_rejected_not_guessed():
    """'private' casa por igual com duas entidades: é ambíguo e não pode ser adivinhado."""
    onto = owlready2.World().get_ontology("http://test.org/amb_cat.owl")
    with onto:
        Employment = type("EmploymentType", (owlready2.Thing,), {})
        Priv = type("PrivateEmployment", (Employment,), {})
        type("PrivateSectorWorker", (Priv,), {})
        type("hasWorkClass", (owlready2.DataProperty,), {"range": [str]})
    X = pd.DataFrame({"hasWorkClass": ["private", "other"] * 20, "hasNumber": np.arange(40.0)})
    proc = OntologyProcessor(onto).fit(
        X, accepted_matches=[{"feature": "hasWorkClass", "entity_name": "hasWorkClass", "accepted": True}], log=False)
    assert proc.category_matching_["hasWorkClass"]["private"]["entity"] is None
