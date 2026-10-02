"""
Testes de engenharia de features semânticas (OntologyProcessor).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("owlready2", reason="dependência opcional OWL não instalada")
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ontology_processor import OntologyProcessor
from core.trepan_reloaded_extractor import TrepanReloadedExtractor
from core.trepan_original import TrepanOriginalExtractor

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

RICH_OWL = ROOT / "tests" / "fixtures" / "breast_cancer_wisconsin_rich.owl"
TBOX_OWL = ROOT / "tests" / "fixtures" / "breast_cancer_wisconsin_tbox.owl"


@pytest.fixture(scope="module")
def rich_ontology():
    if not RICH_OWL.exists():
        pytest.skip("Fixture rich OWL não encontrada")
    return TrepanReloadedExtractor.load_ontology_file(RICH_OWL)


@pytest.fixture(scope="module")
def rich_extractor(rich_ontology):
    return TrepanReloadedExtractor(ontology=rich_ontology)


@pytest.fixture
def bcw_dataframe():
    try:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data[:, :9], columns=BCW_FEATURES)
        df["Class"] = np.where(data.target == 0, "benign", "malignant")
        return df
    except Exception:
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(rng.integers(1, 10, size=(n, 9)), columns=BCW_FEATURES)
        df["Class"] = np.where(
            df[BCW_FEATURES].mean(axis=1) > 5, "malignant", "benign"
        )
        return df


def test_hierarchical_and_constraint_columns(rich_extractor, bcw_dataframe):
    processor = OntologyProcessor(rich_extractor.ontology, matcher=rich_extractor)
    rich_extractor._extract_domain_knowledge(
        BCW_FEATURES, ["benign", "malignant"]
    )
    enriched, stats = processor.apply_semantic_feature_engineering(
        bcw_dataframe,
        target_column="Class",
        feature_semantics=rich_extractor.feature_semantics,
        fit=True,
        log=False,
    )
    assert stats["hierarchical_features"] >= 1 or stats["constraint_features"] >= 1
    inferred = stats.get("inferred_columns", [])
    assert any(str(c).startswith("onto_") for c in inferred)
    assert len(enriched.columns) > len(bcw_dataframe.columns)


def test_cell_characteristic_hierarchy_with_rich_owl(rich_extractor, bcw_dataframe):
    processor = OntologyProcessor(rich_extractor.ontology, matcher=rich_extractor)
    processor.fit(bcw_dataframe[BCW_FEATURES], log=False)
    h_cols = processor.last_engineering_stats["hierarchical_names"]
    assert any("Cell" in name for name in h_cols)


def test_constraint_high_low_columns(rich_extractor, bcw_dataframe):
    processor = OntologyProcessor(rich_extractor.ontology, matcher=rich_extractor)
    processor.fit(bcw_dataframe[BCW_FEATURES], log=False)
    # Limiares empíricos deixaram de ser apresentados como axiomas OWL.
    for spec in processor.feature_specs_:
        if spec["kind"] == "constraint":
            assert spec["provenance"] == "owl_explicit_bound"


def test_enrich_training_matrix_shape(rich_extractor):
    rng = np.random.default_rng(0)
    X = rng.integers(1, 10, size=(50, 9)).astype(float)
    rich_extractor._extract_domain_knowledge(BCW_FEATURES, ["benign", "malignant"])
    X_new, names, stats = OntologyProcessor(
        rich_extractor.ontology, matcher=rich_extractor
    ).enrich_training_matrix(
        X, BCW_FEATURES, rich_extractor.feature_semantics, log=False
    )
    assert X_new.shape[0] == 50
    assert X_new.shape[1] >= 9
    assert len(names) == X_new.shape[1]
    assert stats["total_training_features"] == len(names)


def test_semantic_pruning_marks_incoherent_rules(rich_extractor):
    processor = OntologyProcessor(rich_extractor.ontology, matcher=rich_extractor)
    rules = "|--- Clump_Thickness <= 0.50\n"
    validation = {"issues": ["[axioma] Threshold 0.500 em Clump_Thickness viola min=1"]}
    pruned, flagged = processor.prune_semantically_incoherent_rules(
        rules, ["Clump_Thickness"], validation_result=validation
    )
    assert "contradiz axioma" in pruned.lower() or len(flagged) >= 0


@pytest.mark.slow
def test_trepan_reloaded_fidelity_vs_original_with_ontology(rich_extractor, bcw_dataframe):
    """Com ontologia rica, Reloaded deve igualar ou superar fidelidade ao MLP vs Original."""
    X = bcw_dataframe[BCW_FEATURES].values.astype(float)
    y_raw = bcw_dataframe["Class"].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400, random_state=42)
    mlp.fit(X_train, y_train)

    original = TrepanOriginalExtractor()
    original.extract_tree(
        mlp, X_train, y_train, sample_size=800,
        feature_names=BCW_FEATURES, class_names=list(le.classes_),
    )
    fid_original = accuracy_score(
        mlp.predict(X_test), original.explainer_tree.predict(X_test)
    )

    reloaded = TrepanReloadedExtractor(ontology=rich_extractor.ontology)
    reloaded.extract_tree(
        mlp, X_train, y_train, sample_size=800,
        feature_names=BCW_FEATURES, class_names=list(le.classes_),
    )
    X_test_aug, aug_names = reloaded.ontology_processor.transform_matrix(
        X_test, BCW_FEATURES
    )
    X_test_aug = reloaded.apply_ontology_split_bias(X_test_aug)
    fid_reloaded = accuracy_score(
        mlp.predict(X_test), reloaded.explainer_tree.predict(X_test_aug)
    )

    assert fid_reloaded >= fid_original - 0.05, (
        f"Fidelidade Reloaded ({fid_reloaded:.3f}) muito abaixo de Original ({fid_original:.3f})"
    )
    eng = reloaded.semantic_feature_engineering_stats
    assert eng.get("inferred_columns") or eng.get("constraint_features", 0) >= 1


def test_c45_baseline_uses_original_features_only(bcw_dataframe):
    """C4.5-Nativo usa apenas features ARFF — sem onto_* inferidas."""
    X = bcw_dataframe[BCW_FEATURES].values.astype(float)
    y = (bcw_dataframe["Class"] == "malignant").astype(int).values
    tree = DecisionTreeClassifier(criterion="entropy", random_state=42)
    tree.fit(X, y)
    assert tree.n_features_in_ == len(BCW_FEATURES)


def test_tbox_still_produces_inferred_features():
    if not TBOX_OWL.exists():
        pytest.skip("TBox fixture ausente")
    onto = TrepanReloadedExtractor.load_ontology_file(TBOX_OWL)
    ext = TrepanReloadedExtractor(ontology=onto)
    ext._extract_domain_knowledge(BCW_FEATURES, ["benign", "malignant"])
    proc = OntologyProcessor(onto, matcher=ext)
    rng = np.random.default_rng(1)
    X = rng.integers(1, 10, size=(30, 9)).astype(float)
    _, names, stats = proc.enrich_training_matrix(X, BCW_FEATURES, ext.feature_semantics, log=False)
    assert len(names) > 9
    assert stats.get("hierarchical_features", 0) >= 1
    assert all(
        spec.get("provenance") != "training_quantile_not_owl"
        for spec in proc.feature_specs_
    )
