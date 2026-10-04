"""Agnosticismo ao dataset: o mesmo algoritmo e a mesma política de seleção, sem conhecimento da identidade do dataset.

Datasets conhecidos (Breast Cancer, Iris, Adult, ...) só servem como casos de validação em ``validation/`` e ``scripts/``.
Aqui verifica-se, de forma automática, que o núcleo científico não contém regras, nomes, números de features/classes ou
caminhos de execução específicos de nenhum dataset, e que datasets arbitrários funcionam sem alterar o código-fonte.
"""
import ast
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.scientific_benchmark_service as svc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

SCIENTIFIC_DIRS = ("core", "gui", "counterfactuals")

# Nomes de datasets/classes/valores conhecidos (minúsculas, sem `iris` OWL: ver _iris_hits).
DATASET_WORDS = re.compile(
    r"\b(wine|wdbc|sonar|german|hepatitis|adult|breast[_ ]?cancer|digits|diabetes|titanic|mnist|heart[_ ]disease|"
    r"mushroom|credit|setosa|versicolor|virginica|malignant|benign|wisconsin|pima|ionosphere|vehicle|glass)\b", re.I)
IRIS_DATASET = re.compile(r"(?<![A-Za-z_])(iris|Iris)(?![A-Za-z_])")   # exclui IRIS/IRIs da API do OWL
IDENTITY_BRANCH = re.compile(
    r"(dataset[_.]?name|data_?set[_.]?id|file_name|arff_meta\.get\(['\"]file_name['\"]\)|relation_name)\s*(==|!=|in\s*[\(\[\{])")


def _py_files(*dirs):
    for d in dirs:
        yield from (ROOT / d).rglob("*.py")


def _code_without_docs(path: Path) -> str:
    """Código sem docstrings/comentários: o que pode alterar o comportamento."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body[0].value.value = ""
    return ast.unparse(tree)


# --------------------------------------------------------------------------------- 1. sem nomes de datasets
def test_no_known_dataset_names_inside_the_scientific_modules():
    bad = []
    for path in _py_files(*SCIENTIFIC_DIRS):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if DATASET_WORDS.search(text) or IRIS_DATASET.search(text):
            bad.append(path.relative_to(ROOT).as_posix())
    assert not bad, f"nomes de datasets no núcleo científico: {bad}"


def test_no_branching_on_dataset_identity_or_known_shapes():
    bad = []
    shape_rule = re.compile(r"(n_features|n_classes|shape\[1\]|len\((feature_names|class_names|classes)\))\s*(==|!=)\s*\d{2,}")
    for path in _py_files(*SCIENTIFIC_DIRS):
        code = _code_without_docs(path)
        if IDENTITY_BRANCH.search(code) or shape_rule.search(code):
            bad.append(path.relative_to(ROOT).as_posix())
    assert not bad, bad


def test_dataset_catalogs_live_outside_the_scientific_core_and_are_never_imported_by_it():
    assert (ROOT / "validation" / "benchmark_ontologies.py").exists() and (ROOT / "validation" / "ablation_study.py").exists()
    assert not (ROOT / "core" / "benchmark_ontologies.py").exists() and not (ROOT / "core" / "ablation_study.py").exists()
    for path in _py_files(*SCIENTIFIC_DIRS):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            assert not [n for n in names if n.split(".")[0] == "validation"], f"{path} importa validation/"


def test_target_exclusion_comes_from_runtime_metadata_not_from_a_domain_vocabulary():
    from core.trepan_reloaded_extractor import TrepanReloadedExtractor as E
    vocabulary = set(E.FEATURE_MATCH_EXCLUDED_NAMES) | set(E.FEATURE_MATCH_EXCLUDED_OBJECT_PROPERTIES)
    assert vocabulary <= {"class", "targetclass", "target", "labelclass", "outcome", "classlabel", "label", "negative", "positive"}


def test_target_metadata_exclusions_are_derived_from_the_observed_class_names():
    owlready2 = pytest.importorskip("owlready2")
    onto = owlready2.World().get_ontology("http://test.org/agn.owl")
    with onto:
        Outcome = type("Verdict", (owlready2.Thing,), {})
        type("ClassA", (Outcome,), {}); type("ClassB", (Outcome,), {})
        Measure = type("Measure", (owlready2.Thing,), {})
        type("Length", (Measure,), {})
        type("hasVerdict", (owlready2.ObjectProperty,), {"range": [Outcome]})
    from core.trepan_reloaded_extractor import TrepanReloadedExtractor
    ext = TrepanReloadedExtractor(ontology=onto)
    names = lambda: {e.name for e in TrepanReloadedExtractor._unwrap_matching_entities(ext._get_ontology_matching_entities()[0])}
    assert {"ClassA", "ClassB", "Verdict"} <= names()                       # sem metadados: nada é conhecido
    ext.set_target_metadata(class_names=["class_a", "class_b"])
    got = names()
    assert not ({"ClassA", "ClassB", "Verdict", "hasVerdict"} & got) and "Length" in got
    assert "Measure" in got


# --------------------------------------------------------------------------------- geradores genéricos
def _search():
    return svc.scientific_search_config(cv_folds=2, cv_repeats=2, max_capacity_candidates=1, max_semantic_candidates=1,
                                        purity_epsilon_grid=(0.05, 0.01), max_nodes_grid=(7, 15), bootstrap_resamples=20)


def make_frame(n=150, n_features=5, n_classes=2, class_names=None, target="y", categorical=0, seed=0, feature_prefix="f",
               target_position="last", missing=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    score = X[:, 0] + 0.6 * X[:, min(1, n_features - 1)]
    cuts = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    idx = np.digitize(score, cuts)
    names = class_names or [f"c{i}" for i in range(n_classes)]
    df = pd.DataFrame(X, columns=[f"{feature_prefix}{i}" for i in range(n_features)])
    for k in range(categorical):
        levels = np.array(["red", "green", "blue"]) if k % 2 == 0 else np.array(["u", "v"])
        df[f"cat{k}"] = np.where(rng.random(n) < 0.7, levels[idx % len(levels)], rng.choice(levels, n))
    if missing:
        df.loc[rng.choice(n, 5, replace=False), f"{feature_prefix}{n_features - 1}"] = np.nan
    label = np.array(names)[idx]
    if target_position == "first":
        df.insert(0, target, label)
    else:
        df[target] = label
    return df


SCENARIOS = {
    "two_features_binary": dict(n_features=2, n_classes=2, target="y"),
    "seven_features_three_classes": dict(n_features=7, n_classes=3, target="Class", class_names=["low", "mid", "high"]),
    "twenty_five_features_five_classes": dict(n_features=25, n_classes=5, target="outcome_flag", n=200,
                                              class_names=["A", "B", "C", "D", "E"]),
    "mixed_categorical_unicode_target": dict(n_features=4, n_classes=2, target="Diagnóstico", categorical=3,
                                             class_names=["negativo", "positivo"], target_position="first"),
    "numeric_with_missing_values": dict(n_features=6, n_classes=3, target="label", missing=True, class_names=["1", "2", "3"]),
}


@pytest.fixture(scope="module")
def runs():
    out = {}
    for name, kw in SCENARIOS.items():
        df = make_frame(seed=3, **kw)
        out[name] = (df, kw["target"], svc.run_scientific_benchmark(df, target=kw["target"], seed=3, search=_search()))
    return out


# --------------------------------------------------------------------------------- 2-6. formas arbitrárias
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_arbitrary_schema_runs_end_to_end_with_the_same_pipeline_and_one_oracle(runs, name):
    df, target, out = runs[name]
    kw = SCENARIOS[name]
    ev = out.report["evaluation"]
    ids = ev["oracle_contract"]["tree_oracle_ids"]
    assert ids["trepan_original"] == ids["trepan_reloaded"] == out.oracle_id
    t = ev["trepan_scientific_tuning"]
    assert t["test_used_for_selection"] is False and t["structure_selection"]["bootstrap"]["n_resamples"] > 0
    assert set(map(str, out.result.dataset.class_names)) == set(map(str, df[target].unique()))
    assert out.result.dataset.target == target
    assert out.result.dataset.classes == kw.get("n_classes", 2)
    for key in ("trepan_original", "trepan_reloaded"):
        assert out.result.models[key].status == "AVAILABLE"
    assert out.result.dataset.features == df.shape[1] - 1                  # nº de colunas observado, não conhecido a priori


def test_binary_and_multiclass_are_both_handled(runs):
    counts = {name: runs[name][2].result.dataset.classes for name in runs}
    assert counts["two_features_binary"] == 2 and counts["seven_features_three_classes"] == 3 \
        and counts["twenty_five_features_five_classes"] == 5


def test_categorical_features_are_encoded_generically_and_numeric_missing_values_do_not_break(runs):
    cat = runs["mixed_categorical_unicode_target"][2].report
    kinds = {c["name"]: c.get("treatment") for c in cat["contract"]["columns"]}
    assert any(k for n, k in kinds.items() if n.startswith("cat")) and cat["manifest"]["train_rows"] > 0
    assert runs["numeric_with_missing_values"][2].oracle_id


def test_tuning_chooses_by_the_same_process_and_may_differ_across_datasets(runs):
    picks = {n: runs[n][2].report["evaluation"]["trepan_scientific_tuning"]["structure_selection"]["selected_label"] for n in runs}
    grid = {f"purity_epsilon={e:g}, max_nodes={m}" for e in (0.05, 0.01) for m in (7, 15, 31)}
    assert all(p in grid for p in picks.values())          # sempre um ponto da grelha configurada (nunca um valor "à medida")
    # a política é a mesma para todos; o resultado pode (e é aceitável que) diferir por dataset
    rules = {runs[n][2].report["evaluation"]["trepan_scientific_tuning"]["selection_rule"] for n in runs}
    assert len(rules) == 1


# --------------------------------------------------------------------------------- 7. ontologia presente/ausente
def test_presence_or_absence_of_an_ontology_does_not_break_the_pipeline():
    pytest.importorskip("owlready2")
    from test_production_attribution_gate import _uninformative
    df, onto = _uninformative(n=200, seed=2)
    without = svc.run_scientific_benchmark(df, target="target", seed=2, search=_search())
    with_onto = svc.run_scientific_benchmark(df, target="target", seed=2, ontology=onto, require_reasoner=False, search=_search())
    for out in (without, with_onto):
        assert out.report["evaluation"]["oracle_contract"]["single_oracle_for_all_trees"] and out.usable_as_benchmark
    assert without.result.ontology is None and with_onto.result.ontology is not None


# --------------------------------------------------------------------------------- 8. novo dataset sem alterar código
def _source_digest():
    h = hashlib.sha256()
    for path in sorted(_py_files("core", "gui")):
        h.update(path.read_bytes())
    return h.hexdigest()


def test_adding_a_new_compatible_dataset_requires_no_source_change(tmp_path):
    from scipy.io import arff
    before = _source_digest()
    for i, (nf, nc, tgt, classes) in enumerate([(3, 2, "resposta", ["nao", "sim"]), (6, 4, "grupo", ["g1", "g2", "g3", "g4"])]):
        df = make_frame(n=140, n_features=nf, n_classes=nc, class_names=classes, target=tgt, seed=10 + i)
        path = tmp_path / f"novo_dataset_{i}.arff"            # um ficheiro novo, com nome e schema arbitrários
        lines = [f"@relation novo_{i}"] + [f"@attribute {c} numeric" for c in df.columns[:-1]] \
            + [f"@attribute {tgt} {{{','.join(classes)}}}", "@data"]
        lines += [",".join(f"{v:.4f}" for v in row[:-1]) + f",{row[-1]}" for row in df.itertuples(index=False)]
        path.write_text("\n".join(lines))
        data, _meta = arff.loadarff(str(path))
        loaded = pd.DataFrame(data)
        for col in loaded.columns:
            if loaded[col].dtype == object:
                loaded[col] = loaded[col].str.decode("utf-8")
        out = svc.run_scientific_benchmark(loaded, target=loaded.columns[-1], seed=1, search=_search())
        assert out.oracle_id and out.result.dataset.classes == nc and out.result.dataset.target == tgt
    assert _source_digest() == before


# --------------------------------------------------------------------------------- 9. nenhum resultado depende da identidade
def test_results_do_not_depend_on_dataset_identity_names_or_labels():
    """Os mesmos números com outros nomes de colunas/alvo/classes dão a mesma seleção e as mesmas estatísticas."""
    base = make_frame(n=140, n_features=4, n_classes=3, class_names=["a_x", "b_y", "c_z"], target="t1", seed=21, feature_prefix="aa")
    other = base.rename(columns={**{f"aa{i}": f"zz_measure_{i}" for i in range(4)}, "t1": "renamed_target"})
    other["renamed_target"] = other["renamed_target"].map({"a_x": "alpha", "b_y": "beta", "c_z": "gamma"})
    r1 = svc.run_scientific_benchmark(base, target="t1", seed=5, search=_search())
    r2 = svc.run_scientific_benchmark(other, target="renamed_target", seed=5, search=_search())
    t1 = r1.report["evaluation"]["trepan_scientific_tuning"]; t2 = r2.report["evaluation"]["trepan_scientific_tuning"]
    assert t1["structure_selected"] == t2["structure_selected"]
    s1 = [h["stats"]["fidelity_mean"] for h in t1["structure_history"]]
    s2 = [h["stats"]["fidelity_mean"] for h in t2["structure_history"]]
    assert s1 == s2
    assert t1["structure_selection"]["bootstrap"]["distribution"] == t2["structure_selection"]["bootstrap"]["distribution"]
    assert r1.report["manifest"]["train_rows"] == r2.report["manifest"]["train_rows"]


def test_tuning_grid_is_configuration_not_code():
    X = make_frame(n=120, n_features=3, seed=4)
    custom = svc.scientific_search_config(cv_folds=2, cv_repeats=2, max_capacity_candidates=1, max_semantic_candidates=1,
                                          purity_epsilon_grid=(0.1, 0.03), max_nodes_grid=(9, 21, 127), bootstrap_resamples=10)
    out = svc.run_scientific_benchmark(X, target="y", seed=2, search=custom)
    labels = {h["label"] for h in out.report["evaluation"]["trepan_scientific_tuning"]["structure_history"]}
    assert "purity_epsilon=0.1, max_nodes=127" in labels and "purity_epsilon=0.03, max_nodes=9" in labels


def test_counterfactual_config_is_derived_from_observed_metadata_only():
    from counterfactuals.dataset_config import build_config_from_arff_meta
    for meta in ({"features": ["p", "q", "r"], "classes": ["x", "y"], "target": "alvo", "file_name": "qualquer.arff"},
                 {"features": [f"v{i}" for i in range(11)], "classes": ["a", "b", "c", "d"], "target": "grupo", "file_name": "outro"}):
        cfg = build_config_from_arff_meta(meta)
        assert cfg["numeric_features"] == meta["features"] and cfg["target"] == meta["target"]
        assert cfg["is_multiclass"] == (len(meta["classes"]) > 2)
        assert list(cfg["class_labels"].values()) == meta["classes"] and cfg["mlp_params"] == {}   # sem hiperparâmetros dedicados


def test_vendored_clear_settings_carry_no_dataset_defaults():
    from counterfactuals.clear import CLEAR_settings
    assert CLEAR_settings.numeric_features == [] and CLEAR_settings.categorical_features == [] and CLEAR_settings.class_labels == {}
    assert CLEAR_settings.multi_class_focus == "All" and CLEAR_settings.model_name == "model"
