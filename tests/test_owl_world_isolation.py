"""Isolamento dos mundos OWL: o resultado de uma ontologia nunca pode depender do que foi carregado antes (A→B→A, B→A→B).

Ontologias sintéticas e genéricas (nenhum dataset real): duas ontologias que declaram as MESMAS anotações curtas (``statisticRole``,
``measurementFamily``) em namespaces diferentes — exatamente a colisão que fazia a primeira ontologia carregada perder as features
relacionais. As features relacionais são verificadas contra fórmulas calculadas de forma independente em NumPy.
"""
import hashlib
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("owlready2")

from core.ontology_processor import OntologyProcessor
from core.ontology_quality import OntologyQualityGate, annotation_values
from core.owl_runtime import load_ontology_isolated, new_owl_world, release_ontology
from core.trepan_reloaded_extractor import TrepanReloadedExtractor

HAS_JAVA = shutil.which("java") is not None


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
ROLES = ("mean", "worst", "error")


def _owl(namespace: str, families, profile: str) -> str:
    ns = f"https://example.org/synthetic/{namespace}#"
    out = [f'<?xml version="1.0" encoding="utf-8"?>',
           f'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" '
           f'xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:xsd="http://www.w3.org/2001/XMLSchema#" xmlns:dom="{ns}" xml:base="{ns[:-1]}">',
           f'<owl:Ontology rdf:about="{ns[:-1]}"><rdfs:comment>TBox only (synthetic test ontology)</rdfs:comment></owl:Ontology>',
           '<owl:AnnotationProperty rdf:about="#measurementFamily"/>', '<owl:AnnotationProperty rdf:about="#statisticRole"/>',
           f'<owl:Class rdf:about="#{profile}"><rdfs:label>{profile}</rdfs:label></owl:Class>']
    for fam in families:
        out.append(f'<owl:DatatypeProperty rdf:about="#has{fam.title()}Statistic"><rdfs:label>{fam} statistics</rdfs:label>'
                   f'<rdfs:domain rdf:resource="#{profile}"/><rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#double"/></owl:DatatypeProperty>')
        for role in ROLES:
            out.append(f'<owl:DatatypeProperty rdf:about="#has{fam.title()}{role.title()}"><rdfs:label>{fam}_{role}</rdfs:label>'
                       f'<rdfs:domain rdf:resource="#{profile}"/><rdfs:subPropertyOf rdf:resource="#has{fam.title()}Statistic"/>'
                       f'<rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#double"/>'
                       f'<dom:measurementFamily>{fam}</dom:measurementFamily><dom:statisticRole>{role}</dom:statisticRole></owl:DatatypeProperty>')
    out.append("</rdf:RDF>")
    return "\n".join(out)


FAMILIES = {"A": ("alpha", "beta"), "B": ("gamma", "delta", "epsilon")}


@pytest.fixture(scope="module")
def onto_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("owl")
    paths = {}
    for key, fams in FAMILIES.items():
        p = d / f"{key}.owl"
        p.write_text(_owl(f"onto_{key.lower()}", fams, f"{key}Profile"), encoding="utf-8")
        paths[key] = p
    return paths


def _frame(key: str, n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng({"A": 1, "B": 2}[key] + seed)
    cols = {}
    for fam in FAMILIES[key]:
        mean = rng.uniform(1.0, 5.0, n)
        cols[f"{fam}_mean"] = mean
        cols[f"{fam}_worst"] = mean * rng.uniform(1.1, 2.0, n)
        cols[f"{fam}_error"] = mean * rng.uniform(0.01, 0.2, n)
    return pd.DataFrame(cols)


def _derive(onto, frame: pd.DataFrame):
    gate = OntologyQualityGate()
    rep = gate.evaluate(list(frame.columns), onto, require_reasoner=False)
    proc = OntologyProcessor(onto, quality_gate=gate)
    proc.fit(frame, accepted_matches=[m for m in rep.matches if m.get("accepted")], log=False)
    return proc, proc.transform(frame)


def _digest(df: pd.DataFrame) -> str:
    return hashlib.sha256(np.ascontiguousarray(df.to_numpy(float)).tobytes() + "|".join(df.columns).encode()).hexdigest()


def _load(path):
    return TrepanReloadedExtractor.load_ontology_file(path)           # caminho de carga usado pela GUI


def _expected_relational(frame: pd.DataFrame, family: str) -> dict:
    mean, worst, err = (frame[f"{family}_{r}"].to_numpy(float) for r in ROLES)
    tiny = 1e-12
    return {
        f"onto_{family}_worst_minus_mean": worst - mean,
        f"onto_{family}_relative_worst_delta": (worst - mean) / np.maximum(np.abs(mean), tiny),
        f"onto_{family}_family_contrast": (worst - mean) / np.maximum(np.abs(worst) + np.abs(mean), tiny),
        f"onto_{family}_normalized_error": np.abs(err) / np.maximum(np.abs(mean) + np.abs(worst), tiny),
        f"onto_{family}_error_ratio": err / np.maximum(np.abs(mean), tiny),
    }


def test_each_load_gets_its_own_world_even_for_the_same_file(onto_files):
    one, two = _load(onto_files["A"]), _load(onto_files["A"])
    assert one.world is not two.world
    import owlready2
    assert one.world is not owlready2.default_world and two.world is not owlready2.default_world
    a, b = new_owl_world(), new_owl_world()
    assert a is not b
    release_ontology(one)                       # idempotente e sem efeitos no segundo mundo
    release_ontology(one)
    assert len(list(two.data_properties())) > 0


@pytest.mark.parametrize("order", ["ABA", "BAB", "AABB", "ABBA"])
def test_derived_features_do_not_depend_on_load_order(onto_files, order):
    baseline = {}
    for key in "AB":
        proc, out = _derive(_load(onto_files[key]), _frame(key))
        baseline[key] = (_digest(out), list(out.columns), proc.generation_summary_)
    for key in order:                           # sequência no MESMO processo
        proc, out = _derive(_load(onto_files[key]), _frame(key))
        assert _digest(out) == baseline[key][0], f"{order}: {key} difere do carregamento isolado"
        assert list(out.columns) == baseline[key][1]
        n_rel = sum(1 for s in proc.feature_specs_ if s["kind"] == "relational")
        assert n_rel == 5 * len(FAMILIES[key]) and n_rel > 0


def test_annotation_collision_is_fixed_even_when_two_ontologies_share_one_world(onto_files):
    """Defeito original: duas ontologias no mesmo mundo e a primeira perdia ``statisticRole``. Agora o resultado é igual ao isolado."""
    world = new_owl_world()
    a = world.get_ontology(str(onto_files["A"].resolve())).load()
    b = world.get_ontology(str(onto_files["B"].resolve())).load()
    for onto, key in ((a, "A"), (b, "B")):
        ref_proc, ref_out = _derive(_load(onto_files[key]), _frame(key))
        proc, out = _derive(onto, _frame(key))
        assert _digest(out) == _digest(ref_out)
    entity = next(p for p in a.data_properties() if p.name.endswith("AlphaWorst"))
    assert annotation_values(entity, ("statisticRole",)) == ["worst"]
    assert annotation_values(entity, ("measurementFamily",)) == ["alpha"]
    other = next(p for p in b.data_properties() if p.name.endswith("GammaMean"))
    assert annotation_values(other, ("statisticRole",)) == ["mean"]


def test_relational_features_are_numerically_correct_and_never_constant(onto_files):
    for key in "AB":
        frame = _frame(key)
        _, out = _derive(_load(onto_files[key]), frame)
        for family in FAMILIES[key]:
            for name, expected in _expected_relational(frame, family).items():
                assert name in out.columns, f"{name} em falta"
                np.testing.assert_allclose(out[name].to_numpy(float), expected, rtol=1e-12, atol=1e-12, err_msg=name)
                assert np.unique(np.round(out[name].to_numpy(float), 12)).size > 10, f"{name} é (quase) constante"
                assert np.isfinite(out[name].to_numpy(float)).all()
        derived = [c for c in out.columns if c.startswith("onto_")]
        assert derived and all(out[c].std() > 0 for c in derived)


def test_aggregates_are_standardised_means_of_their_group_fitted_on_train_only(onto_files):
    frame = _frame("A", n=100)
    train, test = frame.iloc[:70], frame.iloc[70:]
    proc, _ = _derive(_load(onto_files["A"]), train)
    out = proc.transform(test)
    for fam in FAMILIES["A"]:
        cols = [f"{fam}_{r}" for r in ROLES]
        mu, sd = train[cols].mean().to_numpy(), train[cols].std(ddof=0).to_numpy()
        expected = ((test[cols].to_numpy() - mu) / sd).mean(axis=1)
        name = next(c for c in out.columns if c.startswith("onto_") and c.endswith("_aggregate") and fam.title() in c)
        np.testing.assert_allclose(out[name].to_numpy(), expected, rtol=1e-12, atol=1e-12)


def test_metamorphic_row_column_order_and_file_copy_do_not_change_the_features(onto_files, tmp_path):
    frame = _frame("A")
    _, base = _derive(_load(onto_files["A"]), frame)
    perm = np.random.default_rng(7).permutation(len(frame))
    _, rows = _derive(_load(onto_files["A"]), frame.iloc[perm].reset_index(drop=True))
    np.testing.assert_allclose(rows.to_numpy(float), base.iloc[perm].reset_index(drop=True).to_numpy(float), rtol=1e-12, atol=1e-12)
    cols = list(frame.columns)[::-1]
    _, shuffled_cols = _derive(_load(onto_files["A"]), frame[cols])
    np.testing.assert_allclose(shuffled_cols[base.columns].to_numpy(float), base.to_numpy(float), rtol=1e-12, atol=1e-12)
    copy = tmp_path / "completely_different_file_name_zz.owl"
    copy.write_bytes(onto_files["A"].read_bytes())
    _, same = _derive(_load(copy), frame)
    assert _digest(same) == _digest(base)                                     # nome/caminho do ficheiro não influencia


@pytest.mark.skipif(not HAS_JAVA, reason="HermiT requer Java; sem Java o raciocinador não pode correr (ambiente)")
def test_reasoner_runs_per_world_and_an_inconsistent_ontology_never_contaminates_another(onto_files, tmp_path):
    from core.ontology_reasoner import run_owl_reasoner
    bad = tmp_path / "bad.owl"
    bad.write_text(_owl("bad", ("zeta",), "BadProfile").replace("</rdf:RDF>",
        '<owl:Class rdf:about="#X"/><owl:Class rdf:about="#Y"><owl:disjointWith rdf:resource="#X"/></owl:Class>'
        '<owl:Class rdf:about="#Z"><rdfs:subClassOf rdf:resource="#X"/><rdfs:subClassOf rdf:resource="#Y"/></owl:Class></rdf:RDF>'), encoding="utf-8")
    a1 = _load(onto_files["A"]); r1 = run_owl_reasoner(a1, engine="hermit", infer_property_values=True, debug=0)
    b = _load(bad); rb = run_owl_reasoner(b, engine="hermit", infer_property_values=True, debug=0)
    a2 = _load(onto_files["A"]); r2 = run_owl_reasoner(a2, engine="hermit", infer_property_values=True, debug=0)
    assert r1["executed"] and r1["consistent"] is True and not r1["unsatisfiable_classes"]
    assert rb["executed"] and (rb["consistent"] is False or rb["unsatisfiable_classes"])        # a má é detetada…
    assert r2["executed"] and r2["consistent"] is True and not r2["unsatisfiable_classes"]      # …sem afetar A depois


@pytest.mark.skipif(not HAS_JAVA, reason="HermiT requer Java; sem Java o raciocinador não pode correr (ambiente)")
def test_owl_semantic_provider_context_is_identical_before_and_after_loading_another_ontology(onto_files):
    from core.benchmark.semantic import OwlSemanticProvider
    frame = _frame("A", n=120)
    names = list(frame.columns)

    def context(key):
        ctx = OwlSemanticProvider(str(onto_files[key])).build(frame.to_numpy(float) if key == "A" else _frame(key, n=120).to_numpy(float),
                                                               names if key == "A" else list(_frame(key).columns))
        assert ctx.available, ctx.reason
        X = frame.to_numpy(float) if key == "A" else _frame(key, n=120).to_numpy(float)
        return (hashlib.sha256(np.ascontiguousarray(ctx.enrich(X)).tobytes()).hexdigest(), ctx.structure_signature(), list(ctx.enriched_names))
    first = context("A"); context("B"); again = context("A")
    assert first == again


@pytest.mark.skipif(not HAS_JAVA, reason="HermiT requer Java; sem Java o raciocinador não pode correr (ambiente)")
def test_gui_loading_ontologies_in_sequence_keeps_state_isolated(onto_files, qapp):
    from PyQt6.QtWidgets import QMessageBox
    from gui.biuri_app_complete import BiuriApp
    for kind in ("information", "warning", "critical"):
        setattr(QMessageBox, kind, staticmethod(lambda *a, **k: None))
    w = BiuriApp()
    seen_worlds, seen_ids, results = [], [], {}
    for key in "ABA":
        w.cf_interactive_result = {"stale": True}                     # resultado CF «antigo» deve cair a cada troca de ontologia
        assert w.load_ontology_from_file(str(onto_files[key])) is True
        assert w.cf_interactive_result is None
        onto = w.loaded_ontology
        seen_worlds.append(onto.world); seen_ids.append(w._ontology_state_id)
        assert w._ontology_match_cache == {}
        _, out = _derive(onto, _frame(key))
        results.setdefault(key, []).append(_digest(out))
    assert len({id(x) for x in seen_worlds}) == 3 and len(set(seen_ids)) == 3          # cada carga = mundo e estado novos
    assert results["A"][0] == results["A"][1]                                           # A→B→A reproduz A exatamente
    _, ref = _derive(_load(onto_files["A"]), _frame("A"))
    assert results["A"][0] == _digest(ref)
    w.close()
