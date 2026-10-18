"""H2 (Doc. 62): testes METAMÓRFICOS da maquinaria de literais derivados / pool com ontologias sintéticas DESCONHECIDAS de validade conhecida. Relações verificadas: renomeação consistente de
atributos e conceitos, permutação da ordem das colunas e da declaração OWL, número variável de atributos/famílias/níveis/classes, ontologia desconhecida (a extensão esperada deriva da configuração,
nunca de nomes fixos) e equivalência sem ontologia. Sem desempenho; sem rótulos reais; sem dataset conhecido."""
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pytest.importorskip("owlready2")
from core.trepan_original import TrepanOriginalClassifier
from validation.benchmark import h2_synthetic as S
from validation.benchmark.b31_facts import compile_registry
from validation.benchmark.h2_axiom_mechanisms import defined_union_concepts, derived_union_literals
from validation.benchmark.h2_pool_study import fit_pool_arm

P = dict(min_sample=120, max_queries=600, max_nodes=4, max_depth=2, max_n=1, beam_width=2, min_samples_leaf=3, max_features_per_node=3, random_state=11)
FAM = {"alpha": 4, "beta": 3}


def _owl_to_literals(owl_text, names, X):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "o.owl"
        p.write_text(owl_text, encoding="utf-8")
        reg = compile_registry(str(p), names, run_reasoner=True)
        assert reg.ontology_status == "VALID" and all(reg.feature_entity.get(n) for n in names)           # matching 100 %
        return derived_union_literals(reg, X, names, only_concepts=defined_union_concepts(p))


def run(X, names, truth, owl_text, params=P, extra_slots=None):
    lits = _owl_to_literals(owl_text, names, X)
    mf = params["max_features_per_node"] + (len(lits) if extra_slots is None else extra_slots)
    arm = fit_pool_arm(X, names, lits, truth, params, extras="derived", mf=mf)
    ext = {l.concept: frozenset(l.attributes) for l in lits}
    reasons = {l["concept"]: l["reasons_by_search"] for l in arm["trace"]["pool"]["literals"]}
    sigs = [None if s is None else s["sig"] for s in arm["model"]._search_log]
    return {"ext": ext, "reasons": reasons, "sigs": sigs, "arm": arm, "lits": lits}


def _truth_ext(fam0, k, k_true):
    return frozenset(f"{fam0}__L{j}" for j in range(k_true + 1, k + 1))


@pytest.fixture(scope="module")
def base():
    X, names, truth = S.make_generic_data(220, 1, FAM, 3, 2, k_true=2)
    owl = S.build_generic_owl(names, FAM, 2, "TRUE")
    return X, names, truth, owl, run(X, names, truth, owl)


def _rename_fn():
    table = [("alpha", "omega"), ("beta", "psi"), ("noise__", "zz__"), ("num__", "yy__"), ("__L", "__Q"), ("__U", "__Z"), ("_L", "_Q"), ("AtLeast", "UpTo"), ("Lvl", "Nv"), ("G_", "H_")]
    def fn(text):
        for a, b in table:
            text = text.replace(a, b)
        return text
    return fn


def test_consistent_renaming_of_attributes_and_concepts_preserves_the_structure(base):
    X, names, truth, owl, ref = base
    fn = _rename_fn()
    r = run(X, [fn(n) for n in names], truth.rename(fn), fn(owl))
    assert r["ext"] == {fn(c): frozenset(fn(a) for a in e) for c, e in ref["ext"].items()}
    assert r["reasons"] == {fn(c): v for c, v in ref["reasons"].items()}
    ren = lambda sig: None if sig is None else (sig[0], tuple((fn(nm), t, g) for nm, t, g in sig[1]))
    assert r["sigs"] == [ren(s) for s in ref["sigs"]]


def test_column_order_and_owl_declaration_order_do_not_change_the_outcome(base):
    X, names, truth, owl, ref = base
    perm = np.random.RandomState(5).permutation(len(names))
    n2 = [names[i] for i in perm]
    r = run(X[:, perm], n2, truth.reorder(n2), S.build_generic_owl(n2, FAM, 2, "TRUE", shuffle_seed=9))
    assert r["ext"] == ref["ext"] and r["reasons"] == ref["reasons"]
    assert r["sigs"] == ref["sigs"]                                                                             # mesmas decisões de split, por nome


@pytest.mark.parametrize("n_noise,n_num,fams", [(0, 1, {"alpha": 4}), (6, 3, {"alpha": 4, "beta": 3, "gamma": 5}), (1, 0, {"alpha": 4, "beta": 2})])
def test_number_of_attributes_and_families_does_not_change_the_derived_truth_literal(n_noise, n_num, fams):
    X, names, truth = S.make_generic_data(220, 2, fams, n_noise, n_num, k_true=2)
    r = run(X, names, truth, S.build_generic_owl(names, fams, 2, "TRUE"))
    want = _truth_ext("alpha", fams["alpha"], 2)
    assert want in set(r["ext"].values())                                                                       # a extensão verdadeira deriva da configuração (nunca de nomes fixos)
    concept = [c for c, e in r["ext"].items() if e == want][0]
    root = r["sigs"][0]
    assert root is not None and any(nm.startswith("onto_any[") and concept in nm for nm, _, _ in root[1])      # com max_n=1 só o literal derivado exprime a verdade ⇒ escolhido na raiz


@pytest.mark.parametrize("fams,k_true", [({"a": 5}, 1), ({"a": 5}, 3), ({"q": 3, "r": 6}, 1), ({"x": 7, "y": 2}, 4)])
def test_unknown_synthetic_ontologies_true_perm_neutral_flat(fams, k_true):
    X, names, truth = S.make_generic_data(260, 3, fams, 2, 2, k_true=k_true)
    fam0 = next(iter(fams))
    want = _truth_ext(fam0, fams[fam0], k_true)
    true_l = _owl_to_literals(S.build_generic_owl(names, fams, k_true, "TRUE"), names, X)
    assert want in {frozenset(l.attributes) for l in true_l} or len(want) < 2
    perm_l = _owl_to_literals(S.build_generic_owl(names, fams, k_true, "PERM"), names, X)
    assert want not in {frozenset(l.attributes) for l in perm_l}                                                # OWL válido, semântica errada: não exprime a verdade
    assert _owl_to_literals(S.build_generic_owl(names, fams, k_true, "NEUTRAL"), names, X) == []
    assert _owl_to_literals(S.build_generic_owl(names, fams, k_true, "FLAT"), names, X) == []
    assert all(l.needs_reasoner for l in true_l)                                                                # a extensão só existe por inferência


@pytest.mark.parametrize("n_classes", [2, 3, 4])
def test_equivalence_without_ontology_for_any_number_of_classes_and_attributes(n_classes):
    X, names, truth = S.make_generic_data(220, 4, {"alpha": 4, "beta": 3}, 4, 2, k_true=2, n_classes=n_classes)
    p = dict(P, max_n=3, max_features_per_node=5)
    o1, o2 = truth.reorder(names), truth.reorder(names)
    a = TrepanOriginalClassifier(**p).fit(X, oracle=o1, feature_names=names)
    b = fit_pool_arm(X, names, [], o2, p, extras="none", mf=5)
    sha = lambda m: hashlib.sha256(json.dumps(m.export_rules(), default=str).encode()).hexdigest()
    assert sha(a) == sha(b["model"]) and a.membership_queries_ == b["queries"] and (o1.calls, o1.rows) == (o2.calls, o2.rows)       # Reloaded(None) ≡ Original
    assert all(np.array_equal(x, y) for x, y in zip(o1.seen, o2.seen))                                           # mesmo fluxo de consultas
    assert len(set(np.asarray(o1.predict(X)))) == n_classes or n_classes > 3


def test_rejected_and_neutral_ontologies_behave_as_the_original(base):
    X, names, truth, owl, ref = base
    neutral = S.build_generic_owl(names, FAM, 2, "NEUTRAL")
    o1, o2 = truth.reorder(names), truth.reorder(names)
    a = TrepanOriginalClassifier(**P).fit(X, oracle=o1, feature_names=names)
    lits = _owl_to_literals(neutral, names, X)
    assert lits == []
    b = fit_pool_arm(X, names, lits, o2, P, extras="derived", mf=P["max_features_per_node"])                    # S2: ontologia válida sem estrutura ativa
    assert json.dumps(a.export_rules(), default=str) == json.dumps(b["model"].export_rules(), default=str) and (o1.calls, o1.rows) == (o2.calls, o2.rows)


def _compile(text, names, tmp):
    p = Path(tmp) / f"o{abs(hash(text)) % 10**8}.owl"
    p.write_text(text, encoding="utf-8")
    return compile_registry(str(p), names, run_reasoner=True), p


def test_ambiguous_matching_never_yields_an_incomplete_derived_literal(tmp_path):
    """Se um atributo da família tem matching ambíguo, o OR das colunas mapeadas NÃO é a extensão do conceito: o literal tem de ser recusado e a recusa registada (solidez), em qualquer configuração."""
    from validation.benchmark.b30_axiom_probe import dprop
    for fams, k_true in (({"alpha": 4, "beta": 3}, 2), ({"q": 3, "r": 6}, 1), ({"x": 7}, 3)):
        X, names, _ = S.make_generic_data(150, 5, fams, 2, 1, k_true=k_true)
        fam0, k0 = next(iter(fams.items()))
        good = S.build_generic_owl(names, fams, k_true, "TRUE")
        bad = good.replace("</rdf:RDF>", dprop(f"has_dup_{fam0}_L{k0}", f"Lvl_{fam0}_L{k0}", label=f"{fam0}__L{k0}") + "</rdf:RDF>")
        reg, p = _compile(bad, names, tmp_path)
        rej = []
        lits = derived_union_literals(reg, X, names, only_concepts=defined_union_concepts(p), rejected=rej)
        assert reg.feature_entity[f"{fam0}__L{k0}"] is None                                               # o atributo ambíguo fica sem entidade (P3)
        assert not any(l.family == fam0 for l in lits) and rej                                           # nenhum literal incompleto; rejeição registada
        assert all(r["reason"] == "INCOMPLETE_EXTENSION_UNRESOLVED_MATCH" and f"{fam0}__L{k0}" in r["unresolved_attributes"] for r in rej)
        reg2, p2 = _compile(good, names, tmp_path)
        assert derived_union_literals(reg2, X, names, only_concepts=defined_union_concepts(p2))           # com matching limpo os literais existem


def test_unmodeled_attribute_without_candidate_does_not_block_sound_literals(tmp_path):
    """Um atributo SEM entidade candidata (não modelado) não torna a extensão insegura: em mundo aberto fica fora do conceito."""
    fams = {"alpha": 4}
    X, names, _ = S.make_generic_data(150, 6, fams, 2, 1, k_true=2)
    names2 = names + ["extra__never_modeled"]
    X2 = np.column_stack([X, np.zeros(len(X))])
    reg, p = _compile(S.build_generic_owl(names, fams, 2, "TRUE"), names2, tmp_path)
    lits = derived_union_literals(reg, X2, names2, only_concepts=defined_union_concepts(p))
    assert _truth_ext("alpha", 4, 2) in {frozenset(l.attributes) for l in lits}


def test_unknown_categories_mixed_numeric_and_categorical_columns():
    from validation.benchmark.h2_trace import augment
    fams = {"alpha": 4, "beta": 3}
    X, names, _ = S.make_generic_data(150, 7, fams, 2, 2, k_true=2)
    lits = _owl_to_literals(S.build_generic_owl(names, fams, 2, "TRUE"), names, X)
    Xn = X.copy()
    grp = [i for i, n in enumerate(names) if n.startswith("alpha__")]
    Xn[:5, grp] = 0.0                                                                                   # categoria nunca vista: nenhum nível ativo
    Xa, na = augment(Xn, names, lits)
    d = {n: Xa[:, i] for i, n in enumerate(na) if n.startswith("onto_any[")}
    alpha_cols = [v for k, v in d.items() if "alpha" in k]
    assert alpha_cols and all(np.all(v[:5] == 0.0) for v in alpha_cols)                                  # derivadas = 0 nessas linhas (nunca atribuídas a um nível)
    assert all(set(np.unique(v)) <= {0.0, 1.0} for v in d.values())                                     # binárias mesmo com colunas numéricas de escala distinta na mesma matriz
    assert Xa.shape[1] == X.shape[1] + len(lits)
