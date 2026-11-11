"""Agnosticismo da leitura/validação do esquema do dataset (CLAUDE.md §1): nenhum dataset conhecido no código nem no vocabulário;
domínios nunca vistos (outros nomes, nº de colunas, nº de classes, línguas, tipos) resolvem pelo mesmo mecanismo."""
import json
import random
import string
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("owlready2")
from owlready2 import World

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import check_dataset_agnosticism as chk  # noqa: E402

from core.dataset_schema import apply_schema, build_schema, read_tabular_text, validate_schema  # noqa: E402
from core.ontology_schema import load_vocabulary, read_ontology_schema  # noqa: E402

FILES = ["core/ontology_schema.py", "core/ontology_schema_vocabulary.json", "core/dataset_schema.py",
         "gui/dataset_schema_support.py", "docs/ESQUEMA_DATASET_E_ONTOLOGIA.md"]


def test_no_known_dataset_name_in_schema_code_vocabulary_or_docs():
    names = chk.known_dataset_regex(ROOT)
    offenders = []
    for rel in FILES:
        for i, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
            if names.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()[:80]}")
    assert not offenders, offenders


def test_static_contract_still_passes_with_the_new_modules():
    assert chk.scan(ROOT) == []


def _word(rng, n=7):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def _make_domain(tmp_path, seed, n_num, n_cat, labels, vocab, target_pos):
    """Domínio inventado: nomes aleatórios, nº de colunas/classes variável, alvo em posição arbitrária, ontologia descritora."""
    rng = random.Random(seed)
    attr_cls, name_p, type_p, label_cls, value_p = vocab
    num = [f"{_word(rng)}_{i}" for i in range(n_num)]
    cat = [f"{_word(rng)}_c{i}" for i in range(n_cat)]
    target = f"{_word(rng)}_t"
    cols = num + cat
    rng.shuffle(cols)
    cols.insert(target_pos if target_pos >= 0 else len(cols), target)
    cat_values = {c: [f"{_word(rng, 4)}{j}" for j in range(3)] for c in cat}
    rows = []
    for r in range(60):
        row = {}
        for c in cols:
            if c == target:
                row[c] = labels[r % len(labels)]
            elif c in num:
                row[c] = f"{rng.random() * 100:.3f}"
            else:
                row[c] = cat_values[c][(r * 7 + len(c)) % 3]
        rows.append(row)
    csv = tmp_path / f"{_word(rng)}.csv"
    pd.DataFrame(rows, columns=cols).to_csv(csv, index=False)
    body = (f'<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            f'xmlns:owl="http://www.w3.org/2002/07/owl#" xml:base="http://ex.org/o" xmlns="http://ex.org/o#">'
            f'<owl:Ontology rdf:about="http://ex.org/o"/><owl:Class rdf:about="#{attr_cls}"/><owl:Class rdf:about="#{label_cls}"/>'
            + "".join(f'<owl:DatatypeProperty rdf:about="#{p}"/>' for p in (name_p, type_p, value_p)))
    for c in num + cat:
        kind = "numeric" if c in num else "categorical"
        body += (f'<rdf:Description rdf:about="#d_{c}"><rdf:type rdf:resource="#{attr_cls}"/><{name_p}>{c}</{name_p}>'
                 f'<{type_p}>{kind}</{type_p}></rdf:Description>')
    for i, lab in enumerate(labels, 1):
        body += (f'<rdf:Description rdf:about="#lab_{i:02d}"><rdf:type rdf:resource="#{label_cls}"/>'
                 f'<{value_p}>{lab}</{value_p}></rdf:Description>')
    owl = tmp_path / f"{_word(rng)}.owl"
    owl.write_text(body + "</rdf:RDF>", encoding="utf-8")
    return csv, owl, target, num, cat


VOCABS = [("Atributo", "nombreAtributo", "tipoDato", "EtiquetaObjetivo", "valorEtiqueta"),
          ("Attribute", "attributeName", "dataType", "TargetLabel", "classValue"),
          ("Variable", "biuriColumnName", "biuriDataType", "ClassLabel", "biuriClassValue")]


@pytest.mark.parametrize("seed,n_num,n_cat,labels,vocab,target_pos", [
    (1, 3, 0, ["z", "a"], VOCABS[0], 0),
    (2, 12, 2, ["k1", "k2", "k3"], VOCABS[1], 5),
    (3, 1, 4, ["yes", "no", "maybe", "later", "never"], VOCABS[2], -1),
    (4, 7, 1, ["10", "9"], VOCABS[1], 2),
])
def test_unseen_domains_resolve_by_the_same_mechanism(tmp_path, seed, n_num, n_cat, labels, vocab, target_pos):
    csv, owl, target, num, cat = _make_domain(tmp_path, seed, n_num, n_cat, labels, vocab, target_pos)
    onto = World().get_ontology(owl.as_uri()).load()
    df, meta = read_tabular_text(csv)
    decl = read_ontology_schema(onto, list(df.columns))
    s = build_schema(df, file_path=csv, source_format="csv", ontology_decl=decl)
    assert s.target == target                                                  # em qualquer posição, sem a indicar
    assert s.class_order == labels and s.class_mapping == {l: i for i, l in enumerate(labels)}
    assert {c.name: c.kind for c in s.columns if c.name != target} == {**{c: "numeric" for c in num}, **{c: "categorical" for c in cat}}
    rep = validate_schema(s, df, ontology_decl=decl)
    assert not [i for i in rep.critical if i.code.startswith("ONTOLOGY")], rep.format()
    assert list(apply_schema(df, s).columns)[-1] == target


def test_metamorphic_rename_and_reorder_gives_equivalent_resolution(tmp_path):
    base = _make_domain(tmp_path, 11, 5, 2, ["u", "v", "w"], VOCABS[1], 3)
    other = _make_domain(tmp_path, 99, 5, 2, ["p", "q", "r"], VOCABS[1], 0)       # outros nomes, alvo noutra posição
    out = []
    for csv, owl, target, num, cat in (base, other):
        onto = World().get_ontology(owl.as_uri()).load()
        df, _ = read_tabular_text(csv)
        s = build_schema(df, file_path=csv, source_format="csv", ontology_decl=read_ontology_schema(onto, list(df.columns)))
        out.append((s.target_source, s.class_order_source, sorted(c.kind for c in s.columns if c.name != s.target),
                    len(s.class_order), len(s.columns)))
    assert out[0] == out[1]


def test_vocabulary_is_extensible_without_code_changes(tmp_path):
    """Outra língua/convenção: acrescenta-se um JSON de sinónimos; nenhuma linha de código muda."""
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({"properties": {"column_name": ["Merkmalsname"], "data_type": ["Datentyp"], "class_value": ["Zielwert"]},
                                 "classes": {"attribute": ["Merkmal"], "target_label": ["Zielklasse"]},
                                 "kinds": {"numeric": ["zahl"]}}), encoding="utf-8")
    vocab = load_vocabulary(extra_path=str(extra))
    body = ('<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            'xmlns:owl="http://www.w3.org/2002/07/owl#" xml:base="http://ex.org/o" xmlns="http://ex.org/o#">'
            '<owl:Ontology rdf:about="http://ex.org/o"/><owl:Class rdf:about="#Merkmal"/><owl:Class rdf:about="#Zielklasse"/>'
            '<owl:DatatypeProperty rdf:about="#Merkmalsname"/><owl:DatatypeProperty rdf:about="#Datentyp"/><owl:DatatypeProperty rdf:about="#Zielwert"/>'
            '<rdf:Description rdf:about="#m1"><rdf:type rdf:resource="#Merkmal"/><Merkmalsname>x1</Merkmalsname><Datentyp>Zahl</Datentyp></rdf:Description>'
            '<rdf:Description rdf:about="#k1"><rdf:type rdf:resource="#Zielklasse"/><Zielwert>ja</Zielwert></rdf:Description>'
            '<rdf:Description rdf:about="#k2"><rdf:type rdf:resource="#Zielklasse"/><Zielwert>nein</Zielwert></rdf:Description></rdf:RDF>')
    p = tmp_path / "de.owl"
    p.write_text(body, encoding="utf-8")
    d = read_ontology_schema(World().get_ontology(p.as_uri()).load(), vocabulary=vocab)
    assert d.column_types == {"x1": "numeric"} and d.class_labels == ["ja", "nein"]
    assert read_ontology_schema(World().get_ontology(p.as_uri()).load()).class_labels == []   # sem o vocabulário extra: nada
