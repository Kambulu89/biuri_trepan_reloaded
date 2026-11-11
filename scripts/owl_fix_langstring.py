#!/usr/bin/env python
"""Gera uma CÓPIA corrigida de ontologias RDF/XML que o reasoner OWL DL rejeita como inconsistentes.

Causa tratada (genérica, independente do dataset): literais com ``xml:lang`` em propriedades de dados cujo ``rdfs:range`` é
``xsd:string``. Em OWL 2 um literal com língua é ``rdf:langString`` e não pertence a ``xsd:string`` → ontologia inconsistente.
O script remove SÓ o atributo ``xml:lang`` desses valores (o texto fica igual) e regista cada alteração. Opcionalmente
(``--drop-imports``) remove ``owl:imports`` (útil sem Internet / para ontologias como PROV-O que o BIURI não precisa).
Nunca altera o ficheiro original.

Uso:
    python scripts/owl_fix_langstring.py ontologia.owl [-o saida.owl] [--drop-imports]
    python scripts/owl_fix_langstring.py pasta_com_owls/ -o pasta_saida/ [--drop-imports]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Tuple

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"


def _register_namespaces(text: str) -> None:
    for prefix, uri in re.findall(r'xmlns:([A-Za-z_][\w.-]*)="([^"]+)"', text):
        try:
            ET.register_namespace(prefix, uri)
        except ValueError:
            pass
    m = re.search(r'\sxmlns="([^"]+)"', text)
    if m:
        ET.register_namespace("", m.group(1))


def _iri(tag: str) -> str:
    return tag[1:].replace("}", "") if tag.startswith("{") else tag


def _string_range_properties(root: ET.Element) -> Set[str]:
    """IRIs das propriedades com rdfs:range xsd:string (declaradas por owl:DatatypeProperty ou rdf:Description)."""
    props: Set[str] = set()
    for el in root.iter():
        about = el.get(f"{{{RDF}}}about") or el.get(f"{{{RDF}}}ID")
        if not about:
            continue
        for child in el:
            if child.tag == f"{{{RDFS}}}range" and child.get(f"{{{RDF}}}resource") == XSD_STRING:
                props.add(about)
    return props


def fix_text(text: str, drop_imports: bool = False) -> Tuple[str, List[Dict[str, str]]]:
    _register_namespaces(text)
    root = ET.fromstring(text.encode("utf-8"))
    base = root.get("{http://www.w3.org/XML/1998/namespace}base") or ""
    props = _string_range_properties(root)
    # IRIs completos e fragmentos relativos (#nome) comparam-se pelo fragmento final
    frag = {p.split("#")[-1] for p in props} | {p.split("/")[-1] for p in props}
    changes: List[Dict[str, str]] = []
    for el in root.iter():
        if XML_LANG in el.attrib and el.tag.startswith("{"):
            iri = _iri(el.tag)
            local = iri.split("#")[-1] if "#" in iri else iri.rsplit("/", 1)[-1]
            if iri in props or local in frag:
                changes.append({"propriedade": local, "lang": el.attrib[XML_LANG], "valor": (el.text or "")[:60]})
                del el.attrib[XML_LANG]
    if drop_imports:
        for parent in root.iter():
            for child in list(parent):
                if child.tag == f"{{{OWL}}}imports":
                    changes.append({"propriedade": "owl:imports", "lang": "", "valor": child.get(f"{{{RDF}}}resource", "")})
                    parent.remove(child)
    out = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + out + "\n", changes


def fix_file(src: Path, dst: Path, drop_imports: bool = False) -> List[Dict[str, str]]:
    if src.resolve() == dst.resolve():
        raise ValueError("O destino não pode ser o ficheiro original (nunca se altera o original).")
    new, changes = fix_text(src.read_text(encoding="utf-8"), drop_imports)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new, encoding="utf-8")
    return changes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", help="ficheiro .owl/.rdf ou pasta")
    ap.add_argument("-o", "--saida", help="ficheiro ou pasta de saída (por omissão: <nome>_biuri.owl ao lado do original)")
    ap.add_argument("--drop-imports", action="store_true", help="remove owl:imports")
    a = ap.parse_args(argv)
    src = Path(a.entrada)
    jobs = []
    if src.is_dir():
        out_dir = Path(a.saida) if a.saida else src / "corrigidas"
        jobs = [(p, out_dir / p.name) for p in sorted(src.glob("*.owl")) + sorted(src.glob("*.rdf"))]
    else:
        dst = Path(a.saida) if a.saida else src.with_name(src.stem + "_biuri" + src.suffix)
        jobs = [(src, dst)]
    report = {}
    for s, d in jobs:
        ch = fix_file(s, d, a.drop_imports)
        report[str(s)] = {"saida": str(d), "alteracoes": len(ch), "detalhe": ch[:20]}
        print(f"{s.name}: {len(ch)} alteração(ões) → {d}")
    (Path(next(iter(report.values()))["saida"]).parent / "owl_fix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8") if report else None
    return 0


if __name__ == "__main__":
    sys.exit(main())
