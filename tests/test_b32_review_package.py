"""B3.2: o pacote de revisão não contém nomes/assinaturas/aprovações inventados; hashes coincidem; estados pendentes explícitos."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pytest.importorskip("owlready2")
from validation.benchmark.b31_protocol import check_package

PKG = ROOT / "docs/evidence/b1b4/b32/review_package"


def test_all_reviews_pending_without_names_or_signatures():
    props = sorted((PKG / "proposals").glob("*/proposal.json"))
    assert len(props) >= 5
    for f in props:
        d = json.loads(f.read_text())
        rv = d["reviews"]
        assert rv["domain_expert"]["status"] == "DOMAIN_REVIEW_PENDING" and rv["ontology_engineer"]["status"] == "ONTOLOGY_REVIEW_PENDING" and rv["methodology"]["status"] == "METHODOLOGY_REVIEW_PENDING"
        assert all(v["reviewer_name"] is None and v["date"] is None and v["signature_ref"] is None for v in rv.values()), f
        res = check_package(f.parent)
        assert res["_summary"]["ontology_declared_independently_validated"] is False


def test_forms_are_blank_and_hashes_match():
    for n in ("DOMAIN_REVIEW", "ONTOLOGY_REVIEW", "METHODOLOGY_REVIEW"):
        txt = (PKG / "review_forms" / f"{n}.md").read_text()
        assert "NÃO preenchido por Claude" in txt and "PENDING" in txt
    h = json.loads((PKG / "hashes.json").read_text())
    for rel, digest in h.items():
        if rel.startswith("_"):
            continue
        p = ROOT / rel
        assert p.exists() and hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest, rel


def test_rejected_and_conditional_axioms_are_not_promoted():
    w = json.loads((PKG / "proposals/wine_phenol_subset_rejected/proposal.json").read_text())["axioms"][0]
    assert w["status_now"].startswith("REJECTED") and w["empirical_check_attributes_only"]["violation_rate_all_178_rows"] > 0.3
    o = json.loads((PKG / "proposals/wdbc_order_relations/proposal.json").read_text())["axioms"]
    assert o and all(a["status_now"] == "HYPOTHESIS_UNTIL_DOMAIN_REVIEW" and a["relation_class"] == "MATHEMATICAL_IDENTITY_CONDITIONAL" for a in o)
    c = json.loads((PKG / "proposals/wdbc_compactness_identity_rejected/proposal.json").read_text())["axioms"][0]
    assert c["status_now"].startswith("REJECTED_AS_IDENTITY")
