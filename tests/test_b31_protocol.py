"""B3.1: protocolo de enriquecimento validado — a máquina nunca declara validação independente; revisores não humanos são rejeitados; pacotes pendentes ficam pendentes."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pytest.importorskip("owlready2")
from validation.benchmark.b30_axiom_probe import dprop, klass, write
from validation.benchmark.b31_protocol import check_package

PROPOSALS = ROOT / "docs/evidence/b1b4/b31/proposals"


def test_committed_proposals_are_pending_and_not_validated():
    for pkg in PROPOSALS.iterdir():
        res = check_package(pkg)
        assert res["_summary"]["ontology_declared_independently_validated"] is False
        assert res["_summary"]["domain_review_status"] == "DOMAIN_REVIEW_PENDING"
        assert "4_domain_review" in res["_summary"]["blocking"]


def _pkg(tmp_path, reviews):
    owl = write(tmp_path, "ontology", klass("Root") + klass("A", ["Root"]) + dprop("temperature_mean", "A") + dprop("temperature_peak", "A"))
    sha = hashlib.sha256(owl.read_bytes()).hexdigest()
    ax = [{"id": "a1", "axiom_text": "temperature_mean Domain A", "type": "DOMAIN", "attributes": ["temperature_mean"], "relation_class": "FORMAL_DOMAIN_CONSTRAINT", "justification": "j",
           "sources": [{"citation": "fonte de teste", "locator": "p.1"}]}]
    (tmp_path / "proposal.json").write_text(json.dumps({"ontology": {"file": "ontology.owl", "sha256": sha}, "axioms": ax, "reviews": reviews,
                                                        "preregistration": {"frozen_before_evaluation": True, "freeze_commit": "abc"},
                                                        "comparison_arms": ["original", "enriched", "permuted", "random", "empty"]}))
    return tmp_path


def _rev(name):
    return {r: {"status": "APPROVED", "reviewer_name": name, "date": "2026-01-01", "signature_ref": "ref"} for r in ("domain_expert", "ontology_engineer", "methodology")}


def test_ai_named_reviewer_cannot_approve(tmp_path):
    res = check_package(_pkg(tmp_path, _rev("Claude")), ["temperature_mean", "temperature_peak"])
    assert res["_summary"]["ontology_declared_independently_validated"] is False
    assert res["4_domain_review"]["detail"]["domain_expert"] == "REJECTED_NON_HUMAN_REVIEWER"


def test_independent_validation_requires_three_named_human_approvals_and_automatic_checks(tmp_path):
    res = check_package(_pkg(tmp_path, _rev("Revisor Humano de Teste")), ["temperature_mean", "temperature_peak"])
    s = res["_summary"]
    assert s["blocking"] == [] and s["ontology_declared_independently_validated"] is True           # só com TODAS as condições (aqui, revisores fictícios de teste)
    r2 = _rev("Revisor Humano de Teste"); r2["methodology"]["status"] = "REVIEW_PENDING"
    assert check_package(_pkg(tmp_path, r2), ["temperature_mean", "temperature_peak"])["_summary"]["ontology_declared_independently_validated"] is False


def test_hash_mismatch_blocks(tmp_path):
    p = _pkg(tmp_path, _rev("Revisor Humano de Teste"))
    d = json.loads((p / "proposal.json").read_text()); d["ontology"]["sha256"] = "0" * 64
    (p / "proposal.json").write_text(json.dumps(d))
    res = check_package(p, ["temperature_mean", "temperature_peak"])
    assert res["9_hashes_and_versions"]["status"] == "FAIL" and not res["_summary"]["ontology_declared_independently_validated"]
