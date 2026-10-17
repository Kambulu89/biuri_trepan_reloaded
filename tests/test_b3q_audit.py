"""B3-Q: testes das ferramentas de auditoria (Docs. 45–48) e integridade da evidência B3.3/B3-Q. Só leitura; nada de benchmarks."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from validation.benchmark import b3q_audit as AU
from validation.benchmark import b3q_information_audit as IA
from validation.benchmark import b3q_partition_recoverability as PR
from validation.benchmark.b30_axiom_probe import dprop, klass, write


def test_census_classifies_by_function_and_flags_substantive(tmp_path):
    body = klass("Root") + klass("G1", ["Root"]) + dprop("a", "G1") + dprop("b", "G1") + klass("G2", ["Root"], '<owl:disjointWith rdf:resource="#G1"/>')
    c = AU.census(write(tmp_path, "t", body))
    f = c["by_function"]
    assert f["AGRUPAMENTO"] == 2 and f["TIPAGEM"] == 2 and f["TAXONOMIA"] == 2 and f["DISJUNCAO"] == 1 and f.get("SUBSTANTIVO_RELACIONAL", 0) == 0
    assert c["groups"] == {"G1": ["a", "b"]}
    sub = klass("R") + klass("S", ["R"], '<rdfs:subClassOf><owl:Restriction><owl:onProperty rdf:resource="#a"/><owl:someValuesFrom rdf:resource="#R"/></owl:Restriction></rdfs:subClassOf>') + dprop("a", "R")
    assert AU.census(write(tmp_path, "s", sub))["by_function"]["SUBSTANTIVO_RELACIONAL"] >= 1


def test_information_quantities_match_exact_enumeration_and_doc_numbers():
    q = IA.quantities(8, 6)
    assert abs(q["H_partition_bits"] - 111.72) < 0.01 and abs(q["H_support_bits"] - 36.02) < 0.01 and abs(q["I_support_partition_bits"] - 31.21) < 0.01
    q2 = IA.quantities(16, 6)
    assert abs(q2["H_partition_bits"] - 302.16) < 0.01 and abs(q2["I_support_partition_bits"] - 42.24) < 0.01
    import itertools, math, collections
    p, K, g = 8, 4, 2
    def partitions(s):
        if not s:
            yield []
            return
        for c in itertools.combinations(s[1:], g - 1):
            blk = (s[0],) + c
            for r in partitions([x for x in s if x not in blk]):
                yield [blk] + r
    parts = list(partitions(list(range(p))))
    assert len(parts) == math.factorial(p) // (math.factorial(g) ** K * math.factorial(K))
    cnt = collections.Counter(tuple(sorted(A + B)) for P in parts for A, B in itertools.combinations(P, 2))
    assert len(cnt) == math.comb(p, 2 * g) and len(set(cnt.values())) == 1          # S uniforme ⇒ H(S) = log2 C(p, 2g)


def test_partition_is_recoverable_with_strong_loadings_but_not_with_weak_ones():
    from sklearn.metrics import adjusted_rand_score
    def ari(lo, hi, n, reps=12):
        rng = np.random.default_rng(5)
        out = []
        for _ in range(reps):
            X, true = PR.gen(n, 8, 6, rng, lo, hi)
            out.append(adjusted_rand_score(true, PR.recover(X, 8)))
        return float(np.mean(out))
    assert ari(0.60, 0.85, 120) > 0.95            # Doc. 44 v1: partição redundante com os dados
    assert ari(0.25, 0.45, 60) < 0.5              # proposta v2: partição não recuperável a n pequeno


def test_audit_evidence_is_coherent():
    ev = json.loads((ROOT / "docs/evidence/b1b4/b3q/audit_dev_ontologies.json").read_text())
    assert set(ev["ontologies"]) == {"iris", "wine", "breast_cancer", "digits"}
    for did, o in ev["ontologies"].items():
        assert o["data_support"]["split_hash_ok"] is True, did                         # só dados de treino, partição verificada
        assert o["census"]["by_function"].get("SUBSTANTIVO_RELACIONAL", 0) == 0, did
        assert o["fact_registry"]["n_hard_constraints"] == 0, did
    assert ev["ontologies"]["breast_cancer"]["census"]["custom_annotation_properties_used"] == {"measurementFamily": 30, "statisticRole": 30}


def test_b33_freeze_hashes_still_match_and_b34_manifest_not_frozen():
    fz = json.loads((ROOT / "docs/evidence/b1b4/b33/freeze_B33.json").read_text())
    assert fz["B33_STATUS"] == "COMPLETED" and fz["B33_VERDICT"] == "NO_ONTOLOGICAL_ADVANTAGE" and fz["M0_M6_CORE_INTEGRATION"] == "NOT_AUTHORIZED"
    for f, h in fz["sha256_lf"].items():
        assert hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == h, f
    m = json.loads((ROOT / "docs/b1b4/manifest_v3/B34_EXPERIMENT_MANIFEST.DRAFT.json").read_text())
    assert m["frozen"] is False and m["state"] == "B34 = DESIGN_ONLY / NOT_EXECUTED"


def test_b34_v1_preserved_manifests_not_frozen_and_gate_evidence_consistent():
    m2 = json.loads((ROOT / "docs/b1b4/manifest_v3/B34_EXPERIMENT_MANIFEST.v2.DRAFT.json").read_text())
    h2 = json.loads((ROOT / "docs/b1b4/manifest_v3/H2_EXPERIMENT_MANIFEST.DRAFT.json").read_text())
    assert m2["frozen"] is False and h2["frozen"] is False and m2["restrictions"]["campaign"] == "NÃO autorizada" and h2["restrictions"]["M0_M6_CORE_INTEGRATION"] == "NOT_AUTHORIZED"
    doc = (ROOT / "docs/b1b4/44_PROTOCOLO_B34_CONHECIMENTO_PREVIO.md").read_text()
    body = doc.split("\n---\n\n## Histórico de versões")[0]
    assert hashlib.sha256(body.rstrip("\n").encode()).hexdigest() == m2["preserves"]["v1_body_sha256_rstrip"]      # v1 preservada integralmente
    rec = json.loads((ROOT / "docs/evidence/b1b4/b3q/recoverability_protocol.json").read_text())
    assert rec["labels_used"] is False and rec["test_data_used"] is False
    lrn = json.loads((ROOT / "docs/evidence/b1b4/b3q/learnability.json").read_text())["results"]
    assert lrn["xor_lam0.25-0.45_n120"]["mlp_accuracy_median"] <= lrn["xor_lam0.25-0.45_n120"]["majority_rate_median"] + 0.01   # XOR não aprendido com cargas fracas
    add = json.loads((ROOT / "docs/evidence/b1b4/b3q/additive_gate.json").read_text())["results"]
    w = add["s2.0_n60"]
    assert w["mlp_acc_median"] - w["majority_median"] >= 0.05 and max(w["screen_hit_rate_corr"], w["screen_hit_rate_L1"]) <= 0.9    # janela G1∧G2 da família aditiva
    r = add["s2.0_n240"]
    assert max(r["screen_hit_rate_corr"], r["screen_hit_rate_L1"]) > 0.9                                                     # redundante para os genéricos em n=240
