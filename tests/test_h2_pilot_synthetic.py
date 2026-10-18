"""H2 (Doc. 62 §6): o piloto técnico sintético cumpre as verificações 1–8 e respeita limites declarados ANTES da execução (interrompe se ultrapassados). Sem desempenho."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pytest.importorskip("owlready2")
from validation.benchmark import h2_pilot_synthetic as PL

EV = ROOT / "docs/evidence/b1b4/h2/pilot_synthetic.json"


def test_limits_are_declared_small_and_before_execution():
    assert PL.LIMITS["max_wall_seconds"] <= 300 and PL.LIMITS["max_fits"] <= 300 and PL.LIMITS["max_peak_memory_mb"] <= 1500


def test_pilot_aborts_when_a_limit_is_exceeded(tmp_path):
    r = PL.run_pilot(str(tmp_path / "p.json"), configs=PL.CONFIGS[:1], limits={"max_wall_seconds": 300, "max_fits": 3, "max_peak_memory_mb": 1500})
    assert r["status"] == "ABORTED_LIMIT" and r["aborted_on"] == "max_fits" and r["consumption"]["fits"] <= 4


def test_recorded_pilot_evidence_satisfies_every_check():
    d = json.loads(EV.read_text())
    assert d["status"] == "COMPLETED" and d["consumption"]["wall_seconds"] <= d["limits_declared_before_execution"]["max_wall_seconds"]
    assert d["consumption"]["fits"] <= d["limits_declared_before_execution"]["max_fits"] and d["consumption"]["peak_memory_mb"] <= d["limits_declared_before_execution"]["max_peak_memory_mb"]
    assert len(d["configs"]) == 3 and {c["n_classes"] for c in d["configs"]} >= {2, 3} and len({c["n_features"] for c in d["configs"]}) == 3          # datasets desconhecidos, dims e classes distintas
    for c in d["configs"]:
        k = c["checks"]["1_compile"]
        assert k["ontology_valid"] and k["matching_complete"] and k["true_literal_present"] and k["perm_does_not_express_truth"] and k["neutral_and_flat_have_no_literals"]
        assert k["ambiguous_match_rejects_incomplete_literals"] and c["checks"]["4_ceilings_control"]["tight_ceiling_gives_NOT_COMPARABLE"]
        for run in c["runs"].values():
            assert run["equivalence"]["A_equals_Original"] and run["equivalence"]["valid_ontology_without_active_structure_equals_Original"]       # 7
            assert not any(run["ceilings_exceeded"].values())                                                                                       # 4
            assert run["comparability"]["C vs A_LIMITED"]["status"] == "NOT_COMPARABLE"                                                             # menos vagas ⇒ não atribuível
            t = run["true_literal_trace"]
            assert t["pool_reason_A_PLUS_root"] == "NOT_IN_FEATURE_POOL" and t["pool_reason_C_root"] == "IN_LITERAL_POOL" and t["candidate_evaluations_C"] > 0       # 2, 3
            assert t["ladder"]["BENEFIT_OBSERVED"] == "NOT_ASSESSED" and t["ladder"]["reached"]["LITERAL_EVALUATED"]                                # 6
            for v in run["comparability"].values():
                assert v["status"] in ("CONFIGURATION_COMPARABLE", "NOT_COMPARABLE") and v["realized_resource_match"]["grade"] in ("EXACT", "WITHIN_TOLERANCE", "PARTIAL")
    assert not any(("accuracy" in json.dumps(c).lower() or "fidelity" in json.dumps(c).lower()) for c in d["configs"])                              # nenhuma métrica de desempenho
