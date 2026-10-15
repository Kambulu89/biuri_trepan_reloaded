import hashlib
from pathlib import Path

from validation.benchmark.manifest import manifest_digest
from validation.confirm import build_manifest_v2_draft as b

ROOT = Path(__file__).resolve().parents[1]


def test_draft_is_deterministic_not_frozen_and_hash_matches():
    m1, m2 = b.build(), b.build()
    assert m1 == m2 and m1["frozen"] is False and str(m1["manifest_version"]).endswith("DRAFT")
    assert manifest_digest(m1) == m1["manifest_sha256"]


def test_equivalence_allowlist_empty_and_hierarchy_margins_strict():
    m = b.build()
    assert m["architecture"]["equivalence_allowlist"] == []
    h = m["hierarchy"]
    assert h["delta_accuracy"] == 0.0 and h["delta_fidelity"] == 0.0 and h["nu_fidelity_selection"] == 0.0


def test_confirm_seeds_follow_the_rule_and_do_not_overlap_dev_seeds():
    seeds = b.confirm_seeds()
    assert len(seeds) == len(set(seeds)) == 10
    assert seeds == [int(hashlib.sha256(f"BIURI-CONFIRM-v2/{i}".encode()).hexdigest(), 16) % (2**31 - 1) for i in range(1, 11)]
    assert not set(seeds) & set(b.DEV_SEEDS)


def test_draft_never_lives_where_the_runner_would_treat_it_as_a_frozen_manifest():
    assert "validation/benchmark" not in b.OUT.as_posix()
    assert not list((ROOT / "validation/benchmark").glob("BENCHMARK_MANIFEST_v2*"))


def test_known_ineligible_and_reserve_ordering_recorded():
    m = b.build()
    assert "statlog_german_credit" in m["datasets_confirm_status"] and "occupancy_detection" in m["datasets_confirm_status"]
    assert m["datasets_confirm_reserves_ordered"][0] == "phishing_websites"
