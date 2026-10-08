import hashlib
import json
import subprocess
from pathlib import Path

from validation.historical_results_registry import STATUS, build, classify

ROOT = Path(__file__).resolve().parents[1]


def test_registry_is_non_destructive_and_hashes_match_tracked_files():
    reg = build(ROOT)
    assert reg["files"]
    for row in reg["files"][:25]:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "results"], cwd=ROOT, capture_output=True, text=True).stdout
    assert changed.strip() == ""


def test_classification_criteria():
    assert classify(Path("x.json"), '{"a": 1}')["classification"] == "not_applicable"
    one = classify(Path("x.json"), '{"dataset_id": "a", "arm": "reloaded_semantic"}')
    assert one["classification"] == "unverified_single_ontology"
    many = classify(Path("x.json"), '{"dataset_id": "a"} {"dataset_id": "b"} semantic')
    assert many["classification"] == "potentially_contaminated_multi_ontology"
    assert STATUS == "NOT_VERIFIED_OWL_ISOLATION"


def test_committed_registry_marks_without_deleting():
    reg = json.loads((ROOT / "docs/evidence/ontology_isolation/historical_results_registry.json").read_text(encoding="utf-8"))
    marked = [r for r in reg["files"] if r["status"] == STATUS]
    assert marked
    for r in marked:
        assert (ROOT / r["path"]).exists()
