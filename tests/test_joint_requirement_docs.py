"""O requisito conjunto (accuracy e fidelidade; Original e C4.5; accuracy != precision) está declarado no contrato do projeto e nos critérios de aceitação."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_claude_md_states_the_joint_requirement_and_metric_distinction():
    t = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for needle in ("hierarquia conjunta", "Reloaded+OWL > C4.5", "accuracy ≠ precision", "precision_macro_real_labels", "TRADEOFF", "28_REQUISITO_HIERARQUIA_CONJUNTA"):
        assert needle in t, needle


def test_manifest_v3_draft_acceptance_criteria_are_joint_and_not_frozen():
    m = json.loads((ROOT / "docs/b1b4/manifest_v3/BENCHMARK_MANIFEST_v3.DRAFT.json").read_text(encoding="utf-8"))
    ac = m["acceptance_criteria"]
    assert m["frozen"] is False and ac["tradeoff_is_not_success"] is True
    assert "Fidelity_R_OWL > Fidelity_C45" in ac["hierarchy"] and "Accuracy_R_OWL > Accuracy_C45" in ac["hierarchy"]
    assert "precision_macro_real_labels" in ac["complementary_metrics"] and "precision" not in ac["primary_predictive_metric"].split("!=")[0]


def test_reference_documents_exist():
    for f in ("28_REQUISITO_HIERARQUIA_CONJUNTA.md", "05_CONTRATO_SUPERIORIDADE.md", "02_PROTOCOLO_EXPERIMENTAL.md"):
        assert (ROOT / "docs/b1b4" / f).exists()
    assert "Revisão 2" in (ROOT / "docs/b1b4/05_CONTRATO_SUPERIORIDADE.md").read_text(encoding="utf-8")
