"""Executor M29: sementes pré-definidas, braços (incl. G29), calibração do orçamento de G29 e VERIFICAÇÃO EM EXECUÇÃO da comparabilidade (sem lançar campanhas)."""
import json
import shutil
from pathlib import Path

import pytest

from validation.benchmark import run_controls_m29 as X

REF = X.DEFAULT_SC_DIR


def test_arms_include_all_new_controls_and_g29_is_an_experimental_original():
    arms = {a.arm_id: a for a in X.build_arms(3)}
    assert len(arms) == 12 and {a.split("_")[1] for a in arms} == {"P", "D", "E", "G"}
    g = arms["m29_G_k0"]
    assert g.family == "trepan_original" and g.semantic == "none" and g.tree_space == "orig" and not g.control and "NÃO substitui" in g.description
    assert arms["m29_P_k1"].control and arms["m29_P_k1"].semantic == "shuffled_structure" and arms["m29_E_k2"].semantic == "conventional"


def test_seeds_are_predefined_independent_and_deterministic():
    s = [X.seeds_for("P", k, 42)["seed_a"] for k in range(3)]
    d = [(X.seeds_for("D", k, 42)["seed_a"], X.seeds_for("D", k, 42)["seed_b"]) for k in range(3)]
    assert s == [20_260_042, 20_261_042, 20_262_042] and len({*s, *[x for pair in d for x in pair]}) == 9
    assert X.seeds_for("P", 0, 42) == X.seeds_for("P", 0, 42) and X.seeds_for("P", 0, 7) != X.seeds_for("P", 0, 42)


def test_only_kinds_filters_arms_and_g29_is_single_shot():
    assert {a.arm_id for a in X.build_arms(2, ("G",))} == {"m29_G_k0", "m29_G_k1"}
    src = Path(X.__file__).read_text()
    assert "__cal" not in src and "Alteração A2" in src                          # sem calibração iterativa nem redução abaixo do orçamento canónico


def test_scale_factor_never_reduces_the_canonical_budget_and_reference_queries_come_from_counts_only():
    assert X.scale_factor(2100, 1000) == pytest.approx(2.1) and X.scale_factor(900, 1000) == 1.0
    qb, qa = X.reference_queries(REF, "iris", 42, 0)
    assert qb > qa > 0                                                                   # B29 consome mais consultas do que A29


def _copy_unit(tmp_path, mutate=None, arm="m29_P_k0"):
    src = REF / "raw" / "iris" / "seed42"
    dst = tmp_path / "raw" / "iris" / "seed42"
    dst.mkdir(parents=True)
    rec = json.loads((src / "sc_D_r0.json").read_text())
    rec["arm"] = arm
    if mutate:
        mutate(rec)
    (dst / f"{arm}.json").write_text(json.dumps(rec))
    return tmp_path


def test_runtime_comparability_accepts_identical_instances_and_oracle(tmp_path):
    out = _copy_unit(tmp_path)
    r = X.runtime_comparability(out, REF, "iris", 42)
    assert r["all_comparable"] and r["arms"]["m29_P_k0"]["strict_comparable"]
    assert (tmp_path / "comparability" / "iris_seed42.json").exists()


@pytest.mark.parametrize("mut", [lambda r: r["predictions"].__setitem__("test_row_index", list(reversed(r["predictions"]["test_row_index"]))),
                                 lambda r: r["predictions"].__setitem__("y_true", [0] * len(r["predictions"]["y_true"])),
                                 lambda r: r["row"].__setitem__("oracle_id", "OTHER_MLP"),
                                 lambda r: r["predictions"].__setitem__("oracle_predictions", {k: [0] * len(v) for k, v in r["predictions"]["oracle_predictions"].items()}),
                                 lambda r: r.__setitem__("labels", [0, 2, 1])])
def test_runtime_comparability_blocks_any_identity_mismatch(tmp_path, mut):
    out = _copy_unit(tmp_path, mut)
    assert not X.runtime_comparability(out, REF, "iris", 42)["all_comparable"]
