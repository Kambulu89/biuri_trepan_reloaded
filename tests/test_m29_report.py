"""Relatório M29: atribuição ontológica exige TODOS os controlos completos; controlo em falta/parcial ⇒ NOT_ASSESSABLE; nenhuma unidade excluída; F29 ≡ A29."""
from tests.test_hierarchy_joint_report import rec
from validation.benchmark.m29_report import build


def unit_sc(aA, fA, aB, fB, aC45=0.80, fC45=0.80):
    return {"c45": rec(aC45, fC45, "c45"), "sc_B_r0": rec(aA, fA), "sc_D_r0": rec(aB, fB), "sc_F_r0": rec(aA, fA)}


def unit_m(ctrl):
    out = {}
    for key, (a, f, variant) in ctrl.items():
        r = rec(a, f)
        if variant:
            r["row"]["control_variant"] = variant
        out[key] = r
    return out


FULL = {"m29_P_k0": (0.88, 0.89, "full"), "m29_D_k0": (0.87, 0.88, "full"), "m29_E_k0": (0.86, 0.87, None)}


def test_attribution_requires_gain_over_all_three_complete_controls():
    sc = {("a", 1): unit_sc(0.90, 0.91, 0.95, 0.96)}
    r = build(sc, {("a", 1): unit_m(FULL)})
    assert r["comparability"]["all_comparable"] and r["equivalence_F29_vs_A29"] == {"identical": 1}
    assert r["level1"]["attribution_counts"] == {"ATTRIBUTED": 1}
    assert r["level2"]["global_state_counts"]["JOINT_HIERARCHY_SATISFIED"]["n"] == 1


def test_missing_or_partial_control_is_not_assessable_never_favourable():
    sc = {("a", 1): unit_sc(0.90, 0.91, 0.95, 0.96)}
    missing = {k: v for k, v in FULL.items() if k != "m29_D_k0"}
    assert build(sc, {("a", 1): unit_m(missing)})["level1"]["attribution_counts"] == {"NOT_ASSESSABLE": 1}
    fallback = dict(FULL, m29_P_k0=(0.88, 0.89, "structure_only_fallback"))
    assert build(sc, {("a", 1): unit_m(fallback)})["level1"]["attribution_counts"] == {"NOT_ASSESSABLE": 1}
    assert build(sc, {})["level1"]["attribution_counts"] == {"NOT_ASSESSABLE": 1}


def test_control_that_matches_the_real_ontology_blocks_attribution():
    sc = {("a", 1): unit_sc(0.90, 0.91, 0.95, 0.96)}
    better = dict(FULL, m29_P_k0=(0.96, 0.97, "full"))                 # a permutada é MELHOR que a real
    assert build(sc, {("a", 1): unit_m(better)})["level1"]["attribution_counts"] == {"NOT_ATTRIBUTED": 1}


def test_no_unit_is_excluded_and_infeasible_units_stay():
    sc = {("a", 1): unit_sc(0.90, 0.91, 0.95, 0.96), ("b", 1): unit_sc(0.70, 0.91, 0.75, 0.96)}      # Original (0.70) < C4.5 (0.80)
    r = build(sc, {})
    assert r["n_units"] == 2 and r["exclusions"] == "nenhuma"
    assert r["level2"]["global_state_counts"]["HIERARCHY_INFEASIBLE"]["pct_of_all_units"] == 50.0


def test_different_oracle_between_arms_blocks_comparability():
    sc = {("a", 1): unit_sc(0.90, 0.91, 0.95, 0.96)}
    sc[("a", 1)]["sc_D_r0"]["row"]["oracle_id"] = "OTHER"
    assert not build(sc, {})["comparability"]["all_comparable"]
