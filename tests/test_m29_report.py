"""Relatório M29: atribuição exige TODOS os controlos completos (incl. G29) e comparáveis; controlo em falta/parcial/confundido ⇒ NOT_ASSESSABLE; nenhuma unidade excluída; F29 ≡ A29."""
from tests.test_hierarchy_joint_report import rec
from validation.benchmark.m29_report import build


def with_budget(r, n_derived=3, queries=1000.0):
    r["row"]["semantic_features_generated"] = n_derived
    r["row"]["oracle_queries"] = queries
    return r


def unit_sc(aA, fA, aB, fB, aC45=0.80, fC45=0.80):
    return {"c45": rec(aC45, fC45, "c45"), "sc_B_r0": with_budget(rec(aA, fA), 0, 1000.0), "sc_D_r0": with_budget(rec(aB, fB), 3, 2100.0), "sc_F_r0": with_budget(rec(aA, fA), 0, 1000.0)}


def unit_m(ctrl):
    out = {}
    for key, (a, f, variant, nd, q) in ctrl.items():
        r = with_budget(rec(a, f), nd, q)
        if variant:
            r["row"]["control_variant"] = variant
        out[key] = r
    return out


FULL = {"m29_P_k0": (0.88, 0.89, "full", 3, 2100.0), "m29_D_k0": (0.87, 0.88, "full", 3, 2100.0), "m29_E_k0": (0.86, 0.87, None, 3, 2000.0), "m29_G_k0": (0.89, 0.90, None, 0, 2100.0)}


def test_attribution_requires_gain_over_all_complete_controls_including_budget_matched_original():
    sc = {("a", 1): unit_sc(0.90, 0.91, 0.95, 0.96)}
    r = build(sc, {("a", 1): unit_m(FULL)})
    assert r["comparability"]["all_comparable"] and r["equivalence_F29_vs_A29"] == {"identical": 1}
    assert r["level1"]["attribution_counts"] == {"ATTRIBUTED": 1}
    assert r["level2"]["global_state_counts"]["JOINT_HIERARCHY_SATISFIED"]["n"] == 1


def test_missing_partial_or_confounded_control_is_not_assessable_never_favourable():
    sc = {("a", 1): unit_sc(0.90, 0.91, 0.95, 0.96)}
    assert build(sc, {("a", 1): unit_m({k: v for k, v in FULL.items() if k != "m29_D_k0"})})["level1"]["attribution_counts"] == {"NOT_ASSESSABLE": 1}
    assert build(sc, {("a", 1): unit_m({k: v for k, v in FULL.items() if k != "m29_G_k0"})})["level1"]["attribution_counts"] == {"NOT_ASSESSABLE": 1}      # G29 é obrigatório
    fallback = dict(FULL, m29_P_k0=(0.88, 0.89, "structure_only_fallback", 3, 2100.0))
    assert build(sc, {("a", 1): unit_m(fallback)})["level1"]["attribution_counts"] == {"NOT_ASSESSABLE": 1}
    assert build(sc, {})["level1"]["attribution_counts"] == {"NOT_ASSESSABLE": 1}


def test_conventional_control_with_different_dimension_or_queries_is_confounded_and_blocks_conclusive_attribution():
    sc = {("a", 1): unit_sc(0.90, 0.91, 0.95, 0.96)}
    conf = dict(FULL, m29_E_k0=(0.86, 0.87, None, 1, 1000.0))               # 1 derivada e metade das consultas
    r = build(sc, {("a", 1): unit_m(conf)})
    assert r["level1"]["attribution_counts"] == {"NOT_ASSESSABLE": 1}
    assert r["level1"]["attribution_counts_excluding_E29"] == {"ATTRIBUTED": 1}      # estado secundário, não conclusivo
    e = r["per_unit"]["a:1"]["states"]["E29_confounding"]
    assert e["confounded"] and len(e["reasons"]) == 2


def test_control_that_matches_or_beats_the_real_ontology_blocks_attribution():
    sc = {("a", 1): unit_sc(0.90, 0.91, 0.95, 0.96)}
    better = dict(FULL, m29_P_k0=(0.96, 0.97, "full", 3, 2100.0))                # a permutada é MELHOR que a real
    assert build(sc, {("a", 1): unit_m(better)})["level1"]["attribution_counts"] == {"NOT_ATTRIBUTED": 1}
    g_better = dict(FULL, m29_G_k0=(0.96, 0.97, None, 0, 2100.0))                # o Original com orçamento igualado já explica o ganho
    assert build(sc, {("a", 1): unit_m(g_better)})["level1"]["attribution_counts"] == {"NOT_ATTRIBUTED": 1}


def test_no_unit_is_excluded_and_infeasible_units_stay():
    sc = {("a", 1): unit_sc(0.90, 0.91, 0.95, 0.96), ("b", 1): unit_sc(0.70, 0.91, 0.75, 0.96)}      # Original (0.70) < C4.5 (0.80)
    r = build(sc, {})
    assert r["n_units"] == 2 and r["exclusions"] == "nenhuma"
    assert r["level2"]["global_state_counts"]["HIERARCHY_INFEASIBLE"]["pct_of_all_units"] == 50.0
    assert "per_draw" in r["per_unit"]["a:1"]                                         # valores por draw/réplica preservados (nada escondido por agregação)


def test_different_oracle_between_arms_blocks_comparability():
    sc = {("a", 1): unit_sc(0.90, 0.91, 0.95, 0.96)}
    sc[("a", 1)]["sc_D_r0"]["row"]["oracle_id"] = "OTHER"
    assert not build(sc, {})["comparability"]["all_comparable"]
