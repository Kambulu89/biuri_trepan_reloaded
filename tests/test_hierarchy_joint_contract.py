"""Contrato da hierarquia em dois níveis: validação matemática exaustiva de todos os estados, não-inferioridade vs superioridade estrita, atribuição à ontologia,
comparabilidade (mesmas instâncias/mesmo MLP) e evidência estatística (unidade = dataset)."""
import itertools

import pytest

from core import hierarchy_contract as H


def g(aR, aO, aC, fR, fO, fC, **k):
    return H.evaluate_global(aR, aO, aC, fR, fO, fC, **k)


# ------------------------------------------------------------------------------- Nível 2: exaustivo
def _reference_state(aR, aO, aC, fR, fO, fC):
    """Implementação independente (definição direta do enunciado) para comparar com a função, em todos os 3^6 = 729 padrões de ordem."""
    u1, u2, u3, u4, u5 = aR > aO, aO > aC, aR > aC, fR > fO, fR > fC
    if u1 and u2 and u3 and u4 and u5:
        return H.JOINT_HIERARCHY_SATISFIED
    if not u2:
        return H.HIERARCHY_INFEASIBLE
    tr = (aR > aO and fR < fO) or (fR > fO and aR < aO) or (aR > aC and fR < fC) or (fR > fC and aR < aC)
    return H.TRADEOFF if tr else H.NOT_SATISFIED


def test_all_729_order_patterns_match_the_independent_definition():
    seen = set()
    for v in itertools.product((0.0, 0.5, 1.0), repeat=6):
        got = g(*v)["state"]
        assert got == _reference_state(*v), v
        seen.add(got)
        assert H.is_scientific_success(g(*v)) == (got == H.JOINT_HIERARCHY_SATISFIED)
    assert seen == {H.JOINT_HIERARCHY_SATISFIED, H.HIERARCHY_INFEASIBLE, H.TRADEOFF, H.NOT_SATISFIED}      # todos os estados lógicos alcançáveis são exercitados


def test_flags_are_orthogonal_to_the_single_state():
    v = g(0.95, 0.90, 0.92, 0.90, 0.93, 0.85)                       # A_O <= A_C (infeasível) E compromisso R vs O
    assert v["state"] == H.HIERARCHY_INFEASIBLE
    f = v["flags"]
    assert f["chain_infeasible_by_baselines"] and f["tradeoff_vs_original"] and not f["fidelity_superiority"]
    assert f["ontological_direct_gain_state"] == H.TRADEOFF


def test_superiority_flags_and_success_requires_everything():
    v = g(0.95, 0.92, 0.90, 0.96, 0.93, 0.91)
    assert v["state"] == H.JOINT_HIERARCHY_SATISFIED and v["flags"]["accuracy_superiority"] and v["flags"]["fidelity_superiority"]
    assert g(0.95, 0.92, 0.90, 0.90, 0.93, 0.85)["state"] == H.TRADEOFF                    # accuracy sobe, fidelidade desce
    assert g(0.90, 0.92, 0.88, 0.97, 0.93, 0.85)["state"] == H.TRADEOFF                    # fidelidade sobe, accuracy desce
    assert g(0.95, 0.94, 0.93, 0.96, 0.93, 0.97)["flags"]["fidelity_superiority"] is False  # não supera o C4.5 em fidelidade
    assert g(0.95, 0.94, 0.93, 0.96, 0.93, 0.97)["state"] == H.TRADEOFF                      # acc↑ e fid↓ face ao C4.5


def test_ties_never_count_and_missing_data_is_insufficient():
    assert g(0.9, 0.9, 0.8, 0.9, 0.9, 0.8)["state"] == H.NOT_SATISFIED
    assert g(0.9, 0.8, 0.8, 0.9, 0.9, 0.8)["state"] == H.HIERARCHY_INFEASIBLE                 # A_O == A_C ⇒ cadeia inviável (estrito)
    assert g(0.9, None, 0.8, 0.9, 0.9, 0.8)["state"] == H.INSUFFICIENT_EVIDENCE
    assert g(float("nan"), 0.9, 0.8, 0.9, 0.9, 0.8)["state"] == H.INSUFFICIENT_EVIDENCE


def test_margins_are_strict_and_pre_registered_only():
    assert g(0.925, 0.92, 0.90, 0.94, 0.93, 0.91, delta_acc=0.01)["state"] == H.NOT_SATISFIED
    assert g(0.94, 0.92, 0.90, 0.95, 0.93, 0.91, delta_acc=0.01)["state"] == H.JOINT_HIERARCHY_SATISFIED


# ------------------------------------------------------------------------------- Nível 1
def test_direct_gain_states_cover_all_sign_patterns():
    for dA, dF in itertools.product((-0.1, 0.0, 0.1), repeat=2):
        v = H.evaluate_direct_gain(0.8 + dA, 0.8, 0.9 + dF, 0.9)
        exp = (H.ONTOLOGICAL_JOINT_GAIN if dA > 0 and dF > 0 else H.TRADEOFF if (dA > 0 and dF < 0) or (dF > 0 and dA < 0) else H.ACCURACY_ONLY_GAIN if dA > 0
               else H.FIDELITY_ONLY_GAIN if dF > 0 else H.EQUIVALENT if dA == 0 and dF == 0 else H.NO_GAIN)
        assert v["state"] == exp, (dA, dF)
        assert v["strict_joint_gain"] == (dA > 0 and dF > 0)


def test_non_inferiority_is_separate_from_strict_superiority():
    eq = H.evaluate_direct_gain(0.9, 0.9, 0.9, 0.9)                 # igualdade: elegível, mas NÃO é superioridade
    assert eq["non_inferior"] is True and eq["strict_joint_gain"] is False and eq["state"] == H.EQUIVALENT
    acc_only = H.evaluate_direct_gain(0.92, 0.9, 0.9, 0.9)         # fidelidade empatada: elegível, mas sem ganho conjunto estrito
    assert acc_only["non_inferior"] and not acc_only["strict_joint_gain"] and acc_only["state"] == H.ACCURACY_ONLY_GAIN
    loss = H.evaluate_direct_gain(0.92, 0.9, 0.89, 0.9)            # perde fidelidade ⇒ não elegível
    assert loss["non_inferior"] is False and loss["state"] == H.TRADEOFF
    tol = H.evaluate_direct_gain(0.92, 0.9, 0.895, 0.9, nu_fid=0.01)   # tolerância pré-registada só afeta a elegibilidade
    assert tol["non_inferior"] is True and tol["strict_joint_gain"] is False


def test_attribution_requires_gain_over_every_required_control_and_never_by_omission():
    d = H.evaluate_direct_gain(0.95, 0.9, 0.96, 0.9)
    full = {"permuted_ontology": (0.01, 0.02), "random_groups": (0.02, 0.01), "conventional_features": (0.01, 0.01)}
    assert H.attribute_to_ontology(d, full)["attribution"] == H.ATTRIBUTED
    assert H.attribute_to_ontology(d, dict(full, conventional_features=(0.0, 0.02)))["attribution"] == H.NOT_ATTRIBUTED
    part = {k: v for k, v in full.items() if k != "random_groups"}
    r = H.attribute_to_ontology(d, part)
    assert r["attribution"] == H.NOT_ASSESSABLE and r["missing_required_controls"] == ["random_groups"]
    no_gain = H.evaluate_direct_gain(0.88, 0.9, 0.96, 0.9)
    assert H.attribute_to_ontology(no_gain, full)["attribution"] == H.NOT_ATTRIBUTED


# ------------------------------------------------------------------------------- mesmas instâncias, mesmo MLP
def test_basic_level_is_legacy_only():
    ok = {"o": {"split_hash": "s", "n_test": 38, "oracle_id": "m"}, "c": {"split_hash": "s", "n_test": 38, "fidelity_oracle_id": "m"}}
    assert H.check_comparability(ok, level="basic")["comparable"]
    assert not H.check_comparability({**ok, "c": {"split_hash": "x", "n_test": 38, "fidelity_oracle_id": "m"}}, level="basic")["comparable"]
    assert not H.check_comparability(ok)["comparable"]                      # o nível estrito NÃO aceita registos mínimos (igualdade de n_test não basta)


def rec(arm_kind="trepan", **kw):
    """Registo bruto mínimo mas completo para o nível estrito."""
    base = {"labels": [0, 1, 2], "manifest_sha256": "mf", "config": {"structural": {"tree_max_nodes": 31.0, "query_budget": 1000.0, "tree_max_n": 3.0}},
            "predictions": {"test_row_index": [1, 4, 7], "y_true": [0, 1, 2], "oracle_predictions": {"mlp": [0, 1, 2]}},
            "row": {"minority_label": "1", "split_id": "seed1", "split_hash": "sh", "n_train": 10, "n_test": 3, "dataset_content_hash": "dc", "dataset_hash": "dh",
                    "base_representation_hash": "br", "preprocessing_id": "pp", "oracle_version": "v1", "oracle_name": "MLP", "config_hash": "ch", "feature_space": "original",
                    "ontology_hash": "none"}}
    if arm_kind == "trepan":
        base["row"]["oracle_id"] = "M"
    else:
        base["row"]["fidelity_oracle_id"] = "M"
    import copy
    r = copy.deepcopy(base)
    for k, v in kw.items():
        node = r
        path = k.split("__")
        for p_ in path[:-1]:
            node = node[p_]
        node[path[-1]] = v
    return r


def test_strict_comparability_accepts_identical_instances_and_oracle_and_documents_representation_differences():
    recs = {"orig": rec(), "owl": rec(row__feature_space="enriched", row__ontology_hash="abc"), "c45": rec("c45")}
    r = H.check_comparability(recs)
    assert r["comparable"] and not r["failed"] and not r["unverifiable"]
    assert r["documented_differences"]["feature_space"] == ["enriched", "original"]           # representação enriquecida = diferença documentada, não oculta


@pytest.mark.parametrize("mutation,prop", [
    ({"predictions__test_row_index": [4, 1, 7]}, "instance_identity_and_order"),            # mesma coleção, ORDEM diferente
    ({"predictions__test_row_index": [1, 4, 8]}, "instance_identity_and_order"),
    ({"predictions__y_true": [0, 1, 1]}, "true_labels"),
    ({"labels": [0, 2, 1]}, "class_mapping_and_order"),                                      # mapeamento/ordem das classes
    ({"row__split_hash": "other"}, "partitioning"),
    ({"row__n_test": 4}, "partitioning"),
    ({"row__dataset_hash": "x"}, "dataset_identity"),
    ({"row__base_representation_hash": "x"}, "base_representation_and_preprocessing"),
    ({"row__preprocessing_id": "x"}, "base_representation_and_preprocessing"),
    ({"row__oracle_id": "OTHER"}, "oracle_identity"),
    ({"predictions__oracle_predictions": {"mlp": [0, 1, 1]}}, "oracle_predictions_on_test"),
    ({"row__oracle_version": "v2"}, "model_version"),
    ({"row__config_hash": "zz"}, "experimental_configuration"),
    ({"manifest_sha256": "other"}, "experimental_configuration"),
    ({"config__structural__query_budget": 2000.0}, "query_budget"),
])
def test_strict_comparability_detects_each_property(mutation, prop):
    recs = {"orig": rec(), "owl": rec(**mutation), "c45": rec("c45")}
    r = H.check_comparability(recs)
    assert not r["comparable"] and prop in r["failed"], (prop, r["failed"])


def test_strict_comparability_missing_information_is_not_comparable_and_same_n_is_not_enough():
    bare = {"o": {"row": {"split_hash": "s", "n_test": 38, "oracle_id": "m"}}, "c": {"row": {"split_hash": "s", "n_test": 38, "fidelity_oracle_id": "m"}}}
    r = H.check_comparability(bare)
    assert not r["comparable"] and "instance_identity_and_order" in r["unverifiable"] and "true_labels" in r["unverifiable"]
    no_preds = {"orig": rec(), "owl": rec(), "c45": rec("c45")}
    del no_preds["owl"]["predictions"]["y_true"]
    assert "true_labels" in H.check_comparability(no_preds)["unverifiable"]


def test_relaxed_properties_are_exempt_and_recorded():
    recs = {"orig": rec(), "owl": rec(row__config_hash="from_other_run"), "c45": rec("c45")}
    assert not H.check_comparability(recs)["comparable"]
    r = H.check_comparability(recs, relaxed=("experimental_configuration",))
    assert r["comparable"] and r["relaxed"] == ["experimental_configuration"]


# ------------------------------------------------------------------------------- evidência: estado lógico separado; unidade = dataset
def _ds(mean_a, mean_f, n_units=5, noise=0.0):
    return {"acc": [mean_a + noise * (i - 2) for i in range(n_units)], "fid": [mean_f + noise * (i - 2) for i in range(n_units)]}


def test_many_seeds_do_not_replace_independent_datasets():
    four = {f"d{i}": _ds(0.02, 0.02, n_units=50, noise=0.001) for i in range(4)}          # 200 unidades, só 4 datasets
    e = H.classify_evidence(four, confirmatory=True, min_datasets=12)
    assert e["n_datasets"] == 4 and e["evidence"] == H.OBSERVED_JOINT_GAIN                  # nunca SUPPORTED
    assert any("datasets independentes" in r for r in e["reasons"])


def test_supported_needs_confirmatory_enough_datasets_and_positive_cis_in_both_metrics():
    many = {f"d{i}": _ds(0.02 + 0.001 * (i % 3), 0.02 + 0.001 * (i % 2)) for i in range(12)}
    assert H.classify_evidence(many, confirmatory=True)["evidence"] == H.SUPPORTED_JOINT_GAIN
    assert H.classify_evidence(many, confirmatory=False)["evidence"] == H.OBSERVED_JOINT_GAIN   # DEV/exploratório nunca é suportado
    noisy = {f"d{i}": _ds((0.05 if i % 2 else -0.04), 0.03) for i in range(12)}
    e = H.classify_evidence(noisy, confirmatory=True)
    assert e["evidence"] in (H.OBSERVED_JOINT_GAIN, H.NOT_SATISFIED, H.TRADEOFF) and e["evidence"] != H.SUPPORTED_JOINT_GAIN


def test_evidence_tradeoff_not_satisfied_and_insufficient():
    assert H.classify_evidence({f"d{i}": _ds(0.02, -0.02) for i in range(4)})["evidence"] == H.TRADEOFF
    assert H.classify_evidence({f"d{i}": _ds(-0.02, -0.02) for i in range(4)})["evidence"] == H.NOT_SATISFIED
    assert H.classify_evidence({"d0": _ds(0.02, 0.02)})["evidence"] == H.INSUFFICIENT_EVIDENCE
    assert H.classify_evidence({})["evidence"] == H.INSUFFICIENT_EVIDENCE


def test_evidence_reports_within_and_between_dataset_variability():
    e = H.classify_evidence({"a": _ds(0.01, 0.02, noise=0.01), "b": _ds(0.03, 0.01, noise=0.02), "c": _ds(0.02, 0.03, noise=0.0)})
    assert e["per_dataset"]["a"]["within_dataset_sd_acc"] > 0 and e["per_dataset"]["c"]["within_dataset_sd_acc"] == 0
    assert e["across_datasets"]["acc"]["t"]["between_dataset_sd"] > 0 and len(e["across_datasets"]["acc"]["bootstrap_ci"]) == 2
