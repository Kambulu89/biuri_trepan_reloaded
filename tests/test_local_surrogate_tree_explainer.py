"""CounterfactualLocalTreeExplainer: um único explicador local para todos os oráculos, com proveniência verificável.

Os oráculos são modelos sklearn controlados (nenhum dataset real): o que se testa é a identidade do conjunto local, a dependência
das previsões/da árvore no oráculo, a ausência de reutilização de objectos e a concordância entre oráculos.
"""
import copy

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

import counterfactuals.local_surrogate_tree as lst
from counterfactuals.local_surrogate_tree import (
    CounterfactualLocalTreeExplainer, LocalTreeLedger, LocalTreeReuseError, collect_oracle_predictions,
    concordance_lines, explain_equality, local_oracle_concordance, summary_lines, tree_title,
)

NAMES = ["a", "b", "c"]


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(260, 3))
    return X


def _tree_oracle(X, column, depth=3, seed=0):
    return DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(X, (X[:, column] > 0).astype(int))


def _generation(X, instance=0):
    candidates = [{"vector": X[i].tolist(), "metrics": {"validity": True, "plausibility": 0.8}} for i in (1, 2, 3, 4)]
    return {"original_instance": X[instance].tolist(), "feature_names": NAMES, "candidates": candidates,
            "method": "controlado", "instance_index": instance}


def _explain(oracle, X, name, ledger, **kw):
    explainer = CounterfactualLocalTreeExplainer(neighborhood_size=120, seed=3, ledger=ledger)
    return explainer.explain(oracle, X, _generation(X), NAMES, oracle_name=name, **kw)


class _Relabel:
    """Outra CLASSE com exactamente as mesmas previsões (rótulos idênticos em todo o espaço)."""

    def __init__(self, inner):
        self._inner = inner

    def predict(self, X):
        return self._inner.predict(X)

    def predict_proba(self, X):
        return self._inner.predict_proba(X)


def test_swapping_the_oracle_recomputes_predictions_and_rebuilds_the_local_tree(data):
    ledger = LocalTreeLedger()
    ra = _explain(_tree_oracle(data, 0), data, "Oráculo A", ledger)
    rb = _explain(_tree_oracle(data, 1), data, "Oráculo B", ledger)
    pa, pb = ra["local_provenance"], rb["local_provenance"]
    assert pa["local_dataset_hash"] == pb["local_dataset_hash"]            # o mesmo conjunto local…
    assert pa["instance_id"] == pb["instance_id"]
    assert pa["oracle_id"] != pb["oracle_id"]
    assert pa["prediction_hash"] != pb["prediction_hash"]                   # …mas previsões recalculadas pelo novo oráculo
    assert pa["tree_signature"] != pb["tree_signature"]                     # e árvore reconstruída (modelos discordam)
    assert ra["_runtime_tree_model"] is not rb["_runtime_tree_model"]
    assert pa["tree_build_token"] != pb["tree_build_token"] and pa["tree_object_id"] != pb["tree_object_id"]
    assert len(ledger) == 2 and pa["ledger_checked"] and pb["ledger_checked"]


def test_each_build_creates_a_new_tree_object_even_for_the_same_oracle(data):
    ledger = LocalTreeLedger()
    oracle = _tree_oracle(data, 0)
    r1, r2 = _explain(oracle, data, "A", ledger), _explain(oracle, data, "A", ledger)
    assert r1["_runtime_tree_model"] is not r2["_runtime_tree_model"]
    assert r1["local_provenance"]["tree_build_token"] != r2["local_provenance"]["tree_build_token"]
    assert r1["local_provenance"]["tree_signature"] == r2["local_provenance"]["tree_signature"]       # determinista
    assert r1["local_provenance"]["prediction_hash"] == r2["local_provenance"]["prediction_hash"]


def test_equal_trees_are_explained_by_equal_labels_and_rules_not_by_object_reuse(data):
    ledger = LocalTreeLedger()
    base = _tree_oracle(data, 0)
    r1 = _explain(base, data, "Oráculo 1", ledger)
    r2 = _explain(_Relabel(base), data, "Oráculo 2 (outra classe, mesmos rótulos)", ledger)
    proof = explain_equality(r1, r2)
    assert proof["verdict"] == "iguais_por_mesmos_rotulos_e_regras_locais"
    assert proof["same_prediction_hash"] and proof["same_local_dataset"] and proof["same_tree_signature"]
    assert proof["distinct_tree_objects"] and r1["_runtime_tree_model"] is not r2["_runtime_tree_model"]
    assert r1["tree_rules"] == r2["tree_rules"]
    # e quando os rótulos diferem, o veredicto nunca é «iguais»
    r3 = _explain(_tree_oracle(data, 2), data, "Oráculo 3", ledger)
    assert explain_equality(r1, r3)["verdict"] == "arvores_diferentes"


def test_ledger_rejects_a_tree_reused_across_oracles(data, monkeypatch):
    ledger = LocalTreeLedger()
    first = _explain(_tree_oracle(data, 0), data, "A", ledger)
    monkeypatch.setattr(lst, "build_local_surrogate_tree", lambda *a, **k: copy.copy(first))      # cache indevida
    with pytest.raises(LocalTreeReuseError):
        CounterfactualLocalTreeExplainer(ledger=ledger).explain(_tree_oracle(data, 1), data, _generation(data), NAMES, oracle_name="B")


def test_seed_does_not_depend_on_the_oracle_so_all_oracles_use_the_same_explainer(data):
    ledger = LocalTreeLedger()
    a = _explain(_tree_oracle(data, 0), data, "A", ledger)
    b = _explain(_tree_oracle(data, 1), data, "B", ledger)
    assert a["local_provenance"]["effective_seed"] == b["local_provenance"]["effective_seed"]
    assert a["explainer"] == b["explainer"] == "CounterfactualLocalTreeExplainer" and a["builder"] == "Local Surrogate m-of-n"


def test_concordance_matrix_total_agreement_and_local_disagreement():
    preds = {"MLP Original": [0, 1, 1, 0, 1], "C4.5-Nativo": [0, 1, 0, 0, 1],
             "Trepan Original": [0, 1, 1, 0, 0], "Trepan Reloaded": [0, 1, 1, 0, 1]}
    c = local_oracle_concordance(preds)
    m = np.asarray(c["pairwise_agreement"])
    assert c["n_points"] == 5 and np.allclose(np.diag(m), 1.0) and np.allclose(m, m.T)
    assert m[0, 1] == pytest.approx(4 / 5) and m[0, 2] == pytest.approx(4 / 5) and m[0, 3] == pytest.approx(1.0)
    assert c["total_agreement"] == pytest.approx(3 / 5)                     # pontos 0, 1 e 3 reúnem os 4 oráculos
    assert c["local_disagreement"] == pytest.approx(2 / 5)
    assert c["total_agreement"] + c["local_disagreement"] == pytest.approx(1.0)
    assert len(set(c["prediction_hash"].values())) == 3                      # 2 oráculos com previsões idênticas partilham hash
    text = "\n".join(concordance_lines(c))
    assert "Concordância total" in text and "60.0%" in text and "Discordância local: 40.0%" in text
    with pytest.raises(ValueError):
        local_oracle_concordance({"a": [0, 1], "b": [0, 1, 1]})


def _session(data, oracles):
    names = NAMES
    return {"dataset_name": "controlado.arff", "mlp_original": oracles["MLP Original"], "mlp_oracle": oracles["MLP Original"],
            "X_train_enc": data, "y_train_enc": (data[:, 0] > 0).astype(int),
            "X_train_original": data, "y_train_original": (data[:, 0] > 0).astype(int),
            "tree_a": oracles["Trepan Original"], "tree_b": oracles["Trepan Reloaded"], "c45_tree": oracles["C4.5-Nativo"],
            "transformed_feature_names": names, "feature_names_original": names, "class_labels": {0: "neg", 1: "pos"}}


def test_service_builds_per_oracle_trees_with_concordance_and_recomputes_after_swapping_the_oracle(data):
    from counterfactuals.service import build_counterfactual_tree_from_session
    oracles = {"MLP Original": _tree_oracle(data, 0), "C4.5-Nativo": _tree_oracle(data, 1),
               "Trepan Original": _tree_oracle(data, 0, depth=2), "Trepan Reloaded": _tree_oracle(data, 2)}
    session = _session(data, oracles)
    opts = {"seed": 3, "cf_tree_neighborhood_size": 120}
    results = {}
    for name in oracles:
        results[name] = build_counterfactual_tree_from_session(
            session, {**_generation(data), "target_model": name}, {**opts, "target_model": name})
    sigs = {n: r["local_provenance"]["tree_signature"] for n, r in results.items()}
    assert len({r["local_provenance"]["local_dataset_hash"] for r in results.values()}) == 1
    assert sigs["C4.5-Nativo"] != sigs["MLP Original"] and sigs["Trepan Reloaded"] != sigs["MLP Original"]
    objs = [id(r["_runtime_tree_model"]) for r in results.values()]
    assert len(set(objs)) == 4
    conc = results["Trepan Original"]["local_concordance"]
    assert set(conc["oracles"]) == set(oracles) and conc["n_points"] == 120
    idx = np.asarray(results["Trepan Original"]["local_provenance"]["neighbor_index"])
    manual = {n: o.predict(data[idx]).astype(str) for n, o in oracles.items()}
    expected = local_oracle_concordance(manual)
    assert conc["pairwise_agreement"] == expected["pairwise_agreement"]
    assert conc["total_agreement"] == expected["total_agreement"] and conc["local_disagreement"] == expected["local_disagreement"]
    assert 0.0 < conc["local_disagreement"] < 1.0                            # os oráculos controlados discordam de facto

    # trocar o objecto-oráculo na sessão recalcula previsões e reconstrói a árvore (nada em cache)
    before = results["Trepan Reloaded"]["local_provenance"]
    session["tree_b"] = _tree_oracle(data, 1)
    again = build_counterfactual_tree_from_session(
        session, {**_generation(data), "target_model": "Trepan Reloaded"}, {**opts, "target_model": "Trepan Reloaded"})
    after = again["local_provenance"]
    assert after["prediction_hash"] != before["prediction_hash"] and after["tree_signature"] != before["tree_signature"]
    assert after["tree_build_token"] != before["tree_build_token"]
    assert after["local_dataset_hash"] == before["local_dataset_hash"]


def test_collect_predictions_excludes_oracles_with_a_different_row_space_instead_of_inventing_alignment(data):
    from counterfactuals.service import _resolve_interactive_context
    oracles = {"MLP Original": _tree_oracle(data, 0), "C4.5-Nativo": _tree_oracle(data, 1),
               "Trepan Original": _tree_oracle(data, 2), "Trepan Reloaded": _tree_oracle(data, 0, depth=1)}
    session = _session(data, oracles)
    session["tree_a"] = None                                                  # Trepan Original indisponível
    out = collect_oracle_predictions(session, {}, list(range(10)), len(data), _resolve_interactive_context)
    assert "Trepan Original" in out["excluded"] and "Trepan Original" not in out["predictions"]
    assert set(out["predictions"]) == {"MLP Original", "C4.5-Nativo", "Trepan Reloaded"}
    out_bad = collect_oracle_predictions(session, {}, list(range(10)), len(data) + 1, _resolve_interactive_context)
    assert not out_bad["predictions"] and len(out_bad["excluded"]) == 4


def test_ui_text_states_oracle_builder_instance_samples_counterfactuals_and_fidelity(data):
    ledger = LocalTreeLedger()
    r = _explain(_tree_oracle(data, 0), data, "Trepan Reloaded", ledger)
    text = "\n".join(summary_lines(r))
    for expected in ("Oráculo: Trepan Reloaded", "Construtor da árvore CF: Local Surrogate m-of-n", "Instância analisada: 0",
                     "Amostras locais: 120", "Contrafactuais válidos: 4", "Fidelidade local ao oráculo:"):
        assert expected in text, expected
    assert r["tree_title"] == tree_title("Trepan Reloaded") == "Árvore CF local — Oráculo: Trepan Reloaded"
    assert "Trepan Reloaded" in r["narrative"] and "não é a árvore interna" in r["narrative"].lower()


def test_gui_titles_the_tree_as_local_cf_tree_with_oracle_and_shows_the_header(qapp, data):
    from gui.biuri_app_complete import BiuriApp
    from gui.counterfactual_panel import CounterfactualPanel
    r = _explain(_tree_oracle(data, 0), data, "C4.5-Nativo", LocalTreeLedger())
    r["local_concordance"] = local_oracle_concordance({"MLP Original": [0, 1, 1], "C4.5-Nativo": [0, 1, 0]})
    w = BiuriApp()
    w.cf_tree_result = r
    w._visualize_cf_tree(r)
    assert w.tree_widget.tree_options[0][0] == "Árvore CF local — Oráculo: C4.5-Nativo"
    from PyQt6.QtWidgets import QLabel
    header = next(c for c in w.visualization_tab.findChildren(QLabel) if c.objectName() == "cf_local_tree_header")
    shown = header.text()
    assert "Oráculo: C4.5-Nativo" in shown and "Local Surrogate m-of-n" in shown and "Discordância local" in shown
    assert "Árvore C4.5" not in shown and "Árvore TREPAN Reloaded" not in shown
    panel = CounterfactualPanel()
    panel.display_result(r)
    summary = panel.summary_text.toPlainText()
    assert "Árvore CF local — Oráculo: C4.5-Nativo" in summary and "Concordância local entre oráculos" in summary
    w.close()
