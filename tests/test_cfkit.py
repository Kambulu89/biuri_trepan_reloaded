"""Contratos do pipeline contrafactual cfkit: válido, plausível, parco, semântico, reproduzível e auditável."""
from __future__ import annotations

import copy
import json
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from counterfactuals.cfkit import (
    CounterfactualResult, CounterfactualStatus, compare_methods, generate_counterfactual, run_cf_benchmark,
)
from counterfactuals.cfkit import methods as M
from counterfactuals.cfkit.api import (
    CFCache, CFContext, cache_key, candidate_table, explain_text, export_results, resolve_target, select_instances,
)
from counterfactuals.cfkit.metrics import DensityModel, GowerMetric, diversity, drop_near_duplicates, mark_dominance
from counterfactuals.cfkit.rules import ConstraintSet, OntologyConstraintExtractor, custom_rule, incompatible, monotone, requires_co_change
from counterfactuals.cfkit.schema import FeatureSchema

warnings.filterwarnings("ignore")
NAMES = ["age", "children", "income", "sex", "emp_private", "emp_gov", "emp_self"]
GROUPS = {"employment": {"columns": NAMES[4:], "labels": ["private", "government", "self_employed"]}}
CFG = {"categorical_groups": GROUPS, "immutable_features": ["sex"], "direction": {"age": "increase_only"}}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class RuleModel:
    """Modelo com fronteira CONHECIDA (para saber o CF esperado). Classe 1 sse (income>52 e não self_employed) ou age>60."""

    classes_ = np.array([0, 1])
    n_features_in_ = 7

    def score(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        income_route = np.where(X[:, 6] > 0.5, -3.0, (X[:, 2] - 52.0) / 4.0)
        return np.maximum(income_route, (X[:, 0] - 60.0) / 3.0)

    def predict_proba(self, X):
        p = sigmoid(self.score(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.score(X) > 0).astype(int)


class ThreeClassModel:
    classes_ = np.array([0, 1, 2])
    n_features_in_ = 7

    def predict_proba(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        inc = X[:, 2]
        logits = np.column_stack([-(inc - 40) / 3, -np.abs(inc - 52) / 3, (inc - 62) / 3])
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 500
    age = rng.integers(20, 70, n).astype(float)
    kids = rng.integers(0, 4, n).astype(float)
    income = rng.normal(50, 12, n)
    emp = rng.integers(0, 3, n)
    sex = rng.integers(0, 2, n).astype(float)
    return np.column_stack([age, kids, income, sex, np.eye(3)[emp]])


@pytest.fixture(scope="module")
def ctx(data):
    return CFContext.build(RuleModel(), data, NAMES, config=CFG, model_name="MLP Original", dataset_name="demo")


def instance(data, income=40.0, age=40.0, emp=0, sex=1.0, kids=1.0):
    return np.array([age, kids, income, sex] + list(np.eye(3)[emp]))


def check_valid(result, model, ctx, original):
    """Contrato final: TODO CF devolvido passa no modelo e nas constraints hard."""
    for c in result.candidates:
        v = np.asarray(c.vector)
        assert model.predict(v.reshape(1, -1))[0] == result.target_class
        assert c.model_valid and not c.hard_violations
        assert not ctx.constraints.hard_violations(original, v)


# ------------------------------------------------------------------ schema / pré-processamento reversível
def test_schema_infers_kinds_and_roundtrips_one_hot(data):
    s = FeatureSchema.from_data(data, NAMES, CFG)
    kinds = {sp.name: sp.kind for sp in s.specs}
    assert kinds["age"] == "integer" and kinds["children"] == "integer" and kinds["income"] == "continuous" and kinds["sex"] == "binary"
    assert all(kinds[n] == "onehot" for n in NAMES[4:])
    assert [u.name for u in s.units] == ["age", "children", "income", "sex", "employment"]
    human = s.decode(data[0])
    assert human["employment"] in ("private", "government", "self_employed") and isinstance(human["age"], int)
    assert np.allclose(s.encode(human), data[0])
    with pytest.raises(ValueError):
        s.encode({"employment": "inexistente"})


def test_one_hot_never_returns_invalid_combinations(data):
    s = FeatureSchema.from_data(data, NAMES, CFG)
    x = data[0].copy()
    bad = x.copy(); bad[4:] = [1, 1, 1]
    assert any(v["code"] == "ONEHOT" for v in s.violations(x, bad))
    fixed = s.project(x, bad)
    assert fixed[4:].sum() == 1 and not [v for v in s.violations(x, fixed) if v["code"] == "ONEHOT"]
    assert s.decode(bad)["employment"] == "<inválido>"


def test_projection_enforces_integer_binary_direction_immutable(data):
    s = FeatureSchema.from_data(data, NAMES, CFG)
    x = instance(data)
    c = s.project(x, np.array([x[0] - 7.4, 2.37, 61.2, 0.0, 0.2, 0.8, 0.1]))
    assert c[0] >= x[0] and float(c[0]).is_integer()          # increase_only + inteiro
    assert float(c[1]).is_integer()                              # children inteiro
    assert c[3] == x[3]                                          # imutável
    assert s.decode(c)["employment"] == "government"
    assert not s.violations(x, c)


def test_range_priority_and_train_only(data):
    s = FeatureSchema.from_data(data, NAMES, CFG)
    inc = s.specs[s.index["income"]]
    assert inc.range_source == "train" and inc.low == pytest.approx(data[:, 2].min())
    s.set_range("income", 0.0, 500.0, "owl")
    assert (inc.low, inc.high, inc.range_source) == (0.0, 500.0, "owl")
    s.set_range("income", 10.0, 20.0, "train")                   # fonte mais fraca não sobrescreve a OWL
    assert (inc.low, inc.high, inc.range_source) == (0.0, 500.0, "owl")
    cfg = dict(CFG, feature_ranges={"income": (20.0, 90.0)})
    assert FeatureSchema.from_data(data, NAMES, cfg).specs[2].range_source == "config"


def test_x_test_cannot_change_learned_constraints_metric_or_density(data):
    X_train, X_test_a, X_test_b = data[:300], data[300:] * 1.0, data[300:] * 7.0 + 99
    import inspect
    assert "X_test" not in inspect.signature(CFContext.build).parameters and "X_test" not in inspect.signature(FeatureSchema.from_data).parameters
    c1 = CFContext.build(RuleModel(), X_train, NAMES, config=CFG)
    c2 = CFContext.build(RuleModel(), X_train, NAMES, config=CFG)
    _ = (X_test_a, X_test_b)                                       # nunca entram na construção
    assert json.dumps(c1.constraints.schema.to_dict(), sort_keys=True) == json.dumps(c2.constraints.schema.to_dict(), sort_keys=True)
    assert np.array_equal(c1.metric.spans, c2.metric.spans) and np.array_equal(c1.density.loo, c2.density.loo)


def test_reversible_preprocessing_with_real_data_preprocessor():
    from core.data_contract import build_data_contract
    from core.preprocessing import DataPreprocessor
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    n = 240
    df = pd.DataFrame({"age": rng.integers(20, 70, n), "income": rng.normal(50, 10, n),
                       "employment": rng.choice(["private", "government", "self_employed"], n)})
    df["y"] = np.where((df["income"] > 52) | (df["employment"] == "government"), "yes", "no")
    contract = build_data_contract(df, "y")
    pre = DataPreprocessor(contract).fit(df.drop(columns="y"))
    Xe = pre.transform(df.drop(columns="y"))
    names = list(pre.get_feature_names_out())
    schema = FeatureSchema.from_preprocessor(pre, Xe)
    assert "employment" in [u.name for u in schema.units]
    unit = [u for u in schema.units if u.name == "employment"][0]
    assert set(unit.categories) == {"private", "government", "self_employed"}
    human = schema.decode(Xe[0])
    assert human["employment"] == df.loc[0, "employment"]
    model = LogisticRegression(max_iter=500).fit(Xe, (df["y"] == "yes").astype(int))
    ctx = CFContext.build(model, Xe, names, constraints=ConstraintSet(schema), model_name="LogReg", dataset_name="adult-like")
    i = int(np.where(model.predict(Xe) == 0)[0][0])
    r = generate_counterfactual(Xe[i], 1, model, "COGS", context=ctx, random_state=0)
    assert r.status == CounterfactualStatus.SUCCESS
    for c in r.candidates:
        assert model.predict(np.asarray(c.vector).reshape(1, -1))[0] == 1
        emp_change = [ch for ch in c.changes if ch["feature"] == "employment"]
        for ch in emp_change:
            assert ch["original"] in unit.categories and ch["counterfactual"] in unit.categories   # categoria legível, não x_17
        assert not ctx.constraints.hard_violations(Xe[i], np.asarray(c.vector))


# ------------------------------------------------------------------ alvo / estados
def test_binary_opposite_and_explicit_target(data, ctx):
    model = RuleModel()
    x = instance(data)
    r = generate_counterfactual(x, None, model, "COGS", context=ctx, random_state=1)
    assert r.status == CounterfactualStatus.SUCCESS and r.target_class == 1 and r.original_class == 0
    assert generate_counterfactual(x, "opposite", model, "COGS", context=ctx, random_state=1).target_class == 1


def test_multiclass_requires_explicit_target_and_lists_options(data):
    model = ThreeClassModel()
    c3 = CFContext.build(model, data, NAMES, config=CFG, model_name="MLP 3 classes")
    x = instance(data, income=40.0)
    r = generate_counterfactual(x, None, model, "COGS", context=c3, random_state=0)
    assert r.status == CounterfactualStatus.INVALID_TARGET and "multiclasse" in r.message and r.diagnostics["valid_targets"] == ["1", "2"]
    for tgt in (1, 2):
        ok = generate_counterfactual(x, tgt, model, "COGS", context=c3, random_state=0)
        assert ok.status == CounterfactualStatus.SUCCESS and ok.target_class == tgt
        check_valid(ok, model, c3, x)
    assert generate_counterfactual(x, 9, model, "COGS", context=c3).status == CounterfactualStatus.INVALID_TARGET
    assert generate_counterfactual(x, 0, model, "COGS", context=c3).status == CounterfactualStatus.INVALID_TARGET   # já é a classe 0


def test_resolve_target_unit():
    classes = np.array(["B", "M"])
    assert resolve_target(classes, "B", None)[0] == "M"
    assert resolve_target(np.array([0, 1, 2]), 0, None)[0] is None
    assert resolve_target(None, 0, None)[1]


def test_every_non_success_status_has_message_and_is_never_none(data, ctx):
    model = RuleModel()
    x = instance(data)
    short = generate_counterfactual(x[:3], 1, model, "COGS", context=ctx)
    assert short.status == CounterfactualStatus.UNSUPPORTED_FEATURE_SPACE and short.message
    nan = x.copy(); nan[2] = np.nan
    assert generate_counterfactual(nan, 1, model, "COGS", context=ctx).status == CounterfactualStatus.UNSUPPORTED_FEATURE_SPACE
    assert generate_counterfactual(x, 1, model, "NOPE", context=ctx).status == CounterfactualStatus.METHOD_FAILURE
    frozen_cfg = dict(CFG, immutable_features=NAMES)
    frozen = CFContext.build(model, data, NAMES, config=frozen_cfg)
    r = generate_counterfactual(x, 1, model, "COGS", context=frozen)
    assert r.status == CounterfactualStatus.CONSTRAINT_INFEASIBLE and "mutável" in r.message

    class Const(RuleModel):
        def score(self, X):
            return -np.ones(len(np.atleast_2d(X)))
    const = Const()
    c0 = CFContext.build(const, data, NAMES, config=CFG)
    r = generate_counterfactual(x, 1, const, "CLEAR", context=c0, max_iterations=3)
    assert r.status == CounterfactualStatus.NO_COUNTERFACTUAL_FOUND and r.message and r.candidates == []
    for res in (short, r):
        assert res.explanation and res.provenance["pipeline_version"]


def test_method_failure_is_captured_not_raised(data, ctx, monkeypatch):
    monkeypatch.setattr(M, "cogs_inspired", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("explodiu")))
    r = generate_counterfactual(instance(data), 1, RuleModel(), "COGS", context=ctx, random_state=0)
    assert r.status == CounterfactualStatus.METHOD_FAILURE and "explodiu" in r.message


# ------------------------------------------------------------------ validade final, hard constraints, imutáveis, direção
@pytest.mark.parametrize("method", ["LORE", "CLEAR", "COGS"])
def test_every_success_is_validated_on_model_and_respects_hard_constraints(data, ctx, method):
    model = RuleModel()
    for inst in (instance(data, income=40, emp=0), instance(data, income=42, emp=1, age=35), instance(data, income=38, emp=2, age=45)):
        r = generate_counterfactual(inst, 1, model, method, context=ctx, random_state=3, n_cfs=3)
        assert r.status in (CounterfactualStatus.SUCCESS, CounterfactualStatus.NO_COUNTERFACTUAL_FOUND)
        check_valid(r, model, ctx, inst)
        for c in r.candidates:
            v = np.asarray(c.vector)
            assert v[3] == inst[3]                                  # sex imutável
            assert v[0] >= inst[0] and float(v[0]).is_integer() and float(v[1]).is_integer()   # age increase_only, inteiros
            assert v[4:].sum() == 1                                  # one-hot válido
            assert c.semantic_status == "NOT_AVAILABLE" and c.semantic_valid is None
    assert generate_counterfactual(instance(data), 1, model, method, context=ctx, random_state=3).status == CounterfactualStatus.SUCCESS


def test_known_expected_cf_income_route(data, ctx):
    """income é a rota mais barata: CF esperado muda SÓ income para pouco acima de 52 (alvo conhecido)."""
    model = RuleModel()
    x = instance(data, income=40.0, emp=0, age=40)
    for method in ("CLEAR", "COGS", "LORE"):
        r = generate_counterfactual(x, 1, model, method, context=ctx, random_state=0, n_cfs=3)
        assert r.status == CounterfactualStatus.SUCCESS, method
        best = r.best
        assert best.changes[0]["feature"] == "income" or best.sparsity == 1
        assert best.sparsity <= 2
        assert best.human["income"] > 52.0 - 1e-9
        assert best.human["income"] < 52.0 + 25      # não passa muito da fronteira


def test_immutable_employment_forces_alternative_route(data):
    model = RuleModel()
    cfg = dict(CFG, immutable_features=["sex", "employment"] if False else ["sex", "emp_private", "emp_gov", "emp_self"])
    c = CFContext.build(model, data, NAMES, config=cfg)
    x = instance(data, income=40.0, emp=2, age=40)         # self_employed: rota de rendimento bloqueada
    r = generate_counterfactual(x, 1, model, "COGS", context=c, random_state=0)
    assert r.status == CounterfactualStatus.SUCCESS
    assert r.best.human["age"] > 60 and not any(ch["feature"] == "employment" for ch in r.best.changes)


def test_directional_decrease_only_and_range(data):
    model = RuleModel()
    cfg = dict(CFG, direction={"age": "increase_only", "income": "decrease_only"})
    c = CFContext.build(model, data, NAMES, config=cfg)
    x = instance(data, income=40.0, emp=0, age=40)
    r = generate_counterfactual(x, 1, model, "COGS", context=c, random_state=0)
    for cand in r.candidates:
        assert cand.human["income"] <= 40.0 + 1e-9 and cand.human["age"] >= 40
    # range: a OWL/config restringe o rendimento máximo abaixo da fronteira -> rota de rendimento inviável
    cfg2 = dict(CFG, feature_ranges={"income": (0.0, 51.0)})
    c2 = CFContext.build(model, data, NAMES, config=cfg2)
    r2 = generate_counterfactual(x, 1, model, "COGS", context=c2, random_state=0)
    for cand in r2.candidates:
        assert cand.human["income"] <= 51.0 + 1e-9


def test_nonactionable_policy_invalid_vs_warn(data):
    model = RuleModel()
    base = dict(CFG, actionable_features=["age", "children", "sex"])
    x = instance(data, income=40.0, emp=0, age=40)
    strict = CFContext.build(model, data, NAMES, config=dict(base, nonactionable_policy="invalid"))
    r = generate_counterfactual(x, 1, model, "COGS", context=strict, random_state=0)
    for c in r.candidates:
        assert c.human["income"] == pytest.approx(40.0)         # income não accionável não é tocada
    warn = CFContext.build(model, data, NAMES, config=dict(base, nonactionable_policy="warn"))
    r2 = generate_counterfactual(x, 1, model, "COGS", context=warn, random_state=0)
    assert r2.status == CounterfactualStatus.SUCCESS
    assert any(c.soft_warnings for c in r2.candidates if any(ch["feature"] == "income" for ch in c.changes)) or all(
        not any(ch["feature"] == "income" for ch in c.changes) for c in r2.candidates)


def test_feature_costs_default_neutral_and_influence_distance(data):
    model = RuleModel()
    neutral = CFContext.build(model, data, NAMES, config=CFG)
    assert all(sp.cost == 1.0 for sp in neutral.constraints.schema.specs)
    costly = CFContext.build(model, data, NAMES, config=dict(CFG, costs={"income": 5.0}))
    x = instance(data, income=40.0)
    a = neutral.metric.distance(x, x + np.array([0, 0, 12, 0, 0, 0, 0]), weighted=True)
    b = costly.metric.distance(x, x + np.array([0, 0, 12, 0, 0, 0, 0]), weighted=True)
    assert b > a
    with pytest.raises(ValueError):
        CFContext.build(model, data, NAMES, config=dict(CFG, costs={"income": -1}))


# ------------------------------------------------------------------ semântica: hard vs soft, OWL
def test_model_valid_but_semantically_invalid_is_not_returned_as_success(data):
    model = RuleModel()
    rules = [incompatible([("income", ">", 52.0)], severity="hard", source="ontology")]   # nenhuma rota de rendimento é semanticamente possível
    cs = ConstraintSet.from_config(data, NAMES, dict(CFG, immutable_features=["sex", "age"]))
    cs.rules.extend(rules)
    cs.semantic_source = "ontology"
    c = CFContext.build(model, data, NAMES, constraints=cs, model_name="MLP")
    x = instance(data, income=40.0, emp=0)
    r = generate_counterfactual(x, 1, model, "COGS", context=c, random_state=0)
    assert r.status == CounterfactualStatus.CONSTRAINT_INFEASIBLE and r.candidates == []
    assert r.diagnostics["rejected_hard_constraints"] > 0 and r.rejected and "incompatível" in " ".join(r.rejected[0]["violations"])
    # a violação é detectada separadamente da validade no modelo
    cf = x.copy(); cf[2] = 60.0
    assert model.predict(cf.reshape(1, -1))[0] == 1
    sem = cs.semantic_validation(x, cf)
    assert sem["status"] == "INVALID" and sem["valid"] is False


def test_soft_constraints_penalise_and_warn(data):
    model = RuleModel()
    cs = ConstraintSet.from_config(data, NAMES, CFG)
    cs.rules.append(monotone("income", "children", "same", severity="soft", src="ontology", weight=2.0))
    cs.semantic_source = "ontology"
    c = CFContext.build(model, data, NAMES, constraints=cs)
    x = instance(data, income=40.0, emp=0)
    r = generate_counterfactual(x, 1, model, "COGS", context=c, random_state=0)
    assert r.status == CounterfactualStatus.SUCCESS and r.semantic_validation in ("VALID", "VALID_WITH_WARNINGS")
    soft_cf = x.copy(); soft_cf[2] = 60.0; soft_cf[1] = 0.0
    sem = cs.semantic_validation(x, soft_cf)
    assert sem["valid"] is True and sem["status"] == "VALID_WITH_WARNINGS" and sem["soft_penalty"] >= 2.0


def test_dependency_rule_requires_co_change(data):
    model = RuleModel()
    cs = ConstraintSet.from_config(data, NAMES, CFG)
    cs.rules.append(requires_co_change("income", "children", severity="hard", source="ontology"))
    cs.semantic_source = "ontology"
    x = instance(data, income=40.0)
    only_income = x.copy(); only_income[2] = 60
    both = only_income.copy(); both[1] = 2
    assert cs.hard_violations(x, only_income) and not [v for v in cs.hard_violations(x, both) if v["code"].startswith("RULE")]
    c = CFContext.build(model, data, NAMES, constraints=cs)
    r = generate_counterfactual(x, 1, model, "COGS", context=c, random_state=0)
    for cand in r.candidates:
        feats = {ch["feature"] for ch in cand.changes}
        assert "income" not in feats or "children" in feats


def test_owl_extraction_and_application(data):
    owl = pytest.importorskip("owlready2")
    w = owl.World()
    onto = w.get_ontology("http://example.org/cf#")
    with onto:
        class cf_dependsOn(owl.AnnotationProperty): pass
        class cf_immutable(owl.AnnotationProperty): pass
        class cf_direction(owl.AnnotationProperty): pass
        class hasIncome(owl.DataProperty): range = [owl.ConstrainedDatatype(float, min_inclusive=0.0, max_inclusive=500.0)]
        class hasChildren(owl.DataProperty): range = [int]
        class hasSex(owl.DataProperty): range = [bool]
    hasIncome.cf_dependsOn = ["children"]
    hasSex.cf_immutable = [True]
    hasChildren.cf_direction = ["increase_only"]
    ex = OntologyConstraintExtractor(onto, NAMES).extract()
    assert ex.ranges["income"] == (0.0, 500.0) and "children" in ex.integer and "sex" in ex.binary and "sex" in ex.immutable
    assert ex.directions["children"] == "increase_only" and [r.id for r in ex.rules] == ["dep:income->children"]
    assert any("age" in n for n in ex.not_extracted) and any("AllDisjoint" in n for n in ex.not_extracted)   # não inventa o que não existe
    cs = ConstraintSet.from_config(data, NAMES, CFG, ontology_constraints=ex)
    assert cs.schema.specs[cs.schema.index["income"]].range_source == "owl" and cs.semantic_source == "ontology"
    assert cs.schema.specs[cs.schema.index["children"]].direction == "increase_only"
    assert cs.semantic_available and cs.provenance()["rules"][0]["source"] == "ontology"
    # sem ontologia: nada extraído e NOT_AVAILABLE
    none = OntologyConstraintExtractor(None, NAMES).extract()
    assert not none.ranges and not none.rules
    plain = ConstraintSet.from_config(data, NAMES, CFG)
    assert plain.semantic_validation(data[0], data[0])["status"] == "NOT_AVAILABLE"


def test_owl_range_ignored_when_train_does_not_comply(data):
    from counterfactuals.cfkit.rules import ExtractedConstraints
    ex = ExtractedConstraints(ranges={"income": (0.0, 1.0)})            # unidades da OWL incompatíveis com o espaço do modelo
    cs = ConstraintSet.from_config(data, NAMES, CFG, ontology_constraints=ex)
    assert cs.schema.specs[2].range_source == "train" and ex.skipped_by_unit_check == ["income"]


def test_semantic_available_even_if_mlp_enrichment_rejected(data):
    """Constraints OWL (plausibilidade) são independentes do enriquecimento preditivo do MLP (Parte 35)."""
    cs = ConstraintSet.from_config(data, NAMES, CFG)
    cs.rules.append(requires_co_change("income", "children", source="ontology"))
    cs.semantic_source = "ontology"
    c = CFContext.build(RuleModel(), data, NAMES, constraints=cs, model_name="MLP Original", ontology_hash="abc123")
    r = generate_counterfactual(instance(data), 1, RuleModel(), "COGS", context=c, random_state=0)
    assert r.provenance["ontology_hash"] == "abc123" and r.semantic_validation != "NOT_AVAILABLE"


# ------------------------------------------------------------------ métricas
def test_proximity_gower_sparsity_counts_one_hot_as_one(data, ctx):
    s, metric = ctx.constraints.schema, ctx.metric
    x = instance(data, income=40.0, emp=0)
    y = x.copy(); y[2] = 50.0
    z = x.copy(); z[4:] = [0, 1, 0]
    assert metric.distance(x, x) == 0
    span = data[:, 2].max() - data[:, 2].min()
    assert metric.distance(x, y) == pytest.approx((10.0 / span) / 5)             # 5 unidades humanas
    assert metric.distance(x, z) == pytest.approx(1.0 / 5)                      # mudar categoria = 1 unidade
    assert metric.l1_changes(x, z) == 1 and len(s.changed_units(x, z)) == 1       # grupo one-hot = 1 feature


def test_plausibility_uses_training_density(data, ctx):
    typical = data[10]
    outlier = typical.copy(); outlier[2] = 500.0; outlier[0] = 69.0
    a, b = ctx.density.score(typical), ctx.density.score(outlier)
    assert a["available"] and a["plausibility"] > 0.2 and a["plausible"] is True
    assert b["plausible"] is False and b["plausibility"] < a["plausibility"]
    tiny = DensityModel(ctx.metric).fit(data[:2])
    assert tiny.score(typical)["available"] is False


def test_sparsity_diversity_duplicates_dominance(data, ctx):
    x = instance(data, income=40.0)
    a = x.copy(); a[2] = 55.0
    a2 = a.copy(); a2[2] = 55.0004
    b = x.copy(); b[2] = 60.0; b[1] = 3.0
    items = [{"vector": v, "model_valid": True, "semantic_valid": None, "sparsity": ctx.metric.l1_changes(x, v), "proximity": ctx.metric.distance(x, v)} for v in (a, a2, b)]
    kept, removed = drop_near_duplicates(ctx.metric, items, tol=1e-3)
    assert removed == 1 and len(kept) == 2
    mark_dominance(kept)
    assert kept[1]["dominated_by"] == [0] and kept[0]["dominated_by"] == []        # a: menos features e mais perto
    d = diversity(ctx.metric, [a, b])
    assert d["mean_pairwise"] > 0 and diversity(ctx.metric, [a])["mean_pairwise"] is None


def test_probabilities_recorded_when_available(data, ctx):
    r = generate_counterfactual(instance(data), 1, RuleModel(), "CLEAR", context=ctx, random_state=0)
    c = r.best
    assert c.original_probability > 0.5 and c.counterfactual_probability > 0.5 > c.original_target_probability   # P(orig|x), P(alvo|cf), P(alvo|x)


# ------------------------------------------------------------------ reprodutibilidade / orçamento
@pytest.mark.parametrize("method", ["LORE", "CLEAR", "COGS"])
def test_same_seed_same_result(data, ctx, method):
    x = instance(data)
    r1 = generate_counterfactual(x, 1, RuleModel(), method, context=ctx, random_state=7, max_iterations=15)
    r2 = generate_counterfactual(x, 1, RuleModel(), method, context=ctx, random_state=7, max_iterations=15)
    assert r1.seed == 7 == r2.seed and r1.status == r2.status
    assert [c.vector for c in r1.candidates] == [c.vector for c in r2.candidates]


def test_iteration_and_time_budgets_terminate_and_are_recorded(data, ctx):
    x = instance(data)
    r = generate_counterfactual(x, 1, RuleModel(), "COGS", context=ctx, random_state=0, max_iterations=2)
    assert r.terminated_by == "max_iterations" and r.iterations <= 2 + 1

    class Slow(RuleModel):
        def predict(self, X):
            time.sleep(0.004)
            return super().predict(X)
        def predict_proba(self, X):
            time.sleep(0.004)
            return super().predict_proba(X)

    class Never(Slow):
        def score(self, X):
            return -np.ones(len(np.atleast_2d(X)))

    slow = Slow()
    c = CFContext.build(RuleModel(), data, NAMES, config=CFG)
    t0 = time.perf_counter()
    r2 = generate_counterfactual(x, 1, slow, "COGS", context=c, random_state=0, max_iterations=10 ** 6, max_time=0.3)
    assert time.perf_counter() - t0 < 8.0 and r2.terminated_by in ("max_time", "found_enough")
    never = Never()
    c2 = CFContext.build(RuleModel(), data, NAMES, config=CFG)
    r3 = generate_counterfactual(x, 1, never, "COGS", context=c2, random_state=0, max_iterations=10 ** 6, max_time=0.3)
    assert r3.status == CounterfactualStatus.TIMEOUT and r3.terminated_by == "max_time" and "tempo" in r3.message


# ------------------------------------------------------------------ árvores: caminho, m-of-n, CF modelo vs surrogate
class Oracle23:
    classes_ = np.array([0, 1])
    n_features_in_ = 6

    def predict(self, X):
        X = np.asarray(X)
        return ((X[:, 0] > 0).astype(int) + (X[:, 1] > 0) + (X[:, 2] > 0) >= 2).astype(int)


@pytest.fixture(scope="module")
def mofn_setup():
    from core.trepan_original import TrepanOriginalClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    names = [f"f{i}" for i in range(6)]
    tree = TrepanOriginalClassifier(min_sample=500, max_queries=60000, max_nodes=7, random_state=7).fit(X, oracle=Oracle23(), feature_names=names)
    assert tree.root_.test.m == 2 and len(tree.root_.test.literals) == 3
    ctx = CFContext.build(tree, X, names, model_name="TREPAN Original", dataset_name="synthetic")
    return tree, X, names, ctx


def test_tree_cf_respects_m_of_n_and_path(mofn_setup):
    tree, X, names, ctx = mofn_setup
    base = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0])           # 0 votos: precisa de 2 literais (m-of-n 2-de-3)
    assert tree.predict(base.reshape(1, -1))[0] == 0
    r = generate_counterfactual(base, 1, tree, "TREE", context=ctx, random_state=0, n_cfs=3)
    assert r.status == CounterfactualStatus.SUCCESS
    best = r.best
    assert best.metadata["has_m_of_n"] and best.metadata["path_respected"]
    assert best.sparsity == 2                                     # 2 literais mudam: NÃO foi convertido num limiar único
    assert tree.predict(np.asarray(best.vector).reshape(1, -1))[0] == 1
    one_vote = np.array([1.0, -1.0, -1.0, 0, 0, 0])
    r1 = generate_counterfactual(one_vote, 1, tree, "TREE", context=ctx, random_state=0)
    assert r1.best.sparsity == 1                                  # falta 1 voto: só um literal muda
    for c in r.candidates + r1.candidates:
        assert c.metadata["tree_family"] == "TrepanOriginalClassifier"


def test_c45_and_sklearn_tree_path(mofn_setup):
    from core.c45_j48_tree import C45Classifier
    tree, X, names, _ = mofn_setup
    y = Oracle23().predict(X)
    c45 = C45Classifier(min_samples_split=2, min_samples_leaf=2).fit(X, y)
    c = CFContext.build(c45, X, names, model_name="C4.5-Nativo")
    i = int(np.where(c45.predict(X) == 0)[0][0])
    r = generate_counterfactual(X[i], 1, c45, "TREE", context=c, random_state=0)
    assert r.status == CounterfactualStatus.SUCCESS and c45.predict(np.asarray(r.best.vector).reshape(1, -1))[0] == 1
    from sklearn.tree import DecisionTreeClassifier
    sk = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    c2 = CFContext.build(sk, X, names, model_name="CART (referência)")
    j = int(np.where(sk.predict(X) == 0)[0][0])
    r2 = generate_counterfactual(X[j], 1, sk, "TREE", context=c2, random_state=0)
    assert r2.status == CounterfactualStatus.SUCCESS and sk.predict(np.asarray(r2.best.vector).reshape(1, -1))[0] == 1


def test_counterfactual_for_model_not_for_surrogate(mofn_setup):
    """Um CF que muda a árvore mas não o MLP não é válido para o MLP: a validade é sempre no modelo explicado."""
    tree, X, names, _ = mofn_setup

    class Linear:
        classes_ = np.array([0, 1])
        n_features_in_ = 6
        def predict(self, A):
            A = np.atleast_2d(A)
            return (A[:, 0] + A[:, 1] + A[:, 2] > 1.5).astype(int)        # fronteira ≠ a da árvore (2-de-3)
    model = Linear()
    ctx = CFContext.build(model, X, names, model_name="MLP Original", dataset_name="synthetic")
    base = np.array([-1.0, -1.0, -1.0, 0, 0, 0])
    r = generate_counterfactual(base, 1, model, "TREE", context=ctx, tree_model=tree, random_state=0, n_cfs=3)
    for c in r.candidates:
        assert model.predict(np.asarray(c.vector).reshape(1, -1))[0] == 1       # válido no MLP, não só na árvore
    assert r.model_explained == "MLP Original"
    if r.status != CounterfactualStatus.SUCCESS:
        assert r.diagnostics["rejected_model_invalid"] >= 1


def test_cross_model_verification_and_local_agreement(mofn_setup):
    tree, X, names, ctx = mofn_setup

    class Linear:
        classes_ = np.array([0, 1]); n_features_in_ = 6
        def predict(self, A):
            A = np.atleast_2d(A); return (A[:, 0] + A[:, 1] + A[:, 2] > 0.5).astype(int)
    base = np.array([-1.0, -1.0, -1.0, 0, 0, 0])
    r = generate_counterfactual(base, 1, tree, "TREE", context=ctx, random_state=0, other_models={"MLP": Linear(), "C4.5": tree}, local_agreement_samples=50)
    cm = r.best.cross_model
    assert set(cm) == {"MLP", "C4.5"} and cm["C4.5"]["target_achieved"] is True and 0.0 <= cm["MLP"]["local_agreement"] <= 1.0
    assert "target_achieved" in cm["MLP"]                           # não exige consenso


# ------------------------------------------------------------------ LORE / CLEAR / CoGS (casos sintéticos com resposta conhecida)
class TwoD:
    classes_ = np.array([0, 1]); n_features_in_ = 2
    def predict_proba(self, X):
        X = np.atleast_2d(X); p = sigmoid(10 * (X[:, 0] + X[:, 1] - 1.0)); return np.column_stack([1 - p, p])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


@pytest.fixture(scope="module")
def plane():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(400, 2))
    model = TwoD()
    ctx = CFContext.build(model, X, ["a", "b"], model_name="plano", dataset_name="synthetic")
    return model, X, ctx


def test_clear_on_linear_boundary_is_near_minimal(plane):
    model, X, ctx = plane
    x = np.array([0.2, 0.2])                                      # fronteira a+b=1; mínimo L2 ≈ 0.424
    r = generate_counterfactual(x, 1, model, "CLEAR", context=ctx, random_state=0, n_cfs=3)
    assert r.status == CounterfactualStatus.SUCCESS and r.method_label == "CLEAR-inspired" and not r.canonical
    best = np.asarray(r.best.vector)
    assert model.predict(best.reshape(1, -1))[0] == 1 and best.sum() < 1.0 + 0.45          # junto da fronteira
    assert r.diagnostics["method_stats"]["local_r2"] > 0.4


def test_lore_extracts_rule_and_changes_only_relevant_feature():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, size=(400, 3))

    class OneFeature:
        classes_ = np.array([0, 1]); n_features_in_ = 3
        def predict(self, A):
            return (np.atleast_2d(A)[:, 0] > 0.5).astype(int)
    model = OneFeature()
    ctx = CFContext.build(model, X, ["relevant", "noise1", "noise2"], model_name="regra", dataset_name="synthetic")
    x = np.array([0.1, 0.5, 0.5])
    r = generate_counterfactual(x, 1, model, "LORE", context=ctx, random_state=0, n_cfs=3)
    assert r.status == CounterfactualStatus.SUCCESS and r.method_label == "LORE-inspired"
    assert r.best.sparsity == 1 and r.best.changes[0]["feature"] == "relevant" and r.best.human["relevant"] > 0.5
    st = r.diagnostics["method_stats"]
    assert st["local_fidelity"] > 0.9 and "relevant" in st["factual_rule"] and st["neighborhood"] > 100
    assert r.best.rule and "relevant" in r.best.rule


def test_cogs_is_sparse_and_finds_known_cf(plane):
    model, X, ctx = plane
    x = np.array([0.1, 0.1])
    r = generate_counterfactual(x, 1, model, "COGS", context=ctx, random_state=0, n_cfs=3)
    assert r.status == CounterfactualStatus.SUCCESS and r.method_label == "CoGS-inspired"
    assert r.best.sparsity >= 1 and model.predict(np.asarray(r.best.vector).reshape(1, -1))[0] == 1
    assert r.diversity["mean_pairwise"] is None or r.diversity["mean_pairwise"] >= 0


def test_method_registry_never_claims_canonical():
    for key, info in M.METHOD_REGISTRY.items():
        assert info["canonical"] is False and info["divergences"], key
    assert M.METHOD_REGISTRY["LORE"]["label"].endswith("inspired") and M.METHOD_REGISTRY["CLEAR"]["label"].endswith("inspired") and M.METHOD_REGISTRY["COGS"]["label"].endswith("inspired")


# ------------------------------------------------------------------ comparação, texto, tabela, cache, export, benchmark
def test_compare_methods_same_instance_without_hidden_ranking(data, ctx):
    out = compare_methods(instance(data), 1, RuleModel(), ["LORE", "CLEAR", "COGS"], context=ctx, random_state=0)
    assert [row["method"] for row in out["table"]] == ["LORE-inspired", "CLEAR-inspired", "CoGS-inspired"]
    assert {"proximity", "sparsity", "plausibility", "semantic_validation", "runtime", "success"} <= set(out["table"][0])
    assert "rank" not in out["table"][0] and "score" not in out["table"][0]


def test_explanation_is_deterministic_and_never_causal(data, ctx):
    r = generate_counterfactual(instance(data), 1, RuleModel(), "CLEAR", context=ctx, random_state=0)
    t1, t2 = explain_text(r), explain_text(r)
    assert t1 == t2 and "passaria a prever" in t1 and "MLP" not in t1.replace("Para o modelo", "") or "passaria a prever" in t1
    low = t1.lower()
    assert "na realidade" not in low and "se fizer" not in low and "causal" in low          # só o aviso de que NÃO é causal
    for ch in r.best.changes:
        assert ch["feature"] in t1
    assert explain_text(r, "es") != t1


def test_candidate_table_changed_only_by_default(data, ctx):
    r = generate_counterfactual(instance(data), 1, RuleModel(), "CLEAR", context=ctx, random_state=0)
    only = candidate_table(r)
    allrows = candidate_table(r, show_all=True)
    assert all(row["changed"] for row in only) and len(only) == len(r.best.changes)
    assert len(allrows) == len(r.original_human) > len(only)


def test_cache_key_covers_all_inputs_and_hits(data, ctx):
    x = instance(data)
    base = cache_key(ctx, x, 1, "COGS", 0, {"n_cfs": 3})
    assert base == cache_key(ctx, x, 1, "COGS", 0, {"n_cfs": 3})
    assert cache_key(ctx, x, 1, "COGS", 1, {"n_cfs": 3}) != base                   # seed
    assert cache_key(ctx, x, 0, "COGS", 0, {"n_cfs": 3}) != base                   # target
    assert cache_key(ctx, x, 1, "CLEAR", 0, {"n_cfs": 3}) != base                  # método
    assert cache_key(ctx, x + 1e-3, 1, "COGS", 0, {"n_cfs": 3}) != base            # instância
    assert cache_key(ctx, x, 1, "COGS", 0, {"n_cfs": 5}) != base                   # opções
    other = copy.copy(ctx); other.constraints = ConstraintSet.from_config(ctx.X_train, NAMES, dict(CFG, immutable_features=["sex", "age"]))
    assert cache_key(other, x, 1, "COGS", 0, {"n_cfs": 3}) != base                 # constraints
    other2 = copy.copy(ctx); other2.ontology_hash = "xyz"
    assert cache_key(other2, x, 1, "COGS", 0, {"n_cfs": 3}) != base                # ontologia
    other3 = copy.copy(ctx); other3.fingerprint = "different-model"
    assert cache_key(other3, x, 1, "COGS", 0, {"n_cfs": 3}) != base                # modelo
    cache = CFCache()
    a = generate_counterfactual(x, 1, RuleModel(), "CLEAR", context=ctx, random_state=0, cache=cache)
    b = generate_counterfactual(x, 1, RuleModel(), "CLEAR", context=ctx, random_state=0, cache=cache)
    assert a.provenance["cache"] == "miss" and b.provenance["cache"] == "hit" and cache.hits == 1
    assert [c.vector for c in a.candidates] == [c.vector for c in b.candidates]


def test_export_csv_json_markdown_with_provenance(data, ctx, tmp_path):
    results = [generate_counterfactual(instance(data), 1, RuleModel(), m, context=ctx, random_state=0, instance_id=5) for m in ("CLEAR", "COGS")]
    results.append(generate_counterfactual(instance(data), None, ThreeClassModel() if False else RuleModel(), "NOPE", context=ctx, instance_id=6))
    files = export_results(results, tmp_path, extra={"dataset": "demo"})
    df = pd.read_csv(files["csv"])
    assert {"instance_id", "method", "status", "proximity", "sparsity", "plausibility", "model_valid", "semantic_status", "changes", "seed"} <= set(df.columns)
    assert set(df["status"]) >= {"SUCCESS", "METHOD_FAILURE"}
    rep = json.loads(open(files["json"], encoding="utf-8").read())
    first = rep["results"][0]
    for key in ("model_fingerprint", "constraints", "objective_weights", "pipeline_version", "causality", "method_registry"):
        assert key in first["provenance"]
    assert "NOT_CLAIMED" in first["provenance"]["causality"] and first["provenance"]["ranges_learned_from"] == "X_train only"
    assert "## CLEAR-inspired" in open(files["markdown"], encoding="utf-8").read()


def test_benchmark_protocol_and_aggregates(data, ctx):
    model = RuleModel()
    X_eval = data[:80]
    a = run_cf_benchmark(model, X_eval, ctx, ["CLEAR", "COGS"], protocol="random", n_instances=4, seed=5, max_iterations=10)
    b = run_cf_benchmark(model, X_eval, ctx, ["LORE", "COGS"], protocol="random", n_instances=4, seed=5, max_iterations=10)
    assert a["selection"]["indices"] == b["selection"]["indices"]                     # selecção independente dos métodos/sucesso
    s = a["summary"]["CLEAR-inspired"]
    assert {"success_rate", "mean_proximity", "mean_sparsity", "mean_plausibility", "semantic_validity_rate", "mean_runtime", "status_counts"} <= set(s)
    assert 0.0 <= s["success_rate"] <= 1.0 and s["n_tasks"] >= 1
    assert select_instances(model.predict(X_eval), "stratified", 6, 1)["protocol"] == "stratified"
    assert len(select_instances(model.predict(X_eval), "all")["indices"]) == 80
    # X de teste só fornece instâncias: constraints idênticas independentemente delas
    c2 = CFContext.build(model, ctx.X_train, NAMES, config=CFG)
    assert json.dumps(c2.constraints.provenance(), sort_keys=True, default=str) == json.dumps(ctx.constraints.provenance(), sort_keys=True, default=str)


def test_benchmark_multiclass_one_task_per_target(data):
    model = ThreeClassModel()
    c3 = CFContext.build(model, data, NAMES, config=CFG, model_name="MLP 3")
    out = run_cf_benchmark(model, data[:30], c3, ["COGS"], protocol="indices", indices=[0, 1], seed=0, max_iterations=8)
    assert len(out["rows"]) == 4 and {r["target"] for r in out["rows"]} <= {"0", "1", "2"}      # 2 alvos por instância; nunca "oposta"


def test_cfkit_has_no_dataset_specific_code():
    import re, pathlib
    pat = re.compile(r"\b(iris|wine|wdbc|sonar|german|hepatitis|adult|breast|cancer|digits|diabetes|titanic|mnist)\b", re.I)
    bad = [p.name for p in pathlib.Path("counterfactuals/cfkit").glob("*.py") if pat.search(re.sub(r"#.*", "", p.read_text(encoding="utf-8")))]
    assert not bad, bad
