"""B3.2 (Doc. 38): comparação CONTROLADA «orientação por ontologia» vs «pesquisa genérica com recursos equivalentes», só ``validation/`` e dados sintéticos.

Pergunta: a ontologia verdadeira dá alguma vantagem que não se obtenha simplesmente alargando o espaço de pesquisa do Original (WIDE) ou gastando o mesmo orçamento de forma
genérica? O Original canónico NÃO é modificado; WIDE e H* são variantes experimentais de referência.

Braços (E1/E2): A ORIG (orçamento histórico limitado `mf`), B WIDE (`mf = p`), C OWL (ontologia verdadeira), D PERM, E RAND, F1 CONV (agrupamento por correlação), F2 SCREEN
(rastreio univariado: mesmos nº/tamanho de unidades, escolhidas pelos dados), G NEUTRALIZED (ontologia verdadeira com as relações relevantes neutralizadas), H1 (WIDE com `beam_width`
calibrado para igualar candidatos AVALIADOS ao braço C) e H2 (idem para igualar CPU de ajuste), mais negativos IRRELEVANT/INCOMPLETE/WRONG_SPLIT e, em P2, HIERARCHY.
Recursos medidos por ajuste: literais listados, candidatos gerados (contadores do Original), candidatos efetivamente avaliados (nº de chamadas a ``_split_selection_score`` e linhas
avaliadas), consultas ao oráculo (``membership_queries_``) e amostras consultadas (linhas passadas ao oráculo), CPU de ajuste (``process_time``), tempo de parede da compilação
(reasoner Java), memória (tracemalloc, em subconjunto de sementes), nós internos, folhas, profundidade, literais por regra, fidelidade ao MESMO MLP, accuracy face aos rótulos reais,
precision/macro-F1 complementares e recuperação exata da regra. Referências: MLP (accuracy) e C4.5 canónico (rótulos reais)."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.c45_j48_tree import C45Classifier
from core.trepan_original import TrepanOriginalClassifier
from validation.benchmark import b31_identifiability as B31
from validation.benchmark.b31_facts import compile_registry
from validation.benchmark.b31_identifiability import ProblemSpec, _cls, _owl, make_instance, true_groups
from validation.benchmark.b31_relatedness import SemanticUnit, semantic_units
from validation.benchmark.b31_steering import SemanticSteeredTrepan

ROOT = Path(__file__).resolve().parents[2]
FROZEN32 = ROOT / "docs/evidence/b1b4/b32/b32_frozen.json"
CODE32 = ("validation/benchmark/b31_facts.py", "validation/benchmark/b31_relatedness.py", "validation/benchmark/b31_steering.py", "validation/benchmark/b31_identifiability.py",
          "validation/benchmark/b32_comparison.py", "validation/benchmark/b32_run.py")
CAL32 = tuple(range(2000, 2030))
EVAL32 = tuple(range(200, 240))
STAB32 = tuple(range(300, 320))
MEM32 = tuple(range(200, 210))


# ------------------------------------------------------------------------------------------------------ instrumentação (sem efeito nas decisões)
class CountingOracle:
    def __init__(self, base):
        self.base, self.calls, self.rows = base, 0, 0

    def predict(self, X):
        self.calls += 1
        self.rows += len(X)
        return self.base.predict(X)

    def predict_proba(self, X):
        self.calls += 1
        self.rows += len(X)
        return self.base.predict_proba(X)


class _Count:
    """Mixin de contagem: não altera pontuações nem escolhas (verificado: mesma árvore que a classe sem contagem)."""

    def fit(self, *a, **k):
        self._cnt = defaultdict(float)
        return super().fit(*a, **k)

    def _split_selection_score(self, y, mask, test=None):
        self._cnt["eval_calls"] += 1
        self._cnt["eval_rows"] += len(y)
        return super()._split_selection_score(y, mask, test)

    def _candidate_literals(self, X, y):
        r = super()._candidate_literals(X, y)
        self._cnt["literal_calls"] += 1
        self._cnt["literals_listed"] += len(r)
        return r


class EfficientSteered(SemanticSteeredTrepan):
    """Como ``SemanticSteeredTrepan(pool_parity=True)`` (mesma árvore, testado) mas sem avaliar literais de atributos fora da unidade: o orçamento de literais do feixe
    mantém-se (``max_features_per_node`` do Original) e só a GERAÇÃO de literais se restringe à unidade. Evita CPU desperdiçada que penalizaria injustamente a orientação."""
    _mf_override: Optional[int] = None

    @property
    def max_features_per_node(self):
        return self._mf_override if self._mf_override is not None else self.__dict__.get("_mf_base", 12)

    @max_features_per_node.setter
    def max_features_per_node(self, v):
        self.__dict__["_mf_base"] = v

    def _candidate_literals(self, X, y):
        if self._restrict is None:
            return TrepanOriginalClassifier._candidate_literals(self, X, y)
        keep = {int(i) for i in self._restrict}
        self._mf_override = len(keep)
        try:
            ranked = TrepanOriginalClassifier._candidate_literals(self, X, y)
        finally:
            self._mf_override = None
        return [(s, l) for s, l in ranked if int(l.feature) in keep]


class CountOrig(_Count, TrepanOriginalClassifier):
    pass


class CountSteered(_Count, EfficientSteered):
    pass


# ------------------------------------------------------------------------------------------------------ instâncias com MLP e rótulos reais
def make_instance_mlp(spec: ProblemSpec, seed: int) -> Dict[str, Any]:
    """Instância B3.1 + MLP (sklearn, com normalização) treinado nos rótulos reais da regra; rótulos reais de teste = regra; oráculo do TREPAN = MLP."""
    inst = make_instance(spec, seed)
    yr = inst["oracle"].predict(inst["X"])
    yt = inst["oracle"].predict(inst["Xtest"])
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=seed))
    mlp.fit(inst["X"], yr)
    inst.update(y_train=yr, y_test=yt, mlp=mlp, mlp_test_pred=mlp.predict(inst["Xtest"]))
    inst["mlp_accuracy_test"] = float(np.mean(inst["mlp_test_pred"] == yt))
    return inst


# ------------------------------------------------------------------------------------------------------ unidades dos braços
def screening_units(X, names: Sequence[str], oracle_pred, group_size: int, n_units: int) -> List[SemanticUnit]:
    """F2: rastreio univariado GENÉRICO (sem ontologia): ordena os atributos pelo melhor ganho de informação de um limiar isolado face às previsões do oráculo
    (nunca rótulos de teste) e forma ``n_units`` unidades consecutivas de ``group_size``. Mesmo nº e tamanho de unidades que o braço C."""
    y = np.asarray(oracle_pred)
    def H(v):
        p = np.bincount(v, minlength=2) / max(len(v), 1)
        return -sum(q * np.log2(q) for q in p if q > 0)
    base = H(y)
    sc = []
    for j in range(X.shape[1]):
        best = 0.0
        for q in np.quantile(X[:, j], np.linspace(0.1, 0.9, 9)):
            m = X[:, j] > q
            if 0 < m.sum() < len(m):
                best = max(best, base - (m.mean() * H(y[m]) + (1 - m.mean()) * H(y[~m])))
        sc.append(best)
    order = np.argsort(-np.asarray(sc), kind="stable")
    units = []
    for u in range(n_units):
        mem = tuple(sorted(names[i] for i in order[u * group_size:(u + 1) * group_size]))
        if len(mem) >= 2:
            units.append(SemanticUnit(f"screen{u}", mem, 1.0, ("DATA",), (), 1.0))
    return units


def build_ontology32(variant: str, inst: Dict[str, Any], seed: int, tmp: Path) -> Optional[Path]:
    if variant == "NEUTRALIZED":                                 # G: ontologia verdadeira; as relações (domínio) dos atributos da regra removidas, tudo o resto igual
        tg = true_groups(inst)
        rn = set(inst["rule_names"])
        groups = {k: [f for f in v if f not in rn] for k, v in tg.items()}
        groups = {k: v for k, v in groups.items() if v}
        body_extra = ""
        from validation.benchmark.b30_axiom_probe import dprop
        for f in inst["rule_names"]:
            body_extra += dprop(f, None, label=f)
        return _owl(tmp, f"NEUT_{seed}", groups, extra=body_extra)
    return B31.build_ontology(variant, inst, seed, tmp)


ARM_TO_VARIANT = {"C_OWL": "TRUE", "D_PERM": "PERM", "E_RAND": "RAND", "G_NEUTRALIZED": "NEUTRALIZED", "N_IRRELEVANT": "IRRELEVANT", "N_INCOMPLETE": "INCOMPLETE",
                  "N_WRONG_SPLIT": "WRONG_SPLIT", "C_HIERARCHY": "HIERARCHY"}


def fit_arm(spec: ProblemSpec, inst: Dict[str, Any], arm: str, seed: int, tmp: Path, *, bw_h1: int = 2, bw_h2: int = 2, measure_memory: bool = False, unit_budget: int = 3) -> Dict[str, Any]:
    kw = dict(min_sample=spec.min_sample, max_queries=spec.max_queries, max_nodes=spec.max_nodes, max_depth=spec.max_depth, max_n=spec.max_n, beam_width=spec.beam_width,
              min_samples_leaf=spec.min_samples_leaf, max_features_per_node=spec.max_features_per_node, random_state=seed)
    p = inst["X"].shape[1]
    base_oracle = inst.get("mlp") or inst["oracle"]
    orc = CountingOracle(base_oracle)
    units: List[SemanticUnit] = []
    t_compile = 0.0
    reg = None
    if arm == "A_ORIG":
        model = CountOrig(**kw)
    elif arm == "B_WIDE":
        model = CountOrig(**dict(kw, max_features_per_node=p))
    elif arm == "H1_WIDE_BEAM_CAND":
        model = CountOrig(**dict(kw, max_features_per_node=p, beam_width=bw_h1))
    elif arm == "H2_WIDE_BEAM_CPU":
        model = CountOrig(**dict(kw, max_features_per_node=p, beam_width=bw_h2))
    elif arm == "F1_CONV":
        units = B31.conventional_units(inst["X"], inst["names"], spec.n_groups, spec.group_size)
        model = CountSteered(units=units, unit_budget=unit_budget, pool_parity=True, **kw)
    elif arm == "F2_SCREEN":
        pred = base_oracle.predict(inst["X"])
        units = screening_units(inst["X"], inst["names"], pred, spec.group_size, spec.n_groups)
        model = CountSteered(units=units, unit_budget=unit_budget, pool_parity=True, **kw)
    else:
        t0 = time.perf_counter()
        owl = build_ontology32(ARM_TO_VARIANT[arm], inst, seed, tmp)
        reg = compile_registry(owl, inst["names"])
        units = semantic_units(reg, inst["names"]) if reg.ontology_status == "VALID" else []
        t_compile = time.perf_counter() - t0
        model = CountSteered(units=units, unit_budget=unit_budget, registry=reg, pool_parity=True, **kw)
    if measure_memory:
        tracemalloc.start()
    c0, w0 = time.process_time(), time.perf_counter()
    model.fit(inst["X"], oracle=orc, feature_names=inst["names"])
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    peak = None
    if measure_memory:
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return _metrics(model, orc, inst, arm, seed, cpu, wall, t_compile, peak, len(units), reg)


def _depth(n) -> int:
    return 0 if n.is_leaf else 1 + max(_depth(c) for c in (n.true_child, n.false_child) if c is not None)


def _metrics(model, orc, inst, arm, seed, cpu, wall, t_compile, peak, n_units, reg) -> Dict[str, Any]:
    nodes = list(model.iter_nodes())
    internal = [n for n in nodes if not n.is_leaf and n.test is not None]
    rule = set(inst["rule_idx"])
    recovered = any({int(l.feature) for l in n.test.literals} == rule for n in internal)
    pred = np.asarray(model.predict(inst["Xtest"]))
    mlp_pred = inst["mlp_test_pred"] if "mlp_test_pred" in inst else inst["oracle"].predict(inst["Xtest"])
    yt = inst["y_test"] if "y_test" in inst else inst["oracle"].predict(inst["Xtest"])
    c = getattr(model, "_cnt", {})
    tot = getattr(model, "candidate_totals_", {}) or {}
    try:
        depth = _depth(model.root_)
    except Exception:
        depth = None
    out = {"arm": arm, "seed": seed, "recovered": bool(recovered), "fidelity": float(np.mean(pred == mlp_pred)), "accuracy": float(accuracy_score(yt, pred)),
           "precision_macro": float(precision_score(yt, pred, average="macro", zero_division=0)), "macro_f1": float(f1_score(yt, pred, average="macro", zero_division=0)),
           "internal_nodes": len(internal), "leaves": len(nodes) - len(internal), "depth": depth, "mean_rule_literals": float(np.mean([len(n.test.literals) for n in internal])) if internal else 0.0,
           "mofn_nodes": int(sum(1 for n in internal if len(n.test.literals) > 1)),
           "eval_calls": float(c.get("eval_calls", 0)), "eval_rows": float(c.get("eval_rows", 0)), "literals_listed": float(c.get("literals_listed", 0)),
           "simple_generated": float(tot.get("simple_generated", 0)), "mofn_generated": float(tot.get("mofn_generated", 0)), "mofn_evaluated": float(tot.get("mofn_evaluated", 0)),
           "queries": int(model.membership_queries_), "oracle_calls": int(orc.calls), "oracle_samples": int(orc.rows), "cpu_fit_s": float(cpu), "wall_fit_s": float(wall),
           "compile_wall_s": float(t_compile), "peak_memory_bytes": peak, "n_units": n_units,
           "rules_sha": hashlib.sha256(json.dumps(model.export_rules(), default=str).encode()).hexdigest()[:16],
           "ontology_status": reg.ontology_status if reg is not None else None}
    tr = getattr(model, "semantic_trace_", None)
    out["unit_searches"] = int(sum(len(e["units_searched"]) for e in tr)) if tr else 0
    out["choice_changed"] = int(sum(e["choice_changed"] for e in tr)) if tr else 0
    return out


def reference_models(spec: ProblemSpec, inst: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Referências: MLP (accuracy face aos rótulos reais) e C4.5 canónico treinado nos rótulos reais (fidelidade ao MESMO MLP)."""
    c45 = C45Classifier(confidence_factor=0.25, min_samples_leaf=2, random_state=seed).fit(inst["X"], inst["y_train"])
    pc = np.asarray(c45.predict(inst["Xtest"]))
    return {"arm": "C45", "seed": seed, "accuracy": float(accuracy_score(inst["y_test"], pc)), "fidelity": float(np.mean(pc == inst["mlp_test_pred"])),
            "mlp_accuracy": inst["mlp_accuracy_test"], "precision_macro": float(precision_score(inst["y_test"], pc, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(inst["y_test"], pc, average="macro", zero_division=0))}


def code32() -> Dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in CODE32}
