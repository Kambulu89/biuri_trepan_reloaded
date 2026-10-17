"""B3.3 (Doc. 41): interações SEM sinal univariado — a ontologia dá alguma vantagem quando o rastreio univariado genérico falha? Só ``validation/``, dados sintéticos.

Reutiliza a infraestrutura B3.2 (contadores de recursos, braços, MLP como oráculo) sem a alterar. Acrescenta: (i) geradores de interação (AND univariado de controlo; XOR de um par,
sem sinal univariado; XOR com ruído; XOR hierárquico entre conceitos irmãos; XOR em espaço maior), (ii) rastreio genérico de INTERAÇÃO por pares (braço H), (iii) recuperação
«par no mesmo caminho/teste» (a XOR não é representável por UM único teste m-of-n: precisa de dois níveis; limite de expressividade declarado), (iv) estados de ganho conjunto.
Dificuldade definida SÓ com o Original e recursos, em sementes de calibração disjuntas das de avaliação. Rótulos reais = regra com ruído de rótulo ``eta`` (treino e teste)."""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from validation.benchmark import b32_comparison as B32
from validation.benchmark.b31_identifiability import ProblemSpec, make_instance
from validation.benchmark.b31_relatedness import SemanticUnit

ROOT = Path(__file__).resolve().parents[2]
FROZEN33 = ROOT / "docs/evidence/b1b4/b33/b33_frozen.json"
CODE33 = ("validation/benchmark/b31_facts.py", "validation/benchmark/b31_relatedness.py", "validation/benchmark/b31_steering.py", "validation/benchmark/b31_identifiability.py",
          "validation/benchmark/b32_comparison.py", "validation/benchmark/b33_interactions.py", "validation/benchmark/b33_run.py", "validation/benchmark/b33_report.py")
CAL33 = tuple(range(3000, 3020))      # calibração técnica única (independente de B3.1: 0–39/1000–1029, B3.2: 200–239/300–319/2000–2029)
EVAL33 = tuple(range(400, 430))
DIAG33 = tuple(range(500, 510))       # diagnóstico/memória (não entra nos veredictos)


@dataclass(frozen=True)
class Family:
    key: str
    rule: str                 # "and" | "xor"
    layout: str               # "within" (par num conceito) | "cross" (par entre dois conceitos irmãos)
    eta: float                # ruído de rótulo
    n_groups: int
    group_size: int
    n_train: int
    max_queries: int
    mf_grid: Tuple[int, ...]
    ontology_true_variant: str   # variante que representa a ontologia verdadeira da família: TRUE | HIERARCHY
    q: float = 0.5            # quantil do limiar da regra por atributo (0,5 = sem sinal univariado; >0,5 = sinal univariado crescente)


FAMILIES: Dict[str, Family] = {f.key: f for f in [
    Family("I1_AND_PAIR", "and", "within", 0.0, 8, 4, 1000, 4000, (16, 20, 24, 28, 30), "TRUE", 0.5),
    Family("I2_XOR_NO_SIGNAL", "xor", "within", 0.0, 8, 4, 1000, 4000, (16, 20, 24, 28, 30), "TRUE", 0.5),
    Family("I3_XOR_WEAK_SIGNAL", "xor", "within", 0.0, 8, 4, 1000, 4000, (16, 20, 24, 28, 30), "TRUE", 0.65),
    Family("I4_XOR_WEAK_NOISY", "xor", "within", 0.10, 8, 4, 1000, 4000, (16, 20, 24, 28, 30), "TRUE", 0.65),
    Family("I5_XOR_WEAK_HIERARCHY", "xor", "cross", 0.0, 8, 4, 1000, 4000, (16, 20, 24, 28, 30), "HIERARCHY", 0.65),
    Family("I6_XOR_WEAK_LARGE_BUDGET", "xor", "within", 0.05, 8, 6, 1500, 2500, (24, 30, 36, 40, 44), "TRUE", 0.65),
]}
ARMS33 = ("A_ORIG", "B_WIDE", "C_OWL", "D_PERM", "E_RAND", "F_CONV", "G_UNI_SCREEN", "H_PAIR_SCREEN", "I_NEUTRALIZED", "J1_WIDE_BEAM_CAND", "J2_WIDE_BEAM_CPU", "N_IRRELEVANT", "N_INCOMPLETE")
_ARM_TO32 = {"A_ORIG": "A_ORIG", "B_WIDE": "B_WIDE", "C_OWL": "C_OWL", "D_PERM": "D_PERM", "E_RAND": "E_RAND", "F_CONV": "F1_CONV", "G_UNI_SCREEN": "F2_SCREEN", "H_PAIR_SCREEN": "F2_SCREEN",
             "I_NEUTRALIZED": "G_NEUTRALIZED", "J1_WIDE_BEAM_CAND": "H1_WIDE_BEAM_CAND", "J2_WIDE_BEAM_CPU": "H2_WIDE_BEAM_CPU", "N_IRRELEVANT": "N_IRRELEVANT", "N_INCOMPLETE": "N_INCOMPLETE"}


def spec_of(fam: Family, mf: int) -> ProblemSpec:
    return ProblemSpec(fam.key, n_groups=fam.n_groups, group_size=fam.group_size, rule="within_group" if fam.layout == "within" else "cross_group", n_train=fam.n_train,
                       max_queries=fam.max_queries, max_features_per_node=mf)


def make_instance33(fam: Family, spec: ProblemSpec, seed: int) -> Dict[str, Any]:
    inst = make_instance(spec, seed)
    rng = np.random.default_rng([seed, 33])
    rn = inst["rule_names"]
    pair = [rn[0], rn[1]] if fam.layout == "within" else [rn[0], rn[2]]
    if fam.layout == "within":
        pair = [rn[i] for i in sorted(rng.permutation(len(rn))[:2])]
    idx = [inst["names"].index(n) for n in pair]

    from scipy.stats import norm
    zq = float(norm.ppf(fam.q))
    thr = [float(inst["sd"][i]) * zq for i in idx]                  # limiar de cada atributo no quantil q da sua distribuição (N(0, sd²))

    def rule(Z):
        a, b = np.asarray(Z)[:, idx[0]] > thr[0], np.asarray(Z)[:, idx[1]] > thr[1]
        return (a & b).astype(int) if fam.rule == "and" else (a ^ b).astype(int)

    def flip(y, tag):
        if fam.eta <= 0:
            return y
        m = np.random.default_rng([seed, tag]).random(len(y)) < fam.eta
        return np.where(m, 1 - y, y)

    ytr, yte = flip(rule(inst["X"]), 35), flip(rule(inst["Xtest"]), 36)
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=seed)).fit(inst["X"], ytr)
    inst.update(rule_names=pair, rule_idx=idx, y_train=ytr, y_test=yte, mlp=mlp, mlp_test_pred=mlp.predict(inst["Xtest"]), rule_fn=rule, rule_thr=thr)
    inst["mlp_accuracy_test"] = float(np.mean(inst["mlp_test_pred"] == yte))
    return inst


# ------------------------------------------------------------------------------------------------------ rastreio genérico de INTERAÇÃO por pares (braço H)
def _entropy(y):
    p = np.bincount(y, minlength=2) / max(len(y), 1)
    return -sum(q * np.log2(q) for q in p if q > 0)


def pair_screening_units(X, names: Sequence[str], oracle_pred, group_size: int, n_units: int) -> List[SemanticUnit]:
    """Rastreio GENÉRICO de interações (sem ontologia): para cada par de atributos, melhor ganho de informação (face às previsões do oráculo, nunca rótulos de teste) do teste
    conjuntivo ``(x_i ⋚ t_i) ∧ (x_j ⋚ t_j)`` com limiares nos quantis 0,3/0,5/0,7 e as 4 combinações de sentido (equivale a OR/m-of-n de 2). Forma ``n_units`` unidades de
    ``group_size`` atributos juntando, por ordem de ganho, os pares ainda não usados (mesmo nº e tamanho de unidades que as restantes orientações)."""
    y = np.asarray(oracle_pred)
    base = _entropy(y)
    p = X.shape[1]
    qs = np.quantile(X, [0.3, 0.5, 0.7], axis=0)                 # (3, p)
    side = [(X > qs[k]) for k in range(3)]                       # 3 × (n, p)
    gains = []
    for i in range(p):
        for j in range(i + 1, p):
            best = 0.0
            for ki in range(3):
                for kj in range(3):
                    for si in (True, False):
                        for sj in (True, False):
                            m = (side[ki][:, i] == si) & (side[kj][:, j] == sj)
                            f = m.mean()
                            if 0 < f < 1:
                                best = max(best, base - (f * _entropy(y[m]) + (1 - f) * _entropy(y[~m])))
            gains.append((best, i, j))
    gains.sort(key=lambda t: (-t[0], t[1], t[2]))
    used, units = set(), []
    for u in range(n_units):
        mem: List[int] = []
        for g, i, j in gains:
            if i in used or j in used:
                continue
            if len(mem) + 2 <= group_size:
                mem += [i, j]
                used.update((i, j))
            if len(mem) >= group_size:
                break
        rest = [c for c in range(p) if c not in used]
        while len(mem) < group_size and rest:                     # grupos de tamanho ímpar: completa com o atributo de maior ganho univariado restante
            c = rest.pop(0)
            mem.append(c)
            used.add(c)
        if len(mem) >= 2:
            units.append(SemanticUnit(f"pairscreen{u}", tuple(sorted(names[c] for c in mem)), 1.0, ("DATA",), (), 1.0))
    return units


def _path_recovery(model, rule_idx) -> Tuple[bool, bool]:
    """(par no mesmo teste, par no mesmo caminho raiz→folha)."""
    rule = {int(i) for i in rule_idx}
    same_test = False
    path_hit = False

    def walk(n, seen):
        nonlocal same_test, path_hit
        if n is None or n.is_leaf or n.test is None:
            return
        f = {int(l.feature) for l in n.test.literals}
        if rule <= f:
            same_test = True
        s2 = seen | (f & rule)
        if s2 >= rule:
            path_hit = True
        walk(n.true_child, s2)
        walk(n.false_child, s2)
    walk(model.root_, set())
    return same_test, path_hit


_orig_metrics = B32._metrics


def _metrics33(model, orc, inst, arm, seed, cpu, wall, t_compile, peak, n_units, reg):
    out = _orig_metrics(model, orc, inst, arm, seed, cpu, wall, t_compile, peak, n_units, reg)
    st, ph = _path_recovery(model, inst["rule_idx"])
    out["pair_same_test"], out["recovered_path"] = bool(st), bool(ph or st)
    feats = {int(l.feature) for n in model.iter_nodes() if not n.is_leaf and n.test is not None for l in n.test.literals}
    out["rule_attr_fraction"] = float(len(feats & set(inst["rule_idx"])) / len(inst["rule_idx"]))
    return out


def fit_arm33(fam: Family, spec: ProblemSpec, inst: Dict[str, Any], arm: str, seed: int, tmp: Path, *, bw_j1: int = 2, bw_j2: int = 2, measure_memory: bool = False) -> Dict[str, Any]:
    """Executa um braço. Variantes ontológicas: a ontologia «verdadeira» da família (C) é ``fam.ontology_true_variant``."""
    screen_t = {"s": 0.0, "pairs": 0}
    real_scr = B32.screening_units

    def timed(fn):
        def inner(X, names, pred, gs, nu):
            t0 = time.perf_counter()
            u = fn(X, names, pred, gs, nu)
            screen_t["s"] = time.perf_counter() - t0
            screen_t["pairs"] = int(X.shape[1] * (X.shape[1] - 1) // 2) if fn is pair_screening_units else int(X.shape[1])
            return u
        return inner
    saved_variant = B32.ARM_TO_VARIANT["C_OWL"]
    try:
        B32._metrics = _metrics33
        B32.screening_units = timed(pair_screening_units if arm == "H_PAIR_SCREEN" else real_scr)
        if arm == "C_OWL":
            B32.ARM_TO_VARIANT["C_OWL"] = fam.ontology_true_variant
        r = B32.fit_arm(spec, inst, _ARM_TO32[arm], seed, tmp, bw_h1=bw_j1, bw_h2=bw_j2, measure_memory=measure_memory)
    finally:
        B32._metrics, B32.screening_units, B32.ARM_TO_VARIANT["C_OWL"] = _orig_metrics, real_scr, saved_variant
    r["arm"], r["family"] = arm, fam.key
    r["screen_wall_s"] = screen_t["s"]
    r["screen_candidates"] = screen_t["pairs"] * (36 if arm == "H_PAIR_SCREEN" else 9) if arm in ("G_UNI_SCREEN", "H_PAIR_SCREEN") else 0
    r["total_cost_cpu_s"] = r["cpu_fit_s"] + r["compile_wall_s"] + r["screen_wall_s"]
    return r


def reference33(inst, seed):
    from validation.benchmark.b32_comparison import reference_models
    return reference_models(None, inst, seed)


def code33() -> Dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in CODE33}
