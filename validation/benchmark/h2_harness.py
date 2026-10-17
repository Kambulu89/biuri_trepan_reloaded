"""H2 (Doc. 60): integração TÉCNICA do adaptador T1/T2 no arnês experimental de ``validation/`` — braços A–I com identidade de oráculo e de instâncias verificável e rastreio por literal. NÃO é um
benchmark: ``run_arm`` devolve estrutura (regras, consultas, rastreio), nunca accuracy/fidelity. Nada em ``core/``; nenhum mecanismo M0–M6 do núcleo."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.trepan_original import TrepanOriginalClassifier
from validation.benchmark.b31_facts import FactRegistry, compile_registry
from validation.benchmark.b31_relatedness import SemanticUnit, semantic_units
from validation.benchmark.b31_steering import SemanticSteeredTrepan
from validation.benchmark.h2_axiom_mechanisms import DerivedLiteral, derived_union_literals, family_of
from validation.benchmark.h2_trace import DerivedLiteralTrepan, augment

ARMS = ("A_ORIG", "B_WIDE", "C_ENRICHED_NO_GUIDANCE", "D_GROUPS_NO_OWL", "E_FLAT_OWL", "F_EXPRESSIVE", "G_NEUTRALIZED", "H_PERMUTED", "I_CONVENTIONAL")


class H2Steered(DerivedLiteralTrepan, SemanticSteeredTrepan):
    """Original + (opcional) unidades de pesquisa semânticas (preferência) + literais derivados (M-R) + poda lógica (M-P). Sem unidades, sem derivados e sem poda ≡ Original."""


def instance_hash(X: np.ndarray, names: Sequence[str]) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(X, dtype=float)).tobytes() + json.dumps(list(names)).encode()).hexdigest()[:16]


@dataclass
class Context:
    X: np.ndarray                              # treino, colunas BASE
    names: List[str]
    oracle: Any                                # o MESMO objeto para todos os braços
    params: Dict[str, Any]                     # recursos iguais para todos os braços
    registries: Dict[str, FactRegistry]        # TRUE | PERM | NEUTRAL | FLAT (compilados do OWL)
    group_units: Optional[List[SemanticUnit]] = None       # partição como lista simples (braço D)
    defined: Optional[Dict[str, set]] = None               # variante → classes definidas por união (únicas fontes de literais)


def data_driven_cumulative(X: np.ndarray, names: Sequence[str], oracle_pred: np.ndarray, min_size: int = 2) -> List[DerivedLiteral]:
    """Engenharia convencional: para cada família de colunas binárias exclusivas, ordena os níveis pela taxa de positivos das PREVISÕES DO ORÁCLE no treino (sem rótulos reais, sem teste)
    e cria as uniões cumulativas do topo da ordem. Não usa ontologia."""
    out: List[DerivedLiteral] = []
    fam: Dict[str, List[int]] = {}
    for i, n in enumerate(names):
        if "__" in n and np.all(np.isin(X[:, i], (0.0, 1.0))):
            fam.setdefault(family_of(n), []).append(i)
    for f, idx in fam.items():
        if len(idx) < 3:
            continue
        rate = [(float(np.mean(oracle_pred[X[:, i] > 0.5])) if np.any(X[:, i] > 0.5) else -1.0, i) for i in idx]
        order = [i for _, i in sorted(rate)]
        for k in range(1, len(order)):
            att = tuple(sorted(names[i] for i in order[k:]))
            if len(att) >= min_size:
                out.append(DerivedLiteral(f"conv_{f}_top{len(att)}", att, False, (), f))
    return out


def arm_config(arm: str, ctx: Context) -> Dict[str, Any]:
    """Especificação do braço: unidades, literais derivados e poda. Toda a ontologia entra por registo compilado do OWL (reasoner incluído)."""
    names, X = ctx.names, ctx.X
    units: List[SemanticUnit] = []
    derived: List[DerivedLiteral] = []
    prune = False
    kw: Dict[str, Any] = {}
    if arm == "B_WIDE":
        kw["max_features_per_node"] = len(names)
    elif arm == "C_ENRICHED_NO_GUIDANCE":
        derived = derived_union_literals(ctx.registries["TRUE"], X, names, only_concepts=(ctx.defined or {}).get("TRUE", set()))
    elif arm == "D_GROUPS_NO_OWL":
        units = list(ctx.group_units or [])
    elif arm == "E_FLAT_OWL":
        units = semantic_units(ctx.registries["FLAT"], names)
    elif arm == "F_EXPRESSIVE":
        units = semantic_units(ctx.registries["TRUE"], names)
        derived = derived_union_literals(ctx.registries["TRUE"], X, names, only_concepts=(ctx.defined or {}).get("TRUE", set()))
        prune = True
    elif arm == "G_NEUTRALIZED":
        units = semantic_units(ctx.registries["NEUTRAL"], names)
        derived = derived_union_literals(ctx.registries["NEUTRAL"], X, names, only_concepts=(ctx.defined or {}).get("NEUTRAL", set()))
        prune = True
    elif arm == "H_PERMUTED":
        units = semantic_units(ctx.registries["PERM"], names)
        derived = derived_union_literals(ctx.registries["PERM"], X, names, only_concepts=(ctx.defined or {}).get("PERM", set()))
        prune = True
    elif arm == "I_CONVENTIONAL":
        derived = data_driven_cumulative(X, names, np.asarray(ctx.oracle.predict(X)))
    elif arm != "A_ORIG":
        raise ValueError(arm)
    return {"units": units, "derived": derived, "prune": prune, "kw": kw}


def run_arm(arm: str, ctx: Context) -> Dict[str, Any]:
    spec = arm_config(arm, ctx)
    Xa, na = augment(ctx.X, ctx.names, spec["derived"])
    params = dict(ctx.params, **spec["kw"])
    m = H2Steered(derived=spec["derived"], base_names=ctx.names, prune_redundant=spec["prune"], units=spec["units"], pool_parity=True, **params)
    calls0 = getattr(ctx.oracle, "calls", 0)
    m.fit(Xa, oracle=ctx.oracle, feature_names=na)
    rules = json.dumps(m.export_rules(), default=str)
    return {"arm": arm, "model": m, "instance_hash": instance_hash(ctx.X, ctx.names), "oracle_id": id(ctx.oracle), "n_units": len(spec["units"]), "n_derived": len(spec["derived"]),
            "prune": spec["prune"], "rules_sha": hashlib.sha256(rules.encode()).hexdigest()[:16], "queries": int(m.membership_queries_), "oracle_calls": getattr(ctx.oracle, "calls", 0) - calls0,
            "trace": getattr(m, "semantic_trace_", None), "n_features_seen": Xa.shape[1]}
