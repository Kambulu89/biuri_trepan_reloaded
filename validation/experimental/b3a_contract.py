"""Contrato de compatibilidade semântica para B3-a (DESENHO + auditoria; NADA é integrado no núcleo; Doc. 27).

B3-a restringiria/priorizaria os candidatos m-of-n JÁ existentes com relações OWL. Este módulo implementa apenas a função pura do contrato de mapeamento
(axioma → compatibilidade entre FEATURES) e mede o tamanho do espaço de candidatos que ela autorizaria, para a ontologia verdadeira e para a permutada
(mesma estrutura de tamanhos ⇒ mesmas oportunidades de pesquisa).

Semântica (mundo aberto): a compatibilidade é um *prior de coerência semântica*, não uma consequência lógica. Um par (i, j) está **permitido** se
  (a) alguma das duas features não tem mapeamento OWL aceite (sem conhecimento ⇒ sem restrição), ou
  (b) pertencem ao mesmo grupo/classe (domínio ou subsunção comum), ou
  (c) a relatedness OWL direta ≥ limiar (equivalência, subsunção, inversa, domain/range, subsunção inferida pelo reasoner).
Caso contrário (ambas mapeadas, grupos distintos, relação abaixo do limiar) o par é **desencorajado**. Nunca usa rótulos, dados nem nomes de datasets.
Transitividade NÃO é assumida (evita fundir tudo por cadeias de relatedness decrescente).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_THRESHOLD = 0.35          # = semantic_relation_threshold do Reloaded (valor existente; não é ajustado)


@dataclass
class Compatibility:
    allowed: np.ndarray                                   # n×n booleana simétrica; diagonal True
    mapped: np.ndarray                                    # n booleana: feature com grupo/entidade OWL
    reason: Dict[tuple, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.mapped)


def build_compatibility(groups: Sequence[Optional[str]], relatedness: Optional[np.ndarray], threshold: float = DEFAULT_THRESHOLD, *,
                        use_relatedness: bool = True) -> Compatibility:
    n = len(groups)
    g = [None if (x is None or str(x).lower() in {"general", "none", ""}) else str(x) for x in groups]
    mapped = np.array([x is not None for x in g])
    rel = np.zeros((n, n)) if relatedness is None else np.asarray(relatedness, dtype=float)
    allowed = np.eye(n, dtype=bool)
    reason: Dict[tuple, str] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if not (mapped[i] and mapped[j]):
                ok, why = True, "open_world_unmapped"
            elif g[i] == g[j]:
                ok, why = True, "same_group"
            elif use_relatedness and rel[i, j] >= threshold:
                ok, why = True, "relatedness>=threshold"
            else:
                ok, why = False, "different_groups_below_threshold"
            allowed[i, j] = allowed[j, i] = ok
            reason[(i, j)] = why
    return Compatibility(allowed, mapped, reason)


def space_stats(c: Compatibility, sizes: Sequence[int] = (2, 3), cap: int = 400000) -> Dict[str, float]:
    """Fração de pares/subconjuntos de features autorizados (todos os pares internos permitidos) e dimensão do espaço."""
    n = c.n
    iu = np.triu_indices(n, 1)
    out: Dict[str, float] = {"n_features": n, "mapped_fraction": float(c.mapped.mean()) if n else 0.0,
                             "allowed_pair_fraction": float(c.allowed[iu].mean()) if n > 1 else 1.0}
    for k in sizes:
        total = 0
        ok = 0
        for comb in itertools.combinations(range(n), k):
            total += 1
            if all(c.allowed[a, b] for a, b in itertools.combinations(comb, 2)):
                ok += 1
            if total >= cap:
                break
        out[f"allowed_{k}_subset_fraction"] = float(ok / total) if total else 1.0
        out[f"n_{k}_subsets_evaluated"] = total
    return out


def audit_context(ctx, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, object]:
    """Quatro vistas do espaço autorizado: {com relatedness (política atual do Reloaded), só grupos} × {espaço enriquecido, só originais}."""
    if not ctx.available:
        return {"available": False, "reason": ctx.reason}
    out: Dict[str, object] = {"available": True, "n_groups": len({x for x in ctx.groups if x}), "n_original": len(ctx.orig_idx), "n_derived": len(ctx.onto_idx)}
    o = list(ctx.orig_idx)
    rel = np.asarray(ctx.relatedness)
    for tag, use_rel in (("rel", True), ("groups_only", False)):
        full = build_compatibility(ctx.groups, rel, threshold, use_relatedness=use_rel)
        orig = build_compatibility([ctx.groups[i] for i in o], rel[np.ix_(o, o)], threshold, use_relatedness=use_rel)
        for space, c in (("enriched", full), ("original", orig)):
            s = space_stats(c)
            out[f"{tag}__{space}"] = {k: s[k] for k in ("n_features", "mapped_fraction", "allowed_pair_fraction", "allowed_2_subset_fraction", "allowed_3_subset_fraction")}
    rel_off = rel[np.ix_(o, o)][np.triu_indices(len(o), 1)]
    out["relatedness_original_pairs"] = {"min": float(rel_off.min()), "median": float(np.median(rel_off)), "max": float(rel_off.max()),
                                         "fraction_ge_threshold": float((rel_off >= threshold).mean())} if len(rel_off) else {}
    return out


def run(dataset: str, seed: int = 42) -> Dict[str, object]:
    from sklearn.model_selection import train_test_split
    from core.benchmark.semantic import OwlSemanticProvider, ShuffledSemanticProvider
    from validation.benchmark import datasets as ds_mod
    from validation.benchmark import datasets_categorical  # noqa: F401  (regista os datasets categóricos)
    spec = ds_mod.REGISTRY[dataset]
    ds, _t, _c = spec.load()
    Xtr, _Xte, _ytr, _yte = train_test_split(ds.X, ds.y, test_size=0.25, stratify=ds.y, random_state=seed)      # só treino; teste não usado
    ctx = OwlSemanticProvider(str(spec.ontology_path)).build(Xtr, ds.feature_names, seed)
    out = {"dataset": dataset, "seed": seed, "true": audit_context(ctx)}
    if ctx.available:
        sh = ShuffledSemanticProvider(None).from_context(OwlSemanticProvider(str(spec.ontology_path)).build(Xtr, ds.feature_names, seed), seed, mode="structure")
        out["permuted_structure"] = audit_context(sh)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out")
    a = ap.parse_args()
    r = run(a.dataset, a.seed)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    print(json.dumps(r, indent=1))
