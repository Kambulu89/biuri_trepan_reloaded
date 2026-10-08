"""Auditoria das features derivadas de uma ontologia: o que depende realmente do CONHECIMENTO OWL e o que é engenharia estatística convencional.

Genérico (não conhece nenhum dataset): recebe a ontologia, a matriz de treino e os nomes das colunas. Para cada feature derivada aceite devolve:

* ``kind`` / ``operation`` / ``sources`` — como é calculada (a aritmética é sempre convencional: média padronizada, diferença, rácio…);
* ``ontology_metadata_used`` — que metadado OWL escolhe as fontes (grupo por ``subPropertyOf``/``domain``/``subClassOf``; ``measurementFamily`` +
  ``statisticRole``);
* ``linear_r2_vs_sources`` / ``linear_r2_vs_all_originals`` — se é combinação linear das colunas originais (R²≈1 ⇒ não traz informação nova para um
  modelo que já aprende combinações lineares, p. ex. um MLP);
* ``nonlinear_information`` — R² contra TODAS as originais < 0.999 (rácios, contrastes, erro normalizado);
* ``name_heuristic_replicable`` — a MESMA feature (mesmas fontes e papéis) seria obtida SEM ontologia, só pelos nomes das colunas (sufixos
  mean/worst/error ou prefixo de família comum). Se for verdade, a ontologia formaliza o que os nomes já dizem; não acrescenta conhecimento novo;
* ``classification`` — ``ontology_knowledge_with_new_information`` | ``ontology_grouping_only_linear`` | ``conventional_engineering_replicable_by_names``.

Esta auditoria NÃO mede ganho preditivo: apenas separa origem do conhecimento e informação nova. O ganho exige a ablação descrita no relatório.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

ROLE_TOKENS = {
    "mean": "mean", "average": "mean", "avg": "mean",
    "worst": "worst", "max": "worst", "maximum": "worst", "extreme": "worst",
    "error": "error", "se": "error", "stderr": "error", "standarderror": "error",
}


def _tokens(name: str) -> List[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]


def name_heuristic_structure(columns: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Família -> {papel: coluna} inferida APENAS pelos nomes (um token de papel + o resto do nome como família). Sem ontologia."""
    families: Dict[str, Dict[str, str]] = {}
    for col in columns:
        toks = _tokens(col)
        roles = [ROLE_TOKENS[t] for t in toks if t in ROLE_TOKENS]
        if len(roles) != 1:
            continue
        family = "_".join(t for t in toks if t not in ROLE_TOKENS)
        if family:
            families.setdefault(family, {})[roles[0]] = str(col)
    return families


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 if ss_tot <= 0 else float(max(0.0, 1.0 - float((resid ** 2).sum()) / ss_tot))


def audit_derived_features(processor, train: pd.DataFrame, transformed: pd.DataFrame | None = None) -> pd.DataFrame:
    """Auditoria por feature derivada aceite de um ``OntologyProcessor`` JÁ AJUSTADO em ``train``."""
    out = processor.transform(train) if transformed is None else transformed
    originals = [c for c in train.columns]
    heuristic = name_heuristic_structure(originals)
    heuristic_groups = {tuple(sorted(roles.values())) for roles in heuristic.values()}
    all_orig = train[originals].to_numpy(float)
    rows: List[Dict[str, Any]] = []
    for spec in processor.feature_specs_:
        name = spec["name"]
        y = out[name].to_numpy(float)
        sources = list(spec.get("sources") or ([spec["source"]] if spec.get("source") else []))
        r2_sources = _r2(y, train[sources].to_numpy(float)) if sources else float("nan")
        r2_all = _r2(y, all_orig)
        kind, op = spec["kind"], spec.get("operation") or spec.get("method") or spec.get("operator")
        if kind == "hierarchical_aggregate":
            metadata = "agrupamento por pai OWL (rdfs:domain / subPropertyOf / subClassOf)"
            replicable = tuple(sorted(sources)) in heuristic_groups or set(sources) == set(originals)
        elif kind == "relational":
            metadata = "measurementFamily + statisticRole (mesma grandeza, papéis mean/worst/error)"
            roles = spec.get("roles") or []
            found = {tuple(sorted(r.values())) for r in heuristic.values() if all(role in r for role in roles)}
            replicable = tuple(sorted(sources)) in {tuple(sorted(r[x] for x in roles)) for r in heuristic.values() if all(x in r for x in roles)} if roles else False
        elif kind == "constraint":
            metadata, replicable = "limite numérico declarado (xsd:min/maxInclusive)", False
        else:
            metadata, replicable = "valores nominais -> grupos de vocabulário controlado", False
        nonlinear = bool(r2_all < 0.999)
        if replicable:
            cls = "conventional_engineering_replicable_by_names"
        elif nonlinear:
            cls = "ontology_knowledge_with_new_information"
        else:
            cls = "ontology_grouping_only_linear"
        rows.append({
            "feature": name, "kind": kind, "operation": op, "n_sources": len(sources), "sources": ";".join(sources),
            "ontology_metadata_used": metadata, "linear_r2_vs_sources": round(r2_sources, 6), "linear_r2_vs_all_originals": round(r2_all, 6),
            "nonlinear_information": nonlinear, "name_heuristic_replicable": bool(replicable),
            "reasoner_inferred": bool(spec.get("reasoner_inferred", False)), "provenance": spec.get("provenance"),
            "std": float(np.std(y)), "classification": cls,
        })
    return pd.DataFrame(rows)


def summarise(audit: pd.DataFrame) -> Dict[str, Any]:
    if audit.empty:
        return {"n_features": 0}
    return {
        "n_features": int(len(audit)),
        "by_kind": audit["kind"].value_counts().to_dict(),
        "by_classification": audit["classification"].value_counts().to_dict(),
        "linear_combinations_of_originals": int((audit["linear_r2_vs_all_originals"] >= 0.999).sum()),
        "with_nonlinear_information": int(audit["nonlinear_information"].sum()),
        "replicable_by_names_without_ontology": int(audit["name_heuristic_replicable"].sum()),
        "constant_features": int((audit["std"] <= 0).sum()),
    }


__all__ = ["audit_derived_features", "name_heuristic_structure", "summarise"]
