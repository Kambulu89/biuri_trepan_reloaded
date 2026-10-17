"""B3.1 (Doc. 36): arnês EXPERIMENTAL de orientação semântica da pesquisa, só em ``validation/`` (NÃO é o mecanismo M3 do núcleo; serve exclusivamente para testar identificabilidade
e rastreabilidade). Subclasse do TREPAN Original canónico, que NÃO é modificado.

Mecanismo (preferência semântica pura, sem alterar pontuação nem eliminar nada):
  1. a pesquisa m-of-n do Original corre como sempre (resultado ``base``);
  2. para as ``unit_budget`` unidades semânticas mais promissoras (ordenadas pelo melhor ganho estatístico de um literal isolado DENTRO da unidade; usa os dados do nó, nunca o
     teste), corre-se a MESMA pesquisa do Original restrita aos atributos da unidade;
  3. escolhe-se o candidato de maior ``_split_selection_score`` ORIGINAL (estatístico). Em empate prevalece o Original. A ontologia só decide ONDE procurar.
Sem unidades ⇒ ``_best_mofn`` devolve exatamente o do Original (contrato EQ, testado com oráculo-espião).
Rastreabilidade (``semantic_trace_``): axioma → relação → mapeamento de atributos → facto → candidato → decisão estatística → efeito na árvore."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.trepan_original import MofNTest, TrepanOriginalClassifier
from validation.benchmark.b31_facts import FactRegistry, PermittedUse
from validation.benchmark.b31_relatedness import SemanticUnit


def test_signature(test: Optional[MofNTest]):
    if test is None:
        return None
    return (int(test.m), tuple((int(l.feature), round(float(l.threshold), 12), bool(l.greater)) for l in test.literals))


class SemanticSteeredTrepan(TrepanOriginalClassifier):
    def __init__(self, *, units: Optional[Sequence[SemanticUnit]] = None, unit_budget: int = 3, registry: Optional[FactRegistry] = None, pool_parity: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pool_parity = bool(pool_parity)               # v2: a pesquisa dentro de uma unidade tem o MESMO orçamento de literais (max_features_per_node) que a do Original
        self.units = list(units) if units else []
        self.unit_budget = int(unit_budget)
        self.registry = registry

    # ---- restrição de atributos para a pesquisa do Original (hook oficial ``_feature_priority_scores``)
    _restrict: Optional[np.ndarray] = None

    def _feature_priority_scores(self, X):
        base = super()._feature_priority_scores(X)
        if self._restrict is None:
            return base
        pr = np.zeros(X.shape[1], dtype=float)
        pr[self._restrict] = 1.0 + np.nan_to_num(np.asarray(base)[self._restrict], nan=0.0)
        return pr

    def _candidate_literals(self, X, y):
        ranked = super()._candidate_literals(X, y)
        if self._restrict is None:
            return ranked
        keep = {int(i) for i in self._restrict}
        return [(sc, lit) for sc, lit in ranked if int(lit.feature) in keep]

    def _unit_indices(self) -> List[np.ndarray]:
        names = list(getattr(self, "feature_names_in_", None) or [])
        pos = {n: i for i, n in enumerate(names)}
        out = []
        for u in self.units:
            idx = [pos[f] for f in u.features if f in pos]
            out.append(np.array(sorted(idx), dtype=int))
        return out

    def _restricted_search(self, X, y, idx: np.ndarray):
        saved = self.max_features_per_node
        self._restrict = idx
        if not self.pool_parity:
            self.max_features_per_node = int(len(idx))
        try:
            return super()._best_mofn(X, y)
        finally:
            self._restrict = None
            self.max_features_per_node = saved

    def _best_mofn(self, X, y):
        base = super()._best_mofn(X, y)
        if not self.units:
            return base                                                    # EQ: idêntico ao Original
        trace = getattr(self, "semantic_trace_", None)
        if trace is None:
            trace = self.semantic_trace_ = []
        call = len(trace)
        idxs = self._unit_indices()
        # conjunto de atributos que o Original considera (pool de literais) — para «o candidato existiria sem OWL?»
        ranked = self._candidate_literals(X, y)
        pool_feats = {lit.feature for _, lit in ranked[: max(8, self.max_features_per_node * 3)]}
        scored = []
        for u, idx in zip(self.units, idxs):
            if len(idx) < 2:
                continue
            self._restrict = idx
            saved = self.max_features_per_node
            self.max_features_per_node = int(len(idx))
            try:
                r = super()._candidate_literals(X, y)
            finally:
                self._restrict = None
                self.max_features_per_node = saved
            scored.append((r[0][0] if r else -1.0, u, idx))
        scored.sort(key=lambda t: (-t[0], -t[1].ic, t[1].concept))
        entries: List[Dict[str, Any]] = []
        best = base
        best_unit: Optional[int] = None
        for rank, (single, u, idx) in enumerate(scored[: self.unit_budget]):
            res = self._restricted_search(X, y, idx)
            e = {"unit_concept": u.concept, "unit_features": list(u.features), "unit_ic": u.ic, "fact_ids": list(u.fact_ids), "derivations": list(u.derivations),
                 "mean_matching_confidence": u.mean_matching_confidence, "rank_by_single_literal": rank, "single_literal_score": float(single),
                 "candidate_signature": test_signature(res[0]) if res else None, "candidate_score": float(res[1]) if res else None,
                 "unit_features_all_in_original_pool": bool(set(int(i) for i in idx) <= pool_feats),
                 "candidate_features": sorted({int(l.feature) for l in res[0].literals}) if res else []}
            e["candidate_exists_without_owl"] = bool(res is not None and (test_signature(res[0]) == test_signature(base[0] if base else None) or set(e["candidate_features"]) <= pool_feats))
            entries.append(e)
            if res is not None and (best is None or res[1] > best[1] + 1e-12):
                best, best_unit = res, len(entries) - 1
        names = list(getattr(self, "feature_names_in_", None) or [])
        base_sig = test_signature(base[0]) if base else None
        final_sig = test_signature(best[0]) if best else None
        trace.append({"call": call, "original_choice": base_sig, "original_score": float(base[1]) if base else None, "final_choice": final_sig, "final_score": float(best[1]) if best else None,
                      "choice_changed": final_sig != base_sig, "score_changed": False, "scoring_modified": False, "winning_unit": entries[best_unit]["unit_concept"] if best_unit is not None else None,
                      "hard_constraints_applied": 0, "logical_admissibility": "n/a (preferência; nenhum facto HARD aplicado)", "units_searched": entries,
                      "feature_names": [names[i] for i in (sorted({l.feature for l in best[0].literals}) if best else [])]})
        return best

    # ---- efeito na árvore final
    def trace_report(self, registry: Optional[FactRegistry] = None) -> Dict[str, Any]:
        reg = registry or self.registry
        tr = list(getattr(self, "semantic_trace_", []) or [])
        final_sigs = {test_signature(n.test) for n in self.iter_nodes() if not n.is_leaf and n.test is not None}
        out = []
        for t in tr:
            facts = []
            if reg is not None and t["winning_unit"] is not None:
                u = next(e for e in t["units_searched"] if e["unit_concept"] == t["winning_unit"])
                for fid in u["fact_ids"]:
                    f = reg.get(fid)
                    if f is not None:
                        facts.append({"fact_id": f.id, "kind": f.kind, "axioms": list(f.source_axioms), "derivation": f.derivation, "epistemic": f.epistemic, "permitted_use": f.permitted_use,
                                      "matching_confidence": f.matching_confidence, "attribute_to_entity": {a: reg.feature_entity.get(a) for a in f.attributes}})
            out.append(dict(t, in_final_tree=t["final_sig"] in final_sigs if "final_sig" in t else (t["final_choice"] in final_sigs), semantic_facts=facts,
                            semantically_influenced=bool(t["choice_changed"])))
        return {"n_calls": len(tr), "n_choice_changed": int(sum(t["choice_changed"] for t in tr)), "n_influenced_in_final_tree": int(sum(1 for t in out if t["semantically_influenced"] and t["in_final_tree"])),
                "n_internal_nodes": int(sum(1 for n in self.iter_nodes() if not n.is_leaf)), "entries": out}
