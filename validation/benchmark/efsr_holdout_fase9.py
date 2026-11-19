"""FASE 9 da auditoria (EXPERIMENTAL, isolado do núcleo): gate do EFSR validado em consultas novas.

Variante descrita no ``FASE7_PREREGISTO.md`` §6 e pré-registada no ``FASE9_PREREGISTO.md``. O núcleo
(``core/trepan_reloaded_historical.py``) não é alterado: esta subclasse só existe em ``validation/`` e só muda
o comportamento quando ``ACTIVE["on"]`` é verdadeiro (braços V/Vq da fase 9). Com o interruptor desligado,
todos os métodos delegam no Reloaded sem alterações.

Regra do gate (substitui os dois gates medidos dentro da amostra e a tolerância de coerência):
1. Só é chamado onde o gate original seria chamado: proposta semântica diferente da divisão estatística, numa
   região de erro elegível (critério de elegibilidade do Reloaded, inalterado).
2. Sorteia um lote NOVO de consultas na região do nó: mesmo modelo de distribuição local e mesmo gerador do lote
   de decisão (incluindo restrições do caminho e projeção), com uma RNG própria
   ``SeedSequence([random_state, 0xEF59, node_id])``. O lote não entra na escolha de nenhuma das duas divisões e a
   RNG do Original não avança.
3. Cada divisão prevê, em cada ramo, a classe maioritária do lote de decisão (o mesmo que a árvore usaria).
4. Aceita a proposta semântica só se, no lote novo, o teste de McNemar exato unilateral (proposta melhor do que a
   estatística) tiver p < ``ALPHA``. Sem tolerância.
As consultas do lote novo são contadas à parte (``holdout_queries``) e reportadas; não consomem o orçamento do
Original, para que a porção Original da árvore não mude.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Optional

import numpy as np
from scipy.stats import binomtest

from core.trepan_reloaded_historical import TrepanReloadedClassifier

ALPHA = 0.05
SEED_TAG = 0xEF59
ACTIVE = {"on": False, "arm": None}
AUDIT_PATH_ENV = "BIURI_FASE9_AUDIT"


def _branch_predictor(test, X_dec: np.ndarray, y_dec: np.ndarray, classes: np.ndarray):
    """Classe maioritária de cada ramo no lote de decisão; ramo vazio herda a maioria do nó."""
    counts = np.asarray([(y_dec == c).sum() for c in classes], dtype=int)
    node_major = classes[int(np.argmax(counts))]
    if test is None:
        return lambda X: np.full(len(X), node_major, dtype=object)
    mask = np.asarray(test.evaluate(X_dec), dtype=bool)
    pred = {}
    for branch in (False, True):
        idx = mask if branch else ~mask
        if np.any(idx):
            c = np.asarray([(y_dec[idx] == k).sum() for k in classes], dtype=int)
            pred[branch] = classes[int(np.argmax(c))]
        else:
            pred[branch] = node_major

    def predict(X):
        m = np.asarray(test.evaluate(X), dtype=bool)
        out = np.empty(len(X), dtype=object)
        out[m] = pred[True]
        out[~m] = pred[False]
        return out
    return predict


class HoldoutEFSRReloaded(TrepanReloadedClassifier):
    """Reloaded com o gate do EFSR validado num lote novo de consultas (só com ``ACTIVE['on']``)."""

    def fit(self, X, *a, **k):
        self.f9_holdout_ = {"active": bool(ACTIVE["on"]), "attempts": 0, "accepted": 0, "holdout_queries": 0, "tests": []}
        out = super().fit(X, *a, **k)
        path = os.environ.get(AUDIT_PATH_ENV)
        if path:
            rec = dict(self.f9_holdout_, arm=ACTIVE.get("arm"),
                       efsr_attempted=int(getattr(self, "error_focused_interventions_attempted_", 0)),
                       efsr_accepted=int(getattr(self, "error_focused_interventions_accepted_", 0)))
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        return out

    def _decision_sample(self, node):
        X, y, model = super()._decision_sample(node)
        self._f9_node, self._f9_model = node, model
        return X, y, model

    def _f9_draw_holdout(self, n: int) -> Optional[tuple]:
        node, model = getattr(self, "_f9_node", None), getattr(self, "_f9_model", None)
        if node is None or model is None or n <= 0 or self.oracle_ is None:
            return None
        clone = copy.copy(model)
        clone._rng = np.random.default_rng(np.random.SeedSequence([int(self.random_state) % (2 ** 32), SEED_TAG, int(max(0, node.node_id))]))
        saved = (getattr(self, "semantic_projection_batches_", None), getattr(self, "semantic_projected_query_count_", None),
                 copy.deepcopy(getattr(self, "ontology_sampling_audit_", None)), getattr(self, "error_focused_query_batches_", None),
                 getattr(self, "semantic_active_query_batches_", None), getattr(self, "semantic_active_query_selected_", None),
                 getattr(self, "error_focused_query_selected_", None))
        try:
            Xv = self._draw_membership_queries(clone, n, node.constraints, node)
        finally:                    # contadores de auditoria do Reloaded ficam como se o lote não existisse
            (self.semantic_projection_batches_, self.semantic_projected_query_count_, self.ontology_sampling_audit_,
             self.error_focused_query_batches_, self.semantic_active_query_batches_, self.semantic_active_query_selected_,
             self.error_focused_query_selected_) = saved
        if len(Xv) == 0:
            return None
        yv = np.asarray(self.oracle_.predict(Xv))
        return np.asarray(Xv, dtype=float), yv

    def _evaluate_error_focused_intervention(self, X, y, *, data_test, semantic_test, disagreement_rate):
        result = super()._evaluate_error_focused_intervention(
            X, y, data_test=data_test, semantic_test=semantic_test, disagreement_rate=disagreement_rate)
        if not ACTIVE["on"] or not result["attempted"]:
            return result
        # O gate original foi apenas usado para chegar a «attempted» (proposta diferente, região elegível);
        # a decisão passa a ser só a do lote novo.
        self.f9_holdout_["attempts"] += 1
        n = int(getattr(self, "effective_min_sample_", len(y)))
        drawn = self._f9_draw_holdout(n)
        result = dict(result, accepted=False, accepted_within_tolerance=False)
        if drawn is None:
            result["reason"] = "holdout_unavailable"
            self.f9_holdout_["tests"].append({"n": 0, "b": 0, "c": 0, "p": None, "accepted": False})
            return result
        Xv, yv = drawn
        self.f9_holdout_["holdout_queries"] += int(len(Xv))
        classes = np.asarray(self.classes_)
        y_dec = np.asarray(y)
        ok_s = _branch_predictor(semantic_test, X, y_dec, classes)(Xv) == yv
        ok_d = _branch_predictor(data_test, X, y_dec, classes)(Xv) == yv
        b = int(np.sum(ok_s & ~ok_d))
        c = int(np.sum(~ok_s & ok_d))
        p = float(binomtest(b, b + c, 0.5, alternative="greater").pvalue) if b + c > 0 else 1.0
        accepted = bool(p < ALPHA)
        result.update(accepted=accepted, reason="holdout_mcnemar_accepts" if accepted else "holdout_mcnemar_rejects",
                      holdout_n=int(len(Xv)), holdout_b=b, holdout_c=c, holdout_p=p,
                      holdout_fidelity_semantic=float(np.mean(ok_s)), holdout_fidelity_data=float(np.mean(ok_d)))
        self.f9_holdout_["tests"].append({"n": int(len(Xv)), "b": b, "c": c, "p": p, "accepted": accepted})
        if accepted:
            self.f9_holdout_["accepted"] += 1
        return result


__all__ = ["HoldoutEFSRReloaded", "ACTIVE", "ALPHA"]
