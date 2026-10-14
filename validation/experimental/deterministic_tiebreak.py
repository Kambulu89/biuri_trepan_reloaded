"""Variante EXPERIMENTAL do TREPAN com política de desempate DETERMINÍSTICA (estudo do Doc. 19). NÃO substitui o Original canónico.

Motivo: no Original, ``IG(máscara) ≠ IG(~máscara)`` ao bit em ~56% dos casos; a ordenação ``(-score, ...)`` usa esse ruído de ulp para desempatar candidatos
matematicamente iguais (complementares ou distintos), logo a árvore pode depender da plataforma. Esta variante:
  1. quantiza a pontuação (``score_digits`` casas decimais) antes de comparar/ordenar;
  2. desempata por chave canónica: (nº de literais, m, assinatura canónica = menor entre o teste e o seu complemento lógico, feature, limiar);
  3. só aceita extensão do feixe se a pontuação QUANTIZADA melhorar (empate quantizado ≠ melhoria).
Não consome RNG nem altera o fluxo de consultas por si só (só muda a escolha entre candidatos). ``score_digits=None`` desliga a política e reproduz o
Original bit-a-bit (verificado por teste), o que garante que a cópia da procura m-of-n não deriva do Original.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.trepan_original import Literal, MofNTest, TrepanOriginalClassifier


def signature(test: MofNTest) -> tuple:
    return (int(test.m), tuple(sorted((int(l.feature), round(float(l.threshold), 12), bool(l.greater)) for l in test.literals)))


def complement(test: MofNTest) -> MofNTest:
    """Complemento lógico: m-of-n(L) ≡ NOT (n−m+1)-of-n(¬L) — mesma partição, ramos trocados."""
    n = len(test.literals)
    return MofNTest(n - int(test.m) + 1, tuple(Literal(l.feature, l.threshold, not l.greater) for l in test.literals))


def canonical_signature(test: MofNTest) -> tuple:
    return min(signature(test), signature(complement(test)))


class DeterministicTieBreakTrepan(TrepanOriginalClassifier):
    EXPERIMENTAL = True

    def __init__(self, *args, score_digits: Optional[int] = 12, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_digits = score_digits

    def get_params(self, deep: bool = True):          # compatibilidade com utilitários tipo scikit-learn
        p = super().get_params(deep) if hasattr(super(), "get_params") else {}
        p["score_digits"] = self.score_digits
        return p

    def _q(self, score: float) -> float:
        return float(score) if self.score_digits is None else round(float(score), int(self.score_digits))

    def _candidate_literals(self, X, y):
        if self.score_digits is None:
            return super()._candidate_literals(X, y)
        # mesma geração do Original; só a ordenação usa a pontuação quantizada
        ranked = super()._candidate_literals(X, y)
        ranked.sort(key=lambda x: (-self._q(x[0]), x[1].feature, round(x[1].threshold, 12), not x[1].greater))
        return ranked

    def _best_mofn(self, X: np.ndarray, y: np.ndarray):
        if self.score_digits is None:
            return super()._best_mofn(X, y)
        ranked = self._candidate_literals(X, y)
        if not ranked:
            return None
        seed_score, seed_lit = ranked[0]
        best = MofNTest(1, (seed_lit,))
        best_score = seed_score
        beam = [(best_score, best)]
        literal_pool = [lit for _, lit in ranked[: max(8, self.max_features_per_node * 3)]]
        while beam:
            generated = []
            for _, test in beam:
                if len(test.literals) >= self.max_n:
                    continue
                used_features = {l.feature for l in test.literals}
                for lit in literal_pool:
                    if lit.feature in used_features:
                        continue
                    literals = tuple(list(test.literals) + [lit])
                    for m in {test.m, test.m + 1}:
                        if m < 1 or m > len(literals):
                            continue
                        cand = MofNTest(m, literals)
                        self._count("mofn_generated")
                        if not self._partition_significantly_different(test, cand, X):
                            self._count("mofn_rejected_not_distinct")
                            continue
                        mask = cand.evaluate(X)
                        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
                            self._count("mofn_rejected_min_samples_leaf")
                            continue
                        score = self._split_selection_score(y, mask, cand)
                        self._count("mofn_evaluated")
                        generated.append((score, cand))
            if not generated:
                break
            generated.sort(key=lambda x: (-self._q(x[0]), len(x[1].literals), x[1].m, canonical_signature(x[1]), signature(x[1])))
            next_beam = generated[: max(1, self.beam_width)]
            if self._q(next_beam[0][0]) <= self._q(best_score):         # empate quantizado NÃO é melhoria
                break
            best_score, best = next_beam[0]
            beam = next_beam
        return best, float(best_score)
