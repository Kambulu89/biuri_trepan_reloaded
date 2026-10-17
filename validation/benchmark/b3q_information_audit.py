"""B3-Q / Doc. 48 §3: auditoria matemática das quantidades de informação do Doc. 44 (gerador LATENT_CONCEPT_V1). Cálculo exato (lgamma), sem simulação.

Modelo: Ω = pares (P, {A,B}) com P partição uniforme dos p=K·g atributos rotulados em K blocos NÃO rotulados de tamanho g (|Ω_P| = p!/((g!)^K K!)) e {A,B} par não ordenado de blocos
uniforme e independente de P. Suporte da regra S = A ∪ B (2g atributos). Entropia de Shannon em bits. Pressupostos: prior uniforme sobre partições e sobre pares; independência P ⊥ {A,B};
X (dados) não considerado em H/I «incondicionais»; a redundância com X é medida à parte (``b3q_partition_recoverability``)."""
import json
import sys
from math import lgamma, log

l2 = lambda x: x / log(2)
lf = lambda n: lgamma(n + 1)
lc = lambda n, k: l2(lf(n) - lf(k) - lf(n - k))


def quantities(K: int, g: int) -> dict:
    p = K * g
    H_P = l2(lf(p) - K * lf(g) - lf(K))                # entropia da partição
    H_S = lc(p, 2 * g)                                   # S é uniforme sobre os subconjuntos de 2g atributos por simetria do prior de P (invariante sob permutações de atributos)
    H_S_given_P = lc(K, 2)                               # dado P, S = união de 2 de K blocos
    I = H_S - H_S_given_P                                # informação mútua I(S;P)
    pairs_attr, pairs_conc = p * (p - 1) // 2, K * (K - 1) // 2
    return {"K": K, "g": g, "p": p, "H_partition_bits": H_P, "H_support_bits": H_S, "H_support_given_partition_bits": H_S_given_P, "I_support_partition_bits": I,
            "fraction_of_partition_info_relevant_to_support": I / H_P, "search_level_candidate_pairs_attributes": pairs_attr, "search_level_candidate_pairs_concepts": pairs_conc,
            "search_level_reduction_factor": pairs_attr / pairs_conc}


if __name__ == "__main__":
    out = {"model": __doc__, "cells": [quantities(8, 6), quantities(16, 6)]}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for c in out["cells"]:
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in c.items()})
