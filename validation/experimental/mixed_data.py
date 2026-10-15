"""Gerador SINTÉTICO de dados mistos (nominais one-hot, nominais por códigos, ordinais, binários, contínuos correlacionados) para testes estruturais de plausibilidade
e restrições categóricas (Doc. 24). Sem nomes nem lógica de datasets reais. Devolve também um transformador do tipo «enriquecimento» (derivadas determinísticas
que misturam contínuas e categóricas) e um oráculo."""
from __future__ import annotations

import numpy as np


def make_mixed(seed: int = 0, n: int = 600):
    r = np.random.default_rng(seed)
    z = r.normal(size=(n, 3))
    z[:, 1] = 0.7 * z[:, 0] + 0.5 * z[:, 1]                                  # contínuas correlacionadas
    c3 = r.integers(0, 3, n); c4 = r.integers(0, 4, n)                         # nominais (one-hot)
    oh3, oh4 = np.eye(3)[c3], np.eye(4)[c4]
    b = np.column_stack([(z[:, 0] + r.normal(0, 0.5, n) > 0).astype(float), r.integers(0, 2, n).astype(float)])   # binárias
    ordi = np.clip(np.round(2 + z[:, 2] + r.normal(0, 0.7, n)), 0, 4)          # ordinal 0..4 (correlacionada com contínua)
    code = r.integers(0, 5, n).astype(float)                                   # nominal por códigos inteiros
    X = np.column_stack([z, oh3, oh4, b, ordi, code])
    names = ["z0", "z1", "z2", "n3_a", "n3_b", "n3_c", "n4_a", "n4_b", "n4_c", "n4_d", "b0", "b1", "ord", "code"]
    mu, sd = X[:, :3].mean(0), X[:, :3].std(0)
    w3 = np.array([0.0, 1.0, 2.0])

    def transform(Z):
        Z = np.asarray(Z, dtype=float)
        d0 = (((Z[:, :2] - mu[:2]) / sd[:2]).mean(axis=1))                     # derivada numérica (média padronizada)
        d1 = Z[:, 3:6] @ w3 + 0.5 * Z[:, 12]                                   # derivada que mistura categórica (one-hot) e ordinal
        return np.column_stack([Z, d0, d1])
    score = 1.2 * z[:, 0] - 0.8 * z[:, 2] + (c3 == 1) * 1.0 - (c4 == 2) * 1.0 + 0.6 * b[:, 0] + 0.3 * ordi
    y = np.digitize(score, np.quantile(score, [0.33, 0.66]))

    class Oracle:
        classes_ = np.array([0, 1, 2])
        def predict(self, A):
            A = np.asarray(A, dtype=float)[:, :14]
            s = 1.2 * A[:, 0] - 0.8 * A[:, 2] + (A[:, 4] == 1) * 1.0 - (A[:, 8] == 1) * 1.0 + 0.6 * A[:, 10] + 0.3 * A[:, 12]
            return np.digitize(s, np.quantile(score, [0.33, 0.66]))
    return X, y, names, transform, Oracle()
