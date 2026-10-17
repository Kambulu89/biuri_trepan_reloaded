"""H2 (Doc. 60): ontologias SINTÉTICAS CONTROLADAS com axiomas de validade CONHECIDA, para testes técnicos de ponta a ponta do adaptador T1/T2. O gerador define a verdade
(``y = 1[nível ordinal ≥ k]`` sobre uma variável one-hot) e a ontologia VERDADEIRA exprime exatamente essa ordem; as variantes de controlo são construídas por regra: PERM (ordem trocada: OWL válido,
semântica errada para a verdade), NEUTRAL (mesmas classes sem definições), FLAT (só partição). Só funcionamento; nenhuma conclusão sobre desempenho."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from validation.benchmark.b30_axiom_probe import FOOTER, HEADER, dprop, klass

LEVELS = ["A1", "A2", "A3", "A4", "A5"]          # A5 = «desconhecido» (NÃO ordinal; fica fora da ordem)
ORDER = ["A1", "A2", "A3", "A4"]                  # ordem verdadeira (crescente)
VAR = "sav"
OTHER = ["col__u", "col__v", "col__w"]            # atributos binários irrelevantes
NUM = ["num__x", "num__y"]                        # numéricos (escala grande: dominam a prioridade por variância, como no German)


def make_data(n: int, seed: int, k_true: int = 2, p_unknown: float = 0.15) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Devolve X (colunas base), nomes e o nível ordinal verdadeiro (−1 = desconhecido). A verdade/oráculo vê só X."""
    rng = np.random.default_rng(seed)
    unk = rng.random(n) < p_unknown
    lv = np.where(unk, 4, rng.integers(0, 4, size=n))
    onehot = np.eye(5)[lv]
    X = np.column_stack([onehot, rng.integers(0, 2, size=(n, 3)), rng.normal(size=(n, 2)) * np.array([50.0, 200.0])])
    names = [f"{VAR}__{l}" for l in LEVELS] + OTHER + NUM
    return X, names, np.where(unk, -1, lv)


class TruthOracle:
    """Oráculo de forma fechada: nível ordinal ≥ k_true (colunas A3 ou A4). Conta chamadas; só vê colunas BASE."""
    def __init__(self, n_base: int = 10, k_true: int = 2):
        self.n_base, self.k, self.calls, self.rows, self.seen = n_base, k_true, 0, 0, []

    def predict(self, X):
        X = np.asarray(X)
        assert X.shape[1] == self.n_base, "o oráculo só pode ver as colunas base"
        self.calls += 1
        self.rows += len(X)
        self.seen.append(X.copy())
        return (X[:, 2:4].max(axis=1) > 0.5).astype(int)


def _union(name: str, members: Sequence[str], parent: str) -> str:
    mem = "".join(f'<owl:Class rdf:about="#{m}"/>' for m in members)
    return (f'<owl:Class rdf:about="#{name}"><rdfs:subClassOf rdf:resource="#{parent}"/><owl:equivalentClass><owl:Class><owl:unionOf rdf:parseType="Collection">{mem}</owl:unionOf>'
            f'</owl:Class></owl:equivalentClass></owl:Class>\n')


def build_owl(variant: str, names: Sequence[str] = None, order: Sequence[str] = ORDER) -> str:
    """variant ∈ TRUE | PERM | NEUTRAL | FLAT. Rótulos das propriedades = nomes dos atributos (matching exato)."""
    names = list(names or (VAR and [f"{VAR}__{l}" for l in LEVELS] + OTHER + NUM))
    body = klass("Root") + klass("Savings", ["Root"]) + klass("Noise", ["Root"]) + klass("Numeric", ["Root"])
    if variant in ("TRUE", "PERM", "NEUTRAL"):
        for l in LEVELS:
            body += klass(f"Lvl_{l}", ["Savings"])
        o = list(order)
        if variant == "PERM":
            o = [o[i] for i in (2, 0, 3, 1)]                                   # ordem trocada, determinística
        for k in range(1, len(o) - 0):
            members = [f"Lvl_{x}" for x in o[k:]]
            name = f"AtLeast_{o[k]}"
            body += _union(name, members, "Savings") if variant != "NEUTRAL" else klass(name, ["Savings"])
    for n in names:
        if n.startswith(VAR + "__"):
            dom = f"Lvl_{n.split('__')[1]}" if variant in ("TRUE", "PERM", "NEUTRAL") else "Savings"
        elif n.startswith("col__"):
            dom = "Noise"
        else:
            dom = "Numeric"
        body += dprop("has_" + n, dom, label=n)
    return HEADER.replace("example.org/b30", "example.org/h2/synth").replace("b30#", "h2synth#") + body + FOOTER


def write_owl(tmp: Path, variant: str, **kw) -> Path:
    p = Path(tmp) / f"synthetic_{variant}.owl"
    p.write_text(build_owl(variant, **kw), encoding="utf-8")
    return p
