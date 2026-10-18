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


# ----------------------------------------------------------------------------------------------- gerador GENÉRICO (famílias/níveis/ruído/numéricas arbitrários; testes metamórficos)
def make_generic_data(n: int, seed: int, families: Dict[str, int], n_noise: int, n_num: int, k_true: int, *, p_unknown: float = 0.15, n_classes: int = 2, shuffle: bool = False
                      ) -> Tuple[np.ndarray, List[str], "GenericTruth"]:
    """``families``: {nome: nº de níveis ordinais (a ordem é a das colunas)}; a PRIMEIRA família define a verdade ``nível ≥ k_true`` (k_true = nº de níveis abaixo do limiar). Cada família tem ainda um nível
    «desconhecido» extra (não ordinal). Com ``n_classes`` > 2 a verdade é o nível da 1.ª família discretizado em classes. Devolve X, nomes e o oráculo de forma fechada."""
    rng = np.random.default_rng(seed)
    cols, names = [], []
    first_lv = None
    for fi, (fam, k) in enumerate(families.items()):
        unk = rng.random(n) < p_unknown
        lv = np.where(unk, k, rng.integers(0, k, size=n))
        cols.append(np.eye(k + 1)[lv])
        names += [f"{fam}__L{j + 1}" for j in range(k)] + [f"{fam}__U"]
        if fi == 0:
            first_lv, first_k = lv, k
    for j in range(n_noise):
        cols.append(rng.integers(0, 2, size=(n, 1)))
        names.append(f"noise__{j}")
    scales = [50.0, 200.0, 7.0, 1000.0]
    for j in range(n_num):
        cols.append((rng.normal(size=(n, 1)) * scales[j % len(scales)]))
        names.append(f"num__{j}")
    X = np.column_stack(cols).astype(float)
    fam0 = list(families)[0]
    return X, names, GenericTruth(names, [f"{fam0}__L{j + 1}" for j in range(first_k)], k_true, n_classes)


class GenericTruth:
    """Oráculo de forma fechada sobre as colunas pelo NOME (invariante à ordem das colunas): 1[nível da 1.ª família ≥ k_true] (ou discretização em ``n_classes``). Só vê colunas base.
    ``level_names`` = nomes das colunas dos níveis ordinais 1..k da 1.ª família; ``rename``/``reorder`` produzem o MESMO oráculo noutra nomenclatura/ordem (relações metamórficas)."""
    def __init__(self, names: Sequence[str], level_names: Sequence[str], k_true: int, n_classes: int = 2):
        self.names, self.level_names, self.k_true, self.n_classes = list(names), list(level_names), int(k_true), int(n_classes)
        self.calls, self.rows, self.seen = 0, 0, []

    def rename(self, fn) -> "GenericTruth":
        return GenericTruth([fn(n) for n in self.names], [fn(n) for n in self.level_names], self.k_true, self.n_classes)

    def reorder(self, names: Sequence[str]) -> "GenericTruth":
        assert sorted(names) == sorted(self.names)
        return GenericTruth(names, self.level_names, self.k_true, self.n_classes)

    def predict(self, X):
        X = np.asarray(X)
        assert X.shape[1] == len(self.names), "o oráculo só pode ver as colunas base"
        self.calls += 1
        self.rows += len(X)
        self.seen.append(X.copy())
        pos = {n: i for i, n in enumerate(self.names)}
        k = len(self.level_names)
        lv = sum((j + 1) * (X[:, pos[nm]] > 0.5) for j, nm in enumerate(self.level_names))              # 0 = desconhecido; 1..k = nível
        if self.n_classes == 2:
            return (lv > self.k_true).astype(int)
        return np.minimum(self.n_classes - 1, (lv * self.n_classes) // (k + 1)).astype(int)


def build_generic_owl(names: Sequence[str], families: Dict[str, int], k_true: int, variant: str = "TRUE", *, shuffle_seed: int = None) -> str:
    """OWL VERDADEIRO para a 1.ª família (``AtLeast_Lj`` = união dos níveis ≥ j para j = 1..k) ou controlo: PERM (ordem trocada) | NEUTRAL (sem definições) | FLAT. Os rótulos das propriedades são os
    nomes dos atributos. ``shuffle_seed`` embaralha a ordem de declaração (metamórfico)."""
    parts: List[str] = [klass("Root")]
    for fam in families:
        parts.append(klass(f"G_{fam}", ["Root"]))
    parts += [klass("G_noise", ["Root"]), klass("G_num", ["Root"])]
    for fam, k in families.items():
        if variant in ("TRUE", "PERM", "NEUTRAL"):
            for j in range(1, k + 1):
                parts.append(klass(f"Lvl_{fam}_L{j}", [f"G_{fam}"]))
    fam0, k0 = next(iter(families.items()))
    if variant in ("TRUE", "PERM", "NEUTRAL"):
        order = list(range(1, k0 + 1))
        if variant == "PERM":
            order = order[1::2] + order[0::2]                                        # ordem trocada, determinística
        for pos in range(1, len(order)):
            members = [f"Lvl_{fam0}_L{x}" for x in order[pos:]]
            nm = f"AtLeast_{fam0}_L{order[pos]}"
            parts.append(_union(nm, members, f"G_{fam0}") if variant != "NEUTRAL" else klass(nm, [f"G_{fam0}"]))
    for n in names:
        fam = n.split("__")[0]
        if fam in families:
            dom = f"Lvl_{n.replace('__', '_')}" if (variant in ("TRUE", "PERM", "NEUTRAL") and not n.endswith("__U")) else f"G_{fam}"
        elif fam == "noise":
            dom = "G_noise"
        else:
            dom = "G_num"
        parts.append(dprop("has_" + n, dom, label=n))
    if shuffle_seed is not None:
        rs = np.random.RandomState(shuffle_seed)
        rs.shuffle(parts)
    return HEADER.replace("example.org/b30", "example.org/h2/generic").replace("b30#", "h2generic#") + "".join(parts) + FOOTER
