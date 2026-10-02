"""Splits PAREADOS e estratificados. O mesmo SplitSpec é usado por todos os métodos."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split


def hash_indices(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.asarray(a, dtype=np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()


@dataclass(frozen=True)
class SplitSpec:
    split_id: str
    seed: int
    repeat: int
    fold: int
    train_idx: np.ndarray = field(repr=False, compare=False)
    test_idx: np.ndarray = field(repr=False, compare=False)
    split_hash: str = ""
    scheme: str = "holdout"
    stratified: bool = True

    @property
    def n_train(self) -> int:
        return int(len(self.train_idx))

    @property
    def n_test(self) -> int:
        return int(len(self.test_idx))


def make_splits(y, *, scheme: str = "holdout", seeds: Sequence[int] = (11, 22, 33, 44, 55),
                test_size: float = 0.25, n_splits: int = 5, n_repeats: int = 2, base_seed: int = 42) -> List[SplitSpec]:
    """Gera os splits UMA vez; devem ser reutilizados por todos os braços (pareamento).

    * ``holdout``: um split estratificado por seed (a seed identifica a unidade experimental).
    * ``repeated_cv``: RepeatedStratifiedKFold(n_splits x n_repeats) com ``base_seed``.
    Se uma classe tiver menos exemplos que ``n_splits``, reduz-se n_splits (registado em ``scheme``)."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    counts = np.unique(y, return_counts=True)[1]
    out: List[SplitSpec] = []
    if scheme == "holdout":
        can_strat = counts.min() >= 2 and int(round(len(y) * test_size)) >= len(counts)
        for seed in seeds:
            tr, te = train_test_split(idx, test_size=test_size, random_state=int(seed),
                                      stratify=y if can_strat else None)
            tr, te = np.sort(tr), np.sort(te)
            out.append(SplitSpec(f"seed{seed}", int(seed), 0, 0, tr, te, hash_indices(tr, te), "holdout", can_strat))
    elif scheme == "repeated_cv":
        k = int(min(n_splits, counts.min()))
        if k < 2:
            raise ValueError("repeated_cv exige >=2 exemplos por classe.")
        rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=int(base_seed))
        for i, (tr, te) in enumerate(rskf.split(idx, y)):
            rep, fold = divmod(i, k)
            tr, te = np.sort(tr), np.sort(te)
            out.append(SplitSpec(f"rep{rep}_fold{fold}", int(base_seed), rep, fold, tr, te, hash_indices(tr, te),
                                 f"repeated_cv({k}x{n_repeats})", True))
    else:
        raise ValueError(f"scheme desconhecido: {scheme!r}")
    return out


def inner_folds(y_train, *, n_splits: int = 3, seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """CV INTERNA sobre o treino (índices relativos ao treino). É a única fonte permitida para
    seleccionar configuração/oráculo: o teste nunca entra nesta função."""
    y_train = np.asarray(y_train)
    counts = np.unique(y_train, return_counts=True)[1]
    k = int(max(2, min(n_splits, counts.min())))
    yield from StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed)).split(np.zeros(len(y_train)), y_train)


def assert_disjoint(split: SplitSpec) -> None:
    if np.intersect1d(split.train_idx, split.test_idx).size:
        raise AssertionError(f"Vazamento: treino e teste partilham índices ({split.split_id}).")
