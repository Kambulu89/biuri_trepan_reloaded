"""Datasets sintéticos para smoke/testes (sem nomes de datasets reais) e TBox gerada a partir de grupos."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import make_classification

from core.benchmark.runner import Dataset


def make_synthetic(name: str, *, n_samples: int = 260, n_features: int = 9, n_classes: int = 2, n_groups: int = 3, seed: int = 0):
    """Dataset com estrutura de grupos de features (grupos contíguos). Devolve (Dataset, groups)."""
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=max(4, n_features * 2 // 3),
                               n_redundant=max(1, n_features // 5), n_classes=n_classes, n_clusters_per_class=1,
                               flip_y=0.03, random_state=seed)
    names = [f"feat_{i}" for i in range(n_features)]
    chunks = np.array_split(np.arange(n_features), n_groups)
    groups = {f"Group{chr(65 + k)}": [names[i] for i in idx] for k, idx in enumerate(chunks)}
    return Dataset(name, X, y, names, list(range(n_classes))), groups


def write_group_tbox(path, dataset_name: str, groups: Dict[str, List[str]]) -> str:
    from core.benchmark.domain_tbox import create_domain_tbox
    return create_domain_tbox(path, dataset_name=dataset_name, groups=groups)
