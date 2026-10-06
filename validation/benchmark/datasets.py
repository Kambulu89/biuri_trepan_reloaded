"""Registo de datasets do benchmark: DADOS + METADADOS apenas (sem hiperparâmetros do TREPAN/Reloaded, sem lógica por dataset).

* ``real_offline``: datasets reais empacotados (sem rede). Entram no ranking científico principal.
* ``synthetic_controlled``: validação do protocolo/isolamento. NUNCA entram no ranking principal nem em afirmações do tipo
  «Reloaded é superior em X % dos datasets».
"""
from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.benchmark.runner import Dataset
from validation import benchmark_ontologies as bo

HERE = Path(__file__).resolve().parent
ONTOLOGY_DIR = HERE / "ontologies"

REAL_OFFLINE = "real_offline"
SYNTHETIC_CONTROLLED = "synthetic_controlled"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    kind: str                                   # real_offline | synthetic_controlled
    origin: str
    version: str
    loader: Callable[[], Tuple[np.ndarray, np.ndarray, List[str], str, List[str]]]    # X, y, features, target, classes
    ontology_id: Optional[str]
    groups: Optional[Callable[[List[str]], Dict[str, List[str]]]]                      # só para metadado do mapeamento esperado
    source_file: Optional[str] = None                                                  # ficheiro empacotado (hash), se existir
    ranking_eligible: bool = True

    def load(self) -> Tuple[Dataset, str, List[str]]:
        X, y, feats, target, classes = self.loader()
        return Dataset(self.dataset_id, X, y, feats, list(range(len(classes)))), target, classes

    @property
    def ontology_path(self) -> Optional[Path]:
        return None if self.ontology_id is None else ONTOLOGY_DIR / f"{self.ontology_id}.owl"


def _sk(name: str):
    def loader():
        from sklearn import datasets as d
        b = getattr(d, f"load_{name}")()
        return (np.asarray(b.data, float), np.asarray(b.target), [str(f) for f in b.feature_names], "target",
                [str(c) for c in b.target_names])
    return loader


def _sk_version() -> str:
    import sklearn
    return f"scikit-learn {sklearn.__version__}"


def _packaged(name: str) -> Optional[str]:
    import sklearn.datasets as d
    p = Path(d.__file__).resolve().parent / "data" / name
    return str(p) if p.exists() else None


def source_file_sha256(path: Optional[str]) -> Optional[str]:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).exists() else None


def _group_loader_names(loader) -> List[str]:
    return loader()[2]


def _synthetic(name: str, n_classes: int, n_features: int, seed: int):
    def loader():
        from core.benchmark.synthetic import make_synthetic
        ds, _groups = make_synthetic(name, n_classes=n_classes, n_features=n_features, seed=seed)
        return ds.X, ds.y, ds.feature_names, "target", [f"class_{k}" for k in range(n_classes)]
    return loader


def _synthetic_groups(n_classes: int, n_features: int, seed: int, name: str):
    def groups(_names):
        from core.benchmark.synthetic import make_synthetic
        return make_synthetic(name, n_classes=n_classes, n_features=n_features, seed=seed)[1]
    return groups


def _registry() -> Dict[str, DatasetSpec]:
    v = _sk_version()
    specs = [
        DatasetSpec("iris", REAL_OFFLINE, "sklearn.datasets.load_iris (UCI Iris, Fisher 1936)", v, _sk("iris"), "iris",
                    lambda names: bo.IRIS_GROUPS, _packaged("iris.csv")),
        DatasetSpec("wine", REAL_OFFLINE, "sklearn.datasets.load_wine (UCI Wine)", v, _sk("wine"), "wine",
                    lambda names: bo.WINE_GROUPS, _packaged("wine_data.csv")),
        DatasetSpec("breast_cancer", REAL_OFFLINE, "sklearn.datasets.load_breast_cancer (UCI WDBC)", v, _sk("breast_cancer"),
                    "breast_cancer", bo.breast_cancer_groups, _packaged("breast_cancer.csv")),
        DatasetSpec("digits", REAL_OFFLINE, "sklearn.datasets.load_digits (UCI Optical Recognition of Handwritten Digits, 8x8)", v,
                    _sk("digits"), "digits", bo.digits_groups, _packaged("digits.csv.gz")),
        DatasetSpec("synthetic_binary", SYNTHETIC_CONTROLLED, "core.benchmark.synthetic.make_synthetic(seed=9)", "make_classification-" + v,
                    _synthetic("synthetic_binary", 2, 9, 9), "synthetic_binary", _synthetic_groups(2, 9, 9, "synthetic_binary"),
                    ranking_eligible=False),
        DatasetSpec("synthetic_3class", SYNTHETIC_CONTROLLED, "core.benchmark.synthetic.make_synthetic(seed=10)", "make_classification-" + v,
                    _synthetic("synthetic_3class", 3, 12, 10), "synthetic_3class", _synthetic_groups(3, 12, 10, "synthetic_3class"),
                    ranking_eligible=False),
    ]
    return {s.dataset_id: s for s in specs}


REGISTRY: Dict[str, DatasetSpec] = _registry()
MAIN_DATASETS = ("iris", "wine", "breast_cancer", "digits")
CONTROLLED_DATASETS = ("synthetic_binary", "synthetic_3class")


def expected_mapping_rate(spec: DatasetSpec) -> Optional[float]:
    """Fração de features cobertas pelos grupos da TBox. APENAS metadado (nunca threshold nem critério de seleção)."""
    if spec.groups is None:
        return None
    names = spec.loader()[2]
    covered = {f for fs in spec.groups(names).values() for f in fs}
    return float(len(covered & set(names)) / len(names)) if names else None


def freeze_ontologies(force: bool = False) -> Dict[str, str]:
    """Gera as TBoxes UMA vez (só nomes/significado das features; nunca resultados) e recusa sobrescrever as congeladas."""
    from core.benchmark.domain_tbox import create_domain_tbox
    ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for spec in REGISTRY.values():
        if spec.ontology_id is None:
            continue
        dest = spec.ontology_path
        if dest.exists() and not force:
            out[spec.ontology_id] = str(dest)
            continue
        names = spec.loader()[2]
        relations = bo.breast_cancer_family_roles(names) if spec.dataset_id == "breast_cancer" else None
        create_domain_tbox(dest, dataset_name=spec.dataset_id, groups=spec.groups(names), feature_relations=relations)
        out[spec.ontology_id] = str(dest)
    return out
