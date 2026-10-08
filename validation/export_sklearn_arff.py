"""Exporta um dataset de demonstração do scikit-learn para ARFF (classe como último atributo), para testar a GUI com as ontologias de
``data/benchmark_ontologies_v7``. Os nomes das colunas são os de scikit-learn normalizados para identificadores sem espaços nem parênteses (``sepal length (cm)`` ->
``sepal_length_cm``), porque o leitor ARFF da GUI (scipy) não aceita nomes com espaços; o casamento com os rótulos da ontologia
é feito pela GUI.

Uso: ``python -m validation.export_sklearn_arff --name iris --out data/local/iris.arff``
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import arff
from sklearn import datasets

LOADERS = {
    "iris": datasets.load_iris,
    "wine": datasets.load_wine,
    "breast_cancer": datasets.load_breast_cancer,
    "digits": datasets.load_digits,
}


def export(name: str, out: Path) -> Path:
    bunch = LOADERS[name]()
    classes = [str(t) for t in bunch.target_names] if name != "digits" else [str(i) for i in range(10)]
    attributes = [(re.sub(r"[^A-Za-z0-9]+", "_", str(n)).strip("_"), "NUMERIC") for n in bunch.feature_names] + [("class", classes)]
    rows = [list(map(float, x)) + [classes[int(y)]] for x, y in zip(bunch.data, bunch.target)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(arff.dumps({"relation": name, "attributes": attributes, "data": rows, "description": ""}), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(LOADERS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(export(a.name, Path(a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
