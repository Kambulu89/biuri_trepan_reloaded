#!/usr/bin/env python3
"""
Script para convertir los 6 datasets de UCI a formato CSV limpio.
Genera cabeceras reales (no números de columna) y coloca la clase al final.

Lee los archivos UCI originales de counterfactuals/datasets/originales/ y escribe los CSV
limpios en counterfactuals/datasets/ (ya incluidos en el repo). Se puede ejecutar desde
cualquier directorio.
"""

from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent
INPUT_DIR = OUTPUT_DIR / 'originales'


def process_iris():
    print("Procesando Iris...")
    df = pd.read_csv(INPUT_DIR / 'iris.data', header=None)
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df.columns = feature_names + ['class']
    df.to_csv(OUTPUT_DIR / 'iris.csv', index=False)
    print("  -> iris.csv generado")


def process_wdbc():
    print("Procesando WDBC...")
    df = pd.read_csv(INPUT_DIR / 'wdbc.data', header=None)
    df = df.iloc[:, 1:]
    class_col = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    feature_names = [f'f_{i}' for i in range(1, 31)]
    df.columns = feature_names
    df['diagnosis'] = class_col
    df.to_csv(OUTPUT_DIR / 'wdbc.csv', index=False)
    print("  -> wdbc.csv generado")


def process_sonar():
    print("Procesando Sonar...")
    df = pd.read_csv(INPUT_DIR / 'sonar.all-data', header=None)
    n_features = df.shape[1] - 1
    feature_names = [f'attribute_{i}' for i in range(1, n_features + 1)]
    class_col = df.iloc[:, -1]
    df = df.iloc[:, :-1]
    df.columns = feature_names
    df['class'] = class_col
    df.to_csv(OUTPUT_DIR / 'sonar.csv', index=False)
    print("  -> sonar.csv generado")


def process_german_credit():
    print("Procesando German Credit...")
    df = pd.read_csv(INPUT_DIR / 'german.data', sep=' ', header=None)
    class_col = df.iloc[:, -1]
    df = df.iloc[:, :-1]
    feature_names = [f'attr_{i}' for i in range(1, df.shape[1] + 1)]
    df.columns = feature_names
    df['class'] = class_col
    df.to_csv(OUTPUT_DIR / 'german_credit.csv', index=False)
    print("  -> german_credit.csv generado")


def process_hepatitis():
    """Converte Hepatitis preservando NaN para imputação *após* o split.

    No ficheiro UCI ``hepatitis.data`` a classe é a primeira coluna e a
    última coluna é ``histology``.  A versão anterior trocava estas colunas
    e ainda imputava globalmente, causando target errado e fuga de dados.
    """
    print("Procesando Hepatitis...")
    df = pd.read_csv(
        INPUT_DIR / 'hepatitis.data', header=None,
        na_values=['?', '', 'NA', 'N/A', 'null'], keep_default_na=True,
    )
    class_col = df.iloc[:, 0].copy()
    features = df.iloc[:, 1:].copy()
    feature_names = [f'feature_{i}' for i in range(1, features.shape[1] + 1)]
    features.columns = feature_names
    features['class'] = class_col
    features.to_csv(OUTPUT_DIR / 'hepatitis.csv', index=False)
    print("  -> hepatitis.csv generado (NaN preservados; classe = coluna 0)")


def process_wine():
    print("Procesando Wine...")
    df = pd.read_csv(INPUT_DIR / 'wine.data', header=None)
    class_col = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    feature_names = [
        'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
        'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
        'Color_intensity', 'Hue', 'OD280_OD315_of_diluted_wines', 'Proline',
    ]
    df.columns = feature_names
    df['class'] = class_col
    df.to_csv(OUTPUT_DIR / 'wine.csv', index=False)
    print("  -> wine.csv generado")


def main():
    print("=" * 60)
    print("Conversión de datasets UCI a CSV limpios")
    print("=" * 60)
    process_iris()
    process_wdbc()
    process_sonar()
    process_german_credit()
    process_hepatitis()
    process_wine()
    print("\nConversión completada.")


if __name__ == '__main__':
    main()
