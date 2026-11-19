"""Datasets SINTÉTICOS de controlo positivo para a fase 6 da auditoria (DADOS + METADADOS; não alteram ``datasets.py`` nem o manifesto v1).

Cada dataset é gerado por uma lei física ou definição conhecida: a classe depende de uma RAZÃO ou PRODUTO de duas colunas, a que uma árvore
de cortes ortogonais só chega por uma «escada» de cortes, e há colunas irrelevantes. A ontologia v2 de cada um declara essa relação.
São CONTROLOS POSITIVOS: mostram se o mecanismo consegue ganhar quando o conhecimento é verdadeiro e relevante por construção; não mostram
ganho em dados reais. Todos os parâmetros (distribuições, limiares, ruído, tamanho, semente de geração) foram fixados antes de qualquer
execução; os limiares são valores redondos próximos da mediana, calculados numa amostra-piloto com semente diferente da de geração.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from validation.benchmark import datasets as ds_mod

N_ROWS = 400                 # treino = 300 com o holdout de 25 % do manifesto: regime de poucos dados
GENERATION_SEED = 20261118   # semente FIXA dos dados (as sementes mestras só definem as partições e o MLP)
LABEL_NOISE = 0.05           # fração de rótulos trocados


def _flip(y: np.ndarray, rng) -> np.ndarray:
    flip = rng.random(len(y)) < LABEL_NOISE
    return np.where(flip, 1 - y, y)


def _bmi(rng):
    height = rng.uniform(1.50, 1.95, N_ROWS)
    bmi = rng.uniform(16.0, 42.0, N_ROWS)
    mass = bmi * height ** 2 * (1.0 + rng.normal(0.0, 0.02, N_ROWS))           # erro de medição de 2 %
    X = np.column_stack([height, mass, rng.uniform(18, 80, N_ROWS), rng.normal(125, 15, N_ROWS), rng.normal(72, 10, N_ROWS)])
    y = (mass / height ** 2 >= 30.0).astype(int)                                    # obesidade: IMC ≥ 30 (OMS)
    return X, y, ["height_m", "mass_kg", "age_years", "systolic_bp_mmhg", "resting_heart_rate_bpm"], "obese", ["no", "yes"]


def _kinetic(rng):
    mass = rng.uniform(0.5, 20.0, N_ROWS)
    speed = rng.uniform(1.0, 30.0, N_ROWS)
    energy = 0.5 * mass * speed ** 2 * (1.0 + rng.normal(0.0, 0.02, N_ROWS))
    X = np.column_stack([mass, speed, rng.uniform(-10, 40, N_ROWS), rng.uniform(0, 90, N_ROWS), rng.uniform(0.1, 5.0, N_ROWS)])
    y = (energy >= 1000.0).astype(int)                                              # energia de impacto acima de 1 kJ
    return X, y, ["mass_kg", "speed_m_s", "ambient_temperature_c", "impact_angle_deg", "contact_area_cm2"], "damage", ["no", "yes"]


def _power(rng):
    voltage = rng.uniform(5.0, 240.0, N_ROWS)
    resistance = rng.uniform(1.0, 500.0, N_ROWS)
    power = voltage ** 2 / resistance * (1.0 + rng.normal(0.0, 0.02, N_ROWS))
    X = np.column_stack([voltage, resistance, rng.uniform(0.5, 50.0, N_ROWS), rng.uniform(-10, 40, N_ROWS), rng.uniform(0.5, 6.0, N_ROWS)])
    y = (power >= 60.0).astype(int)                                                 # dissipação acima de 60 W
    return X, y, ["voltage_v", "resistance_ohm", "cable_length_m", "ambient_temperature_c", "wire_cross_section_mm2"], "overheat", ["no", "yes"]


GENERATORS = {"phys_bmi": _bmi, "phys_kinetic": _kinetic, "phys_power": _power}


def _loader(name: str):
    def load() -> Tuple[np.ndarray, np.ndarray, List[str], str, List[str]]:
        rng = np.random.default_rng([GENERATION_SEED, list(GENERATORS).index(name)])
        X, y, feats, target, classes = GENERATORS[name](rng)
        return X.astype(float), _flip(y, rng), feats, target, classes
    return load


for _name in GENERATORS:
    ds_mod.REGISTRY[_name] = ds_mod.DatasetSpec(
        _name, "synthetic_controlled", "auditoria fase 6: controlo positivo gerado por lei física (validation/benchmark/datasets_physics.py)",
        f"gen-seed-{GENERATION_SEED}", _loader(_name), _name, None, ranking_eligible=False)
