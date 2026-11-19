"""Elegibilidade da fase 8: exatamente ``fase7_eligibility`` (validação cruzada aninhada só no treino, margem 0,02), com o Adult registado."""
from validation.benchmark import run as R  # noqa: F401
from validation.benchmark import datasets_adult  # noqa: F401
from validation.benchmark import fase7_eligibility as F7E

if __name__ == "__main__":
    F7E.main()
