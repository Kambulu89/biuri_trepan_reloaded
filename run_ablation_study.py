"""Executa o benchmark científico multi-dataset fora da interface gráfica."""
from pathlib import Path

from core.ablation_study import AblationConfig, run_builtin_ablation, save_ablation_results


if __name__ == "__main__":
    project = Path(__file__).resolve().parent
    result = run_builtin_ablation(AblationConfig())
    paths = save_ablation_results(result, project / "results" / "ablation")
    for name, path in paths.items():
        print(f"{name}: {path}")
