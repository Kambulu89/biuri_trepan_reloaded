"""Executor da FASE 4 da auditoria (EXPERIMENTAL; só DEV): separa o efeito dos ATRIBUTOS derivados da ontologia v2 do efeito dos
MECANISMOS do Reloaded (peso de ganho, Gain_Reloaded, EFSR, consultas ativas). Todos os braços TREPAN usam o amostrador conjunto (3 réplicas).

Braços novos (ids ``sc_{X}_r{r}``):
  B  Original + amostrador conjunto (espaço original) — verificação de integridade face às fases 2 e 3;
  O  «Original-equivalente» no espaço enriquecido pela OWL v2: TrepanReloadedClassifier com λ=0 (sem pesos, sem Gain_Reloaded,
     sem grupos/relatedness), EFSR desligado, consultas ativas desligadas; só a projeção das consultas (recalcula as derivadas) fica ativa;
  Q  igual a O com a ontologia PERMUTADA (controlo negativo do runner: mesmas operações, significado destruído);
  K  igual a O com atributos de ENGENHARIA CONVENCIONAL (médias de grupos escolhidos pelos dados; sem ontologia).
Reutiliza ``run_sampler_c`` (gancho do amostrador e auditoria das consultas) e ``run_fase3.activate_v2`` (ontologias v2), sem os alterar.
``coherent_pilot=True`` só serve para o runner construir o contexto convencional; os braços COH que ele acrescenta são filtrados.

    python -m validation.benchmark.run_fase4 --unit iris 42 --out DIR [--reps 3] [--reuse-checkpoint-from DIR]
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
from pathlib import Path

from validation.benchmark import run as R      # PRIMEIRO: fixa o ambiente numérico
from core.benchmark.runner import ArmSpec
from validation.benchmark import run_sampler_c as SCM
from validation.benchmark.run_fase3 import activate_v2

NEW_KINDS = ("B", "O", "Q", "K")


def build_arms(reps: int, dataset: str = "", c_replicas: str = "none"):
    arms = []
    for r in range(reps):
        common = dict(oracle="mlp_original", lam=0.0, active_queries=False, error_focus=False, role="fase4", group="SC")
        arms += [
            ArmSpec(f"sc_B_r{r}", "trepan_original", f"B: Original + amostrador conjunto (réplica {r})", oracle="mlp_original", role="fase4", group="SC"),
            ArmSpec(f"sc_O_r{r}", "trepan_reloaded", f"O: Original-equivalente no espaço OWL v2 + conjunto (réplica {r})", tree_space="enr",
                    semantic="real", needs_semantic=True, **common),
            ArmSpec(f"sc_Q_r{r}", "trepan_reloaded", f"Q: CONTROLO O com OWL v2 permutada + conjunto (réplica {r})", tree_space="enr",
                    semantic="shuffled", needs_semantic=True, control=True, **common),
            ArmSpec(f"sc_K_r{r}", "trepan_reloaded", f"K: CONTROLO O com engenharia convencional + conjunto (réplica {r})", tree_space="enr",
                    semantic="conventional", needs_semantic=True, control=True, **common),
        ]
    return arms


_plausibility = SCM.plausibility


def plausibility(Q, Xtr, mlp, n_o, enrich, yq, otp, seed: int = 0):
    """A auditoria de ``run_sampler_c`` mede a coerência das derivadas com o transformador da OWL REAL. Para os controlos (Q, K) esse
    transformador não é o da árvore (no K nem tem o mesmo número de colunas), por isso a medida não se aplica e não é calculada."""
    kind = str(SCM.SC.get("arm") or "").split("_")[1] if str(SCM.SC.get("arm") or "").startswith("sc_") else None
    return _plausibility(Q, Xtr, mlp, n_o, None if kind in ("Q", "K") else enrich, yq, otp, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--reuse-checkpoint-from", default=None)
    a = ap.parse_args()
    dataset, seed, out = a.unit[0], int(a.unit[1]), Path(a.out)
    activate_v2(dataset, out)
    if a.reuse_checkpoint_from:
        dst = out / "checkpoints"
        dst.mkdir(parents=True, exist_ok=True)
        for f in (Path(a.reuse_checkpoint_from) / "checkpoints").glob(f"{dataset}_seed{seed}.ckpt*"):
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
    base_cfg = R.benchmark_config
    R.benchmark_config = lambda s: dataclasses.replace(base_cfg(s), coherent_pilot=True)
    SCM.build_arms = build_arms
    SCM.SAMPLER_ARMS = NEW_KINDS
    SCM.plausibility = plausibility
    SCM.run_unit_sampler(dataset, seed, out, a.reps, "none")


if __name__ == "__main__":
    sys.exit(main())
