"""Executor da FASE 6 da auditoria (EXPERIMENTAL): o Reloaded corrigido (escolhido e congelado na fase 5) em datasets novos e sob orçamento
restrito. Reutiliza ``run_sampler_c``, ``run_fase3.activate_v2`` e ``run_fase5.SPECS``, sem os alterar.

Braços (ids ``sc_{X}_r{r}``), todos com o amostrador conjunto, 3 réplicas:
  B   Original, espaço original, estrutura do tuning                B7  idem com no máximo 7 nós
  R   Reloaded corrigido (``--reloaded``), espaço OWL v2             R7  idem com no máximo 7 nós
  Rq  R com a ontologia v2 permutada (controlo)                      Rq7 idem com no máximo 7 nós
C4.5 canónico, MLP e os braços históricos vêm do runner (sem alterações). O limite de 7 nós aplica-se só aos braços terminados em «7».

    python -m validation.benchmark.run_fase6 --unit phys_bmi 42 --out DIR --reloaded T [--reps 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from validation.benchmark import run as R      # PRIMEIRO: fixa o ambiente numérico
from core.benchmark import runner as RN
from core.benchmark.runner import ArmSpec
from validation.benchmark import datasets_categorical  # noqa: F401  (regista german_credit)
from validation.benchmark import datasets_physics      # noqa: F401  (regista os controlos positivos)
from validation.benchmark import run_fase4 as F4
from validation.benchmark import run_fase5 as F5
from validation.benchmark import run_sampler_c as SCM
from validation.benchmark.run_fase3 import activate_v2

SMALL_BUDGET_NODES = 7
STATE = {"reloaded": None}


def _reloaded_kwargs(kind: str, shuffled: bool) -> dict:
    kw = dict(F5.SPECS[kind][1])
    kw.pop("family", None)
    kw["semantic"] = "shuffled" if shuffled else "real"
    if shuffled:
        kw["control"] = True
    return kw


def build_arms(reps: int, dataset: str = "", c_replicas: str = "none"):
    kind = STATE["reloaded"]
    arms = []
    for r in range(reps):
        for suffix in ("", "7"):
            note = f" (máx. {SMALL_BUDGET_NODES} nós)" if suffix else ""
            arms.append(ArmSpec(f"sc_B{suffix}_r{r}", "trepan_original", f"B{suffix}: Original + conjunto{note} (réplica {r})",
                                oracle="mlp_original", role="fase6", group="SC"))
            for tag, shuffled in (("R", False), ("Rq", True)):
                arms.append(ArmSpec(f"sc_{tag}{suffix}_r{r}", "trepan_reloaded",
                                    f"{tag}{suffix}: Reloaded corrigido ({kind}){' com OWL v2 permutada' if shuffled else ''} + conjunto{note} (réplica {r})",
                                    tree_space="enr", oracle="mlp_original", needs_semantic=True, role="fase6", group="SC",
                                    **_reloaded_kwargs(kind, shuffled)))
    return arms


def plausibility(Q, Xtr, mlp, n_o, enrich, yq, otp, seed: int = 0):
    arm = str(SCM.SC.get("arm") or "")
    kind = arm.split("_")[1] if arm.startswith("sc_") else None
    return F4._plausibility(Q, Xtr, mlp, n_o, None if kind and kind.startswith("Rq") else enrich, yq, otp, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reloaded", required=True, choices=["O", "T", "L", "W"], help="correção congelada na fase 5")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    dataset, seed, out = a.unit[0], int(a.unit[1]), Path(a.out)
    activate_v2(dataset, out)
    STATE["reloaded"] = a.reloaded
    (out / "fase6_config.json").write_text(json.dumps({"reloaded": a.reloaded, "small_budget_nodes": SMALL_BUDGET_NODES}), encoding="utf-8")
    orig_run_arm = RN.BenchmarkRunner._run_arm

    def run_arm(self, spec, *args, **kwargs):
        kind = spec.arm_id.split("_")[1] if spec.arm_id.startswith("sc_") else ""
        if kind.endswith("7"):
            args = list(args)
            budget_pos = 15                          # posição de ``budget`` em _run_arm depois de ``spec``
            args[budget_pos] = dict(args[budget_pos], max_nodes=SMALL_BUDGET_NODES)
        return orig_run_arm(self, spec, *args, **kwargs)

    RN.BenchmarkRunner._run_arm = run_arm
    SCM.build_arms = build_arms
    SCM.SAMPLER_ARMS = ("B", "B7", "R", "R7", "Rq", "Rq7")
    SCM.plausibility = plausibility
    SCM.run_unit_sampler(dataset, seed, out, a.reps, "none")


if __name__ == "__main__":
    sys.exit(main())
