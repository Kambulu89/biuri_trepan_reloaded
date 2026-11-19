"""Executor da FASE 5 da auditoria (EXPERIMENTAL; só DEV): ablação dos mecanismos do Reloaded e candidatas a correção, no espaço
enriquecido pela OWL v2, com o amostrador conjunto (3 réplicas). Reutiliza ``run_fase4`` (contexto convencional, auditoria das consultas) e
``run_sampler_c``, sem os alterar.

Braços (ids ``sc_{X}_r{r}``; todos no espaço OWL v2 exceto B):
  B  Original, espaço original (integridade face às fases 2–4)
  O  sem mecanismos: λ=0, refinamento focado no erro (EFSR) desligado, consultas ativas desligadas
  W  só canal de score: λ=1 (peso de ganho, Gain_Reloaded, grupos/relatedness nos candidatos m-of-n), EFSR e consultas ativas desligados
  E  canal de score + EFSR: λ=1, EFSR ligado, consultas ativas desligadas
  A  canal de score + consultas ativas: λ=1, EFSR desligado, consultas ativas ligadas
     (com λ=0 o EFSR e as consultas ativas não atuam: dependem da estrutura semântica, que só é passada com λ>0 — verificado no teste
     de funcionamento, em que E e A com λ=0 deram árvores idênticas a O; por isso cada um é medido como acréscimo sobre W)
  D  Reloaded completo: λ=1, EFSR e consultas ativas ligados (= D2 da fase 3; integridade)
  T  correção «desempate»: λ=1 com ``semantic_calibration='tiebreak'`` (a semântica só reordena candidatos estatisticamente indistinguíveis),
     EFSR e consultas ativas desligados
  L  correção «limitada»: λ=1 com ``semantic_calibration='bounded'``, teto λcap=0,2, EFSR e consultas ativas desligados
  Tq, Lq  T e L com a ontologia permutada (controlo negativo)

    python -m validation.benchmark.run_fase5 --unit iris 42 --out DIR [--reps 3] [--reuse-checkpoint-from DIR]
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
from pathlib import Path

from validation.benchmark import run as R      # PRIMEIRO: fixa o ambiente numérico
from core.benchmark.runner import ArmSpec
from validation.benchmark import run_fase4 as F4
from validation.benchmark import run_sampler_c as SCM
from validation.benchmark.run_fase3 import activate_v2

# kind -> (descrição, kwargs do ArmSpec)
SPECS = {
    "B": ("Original, espaço original", dict(family="trepan_original")),
    "O": ("sem mecanismos", dict(lam=0.0, error_focus=False, active_queries=False)),
    "W": ("só canal de score (λ=1)", dict(lam=1.0, error_focus=False, active_queries=False)),
    "E": ("score + EFSR", dict(lam=1.0, error_focus=True, active_queries=False)),
    "A": ("score + consultas ativas", dict(lam=1.0, error_focus=False, active_queries=True)),
    "D": ("Reloaded completo", dict(lam=1.0, error_focus=True, active_queries=True)),
    "T": ("correção: desempate", dict(lam=1.0, error_focus=False, active_queries=False, calibration="tiebreak", lambda_cap=0.0)),
    "L": ("correção: limitada λcap=0,2", dict(lam=1.0, error_focus=False, active_queries=False, calibration="bounded", lambda_cap=0.2)),
    "Tq": ("CONTROLO: desempate com OWL permutada", dict(lam=1.0, error_focus=False, active_queries=False, calibration="tiebreak",
                                                        lambda_cap=0.0, semantic="shuffled", control=True)),
    "Lq": ("CONTROLO: limitada com OWL permutada", dict(lam=1.0, error_focus=False, active_queries=False, calibration="bounded",
                                                       lambda_cap=0.2, semantic="shuffled", control=True)),
}


def build_arms(reps: int, dataset: str = "", c_replicas: str = "none"):
    arms = []
    for r in range(reps):
        for kind, (desc, kw) in SPECS.items():
            kw = dict(kw)
            if kw.pop("family", None) == "trepan_original":
                arms.append(ArmSpec(f"sc_{kind}_r{r}", "trepan_original", f"{kind}: {desc} + amostrador conjunto (réplica {r})",
                                    oracle="mlp_original", role="fase5", group="SC"))
                continue
            kw.setdefault("semantic", "real")
            arms.append(ArmSpec(f"sc_{kind}_r{r}", "trepan_reloaded", f"{kind}: {desc} + amostrador conjunto (réplica {r})", tree_space="enr",
                                oracle="mlp_original", needs_semantic=True, role="fase5", group="SC", **kw))
    return arms


def plausibility(Q, Xtr, mlp, n_o, enrich, yq, otp, seed: int = 0):
    arm = str(SCM.SC.get("arm") or "")
    kind = arm.split("_")[1] if arm.startswith("sc_") else None
    return F4._plausibility(Q, Xtr, mlp, n_o, None if kind in ("Tq", "Lq") else enrich, yq, otp, seed)


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
    SCM.build_arms = build_arms
    SCM.SAMPLER_ARMS = tuple(SPECS)
    SCM.plausibility = plausibility
    SCM.run_unit_sampler(dataset, seed, out, a.reps, "none")


if __name__ == "__main__":
    sys.exit(main())
