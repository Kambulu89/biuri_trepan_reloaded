"""Executor da FASE 9 da auditoria (EXPERIMENTAL): EFSR com gate validado em consultas novas.

Pipeline das fases 7 e 8 (MLP otimizado por seleção só no treino, amostrador conjunto, 3 réplicas, dois orçamentos).
Braços (ids ``sc_{X}_r{r}``; sufixo «7» = no máximo 7 nós):
  B   Original                                    R   Reloaded O (atributos da ontologia, sem mecanismos; fases 5–8)
  E   Reloaded com canal de score + EFSR original (gate dentro da amostra; braço E da fase 5)
  V   E com o gate do EFSR validado num lote novo (``efsr_holdout_fase9``)
  Vq  V com a ontologia v2 permutada (controlo)
Nada no núcleo é alterado: a subclasse só atua nos braços V/Vq.

    python -m validation.benchmark.run_fase9 --unit digits 42 --out DIR [--reps 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from validation.benchmark import run as R      # PRIMEIRO: fixa o ambiente numérico
from core.benchmark import runner as RN
from core.benchmark.runner import ArmSpec
from validation.benchmark import datasets_adult         # noqa: F401  (regista o Adult)
from validation.benchmark import datasets_categorical  # noqa: F401  (regista german_credit)
from validation.benchmark import datasets_physics      # noqa: F401  (regista os controlos positivos)
from validation.benchmark import run_fase4 as F4
from validation.benchmark import run_fase5 as F5
from validation.benchmark import run_sampler_c as SCM
from validation.benchmark import efsr_holdout_fase9 as H
from validation.benchmark.run_fase3 import activate_v2
from validation.benchmark.run_fase7 import fit_mlp_fase7

SMALL_BUDGET_NODES = 7
KINDS = ("B", "R", "E", "V", "Vq")


def _kw(kind: str) -> dict:
    base = {"R": "O", "E": "E", "V": "E", "Vq": "E"}[kind]
    kw = dict(F5.SPECS[base][1])
    kw.pop("family", None)
    kw["semantic"] = "shuffled" if kind == "Vq" else "real"
    if kind == "Vq":
        kw["control"] = True
    return kw


DESC = {"B": "Original", "R": "Reloaded O (sem mecanismos)", "E": "Reloaded score + EFSR original",
        "V": "Reloaded score + EFSR com gate em lote novo", "Vq": "V com OWL v2 permutada"}


def build_arms(reps: int, dataset: str = "", c_replicas: str = "none"):
    arms = []
    for r in range(reps):
        for suffix in ("", "7"):
            note = f" (máx. {SMALL_BUDGET_NODES} nós)" if suffix else ""
            for kind in KINDS:
                desc = f"{kind}{suffix}: {DESC[kind]} + conjunto{note} (réplica {r})"
                if kind == "B":
                    arms.append(ArmSpec(f"sc_B{suffix}_r{r}", "trepan_original", desc, oracle="mlp_original", role="fase9", group="SC"))
                else:
                    arms.append(ArmSpec(f"sc_{kind}{suffix}_r{r}", "trepan_reloaded", desc, tree_space="enr", oracle="mlp_original",
                                        needs_semantic=True, role="fase9", group="SC", **_kw(kind)))
    return arms


def plausibility(Q, Xtr, mlp, n_o, enrich, yq, otp, seed: int = 0):
    arm = str(SCM.SC.get("arm") or "")
    kind = arm.split("_")[1] if arm.startswith("sc_") else None
    return F4._plausibility(Q, Xtr, mlp, n_o, None if kind and kind.startswith("Vq") else enrich, yq, otp, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    dataset, seed, out = a.unit[0], int(a.unit[1]), Path(a.out)
    activate_v2(dataset, out)
    (out / "fase9_config.json").write_text(json.dumps({"kinds": KINDS, "small_budget_nodes": SMALL_BUDGET_NODES, "alpha": H.ALPHA,
                                                        "seed_tag": H.SEED_TAG}), encoding="utf-8")
    (out / "audits").mkdir(parents=True, exist_ok=True)
    os.environ[H.AUDIT_PATH_ENV] = str(out / "audits" / f"{dataset}_seed{seed}__fase9_efsr.jsonl")
    RN.fit_mlp = fit_mlp_fase7
    RN.TrepanReloadedClassifier = H.HoldoutEFSRReloaded
    orig_run_arm = RN.BenchmarkRunner._run_arm

    def run_arm(self, spec, *args, **kwargs):
        kind = spec.arm_id.split("_")[1] if spec.arm_id.startswith("sc_") else ""
        if kind.endswith("7"):
            args = list(args)
            budget_pos = 15                          # posição de ``budget`` em _run_arm depois de ``spec`` (igual à fase 6)
            args[budget_pos] = dict(args[budget_pos], max_nodes=SMALL_BUDGET_NODES)
        H.ACTIVE.update(on=kind.rstrip("7") in ("V", "Vq"), arm=spec.arm_id)
        try:
            return orig_run_arm(self, spec, *args, **kwargs)
        finally:
            H.ACTIVE.update(on=False, arm=None)

    RN.BenchmarkRunner._run_arm = run_arm
    SCM.build_arms = build_arms
    SCM.SAMPLER_ARMS = tuple(k + s for s in ("", "7") for k in KINDS)
    SCM.plausibility = plausibility
    SCM.run_unit_sampler(dataset, seed, out, a.reps, "none")


if __name__ == "__main__":
    sys.exit(main())
