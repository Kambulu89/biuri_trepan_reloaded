"""Executor da FASE 11 da auditoria (EXPERIMENTAL): complexidade igual (P07) e consultas efetivas dos braços com mecanismos (P02).

Pipeline das fases 7 a 10 (MLP selecionado só no treino; ``--mlp omissao`` usa o MLP por omissão da fase 6), amostrador conjunto,
3 réplicas. Nada no núcleo é alterado. O oráculo de cada árvore é envolvido pelo registador da fase 10 (respostas intactas).

Braços (ids ``sc_{X}_r{r}``, salvo o C4.5 restrito):
  B, R          Original e Reloaded O, estrutura do tuning (iguais às fases 7 a 10)
  B7, R7        idem com no máximo 7 nós
  Bc, Rc        idem com no máximo N nós, N = min(nós do C4.5 canónico DESTA unidade, ajustado só no treino; teto do tuning)
  c45_d2        C4.5 canónico com profundidade máxima 2 (≤ 7 nós): o C4.5 com a complexidade de B7/R7
  E, V, D, A    Reloaded com mecanismos: score + EFSR original (E), EFSR com lote novo (V, subclasse da fase 9),
                Reloaded completo (D), score + consultas ativas (A); só para medir consultas efetivas
  Rnp, R7np     só com ``--mlp omissao``: R sem ``predict_proba`` (réplica da H10.2 com o MLP da fase 6)

    python -m validation.benchmark.run_fase11 --unit digits 42 --out DIR [--mlp {selecionado,omissao}] [--reps 3] [--reuse-checkpoint-from DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from validation.benchmark import run as R      # PRIMEIRO: fixa o ambiente numérico
from core.benchmark import runner as RN
from core.benchmark.runner import ArmSpec
from core.c45_j48_tree import C45Classifier
from validation.benchmark import datasets_adult         # noqa: F401
from validation.benchmark import datasets_categorical  # noqa: F401
from validation.benchmark import datasets_physics      # noqa: F401
from validation.benchmark import efsr_holdout_fase9 as H
from validation.benchmark import run_fase10 as F10
from validation.benchmark import run_fase4 as F4
from validation.benchmark import run_fase5 as F5
from validation.benchmark import run_sampler_c as SCM
from validation.benchmark.run_fase3 import activate_v2
from validation.benchmark.run_fase7 import fit_mlp_fase7

SMALL = 7
STATE = {"c45_nodes": None, "mode": "selecionado"}


class C45Depth2(C45Classifier):
    """C4.5 canónico (mesma poda e parâmetros do braço c45) com profundidade máxima 2: no máximo 7 nós em divisões binárias."""

    def __init__(self, **kw):
        kw["max_depth"] = 2
        super().__init__(**kw)


def _spec_kw(base: str) -> dict:
    kw = dict(F5.SPECS[base][1])
    kw.pop("family", None)
    kw["semantic"] = "real"
    return kw


def build_arms(reps: int, dataset: str = "", c_replicas: str = "none"):
    arms = [ArmSpec("c45_d2", "c45", "C4.5 canónico com profundidade máxima 2 (≤ 7 nós)", group="SC", role="fase11")]
    sel = STATE["mode"] == "selecionado"
    for r in range(reps):
        for suf in (("", "7", "c") if sel else ("", "7")):
            arms.append(ArmSpec(f"sc_B{suf}_r{r}", "trepan_original", f"B{suf}: Original (réplica {r})", oracle="mlp_original",
                                role="fase11", group="SC"))
            arms.append(ArmSpec(f"sc_R{suf}_r{r}", "trepan_reloaded", f"R{suf}: Reloaded O (réplica {r})", tree_space="enr",
                                oracle="mlp_original", needs_semantic=True, role="fase11", group="SC", **_spec_kw("O")))
        if sel:
            for k, base in (("E", "E"), ("V", "E"), ("D", "D"), ("A", "A")):
                arms.append(ArmSpec(f"sc_{k}_r{r}", "trepan_reloaded", f"{k}: Reloaded com mecanismos (réplica {r})", tree_space="enr",
                                    oracle="mlp_original", needs_semantic=True, role="fase11", group="SC", **_spec_kw(base)))
        else:
            for suf in ("", "7"):
                arms.append(ArmSpec(f"sc_R{suf}np_r{r}", "trepan_reloaded", f"R{suf}np: R sem predict_proba (réplica {r})", tree_space="enr",
                                    oracle="mlp_original", needs_semantic=True, role="fase11", group="SC", **_spec_kw("O")))
    return arms


def plausibility(Q, Xtr, mlp, n_o, enrich, yq, otp, seed: int = 0):
    return F4._plausibility(Q, Xtr, mlp, n_o, enrich, yq, otp, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlp", choices=["selecionado", "omissao"], default="selecionado")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--reuse-checkpoint-from", default=None)
    a = ap.parse_args()
    dataset, seed, out = a.unit[0], int(a.unit[1]), Path(a.out)
    STATE["mode"] = a.mlp
    activate_v2(dataset, out)
    if a.reuse_checkpoint_from:
        dst = out / "checkpoints"
        dst.mkdir(parents=True, exist_ok=True)
        for f in (Path(a.reuse_checkpoint_from) / "checkpoints").glob(f"{dataset}_seed{seed}.ckpt*"):
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
    (out / "audits").mkdir(parents=True, exist_ok=True)
    F10.STATE["log"] = str(out / "audits" / f"{dataset}_seed{seed}__fase11_oraculo.jsonl")
    (out / "fase11_config.json").write_text(json.dumps({"mlp": a.mlp, "small_budget_nodes": SMALL}), encoding="utf-8")
    if a.mlp == "selecionado":
        RN.fit_mlp = fit_mlp_fase7
    RN.TrepanReloadedClassifier = H.HoldoutEFSRReloaded       # só atua com H.ACTIVE['on'] (braço V); nos outros delega no Reloaded
    orig_run_arm = RN.BenchmarkRunner._run_arm

    def run_arm(self, spec, *args, **kwargs):
        kind = spec.arm_id.split("_")[1] if spec.arm_id.startswith("sc_") else ""
        base = kind[:-2] if kind.endswith("np") else kind
        if base.endswith("7") or base.endswith("c"):
            cap = SMALL if base.endswith("7") else STATE["c45_nodes"]
            if cap is None:
                raise RuntimeError("número de nós do C4.5 da unidade indisponível")
            args = list(args)
            tuned = args[15].get("max_nodes")                 # posição de ``budget`` em _run_arm depois de ``spec`` (fases 6 a 10)
            if base.endswith("c") and tuned is not None:
                cap = min(int(cap), int(tuned))               # nunca acima da estrutura do tuning: só restringe, nunca alarga
            args[15] = dict(args[15], max_nodes=int(cap))
        F10.STATE.update(arm=spec.arm_id, no_proba=kind.endswith("np"))
        H.ACTIVE.update(on=(base == "V"), arm=spec.arm_id)
        orig_c45 = RN.C45Classifier
        if spec.arm_id == "c45_d2":
            RN.C45Classifier = C45Depth2
        try:
            row, pred = orig_run_arm(self, spec, *args, **kwargs)
        finally:
            RN.C45Classifier = orig_c45
            F10.STATE.update(arm=None, no_proba=False)
            H.ACTIVE.update(on=False, arm=None)
        if spec.arm_id == "c45":
            STATE["c45_nodes"] = int(row["node_count"])
        if spec.arm_id == "c45_d2":
            row["c45_max_depth"] = 2
        return row, pred

    RN.BenchmarkRunner._run_arm = run_arm
    SCM._hook = F10._hook10
    SCM.build_arms = build_arms
    kinds = ("B", "R", "B7", "R7", "Bc", "Rc", "E", "V", "D", "A", "Rnp", "R7np")
    SCM.SAMPLER_ARMS = kinds
    SCM.plausibility = plausibility
    SCM.run_unit_sampler(dataset, seed, out, a.reps, "none")


if __name__ == "__main__":
    sys.exit(main())
