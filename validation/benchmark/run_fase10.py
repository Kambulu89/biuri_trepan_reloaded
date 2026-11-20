"""Executor da FASE 10 da auditoria (EXPERIMENTAL): orçamento de consultas EFETIVO ao oráculo, com e sem OWL.

As fases 6 a 9 registaram que os braços com OWL contam 0,6 a 5 vezes mais consultas ao oráculo do que o Original, com o mesmo
orçamento configurado. Esta fase mede o que essas consultas são: pontos NOVOS (que a árvore ainda não conhecia) ou reconsultas de
pontos já conhecidos (linhas de treino e consultas já rotuladas); e testa se as reconsultas influenciam alguma decisão.

Pipeline das fases 7 a 9 (MLP selecionado só no treino, amostrador conjunto, 3 réplicas, dois orçamentos). Braços ``sc_{X}_r{r}``:
  B / B7     Original                                   R / R7     Reloaded O (atributos da OWL v2, sem mecanismos)
  Rnp / R7np R com um oráculo SEM ``predict_proba`` (o Reloaded usa então incerteza «indisponível»; nada mais muda)
O oráculo de cada árvore é envolvido por um registador que NÃO altera respostas: guarda, por chamada, o método, o número de linhas e
a impressão digital de cada linha. Nada no núcleo é alterado.

    python -m validation.benchmark.run_fase10 --unit phys_bmi 42 --out DIR [--reps 3] [--reuse-checkpoint-from DIR_ANTERIOR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from validation.benchmark import run as R      # PRIMEIRO: fixa o ambiente numérico
import numpy as np
from core.benchmark import runner as RN
from core.benchmark.runner import ArmSpec
from validation.benchmark import datasets_adult         # noqa: F401  (regista o Adult)
from validation.benchmark import datasets_categorical  # noqa: F401  (regista german_credit)
from validation.benchmark import datasets_physics      # noqa: F401  (regista os controlos positivos)
from validation.benchmark import run_fase4 as F4
from validation.benchmark import run_fase5 as F5
from validation.benchmark import run_sampler_c as SCM
from validation.benchmark.run_causal import _Rec
from validation.benchmark.run_fase3 import activate_v2
from validation.benchmark.run_fase7 import fit_mlp_fase7

SMALL_BUDGET_NODES = 7
KINDS = ("B", "R", "Rnp")
STATE = {"train_keys": None, "arm": None, "log": None, "no_proba": False}


def _keys(X) -> list:
    A = np.ascontiguousarray(np.asarray(X, dtype=float))
    return [hashlib.blake2b(row.tobytes(), digest_size=12).hexdigest() for row in A]


class _CountingRec(_Rec):
    """Regista as linhas enviadas ao oráculo. Respostas e contagens do oráculo congelado ficam intactas."""

    def __init__(self, inner):
        super().__init__(inner)
        self.calls = []

    def _log(self, method, X):
        self.calls.append((method, _keys(X)))

    def predict(self, X):
        self._log("predict", X)
        return super().predict(X)

    def __getattr__(self, name):
        if name == "predict_proba":
            if STATE["no_proba"]:
                raise AttributeError(name)                    # braço Rnp: ``hasattr(oracle, 'predict_proba')`` é falso
            inner = getattr(self.inner, "predict_proba")

            def counted(X, *a, **k):
                self._log("predict_proba", X)
                return inner(X, *a, **k)
            return counted
        return super().__getattr__(name)


def _hook10(cls):
    class Cap(cls):
        def fit(self, X, *a, oracle=None, **k):
            SCM.SC["Q"] = []
            rec = _CountingRec(oracle) if oracle is not None else None
            SCM.SC["rec"] = rec
            STATE["train_keys"] = set(_keys(X))
            out = super().fit(X, *a, oracle=rec if rec is not None else oracle, **k)
            if rec is not None:
                _summarise(rec)
            return out

        def _draw_membership_queries(self, model, n, constraints, node):
            out = super()._draw_membership_queries(model, n, constraints, node)
            if SCM.SC["Q"] is not None and len(out):
                SCM.SC["Q"].append(np.asarray(out, dtype=float))
            return out
    return Cap


def _summarise(rec: _CountingRec) -> None:
    train = STATE["train_keys"] or set()
    seen_label, seen_any = set(), set()
    out = {"arm": STATE["arm"], "calls": {}, "rows": {}, "unique": {}}
    novel_label = novel_proba_only = reproba_known = relabel_known = 0
    for method, keys in rec.calls:
        c = out["calls"].setdefault(method, 0)
        out["calls"][method] = c + 1
        out["rows"][method] = out["rows"].get(method, 0) + len(keys)
        for k in keys:
            known = k in train or k in seen_any
            if method == "predict":
                if not known:
                    novel_label += 1
                elif k in train or k in seen_label:
                    relabel_known += 1
                seen_label.add(k)
            else:
                if not known:
                    novel_proba_only += 1
                else:
                    reproba_known += 1
            seen_any.add(k)
    out["unique"] = {"pontos_novos_rotulados": novel_label, "pontos_novos_so_probabilidade": novel_proba_only,
                     "reconsultas_probabilidade_de_pontos_conhecidos": reproba_known, "reconsultas_rotulo_de_pontos_conhecidos": relabel_known,
                     "linhas_unicas_total": len(seen_any), "linhas_treino_consultadas": len(seen_any & train), "n_treino": len(train)}
    with open(STATE["log"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")


def _kw(kind: str) -> dict:
    kw = dict(F5.SPECS["O"][1])
    kw.pop("family", None)
    kw["semantic"] = "real"
    return kw


def build_arms(reps: int, dataset: str = "", c_replicas: str = "none"):
    arms = []
    for r in range(reps):
        for suffix in ("", "7"):
            note = f" (máx. {SMALL_BUDGET_NODES} nós)" if suffix else ""
            arms.append(ArmSpec(f"sc_B{suffix}_r{r}", "trepan_original", f"B{suffix}: Original + conjunto{note} (réplica {r})",
                                oracle="mlp_original", role="fase10", group="SC"))
            arms.append(ArmSpec(f"sc_R{suffix}_r{r}", "trepan_reloaded", f"R{suffix}: Reloaded O + conjunto{note} (réplica {r})",
                                tree_space="enr", oracle="mlp_original", needs_semantic=True, role="fase10", group="SC", **_kw("R")))
            arms.append(ArmSpec(f"sc_R{suffix}np_r{r}", "trepan_reloaded", f"R{suffix}np: R com oráculo sem predict_proba{note} (réplica {r})",
                                tree_space="enr", oracle="mlp_original", needs_semantic=True, role="fase10", group="SC", **_kw("R")))
    return arms


def plausibility(Q, Xtr, mlp, n_o, enrich, yq, otp, seed: int = 0):
    return F4._plausibility(Q, Xtr, mlp, n_o, enrich, yq, otp, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--reuse-checkpoint-from", default=None)
    a = ap.parse_args()
    dataset, seed, out = a.unit[0], int(a.unit[1]), Path(a.out)
    activate_v2(dataset, out)
    if a.reuse_checkpoint_from:            # tuning determinístico: a chave do checkpoint inclui oráculo, treino e código (inválido => recalcula)
        dst = out / "checkpoints"
        dst.mkdir(parents=True, exist_ok=True)
        for f in (Path(a.reuse_checkpoint_from) / "checkpoints").glob(f"{dataset}_seed{seed}.ckpt*"):
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
    (out / "audits").mkdir(parents=True, exist_ok=True)
    STATE["log"] = str(out / "audits" / f"{dataset}_seed{seed}__fase10_oraculo.jsonl")
    (out / "fase10_config.json").write_text(json.dumps({"kinds": KINDS, "small_budget_nodes": SMALL_BUDGET_NODES}), encoding="utf-8")
    RN.fit_mlp = fit_mlp_fase7
    orig_run_arm = RN.BenchmarkRunner._run_arm

    def run_arm(self, spec, *args, **kwargs):
        kind = spec.arm_id.split("_")[1] if spec.arm_id.startswith("sc_") else ""
        if kind.rstrip("np").endswith("7"):
            args = list(args)
            budget_pos = 15                          # posição de ``budget`` em _run_arm depois de ``spec`` (igual às fases 6 a 9)
            args[budget_pos] = dict(args[budget_pos], max_nodes=SMALL_BUDGET_NODES)
        STATE.update(arm=spec.arm_id, no_proba=kind.endswith("np"))
        try:
            return orig_run_arm(self, spec, *args, **kwargs)
        finally:
            STATE.update(arm=None, no_proba=False)

    RN.BenchmarkRunner._run_arm = run_arm
    SCM._hook = _hook10
    SCM.build_arms = build_arms
    SCM.SAMPLER_ARMS = tuple(k + s for s in ("", "7") for k in ("B", "R")) + ("Rnp", "R7np")
    SCM.plausibility = plausibility
    SCM.run_unit_sampler(dataset, seed, out, a.reps, "none")


if __name__ == "__main__":
    sys.exit(main())
