"""Executor da AVALIAÇÃO EMPARELHADA do amostrador C (EXPERIMENTAL; Doc. 26). Só datasets DEV; uma unidade por processo; retoma de tuning por *checkpoint* (fingerprint v2).

Braços (ids ``sc_{X}_r{r}``): A Original histórico (referência r=0 = ``trepan_original``; réplicas r>0 por ``seed_offset``); B Original + amostrador C; C Reloaded OWL
histórico (r=0 = ``reloaded_owl_full``; r>0 = células ``cx_h_on_n_r{r}`` do Doc. 23); D Reloaded OWL + amostrador C; E Reloaded sem OWL (r=0 = ``reloaded_core``; réplicas
``sc_E``); F Reloaded sem OWL + amostrador C. Sementes: de TREINO/partição = semente mestra (MLP, divisão); do ALGORITMO = semente mestra (+offset só nas réplicas dos
braços com amostrador histórico, onde o RNG do amostrador e o do algoritmo são o mesmo fluxo); de AMOSTRAGEM do amostrador C = réplica r∈{0,1,2} (fixadas antes da execução).
O amostrador C é ativado SÓ dentro do ajuste final do braço (``joint_sampler_context``); o tuning estrutural (checkpoint) nunca o vê.
    python -m validation.benchmark.run_sampler_c --unit iris 42 --out DIR [--reps 3 --reuse-checkpoint-from DIR]
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
from pathlib import Path

from validation.benchmark import run as R      # PRIMEIRO: fixa o ambiente numérico
import numpy as np

from core.benchmark import runner as RN
from core.benchmark.runner import ArmSpec
from core.query_sampling import joint_sampler_context
from validation.benchmark import manifest as mf
from validation.benchmark.run_causal import _Rec

SC: dict = {"Q": None, "rec": None, "ctx": None, "Xtr": None, "otp": None, "arm": None, "unit": None, "out": None}
SAMPLER_ARMS = ("B", "D", "F")


def build_arms(reps: int, dataset: str = "", c_replicas: str = "auto"):
    """``c_replicas``: ``auto`` = só o que falta (digits r=2; as restantes réplicas de C vêm das células cx_h_on_n do Doc. 23); ``all`` = r>0 de C (verificação de identidade); ``none``."""
    arms = []
    c_rs = [r for r in range(1, reps) if c_replicas == "all" or (c_replicas == "auto" and dataset == "digits" and r == 2)]
    for r in c_rs:
        arms.append(ArmSpec(f"sc_C_r{r}", "trepan_reloaded", f"C: Reloaded OWL histórico, réplica {r} (seed_offset; = cx_h_on_n_r{r})", tree_space="enr", oracle="mlp_original",
                            semantic="real", lam=1.0, needs_semantic=True, seed_offset=1000 * r, role="sampler_c", group="SC"))
    for r in range(reps):
        arms.append(ArmSpec(f"sc_B_r{r}", "trepan_original", f"B: Original + amostrador C (réplica de amostragem {r})", oracle="mlp_original", role="sampler_c", group="SC"))
        arms.append(ArmSpec(f"sc_D_r{r}", "trepan_reloaded", f"D: Reloaded OWL + amostrador C (réplica {r})", tree_space="enr", oracle="mlp_original", semantic="real",
                            lam=1.0, needs_semantic=True, role="sampler_c", group="SC"))
        arms.append(ArmSpec(f"sc_F_r{r}", "trepan_reloaded", f"F: Reloaded sem OWL + amostrador C (réplica {r})", oracle="mlp_original", semantic="none", lam=1.0,
                            role="sampler_c", group="SC"))
        if r > 0:
            arms.append(ArmSpec(f"sc_A_r{r}", "trepan_original", f"A: Original histórico, réplica {r} (seed_offset)", oracle="mlp_original", seed_offset=1000 * r,
                                role="sampler_c", group="SC"))
            arms.append(ArmSpec(f"sc_E_r{r}", "trepan_reloaded", f"E: Reloaded sem OWL histórico, réplica {r} (seed_offset)", oracle="mlp_original", semantic="none", lam=1.0,
                                seed_offset=1000 * r, role="sampler_c", group="SC"))
    return arms


def _hook(cls):
    class Cap(cls):
        def fit(self, X, *a, oracle=None, **k):
            SC["Q"] = []
            rec = _Rec(oracle) if oracle is not None else None
            SC["rec"] = rec
            return super().fit(X, *a, oracle=rec if rec is not None else oracle, **k)

        def _draw_membership_queries(self, model, n, constraints, node):
            out = super()._draw_membership_queries(model, n, constraints, node)
            if SC["Q"] is not None and len(out):
                SC["Q"].append(np.asarray(out, dtype=float))
            return out
    return Cap


def plausibility(Q, Xtr, mlp, n_o: int, enrich, yq, otp, seed: int = 0) -> dict:
    """Plausibilidade/cobertura/diversidade das consultas EFETIVAS da árvore (nível de linhas; sem AUC: as linhas reais do treino foram vistas pelo amostrador)."""
    from scipy.stats import ks_2samp
    from sklearn.neighbors import NearestNeighbors
    from core.query_sampling import infer_schema
    Xo = np.asarray(Xtr, dtype=float)[:, :n_o]
    Qo = Q[:, :n_o]
    sd = np.maximum(Xo.std(axis=0), 1e-12)
    Z = lambda A: (np.asarray(A) - Xo.mean(0)) / sd
    nn = NearestNeighbors(n_neighbors=2).fit(Z(Xo))
    d_train = nn.kneighbors(Z(Xo), n_neighbors=2)[0][:, 1]
    sub = Qo[:: max(1, len(Qo) // 1000)][:1000]
    d_q = nn.kneighbors(Z(sub), n_neighbors=1)[0][:, 0]
    nnq = NearestNeighbors(n_neighbors=2).fit(Z(sub))
    d_qq = nnq.kneighbors(Z(sub), n_neighbors=2)[0][:, 1]
    schema = infer_schema(Xo)
    rows = {tuple(np.round(r, 9)) for r in Xo}
    out = {"n_query_rows": int(len(Q)),
           "ks_original_mean": float(np.mean([ks_2samp(Qo[:: max(1, len(Qo) // 1500), j], Xo[:, j]).statistic for j in range(n_o)])),
           "out_of_range_rate": float(((Qo < Xo.min(0)) | (Qo > Xo.max(0))).any(axis=1).mean()),
           "exact_copy_of_train_rows": float(np.mean([tuple(np.round(r, 9)) in rows for r in Qo[:1000]])),
           "unique_query_rows_fraction": float(len({tuple(np.round(r, 9)) for r in Qo[:2000]}) / min(len(Qo), 2000)),
           "nn_distance_ratio_queries_to_train": float(np.median(d_q) / max(1e-12, np.median(d_train))),
           "diversity_nn_distance_ratio": float(np.median(d_qq) / max(1e-12, np.median(d_train))),
           "onehot_violation_rate": float(np.any([Qo[:, g].sum(1) != 1.0 for g in schema.onehot_groups], axis=0).mean()) if schema.onehot_groups else 0.0,
           "vocab_violation_rate": float(np.any([~np.isin(Qo[:, j], schema.vocab[j]) for j in schema.discrete], axis=0).mean()) if schema.discrete else 0.0,
           "binary_violation_rate": float(np.any([~np.isin(Qo[:, j], [0.0, 1.0]) for j in schema.binary], axis=0).mean()) if schema.binary else 0.0}
    if enrich is not None and Q.shape[1] > n_o:
        out["derived_violation_rate"] = float((np.abs(Q - np.asarray(enrich(Qo))).max(axis=1) > 1e-9).mean())
    if hasattr(mlp, "predict_proba"):
        def margin(A):
            s = np.sort(np.asarray(mlp.predict_proba(A)), axis=1)
            return s[:, -1] - s[:, -2]
        mq, mt = margin(Qo), margin(Xo)
        out["near_boundary_fraction_margin<0.2"] = float((mq < 0.2).mean())
        ref = Z(Xo)[mt < 0.3]
        if len(ref):
            rho = float(np.median(d_train))
            nq = NearestNeighbors(n_neighbors=1).fit(Z(sub))
            out["boundary_points_covered_fraction"] = float((nq.kneighbors(ref, n_neighbors=1)[0][:, 0] <= rho).mean())
    if len(yq) and otp is not None:
        cls = np.unique(np.concatenate([yq, otp]))
        pr = np.array([(otp == c).mean() for c in cls]); ps = np.array([(yq == c).mean() for c in cls])
        out.update(label_tv_vs_real=float(0.5 * np.abs(pr - ps).sum()), label_entropy=float(-(ps[ps > 0] * np.log2(ps[ps > 0])).sum()))
    return out


def _tree_stats(tree, n_o: int) -> dict:
    splits = [n for n in tree.iter_nodes() if getattr(n, "test", None) is not None]
    return {"n_splits": len(splits), "mofn_splits": sum(1 for n in splits if len(n.test.literals) > 1 or n.test.m > 1),
            "mean_literals": float(np.mean([len(n.test.literals) for n in splits])) if splits else 0.0,
            "derived_literals": sum(1 for n in splits for l in n.test.literals if l.feature >= n_o),
            "total_literals": sum(len(n.test.literals) for n in splits)}


def run_unit_sampler(dataset: str, seed: int, out: Path, reps: int = 3, c_replicas: str = "auto") -> Path:
    SC.update(unit=f"{dataset}_seed{seed}", out=str(out))
    orig_cfg, orig_groups, orig_default = R.benchmark_config, mf.MAIN_ARM_GROUPS, RN.default_arms
    orig_run_arm, orig_rel, orig_orig = RN.BenchmarkRunner._run_arm, RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier
    arms_new = build_arms(reps, dataset, c_replicas)

    def default_arms_sc(cfg):
        base = [a for a in orig_default(cfg) if a.group in ("A", "B", "C", "D", "E")]      # MLP, C4.5, Original, Reloaded Core, Reloaded OWL (integridade/referência r=0)
        return base + arms_new

    def wrapped(self, spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, oracle_preds, infos, selected, budget, seed_,
                labels, minority, fo=None, fe=None, oracle_train_pred=None, structural=None):
        is_sc = spec.arm_id.startswith("sc_")
        SC.update(arm=spec.arm_id, Xtr=Xtr, otp=(np.asarray(oracle_train_pred) if oracle_train_pred is not None else None))
        kind = spec.arm_id.split("_")[1] if is_sc else None
        replica = int(spec.arm_id.rsplit("_r", 1)[1]) if is_sc else 0
        cm = joint_sampler_context(replica=replica) if (is_sc and kind in SAMPLER_ARMS) else contextlib.nullcontext({})
        with cm as shared:
            row, pred = orig_run_arm(self, spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, oracle_preds, infos, selected,
                                     budget, seed_, labels, minority, fo, fe, oracle_train_pred=oracle_train_pred, structural=structural)
        if spec.family in ("trepan_original", "trepan_reloaded") and SC["Q"]:
            Q = np.vstack(SC["Q"])
            yq = np.concatenate(SC["rec"].y) if SC["rec"] is not None and SC["rec"].y else np.empty(0)
            n_o = Xtr.shape[1]
            ctx = ctx_real if spec.tree_space == "enr" else None
            rec = {"arm": spec.arm_id, "unit": SC["unit"], "sampler": "joint_empirical" if (is_sc and kind in SAMPLER_ARMS) else "marginal",
                   "sampler_replica": replica if (is_sc and kind in SAMPLER_ARMS) else None, "sampler_audit": dict(shared.get("audit", {})) if shared else None,
                   "plausibility": plausibility(Q, Xtr, mlp_o, n_o, ctx.enrich if ctx is not None and ctx.available else None, yq, SC["otp"])}
            d = Path(SC["out"]) / "audits"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{SC['unit']}__{spec.arm_id}.json").write_text(json.dumps(rec, default=str), encoding="utf-8")
        SC["Q"] = SC["rec"] = None
        return row, pred

    R.benchmark_config = lambda s: orig_cfg(s)
    mf.MAIN_ARM_GROUPS = tuple(orig_groups) + ("SC",)
    RN.default_arms = default_arms_sc
    RN.BenchmarkRunner._run_arm = wrapped
    RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier = _hook(orig_rel), _hook(orig_orig)
    try:
        return R.run_unit(dataset, seed, out)
    finally:
        R.benchmark_config, mf.MAIN_ARM_GROUPS, RN.default_arms = orig_cfg, orig_groups, orig_default
        RN.BenchmarkRunner._run_arm, RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier = orig_run_arm, orig_rel, orig_orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--reuse-checkpoint-from", default=None)
    ap.add_argument("--c-replicas", default="auto", choices=["auto", "all", "none"])
    a = ap.parse_args()
    if a.reuse_checkpoint_from:
        import shutil
        src = Path(a.reuse_checkpoint_from) / "checkpoints"
        dst = Path(a.out) / "checkpoints"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob(f"{a.unit[0]}_seed{a.unit[1]}.ckpt*"):
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
    run_unit_sampler(a.unit[0], int(a.unit[1]), Path(a.out), a.reps, a.c_replicas)


if __name__ == "__main__":
    sys.exit(main())
