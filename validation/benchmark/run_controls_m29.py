"""Executor dos CONTROLOS ONTOLÓGICOS do Doc. 29 (matriz M29) sob o amostrador C. EXPERIMENTAL; só DEV; uma unidade por processo; retoma de tuning por checkpoint (fingerprint v2).

Braços novos (ids ``m29_{X}_k{k}``, k = 0..K-1 = réplica de amostragem = índice da permutação/draw, emparelhados): C29 ``P`` ontologia permutada; D29 ``D`` grupos aleatórios;
E29 ``E`` engenharia convencional de features. A29 (Original+C), B29 (Reloaded OWL real+C) e F29 (Reloaded sem OWL+C) são os braços ``sc_B``, ``sc_D``, ``sc_F`` da avaliação do Doc. 26
(não se recomputam: a comparabilidade ESTRITA com eles é verificada). C4.5 e braços históricos vêm da execução de referência.
    python -m validation.benchmark.run_controls_m29 --unit DS SEED --out DIR [--reps 3] [--reuse-checkpoint-from DIR] [--dry-run]
``--dry-run`` constrói e valida só os contextos (sem ajustar árvores nem tocar no tuning): verificação de integridade de baixo custo."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

from validation.benchmark import run as R      # PRIMEIRO: fixa o ambiente numérico
import numpy as np

from core.benchmark import runner as RN
from core.benchmark.runner import ArmSpec
from core.benchmark.semantic import (ConventionalFeatureProvider, OwlSemanticProvider, build_valid_control, column_groups, control_validity)
from core.query_sampling import joint_sampler_context
from validation.benchmark import datasets as ds_mod
from validation.benchmark import datasets_categorical  # noqa: F401  (regista os datasets categóricos)
from validation.benchmark import manifest as mf
from validation.benchmark.run_causal import _Rec
from validation.benchmark.run_sampler_c import plausibility

KINDS = ("P", "D", "E")
M = {"tree": None, "Q": None, "rec": None, "unit": None, "out": None, "arm": None, "Xtr": None, "otp": None}


def seeds_for(kind: str, k: int, master_seed: int):
    """Sementes PRÉ-DEFINIDAS dos controlos (independentes entre si e dos fluxos do amostrador/algoritmo)."""
    if kind == "P":
        return {"seed_a": 20_260_000 + 1000 * k + master_seed, "seed_b": None}
    if kind == "D":
        return {"seed_a": 30_260_000 + 1000 * k + master_seed, "seed_b": 40_260_000 + 1000 * k + master_seed}
    return {}


def build_arms(reps: int):
    arms = []
    for k in range(reps):
        arms.append(ArmSpec(f"m29_P_k{k}", "trepan_reloaded", f"C29: Reloaded + ontologia permutada + amostrador C (draw {k})", tree_space="enr", oracle="mlp_original",
                            semantic="shuffled_structure", lam=1.0, needs_semantic=True, role="m29_control", group="M29", control=True))
        arms.append(ArmSpec(f"m29_D_k{k}", "trepan_reloaded", f"D29: Reloaded + grupos aleatórios + amostrador C (draw {k})", tree_space="enr", oracle="mlp_original",
                            semantic="shuffled_structure", lam=1.0, needs_semantic=True, role="m29_control", group="M29", control=True))
        arms.append(ArmSpec(f"m29_E_k{k}", "trepan_reloaded", f"E29: Reloaded + engenharia convencional de features + amostrador C (réplica {k})", tree_space="enr", oracle="mlp_original",
                            semantic="conventional", lam=1.0, needs_semantic=True, role="m29_control", group="M29", control=True))
    return arms


def build_contexts(dataset: str, Xtr, names, master_seed: int, real, reps: int):
    """Contextos de controlo de uma unidade: (ctxs[arm_id], meta[arm_id]). Porta de validade a priori; fallback explícito."""
    base = OwlSemanticProvider(str(ds_mod.REGISTRY[dataset].ontology_path))
    ctxs, meta = {}, {}
    for k in range(reps):
        for kind, tag in (("P", "permuted"), ("D", "random_groups")):
            s = seeds_for(kind, k, master_seed)
            ctx, m = build_valid_control(tag, base, real, Xtr, names, master_seed, s["seed_a"], s["seed_b"] or 0)
            ctxs[f"m29_{kind}_k{k}"] = ctx
            info = ctx.info
            meta[f"m29_{kind}_k{k}"] = {"kind": tag, "variant": m["variant"], "attempts": m["attempts"], "validity": m["validity"], "seeds": s,
                                         "perm_meta": info.get("perm_meta"), "perm_hash": info.get("perm_hash"), "structure_signature": ctx.structure_signature(),
                                         "n_derived": len(ctx.onto_idx), "group_sizes": sorted(__import__("collections").Counter(g for g in column_groups(ctx) if g).values())}
    conv = ConventionalFeatureProvider(n_target=len(real.onto_idx)).build(Xtr, names)
    for k in range(reps):
        ctxs[f"m29_E_k{k}"] = conv
        meta[f"m29_E_k{k}"] = {"kind": "conventional_features", "variant": "full" if conv.available else "unavailable", "n_derived": len(conv.onto_idx) if conv.available else 0,
                               "n_derived_real": len(real.onto_idx), "dimension_matched": bool(conv.available and len(conv.onto_idx) == len(real.onto_idx)),
                               "note": "n_derived = min(n_OWL, p//2) e só grupos com ≥2 colunas: a dimensão pode diferir da OWL (documentado)"}
    return ctxs, meta


def _hook(cls):
    class Cap(cls):
        def fit(self, X, *a, oracle=None, **k):
            M["Q"] = []
            rec = _Rec(oracle) if oracle is not None else None
            M["rec"] = rec
            r = super().fit(X, *a, oracle=rec if rec is not None else oracle, **k)
            M["tree"] = self
            return r

        def _draw_membership_queries(self, model, n, constraints, node):
            out = super()._draw_membership_queries(model, n, constraints, node)
            if M["Q"] is not None and len(out):
                M["Q"].append(np.asarray(out, dtype=float))
            return out
    return Cap


def _split_audit(tree, n_o: int) -> dict:
    """Instrumentação SÓ de leitura da árvore final (sem efeito nas decisões): resumo por split ontologicamente influenciado."""
    rows = []
    for r in list(getattr(tree, "semantic_split_audit_", []) or []):
        rows.append({k: (v if isinstance(v, (int, float, bool, str, type(None))) else str(v)[:120]) for k, v in r.items()
                     if k in ("node_id", "ontology_influenced", "decision_changed", "decision_reinforced", "semantic_bonus", "semantic_reason", "n_literals", "m", "test", "data_only_score",
                              "final_score", "accepted", "attempted", "derived_literals")})
    splits = [n for n in tree.iter_nodes() if getattr(n, "test", None) is not None]
    return {"n_split_audit_rows": len(rows), "rows": rows[:200], "n_splits": len(splits), "derived_literals": sum(1 for n in splits for l in n.test.literals if l.feature >= n_o),
            "mofn_splits": sum(1 for n in splits if len(n.test.literals) > 1 or n.test.m > 1),
            "efsr": {"attempted": int(getattr(tree, "error_focused_interventions_attempted_", 0) or 0), "accepted": int(getattr(tree, "error_focused_interventions_accepted_", 0) or 0)}}


def run_unit_m29(dataset: str, seed: int, out: Path, reps: int = 3) -> Path:
    M.update(unit=f"{dataset}_seed{seed}", out=str(out))
    orig_cfg, orig_groups, orig_default = R.benchmark_config, mf.MAIN_ARM_GROUPS, RN.default_arms
    orig_run_arm, orig_rel, orig_orig = RN.BenchmarkRunner._run_arm, RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier
    arms_new = build_arms(reps)
    cache = {}

    def default_arms_m29(cfg):
        base = [a for a in orig_default(cfg) if a.group in ("A", "B", "C")]                   # MLP, C4.5, Original histórico (integridade); sem Reloaded de referência
        return base + arms_new

    def wrapped(self, spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, oracle_preds, infos, selected, budget, seed_,
                labels, minority, fo=None, fe=None, oracle_train_pred=None, structural=None):
        is_m = spec.arm_id.startswith("m29_")
        if is_m:
            if "ctxs" not in cache:
                cache["ctxs"], cache["meta"] = build_contexts(dataset, Xtr, list(ds.feature_names), seed, ctx_real, reps)
            ctx = cache["ctxs"][spec.arm_id]
            if not ctx.available:
                raise RuntimeError(f"controlo indisponível: {ctx.reason}")
            self._ctx_variants = {"structure": ctx}              # injeta o contexto de controlo pelo caminho existente do runner (semantic='shuffled_structure')
            self._ctx_conv = ctx
        M.update(arm=spec.arm_id, Xtr=Xtr, otp=(np.asarray(oracle_train_pred) if oracle_train_pred is not None else None))
        replica = int(spec.arm_id.rsplit("_k", 1)[1]) if is_m else 0
        cm = joint_sampler_context(replica=replica) if is_m else contextlib.nullcontext({})
        with cm as shared:
            row, pred = orig_run_arm(self, spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, oracle_preds, infos, selected,
                                     budget, seed_, labels, minority, fo, fe, oracle_train_pred=oracle_train_pred, structural=structural)
        if is_m:
            meta = cache["meta"][spec.arm_id]
            row["ontology_structure_signature"] = ctx.structure_signature()
            row["control_kind"] = meta["kind"]
            row["control_variant"] = meta["variant"]
            row["control_meta"] = json.dumps(meta, default=str)
            Q = np.vstack(M["Q"]) if M["Q"] else np.empty((0, 0))
            yq = np.concatenate(M["rec"].y) if M["rec"] is not None and M["rec"].y else np.empty(0)
            rec = {"arm": spec.arm_id, "unit": M["unit"], "control": meta, "sampler": "joint_empirical", "sampler_replica": replica,
                   "sampler_audit": dict(shared.get("audit", {})) if shared else None}
            if len(Q):
                rec["plausibility"] = plausibility(Q, Xtr, mlp_o, Xtr.shape[1], ctx.enrich if ctx.available else None, yq, M["otp"])
            if M["tree"] is not None:
                rec["tree_audit"] = _split_audit(M["tree"], Xtr.shape[1])
            d = Path(M["out"]) / "audits"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{M['unit']}__{spec.arm_id}.json").write_text(json.dumps(rec, default=str), encoding="utf-8")
        M["Q"] = M["rec"] = M["tree"] = None
        return row, pred

    R.benchmark_config = lambda s: orig_cfg(s)
    mf.MAIN_ARM_GROUPS = tuple(orig_groups) + ("M29",)
    RN.default_arms = default_arms_m29
    RN.BenchmarkRunner._run_arm = wrapped
    RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier = _hook(orig_rel), _hook(orig_orig)
    try:
        return R.run_unit(dataset, seed, out)
    finally:
        R.benchmark_config, mf.MAIN_ARM_GROUPS, RN.default_arms = orig_cfg, orig_groups, orig_default
        RN.BenchmarkRunner._run_arm, RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier = orig_run_arm, orig_rel, orig_orig


def dry_run(dataset: str, seed: int, out: Path, reps: int = 3) -> dict:
    """Constrói a OWL real e todos os contextos de controlo da unidade e escreve os invariantes (dimensão, escala, grupos, retenção, validade). Sem árvores nem tuning."""
    from sklearn.model_selection import train_test_split
    spec = ds_mod.REGISTRY[dataset]
    ds, _t, _c = spec.load()
    Xtr, _Xte, _ytr, _yte = train_test_split(ds.X, ds.y, test_size=0.25, stratify=ds.y, random_state=seed)      # só treino (teste não usado)
    real = OwlSemanticProvider(str(spec.ontology_path)).build(Xtr, ds.feature_names, seed)
    res = {"dataset": dataset, "seed": seed, "real": {"available": real.available, "n_derived": len(real.onto_idx) if real.available else 0,
                                                       "structure_signature": real.structure_signature() if real.available else None,
                                                       "ontology_hash": real.info.get("ontology_hash"), "state_reason": real.reason}}
    if real.available:
        ctxs, meta = build_contexts(dataset, Xtr, list(ds.feature_names), seed, real, reps)
        res["controls"] = {}
        for aid, c in ctxs.items():
            v = control_validity(real, c, Xtr) if c.available else {"valid": False, "reasons": ["indisponível"]}
            res["controls"][aid] = {**meta[aid], "validity_check": v, "differs_from_real_structure": bool(c.structure_signature() != real.structure_signature())}
        sigs = [res["controls"][a]["structure_signature"] for a in res["controls"] if a.startswith(("m29_P", "m29_D"))]
        res["distinct_control_structures"] = len(set(sigs))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{dataset}_seed{seed}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--reuse-checkpoint-from", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        dry_run(a.unit[0], int(a.unit[1]), Path(a.out), a.reps)
        return
    if a.reuse_checkpoint_from:
        import shutil
        src = Path(a.reuse_checkpoint_from) / "checkpoints"
        dst = Path(a.out) / "checkpoints"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob(f"{a.unit[0]}_seed{a.unit[1]}.ckpt*"):
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
    run_unit_m29(a.unit[0], int(a.unit[1]), Path(a.out), a.reps)


if __name__ == "__main__":
    sys.exit(main())
