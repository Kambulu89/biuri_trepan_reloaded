"""Executor dos CONTROLOS ONTOLÓGICOS do Doc. 29 (matriz M29) sob o amostrador C. EXPERIMENTAL; só DEV; uma unidade por processo; retoma de tuning por checkpoint (fingerprint v2).

Braços novos (ids ``m29_{X}_k{k}``, k = 0..K-1 = réplica de amostragem = índice da permutação/draw, emparelhados): C29 ``P`` ontologia permutada; D29 ``D`` grupos aleatórios;
E29 ``E`` engenharia convencional de features; G29 ``G`` Original experimental + C com consumo de consultas IGUALADO ao de B29 (só controlo do efeito do orçamento; não substitui o Original canónico). A29 (Original+C), B29 (Reloaded OWL real+C) e F29 (Reloaded sem OWL+C) são os braços ``sc_B``, ``sc_D``, ``sc_F`` da avaliação do Doc. 26
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

KINDS = ("P", "D", "E", "G")
DEFAULT_SC_DIR = Path(__file__).resolve().parents[2] / "docs/evidence/b1b4/sampler_c/dev_numeric"
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
        arms.append(ArmSpec(f"m29_G_k{k}", "trepan_original", f"G29: Original EXPERIMENTAL + amostrador C com consultas igualadas a B29 (réplica {k}); NÃO substitui o Original canónico",
                            oracle="mlp_original", role="m29_control", group="M29"))
    return arms


def reference_queries(sc_dir: Path, dataset: str, seed: int, k: int):
    """Consultas ao oráculo de B29 (``sc_D_r{k}``) e de A29 (``sc_B_r{k}``) da MESMA unidade/réplica na avaliação do Doc. 26 (só contagens: sem rótulos nem teste)."""
    d = Path(sc_dir) / "raw" / dataset / f"seed{seed}"
    qb = json.loads((d / f"sc_D_r{k}.json").read_text())["row"]["oracle_queries"]
    qa = json.loads((d / f"sc_B_r{k}.json").read_text())["row"]["oracle_queries"]
    return float(qb), float(qa)


def scale_factor(q_target: float, q_ref: float) -> float:
    """Razão pré-definida entre o consumo de consultas de B29 e de A29 (≥ 1: o Original nunca recebe MENOS orçamento do que o seu consumo histórico)."""
    return max(1.0, float(q_target) / max(1.0, float(q_ref)))


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
            n_o = len(real.orig_idx)
            g_real, g_ctx = column_groups(real), column_groups(ctx)
            pm = info.get("perm_meta") or {}
            meta[f"m29_{kind}_k{k}"] = {"kind": tag, "feature_order_ok": bool(list(ctx.enriched_names[:n_o]) == list(names) and list(ctx.base_names) == list(names)),
                                         "fit_on_train_only": True, "features_lost": int(len(real.onto_idx) - len(ctx.onto_idx)),
                                         "columns_whose_group_changed_fraction": float(np.mean([a != b for a, b in zip(g_real, g_ctx)])) if len(g_real) else 0.0,
                                         "co_grouped_pairs_real": int(sum(1 for i in range(n_o) for j in range(i + 1, n_o) if g_real[i] is not None and g_real[i] == g_real[j])),
                                         "selection_criterion": "structural_only (retenção de pares co-agrupados ≤ acaso; sem accuracy/fidelidade/dados de teste)",
                                         "relations_destroyed": "correspondência coluna↔grupo/entidade e composição de cada derivada" if kind == "P" else "correspondência coluna↔entidade, composição das derivadas e acoplamento rótulo↔composição",
                                         "oracle_unchanged": "o MLP só vê as colunas originais (OriginalOracleProjection): a permutação não altera o oráculo", "variant": m["variant"], "attempts": m["attempts"], "validity": m["validity"], "seeds": s,
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


def _sum_counters(tree) -> dict:
    tot: dict = {}
    for r in getattr(tree, "split_audit_", []) or []:
        for k, v in (r.get("candidate_counters") or {}).items():
            if isinstance(v, (int, float)):
                tot[k] = tot.get(k, 0) + v
    return tot


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
            "candidate_counters_total": _sum_counters(tree),
            "efsr": {"attempted": int(getattr(tree, "error_focused_interventions_attempted_", 0) or 0), "accepted": int(getattr(tree, "error_focused_interventions_accepted_", 0) or 0)}}


def _reference_oracle(sc_dir: Path, dataset: str, seed: int) -> dict:
    f = Path(sc_dir) / "raw" / dataset / f"seed{seed}" / "trepan_original.json"
    return {"id": json.loads(f.read_text())["oracle_id"]} if f.exists() else {"id": None}


def run_unit_m29(dataset: str, seed: int, out: Path, reps: int = 3, sc_dir: Path = DEFAULT_SC_DIR) -> Path:
    M.update(unit=f"{dataset}_seed{seed}", out=str(out))
    orig_cfg, orig_groups, orig_default = R.benchmark_config, mf.MAIN_ARM_GROUPS, RN.default_arms
    orig_run_arm, orig_rel, orig_orig = RN.BenchmarkRunner._run_arm, RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier
    arms_new = build_arms(reps)
    cache = {}
    ref_oracle = _reference_oracle(sc_dir, dataset, seed)

    def default_arms_m29(cfg):
        base = [a for a in orig_default(cfg) if a.group in ("A", "B", "C")]                   # MLP, C4.5, Original histórico (integridade); sem Reloaded de referência
        return base + arms_new

    def wrapped(self, spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, oracle_preds, infos, selected, budget, seed_,
                labels, minority, fo=None, fe=None, oracle_train_pred=None, structural=None):
        is_m = spec.arm_id.startswith("m29_")
        is_g = is_m and "_G_k" in spec.arm_id
        if is_m and fo is not None and ref_oracle.get("id") not in (None, fo.oracle_id):                 # identidade do oráculo VERIFICADA EM EXECUÇÃO
            raise RuntimeError(f"oráculo da unidade ({fo.oracle_id}) ≠ oráculo de referência ({ref_oracle['id']}): comparação direta impossível")
        if is_m and not is_g:
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
        g_info = None
        with cm as shared:
            if is_g:
                qb, qa = reference_queries(sc_dir, dataset, seed, replica)
                m0, f0 = int(budget["min_sample"]), scale_factor(qb, qa)
                steps = []
                f = f0
                for step in range(2):                                          # 1.º ajuste descartável (calibração por contagem de consultas), depois o ajuste final
                    b = dict(budget, min_sample=int(np.ceil(m0 * f)), max_queries=max(int(budget["max_queries"]), int(2 * qb)))
                    if step == 0:
                        cal = dataclasses.replace(spec, arm_id=spec.arm_id + "__cal")
                        crow, _ = orig_run_arm(self, cal, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, oracle_preds, infos, selected, b, seed_,
                                               labels, minority, fo, fe, oracle_train_pred=oracle_train_pred, structural=structural)
                        q1 = float(crow["oracle_queries"])
                        steps.append({"min_sample": b["min_sample"], "factor": f, "queries": q1, "target": qb, "deviation": q1 / qb - 1.0})
                        if abs(q1 / qb - 1.0) > 0.05:
                            f = f * qb / max(1.0, q1)
                        M["Q"] = M["rec"] = M["tree"] = None
                    else:
                        row, pred = orig_run_arm(self, spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, oracle_preds, infos, selected, b, seed_,
                                                 labels, minority, fo, fe, oracle_train_pred=oracle_train_pred, structural=structural)
                        steps.append({"min_sample": b["min_sample"], "factor": f, "queries": float(row["oracle_queries"]), "target": qb, "deviation": float(row["oracle_queries"]) / qb - 1.0})
                g_info = {"budget_matched_to": "B29 (sc_D)", "reference_queries_B29": qb, "reference_queries_A29": qa, "initial_factor": f0, "steps": steps,
                          "min_sample_canonical": m0, "min_sample_used": steps[-1]["min_sample"], "final_deviation_from_target": steps[-1]["deviation"],
                          "note": "consumo de consultas igualado; NÃO equivalência total (distribuição das consultas e espaço de candidatos diferem)"}
            else:
                row, pred = orig_run_arm(self, spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, oracle_preds, infos, selected,
                                         budget, seed_, labels, minority, fo, fe, oracle_train_pred=oracle_train_pred, structural=structural)
        if is_m:
            meta = g_info if is_g else cache["meta"][spec.arm_id]
            if not is_g:
                row["ontology_structure_signature"] = ctx.structure_signature()
                row["control_kind"] = meta["kind"]
                row["control_variant"] = meta["variant"]
            else:
                row["control_kind"] = "budget_matched_original"
                row["budget_matched_min_sample"] = g_info["min_sample_used"]
                row["budget_match_deviation"] = g_info["final_deviation_from_target"]
            row["control_meta"] = json.dumps(meta, default=str)
            Q = np.vstack(M["Q"]) if M["Q"] else np.empty((0, 0))
            yq = np.concatenate(M["rec"].y) if M["rec"] is not None and M["rec"].y else np.empty(0)
            rec = {"arm": spec.arm_id, "unit": M["unit"], "control": meta, "sampler": "joint_empirical", "sampler_replica": replica,
                   "sampler_audit": dict(shared.get("audit", {})) if shared else None}
            if len(Q):
                rec["plausibility"] = plausibility(Q, Xtr, mlp_o, Xtr.shape[1], (ctx.enrich if (not is_g and ctx.available) else None), yq, M["otp"])
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


def runtime_comparability(out: Path, sc_dir: Path, dataset: str, seed: int) -> dict:
    """VERIFICAÇÃO EM EXECUÇÃO (logo que a unidade é escrita, antes de qualquer agregação): comparabilidade ESTRITA de cada braço novo com A29/B29/F29 e o C4.5 da referência
    (mesmas instâncias e ordem, rótulos, classes, partição, oráculo e as suas previsões, configuração, orçamento). Falha ⇒ a campanha pára."""
    from core import hierarchy_contract as H
    new_dir = Path(out) / "raw" / dataset / f"seed{seed}"
    ref_dir = Path(sc_dir) / "raw" / dataset / f"seed{seed}"
    load_ = lambda d: {f.stem: json.loads(f.read_text()) for f in d.glob("*.json") if f.name not in ("split.json", "UNIT_SHA256.json")}
    new, ref = load_(new_dir), load_(ref_dir)
    res = {"unit": f"{dataset}:{seed}", "arms": {}, "all_comparable": True}
    base = {k: ref[k] for k in ("c45", "sc_B_r0", "sc_D_r0", "sc_F_r0") if k in ref}
    for arm in sorted(k for k in new if k.startswith("m29_")):
        recs = {**base, arm: new[arm]}
        strict = H.check_comparability(recs)
        relax = H.check_comparability(recs, relaxed=("experimental_configuration",)) if not strict["comparable"] else strict
        res["arms"][arm] = {"strict_comparable": strict["comparable"], "failed_strict": strict.get("failed"), "unverifiable": strict.get("unverifiable"),
                            "comparable_with_config_relaxed": relax["comparable"], "documented_differences": relax.get("documented_differences")}
        res["all_comparable"] &= bool(relax["comparable"])
    for ref_arm in ("trepan_original",):                                                  # Original histórico recomputado: tem de coincidir bit a bit com o da referência
        if ref_arm in new and ref_arm in ref:
            same = all(new[ref_arm]["row"].get(m) == ref[ref_arm]["row"].get(m) for m in ("fidelity_to_oracle", "accuracy_real_labels", "node_count", "oracle_queries"))
            res["arms"][ref_arm] = {"identical_to_reference": bool(same)}
            res["all_comparable"] &= bool(same)
    cdir = Path(out) / "comparability"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / f"{dataset}_seed{seed}.json").write_text(json.dumps(res, indent=1, default=str))
    return res


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
    ap.add_argument("--sc-dir", default=str(DEFAULT_SC_DIR), help="raw da avaliação do Doc. 26 (A29/B29/F29/C4.5 de referência)")
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
    run_unit_m29(a.unit[0], int(a.unit[1]), Path(a.out), a.reps, Path(a.sc_dir))
    res = runtime_comparability(Path(a.out), Path(a.sc_dir), a.unit[0], int(a.unit[1]))
    if not res["all_comparable"]:
        print("COMPARABILIDADE FALHADA:", json.dumps(res, default=str)[:2000], file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    sys.exit(main())
