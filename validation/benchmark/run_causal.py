"""Executor da análise causal do ganho D − C e dos controlos P2 (EXPERIMENTAL; Doc. 23). Só datasets DEV; uma unidade por processo; retoma de tuning por
*checkpoint* (fingerprint v2). Pré-registo: docs/b1b4/23_ANALISE_CAUSAL_GANHO_COERENCIA.md §0.

Braços (grupo CX), nomes ``cx_{S}_{A}_{L}_r{r}``: S∈{h,c} (sorteio histórico/coerente) · A∈{on,off} (consultas ativas) · L∈{n,f} (família do teste de adoção do
modelo local: n = nativo da variante; f = trocado). r = réplica (random_state = semente mestra + 1000·r). Controlos P2: ``p2_structshuf_{A}_r{r}`` (features
verdadeiras, estrutura permutada) e ``p2_featshuf_{A}_r{r}`` (features permutadas, estrutura verdadeira), coerentes.
A família do teste de adoção é um parcheamento TRANSITÓRIO fora do núcleo (``core/`` intacto), ativo só dentro do ajuste do braço.
    python -m validation.benchmark.run_causal --unit iris 42 --out DIR [--small-reps 3 --digits-reps 2]
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
from core import trepan_original as TO
from core import trepan_reloaded_historical as TRH
from validation.benchmark import manifest as mf

CAP: dict = {"arm": None, "unit": None, "out": None, "ctx": None, "Xtr": None, "queries": None, "labels": None, "ldiff": None, "otp": None}
FDM = TO.FeatureDistributionModel
_ORIG_DIFFERS = FDM.differs_from


def _counting_differs(self, reference, X_local, alpha=0.10):
    r = _ORIG_DIFFERS(self, reference, X_local, alpha)
    if CAP["ldiff"] is not None and reference is not None and len(X_local) >= 8:
        CAP["ldiff"][0] += 1
        CAP["ldiff"][1] += int(bool(r))
    return r


def _differs_cols(self, reference, X_local, alpha=0.10, cols=None):
    """Cópia fiel de ``FeatureDistributionModel.differs_from`` com a FAMÍLIA de colunas testadas restrita a ``cols`` (e a correção de Bonferroni sobre essa família)."""
    if reference is None or len(X_local) < 8:
        return False
    X_local = np.asarray(X_local, dtype=float)
    corrected = alpha / max(1, len(cols))
    for j in cols:
        spec = self.features_[j]
        local = X_local[:, j]
        local = local[np.isfinite(local)]
        global_col = reference.training_columns_[j]
        if len(local) < 4:
            continue
        if spec.discrete and TO.chi2_contingency is not None:
            vals = np.union1d(np.unique(local), np.unique(global_col))
            a = np.array([(local == v).sum() for v in vals], dtype=float)
            b = np.array([(global_col == v).sum() for v in vals], dtype=float)
            table = np.vstack([a, b])
            valid = table.sum(axis=0) > 0
            if valid.sum() >= 2:
                _, p, _, _ = TO.chi2_contingency(table[:, valid], correction=False)
                if p < corrected:
                    return True
        elif TO.ks_2samp is not None:
            if TO.ks_2samp(local, global_col).pvalue < corrected:
                return True
    return False


class _CohAllFamily(TRH._OriginalSpaceModel):
    """Modelo coerente cujo teste de adoção usa a família ENRIQUECIDA (originais+derivadas), como o histórico. Só para o fator L (intervenção experimental)."""
    def fit(self, X):
        self._fit_rows = np.asarray(X, dtype=float)
        return super().fit(X)

    def differs_from(self, reference, X_local, alpha=0.10):
        if reference is None or len(X_local) < 8:
            return False
        ref_rows = getattr(reference, "_fit_rows", None)
        if ref_rows is None:
            return super().differs_from(reference, X_local, alpha)
        cand = FDM(random_state=self.random_state).fit(self._lift(self._fit_rows[:, self.idx]))
        ref = FDM(random_state=self.random_state).fit(self._lift(ref_rows[:, self.idx]))
        return cand.differs_from(ref, np.asarray(X_local, dtype=float), alpha=alpha)


@contextlib.contextmanager
def family_patch(arm_id: str, orig_idx):
    """L trocado: histórico ⇒ família = originais; coerente ⇒ família = enriquecida. Nativo ⇒ sem parcheamento."""
    parts = arm_id.split("_") if arm_id.startswith("cx_") else []
    S, L = (parts[1], parts[3]) if len(parts) == 5 else (None, None)
    if L != "f":
        yield
        return
    if S == "h":
        cols = [int(i) for i in orig_idx]
        FDM.differs_from = lambda self, reference, X_local, alpha=0.10: (
            (lambda r: (CAP["ldiff"].__setitem__(0, CAP["ldiff"][0] + 1) if (CAP["ldiff"] is not None and reference is not None and len(X_local) >= 8) else None,
                        CAP["ldiff"].__setitem__(1, CAP["ldiff"][1] + int(bool(r))) if (CAP["ldiff"] is not None and reference is not None and len(X_local) >= 8) else None, r)[-1])(
                _differs_cols(self, reference, X_local, alpha, cols)))
        try:
            yield
        finally:
            FDM.differs_from = _counting_differs
    else:
        old = TRH._OriginalSpaceModel
        TRH._OriginalSpaceModel = _CohAllFamily
        try:
            yield
        finally:
            TRH._OriginalSpaceModel = old


class _Rec:
    """Invólucro do oráculo que regista rótulos das consultas (sem alterar respostas nem contagens; comportamento do oráculo, H-O)."""
    def __init__(self, inner):
        self.inner, self.y = inner, []
        self.classes_ = getattr(inner, "classes_", None)

    def predict(self, X):
        out = self.inner.predict(X)
        self.y.append(np.asarray(out))
        return out

    def __getattr__(self, name):
        if name in ("inner", "y") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)


def _dump(tree, kind: str) -> None:
    if CAP["out"] is None or CAP["arm"] is None:
        return
    ctx, Xtr = CAP["ctx"], CAP["Xtr"]
    Q = np.vstack(CAP["queries"]) if CAP["queries"] else np.empty((0, 0))
    yq = np.concatenate(CAP["labels"].y) if CAP["labels"] is not None and CAP["labels"].y else np.empty(0)
    rec = {"arm": CAP["arm"], "unit": CAP["unit"], "n_query_rows": int(len(Q)), "local_model_calls": CAP["ldiff"][0] if CAP["ldiff"] else None,
           "local_model_adopted": CAP["ldiff"][1] if CAP["ldiff"] else None, "membership_queries": int(getattr(tree, "membership_queries_", 0)),
           "timing": {k: float(getattr(tree, k, 0.0) or 0.0) for k in ("training_time_", "split_search_time_", "m_of_n_search_time_", "query_time_")},
           "summary": tree.semantic_audit_summary() if hasattr(tree, "semantic_audit_summary") else None}
    n_o = len(ctx.orig_idx) if ctx is not None else (Q.shape[1] if len(Q) else 0)
    splits = [n for n in tree.iter_nodes() if getattr(n, "test", None) is not None]
    rec["tree"] = {"n_splits": len(splits), "mofn_splits": sum(1 for n in splits if len(n.test.literals) > 1 or n.test.m > 1),
                   "mean_literals": float(np.mean([len(n.test.literals) for n in splits])) if splits else 0.0,
                   "derived_literals": sum(1 for n in splits for l in n.test.literals if l.feature >= n_o),
                   "total_literals": sum(len(n.test.literals) for n in splits),
                   "root_features": sorted({int(l.feature) for l in splits[0].test.literals}) if splits else []}
    if len(Q) and Xtr is not None:
        from scipy.stats import ks_2samp
        Xo = np.asarray(Xtr, dtype=float)
        rec["queries"] = {"ks_original_mean": float(np.mean([ks_2samp(Q[:: max(1, len(Q) // 1500), j], Xo[:, j]).statistic for j in range(n_o)])),
                          "out_of_range_rate": float(((Q[:, :n_o] < Xo.min(0)) | (Q[:, :n_o] > Xo.max(0))).any(axis=1).mean())}
        if len(yq) and CAP["otp"] is not None:
            cls = np.unique(np.concatenate([yq, CAP["otp"]]))
            pr = np.array([(CAP["otp"] == c).mean() for c in cls]); ps = np.array([(yq == c).mean() for c in cls])
            rec["queries"].update(label_tv_vs_real=float(0.5 * np.abs(pr - ps).sum()), label_entropy=float(-(ps[ps > 0] * np.log2(ps[ps > 0])).sum()))
    d = Path(CAP["out"]) / "audits"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{CAP['unit']}__{CAP['arm']}.json").write_text(json.dumps(rec, default=str), encoding="utf-8")
    CAP["queries"] = CAP["labels"] = CAP["ldiff"] = None


def _hook(cls):
    class Cap(cls):
        def fit(self, X, *a, oracle=None, **k):
            CAP["queries"] = []
            CAP["ldiff"] = [0, 0]
            rec = _Rec(oracle) if oracle is not None else None
            CAP["labels"] = rec
            r = super().fit(X, *a, oracle=rec if rec is not None else oracle, **k)
            _dump(self, "tree")
            return r

        def _draw_membership_queries(self, model, n, constraints, node):
            out = super()._draw_membership_queries(model, n, constraints, node)
            if CAP["queries"] is not None and len(out):
                CAP["queries"].append(np.asarray(out, dtype=float))
            return out
    return Cap


def build_arms(small_reps: int, digits_reps: int, dataset: str, include_native_r0: bool = False):
    reps = digits_reps if dataset == "digits" else small_reps
    arms = []
    mk = lambda aid, desc, **kw: ArmSpec(aid, "trepan_reloaded", desc, tree_space="enr", oracle="mlp_original", needs_semantic=True, role="causal_cell", group="CX", **kw)
    for S in ("h", "c"):
        for A in ("on", "off"):
            for L in ("n", "f"):
                for r in range(reps):
                    if S == "h" and L == "n" and False:
                        continue
                    if L == "n" and r == 0 and not include_native_r0:
                        continue                     # células nativas r=0: já medidas (benchmark da amostragem coerente, bit-a-bit reutilizáveis)
                    arms.append(mk(f"cx_{S}_{A}_{L}_r{r}", f"S={S} A={A} L={L} r={r}", semantic="real", lam=1.0, active_queries=(A == "on"),
                                   coherent=(S == "c"), seed_offset=1000 * r))
    p2_reps = 1 if dataset == "digits" else 2
    for mode in ("structure", "features"):
        for A in ("on", "off"):
            for r in range(p2_reps):
                arms.append(mk(f"p2_{mode[:4]}shuf_{A}_r{r}", f"P2 {mode} permutado A={A} r={r}", semantic=f"shuffled_{mode}", lam=1.0, active_queries=(A == "on"),
                               coherent=True, seed_offset=1000 * r, control=True))
    return arms


def run_unit_causal(dataset: str, seed: int, out: Path, small_reps: int = 3, digits_reps: int = 2, include_native_r0: bool = False) -> Path:
    CAP.update(unit=f"{dataset}_seed{seed}", out=str(out))
    orig_cfg, orig_groups, orig_default = R.benchmark_config, mf.MAIN_ARM_GROUPS, RN.default_arms
    orig_run_arm, orig_rel, orig_orig = RN.BenchmarkRunner._run_arm, RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier
    arms_new = build_arms(small_reps, digits_reps, dataset, include_native_r0)

    def default_arms_causal(cfg):
        base = [a for a in orig_default(cfg) if a.group in ("A", "B", "C", "D")]          # MLP, C4.5, Original, Reloaded Core (integridade/referência)
        return base + arms_new

    def wrapped(self, spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, oracle_preds, infos, selected, budget, seed_,
                labels, minority, fo=None, fe=None, oracle_train_pred=None, structural=None):
        CAP["arm"], CAP["Xtr"], CAP["otp"] = spec.arm_id, Xtr, (np.asarray(oracle_train_pred) if oracle_train_pred is not None else None)
        if spec.semantic == "shuffled":
            CAP["ctx"] = ctx_shuf
        elif spec.semantic in ("shuffled_structure", "shuffled_features"):
            CAP["ctx"] = self._ctx_variants[spec.semantic.split("_", 1)[1]]
        else:
            CAP["ctx"] = ctx_real if spec.tree_space == "enr" else None
        with family_patch(spec.arm_id, ctx_real.orig_idx):
            return orig_run_arm(self, spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, oracle_preds, infos, selected,
                                budget, seed_, labels, minority, fo, fe, oracle_train_pred=oracle_train_pred, structural=structural)

    R.benchmark_config = lambda s: dataclasses.replace(orig_cfg(s), causal_pilot=True)
    mf.MAIN_ARM_GROUPS = tuple(orig_groups) + ("CX",)
    RN.default_arms = default_arms_causal
    RN.BenchmarkRunner._run_arm = wrapped
    RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier = _hook(orig_rel), _hook(orig_orig)
    FDM.differs_from = _counting_differs
    try:
        return R.run_unit(dataset, seed, out)
    finally:
        FDM.differs_from = _ORIG_DIFFERS
        R.benchmark_config, mf.MAIN_ARM_GROUPS, RN.default_arms = orig_cfg, orig_groups, orig_default
        RN.BenchmarkRunner._run_arm, RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier = orig_run_arm, orig_rel, orig_orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--small-reps", type=int, default=3)
    ap.add_argument("--digits-reps", type=int, default=2)
    ap.add_argument("--reuse-checkpoint-from", default=None)
    ap.add_argument("--include-native-r0", action="store_true", help="integridade: recalcula também as 4 células nativas r=0")
    a = ap.parse_args()
    if a.reuse_checkpoint_from:
        import shutil
        src = Path(a.reuse_checkpoint_from) / "checkpoints"
        dst = Path(a.out) / "checkpoints"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob(f"{a.unit[0]}_seed{a.unit[1]}.ckpt*"):
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
    run_unit_causal(a.unit[0], int(a.unit[1]), Path(a.out), a.small_reps, a.digits_reps, a.include_native_r0)


if __name__ == "__main__":
    sys.exit(main())
