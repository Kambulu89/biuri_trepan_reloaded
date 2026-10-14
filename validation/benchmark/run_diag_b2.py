"""DIAGNÓSTICO DEV para o desenho de B2 (NÃO é B2; não altera o núcleo): decomposição por ablação da perda residual do Reloaded+OWL face ao Original e
captura das auditorias de aceitação por nó (ganho local, ganho em linhas reais, razões de rejeição) e dos tempos por componente (custo).

Braços A–F (integridade) + DIAG:
  diag_original_enr          Original na representação OWL (isola o efeito da REPRESENTAÇÃO)
  reloaded_owl_features      Reloaded λ=0 na representação OWL (projetor + representação; sem score/gate semântico)
  diag_gate_off_active_off   E sem refinamento focado no erro (gate) e sem consultas ativas (restam: pesos na prioridade de features, score, candidatos guiados)
  diag_gate_off              E sem gate
  diag_active_off            E sem consultas ativas
  reloaded_semantic_score    score semântico nas features ORIGINAIS (sem representação OWL)
  diag_E_audit               E (≡ reloaded_owl_full) só para capturar auditorias
    python -m validation.benchmark.run_diag_b2 --unit iris 42 --out DIR [--reuse-checkpoint-from REF]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path

from validation.benchmark import run as R      # PRIMEIRO: fixa o ambiente numérico (OPENBLAS_*) antes de importar numpy
from core.benchmark import runner as RN
from validation.benchmark import manifest as mf

_ORIG_DEFAULT_ARMS = RN.default_arms      # a função original (diag_arms substitui RN.default_arms durante a execução)

CAPTURE = {"arm": None, "unit": None, "out": None}
AUDIT_KEYS = ("node_id", "test", "data_only_test", "semantic_test", "selected_test_after_error_gate", "proposed_decision_changed", "decision_changed",
              "error_focused_intervention_attempted", "error_focused_intervention_accepted", "error_focused_intervention_reason",
              "data_only_local_fidelity", "semantic_local_fidelity", "local_fidelity_gain", "data_only_real_fidelity", "semantic_real_fidelity",
              "real_fidelity_gain", "real_training_gate_evaluated", "semantic_coherence_gain", "error_focused_accepted_within_tolerance",
              "decision_sample_size", "left_count", "right_count", "error_region_disagreement_rate", "information_gain", "data_only_information_gain",
              "semantic_information_gain", "semantic_candidates_evaluated", "m", "n")


def _dump(tree, kind: str) -> None:
    if CAPTURE["out"] is None or CAPTURE["arm"] is None:
        return
    d = Path(CAPTURE["out"]) / "audits"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{k: r.get(k) for k in AUDIT_KEYS} for r in (getattr(tree, "split_audit_", None) or [])]
    timing = {k: float(getattr(tree, k, 0.0) or 0.0) for k in ("query_time_", "split_search_time_", "m_of_n_search_time_", "synthetic_generation_time_",
                                                                 "local_model_time_", "pruning_time_", "training_time_", "oracle_prediction_time_")}
    rec = {"kind": kind, "arm": CAPTURE["arm"], "unit": CAPTURE["unit"], "rows": rows, "timing": timing, "nodes_created": getattr(tree, "nodes_created_", None),
           "nodes_expanded": getattr(tree, "nodes_expanded_", None), "membership_queries": int(getattr(tree, "membership_queries_", 0)),
           "query_budget_exhausted": bool(getattr(tree, "query_budget_exhausted_", False)), "max_nodes_reached": bool(getattr(tree, "max_nodes_reached_", False)),
           "candidate_totals": getattr(tree, "candidate_totals_", None), "oracle_accounting": getattr(tree, "oracle_accounting_", None),
           "summary": tree.semantic_audit_summary() if hasattr(tree, "semantic_audit_summary") else None}
    (d / f"{CAPTURE['unit']}__{CAPTURE['arm']}.json").write_text(json.dumps(rec, default=str), encoding="utf-8")


class _RelCapture(RN.TrepanReloadedClassifier):
    def fit(self, *a, **k):
        r = super().fit(*a, **k)
        _dump(self, "reloaded")
        return r


class _OrigCapture(RN.TrepanOriginalClassifier):
    def fit(self, *a, **k):
        r = super().fit(*a, **k)
        _dump(self, "original")
        return r


def diag_arms(cfg):
    base = [a for a in _ORIG_DEFAULT_ARMS(cfg) if a.group in mf.MAIN_ARM_GROUPS]
    allarms = {a.arm_id: a for a in _ORIG_DEFAULT_ARMS(cfg)}
    E = allarms["reloaded_owl_full"]
    mk = lambda aid, desc, **kw: dataclasses.replace(E, arm_id=aid, description=desc, group="DIAG", role="diagnostic", **kw)
    extra = [
        RN.ArmSpec("diag_original_enr", "trepan_original", "DIAG: Original na representação OWL", tree_space="enr", oracle="mlp_original", needs_semantic=True,
                   role="diagnostic", group="DIAG"),
        dataclasses.replace(allarms["reloaded_owl_features"], group="DIAG", role="diagnostic"),
        mk("diag_gate_off_active_off", "DIAG: E sem gate EFSR e sem consultas ativas", error_focus=False, active_queries=False),
        mk("diag_gate_off", "DIAG: E sem gate EFSR", error_focus=False),
        mk("diag_active_off", "DIAG: E sem consultas ativas", active_queries=False),
        dataclasses.replace(allarms["reloaded_semantic_score"], group="DIAG", role="diagnostic"),
        mk("diag_E_audit", "DIAG: E (captura de auditorias)"),
    ]
    return base + extra


def run_unit_diag(dataset: str, seed: int, out: Path, reuse_checkpoint_from: Path | None = None) -> Path:
    if reuse_checkpoint_from is not None:
        src = Path(reuse_checkpoint_from) / "checkpoints" / f"{dataset}_seed{seed}.ckpt"
        dst = out / "checkpoints" / f"{dataset}_seed{seed}.ckpt"
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    CAPTURE.update(unit=f"{dataset}_seed{seed}", out=str(out))
    orig_default, orig_groups = RN.default_arms, mf.MAIN_ARM_GROUPS
    orig_run_arm, orig_rel, orig_orig = RN.BenchmarkRunner._run_arm, RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier

    def wrapped_run_arm(self, spec, *a, **k):
        CAPTURE["arm"] = spec.arm_id
        return orig_run_arm(self, spec, *a, **k)

    RN.default_arms = diag_arms
    mf.MAIN_ARM_GROUPS = tuple(orig_groups) + ("DIAG",)
    RN.BenchmarkRunner._run_arm = wrapped_run_arm
    RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier = _RelCapture, _OrigCapture
    try:
        return R.run_unit(dataset, seed, out)
    finally:
        RN.default_arms, mf.MAIN_ARM_GROUPS = orig_default, orig_groups
        RN.BenchmarkRunner._run_arm, RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier = orig_run_arm, orig_rel, orig_orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse-checkpoint-from", default=None)
    a = ap.parse_args()
    run_unit_diag(a.unit[0], int(a.unit[1]), Path(a.out), Path(a.reuse_checkpoint_from) if a.reuse_checkpoint_from else None)


if __name__ == "__main__":
    sys.exit(main())
