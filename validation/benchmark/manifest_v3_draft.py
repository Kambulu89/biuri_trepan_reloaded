"""Gera o RASCUNHO do manifest v3 (B1+B2) — ``frozen = false``. Não substitui nem altera o manifest congelado (v1/v2); não é usado por nenhuma execução.

    python -m validation.benchmark.manifest_v3_draft --out docs/b1b4/manifest_v3/BENCHMARK_MANIFEST_v3.DRAFT.json
Os hashes de código/ontologias são os do working tree no momento da geração; o congelamento (``frozen=true``) exige aprovação explícita e nova geração.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = ("core/trepan_original.py", "core/trepan_reloaded_historical.py", "core/trepan_scientific_tuning.py", "core/scientific_experiment_contract.py",
              "core/c45_j48_tree.py", "core/benchmark/runner.py", "core/benchmark/semantic.py", "core/benchmark/ontology_gate.py", "core/benchmark/structure.py",
              "core/benchmark/metrics.py", "core/benchmark/analysis.py", "core/benchmark/preprocessing.py", "core/benchmark/timing.py",
              "core/tuning_checkpoint.py", "core/error_focused_semantic_refinement.py", "core/probabilistic_distillation.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def build() -> dict:
    v1 = json.loads((ROOT / "validation/benchmark/BENCHMARK_MANIFEST_v1.json").read_text())
    import numpy, scipy, sklearn, pandas
    onto = {p.name: sha(p) for p in sorted((ROOT / "validation/benchmark/ontologies").glob("*.owl"))}
    m = {
        "manifest_version": "3-DRAFT", "frozen": False,
        "status": "DRAFT_NOT_FROZEN: B2-C implementado e em piloto DEV; B2-A/B2-B não implementados; confirmatório BLOQUEADO até congelar datasets/ontologias/oráculos/configurações/métricas",
        "supersedes": "nenhum (v1 e v2-DRAFT permanecem intactos)",
        "code": {"commit_at_generation": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                 "dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "sha256": {f: sha(ROOT / f) for f in CODE_FILES},
                 "frozen_baselines": {"trepan_original": sha(ROOT / "core/trepan_original.py"), "c45": sha(ROOT / "core/c45_j48_tree.py"),
                                      "v1_frozen_code_sha256": v1["frozen_code_sha256"]},
                 "pre_b2c_reference_commit": "4143db6b082b09f02066c5118a121d972c6c49fd"},
        "baselines": {"trepan_original": "canónico; limitação de desempate m-of-n registada (Doc. 19)", "c45": "C4.5 canónico (rótulos reais)",
                      "mlp": "FrozenOracle (oracle_id = hash dos pesos)", "equivalence_contract": "EQ v2.0 S0–S3 (docs/b1b4/07_CONTRATO_EQUIVALENCIA.md)"},
        "numeric_environment": {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                                "NPY_DISABLE_CPU_FEATURES": "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_KNL AVX512_KNM",
                                "python": sys.version.split()[0], "platform": platform.platform(),
                                "libs": {"numpy": numpy.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pandas": pandas.__version__}},
        "seeds": {"dev_master_seeds": v1["master_seeds"], "dev_digits_seeds_in_pilot": [42, 7, 123], "confirmatory_seeds": "RESERVADAS; não definidas neste rascunho"},
        "datasets": {"dev": ["iris", "wine", "breast_cancer", "digits"], "controlled": v1["controlled_validation_datasets"],
                     "confirmatory": "BLOQUEADO (ver BENCHMARK_MANIFEST_v2.DRAFT.json: candidatos por verificar)"},
        "oracles": {"policy": "FrozenOracle partilhado por todos os braços da unidade; oracle_id verificado; OriginalOracleProjection nas representações enriquecidas"},
        "ontologies": {"dev_owl_sha256": onto, "control": "ontologia permutada (ShuffledSemanticProvider, seed_offset 7919)"},
        "partitioning": v1["evaluation"],
        "b1_variants": {"default": "OFF", "arms": ["historical", "tiebreak", "calibrated lambda=0.2", "calibrated lambda=0.4"], "exploratory_separate": "bounded lambda=0.8",
                        "doc": "docs/b1b4/17_B1_PREREGISTO_E_RESULTADOS.md"},
        "b2_variants": {"B2-C": {"status": "implemented_experimental_default_off", "flags": ["statistical_fallback", "b2c_semantic_node_priority"],
                                 "doc": "docs/b1b4/20_B2_CONTRATO_E_PILOTO.md"},
                        "B2-A": {"status": "NOT_IMPLEMENTED_requires_authorization"}, "B2-B": {"status": "NOT_IMPLEMENTED_requires_authorization"},
                        "cv_lite": {"status": "NOT_IMPLEMENTED_experimental_factor_only"}},
        "rng_policy": {"rng_original": "estado interno (np.random.Generator) dos FeatureDistributionModel (global: random_state; local: random_state+node_id); intocado por B2-C",
                       "rng_semantic": "np.random.default_rng(SeedSequence([random_state, 0xB2C, node_id])) sobre cópia superficial do modelo local; só consultas semânticas",
                       "dataset_identity_dependence": False},
        "pilot_matrix_B2C": ["E histórico", "B1 tiebreak", "B1 calibrated lambda=0.2", "B1 calibrated lambda=0.4"],
        "pilot_arms": ["trepan_original_enr (Original equivalente)", "b2c_* e b2c_*_shuf (controlo permutado)", "b1_* sem B2-C"],
        "metrics": ["fidelity_to_oracle", "accuracy_real_labels", "node_count", "depth", "oracle_queries", "total_oracle_samples", "b2c_* (aceites/fallback/contexto)", "tempos por componente"],
        "acceptance_criteria": {"hierarchy": "Accuracy_R_OWL > Accuracy_Original > Accuracy_C45; Fidelity_R_OWL > Fidelity_Original (objetivo pré-registado, não imposto)",
                                "labels": ["HIERARCHY_SATISFIED", "NOT_SATISFIED", "INFEASIBLE", "INSUFFICIENT_EVIDENCE"], "selection_uses_test": False,
                                "b2c_technical_completion": "ver docs/b1b4/20_B2_CONTRATO_E_PILOTO.md §10"},
        "computational_budget": {"pilot_b2c_dev": "18 unidades, ≈13 árvores semânticas por unidade; ver Doc. 20 §8", "executor_task_limit_min": 120,
                                 "structural_tuning": "recalculado por unidade (o sal do checkpoint inclui o código do Reloaded)"},
        "ontological_controls": ["ontologia permutada", "Original equivalente na mesma representação", "controlo de engenharia convencional de features (planeado)"],
        "freeze_preconditions": ["aprovação explícita do utilizador", "datasets/ontologias/oráculos/configurações/métricas confirmatórios fixados", "CI verde no commit", "equivalência vs v1 documentada"],
    }
    m["manifest_sha256"] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).write_text(json.dumps(build(), indent=1, sort_keys=True), encoding="utf-8")
