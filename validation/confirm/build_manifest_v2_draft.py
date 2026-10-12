"""Gera o RASCUNHO do manifest v2 (``docs/b1b4/manifest_v2/BENCHMARK_MANIFEST_v2.DRAFT.json``). NÃO é o manifest congelado.

Não escreve em ``validation/benchmark/`` (onde ``latest_manifest_path`` o trataria como congelado). Regista: hashes do código atual e deriva face ao v1, contrato de equivalência
(allowlist vazia), regras de seeds, datasets CONFIRM (por verificar), grelhas/margens, braços, plano estatístico, hashes dos documentos de protocolo e identificadores experimentais.
A geração é determinística (sem relógio no conteúdo com hash).  ``python -m validation.confirm.build_manifest_v2_draft``
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from validation.benchmark.manifest import FROZEN_CODE_FILES, manifest_digest
from validation.confirm.inspect_candidates import CANDIDATES, EC3

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "b1b4" / "manifest_v2" / "BENCHMARK_MANIFEST_v2.DRAFT.json"
PROTOCOL_DOCS = ("02_PROTOCOLO_EXPERIMENTAL.md", "04_MATRIZ_ABLACAO.md", "05_CONTRATO_SUPERIORIDADE.md", "07_CONTRATO_EQUIVALENCIA.md",
                 "08_ELEGIBILIDADE_DATASETS.md", "09_AUTORIA_ONTOLOGIAS_E_CONGELAMENTO_OWL.md", "10_GRELHAS_MARGENS_CUSTO.md", "11_PLANO_V1_FASE1A.md")
CONFIRM_SEED_RULE = "seed_i = int(sha256('BIURI-CONFIRM-v2/{i}'), 16) mod (2**31 - 1), i = 1..10"
DEV_SEEDS = [42, 7, 123, 2024, 11]


def confirm_seeds() -> list:
    return [int(hashlib.sha256(f"BIURI-CONFIRM-v2/{i}".encode()).hexdigest(), 16) % (2**31 - 1) for i in range(1, 11)]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> Dict[str, Any]:
    v1 = json.loads((ROOT / "validation/benchmark/BENCHMARK_MANIFEST_v1.json").read_text(encoding="utf-8"))
    cur = {f: _sha(ROOT / f) for f in FROZEN_CODE_FILES}
    drift = sorted(f for f, h in cur.items() if v1["frozen_code_sha256"].get(f) != h)
    m: Dict[str, Any] = {
        "manifest_version": "2-DRAFT", "frozen": False, "supersedes": "BENCHMARK_MANIFEST_v1.json",
        "status": "DRAFT_NOT_FROZEN: datasets por verificar, ontologias por escrever, hashes de código a re-congelar na entrada da Fase 1a",
        "experiment_family_id": "BIURI-B1B4-v2",
        "stage_ids": {"0": "BIURI-B1B4-v2/S0-dev-conformance", "1": "BIURI-B1B4-v2/S1-confirm-primary", "2": "BIURI-B1B4-v2/S2-confirm-attribution",
                      "3": "BIURI-B1B4-v2/S3-opaque-and-track-M", "4": "BIURI-B1B4-v2/S4-factorial"},
        "unit_id_scheme": "{experiment_family_id}/{stage}/{dataset_id}/{seed}/{arm_id}",
        "architecture": {"statement": "Reloaded = Original + extensão ontológica", "equivalence_contract": "E1-E9 (docs/b1b4/07_CONTRATO_EQUIVALENCIA.md)",
                         "equivalence_allowlist": [], "known_nonconformities": [{"id": "V1", "status": "open", "fix_phase": "1a"}, {"id": "V2", "status": "to_assess"}]},
        "hierarchy": {"accuracy": "Accuracy_R_OWL > Accuracy_Original > Accuracy_C45", "fidelity": "Fidelity_R_OWL > Fidelity_Original",
                      "delta_accuracy": 0.0, "delta_fidelity": 0.0, "nu_fidelity_selection": 0.0, "comparison": "strict, unrounded",
                      "statuses": ["HIERARCHY_SATISFIED", "HIERARCHY_NOT_SATISFIED", "HIERARCHY_INFEASIBLE", "INSUFFICIENT_EVIDENCE"],
                      "evidence_levels": ["observed_only", "statistically_sustained"], "fidelity_loss_tolerance": "exploratory sensitivity only; never used to claim fidelity superiority"},
        "code_freeze": {"files": cur, "v1_hashes": v1["frozen_code_sha256"], "drift_vs_v1": drift,
                        "drift_policy": "Original/tuning mudaram desde o v1 (otimizações exatas de engenharia); re-congelar só com relatório de equivalência v1->atual versionado (ver Doc. 12 D-13)",
                        "equivalence_report_v1_to_current": "PENDING (docs/b1b4/manifest_v2/equivalence_v1_to_current.json)"},
        "splits": {"scheme": "holdout", "test_size": 0.25, "stratified": True, "dev_seeds": DEV_SEEDS, "confirm_seeds_rule": CONFIRM_SEED_RULE, "confirm_seeds": confirm_seeds()},
        "eligibility_criteria": {"EC3": EC3, "documented_in": "docs/b1b4/08_ELEGIBILIDADE_DATASETS.md", "sha256": _sha(ROOT / "docs/b1b4/08_ELEGIBILIDADE_DATASETS.md")},
        "datasets_confirm_primary": [k for k, v in sorted(CANDIDATES.items(), key=lambda kv: kv[1]["order"]) if v["primary"]],
        "datasets_confirm_status": {"statlog_german_credit": "INELIGIBLE (EC1, exposto)", "occupancy_detection": "INELIGIBLE (EC4, série temporal)",
                                    "others": "PENDING verification (UCI inacessível no ambiente de desenvolvimento)"},
        "datasets_confirm_reserves_ordered": [k for k, v in sorted(CANDIDATES.items(), key=lambda kv: kv[1]["order"]) if not v["primary"]],
        "datasets_dev": ["iris", "wine", "breast_cancer", "digits", "synthetic_binary", "synthetic_3class"],
        "dataset_pins": {k: {"uci_id_expected": v["uci_id"], "data_sha256": "PENDING", "license": "PENDING", "ontology_freeze": "PENDING"} for k, v in CANDIDATES.items() if v["primary"]},
        "ontology_protocol": {"doc": "docs/b1b4/09_AUTORIA_ONTOLOGIAS_E_CONGELAMENTO_OWL.md", "sha256": _sha(ROOT / "docs/b1b4/09_AUTORIA_ONTOLOGIAS_E_CONGELAMENTO_OWL.md"),
                              "freeze_tag": "ontologies-confirm-v1", "hash": "sha256 of OWL bytes", "mutation_after_freeze": "forbidden"},
        "arms": {"A": "c45_canonical", "B": "trepan_original", "C": "reloaded_no_ontology_equivalence_control", "D": "reloaded_owl_real_variants_R0..R7",
                 "E": "reloaded_owl_permuted_or_random_controls", "F": "conventional_feature_engineering_controls",
                 "extra": ["original_on_owl_representation", "mlp_ontological_track_M"]},
        "grids": {"lambda_b1b": [0.0, 0.05, 0.10, 0.20, 0.40, 0.80], "lambda_b1b_definition": "score_i = s_i + min(m_i, lambda*s_i); max statistical shortfall = lambda/(1+lambda)",
                  "legacy_semantic_lambda": [0.0, 0.5, 1.0, 2.5], "inner_folds_K": 3, "b1_variants": ["A_tie_break", "B_bounded", "C_normalised"],
                  "grid_status": "PROPOSED; review in docs/b1b4/10_GRELHAS_MARGENS_CUSTO.md"},
        "statistics": {"alpha": 0.05, "within_dataset": "Nadeau-Bengio corrected resampled t + paired bootstrap", "across_datasets": "one-sided Wilcoxon signed-rank + bootstrap CI over datasets",
                       "multiplicity": "Holm over {H1,H2,H3}", "equivalence": "TOST margin 0.01 (non-inferiority/no-effect only)", "H0": "exact equivalence R0=O"},
        "protocol_documents": {d: _sha(ROOT / "docs/b1b4" / d) for d in PROTOCOL_DOCS if (ROOT / "docs/b1b4" / d).exists()},
        "freeze_preconditions": ["datasets verified by inspection (EC1-EC8)", "ontologies authored, validated and hashed", "V1 fixed + EQ tests green + golden anchor equal",
                                 "baseline hash drift resolved (D-13)", "user approval of grids and margins (D-8)"],
    }
    m["manifest_sha256"] = manifest_digest(m)
    return m


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m = build()
    OUT.write_text(json.dumps(m, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT, m["manifest_sha256"], "drift:", m["code_freeze"]["drift_vs_v1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
