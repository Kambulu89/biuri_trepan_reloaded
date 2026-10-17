"""Acrescenta ao manifest v3 em RASCUNHO (``frozen=false``) a secção dos controlos ontológicos M29 (Doc. 29): variantes, sementes/réplicas, versões das ontologias, identidades dos
oráculos, fingerprints dos checkpoints, hashes de artefactos, critérios (ganho direto, atribuição, superioridade global, evidência) e métricas obrigatórias. Idempotente.
    python -m validation.benchmark.manifest_v3_m29 [--checkpoints DIR_COM_SIDECARS]"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/b1b4/manifest_v3/BENCHMARK_MANIFEST_v3.DRAFT.json"
ONT = ROOT / "validation/benchmark/ontologies"
RAW = ROOT / "docs/evidence/b1b4/sampler_c/dev_numeric/raw"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(checkpoints: Path | None) -> dict:
    oracles = {}
    for f in sorted(RAW.glob("*/seed*/trepan_original.json")):
        r = json.loads(f.read_text())
        oracles[f"{r['dataset']}:{r['master_seed']}"] = {"oracle_id": r["oracle_id"], "oracle_version": r["row"].get("oracle_version"), "split_hash": r["split_hash"],
                                                          "base_representation_hash": r["row"].get("base_representation_hash"), "dataset_hash": r["row"].get("dataset_hash")}
    fps = {}
    if checkpoints and checkpoints.exists():
        for f in sorted(checkpoints.glob("*.fingerprint.json")):
            d = json.loads(f.read_text())
            fps[f.name.replace(".ckpt.fingerprint.json", "")] = {"fingerprint": d["fingerprint"], "version": d.get("fingerprint_version"), "sidecar_sha256": sha(f)}
    return {
        "status": "PREPARED_NOT_EXECUTED (orçamento por aprovar)", "doc": "docs/b1b4/29_PROTOCOLO_CONTROLOS_ONTOLOGICOS.md",
        "arms": {"A29": "Original + amostrador C (= sc_B do Doc. 26)", "B29": "Reloaded OWL real + amostrador C (= sc_D)", "C29": "Reloaded + ontologia PERMUTADA acoplada + C (m29_P_k)",
                 "D29": "Reloaded + GRUPOS ALEATÓRIOS desacoplados + C (m29_D_k)", "E29": "Reloaded + engenharia convencional de features + C (m29_E_k)",
                 "F29": "Reloaded sem ontologia + C (= sc_F; controlo de equivalência com A29)", "C45": "C4.5 canónico (inalterado)"},
        "seeds": {"master_seeds": [42, 7, 123, 2024, 11], "digits_master_seeds": [42, 7, 123], "sampler_replicas": [0, 1, 2],
                  "permutation_P": "20_260_000 + 1000*k + master_seed (+997*tentativa)", "random_groups_D": {"A": "30_260_000 + 1000*k + master_seed", "B": "40_260_000 + 1000*k + master_seed"},
                  "pairing": "draw k do controlo emparelhado com a réplica de amostragem k do braço real", "algorithm_seed": "semente mestra (fixa)"},
        "control_validity_gate": {"dimension": "n derivadas = OWL real", "finite": True, "scale": "std máx ≤ 100× a da OWL real", "destruction": "retenção de pares co-agrupados ≤ acaso",
                                   "max_attempts": 4, "fallback": "estrutura permutada (features reais) rotulada structure_only_fallback; NÃO satisfaz o controlo exigido"},
        "ontology_versions_sha256": {p.stem: sha(p) for p in sorted(ONT.glob("*.owl"))},
        "oracle_identities": oracles, "tuning_checkpoint_fingerprints": fps,
        "artifact_hashes": {f: sha(ROOT / f) for f in ("core/query_sampling.py", "core/hierarchy_contract.py", "core/benchmark/semantic.py", "validation/benchmark/run_controls_m29.py",
                                                         "validation/benchmark/m29_report.py", "validation/benchmark/hierarchy_joint_report.py")},
        "metrics_required": ["accuracy", "fidelity", "precision_macro", "macro_f1", "node_count", "leaf_count", "tree_depth", "m_of_n_split_count", "semantic_split_count", "oracle_queries",
                             "oracle_samples", "training_time", "semantic_gain_accuracy", "semantic_gain_fidelity", "ontology_attribution_status", "direct_gain_status",
                             "global_hierarchy_status", "evidence_level"],
        "metric_columns": {"accuracy": "accuracy_real_labels", "fidelity": "fidelity_to_oracle", "precision_macro": "precision_macro_real_labels", "macro_f1": "macro_f1_real_labels",
                           "tree_depth": "depth", "m_of_n_split_count": "m_of_n_count", "oracle_samples": "total_oracle_samples", "training_time": "tree_training_time"},
        "criteria": {"direct_gain": "evaluate_direct_gain: DeltaAccuracy>0 e DeltaFidelity>0 vs Original equivalente (A29), estritos; não-inferioridade (>=) só elegibilidade",
                     "ontology_attribution": "attribute_to_ontology: ganho conjunto estrito sobre C29, D29 e E29 (completos); controlo em falta/parcial ⇒ NOT_ASSESSABLE",
                     "global_superiority": "evaluate_global: U1-U5 estritas; Original<=C4.5 ⇒ HIERARCHY_INFEASIBLE (unidade mantida)",
                     "evidence": "classify_evidence: unidade = dataset; SUPPORTED só confirmatório com >=12 datasets independentes; DEV nunca confirmatório"},
        "comparability": "core.hierarchy_contract.check_comparability (nível estrito: instâncias+ordem, rótulos, classes, partição, dataset, representação base, oráculo e previsões, versão do modelo, configuração, orçamento)",
        "confirmatory_data_used": False}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints")
    a = ap.parse_args()
    m = json.loads(MANIFEST.read_text())
    m["controls_m29"] = build(Path(a.checkpoints) if a.checkpoints else None)
    assert m["frozen"] is False
    MANIFEST.write_text(json.dumps(m, indent=1, ensure_ascii=False))
    print("ok", len(m["controls_m29"]["oracle_identities"]), "oráculos;", len(m["controls_m29"]["tuning_checkpoint_fingerprints"]), "fingerprints")
