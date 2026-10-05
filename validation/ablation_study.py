"""Benchmark de engenharia: ablação pareada sem OWL versus TBoxes curadas.

As TBoxes incluídas foram construídas a partir das descrições públicas das
features. Não são ontologias externas independentes e, por isso, não bastam
para uma alegação confirmatória de superioridade.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from core.mlp_factory import build_mlp_for_data, signal_capacity_audit
from validation.benchmark_ontologies import ensure_builtin_domain_ontologies
from core.c45_j48_tree import C45Classifier
from core.ontology_processor import OntologyProcessor
from core.ontology_quality import OntologyQualityGate
from core.ontology_reasoner import run_owl_reasoner
from core.controlled_trepan_experiment import (
    ControlledTrepanConfig,
    OriginalOracleProjection,
    evaluate_controlled_trepan_pair,
    fit_controlled_trepan_pair,
    oracle_health_gate,
)
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier


@dataclass(frozen=True)
class AblationConfig:
    repeats: int = 3
    test_size: float = 0.25
    max_samples: int = 700
    random_state: int = 42
    active_iterations: int = 2
    active_budget: int = 48
    ontology_dir: Optional[str] = None
    reasoner_engine: str = "hermit"
    ontology_feature_weight: float = 1.35
    semantic_gain_strength: float = 1.0
    oracle_dummy_margin: float = 0.02
    oracle_c45_margin: float = 0.10


def _base_mlp(X, y, seed: int):
    """Usa a fábrica única V9.2; sem hiperparâmetros por dataset."""
    return build_mlp_for_data(X, y, random_state=seed)


def _metrics(
    name, model, X_test, y_test, active_oracle, original_oracle, X_original_test,
    dataset, repeat, feature_space, ontology_path=None,
):
    pred = np.asarray(model.predict(X_test))
    oracle_pred = np.asarray(active_oracle.predict(X_test))
    original_pred = np.asarray(original_oracle.predict(X_original_test))
    return {
        "dataset": dataset, "repeat": repeat, "model": name,
        "feature_space": feature_space,
        "ablation": "with_owl" if ontology_path else "without_owl",
        "ontology_path": str(ontology_path or ""),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "active_oracle_fidelity": float(accuracy_score(oracle_pred, pred)),
        "controlled_original_mlp_fidelity": float(accuracy_score(original_pred, pred)),
        "nodes": int(getattr(getattr(model, "tree_", None), "node_count", 0) or 0),
        "depth": int(model.get_depth()) if hasattr(model, "get_depth") else None,
        "test_role": "locked_final_test",
    }


def _load_and_validate_ontology(ontology_path, feature_names, reasoner_engine):
    from owlready2 import get_ontology
    ontology = get_ontology(str(Path(ontology_path).resolve())).load()
    reasoner = run_owl_reasoner(
        ontology, engine=reasoner_engine, infer_property_values=True, debug=0
    )
    gate = OntologyQualityGate()
    report = gate.evaluate(
        feature_names, ontology, reasoner_report=reasoner, require_reasoner=True
    )
    if not report.accepted:
        raise ValueError(
            f"Ontologia de benchmark rejeitada ({ontology_path}): {report.issues}"
        )
    return ontology, gate, report


def run_dataset_ablation(
    X, y, *, feature_names: Sequence[str], ontology_path,
    dataset_name: str = "loaded_dataset", config: AblationConfig = AblationConfig(),
) -> list[dict]:
    """Executa uma ablação pareada em que a OWL é a única variável experimental.

    Os dois braços TREPAN consultam exactamente o mesmo MLP Original. O braço
    enriquecido usa :class:`OriginalOracleProjection` apenas para retirar as
    colunas OWL antes da consulta ao MLP; nenhum oráculo residual/ontológico é
    treinado nesta experiência. Seed, min_sample, max_queries, max_nodes,
    profundidade e parâmetros m-of-n são idênticos nos dois braços.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    feature_names = [str(name) for name in feature_names]
    if X.ndim != 2 or X.shape[1] != len(feature_names) or len(np.unique(y)) < 2:
        raise ValueError("Dataset/schema inválido para o estudo de ablação OWL.")
    ontology, gate, quality = _load_and_validate_ontology(
        ontology_path, feature_names, config.reasoner_engine
    )
    rows = []
    for repeat in range(config.repeats):
        seed = config.random_state + repeat * 101
        indices = np.arange(len(y))
        if len(indices) > config.max_samples:
            indices, _ = train_test_split(
                indices, train_size=config.max_samples, random_state=seed, stratify=y,
            )
        X_sub, y_sub = X[indices], y[indices]
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=config.test_size, random_state=seed, stratify=y_sub,
        )

        # O professor é treinado uma única vez e partilhado pelos dois braços.
        mlp = _base_mlp(X_train, y_train, seed).fit(X_train, y_train)
        oracle_health = oracle_health_gate(
            _base_mlp(X_train, y_train, seed),
            X_train, y_train,
            random_state=seed,
            cv_folds=3,
            dummy_margin=float(config.oracle_dummy_margin),
            c45_margin=float(config.oracle_c45_margin),
        )
        mlp.BIURI_ORACLE_HEALTH = oracle_health

        c45 = C45Classifier(
            confidence_factor=0.25, min_samples_leaf=2, random_state=seed,
        ).fit(X_train, y_train)

        # Ontologia é ajustada SOMENTE no treino.
        processor = OntologyProcessor(ontology, quality_gate=gate)
        accepted = [item for item in quality.matches if item.get("accepted")]
        train_df = pd.DataFrame(X_train, columns=feature_names)
        test_df = pd.DataFrame(X_test, columns=feature_names)
        processor.fit(train_df, accepted_matches=accepted, log=False)
        X_enr_train = processor.transform(train_df).to_numpy(dtype=float)
        X_enr_test = processor.transform(test_df).to_numpy(dtype=float)
        enriched_names = list(processor.output_features_)
        orig_idx = list(range(len(feature_names)))
        onto_idx = list(range(len(feature_names), len(enriched_names)))
        if not onto_idx:
            raise ValueError(f"TBox {ontology_path} não gerou features OWL não duplicadas.")

        def query_projector(rows):
            rows = np.asarray(rows, dtype=float)
            base = rows[:, orig_idx]
            frame = pd.DataFrame(base, columns=feature_names)
            projected = processor.transform(frame).to_numpy(dtype=float)
            if projected.shape[1] != len(enriched_names):
                raise ValueError("OntologyProcessor alterou o schema durante membership queries.")
            return projected

        semantic_weights = np.ones(len(enriched_names), dtype=float)
        semantic_weights[onto_idx] = float(config.ontology_feature_weight)

        # Orçamento completamente pareado: só a extensão OWL pode diferir.
        min_sample = max(len(X_train), min(1000, max(120, len(X_train) * 3)))
        query_budget = max(1600, min_sample * 2)
        paired_cfg = ControlledTrepanConfig(
            max_nodes=31,
            max_depth=8,
            min_samples_leaf=2,
            min_sample=min_sample,
            max_n=3,
            beam_width=2,
            max_features_per_node=12,
            max_queries=query_budget,
            random_state=seed,
            semantic_gain_strength=float(config.semantic_gain_strength),
        )
        pair = fit_controlled_trepan_pair(
            X_train,
            y_train,
            oracle=mlp,
            feature_names=feature_names,
            config=paired_cfg,
            reloaded_X_train=X_enr_train,
            reloaded_feature_names=enriched_names,
            original_feature_indices=orig_idx,
            semantic_feature_weights=semantic_weights,
            query_projector=query_projector,
            run_id=f"ablation_v9_2:{dataset_name}:{repeat}",
        )
        # A presença explícita deste tipo no fluxo é auditável e garante que o
        # braço com OWL continua a consultar o mesmo MLP Original.
        if not isinstance(pair.reloaded_oracle_for_audit, OriginalOracleProjection):
            raise RuntimeError("Braço OWL não está projectado para o mesmo MLP Original.")
        if not isinstance(pair.reloaded, TrepanReloadedClassifier):
            raise RuntimeError("Braço OWL não usa o núcleo histórico Reloaded.")

        paired_eval = evaluate_controlled_trepan_pair(
            pair, X_test, y_test, reloaded_X_test=X_enr_test
        )

        def row_from_metrics(name, metrics, *, feature_space, with_owl):
            return {
                "dataset": dataset_name,
                "repeat": repeat,
                "model": name,
                "feature_space": feature_space,
                "ablation": "with_owl" if with_owl else "without_owl",
                "ontology_path": str(ontology_path) if with_owl else "",
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "active_oracle_fidelity": float(metrics["oracle_fidelity"]),
                "controlled_original_mlp_fidelity": float(metrics["oracle_fidelity"]),
                "nodes": int(metrics["nodes"]),
                "depth": int(metrics["depth"]),
                "membership_queries": int(metrics["membership_queries"]),
                "test_role": "locked_final_test",
                "same_oracle": True,
                "same_seed": True,
                "same_tree_budget": True,
                "external_test_used_for_selection": False,
                "oracle_valid": bool(oracle_health.get("valid", False)),
                "exclude_from_owl_aggregate": not bool(oracle_health.get("valid", False)),
            }

        original_metrics = paired_eval["models"]["original"]
        reloaded_metrics = paired_eval["models"]["reloaded"]
        rows.append(row_from_metrics(
            "TREPAN Original", original_metrics, feature_space="original", with_owl=False
        ))
        # Contrato de espelhamento: sem OWL, Reloaded é literalmente o Original.
        rows.append(row_from_metrics(
            "TREPAN Reloaded — sem OWL", original_metrics,
            feature_space="original", with_owl=False
        ))

        c45_pred = np.asarray(c45.predict(X_test))
        mlp_pred = np.asarray(mlp.predict(X_test))
        rows.append({
            "dataset": dataset_name, "repeat": repeat, "model": "C4.5-Nativo",
            "feature_space": "original", "ablation": "without_owl",
            "ontology_path": "",
            "accuracy": float(accuracy_score(y_test, c45_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, c45_pred)),
            "macro_f1": float(f1_score(y_test, c45_pred, average="macro", zero_division=0)),
            "active_oracle_fidelity": float(accuracy_score(mlp_pred, c45_pred)),
            "controlled_original_mlp_fidelity": float(accuracy_score(mlp_pred, c45_pred)),
            "nodes": int(getattr(getattr(c45, "tree_", None), "node_count", 0) or 0),
            "depth": int(c45.get_depth()) if hasattr(c45, "get_depth") else None,
            "test_role": "locked_final_test",
            "oracle_valid": bool(oracle_health.get("valid", False)),
            "exclude_from_owl_aggregate": not bool(oracle_health.get("valid", False)),
        })

        row = row_from_metrics(
            "TREPAN Reloaded — com OWL", reloaded_metrics,
            feature_space="enriched", with_owl=True
        )
        row.update({
            "ontology_quality_status": quality.status,
            "ontology_feature_coverage": quality.metrics["feature_coverage"],
            "ontology_tbox_entities": quality.metrics["tbox_entity_count"],
            "ontology_abox_individuals": quality.abox["individual_count"],
            "reasoner": quality.reasoner.get("engine"),
            "reasoner_consistent": quality.reasoner.get("consistent"),
            "derived_owl_features": len(onto_idx),
            "oracle_health": oracle_health,
            "paired_protocol": paired_eval["experiment_audit"],
            "protocol_audit": paired_eval["protocol_audit"],
            "semantic_feature_weight": float(config.ontology_feature_weight),
            "semantic_gain_strength": float(config.semantic_gain_strength),
            "surrogate_core": "core.trepan_reloaded_historical.TrepanReloadedClassifier",
            "oracle_adapter": "OriginalOracleProjection",
        })
        rows.append(row)
    return rows


def builtin_benchmark_datasets() -> Dict[str, tuple]:
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()
    diabetes = load_diabetes()
    diabetes_target = (diabetes.target >= np.median(diabetes.target)).astype(int)
    digits = load_digits()
    return {
        "iris": (iris.data, iris.target, iris.feature_names),
        "wine": (wine.data, wine.target, wine.feature_names),
        "breast_cancer": (cancer.data, cancer.target, cancer.feature_names),
        "diabetes_progression": (
            diabetes.data, diabetes_target, diabetes.feature_names,
        ),
        "digits": (digits.data, digits.target, digits.feature_names),
    }


def summarize_ablation(rows: Sequence[dict]) -> dict:
    summary = {}
    for model in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        summary[model] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in selected])),
                "std": float(np.std([row[metric] for row in selected], ddof=1))
                if len(selected) > 1 else 0.0,
            }
            for metric in (
                "accuracy", "balanced_accuracy", "macro_f1",
                "active_oracle_fidelity", "controlled_original_mlp_fidelity",
            )
        }
    return summary


def paired_owl_analysis(rows: Sequence[dict], random_state: int = 42) -> dict:
    """Efeito OWL pareado usando *dataset* como unidade de análise.

    Primeiro calcula a diferença OWL−sem-OWL por repetição e depois agrega a
    média dentro de cada dataset. Bootstrap e Wilcoxon operam sobre essas médias
    por dataset, evitando tratar repetições do mesmo dataset como independentes.
    """
    from scipy.stats import wilcoxon

    without = {
        (row["dataset"], row["repeat"]): row
        for row in rows if row["model"] == "TREPAN Reloaded — sem OWL"
    }
    with_owl = {
        (row["dataset"], row["repeat"]): row
        for row in rows if row["model"] == "TREPAN Reloaded — com OWL"
    }
    all_keys = sorted(set(without) & set(with_owl))
    invalid_datasets = sorted({
        dataset for dataset, repeat in all_keys
        if without[(dataset, repeat)].get("oracle_valid", True) is False
        or with_owl[(dataset, repeat)].get("oracle_valid", True) is False
    })
    keys = [key for key in all_keys if key[0] not in set(invalid_datasets)]
    datasets = sorted({dataset for dataset, _ in keys})
    rng = np.random.default_rng(random_state)
    output = {
        "n_paired_runs": len(keys),
        "n_paired_runs_before_oracle_gate": len(all_keys),
        "n_datasets": len(datasets),
        "unit_of_analysis": "dataset",
        "excluded_datasets": invalid_datasets,
        "pairing": "dataset + repeat + locked split; repetitions aggregated within dataset",
    }
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        per_dataset = []
        per_dataset_detail = {}
        for dataset in datasets:
            ds_keys = [key for key in keys if key[0] == dataset]
            ds_diffs = np.asarray([
                with_owl[key][metric] - without[key][metric] for key in ds_keys
            ], dtype=float)
            mean_delta = float(np.mean(ds_diffs))
            per_dataset.append(mean_delta)
            per_dataset_detail[dataset] = {
                "mean_difference": mean_delta,
                "repeat_differences": ds_diffs.tolist(),
                "n_repeats": int(len(ds_diffs)),
            }
        differences = np.asarray(per_dataset, dtype=float)
        if len(differences):
            boot = np.asarray([
                np.mean(rng.choice(differences, size=len(differences), replace=True))
                for _ in range(5000)
            ])
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            try:
                p_value = float(wilcoxon(differences, alternative="two-sided").pvalue)
            except ValueError:
                p_value = 1.0
        else:
            ci_low = ci_high = p_value = float("nan")
        output[metric] = {
            "mean_difference": float(np.mean(differences)) if len(differences) else None,
            "median_difference": float(np.median(differences)) if len(differences) else None,
            "bootstrap_95_ci": [float(ci_low), float(ci_high)],
            "wilcoxon_p_value": p_value,
            "wins": int(np.sum(differences > 0)),
            "ties": int(np.sum(differences == 0)),
            "losses": int(np.sum(differences < 0)),
            "differences": differences.tolist(),
            "per_dataset": per_dataset_detail,
        }
    return output


def run_builtin_ablation(config: AblationConfig = AblationConfig()) -> dict:
    ontology_root = Path(config.ontology_dir) if config.ontology_dir else (
        Path(__file__).resolve().parent.parent / "data" / "benchmark_ontologies"
    )
    ontology_paths = ensure_builtin_domain_ontologies(ontology_root)
    rows = []
    for name, (X, y, feature_names) in builtin_benchmark_datasets().items():
        rows.extend(run_dataset_ablation(
            X, y, feature_names=feature_names, ontology_path=ontology_paths[name],
            dataset_name=name, config=config,
        ))
    return {
        "protocol": {
            "outer_test": "stratified locked holdout",
            "ontology_fit": "training_only fit/transform",
            "ablation": (
                "same split, same MLP original, same seed and same TREPAN budget; "
                "only semantic/OWL extension differs"
            ),
            "ontologies": (
                "curated benchmark RDF/XML TBox; zero dataset individuals; "
                "not an externally governed independent ontology"
            ),
            "reasoner": config.reasoner_engine,
            "repeats": config.repeats,
            "oracle_gate": (
                "training-CV only: MLP must beat Dummy and stay within configured "
                "balanced-accuracy margin of C4.5-Nativo"
            ),
            "paired_statistics_unit": "dataset (repetitions aggregated within dataset)",
            "warning": (
                "Benchmark de engenharia; não constitui validação clínica externa "
                "nem comparação confirmatória com ontologias independentes."
            ),
        },
        "config": asdict(config),
        "ontology_paths": ontology_paths,
        "summary": summarize_ablation(rows),
        "paired_owl_analysis": paired_owl_analysis(rows, config.random_state),
        "rows": rows,
    }


def save_ablation_results(result: dict, output_dir) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "ablation_results.json"
    csv_path = output / "ablation_rows.csv"
    md_path = output / "ablation_report.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({key for row in result["rows"] for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])
    lines = [
        "# Estudo de ablação OWL — TREPAN Reloaded", "",
        "Teste bloqueado; ontologias TBox RDF/XML; reasoner OWL DL obrigatório.", "",
        "| Modelo | Accuracy | Balanced accuracy | Macro-F1 | Fidelidade ativa | Fidelidade MLP original |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, metrics in result["summary"].items():
        lines.append(
            f"| {model} | {metrics['accuracy']['mean']:.3f} | "
            f"{metrics['balanced_accuracy']['mean']:.3f} | "
            f"{metrics['macro_f1']['mean']:.3f} | "
            f"{metrics['active_oracle_fidelity']['mean']:.3f} | "
            f"{metrics['controlled_original_mlp_fidelity']['mean']:.3f} |"
        )
    paired = result.get("paired_owl_analysis", {})
    lines.extend(["", "## Efeito pareado da OWL", ""])
    lines.append("| Métrica | Diferença média | IC bootstrap 95% | Vitórias/Empates/Derrotas | Wilcoxon p |")
    lines.append("|---|---:|---:|---:|---:|")
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        value = paired.get(metric, {})
        ci = value.get("bootstrap_95_ci", [float("nan"), float("nan")])
        lines.append(
            f"| {metric} | {value.get('mean_difference', float('nan')):.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{value.get('wins', 0)}/{value.get('ties', 0)}/{value.get('losses', 0)} | "
            f"{value.get('wilcoxon_p_value', float('nan')):.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}
