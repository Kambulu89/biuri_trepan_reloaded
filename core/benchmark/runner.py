"""BenchmarkRunner genérico: o MESMO protocolo para qualquer dataset (com ou sem ontologia)."""
from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from core.benchmark import manifest as mf
from core.benchmark.metrics import OracleInfo, evaluate_model, minority_class
from core.benchmark.semantic import (
    NoSemanticProvider, OwlSemanticProvider, SemanticContext, ShuffledSemanticProvider,
)
from core.benchmark.splits import SplitSpec, assert_disjoint, inner_folds, make_splits
from core.c45_j48_tree import C45Classifier
from core.controlled_trepan_experiment import OriginalOracleProjection
from core.evaluation_protocol import EvaluationProtocolGuard, PartitionRole
from core.mlp_factory import build_mlp_for_data
from core.scientific_experiment_contract import OracleContractViolation, freeze_oracle
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier
from core.tree_build_report import c45_complexity, trepan_complexity

ORACLE_ORIGINAL = "MLP Original"
ORACLE_ONTOLOGICAL = "MLP Ontológico"


@dataclass(frozen=True)
class TreeBudget:
    """Orçamento IDÊNTICO para todos os braços TREPAN (fairness). ``None`` = regra automática declarada."""
    min_sample: Optional[int] = None          # auto: max(n_train, min(1000, max(120, 3*n_train)))
    max_queries: Optional[int] = None         # auto: max_nodes * min_sample (cada nó pode receber a sua amostra)
    max_nodes: int = 31
    max_depth: int = 8
    max_n: int = 3
    beam_width: int = 2
    min_samples_leaf: int = 2
    max_features_per_node: int = 12


@dataclass(frozen=True)
class BenchmarkConfig:
    seeds: Sequence[int] = (11, 22, 33, 44, 55, 66, 77, 88, 99, 111)   # fixadas ANTES dos resultados
    scheme: str = "holdout"                    # holdout | repeated_cv
    test_size: float = 0.25
    n_splits: int = 5
    n_repeats: int = 2
    base_seed: int = 42
    tree: TreeBudget = TreeBudget()
    mlp_trials: int = 0                        # MESMO nº de trials para MLP Original e Ontológico
    oracle_gate_margin: float = 0.01           # MLP Ontológico só é oráculo se ganhar > margin no CV INTERNO
    inner_cv_splits: int = 3
    alpha: float = 0.35                        # peso OntoDepth do Gain_Reloaded
    beta: float = 0.20                         # peso ErrorCoverage
    semantic_lambdas: Sequence[float] = (1.0,) # λ multiplica (alpha, beta); sensibilidade: p.ex. (0.5, 1.0, 2.5)
    extra_ablations: bool = True
    save_predictions: bool = True
    n_boot: int = 10000
    ci_alpha: float = 0.05
    reasoner_engine: str = "hermit"
    onto_weight: float = 1.35

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        d["semantic_lambdas"] = list(self.semantic_lambdas)
        return d


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    class_labels: Optional[List[Any]] = None   # ordem das classes (ex.: ordem do ARFF); preservada nas métricas

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y)
        self.feature_names = [str(f) for f in self.feature_names]
        if self.X.ndim != 2 or self.X.shape[1] != len(self.feature_names) or len(self.y) != len(self.X):
            raise ValueError("Dataset inconsistente (X, y, feature_names).")
        if not np.isfinite(self.X).all():
            raise ValueError("X contém NaN/Inf: faça a imputação antes do benchmark.")
        if self.class_labels is None:
            self.class_labels = list(np.unique(self.y))

    @classmethod
    def from_frame(cls, name: str, frame: pd.DataFrame, target: Optional[str] = None) -> "Dataset":
        target = target or frame.columns[-1]
        feats = [c for c in frame.columns if c != target]
        X = frame[feats].apply(pd.to_numeric, errors="raise").to_numpy(float)
        codes, uniques = pd.factorize(frame[target])
        return cls(name, X, codes, [str(c) for c in feats], list(range(len(uniques))))

    @classmethod
    def from_csv(cls, path: str, target: Optional[str] = None, name: Optional[str] = None) -> "Dataset":
        return cls.from_frame(name or Path(path).stem, pd.read_csv(path), target)

    @classmethod
    def from_file(cls, path: str, target: Optional[str] = None, name: Optional[str] = None) -> "Dataset":
        """CSV/TSV/ARFF genérico (features numéricas). Em ARFF a ORDEM DAS CLASSES declarada é preservada."""
        p = Path(path)
        if p.suffix.lower() != ".arff":
            return cls.from_csv(path, target, name)
        from core.arff_schema import parse_arff_class_order
        from core.data_loading import load_tabular
        frame, _ = load_tabular(p)
        target = target or str(frame.columns[-1])
        order = [str(c) for c in parse_arff_class_order(p, target)]
        y_raw = frame[target].map(lambda v: v.decode() if isinstance(v, bytes) else str(v))
        if order and set(y_raw.unique()) <= set(order):
            codes = y_raw.map({c: i for i, c in enumerate(order)}).to_numpy()
            labels = list(range(len(order)))
        else:
            codes, uniques = pd.factorize(y_raw)
            labels = list(range(len(uniques)))
        feats = [c for c in frame.columns if c != target]
        X = frame[feats].apply(pd.to_numeric, errors="raise").to_numpy(float)
        return cls(name or p.stem, X, codes, [str(c) for c in feats], labels)


@dataclass
class ArmSpec:
    arm_id: str
    family: str                  # mlp | c45 | trepan_original | trepan_reloaded
    description: str
    tree_space: str = "orig"     # orig | enr
    oracle: Optional[str] = None  # mlp_original | mlp_ontological | selected
    semantic: str = "none"       # none | real | shuffled
    lam: float = 0.0
    max_n: Optional[int] = None
    active_queries: bool = True
    error_focus: bool = True
    needs_semantic: bool = False
    role: str = "main"           # main | ablation | sensitivity


def default_arms(cfg: BenchmarkConfig) -> List[ArmSpec]:
    arms = [
        ArmSpec("mlp_original", "mlp", "MLP treinado no espaço original"),
        ArmSpec("mlp_ontological", "mlp", "MLP treinado no espaço enriquecido (OWL real)", tree_space="enr", needs_semantic=True),
        ArmSpec("c45", "c45", "C4.5 supervisionado com rótulos reais"),
        ArmSpec("trepan_original", "trepan_original", "TREPAN Original -> MLP Original", oracle="mlp_original"),
        ArmSpec("reloaded_lambda0", "trepan_reloaded", "Reloaded sem semântica (λ=0), mesma infraestrutura -> MLP Original",
                oracle="mlp_original", lam=0.0),
        ArmSpec("reloaded_semantic_score", "trepan_reloaded", "Reloaded + score semântico (features originais) -> MLP Original",
                oracle="mlp_original", semantic="real", lam=1.0, needs_semantic=True),
        ArmSpec("reloaded_owl_features", "trepan_reloaded", "Reloaded + features OWL, sem score semântico (λ=0) -> MLP Original",
                tree_space="enr", oracle="mlp_original", semantic="real", lam=0.0, needs_semantic=True),
        ArmSpec("reloaded_owl_full", "trepan_reloaded", "Reloaded + OWL real completa -> MLP Original (mesmo oráculo)",
                tree_space="enr", oracle="mlp_original", semantic="real", lam=1.0, needs_semantic=True),
        ArmSpec("reloaded_owl_shuffled", "trepan_reloaded", "CONTROLO NEGATIVO: Reloaded + OWL permutada -> MLP Original",
                tree_space="enr", oracle="mlp_original", semantic="shuffled", lam=1.0, needs_semantic=True),
        ArmSpec("reloaded_e2e", "trepan_reloaded", "Pipeline completo: Reloaded + OWL real -> oráculo seleccionado pelo gate",
                tree_space="enr", oracle="selected", semantic="real", lam=1.0, needs_semantic=True),
    ]
    if cfg.extra_ablations:
        arms += [
            ArmSpec("trepan_original_no_mofn", "trepan_original", "TREPAN Original sem m-of-n (max_n=1)", oracle="mlp_original", max_n=1, role="ablation"),
            ArmSpec("reloaded_owl_full_no_mofn", "trepan_reloaded", "Reloaded OWL completa sem m-of-n", tree_space="enr", oracle="mlp_original",
                    semantic="real", lam=1.0, max_n=1, needs_semantic=True, role="ablation"),
            ArmSpec("reloaded_owl_full_no_active_queries", "trepan_reloaded", "Reloaded OWL completa sem active queries", tree_space="enr",
                    oracle="mlp_original", semantic="real", lam=1.0, active_queries=False, needs_semantic=True, role="ablation"),
            ArmSpec("reloaded_owl_full_no_error_focus", "trepan_reloaded", "Reloaded OWL completa sem refinamento focado no erro", tree_space="enr",
                    oracle="mlp_original", semantic="real", lam=1.0, error_focus=False, needs_semantic=True, role="ablation"),
        ]
    for lam in cfg.semantic_lambdas:
        if float(lam) not in (0.0, 1.0):
            arms.append(ArmSpec(f"reloaded_owl_full_lambda{lam:g}", "trepan_reloaded", f"Sensibilidade λ={lam:g}", tree_space="enr",
                                oracle="mlp_original", semantic="real", lam=float(lam), needs_semantic=True, role="sensitivity"))
    return arms


NOT_APPLICABLE_ABLATIONS = {
    "sem_reasoner": "o reasoner é pré-requisito do quality gate OWL; desligá-lo invalida a ontologia (não exposto como switch).",
    "sem_relational_features": "o OntologyProcessor não expõe um switch independente para features relacionais.",
    "sem_aggregates": "idem: agregados são gerados no mesmo passo que as demais features derivadas.",
    "sem_constraints": "constraints de domínio só actuam na geração de queries do extractor da GUI (não neste runner).",
    "sem_semantic_pruning": "o Reloaded actual não implementa poda semântica (só colapso de subárvores idênticas).",
}


# ------------------------------------------------------------------------------------- MLP helpers
def fit_mlp(X, y, seed: int, trials: int = 0, inner_splits: int = 3):
    """MLP com a MESMA rotina e orçamento para original e ontológico. trials>0: pesquisa aleatória por CV interna."""
    t0 = time.perf_counter()
    info: Dict[str, Any] = {"trials": int(trials), "selection": "adaptive_factory_default"}
    if trials > 0:
        rng = np.random.default_rng(seed)
        grid = [dict(hidden_layer_sizes=h, alpha=a) for h in [(16,), (32,), (32, 16), (64, 32)] for a in (1e-4, 1e-3, 1e-2)]
        picks = [grid[i] for i in rng.choice(len(grid), size=min(trials, len(grid)), replace=False)]
        best, best_score = None, -1.0
        for cfg in picks:
            scores = []
            for tr, va in inner_folds(y, n_splits=inner_splits, seed=seed):
                m = build_mlp_for_data(X[tr], y[tr], random_state=seed, **cfg).fit(X[tr], y[tr])
                scores.append(balanced_accuracy_score(y[va], m.predict(X[va])))
            if np.mean(scores) > best_score:
                best, best_score = cfg, float(np.mean(scores))
        info.update(selection="random_search_inner_cv", chosen=str(best), inner_cv_balanced_accuracy=best_score)
        model = build_mlp_for_data(X, y, random_state=seed, **best).fit(X, y)
    else:
        model = build_mlp_for_data(X, y, random_state=seed).fit(X, y)
    info["train_time"] = time.perf_counter() - t0
    return model, info


def select_oracle(X_train, X_train_enriched, y_train, *, seed: int, margin: float, inner_splits: int = 3) -> Dict[str, Any]:
    """Gate do professor: usa APENAS treino (CV interna). Não recebe, por construção, nenhum dado de teste."""
    res = {"gate_margin": float(margin), "available": X_train_enriched is not None}
    if X_train_enriched is None:
        res.update(accepted=False, reason="semântica indisponível")
        return res
    so, se = [], []
    for tr, va in inner_folds(y_train, n_splits=inner_splits, seed=seed):
        mo = build_mlp_for_data(X_train[tr], y_train[tr], random_state=seed).fit(X_train[tr], y_train[tr])
        me = build_mlp_for_data(X_train_enriched[tr], y_train[tr], random_state=seed).fit(X_train_enriched[tr], y_train[tr])
        so.append(balanced_accuracy_score(y_train[va], mo.predict(X_train[va])))
        se.append(balanced_accuracy_score(y_train[va], me.predict(X_train_enriched[va])))
    gain = float(np.mean(se) - np.mean(so))
    res.update(inner_cv_original=float(np.mean(so)), inner_cv_ontological=float(np.mean(se)), inner_cv_gain=gain,
               accepted=bool(gain > margin), reason=f"ganho CV interno {gain:+.4f} {'>' if gain > margin else '<='} margem {margin}")
    return res


@dataclass
class BenchmarkResult:
    dataset: str
    config: Dict[str, Any]
    rows: List[Dict[str, Any]]
    predictions: pd.DataFrame
    splits: List[Dict[str, Any]]
    semantic_report: List[Dict[str, Any]]
    skipped: List[Dict[str, Any]]
    dataset_hash: str = ""
    ontology_hash: str = "none"
    n_samples: int = 0
    n_features: int = 0
    n_classes: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    arms: List[Dict[str, Any]] = field(default_factory=list)
    experiment_id: str = ""
    manifest: Dict[str, Any] = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class BenchmarkRunner:
    """Executa o protocolo para QUALQUER dataset. A ontologia é opcional (caminho OWL ou provider)."""

    def __init__(self, config: BenchmarkConfig = BenchmarkConfig(), arms: Optional[List[ArmSpec]] = None, verbose: bool = False):
        self.cfg = config
        self.arms = arms or default_arms(config)
        self.verbose = verbose

    # ----------------------------------------------------------------------------- public
    def run(self, dataset: Dataset, ontology=None) -> BenchmarkResult:
        provider = self._provider(ontology)
        ontology_hash = mf.hash_file(ontology) if isinstance(ontology, (str, Path)) else (
            mf.stable_hash({"provider": getattr(ontology, "name", "none")}) if ontology is not None else "none")
        ds_hash = mf.hash_dataset(dataset.X, dataset.y, dataset.feature_names)
        splits = make_splits(dataset.y, scheme=self.cfg.scheme, seeds=self.cfg.seeds, test_size=self.cfg.test_size,
                             n_splits=self.cfg.n_splits, n_repeats=self.cfg.n_repeats, base_seed=self.cfg.base_seed)
        rows: List[Dict[str, Any]] = []
        pred_frames: List[pd.DataFrame] = []
        sem_report: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        split_info = []
        for sp in splits:
            assert_disjoint(sp)
            r, p, s, k = self._run_split(dataset, sp, provider)
            rows += r; sem_report.append(s); skipped += k
            if p is not None:
                pred_frames.append(p)
            split_info.append({"split_id": sp.split_id, "seed": sp.seed, "repeat": sp.repeat, "fold": sp.fold,
                               "split_hash": sp.split_hash, "n_train": sp.n_train, "n_test": sp.n_test, "scheme": sp.scheme,
                               "stratified": sp.stratified})
        split_hash = mf.stable_hash([s["split_hash"] for s in split_info])
        cfg_dict = self.cfg.to_dict()
        counts = {str(k): int(v) for k, v in zip(*np.unique(dataset.y, return_counts=True))}
        exp_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{dataset.name}_{mf.stable_hash(cfg_dict)[:8]}"
        manifest = mf.build_manifest(experiment_id=exp_id, dataset_name=dataset.name, dataset_hash=ds_hash,
                                     ontology_hash=ontology_hash, config=cfg_dict, seeds=[s["seed"] for s in split_info],
                                     split_hash=split_hash, extra={"scheme": self.cfg.scheme,
                                                                    "arms": [a.arm_id for a in self.arms]})
        preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
        return BenchmarkResult(dataset.name, cfg_dict, rows, preds, split_info, sem_report, skipped, ds_hash, ontology_hash,
                               len(dataset.y), dataset.X.shape[1], len(counts), counts, [asdict(a) for a in self.arms],
                               exp_id, manifest)

    # ----------------------------------------------------------------------------- internals
    def _provider(self, ontology):
        if ontology is None:
            return NoSemanticProvider()
        if isinstance(ontology, (str, Path)):
            return OwlSemanticProvider(str(ontology), self.cfg.reasoner_engine, self.cfg.onto_weight)
        return ontology   # provider programático (ex.: GroupSemanticProvider)

    def _budget(self, n_train: int) -> Dict[str, int]:
        b = self.cfg.tree
        min_sample = b.min_sample if b.min_sample is not None else max(n_train, min(1000, max(120, 3 * n_train)))
        max_queries = b.max_queries if b.max_queries is not None else b.max_nodes * min_sample
        return dict(min_sample=int(min_sample), max_queries=int(max_queries), max_nodes=b.max_nodes, max_depth=b.max_depth,
                    max_n=b.max_n, beam_width=b.beam_width, min_samples_leaf=b.min_samples_leaf,
                    max_features_per_node=b.max_features_per_node)

    def _run_split(self, ds: Dataset, sp: SplitSpec, provider):
        cfg = self.cfg
        seed = int(sp.seed) + 1000 * int(sp.repeat) + int(sp.fold)
        Xtr, ytr = ds.X[sp.train_idx], ds.y[sp.train_idx]
        Xte, yte = ds.X[sp.test_idx], ds.y[sp.test_idx]
        labels = ds.class_labels
        minority = minority_class(ytr, labels)            # só do treino
        guard = EvaluationProtocolGuard(run_id=f"{ds.name}:{sp.split_id}")
        guard.record_selection(PartitionRole.TRAIN, "fit_all_models_on_train", split_hash=sp.split_hash)

        # ---- semântica (ajustada só no treino)
        ctx_real = provider.build(Xtr, ds.feature_names, seed)
        ctx_shuf = ShuffledSemanticProvider(provider).build(Xtr, ds.feature_names, seed) if ctx_real.available else ctx_real
        sem = {"split_id": sp.split_id, "provider": ctx_real.provider, "semantic_available": bool(ctx_real.available),
               "ontology_valid": ctx_real.ontology_valid, "reason": ctx_real.reason,
               "n_derived_features": len(ctx_real.onto_idx), "semantic_time": ctx_real.semantic_time,
               "reasoner_time": ctx_real.reasoner_time, "info": ctx_real.info,
               "real_structure_signature": ctx_real.structure_signature() if ctx_real.available else None,
               "shuffled_structure_signature": ctx_shuf.structure_signature() if ctx_shuf.available else None}

        # ---- MLPs (mesmo procedimento e orçamento)
        mlp_o, info_o = fit_mlp(Xtr, ytr, seed, cfg.mlp_trials, cfg.inner_cv_splits)
        mlp_e = info_e = None
        Xtr_enr = Xte_enr = None
        if ctx_real.available:
            Xtr_enr, Xte_enr = ctx_real.enrich(Xtr), ctx_real.enrich(Xte)
            mlp_e, info_e = fit_mlp(Xtr_enr, ytr, seed, cfg.mlp_trials, cfg.inner_cv_splits)
        # ---- contrato do oráculo: o MLP é CONGELADO (hash dos pesos = oracle_id) e é exatamente este objeto que todas as
        # árvores do split consultam; o oráculo ontológico (opcional) é outro objeto, identificado como tal.
        fo = freeze_oracle(mlp_o, Xtr, builder="benchmark_mlp_original")
        fe = freeze_oracle(mlp_e, Xtr_enr, builder="benchmark_mlp_ontological") if mlp_e is not None else None
        guard.record_selection(PartitionRole.VALIDATION, "oracle_gate_inner_cv_on_train_only", folds=cfg.inner_cv_splits)
        gate = select_oracle(Xtr, Xtr_enr, ytr, seed=seed, margin=cfg.oracle_gate_margin, inner_splits=cfg.inner_cv_splits)
        sem.update(mlp_enrichment_accepted=bool(gate.get("accepted")), oracle_gate=gate,
                   trepan_semantics_available=bool(ctx_real.available))

        # ---- predições dos oráculos no conjunto de avaliação (cada um no SEU espaço)
        oracle_preds = {"mlp_original": np.asarray(mlp_o.predict(Xte))}
        acc_o = float(np.mean(oracle_preds["mlp_original"] == yte))
        infos = {"mlp_original": OracleInfo(ORACLE_ORIGINAL, "MLPClassifier(original features)", Xtr.shape[1],
                                            f"sklearn-pipeline;seed={seed};oracle_id={fo.oracle_id}", acc_o)}
        if mlp_e is not None:
            oracle_preds["mlp_ontological"] = np.asarray(mlp_e.predict(Xte_enr))
            infos["mlp_ontological"] = OracleInfo(ORACLE_ONTOLOGICAL, "MLPClassifier(enriched features)", Xtr_enr.shape[1],
                                                  f"sklearn-pipeline;seed={seed};oracle_id={fe.oracle_id}", float(np.mean(oracle_preds["mlp_ontological"] == yte)))
        selected = "mlp_ontological" if (gate.get("accepted") and mlp_e is not None) else "mlp_original"
        sem["selected_oracle"] = selected

        budget = self._budget(sp.n_train)
        rows, preds, skipped = [], {}, []
        base = dict(dataset=ds.name, split_id=sp.split_id, seed=int(sp.seed), repeat=sp.repeat, fold=sp.fold, split_hash=sp.split_hash,
                    scheme=sp.scheme, n_train=sp.n_train, n_test=sp.n_test, minority_label=str(minority))
        c45_model = None
        for spec in self.arms:
            if spec.needs_semantic and not ctx_real.available:
                skipped.append({"split_id": sp.split_id, "arm": spec.arm_id, "reason": "semantic_available=false: " + ctx_real.reason})
                continue
            try:
                row, pred = self._run_arm(spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf,
                                          oracle_preds, infos, selected, budget, seed, labels, minority, fo, fe)
            except Exception as exc:   # um braço falhado não derruba o benchmark: fica registado
                skipped.append({"split_id": sp.split_id, "arm": spec.arm_id, "reason": f"erro: {type(exc).__name__}: {exc}"})
                continue
            row.update(base); row.update(arm=spec.arm_id, family=spec.family, role=spec.role, description=spec.description,
                                         semantic_source=(spec.semantic if spec.semantic != "none" else "none"),
                                         semantic_available=bool(ctx_real.available), mlp_enrichment_accepted=sem["mlp_enrichment_accepted"])
            rows.append(row); preds[spec.arm_id] = pred
        sem["oracle_contract"] = self._verify_oracle_contract(rows, fo, fe)
        guard.record_final_evaluation(PartitionRole.TEST, "final_metrics_all_arms_once", n_test=sp.n_test)
        sem["protocol_audit"] = guard.audit()
        # paridade de predições Original vs Reloaded λ=0 (verificação do mirror)
        if "trepan_original" in preds and "reloaded_lambda0" in preds:
            for r in rows:
                if r["arm"] == "reloaded_lambda0":
                    r["identical_predictions_to_trepan_original"] = bool(np.array_equal(preds["trepan_original"], preds["reloaded_lambda0"]))
        frame = None
        if cfg.save_predictions:
            data = {"dataset": ds.name, "split_id": sp.split_id, "seed": int(sp.seed), "test_row_index": sp.test_idx,
                    "y_real": yte}
            for k, v in oracle_preds.items():
                data[f"oraclepred__{infos[k].name}"] = v
            for k, v in preds.items():
                data[f"pred__{k}"] = v
            frame = pd.DataFrame(data)
        return rows, frame, sem, skipped

    @staticmethod
    def _verify_oracle_contract(rows, fo, fe) -> Dict[str, Any]:
        """Prova, por split, que todas as árvores do oráculo partilhado consultaram EXATAMENTE o mesmo modelo congelado."""
        shared = [r for r in rows if r.get("oracle_scope") == "shared_frozen_oracle"]
        ids = {r["oracle_id"] for r in shared}
        if ids and ids != {fo.oracle_id}:
            raise OracleContractViolation(f"Árvores do oráculo partilhado com oracle_id distintos: {sorted(ids)}")
        fo.verify_unchanged()
        if fe is not None:
            fe.verify_unchanged()
        return {"oracle_id_original": fo.oracle_id, "oracle_id_ontological": None if fe is None else fe.oracle_id,
                "shared_oracle_arms": sorted(r["arm"] if "arm" in r else "?" for r in shared),
                "single_oracle_for_shared_arms": True, "unchanged_after": True,
                "queries_per_scope": {k: v["queries"] for k, v in fo.calls.items()}}

    def _run_arm(self, spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf,
                 oracle_preds, infos, selected, budget, seed, labels, minority, fo=None, fe=None):
        cfg = self.cfg
        row: Dict[str, Any] = {"mlp_training_time": None, "tree_training_time": None, "query_time": None,
                               "semantic_processing_time": 0.0, "reasoner_time": 0.0, "counterfactual_time": None}
        ctx = ctx_shuf if spec.semantic == "shuffled" else ctx_real
        if spec.family == "mlp":
            model, info = (mlp_o, info_o) if spec.arm_id == "mlp_original" else (mlp_e, info_e)
            pred = np.asarray(model.predict(Xte if spec.arm_id == "mlp_original" else Xte_enr))
            row.update(evaluate_model(y_real=yte, prediction=pred, labels=labels, minority_label=minority))
            row.update(oracle_name=None, oracle_type=None, oracle_feature_space=None, oracle_version=None,
                       oracle_accuracy_real_labels=None, mlp_training_time=info["train_time"], mlp_selection=info["selection"],
                       mlp_trials=info["trials"], feature_space="enriched" if spec.arm_id == "mlp_ontological" else "original",
                       oracle_id=(fo.oracle_id if spec.arm_id == "mlp_original" else getattr(fe, "oracle_id", None)),
                       oracle_scope="frozen_oracle_model_itself")
            if spec.arm_id == "mlp_ontological":
                row["semantic_processing_time"], row["reasoner_time"] = ctx_real.semantic_time, ctx_real.reasoner_time
            return row, pred
        if spec.family == "c45":
            t0 = time.perf_counter()
            model = C45Classifier(confidence_factor=0.25, min_samples_leaf=2, random_state=seed).fit(Xtr, ytr)  # SÓ rótulos reais
            row["tree_training_time"] = time.perf_counter() - t0
            pred = np.asarray(model.predict(Xte))
            row.update(evaluate_model(y_real=yte, prediction=pred, labels=labels, minority_label=minority))
            row.update(oracle_name=None, oracle_type=None, oracle_feature_space=None, oracle_version=None,
                       oracle_accuracy_real_labels=None, feature_space="original", uses_oracle=False, uses_real_labels_for_training=True,
                       oracle_id=None, oracle_scope="no_oracle_real_labels")
            cx = c45_complexity(model)
            row.update(self._complexity(cx, 0))
            return row, pred

        # ---- árvores TREPAN
        key = selected if spec.oracle == "selected" else spec.oracle
        info = infos[key]
        use_enr = spec.tree_space == "enr"
        if key == "mlp_original":
            oracle = OriginalOracleProjection(fo, ctx.orig_idx) if use_enr else fo      # oráculo CONGELADO e partilhado
        else:
            oracle = fe                              # o oráculo ontológico (outro modelo, congelado) opera no espaço enriquecido
            if not use_enr:
                raise ValueError("MLP Ontológico só pode ser oráculo de uma árvore no espaço enriquecido.")
        X_fit = ctx.enrich(Xtr) if use_enr else Xtr
        names = ctx.enriched_names if use_enr else ds.feature_names
        common = dict(budget, random_state=seed)
        if spec.max_n is not None:
            common["max_n"] = spec.max_n
        frozen = fo if key == "mlp_original" else fe
        scope_cm = frozen.scope(spec.arm_id)
        t0 = time.perf_counter()
        with scope_cm:
            if spec.family == "trepan_original":
                tree = TrepanOriginalClassifier(**common).fit(X_fit, oracle=oracle, feature_names=names)
            else:
                # λ escala TODOS os canais semânticos (alpha, beta, expoente do peso, força de grupo): λ=0 => semântica OFF
                kw = dict(common, alpha=cfg.alpha * spec.lam, beta=cfg.beta * spec.lam, error_focused_refinement=bool(spec.error_focus),
                          semantic_gain_strength=float(spec.lam), semantic_group_strength=0.15 * float(spec.lam),
                          semantic_active_query_fraction=0.65 if spec.active_queries else 0.0)
                fit_kw: Dict[str, Any] = {}
                if spec.lam > 0 and spec.semantic != "none":
                    sl = ctx.orig_idx if not use_enr else list(range(ctx.n_features))
                    fit_kw.update(semantic_feature_weights=np.asarray(ctx.weights)[sl],
                                  semantic_feature_groups=[ctx.groups[i] for i in sl],
                                  semantic_relatedness_matrix=np.asarray(ctx.relatedness)[np.ix_(sl, sl)],
                                  semantic_feature_depths=np.asarray(ctx.depths)[sl])
                if use_enr:
                    fit_kw["query_projector"] = ctx.query_projector()
                tree = TrepanReloadedClassifier(**kw).fit(X_fit, oracle=oracle, feature_names=names, **fit_kw)
        row["tree_training_time"] = time.perf_counter() - t0
        X_eval = ctx.enrich(Xte) if use_enr else Xte
        pred = np.asarray(tree.predict(X_eval))
        row.update(evaluate_model(y_real=yte, prediction=pred, labels=labels, minority_label=minority, oracle=info,
                                  oracle_prediction=oracle_preds[key]))
        row.update(oracle_name=info.name, oracle_type=info.type, oracle_feature_space=info.feature_space,
                   oracle_version=info.version, oracle_accuracy_real_labels=info.accuracy_real_labels,
                   feature_space="enriched" if use_enr else "original", uses_oracle=True, uses_real_labels_for_training=False,
                   oracle_id=frozen.oracle_id, oracle_queries=int(frozen.calls.get(spec.arm_id, {}).get("queries", 0)),
                   oracle_scope="shared_frozen_oracle" if key == "mlp_original" else "ontological_oracle_separate")
        cx = trepan_complexity(tree)
        # SÓ splits presentes na árvore FINAL (a auditoria cobre a árvore bruta, incluindo subárvores podadas)
        final_internal = {n.node_id for n in tree.iter_nodes() if not n.is_leaf}
        mirrored = bool(getattr(tree, "semantic_effect_mirror_applied_", False))
        audit_rows = [r for r in getattr(tree, "semantic_split_audit_", []) if r.get("node_id") in final_internal] if not mirrored else []
        sem_splits = sum(1 for r in audit_rows if r.get("ontology_influenced") or abs(r.get("semantic_bonus", 0.0)) > 1e-12)
        row["semantic_decision_changed_count"] = sum(1 for r in audit_rows if r.get("decision_changed"))
        row.update(self._complexity(cx, sem_splits))
        row.update(membership_queries=int(tree.membership_queries_), query_budget=int(tree.max_queries),
                   query_budget_exhausted=bool(getattr(tree, "query_budget_exhausted_", False)),
                   query_time=float(getattr(tree, "query_time_", 0.0)),
                   semantic_mirror_applied=bool(getattr(tree, "semantic_effect_mirror_applied_", False)),
                   lambda_semantic=float(spec.lam), tree_max_n=int(common["max_n"]))
        if spec.semantic != "none":
            row["semantic_processing_time"], row["reasoner_time"] = ctx.semantic_time, ctx.reasoner_time
        return row, pred

    @staticmethod
    def _complexity(cx: Dict[str, Any], semantic_splits: int) -> Dict[str, Any]:
        return dict(node_count=cx["node_count"], internal_nodes=cx["internal_node_count"], leaf_count=cx["leaf_count"],
                    depth=cx["depth"], average_leaf_depth=cx["average_leaf_depth"], average_rule_length=cx["average_rule_length"],
                    features_used=cx["n_features_used"], m_of_n_count=cx["m_of_n_count"], semantic_split_count=int(semantic_splits))
