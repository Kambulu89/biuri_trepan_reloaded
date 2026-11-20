"""Identidade do dataset: dois testes metamórficos independentes, com a MESMA OWL, comparação instância a instância.

1. Muda APENAS o nome do ficheiro (conteúdo, ``dataset_id``, OWL, partição, configuração e semente iguais).
2. Muda APENAS o ``dataset_id`` (mesmo ficheiro, mesma OWL, mesma partição, configuração e semente).

Compara-se, por instância de teste: o índice da linha, a previsão do oráculo (MLP congelado) e a previsão de cada árvore
(C4.5, TREPAN Original, Reloaded sem OWL, Reloaded com OWL e com OWL permutada). Duas execuções podem ter a mesma accuracy e
discordar em casos individuais, por isso nenhuma métrica agregada substitui esta comparação.

Percurso de produção: ``production_training`` e ``pipeline_v92`` recebem um DataFrame, não um ``dataset_id``. Neles, o teste 1
muda o nome do ficheiro lido por ``load_tabular``; o teste 2 muda o único identificador que recebem (o diretório do artefacto).
Na GUI o ``dataset_id`` é a impressão digital do CONTEÚDO (``core.dataset_schema.file_fingerprint``) e não pode variar com os
dados fixos; ``test_gui_dataset_id_is_content_derived`` verifica isso.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

pytest.importorskip("owlready2")

from core.benchmark.runner import BenchmarkConfig, BenchmarkRunner, Dataset, TreeBudget, default_arms
from core.data_loading import load_tabular
from core.trepan_scientific_tuning import ScientificTrepanSearchConfig
from validation.benchmark import equivalence as eq

ARMS = ("mlp_original", "c45", "trepan_original", "reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled")
NAMES = [f"m{i}" for i in range(6)]
SEED = 7


def _cfg() -> BenchmarkConfig:
    return BenchmarkConfig(seeds=(SEED,), tree=TreeBudget(min_sample=120, max_queries=1200, max_nodes=5, max_depth=4),
                           extra_ablations=False, different_oracle_experiment=False, structure_tuning=True,
                           structure_search=ScientificTrepanSearchConfig(cv_folds=2, cv_repeats=1, purity_epsilon_grid=(0.05,),
                                                                         max_nodes_grid=(5,)),
                           n_boot=30, inner_cv_splits=2)


def _frame() -> pd.DataFrame:
    X, y = make_classification(n_samples=180, n_features=6, n_informative=3, n_redundant=0, n_classes=2,
                               n_clusters_per_class=1, flip_y=0.02, random_state=17)
    df = pd.DataFrame(np.round(X, 6), columns=NAMES)
    df["target"] = np.where(y == 1, "pos", "neg")
    return df


@pytest.fixture(scope="module")
def owl(tmp_path_factory) -> str:
    from core.benchmark.domain_tbox import create_domain_tbox
    path = tmp_path_factory.mktemp("owl") / "mesma_ontologia.owl"
    create_domain_tbox(path, dataset_name="qualquer_nome", groups={"GA": NAMES[:3], "GB": NAMES[3:]})
    return str(path)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("dados")
    a, b = d / "primeiro_ficheiro.csv", d / "zz_outro_nome_de_ficheiro.csv"
    _frame().to_csv(a, index=False)
    _frame().to_csv(b, index=False)
    assert a.read_bytes() == b.read_bytes()
    return str(a), str(b)


def _bench(path: str, name: str, owl: str):
    cfg = _cfg()
    ds = Dataset.from_file(path, "target", name=name)
    res = BenchmarkRunner(cfg, [a for a in default_arms(cfg) if a.arm_id in ARMS]).run(ds, owl)
    assert not res.skipped, res.skipped
    assert set(res.frame()["arm"]) == set(ARMS)                                    # os braços com OWL correram de facto
    return ds, res


def _assert_instance_level_equal(p1: pd.DataFrame, p2: pd.DataFrame):
    """Uma linha por instância de teste: mesmas linhas (split, índice da linha original) e a mesma previsão em cada uma."""
    key = ["split_id", "test_row_index"]
    a = p1.sort_values(key, kind="stable").reset_index(drop=True)
    b = p2.sort_values(key, kind="stable").reset_index(drop=True)
    assert len(a) == len(b) > 0
    assert a[key].equals(b[key]), "instâncias de teste diferentes"
    pred_cols = [c for c in a.columns if c.startswith(("pred__", "oraclepred__"))]
    assert {f"pred__{k}" for k in ARMS} <= set(pred_cols) and any(c.startswith("oraclepred__") for c in pred_cols)
    assert sorted(pred_cols) == sorted(c for c in b.columns if c.startswith(("pred__", "oraclepred__")))
    for c in ["y_real"] + pred_cols:
        va, vb = a[c].to_numpy(), b[c].to_numpy()
        assert np.array_equal(va, vb), f"{c}: {int(np.sum(va != vb))} de {len(va)} instâncias diferem"


# --------------------------------------------------------------------------------------------- benchmark
def test_only_the_file_name_changes(files, owl):
    d1, r1 = _bench(files[0], "mesmo_dataset_id", owl)
    d2, r2 = _bench(files[1], "mesmo_dataset_id", owl)
    assert d1.name == d2.name and np.array_equal(d1.X, d2.X) and np.array_equal(d1.y, d2.y)
    _assert_instance_level_equal(r1.predictions, r2.predictions)
    # nada no resultado depende do ficheiro: TODAS as colunas (incluindo estrutura das árvores e hashes) são iguais
    assert not eq.compare_frames(r1.frame(), r2.frame())


def test_only_the_dataset_id_changes(files, owl):
    d1, r1 = _bench(files[0], "dataset_id_a", owl)
    d2, r2 = _bench(files[0], "zz_dataset_id_b", owl)
    assert d1.name != d2.name and np.array_equal(d1.X, d2.X) and np.array_equal(d1.y, d2.y)
    _assert_instance_level_equal(r1.predictions, r2.predictions)
    # só os identificadores de ARTEFACTO podem mudar; todo o resultado científico (estrutura, splits, hashes científicos) é igual
    artefact = {"dataset", "dataset_hash", "preprocessing_id", "experiment_id"}
    diff_cols = {d.split(": ")[1] for d in eq.compare_frames(r1.frame(), r2.frame())}
    assert diff_cols <= artefact, diff_cols
    assert "dataset" in diff_cols


# --------------------------------------------------------------------------------------------- produção
def _production(path: str, out: Path, owl: str):
    from core.production_training import train_production_dataframe
    df, _ = load_tabular(path)
    rep = train_production_dataframe(df, target="target", out_dir=out, seed=SEED, owl_path=owl, require_reasoner=False,
                                     scientific_tuning=False, use_semantic_teacher=False, augment_reloaded_space=False,
                                     return_artifacts=True)
    art = rep["artifacts"]
    clean = df.reset_index(drop=True)
    te = np.asarray(rep["split"]["test_indices"])
    Xte = clean.drop(columns=["target"]).iloc[te]
    Zte = art["preprocessor"].transform(Xte)
    preds = {"oracle": np.asarray(art["oracle"].predict(Zte)), "mlp_original": np.asarray(art["mlp_original"].predict(Zte)),
             "c45_native": np.asarray(art["c45_native"].predict(Xte.to_numpy(dtype=object))),
             "trepan_original": np.asarray(art["trepan_original"].predict(Zte)),
             "trepan_reloaded": np.asarray(art["trepan_reloaded"].predict(Zte))}
    return rep, te, preds


def _assert_production_equal(a, b):
    (rep1, te1, p1), (rep2, te2, p2) = a, b
    assert np.array_equal(te1, te2)
    assert rep1["manifest"]["owl_sha256"] == rep2["manifest"]["owl_sha256"] is not None   # a mesma OWL foi de facto usada
    for k in p1:
        assert p1[k].shape == te1.shape and np.array_equal(p1[k], p2[k]), f"{k}: {int(np.sum(p1[k] != p2[k]))} instâncias diferem"


def test_production_only_the_file_name_changes(files, owl, tmp_path):
    _assert_production_equal(_production(files[0], tmp_path / "a" / "run", owl), _production(files[1], tmp_path / "b" / "run", owl))


def test_production_only_the_artifact_id_changes(files, owl, tmp_path):
    _assert_production_equal(_production(files[0], tmp_path / "id_a", owl), _production(files[0], tmp_path / "zz_id_b", owl))


def test_pipeline_v92_only_the_file_name_changes(files):
    from core.pipeline_v92 import train_dataset

    def run(path):
        df, _ = load_tabular(path)
        res = train_dataset(df, target="target", seed=SEED, max_trepan_nodes=7)
        te = np.asarray(res.split["test_indices"])
        Zte = res.preprocessor.transform(df.drop(columns=["target"]).iloc[te])
        return te, {"mlp": res.mlp.predict(Zte), "trepan": res.trepan_original.predict(Zte),
                    "c45": res.c45.predict(df.drop(columns=["target"]).iloc[te].to_numpy(dtype=object))}

    (t1, p1), (t2, p2) = run(files[0]), run(files[1])
    assert np.array_equal(t1, t2)
    for k in p1:
        assert np.array_equal(p1[k], p2[k]), k


def test_gui_dataset_id_is_content_derived(files):
    from core.dataset_schema import file_fingerprint
    assert file_fingerprint(files[0]) == file_fingerprint(files[1])          # outro nome, o mesmo conteúdo, o mesmo dataset_id
