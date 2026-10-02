"""Atribuição do ganho do TREPAN Reloaded à ontologia (gate treinado só no treino).

O Reloaded só deve diferir do Original por causa do *conhecimento* da ontologia. Medir o
Reloaded contra o Original não prova isso: a maquinaria do Reloaded (critério de ganho,
candidatos m-of-n, amostragem ativa) muda a árvore com qualquer estrutura semântica,
certa ou errada, e a variação do gerador aleatório também. Medimos portanto o Reloaded
real contra **controlos com a mesma maquinaria e sem significado**:

* espaço original : mesma ontologia com a atribuição feature->entidade baralhada;
* espaço aumentado: mesmas formas de feature derivada, mas sobre colunas ao acaso.

Gate (validação interna do treino, nunca o teste): para cada fold interno treina-se o
Reloaded real e K controlos no mesmo treino interno e mede-se a fidelidade ao professor em
linhas de validação interna nunca vistas. A semântica só é aceite (``ATTRIBUTED``) se o ganho
médio sobre os controlos for > ``min_gain`` e o real vencer a maioria das comparações
não empatadas. Caso contrário o Reloaded degrada para o Original e não reivindica ganho.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from core.semantic_teacher import SemanticTeacher

ATTRIBUTED = "ATTRIBUTED_TO_ONTOLOGY"
NOT_ATTRIBUTED = "NOT_ATTRIBUTED_NO_GAIN_OVER_CONTROLS"
NOT_ATTRIBUTED_TIES = "NOT_ATTRIBUTED_ALL_TIES"
NOT_EVALUATED = "NOT_EVALUATED"


@dataclass(frozen=True)
class AttributionConfig:
    """Configuração congelada antes de ver resultados."""

    controls: int = 3
    inner_folds: int = 3
    min_gain: float = 0.0          # ganho médio de fidelidade sobre os controlos
    min_win_fraction: float = 0.6  # fração de vitórias entre comparações não empatadas
    random_state: int = 42


class AugmentedPredictor:
    """Modelo treinado no espaço aumentado, com a interface ``predict(Z)`` do espaço do modelo."""

    def __init__(self, model, augmenter: SemanticTeacher):
        self.model, self.augmenter = model, augmenter

    def predict(self, Z):
        return self.model.predict(self.augmenter.augment(Z))


Variant = Callable[[np.ndarray, np.ndarray], Any]  # (Z_treino, y_treino) -> objeto com predict(Z)


def evaluate_attribution(
    Z,
    y,
    oracle,
    real_variant: Variant,
    control_variant: Callable[[int], Variant],
    config: AttributionConfig = AttributionConfig(),
) -> Dict[str, Any]:
    """Compara o Reloaded real com K controlos por fidelidade ao professor, em folds internos."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    folds = int(min(config.inner_folds, counts.min()))
    report: Dict[str, Any] = {
        "config": asdict(config), "scope": "training_internal_cv_only", "test_used": False,
        "metric": "oracle_fidelity_on_inner_validation",
    }
    if folds < 2:
        report.update(status=NOT_EVALUATED, reason="too_few_samples_per_class", attributed=False)
        return report
    cv = StratifiedKFold(folds, shuffle=True, random_state=config.random_state)
    rows: List[Dict[str, Any]] = []
    diffs: List[float] = []
    for fold, (tr, va) in enumerate(cv.split(Z, y)):
        teacher_val = np.asarray(oracle.predict(Z[va]))
        real = real_variant(Z[tr], y[tr])
        fid_real = float(np.mean(np.asarray(real.predict(Z[va])) == teacher_val))
        fid_ctrl = []
        for k in range(config.controls):
            model = control_variant(fold * 1000 + k)(Z[tr], y[tr])
            fid_ctrl.append(float(np.mean(np.asarray(model.predict(Z[va])) == teacher_val)))
        for f in fid_ctrl:
            diffs.append(fid_real - f)
        rows.append({"fold": fold, "fidelity_real": fid_real, "fidelity_controls": fid_ctrl,
                     "gain_over_controls_mean": float(fid_real - np.mean(fid_ctrl))})
    d = np.asarray(diffs)
    wins, losses, ties = int((d > 1e-12).sum()), int((d < -1e-12).sum()), int((np.abs(d) <= 1e-12).sum())
    decided = wins + losses
    win_fraction = (wins / decided) if decided else None
    mean_gain = float(d.mean()) if len(d) else 0.0
    if decided == 0:
        status, attributed = NOT_ATTRIBUTED_TIES, False
    elif mean_gain > config.min_gain and win_fraction >= config.min_win_fraction:
        status, attributed = ATTRIBUTED, True
    else:
        status, attributed = NOT_ATTRIBUTED, False
    report.update(status=status, attributed=attributed, mean_gain_over_controls=mean_gain,
                  wins=wins, losses=losses, ties=ties, win_fraction=win_fraction,
                  comparisons=int(len(d)), per_fold=rows)
    return report


def make_random_derived_teacher(teacher: SemanticTeacher, Z_train, *, seed: int) -> SemanticTeacher:
    """Controlo: mesmas features derivadas (forma e número), mas sobre colunas escolhidas ao acaso.

    Mantém a operação de cada feature (agregado padronizado, diferença, rácio, contraste...), mas
    troca as colunas-fonte por outras ao acaso; estatísticas aprendidas só no treino; a família e os
    papéis OWL desaparecem. O professor (rótulos) continua a ser o real: só muda o que o árvore vê.
    """
    rng = np.random.default_rng(seed)
    processor = copy.deepcopy(teacher.processor)
    cols = list(processor.input_features_)
    Z_train = np.asarray(Z_train, dtype=float)
    frame = pd.DataFrame(Z_train[:, teacher.z_indices], columns=teacher.input_columns)
    selected = set(teacher.selected_features)
    new_specs = []
    for spec in processor.feature_specs_:
        spec = dict(spec)
        if spec["name"] in selected:
            kind = spec["kind"]
            if kind == "hierarchical_aggregate":
                k = min(len(spec["sources"]), len(cols))
                sources = list(rng.choice(cols, size=k, replace=False))
                values = frame[sources]
                scales = values.std(axis=0, ddof=0).where(lambda s: s.abs() > 1e-12, 1.0)
                spec.update(sources=sources, centers={c: float(values[c].mean()) for c in sources},
                            scales={c: float(scales[c]) for c in sources})
            elif kind == "relational":
                keys = [k for k in ("left", "right", "denominator") if spec.get(k) is not None]
                olds = list(dict.fromkeys(spec[k] for k in keys))
                # Denominadores só em colunas estritamente positivas no treino (se existirem): um
                # denominador centrado em zero daria ao controlo caudas pesadas e pioraria-o de forma
                # artificial, inflacionando o ganho atribuído ao real. Propriedade medida nos dados.
                positive = [c for c in cols if float(frame[c].min()) > 0.0]
                denominators = {spec["denominator"]} if spec.get("denominator") is not None else set()
                mapping, used = {}, set()
                for old in olds:
                    pool = [c for c in (positive if old in denominators and positive else cols) if c not in used]
                    pool = pool or [c for c in cols if c not in used] or cols
                    mapping[old] = str(rng.choice(pool)); used.add(mapping[old])
                for k in keys:
                    spec[k] = mapping[spec[k]]
                spec["sources"] = list(dict.fromkeys(spec[k] for k in keys))
            elif kind == "constraint":
                src = str(rng.choice(cols))
                spec.update(source=src, threshold=float(np.median(frame[src])), provenance="random_control")
            spec.update(family=None, roles=None, owl_entities=[], provenance="random_control", reasoner_inferred=False)
        new_specs.append(spec)
    processor.feature_specs_ = new_specs
    control = copy.copy(teacher)
    control.processor = processor
    control.audit_ = {**teacher.audit_, "control": "random_derived_features", "control_seed": int(seed)}
    return control
