"""
Cache de modelos treinados por hash de dataset + ontologia + preset + pré-processamento.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.semantic_version import SEMANTIC_PIPELINE_VERSION
from core.training_config import TrainingPreset, preset_config_hash


def _cache_root() -> Path:
    root = Path(__file__).resolve().parent.parent / "results" / "model_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_dataset_hash(
    X,
    y,
    dataset_name: str = "unknown",
) -> str:
    """Hash completo e determinístico; nenhuma amostragem parcial."""
    import pandas as pd
    frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=object))
    target = y.copy() if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=object), name="__target__")
    # inclui nomes, índice lógico e todos os valores.
    value_hash = pd.util.hash_pandas_object(frame, index=True, categorize=False).to_numpy(dtype="uint64")
    target_hash = pd.util.hash_pandas_object(target, index=True, categorize=False).to_numpy(dtype="uint64")
    h = hashlib.sha256()
    h.update(str(dataset_name).encode("utf-8"))
    h.update(json.dumps([str(c) for c in frame.columns], ensure_ascii=False).encode("utf-8"))
    h.update(value_hash.tobytes()); h.update(target_hash.tobytes())
    return h.hexdigest()[:16]


def _ontology_structure(ontology_obj) -> Dict[str, Any]:
    """Assinatura estrutural: TBox (classes, propriedades, subclasses) e ABox (indivíduos)."""
    def names(method):
        try:
            return sorted(str(x) for x in getattr(ontology_obj, method)())
        except Exception:
            return []
    classes = names("classes")
    subclass = []
    try:
        for cls in ontology_obj.classes():
            for parent in getattr(cls, "is_a", []) or []:
                subclass.append(f"{cls}<{parent}")
    except Exception:
        pass
    abox = []
    try:
        for ind in ontology_obj.individuals():
            abox.append(f"{ind}:{sorted(str(t) for t in getattr(ind, 'is_a', []) or [])}")
    except Exception:
        pass
    return {
        "tbox": {"classes": classes, "data_properties": names("data_properties"),
                 "object_properties": names("object_properties"), "subclass": sorted(subclass)},
        "abox": sorted(abox),
    }


def compute_ontology_hash(ontology_path: Optional[str], ontology_obj=None) -> str:
    """Hash do CONTEÚDO da OWL (não de caminho/mtime).

    Copiar o ficheiro não invalida a cache; editar o conteúdo (TBox ou ABox) invalida,
    mesmo que o tamanho e a data de modificação coincidam.
    """
    if ontology_path:
        p = Path(ontology_path)
        if p.exists():
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()[:12]
    if ontology_obj is not None:
        try:
            return _stable_hash(_ontology_structure(ontology_obj))[:12]
        except Exception:
            pass
    return "no_ontology"


def compute_semantic_config_hash(*configs: Any) -> str:
    """Hash estável das configurações semânticas (processor, gates, enriquecimento)."""
    import dataclasses
    payload = [dataclasses.asdict(c) if dataclasses.is_dataclass(c) and not isinstance(c, type) else c
               for c in configs]
    return _stable_hash(payload)[:12]


def compute_matching_hash(matches) -> str:
    """Hash do matching ARFF<->OWL aceite (feature, entidade, score)."""
    rows = sorted(
        (str(m.get("feature")), str(m.get("entity_name")), round(float(m.get("score", 0.0)), 4),
         bool(m.get("accepted", True)))
        for m in (matches or [])
    )
    return _stable_hash(rows)[:12]


def compute_preprocessing_hash(
    feature_names: Optional[list],
    n_encoded_features: int,
    augmented: bool = False,
) -> str:
    return _stable_hash(
        {
            "feature_names": feature_names or [],
            "n_encoded_features": n_encoded_features,
            "augmented": augmented,
        }
    )[:12]


def build_cache_key(
    *,
    model_role: str,
    dataset_hash: str,
    ontology_hash: str,
    preprocessing_hash: str,
    preset: TrainingPreset,
    feature_space: str = "original",
    n_features: Optional[int] = None,
    feature_names: Optional[list] = None,
    selected_onto_features: Optional[list] = None,
    training_mode: Optional[str] = None,
    semantic_pipeline_version: str = SEMANTIC_PIPELINE_VERSION,
    semantic_config_hash: Optional[str] = None,
    matching_hash: Optional[str] = None,
) -> str:
    return _stable_hash(
        {
            # Um MLP treinado com outra versão/config das features ontológicas nunca é reutilizado.
            "semantic_pipeline_version": semantic_pipeline_version,
            "semantic_config": semantic_config_hash,
            "matching": matching_hash,
            "role": model_role,
            "model_type": model_role,
            "dataset": dataset_hash,
            "ontology": ontology_hash,
            "preprocessing": preprocessing_hash,
            "preset": preset_config_hash(preset),
            "feature_space": feature_space,
            "n_features": n_features,
            "feature_names": feature_names or [],
            "selected_onto_features": selected_onto_features or [],
            "training_mode": training_mode or preset.key,
        }
    )


def _entry_path(cache_key: str) -> Path:
    return _cache_root() / f"{cache_key}.pkl"


def _cached_n_features(cached: Dict[str, Any]) -> Optional[int]:
    n = cached.get("n_features")
    if n is not None:
        return int(n)
    bundle = cached.get("bundle")
    if bundle is not None:
        return getattr(bundle, "n_features", None)
    model = cached.get("model")
    if model is not None:
        steps = getattr(model, "named_steps", None)
        if steps and "scaler" in steps:
            n = getattr(steps["scaler"], "n_features_in_", None)
            if n is not None:
                return int(n)
    eval_split = cached.get("eval_split") or {}
    X_train = eval_split.get("X_train")
    if X_train is not None:
        return int(np.asarray(X_train).shape[1])
    return None


def validate_cached_training(
    cached: Dict[str, Any],
    *,
    expected_n_features: Optional[int] = None,
    expected_feature_space: Optional[str] = None,
    feature_names: Optional[list] = None,
) -> bool:
    """Valida schema da cache; devolve False se incompatível."""
    if cached is None:
        return False

    cached_n = _cached_n_features(cached)
    if expected_n_features is not None and cached_n is not None:
        if int(cached_n) != int(expected_n_features):
            print("\n[CACHE INVALIDATED]")
            print(
                f"Cached model expected {cached_n} features, "
                f"current matrix has {expected_n_features} features."
            )
            print("Retraining model with current feature schema.")
            return False

    cached_space = cached.get("feature_space")
    if (
        expected_feature_space is not None
        and cached_space is not None
        and cached_space != expected_feature_space
    ):
        print("\n[CACHE INVALIDATED]")
        print(
            f"Cached feature_space={cached_space}, "
            f"expected {expected_feature_space}."
        )
        print("Retraining model with current feature schema.")
        return False

    cached_names = cached.get("feature_names")
    if feature_names and cached_names and list(cached_names) != list(feature_names):
        print("\n[CACHE INVALIDATED]")
        print("Cached feature_names differ from current schema.")
        print("Retraining model with current feature schema.")
        return False

    return True


def load_cached_training(
    cache_key: str,
    *,
    expected_n_features: Optional[int] = None,
    expected_feature_space: Optional[str] = None,
    feature_names: Optional[list] = None,
) -> Optional[Dict[str, Any]]:
    path = _entry_path(cache_key)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if not validate_cached_training(
            data,
            expected_n_features=expected_n_features,
            expected_feature_space=expected_feature_space,
            feature_names=feature_names,
        ):
            return None
        data["loaded_from_cache"] = True
        print(f"[INFO] Cache hit: {path.name}")
        return data
    except Exception as exc:
        print(f"[WARN] Falha ao ler cache {path.name}: {exc}")
        return None


def save_cached_training(cache_key: str, payload: Dict[str, Any]) -> Path:
    path = _entry_path(cache_key)
    payload = dict(payload)
    payload["cached_at"] = time.time()
    payload["cache_key"] = cache_key
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path = path.with_suffix(".json")
    meta = {
        k: v
        for k, v in payload.items()
        if k not in ("model", "mlp_trainer_state", "eval_split")
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"[INFO] Modelo guardado em cache: {path.name}")
    return path


def serialize_mlp_trainer_state(trainer) -> Dict[str, Any]:
    classes = getattr(trainer.label_encoder, "classes_", None)
    return {
        "feature_encoders": trainer.feature_encoders,
        "_extra_column_encoders": getattr(trainer, "_extra_column_encoders", {}),
        "label_encoder_classes": list(classes) if classes is not None else None,
        "arff_meta": trainer.arff_meta,
        "last_optimization_summary": trainer.last_optimization_summary,
        "optimization_results": getattr(trainer, "optimization_results", []),
        "bundle_n_features": getattr(getattr(trainer, "bundle", None), "n_features", None),
        "bundle_feature_space": getattr(getattr(trainer, "bundle", None), "feature_space", None),
    }


def restore_mlp_trainer_state(trainer, state: Dict[str, Any]) -> None:
    from sklearn.preprocessing import LabelEncoder

    trainer.feature_encoders = state.get("feature_encoders", {})
    trainer._extra_column_encoders = state.get("_extra_column_encoders", {})
    le = LabelEncoder()
    classes = state.get("label_encoder_classes")
    if classes is not None:
        le.classes_ = np.array(classes)
        trainer.label_encoder = le
    trainer.arff_meta = state.get("arff_meta")
    trainer.last_optimization_summary = state.get("last_optimization_summary")
    trainer.optimization_results = state.get("optimization_results", [])
