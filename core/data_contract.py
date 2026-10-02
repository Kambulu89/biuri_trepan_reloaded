"""Contrato de dados agnóstico para datasets tabulares de classificação."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import json
import math
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from core.scientific_errors import DataContractError

DEFAULT_MISSING_TOKENS = ("", "?", "NA", "N/A", "null")


def _safe_name(name: Any, idx: int) -> str:
    text = "" if name is None else str(name).strip()
    return text or f"coluna_{idx}"


def _normalise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    names = [_safe_name(c, i) for i, c in enumerate(df.columns)]
    seen: Dict[str, int] = {}
    out: List[str] = []
    duplicates: List[str] = []
    for name in names:
        count = seen.get(name, 0)
        if count:
            duplicates.append(name)
            candidate = f"{name}__{count+1}"
        else:
            candidate = name
        seen[name] = count + 1
        out.append(candidate)
    copy = df.copy()
    copy.columns = out
    return copy, duplicates


def _replace_missing(series: pd.Series, tokens: Sequence[str]) -> pd.Series:
    token_set = {str(x).strip().lower() for x in tokens}
    def conv(v):
        if v is None:
            return np.nan
        if isinstance(v, str) and v.strip().lower() in token_set:
            return np.nan
        return v
    return series.map(conv)


def _is_datetime_like(s: pd.Series) -> bool:
    non = s.dropna()
    if non.empty or pd.api.types.is_numeric_dtype(non):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(non, errors="coerce")
        return float(parsed.notna().mean()) >= 0.9
    except Exception:
        return False


def _infer_kind(s: pd.Series, n_rows: int, *, discrete_max_unique: int = 20) -> str:
    non = s.dropna()
    nunique = int(non.nunique(dropna=True))
    if nunique <= 1:
        return "constante"
    if _is_datetime_like(s):
        return "data_hora"
    if pd.api.types.is_bool_dtype(non):
        return "binaria"
    if pd.api.types.is_numeric_dtype(non):
        if nunique == 2:
            return "binaria"
        if nunique <= discrete_max_unique:
            return "numerica_discreta"
        ratio = nunique / max(1, len(non))
        if len(non) >= 20 and ratio >= 0.98:
            vals = pd.to_numeric(non, errors="coerce").dropna().to_numpy(dtype=float)
            # ID numérico: praticamente único E com padrão inteiro/sequencial.
            # Variáveis contínuas aleatórias também são únicas, portanto não basta cardinalidade.
            integerish = bool(len(vals) and np.allclose(vals, np.round(vals), atol=1e-9))
            if integerish and len(vals) > 2:
                sv = np.sort(np.unique(vals))
                diffs = np.diff(sv)
                regular = bool(len(diffs) and np.mean(np.isclose(diffs, np.median(diffs), atol=1e-9)) >= 0.95)
                if regular:
                    return "id_like"
        return "numerica_continua"
    if nunique == 2:
        return "binaria"
    ratio = nunique / max(1, len(non))
    avg_len = float(non.astype(str).str.len().mean()) if len(non) else 0.0
    if len(non) >= 20 and ratio >= 0.98:
        return "id_like"
    if avg_len > 40 and ratio > 0.5:
        return "texto_livre"
    return "categorica"


def _decision(kind: str, cardinality: int, n_rows: int) -> str:
    if kind in {"constante", "id_like", "texto_livre", "data_hora"}:
        return "descartar"
    if kind in {"categorica", "binaria"}:
        if cardinality > max(50, int(0.5 * max(1, n_rows))):
            return "frequency_encoding"
        return "categorica"
    return "usar"


@dataclass
class ColumnContract:
    name: str
    original_name: str
    index: int
    inferred_type: str
    missing_rate: float
    cardinality: int
    treatment: str
    override: Optional[str] = None


@dataclass
class DataContract:
    version: str
    target: Optional[str]
    target_index: Optional[int]
    target_confirmed: bool
    status: str
    n_rows: int
    n_columns: int
    columns: List[ColumnContract]
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    removed_target_missing_rows: List[int] = field(default_factory=list)
    class_counts: Dict[str, int] = field(default_factory=dict)
    missing_tokens: List[str] = field(default_factory=lambda: list(DEFAULT_MISSING_TOKENS))
    column_renames: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def feature_columns(self) -> List[str]:
        return [c.name for c in self.columns if c.name != self.target and c.treatment != "descartar"]


def _target_name(df: pd.DataFrame, target: Union[str, int, None]) -> tuple[Optional[str], Optional[int], bool]:
    if target is None:
        # Sugestão deliberadamente conservadora: nomes típicos, caso contrário última coluna.
        lower = {str(c).lower(): str(c) for c in df.columns}
        for key in ("target", "class", "classe", "label", "y"):
            if key in lower:
                name = lower[key]
                return name, int(df.columns.get_loc(name)), False
        if len(df.columns):
            return str(df.columns[-1]), len(df.columns)-1, False
        return None, None, False
    if isinstance(target, (int, np.integer)):
        idx = int(target)
        if idx < 0:
            idx += len(df.columns)
        if idx < 0 or idx >= len(df.columns):
            raise DataContractError(f"Índice da coluna-alvo fora do intervalo: {target}.")
        return str(df.columns[idx]), idx, True
    target = str(target)
    if target not in df.columns:
        raise DataContractError(f"Coluna-alvo '{target}' não existe no dataset.")
    return target, int(df.columns.get_loc(target)), True


def _majority_label(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=int)
    if values.size == 0:
        return 0
    counts = np.bincount(values)
    return int(np.argmax(counts))


def _fit_numeric_stump(x: np.ndarray, y: np.ndarray):
    """Stump univariado próprio para detecção de fuga, sem árvore CART.

    O contrato precisa apenas de detectar uma coluna quase equivalente ao alvo;
    não deve introduzir uma segunda família de árvores no runtime de produção.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    finite = np.isfinite(x)
    if finite.sum() < 4:
        majority = _majority_label(y)
        return (None, majority, majority)
    vals = np.unique(x[finite])
    if len(vals) <= 1:
        majority = _majority_label(y)
        return (None, majority, majority)
    # Mantém custo limitado mesmo em colunas contínuas grandes.
    if len(vals) > 64:
        qs = np.linspace(0.02, 0.98, 63)
        vals = np.unique(np.quantile(vals, qs))
    thresholds = (vals[:-1] + vals[1:]) / 2.0
    global_majority = _majority_label(y)
    best = (None, global_majority, global_majority)
    best_acc = -1.0
    for threshold in thresholds:
        left = x <= threshold
        right = ~left
        left_label = _majority_label(y[left]) if left.any() else global_majority
        right_label = _majority_label(y[right]) if right.any() else global_majority
        pred = np.where(left, left_label, right_label)
        acc = float(np.mean(pred == y))
        if acc > best_acc:
            best_acc = acc
            best = (float(threshold), int(left_label), int(right_label))
    return best


def _predict_numeric_stump(model, x: np.ndarray) -> np.ndarray:
    threshold, left_label, right_label = model
    if threshold is None:
        return np.full(len(x), left_label, dtype=int)
    return np.where(np.asarray(x, dtype=float) <= threshold, left_label, right_label).astype(int)


def _fit_categorical_stump(x: np.ndarray, y: np.ndarray):
    """Mapa categoria→classe maioritária, com fallback global."""
    x = np.asarray(x, dtype=object)
    y = np.asarray(y, dtype=int)
    fallback = _majority_label(y)
    mapping = {}
    for value in np.unique(x.astype(str)):
        mask = x.astype(str) == value
        mapping[str(value)] = _majority_label(y[mask]) if mask.any() else fallback
    return mapping, fallback


def _predict_categorical_stump(model, x: np.ndarray) -> np.ndarray:
    mapping, fallback = model
    return np.asarray([mapping.get(str(value), fallback) for value in np.asarray(x, dtype=object)], dtype=int)


def _leakage_score(feature: pd.Series, target: pd.Series) -> Optional[float]:
    """CV univariada para detectar fuga óbvia sem depender de CART/sklearn.tree."""
    mask = feature.notna() & target.notna()
    x = feature[mask]
    y = target[mask]
    if len(x) < 12 or y.nunique() < 2:
        return None
    try:
        numeric = pd.api.types.is_numeric_dtype(x)
        if numeric:
            values = pd.to_numeric(x, errors="coerce")
            median = values.median()
            X = values.fillna(median).to_numpy(dtype=float)
        else:
            X = x.astype(str).to_numpy(dtype=object)
        yy = pd.factorize(y.astype(str), sort=True)[0].astype(int)
        min_class = int(pd.Series(yy).value_counts().min())
        folds = min(5, min_class)
        if folds < 2:
            return None
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1701)
        scores = []
        for train_idx, val_idx in cv.split(np.zeros(len(yy)), yy):
            if numeric:
                model = _fit_numeric_stump(X[train_idx], yy[train_idx])
                pred = _predict_numeric_stump(model, X[val_idx])
            else:
                model = _fit_categorical_stump(X[train_idx], yy[train_idx])
                pred = _predict_categorical_stump(model, X[val_idx])
            scores.append(float(np.mean(pred == yy[val_idx])))
        return float(np.mean(scores)) if scores else None
    except (TypeError, ValueError, FloatingPointError):
        return None


def build_data_contract(
    df: pd.DataFrame,
    target: Union[str, int, None],
    user_overrides: Optional[Mapping[str, str]] = None,
    *,
    missing_tokens: Sequence[str] = DEFAULT_MISSING_TOKENS,
    high_missing_threshold: float = 0.40,
    imbalance_ratio_threshold: float = 3.0,
    small_n_threshold: int = 100,
    leakage_threshold: float = 0.98,
) -> DataContract:
    if not isinstance(df, pd.DataFrame):
        raise DataContractError("Os dados devem ser fornecidos como pandas.DataFrame.")
    if df.empty or len(df.columns) < 2:
        raise DataContractError("O dataset deve conter pelo menos duas colunas e uma linha.")

    original_names = list(df.columns)
    work, duplicates = _normalise_columns(df)
    rename_map = {str(o): str(n) for o, n in zip(original_names, work.columns) if str(o) != str(n)}
    # Target por índice mantém semântica mesmo quando nomes duplicados foram normalizados.
    if isinstance(target, str) and target not in work.columns and target in [str(x) for x in original_names]:
        positions = [i for i, x in enumerate(original_names) if str(x) == target]
        if len(positions) != 1:
            raise DataContractError(f"Nome de alvo ambíguo devido a colunas duplicadas: '{target}'. Use o índice.")
        target = positions[0]
    target_name, target_idx, confirmed = _target_name(work, target)
    overrides = {str(k): str(v) for k,v in (user_overrides or {}).items()}

    normalised = pd.DataFrame({c: _replace_missing(work[c], missing_tokens) for c in work.columns}, index=work.index)

    warnings_out: List[Dict[str, Any]] = []
    if duplicates:
        warnings_out.append({"code":"duplicate_column_names", "columns":sorted(set(duplicates)), "message":"Foram encontrados nomes de colunas duplicados; use índices ou os nomes normalizados."})
    if any(not str(c).strip() for c in original_names):
        warnings_out.append({"code":"empty_column_names", "message":"Foram encontrados nomes de colunas vazios e foram normalizados."})

    y = normalised[target_name]
    missing_target_idx = [int(i) for i in y.index[y.isna()].tolist()]
    if missing_target_idx:
        warnings_out.append({"code":"target_missing_rows_removed", "count":len(missing_target_idx), "message":"Linhas com alvo em falta devem ser removidas antes do treino."})
        valid = y.notna()
        normalised = normalised.loc[valid].copy()
        y = normalised[target_name]
    counts = y.astype(str).value_counts()
    if len(counts) < 2:
        raise DataContractError("A coluna-alvo contém apenas uma classe; classificação é inviável.")
    rare = counts[counts < 2]
    if not rare.empty:
        names = ", ".join(f"{k} ({int(v)})" for k,v in rare.items())
        raise DataContractError(f"Classes com menos de 2 amostras: {names}.")
    if pd.api.types.is_numeric_dtype(y):
        unique_ratio = y.nunique()/max(1,len(y))
        if y.nunique() > max(20, int(0.20*len(y))) and unique_ratio > 0.20:
            warnings_out.append({"code":"target_looks_continuous", "message":"O alvo tem muitos valores numéricos únicos e pode representar regressão."})

    cols: List[ColumnContract] = []
    for i,c in enumerate(normalised.columns):
        s = normalised[c]
        kind = _infer_kind(s, len(normalised))
        cardinality = int(s.nunique(dropna=True))
        treatment = "target" if c == target_name else _decision(kind, cardinality, len(normalised))
        override = overrides.get(str(c))
        if override:
            mapping = {"ignorar":"descartar", "categorica":"categorica", "categórica":"categorica", "numerica":"usar", "numérica":"usar"}
            treatment = mapping.get(override.lower(), treatment)
        cc = ColumnContract(str(c), str(original_names[i]), i, kind, float(s.isna().mean()), cardinality, treatment, override)
        cols.append(cc)
        if c != target_name:
            if kind == "constante": warnings_out.append({"code":"constant_column", "column":str(c), "message":f"A coluna '{c}' é constante e será descartada."})
            if kind == "id_like": warnings_out.append({"code":"id_like_column", "column":str(c), "message":f"A coluna '{c}' parece um identificador e será descartada por defeito."})
            if cc.missing_rate > high_missing_threshold: warnings_out.append({"code":"high_missing_rate", "column":str(c), "rate":cc.missing_rate, "message":f"A coluna '{c}' tem mais de {high_missing_threshold:.0%} de valores em falta."})

    ratio = float(counts.max()/counts.min())
    if ratio >= imbalance_ratio_threshold:
        warnings_out.append({"code":"class_imbalance", "ratio":ratio, "class_counts":{str(k):int(v) for k,v in counts.items()}, "message":f"As classes estão desbalanceadas (maioria/minoria = {ratio:.2f})."})
    if len(normalised) < small_n_threshold:
        warnings_out.append({"code":"small_sample", "n_rows":len(normalised), "message":f"Dataset pequeno: {len(normalised)} linhas."})
    n_features = len(normalised.columns)-1
    if n_features > len(normalised):
        warnings_out.append({"code":"wide_dataset", "n_rows":len(normalised), "n_features":n_features, "message":"Há mais atributos do que linhas."})
    duplicate_rows = int(normalised.duplicated().sum())
    if duplicate_rows:
        warnings_out.append({"code":"duplicate_rows", "count":duplicate_rows, "message":f"Foram encontradas {duplicate_rows} linhas exactamente duplicadas."})

    # Fuga óbvia e stump-CV. Evita custo excessivo em cardinalidade enorme já descartada.
    ystr = y.astype(str).reset_index(drop=True)
    for cc in cols:
        if cc.name == target_name or cc.inferred_type in {"constante","texto_livre","data_hora"}:
            continue
        feat = normalised[cc.name].reset_index(drop=True)
        same = feat.astype(str).equals(ystr)
        if same:
            score = 1.0
        else:
            score = _leakage_score(feat, ystr)
        if score is not None and score > leakage_threshold:
            warnings_out.append({"code":"possible_target_leakage", "column":cc.name, "cv_accuracy":score, "message":f"A coluna '{cc.name}' prevê o alvo quase perfeitamente sozinha (CV={score:.3f}); verificar possível fuga de alvo."})

    status = "pronto" if confirmed else "alvo_nao_confirmado"
    return DataContract(
        version="1.0", target=target_name, target_index=target_idx, target_confirmed=confirmed,
        status=status, n_rows=int(len(normalised)), n_columns=int(len(normalised.columns)),
        columns=cols, warnings=warnings_out, removed_target_missing_rows=missing_target_idx,
        class_counts={str(k):int(v) for k,v in counts.items()}, missing_tokens=list(missing_tokens),
        column_renames=rename_map,
    )


__all__ = ["ColumnContract", "DataContract", "build_data_contract", "DEFAULT_MISSING_TOKENS"]
