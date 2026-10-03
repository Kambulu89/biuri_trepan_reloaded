"""Schema de features reversível: espaço do modelo <-> espaço humano (raw).

Responsabilidades:
* tipos (contínua, inteira, binária, one-hot, código categórico) e domínios válidos;
* ranges com prioridade OWL > config/metadata > distribuição de TREINO > fallback;
* projecção para um vector válido (inteiros, binárias, one-hot exatamente um activo, direccionalidade);
* decode/encode: ``x_17 = 1`` volta a ser ``employment = private``;
* violações *hard* auditáveis.

Nada aqui usa dados de teste: o schema só vê ``X_train``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9
KINDS = ("continuous", "integer", "binary", "onehot", "code")
DIRECTIONS = ("both", "increase_only", "decrease_only", "immutable")
RANGE_PRIORITY = ("owl", "config", "metadata", "train", "fallback")


@dataclass
class FeatureSpec:
    """Uma coluna do espaço do modelo."""

    name: str
    kind: str = "continuous"
    low: float = -np.inf
    high: float = np.inf
    range_source: str = "fallback"
    direction: str = "both"
    actionable: bool = True
    immutable: bool = False
    cost: float = 1.0                       # neutro por defeito; só vem de config/domínio
    group: Optional[str] = None             # nome humano (one-hot)
    category: Any = None                    # valor representado por esta coluna one-hot
    allowed_values: Optional[Tuple[float, ...]] = None   # kind == "code"
    labels: Optional[Dict[float, str]] = None            # code -> rótulo humano
    reversible: bool = True
    derived: bool = False                   # calculada a partir de outras (ex.: onto_*): nunca alterada directamente
    notes: List[str] = field(default_factory=list)

    @property
    def fixed(self) -> bool:
        """Não pesquisável: imutável, direcção imutável, derivada, não reversível ou range degenerado."""
        return bool(self.immutable or self.direction == "immutable" or self.derived or not self.reversible or self.high - self.low <= EPS)


@dataclass
class FeatureUnit:
    """Feature ao nível humano (um grupo one-hot conta como UMA feature)."""

    name: str
    kind: str
    indices: List[int]
    categories: List[Any] = field(default_factory=list)


def _first_present(mapping: Mapping[str, Any], *keys: str, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


class FeatureSchema:
    def __init__(self, specs: Sequence[FeatureSpec], *, provenance: Optional[Dict[str, Any]] = None):
        self.specs: List[FeatureSpec] = list(specs)
        self.names: List[str] = [s.name for s in self.specs]
        if len(set(self.names)) != len(self.names):
            raise ValueError("Nomes de colunas duplicados no schema.")
        self.index: Dict[str, int] = {n: i for i, n in enumerate(self.names)}
        self.provenance: Dict[str, Any] = dict(provenance or {})
        self.units: List[FeatureUnit] = self._build_units()
        self.unit_of_column: Dict[int, int] = {i: u for u, unit in enumerate(self.units) for i in unit.indices}

    # ------------------------------------------------------------------ construção
    def _build_units(self) -> List[FeatureUnit]:
        units: List[FeatureUnit] = []
        seen_groups: Dict[str, FeatureUnit] = {}
        for i, s in enumerate(self.specs):
            if s.kind == "onehot" and s.group:
                unit = seen_groups.get(s.group)
                if unit is None:
                    unit = FeatureUnit(s.group, "onehot", [], [])
                    seen_groups[s.group] = unit
                    units.append(unit)
                unit.indices.append(i)
                unit.categories.append(s.category)
            else:
                units.append(FeatureUnit(s.name, s.kind, [i]))
        return units

    @classmethod
    def from_data(cls, X_train, feature_names: Sequence[str], config: Optional[Mapping[str, Any]] = None) -> "FeatureSchema":
        """Infere o schema a partir do TREINO e de metadados/config opcionais.

        ``config`` aceita (todas opcionais): ``features`` {nome: {kind,min,max,direction,actionable,immutable,cost,labels}},
        ``feature_ranges``, ``immutable_features``, ``protected_features``, ``actionable_features``, ``binary_features``,
        ``integer_features``, ``categorical_values`` (códigos), ``categorical_groups`` {grupo: {columns:[...], labels:[...]}},
        ``direction`` {nome: increase_only|decrease_only|both|immutable}, ``costs`` {nome: custo}.
        """
        cfg = dict(config or {})
        X = np.asarray(X_train, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(feature_names):
            raise ValueError("X_train e feature_names são incompatíveis.")
        names = [str(n) for n in feature_names]
        per_feature = {str(k): dict(v) for k, v in (cfg.get("features") or {}).items()}
        ranges_cfg = {str(k): tuple(v) for k, v in (cfg.get("feature_ranges") or {}).items()}
        immutable = set(map(str, cfg.get("immutable_features") or [])) | set(map(str, cfg.get("protected_features") or []))
        actionable = cfg.get("actionable_features")
        actionable = None if actionable is None else set(map(str, actionable))
        binary_cfg = set(map(str, cfg.get("binary_features") or []))
        integer_cfg = set(map(str, cfg.get("integer_features") or []))
        code_cfg = {str(k): tuple(float(x) for x in v) for k, v in (cfg.get("categorical_values") or {}).items()}
        directions = {str(k): str(v) for k, v in (cfg.get("direction") or {}).items()}
        costs = {str(k): float(v) for k, v in (cfg.get("costs") or {}).items()}
        derived_cfg = set(map(str, cfg.get("derived_features") or []))
        groups_cfg = dict(cfg.get("categorical_groups") or {})
        if cfg.get("auto_detect_onehot"):
            for gname, spec in cls.detect_onehot_groups(X, names, exclude=set().union(*[set(g["columns"] if isinstance(g, Mapping) else g) for g in groups_cfg.values()] or [set()])).items():
                groups_cfg.setdefault(gname, spec)
        group_member: Dict[str, Tuple[str, Any]] = {}
        for gname, spec in groups_cfg.items():
            cols = list(spec["columns"] if isinstance(spec, Mapping) else spec)
            labels = list(spec.get("labels") if isinstance(spec, Mapping) and spec.get("labels") else cols)
            if len(labels) != len(cols):
                raise ValueError(f"Grupo '{gname}': labels e columns têm comprimentos diferentes.")
            for col, lab in zip(cols, labels):
                if str(col) not in names:
                    raise ValueError(f"Grupo '{gname}' refere coluna inexistente '{col}'.")
                group_member[str(col)] = (str(gname), lab)

        specs: List[FeatureSpec] = []
        notes_global: List[str] = []
        for j, name in enumerate(names):
            col = X[:, j]
            finite = col[np.isfinite(col)]
            f = per_feature.get(name, {})
            kind = str(f.get("kind") or "")
            if name in group_member:
                kind = "onehot"
            elif not kind:
                if name in binary_cfg:
                    kind = "binary"
                elif name in code_cfg:
                    kind = "code"
                elif name in integer_cfg:
                    kind = "integer"
                elif finite.size and set(np.round(np.unique(finite), 12)).issubset({0.0, 1.0}):
                    kind = "binary"
                elif finite.size and np.all(np.abs(finite - np.round(finite)) <= 1e-9) and np.unique(finite).size > 2:
                    kind = "integer"
                else:
                    kind = "continuous"
            if kind not in KINDS:
                raise ValueError(f"Tipo de feature desconhecido para '{name}': {kind!r}")
            # --- range: config > metadata > treino > fallback
            source = "train"
            if "min" in f or "max" in f:
                low = float(f.get("min", finite.min() if finite.size else -np.inf))
                high = float(f.get("max", finite.max() if finite.size else np.inf))
                source = "config"
            elif name in ranges_cfg:
                low, high = float(ranges_cfg[name][0]), float(ranges_cfg[name][1])
                source = "config"
            elif finite.size:
                low, high = float(finite.min()), float(finite.max())
            else:
                low, high, source = 0.0, 0.0, "fallback"
            if high < low:
                low, high = high, low
            if kind in ("binary", "onehot"):
                low, high = 0.0, 1.0
            spec = FeatureSpec(name=name, kind=kind, low=low, high=high, range_source=source)
            if kind == "code":
                allowed = code_cfg.get(name) or tuple(float(v) for v in np.unique(finite))
                spec.allowed_values = tuple(sorted(allowed))
                spec.low, spec.high = min(spec.allowed_values), max(spec.allowed_values)
                if f.get("labels"):
                    spec.labels = {float(k): str(v) for k, v in dict(f["labels"]).items()}
            if kind == "onehot":
                spec.group, spec.category = group_member[name]
            spec.immutable = bool(f.get("immutable", name in immutable))
            spec.derived = name in derived_cfg
            spec.actionable = bool(f.get("actionable", True if actionable is None else name in actionable))
            spec.direction = str(f.get("direction") or directions.get(name) or "both")
            if spec.direction not in DIRECTIONS:
                raise ValueError(f"Direção inválida para '{name}': {spec.direction!r}")
            if spec.direction == "immutable":
                spec.immutable = True
            spec.cost = float(f.get("cost", costs.get(name, 1.0)))
            if spec.cost <= 0 or not np.isfinite(spec.cost):
                raise ValueError(f"Custo inválido para '{name}': {spec.cost!r} (deve ser > 0).")
            if source == "train" and spec.high - spec.low <= EPS:
                spec.notes.append("constante no treino")
            specs.append(spec)
        schema = cls(specs, provenance={"fit_rows": int(len(X)), "range_priority": list(RANGE_PRIORITY), "notes": notes_global})
        schema._check_groups(X)
        return schema

    @staticmethod
    def detect_onehot_groups(X, names: Sequence[str], exclude: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
        """Deteta grupos one-hot SÓ com prova no treino: colunas 0/1 com prefixo comum cujas linhas somam sempre 1."""
        X = np.asarray(X, dtype=float)
        exclude = exclude or set()
        candidates: Dict[str, List[int]] = {}
        for j, name in enumerate(names):
            if name in exclude:
                continue
            col = X[:, j]
            if not set(np.round(np.unique(col[np.isfinite(col)]), 12)).issubset({0.0, 1.0}):
                continue
            for sep in ("=", "_"):
                if sep in name:
                    prefix = name.rsplit(sep, 1)[0]
                    candidates.setdefault(prefix, []).append(j)
                    break
        out: Dict[str, Dict[str, Any]] = {}
        for prefix, cols in candidates.items():
            if len(cols) < 2:
                continue
            if np.all(np.abs(X[:, cols].sum(axis=1) - 1.0) <= 1e-9):
                labels = [names[j][len(prefix) + 1:] for j in cols]
                out[prefix] = {"columns": [names[j] for j in cols], "labels": labels}
        return out

    def _check_groups(self, X: np.ndarray) -> None:
        for unit in self.units:
            if unit.kind != "onehot":
                continue
            sums = X[:, unit.indices].sum(axis=1)
            frac_ok = float(np.mean(np.abs(sums - 1.0) <= 1e-9)) if len(X) else 1.0
            if frac_ok < 1.0:
                self.provenance.setdefault("notes", []).append(
                    f"grupo one-hot '{unit.name}': só {frac_ok:.1%} das linhas de treino têm exactamente uma categoria activa")

    @classmethod
    def from_preprocessor(cls, preprocessor, X_train_encoded, config: Optional[Mapping[str, Any]] = None) -> "FeatureSchema":
        """Schema reversível a partir de um ``core.preprocessing.DataPreprocessor`` ajustado.

        * colunas one-hot -> grupos (uma feature humana = coluna original; categorias reconstruíveis);
        * colunas numéricas -> tipos inferidos do treino codificado; ``scale_numeric`` é reportado em ``notes``;
        * ``frequency_encoding``, indicadores de falta e categorias raras/infrequentes NÃO são reversíveis:
          ficam ``reversible=False`` e fixas (nunca alteradas pelo motor)."""
        names = [str(n) for n in preprocessor.get_feature_names_out()]
        cfg = dict(config or {})
        groups: Dict[str, Dict[str, Any]] = {}
        non_rev: Dict[str, str] = {}
        tf = getattr(preprocessor, "transformer_", None)
        for entry in (getattr(tf, "transformers_", None) or []):
            tname, trans, cols = entry[0], entry[1], list(entry[2]) if not isinstance(entry[2], str) else [entry[2]]
            if tname == "cat":
                oh = trans.named_steps["onehot"]
                for col, cats in zip(cols, oh.categories_):
                    members, labels = [], []
                    for cat in cats:
                        cand = f"{col}_{cat}"
                        if cand in names:
                            members.append(cand)
                            labels.append(str(cat))
                    infreq = [n for n in names if n.startswith(f"{col}_infrequent")]
                    for n in infreq:
                        non_rev[n] = f"categoria rara agregada de '{col}' (não reversível)"
                    if members:
                        groups[str(col)] = {"columns": members, "labels": labels}
            elif tname == "freq":
                for col in cols:
                    for n in names:
                        if n.startswith(f"{col}__frequencia") or n == col:
                            non_rev[n] = f"frequency encoding de '{col}' (não reversível)"
            elif tname == "num":
                for n in names:
                    if n.startswith("missingindicator_"):
                        non_rev[n] = "indicador de valor em falta (derivado)"
        cfg.setdefault("categorical_groups", {}).update(groups)
        schema = cls.from_data(X_train_encoded, names, cfg)
        for n, why in non_rev.items():
            if n in schema.index:
                spec = schema.specs[schema.index[n]]
                spec.reversible = False
                spec.notes.append(why)
        if getattr(preprocessor, "scale_numeric", False):
            schema.provenance.setdefault("notes", []).append("scale_numeric=True: valores no espaço padronizado do modelo")
        schema.provenance["origin"] = "DataPreprocessor"
        return schema

    # ------------------------------------------------------------------ queries
    @property
    def n_features(self) -> int:
        return len(self.specs)

    def derived_names(self) -> List[str]:
        return [s.name for s in self.specs if s.derived]

    def searchable_columns(self, nonactionable_policy: str = "invalid") -> List[int]:
        out = []
        for i, s in enumerate(self.specs):
            if s.fixed:
                continue
            if not s.actionable and nonactionable_policy == "invalid":
                continue
            out.append(i)
        return out

    def searchable_units(self, nonactionable_policy: str = "invalid") -> List[int]:
        cols = set(self.searchable_columns(nonactionable_policy))
        return [u for u, unit in enumerate(self.units) if cols.intersection(unit.indices)]

    def set_range(self, name: str, low: Optional[float], high: Optional[float], source: str) -> None:
        """Aplica um range respeitando a prioridade das fontes (a fonte mais forte nunca é sobrescrita)."""
        spec = self.specs[self.index[name]]
        if RANGE_PRIORITY.index(source) > RANGE_PRIORITY.index(spec.range_source):
            return
        if spec.kind in ("binary", "onehot"):
            return
        if low is not None:
            spec.low = float(low)
        if high is not None:
            spec.high = float(high)
        spec.range_source = source

    # ------------------------------------------------------------------ projecção / validação
    def _active_category(self, vec: np.ndarray, unit: FeatureUnit, fallback: Optional[np.ndarray] = None) -> int:
        vals = vec[unit.indices]
        pos = int(np.argmax(vals))
        if fallback is not None and np.isclose(vals.max(), vals, atol=1e-12).sum() > 1:
            fb = int(np.argmax(fallback[unit.indices]))
            if np.isclose(vals[fb], vals.max(), atol=1e-12):
                pos = fb
        return pos

    def project(self, original, candidate, *, nonactionable_policy: str = "invalid") -> np.ndarray:
        """Repara ``candidate`` para um vector válido: respeita fixas, direção, domínio, inteiros, binárias e one-hot."""
        orig = np.asarray(original, dtype=float).reshape(-1)
        cand = np.asarray(candidate, dtype=float).reshape(-1).copy()
        if len(cand) != self.n_features or len(orig) != self.n_features:
            raise ValueError(f"Vector com {len(cand)} valores; schema tem {self.n_features} colunas.")
        cand = np.where(np.isfinite(cand), cand, orig)
        for i, s in enumerate(self.specs):
            if s.fixed or (not s.actionable and nonactionable_policy == "invalid") or s.kind == "onehot":
                if s.kind != "onehot":
                    cand[i] = orig[i]
                continue
            v = float(np.clip(cand[i], s.low, s.high))
            if s.direction == "increase_only":
                v = max(v, orig[i])
            elif s.direction == "decrease_only":
                v = min(v, orig[i])
            if s.kind == "binary":
                v = float(v >= 0.5)
            elif s.kind == "integer":
                if s.direction == "increase_only":
                    v = float(np.ceil(v - 1e-9))
                elif s.direction == "decrease_only":
                    v = float(np.floor(v + 1e-9))
                else:
                    v = float(np.rint(v))
                v = float(np.clip(v, np.ceil(s.low - 1e-9), np.floor(s.high + 1e-9)))
            elif s.kind == "code":
                allowed = s.allowed_values or ()
                if allowed:
                    v = float(min(allowed, key=lambda a: abs(a - v)))
            cand[i] = v
        for unit in self.units:
            if unit.kind != "onehot":
                continue
            specs = [self.specs[i] for i in unit.indices]
            if any(s.fixed or (not s.actionable and nonactionable_policy == "invalid") for s in specs):
                cand[unit.indices] = orig[unit.indices]
                continue
            pos = self._active_category(cand, unit, fallback=orig)
            block = np.zeros(len(unit.indices))
            block[pos] = 1.0
            cand[unit.indices] = block
        return cand

    def changed_columns(self, original, candidate) -> np.ndarray:
        o = np.asarray(original, dtype=float).reshape(-1)
        c = np.asarray(candidate, dtype=float).reshape(-1)
        scale = np.array([max(abs(s.high - s.low), 1.0) if np.isfinite(s.high - s.low) else 1.0 for s in self.specs])
        return np.abs(c - o) > EPS * scale

    def changed_units(self, original, candidate, include_derived: bool = False) -> List[int]:
        """Unidades humanas alteradas. Colunas derivadas (consequência de outras) não contam como acção do utilizador."""
        mask = self.changed_columns(original, candidate)
        return [u for u, unit in enumerate(self.units)
                if mask[unit.indices].any() and (include_derived or not all(self.specs[i].derived for i in unit.indices))]

    def violations(self, original, candidate, *, nonactionable_policy: str = "invalid") -> List[Dict[str, str]]:
        """Violações HARD do domínio (cada uma com código e mensagem)."""
        o = np.asarray(original, dtype=float).reshape(-1)
        c = np.asarray(candidate, dtype=float).reshape(-1)
        out: List[Dict[str, str]] = []
        if len(c) != self.n_features:
            return [{"code": "SHAPE", "feature": "*", "message": f"vector com {len(c)} valores; esperados {self.n_features}"}]
        changed = self.changed_columns(o, c)
        for i, s in enumerate(self.specs):
            v = c[i]
            if not np.isfinite(v):
                out.append({"code": "NONFINITE", "feature": s.name, "message": f"{s.name}: valor não finito"})
                continue
            if s.derived:
                continue          # consistência das derivadas é verificada pelo ConstraintSet (recomputação)
            if v < s.low - EPS or v > s.high + EPS:
                out.append({"code": "RANGE", "feature": s.name, "message": f"{s.name}: {v:g} fora de [{s.low:g}, {s.high:g}] (fonte {s.range_source})"})
            if s.kind == "binary" and not (abs(v) <= EPS or abs(v - 1) <= EPS):
                out.append({"code": "BINARY", "feature": s.name, "message": f"{s.name}: valor binário inválido {v:g}"})
            if s.kind == "integer" and abs(v - round(v)) > 1e-6:
                out.append({"code": "INTEGER", "feature": s.name, "message": f"{s.name}: {v:g} não é inteiro"})
            if s.kind == "code" and s.allowed_values and not any(abs(v - a) <= 1e-6 for a in s.allowed_values):
                out.append({"code": "CATEGORY", "feature": s.name, "message": f"{s.name}: categoria {v:g} fora do domínio {s.allowed_values}"})
            if changed[i]:
                if s.immutable or s.direction == "immutable":
                    out.append({"code": "IMMUTABLE", "feature": s.name, "message": f"{s.name}: feature imutável alterada"})
                elif not s.reversible:
                    out.append({"code": "NOT_REVERSIBLE", "feature": s.name, "message": f"{s.name}: feature não reversível alterada"})
                elif s.direction == "increase_only" and v < o[i] - EPS:
                    out.append({"code": "DIRECTION", "feature": s.name, "message": f"{s.name}: só pode aumentar ({o[i]:g} -> {v:g})"})
                elif s.direction == "decrease_only" and v > o[i] + EPS:
                    out.append({"code": "DIRECTION", "feature": s.name, "message": f"{s.name}: só pode diminuir ({o[i]:g} -> {v:g})"})
                if not s.actionable and nonactionable_policy == "invalid":
                    out.append({"code": "NOT_ACTIONABLE", "feature": s.name, "message": f"{s.name}: feature não accionável alterada"})
        for unit in self.units:
            if unit.kind == "onehot":
                block = c[unit.indices]
                ok = np.all((np.abs(block) <= EPS) | (np.abs(block - 1) <= EPS)) and abs(block.sum() - 1) <= 1e-6
                if not ok:
                    out.append({"code": "ONEHOT", "feature": unit.name,
                                "message": f"{unit.name}: combinação one-hot inválida {block.tolist()} (deve haver exatamente uma categoria)"})
        return out

    def soft_warnings(self, original, candidate, nonactionable_policy: str = "invalid") -> List[str]:
        if nonactionable_policy == "invalid":
            return []
        changed = self.changed_columns(original, candidate)
        return [f"{s.name}: feature não accionável alterada (aviso)" for i, s in enumerate(self.specs) if changed[i] and not s.actionable]

    # ------------------------------------------------------------------ espaço humano
    def _human_value(self, vec: np.ndarray, unit: FeatureUnit):
        if unit.kind == "onehot":
            block = vec[unit.indices]
            if abs(block.sum() - 1) > 1e-6 or not np.all((np.abs(block) <= 1e-6) | (np.abs(block - 1) <= 1e-6)):
                return "<inválido>"
            return unit.categories[int(np.argmax(block))]
        s = self.specs[unit.indices[0]]
        v = float(vec[unit.indices[0]])
        if s.kind == "integer" or s.kind == "binary":
            return int(round(v))
        if s.kind == "code":
            return (s.labels or {}).get(v, v)
        return v

    def decode(self, vector) -> Dict[str, Any]:
        """Vector do modelo -> {feature humana: valor legível} (one-hot reconstruído)."""
        vec = np.asarray(vector, dtype=float).reshape(-1)
        return {unit.name: self._human_value(vec, unit) for unit in self.units}

    def encode(self, human: Mapping[str, Any], base: Optional[Any] = None) -> np.ndarray:
        """Inverso de ``decode``: {feature humana: valor} -> vector do modelo (``base`` fornece valores omitidos)."""
        vec = np.zeros(self.n_features) if base is None else np.asarray(base, dtype=float).reshape(-1).copy()
        for unit in self.units:
            if unit.name not in human:
                continue
            value = human[unit.name]
            if unit.kind == "onehot":
                if value not in unit.categories:
                    raise ValueError(f"Categoria {value!r} inexistente em '{unit.name}'. Válidas: {unit.categories}")
                block = np.zeros(len(unit.indices))
                block[unit.categories.index(value)] = 1.0
                vec[unit.indices] = block
            else:
                s = self.specs[unit.indices[0]]
                if s.kind == "code" and s.labels and value in s.labels.values():
                    value = [k for k, lab in s.labels.items() if lab == value][0]
                vec[unit.indices[0]] = float(value)
        return vec

    def describe_change(self, original, candidate) -> List[Dict[str, Any]]:
        """Alterações ao nível humano (um grupo one-hot = uma alteração: ``X -> Y``)."""
        o = np.asarray(original, dtype=float).reshape(-1)
        c = np.asarray(candidate, dtype=float).reshape(-1)
        changes = []
        for u in self.changed_units(o, c):
            unit = self.units[u]
            before, after = self._human_value(o, unit), self._human_value(c, unit)
            numeric = unit.kind in ("continuous", "integer") or (unit.kind == "binary")
            delta = float(c[unit.indices[0]] - o[unit.indices[0]]) if numeric else None
            specs = [self.specs[i] for i in unit.indices]
            changes.append({
                "feature": unit.name, "kind": unit.kind, "columns": [self.names[i] for i in unit.indices],
                "original": before, "counterfactual": after, "delta": delta,
                "actionable": all(s.actionable for s in specs), "direction": specs[0].direction,
                "cost": float(np.mean([s.cost for s in specs])),
            })
        return changes

    def to_dict(self) -> Dict[str, Any]:
        return {"columns": [{"name": s.name, "kind": s.kind, "low": s.low, "high": s.high, "range_source": s.range_source,
                             "direction": s.direction, "actionable": s.actionable, "immutable": s.immutable, "cost": s.cost,
                             "group": s.group, "category": s.category, "reversible": s.reversible} for s in self.specs],
                "provenance": self.provenance}
