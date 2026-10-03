"""Constraints semânticas/de domínio: regras hard vs soft e extracção conservadora da OWL.

IMPORTANTE (Parte 55): relações OWL **não são causalidade**. As regras aqui são restrições de
consistência de domínio (co-alteração, incompatibilidade, monotonia declarada). O motor nunca afirma
que uma alteração "causa" outra, e sem SCM não existe "contrafactual causal".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from counterfactuals.cfkit.schema import EPS, FeatureSchema

HARD, SOFT = "hard", "soft"


def _norm(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


@dataclass
class Rule:
    """Regra declarativa. ``check`` devolve uma mensagem se violada, ``None`` caso contrário."""

    id: str
    kind: str
    severity: str = HARD            # hard: inviabiliza o CF; soft: penaliza
    source: str = "config"          # config | ontology | metadata
    weight: float = 1.0             # penalização quando soft
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    fn: Optional[Callable[..., Optional[str]]] = None

    def check(self, original: np.ndarray, candidate: np.ndarray, schema: FeatureSchema) -> Optional[str]:
        if self.kind == "requires_co_change":
            a, b = self.params["feature"], self.params["depends_on"]
            ua, ub = schema.unit_of_column.get(schema.index.get(a, -1)), schema.unit_of_column.get(schema.index.get(b, -1))
            if ua is None or ub is None:
                return None
            changed = set(schema.changed_units(original, candidate))
            if ua in changed and ub not in changed:
                return f"{a} depende de {b}: {a} foi alterada mas {b} não"
            return None
        if self.kind == "incompatible":
            hits = []
            for feat, op, value in self.params["conditions"]:
                idx = schema.index.get(feat)
                if idx is None:
                    return None
                v = candidate[idx]
                ok = {"==": abs(v - value) <= 1e-9, "!=": abs(v - value) > 1e-9, ">": v > value, ">=": v >= value,
                      "<": v < value, "<=": v <= value}[op]
                hits.append(ok)
            if hits and all(hits):
                text = " e ".join(f"{f} {op} {val:g}" for f, op, val in self.params["conditions"])
                return f"combinação incompatível: {text}"
            return None
        if self.kind == "monotone":
            s, t = self.params["source"], self.params["target"]
            if s not in schema.index or t not in schema.index:
                return None
            ds, dt = candidate[schema.index[s]] - original[schema.index[s]], candidate[schema.index[t]] - original[schema.index[t]]
            tol = float(self.params.get("tolerance", EPS))
            mode = self.params.get("direction", "same")
            if abs(ds) <= tol:
                return None
            if mode == "same" and ds * dt < -tol:
                return f"{s} e {t} devem variar no mesmo sentido"
            if mode == "opposite" and ds * dt > tol:
                return f"{s} e {t} devem variar em sentidos opostos"
            if mode == "nondecreasing" and dt < -tol:
                return f"{t} não pode diminuir quando {s} muda"
            return None
        if self.kind == "callable" and self.fn is not None:
            try:
                return self.fn(original, candidate, schema)
            except Exception as exc:  # regra defeituosa nunca derruba o motor
                return f"regra '{self.id}' falhou: {exc}"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "severity": self.severity, "source": self.source, "weight": self.weight,
                "description": self.description, "params": {k: v for k, v in self.params.items()}}


def requires_co_change(feature: str, depends_on: str, *, severity: str = HARD, source: str = "config", weight: float = 1.0) -> Rule:
    return Rule(f"dep:{feature}->{depends_on}", "requires_co_change", severity, source, weight,
                f"{feature} dependsOn {depends_on}: alterar {feature} exige alterar {depends_on} (consistência, não causalidade)",
                {"feature": feature, "depends_on": depends_on})


def incompatible(conditions: Sequence[Tuple[str, str, float]], *, severity: str = HARD, source: str = "config", weight: float = 1.0) -> Rule:
    text = " & ".join(f"{f}{op}{v:g}" for f, op, v in conditions)
    return Rule(f"incompat:{text}", "incompatible", severity, source, weight, f"incompatível: {text}", {"conditions": list(conditions)})


def monotone(source: str, target: str, direction: str = "same", *, severity: str = HARD, tolerance: float = EPS, src: str = "config", weight: float = 1.0) -> Rule:
    return Rule(f"mono:{source}->{target}:{direction}", "monotone", severity, src, weight,
                f"restrição de monotonia declarada ({direction}); NÃO é uma relação causal",
                {"source": source, "target": target, "direction": direction, "tolerance": tolerance})


def custom_rule(rule_id: str, fn: Callable[..., Optional[str]], *, severity: str = HARD, source: str = "config", weight: float = 1.0, description: str = "") -> Rule:
    return Rule(rule_id, "callable", severity, source, weight, description or rule_id, {}, fn)


@dataclass
class ConstraintSet:
    """Schema + regras (hard/soft). Fonte única de validade semântica."""

    schema: FeatureSchema
    rules: List[Rule] = field(default_factory=list)
    nonactionable_policy: str = "invalid"      # invalid (hard) | warn (soft)
    semantic_source: str = "none"              # none | ontology | config
    ontology_notes: List[str] = field(default_factory=list)
    not_extracted: List[str] = field(default_factory=list)
    derive_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None    # original -> colunas derivadas (matriz -> matriz)

    @property
    def semantic_available(self) -> bool:
        """Há alguma constraint semântica (OWL/config) além do domínio básico?"""
        return bool(self.rules) or self.semantic_source in ("ontology",)

    def derived_columns(self) -> List[int]:
        return [i for i, s in enumerate(self.schema.specs) if s.derived]

    def repair(self, original, candidate) -> np.ndarray:
        out = self.schema.project(original, candidate, nonactionable_policy=self.nonactionable_policy)
        cols = self.derived_columns()
        if cols:
            if self.derive_fn is not None:
                out[cols] = np.asarray(self.derive_fn(out.reshape(1, -1)), dtype=float).reshape(-1)[cols]
            else:
                out[cols] = np.asarray(original, dtype=float).reshape(-1)[cols]     # sem função de derivação: congeladas
        return out

    def hard_violations(self, original, candidate) -> List[Dict[str, str]]:
        out = list(self.schema.violations(original, candidate, nonactionable_policy=self.nonactionable_policy))
        o, c = np.asarray(original, float).reshape(-1), np.asarray(candidate, float).reshape(-1)
        cols = self.derived_columns()
        if cols and len(c) == self.schema.n_features and self.derive_fn is not None:
            expected = np.asarray(self.derive_fn(c.reshape(1, -1)), dtype=float).reshape(-1)
            if not np.allclose(c[cols], expected[cols], rtol=1e-6, atol=1e-6):
                out.append({"code": "DERIVED_INCONSISTENT", "feature": "*", "message": "features derivadas inconsistentes com as originais"})
        if len(c) == self.schema.n_features:
            for rule in self.rules:
                if rule.severity == HARD:
                    msg = rule.check(o, c, self.schema)
                    if msg:
                        out.append({"code": f"RULE:{rule.id}", "feature": "*", "message": msg, "source": rule.source})
        return out

    def soft_violations(self, original, candidate) -> Tuple[List[str], float]:
        o, c = np.asarray(original, float).reshape(-1), np.asarray(candidate, float).reshape(-1)
        msgs = list(self.schema.soft_warnings(o, c, self.nonactionable_policy))
        penalty = 0.5 * len(msgs)
        for rule in self.rules:
            if rule.severity == SOFT:
                msg = rule.check(o, c, self.schema)
                if msg:
                    msgs.append(msg)
                    penalty += float(rule.weight)
        return msgs, float(penalty)

    def semantic_validation(self, original, candidate) -> Dict[str, Any]:
        """Separa validade semântica da validade no modelo (Parte 31)."""
        if not self.semantic_available:
            return {"status": "NOT_AVAILABLE", "valid": None, "hard_violations": [], "soft_warnings": [], "soft_penalty": 0.0,
                    "note": "sem ontologia/regras semânticas: apenas o domínio básico foi verificado"}
        hard = [v for v in self.hard_violations(original, candidate) if v["code"].startswith("RULE:") or v["code"] in
                ("ONEHOT", "CATEGORY", "RANGE", "INTEGER", "BINARY")]
        soft, pen = self.soft_violations(original, candidate)
        status = "INVALID" if hard else ("VALID_WITH_WARNINGS" if soft else "VALID")
        return {"status": status, "valid": not hard, "hard_violations": [v["message"] for v in hard], "soft_warnings": soft, "soft_penalty": pen}

    def feasible(self) -> Tuple[bool, str]:
        cols = self.schema.searchable_columns(self.nonactionable_policy)
        if not cols:
            return False, "nenhuma feature mutável/accionável: todas são imutáveis, não accionáveis ou constantes"
        return True, ""

    def provenance(self) -> Dict[str, Any]:
        return {"schema": self.schema.provenance, "columns": self.schema.to_dict()["columns"], "nonactionable_policy": self.nonactionable_policy,
                "semantic_source": self.semantic_source, "rules": [r.to_dict() for r in self.rules],
                "range_sources": {s.name: s.range_source for s in self.schema.specs},
                "ontology_notes": list(self.ontology_notes), "not_extracted": list(self.not_extracted), "derive_fn": bool(self.derive_fn)}

    @classmethod
    def from_config(cls, X_train, feature_names: Sequence[str], config: Optional[Mapping[str, Any]] = None,
                    ontology_constraints: Optional["ExtractedConstraints"] = None) -> "ConstraintSet":
        """Constrói a partir do TREINO + config; a OWL (se existir) tem a prioridade mais alta nos ranges."""
        cfg = dict(config or {})
        schema = FeatureSchema.from_data(X_train, feature_names, cfg)
        rules: List[Rule] = []
        for item in cfg.get("rules") or cfg.get("causal_rules") or []:     # 'causal_rules' é alias legado: tratadas como monotonia, NÃO causais
            if item.get("if_feature") and item.get("then_feature"):
                rules.append(monotone(item["if_feature"], item["then_feature"], str(item.get("direction", "same")),
                                      severity=str(item.get("severity", HARD)), tolerance=float(item.get("tolerance", EPS))))
            elif item.get("kind") == "dependency":
                rules.append(requires_co_change(item["feature"], item["depends_on"], severity=str(item.get("severity", HARD))))
            elif item.get("kind") == "incompatible":
                rules.append(incompatible([tuple(c) for c in item["conditions"]], severity=str(item.get("severity", HARD))))
        cs = cls(schema, rules, nonactionable_policy=str(cfg.get("nonactionable_policy", "invalid")),
                 semantic_source="config" if rules else "none", derive_fn=cfg.get("derive_fn"))
        if schema.derived_names() and cs.derive_fn is None:
            cs.ontology_notes.append("features derivadas sem função de derivação: ficam congeladas nos valores originais (podem não reflectir as originais alteradas)")
        if cs.nonactionable_policy not in ("invalid", "warn"):
            raise ValueError("nonactionable_policy deve ser 'invalid' ou 'warn'.")
        if ontology_constraints is not None:
            ontology_constraints.apply_to(cs, X_train)
        return cs


@dataclass
class ExtractedConstraints:
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    integer: set = field(default_factory=set)
    binary: set = field(default_factory=set)
    allowed_values: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    immutable: set = field(default_factory=set)
    directions: Dict[str, str] = field(default_factory=dict)
    rules: List[Rule] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    not_extracted: List[str] = field(default_factory=list)
    skipped_by_unit_check: List[str] = field(default_factory=list)

    def apply_to(self, cs: ConstraintSet, X_train, min_reference_coverage: float = 0.99) -> None:
        """Aplica ao ConstraintSet. Um range OWL só é usado se o TREINO já o cumpre (prova de unidades iguais)."""
        schema = cs.schema
        X = np.asarray(X_train, dtype=float)
        for name, (lo, hi) in self.ranges.items():
            if name not in schema.index:
                continue
            j = schema.index[name]
            col = X[:, j][np.isfinite(X[:, j])]
            if len(col):
                ok = np.ones(len(col), dtype=bool)
                if lo is not None:
                    ok &= col >= lo - EPS
                if hi is not None:
                    ok &= col <= hi + EPS
                if ok.mean() < min_reference_coverage:
                    self.skipped_by_unit_check.append(name)
                    self.notes.append(f"range OWL de '{name}' ignorado: só {ok.mean():.1%} do treino o cumpre (espaço do modelo ≠ unidades da OWL)")
                    continue
            schema.set_range(name, lo, hi, "owl")
        for name in self.integer:
            if name in schema.index and schema.specs[schema.index[name]].kind == "continuous":
                col = X[:, schema.index[name]]
                if np.mean(np.abs(col - np.round(col)) <= 1e-9) >= min_reference_coverage:
                    schema.specs[schema.index[name]].kind = "integer"
        for name in self.immutable:
            if name in schema.index:
                schema.specs[schema.index[name]].immutable = True
        for name, direction in self.directions.items():
            if name in schema.index:
                schema.specs[schema.index[name]].direction = direction
        for name, values in self.allowed_values.items():
            if name in schema.index and schema.specs[schema.index[name]].kind == "code":
                schema.specs[schema.index[name]].allowed_values = tuple(sorted(values))
        cs.rules.extend(self.rules)
        cs.semantic_source = "ontology"
        cs.ontology_notes.extend(self.notes)
        cs.not_extracted.extend(self.not_extracted)
        schema.units = schema._build_units()


class OntologyConstraintExtractor:
    """Extrai SÓ o que a OWL contém. Nunca inventa constraints.

    Extraído: faixas de datatype (``ConstrainedDatatype`` min/max incl./excl.), tipos ``int``/``bool``,
    enumerações (``OneOf``), e anotações explícitas no vocabulário cfkit: ``cf_immutable``, ``cf_direction``,
    ``cf_dependsOn``, ``cf_incompatibleWith``. Classes disjuntas, restrições de classe e equivalências são
    reportadas em ``not_extracted`` (não são mapeáveis para features sem anotações explícitas)."""

    ANNOTATIONS = ("cf_immutable", "cf_direction", "cf_dependsOn", "cf_incompatibleWith")

    def __init__(self, ontology: Any, feature_names: Sequence[str], mapping: Optional[Mapping[str, str]] = None):
        self.ontology = ontology
        self.names = [str(n) for n in feature_names]
        self.mapping = dict(mapping or {})
        self._entities = self._collect()

    def _collect(self) -> Dict[str, Any]:
        entities: Dict[str, Any] = {}
        if self.ontology is None:
            return entities
        for accessor in ("data_properties", "object_properties", "classes"):
            fn = getattr(self.ontology, accessor, None)
            if not callable(fn):
                continue
            try:
                items = list(fn())
            except Exception:
                continue
            for e in items:
                keys = [getattr(e, "name", "")] + [str(x) for x in (getattr(e, "label", None) or [])]
                for k in keys:
                    if k:
                        entities.setdefault(_norm(k), e)
        return entities

    def _entity_for(self, feature: str) -> Any:
        token = self.mapping.get(feature, feature)
        if not isinstance(token, str):
            return token
        return self._entities.get(_norm(token)) or self._entities.get(_norm("has" + token))

    @staticmethod
    def _range_info(entity: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = {"lo": None, "hi": None, "integer": False, "binary": False, "enum": None}
        for rng in (getattr(entity, "range", None) or []):
            if rng is int:
                info["integer"] = True
            elif rng is bool:
                info["binary"] = True
            else:
                base = getattr(rng, "base_datatype", None)
                if base is int:
                    info["integer"] = True
                for attr, key, sign in (("min_inclusive", "lo", 1), ("min_exclusive", "lo", 1), ("max_inclusive", "hi", -1), ("max_exclusive", "hi", -1)):
                    val = getattr(rng, attr, None)
                    if val is not None:
                        info[key] = float(val)
                inst = getattr(rng, "instances", None)
                if inst:
                    try:
                        info["enum"] = tuple(float(v) for v in inst)
                    except (TypeError, ValueError):
                        pass
        return info

    @staticmethod
    def _annotation(entity: Any, name: str):
        try:
            value = getattr(entity, name, None)
        except Exception:
            return None
        if isinstance(value, list):
            return value[0] if value else None
        return value

    def extract(self) -> ExtractedConstraints:
        out = ExtractedConstraints()
        if self.ontology is None:
            out.notes.append("sem ontologia")
            return out
        for feature in self.names:
            entity = self._entity_for(feature)
            if entity is None:
                out.not_extracted.append(f"{feature}: sem entidade OWL mapeada")
                continue
            info = self._range_info(entity)
            if info["lo"] is not None or info["hi"] is not None:
                out.ranges[feature] = (info["lo"], info["hi"])
            if info["integer"]:
                out.integer.add(feature)
            if info["binary"]:
                out.binary.add(feature)
            if info["enum"]:
                out.allowed_values[feature] = info["enum"]
            imm = self._annotation(entity, "cf_immutable")
            if imm is not None and str(imm).lower() in ("true", "1", "yes"):
                out.immutable.add(feature)
            direction = self._annotation(entity, "cf_direction")
            if direction in ("increase_only", "decrease_only", "immutable", "both"):
                out.directions[feature] = str(direction)
            dep = self._annotation(entity, "cf_dependsOn")
            if dep and str(dep) in self.names:
                out.rules.append(requires_co_change(feature, str(dep), source="ontology"))
            inc = self._annotation(entity, "cf_incompatibleWith")
            if inc and ":" in str(inc):
                out.not_extracted.append(f"{feature}: cf_incompatibleWith requer formato 'outra=valor' (ignorado: {inc})")
            elif inc and "=" in str(inc):
                other, val = str(inc).split("=", 1)
                try:
                    out.rules.append(incompatible([(feature, "!=", 0.0), (other.strip(), "==", float(val))], source="ontology"))
                except ValueError:
                    out.not_extracted.append(f"{feature}: valor inválido em cf_incompatibleWith ({inc})")
        out.not_extracted.append("restrições de classe, AllDisjoint e equivalências não são mapeadas para features sem anotações explícitas")
        out.notes.append(f"extraídos: {len(out.ranges)} ranges, {len(out.integer)} inteiros, {len(out.rules)} regras, {len(out.immutable)} imutáveis")
        return out
