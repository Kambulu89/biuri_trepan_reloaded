"""Datasets DEV com atributos CATEGÓRICOS reais (P1; Doc. 26 §5): DADOS + METADADOS apenas. Registam-se no ``REGISTRY`` ao importar este módulo (não alteram
``datasets.py``, logo não entram no manifesto v1 nem nas campanhas anteriores).

Pré-processamento GENÉRICO (igual para qualquer tabela): colunas numéricas ficam como estão; colunas nominais (texto) → one-hot com níveis ordenados;
categorias desconhecidas em dados novos → linha de zeros no grupo (``OneHotSchema.transform``). Ordinais/contagens declaradas nos metadados do dataset
(``ORDINAL_FEATURES``) só afetam o amostrador C (±1 passo) — nunca o algoritmo TREPAN. Sem lógica por dataset em ``core/``.

German Credit (UCI Statlog, 1000 linhas, 20 atributos): ficheiro local ``validation/counterfactual_research/datasets/originales/german.data`` (já exposto; ver
Doc. 8: inelegível para confirmatório). Adult Census: reconstruído do ficheiro de instâncias do repositório (``data/adult_census_full_ontology.owl``) —
ver ``adult_frame``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from validation.benchmark import datasets as ds_mod

ROOT = Path(__file__).resolve().parents[2]
GERMAN_FILE = ROOT / "validation" / "counterfactual_research" / "datasets" / "originales" / "german.data"
ADULT_OWL = ROOT / "data" / "adult_census_full_ontology.owl"

# dicionário de dados público do UCI (nomes legíveis; ordem das colunas do ficheiro)
GERMAN_COLUMNS = ["status_checking_account", "duration_months", "credit_history", "purpose", "credit_amount", "savings_account", "employment_since",
                  "installment_rate_pct_income", "personal_status_sex", "other_debtors", "residence_since", "property", "age_years",
                  "other_installment_plans", "housing", "existing_credits_count", "job", "people_liable_count", "telephone", "foreign_worker"]
GERMAN_ORDINAL = ["installment_rate_pct_income", "residence_since", "existing_credits_count", "people_liable_count"]
# grupos temáticos definidos SÓ a partir do significado público dos atributos (antes de qualquer resultado); membros = atributos de origem
GERMAN_THEMES: Dict[str, List[str]] = {
    "AccountStatus": ["status_checking_account", "savings_account"],
    "CreditRequest": ["duration_months", "credit_amount", "installment_rate_pct_income", "purpose"],
    "EmploymentStability": ["employment_since", "job", "residence_since", "housing"],
    "PersonalProfile": ["personal_status_sex", "age_years", "people_liable_count", "telephone", "foreign_worker"],
    "DebtAndAssets": ["credit_history", "other_debtors", "property", "other_installment_plans", "existing_credits_count"],
}
ORDINAL_FEATURES: Dict[str, List[str]] = {}          # dataset_id -> nomes de features CODIFICADAS ordinais (metadados)


class OneHotSchema:
    """Níveis por coluna nominal (ordenados); ``transform`` mapeia níveis desconhecidos para zeros (nunca inventa categoria)."""
    def __init__(self, nominal: Sequence[str]):
        self.nominal = list(nominal)
        self.levels: Dict[str, List[str]] = {}

    def fit(self, df: pd.DataFrame) -> "OneHotSchema":
        self.levels = {c: sorted(df[c].astype(str).unique()) for c in self.nominal}
        return self

    def names(self, df_columns: Sequence[str]) -> List[str]:
        out = []
        for c in df_columns:
            out.extend([f"{c}__{lv}" for lv in self.levels[c]] if c in self.levels else [c])
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in df.columns:
            if c in self.levels:
                v = df[c].astype(str).to_numpy()
                cols.append(np.column_stack([(v == lv).astype(float) for lv in self.levels[c]]))
            else:
                cols.append(df[[c]].to_numpy(dtype=float))
        return np.hstack(cols)


def german_frame() -> Tuple[pd.DataFrame, np.ndarray]:
    raw = pd.read_csv(GERMAN_FILE, sep=" ", header=None)
    X = raw.iloc[:, :20].copy()
    X.columns = GERMAN_COLUMNS
    y = (raw.iloc[:, 20].to_numpy() == 2).astype(int)           # 1 = bom pagador (classe 0 «good»), 2 = mau (classe 1 «bad»)
    return X, y


def german_loader():
    X, y = german_frame()
    nominal = [c for c in X.columns if X[c].dtype == object]
    enc = OneHotSchema(nominal).fit(X)
    names = enc.names(list(X.columns))
    return enc.transform(X), y, names, "credit_risk", ["good", "bad"]


def german_groups(encoded_names: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for theme, attrs in GERMAN_THEMES.items():
        members = [n for n in encoded_names if n.split("__", 1)[0] in attrs]
        if members:
            groups[theme] = members
    return groups


def adult_frame(path: Path = ADULT_OWL, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Reconstrói a tabela do Adult Census a partir das instâncias ``AdultRecord`` do ficheiro OWL do repositório (stream por regex; sem rdflib)."""
    pat_block = re.compile(r"<rdf:Description rdf:about=\"[^\"]*#AdultRecord_(\d+)\">(.*?)</rdf:Description>", re.S)
    pat_lit = re.compile(r"<adult:(\w+) rdf:datatype=\"[^\"]*\">([^<]*)</adult:\1>")
    pat_res = re.compile(r"<adult:(\w+) rdf:resource=\"[^\"]*#([^\"]*)\"/>")
    pat_type = re.compile(r"<rdf:type rdf:resource=\"[^\"]*#(HighIncomeAdult|LowIncomeAdult)\"/>")
    rows = []
    with open(path, encoding="utf-8") as fh:
        buf = ""
        for chunk in iter(lambda: fh.read(1 << 22), ""):
            buf += chunk
            last = 0
            for m in pat_block.finditer(buf):
                body = m.group(2)
                r = {k: v for k, v in pat_lit.findall(body)}
                r.update({k: v for k, v in pat_res.findall(body)})
                t = pat_type.search(body)
                r["income"] = t.group(1) if t else None
                r["row"] = int(m.group(1))
                rows.append(r)
                last = m.end()
            buf = buf[last:]
            if max_rows and len(rows) >= max_rows:
                break
    return pd.DataFrame(rows).sort_values("row").reset_index(drop=True)


def register() -> None:
    v = "local file validation/counterfactual_research/datasets/originales/german.data"
    spec = ds_mod.DatasetSpec("german_credit", ds_mod.REAL_OFFLINE, "UCI Statlog (German Credit Data), ficheiro local do repositório", v, german_loader,
                              "german_credit", german_groups, str(GERMAN_FILE), ranking_eligible=False)
    ds_mod.REGISTRY.setdefault("german_credit", spec)
    ORDINAL_FEATURES.setdefault("german_credit", list(GERMAN_ORDINAL))


register()


def freeze_german_ontology(force: bool = False) -> str:
    """Gera a TBox UMA vez (só nomes/significado dos atributos) e recusa sobrescrever."""
    from core.benchmark.domain_tbox import create_domain_tbox
    spec = ds_mod.REGISTRY["german_credit"]
    dest = spec.ontology_path
    if dest.exists() and not force:
        return str(dest)
    names = spec.loader()[2]
    return create_domain_tbox(dest, dataset_name="german_credit", groups=spec.groups(names))


def ordinal_columns(dataset_id: str, feature_names: Sequence[str]) -> List[int]:
    wanted = set(ORDINAL_FEATURES.get(dataset_id, []))
    return [i for i, n in enumerate(feature_names) if n in wanted]
