"""Inspeção DOCUMENTAL dos datasets CONFIRM candidatos (Fase 0.5). NÃO treina nenhum modelo nem calcula nenhuma métrica de desempenho.

Para cada candidato regista: proveniência/licença/DOI (metadados UCI), sha256 dos dados, dimensões, tipos, classes e contagens, ausentes por atributo,
duplicados exatos e conflituantes, colunas candidatas a identificador/tempo, e as flags dos critérios EC3/EC6/EC7 do Documento 8. EC1 (exposição no
repositório), EC4/EC5/EC8 (leitura humana do dicionário de dados) ficam como campos a preencher. A análise (``analyse_frame``) é pura e testável sem rede.

    python -m validation.confirm.inspect_candidates --ids all --out docs/b1b4/confirm      # precisa de rede para archive.ics.uci.edu e de ``ucimlrepo``
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# id UCI (de memória, a VERIFICAR contra ``metadata.name``), token esperado no nome, colunas a remover por fuga documentada (EC5; decidido antes de qualquer resultado)
CANDIDATES: Dict[str, Dict[str, Any]] = {
    "car_evaluation": {"uci_id": 19, "name_token": "car", "primary": True, "order": 1},
    "banknote_authentication": {"uci_id": 267, "name_token": "banknote", "primary": True, "order": 2},
    "spambase": {"uci_id": 94, "name_token": "spambase", "primary": True, "order": 3},
    "statlog_vehicle_silhouettes": {"uci_id": 149, "name_token": "vehicle", "primary": True, "order": 4},
    "online_shoppers_purchasing_intention": {"uci_id": 468, "name_token": "shoppers", "primary": True, "order": 5},
    "dry_bean": {"uci_id": 602, "name_token": "bean", "primary": True, "order": 6},
    "steel_plates_faults": {"uci_id": 198, "name_token": "steel", "primary": True, "order": 7},
    "occupancy_detection": {"uci_id": 357, "name_token": "occupancy", "primary": True, "order": 8},
    "mushroom": {"uci_id": 73, "name_token": "mushroom", "primary": True, "order": 9},
    "statlog_german_credit": {"uci_id": 144, "name_token": "german", "primary": True, "order": 10},
    "bank_marketing": {"uci_id": 222, "name_token": "bank", "primary": True, "order": 11, "documented_leak_columns": ["duration"]},
    "rice_cammeo_osmancik": {"uci_id": 545, "name_token": "rice", "primary": True, "order": 12},
    # reservas, por ordem
    "phishing_websites": {"uci_id": 327, "name_token": "phishing", "primary": False, "order": 101},
    "magic_gamma_telescope": {"uci_id": 159, "name_token": "magic", "primary": False, "order": 102},
    "image_segmentation": {"uci_id": 50, "name_token": "segmentation", "primary": False, "order": 103},
    "page_blocks_classification": {"uci_id": 78, "name_token": "page", "primary": False, "order": 104},
    "contraceptive_method_choice": {"uci_id": 30, "name_token": "contraceptive", "primary": False, "order": 105},
    "heart_disease": {"uci_id": 45, "name_token": "heart", "primary": False, "order": 106},
    "statlog_landsat_satellite": {"uci_id": 146, "name_token": "landsat", "primary": False, "order": 107},
    "seeds": {"uci_id": 236, "name_token": "seeds", "primary": False, "order": 108},
}
EC3 = {"n_min": 150, "n_max": 50_000, "p_min": 4, "p_max": 200, "k_min": 2, "k_max": 10, "min_per_class": 30}
EC6_MAX_MISSING_PER_COLUMN = 0.30


def analyse_frame(X: pd.DataFrame, y: pd.Series, *, dropped_leak_columns: Optional[list] = None) -> Dict[str, Any]:
    """Estatísticas documentais e flags EC3/EC6/EC7. Sem modelos, sem métricas de desempenho."""
    X = X.copy()
    y = y.astype(str)
    n, p = X.shape
    counts = y.value_counts().to_dict()
    miss = X.isna().mean()
    kinds = {c: ("numeric" if pd.api.types.is_numeric_dtype(X[c]) else "nominal") for c in X.columns}
    frame = X.assign(__y__=y.values)
    exact_dups = int(frame.duplicated(keep="first").sum())
    feat_dups = X.duplicated(keep=False)
    conflicting = int(frame[feat_dups].groupby(list(X.columns), dropna=False)["__y__"].nunique().gt(1).sum()) if feat_dups.any() else 0
    id_like = [c for c in X.columns if X[c].nunique(dropna=True) == n and n > 20]
    time_like = [c for c in X.columns if "date" in str(c).lower() or "time" in str(c).lower() or "month" in str(c).lower()]
    ec3 = (EC3["n_min"] <= n <= EC3["n_max"] and EC3["p_min"] <= p <= EC3["p_max"] and EC3["k_min"] <= len(counts) <= EC3["k_max"]
           and min(counts.values()) >= EC3["min_per_class"])
    return {
        "n": int(n), "p": int(p), "n_classes": len(counts), "class_counts": {str(k): int(v) for k, v in counts.items()},
        "feature_kinds": {"numeric": sum(v == "numeric" for v in kinds.values()), "nominal": sum(v == "nominal" for v in kinds.values())},
        "missing_per_column": {c: round(float(v), 6) for c, v in miss.items() if v > 0}, "missing_overall": round(float(X.isna().to_numpy().mean()), 6),
        "exact_duplicate_rows": exact_dups, "conflicting_duplicate_groups": conflicting,
        "constant_columns": [c for c in X.columns if X[c].nunique(dropna=True) <= 1],
        "id_like_columns": id_like, "time_like_columns": time_like, "dropped_documented_leak_columns": dropped_leak_columns or [],
        "flags": {"EC3_dimensions_and_min_per_class": bool(ec3), "EC6_missing_le_30pct_per_column": bool((miss <= EC6_MAX_MISSING_PER_COLUMN).all()),
                  "EC7_exact_duplicates_present": bool(exact_dups > 0), "EC4_time_like_columns_to_review": bool(time_like),
                  "EC1_repo_exposure": "A VERIFICAR pelo autor (pesquisa no repositório já feita no Doc. 8)",
                  "EC5_EC8_human_review": "PENDENTE (dicionário de dados)"},
    }


def data_sha256(X: pd.DataFrame, y: pd.Series) -> str:
    return hashlib.sha256(pd.concat([X.reset_index(drop=True), y.reset_index(drop=True).rename("__y__")], axis=1).to_csv(index=False).encode("utf-8")).hexdigest()


def fetch_uci(uci_id: int):
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:                                  # pragma: no cover - ambiente sem a dependência
        raise SystemExit("Instale 'ucimlrepo' (pip install ucimlrepo) e garanta acesso a archive.ics.uci.edu.") from exc
    d = fetch_ucirepo(id=uci_id)
    y = d.data.targets.iloc[:, 0] if d.data.targets.shape[1] else None
    return d.data.features, y, dict(d.metadata)


def inspect_one(key: str, out_dir: Path) -> Dict[str, Any]:
    spec = CANDIDATES[key]
    X, y, meta = fetch_uci(spec["uci_id"])
    drops = [c for c in spec.get("documented_leak_columns", []) if c in X.columns]
    X = X.drop(columns=drops)
    rec = {"key": key, "uci_id_expected": spec["uci_id"], "uci_name": meta.get("name"),
           "name_check": bool(spec["name_token"].lower() in str(meta.get("name", "")).lower()),
           "license": meta.get("license") or meta.get("licence"), "doi": meta.get("dataset_doi"), "repository_url": meta.get("repository_url"),
           "year": meta.get("year_of_dataset_creation"), "data_sha256": data_sha256(X, y), **analyse_frame(X, y, dropped_leak_columns=drops)}
    (out_dir / f"{key}_inspection.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=["all"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(CANDIDATES, key=lambda k: CANDIDATES[k]["order"]) if a.ids == ["all"] else a.ids
    for k in keys:
        r = inspect_one(k, out)
        print(f"{k}: n={r['n']} p={r['p']} k={r['n_classes']} EC3={r['flags']['EC3_dimensions_and_min_per_class']} name_check={r['name_check']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
