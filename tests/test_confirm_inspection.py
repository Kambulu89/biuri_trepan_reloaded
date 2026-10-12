import numpy as np
import pandas as pd

from validation.confirm.inspect_candidates import CANDIDATES, analyse_frame, data_sha256


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 5, size=n).astype(float), "c": rng.choice(["x", "y", "z"], size=n), "d": rng.normal(size=n)})
    y = pd.Series(rng.choice(["u", "v"], size=n))
    return X, y


def test_flags_ok_on_clean_frame_and_no_model_metrics():
    X, y = _frame()
    r = analyse_frame(X, y)
    assert r["flags"]["EC3_dimensions_and_min_per_class"] and r["flags"]["EC6_missing_le_30pct_per_column"]
    assert r["feature_kinds"] == {"numeric": 3, "nominal": 1}
    assert not any(k in r for k in ("accuracy", "fidelity", "score"))


def test_detects_missing_duplicates_conflicts_and_time_columns():
    X, y = _frame()
    X.loc[:90, "a"] = np.nan                                # > 30 % ausentes
    X = pd.concat([X, X.iloc[[100, 100]]], ignore_index=True)
    y = pd.concat([y, pd.Series(["u", "v"])], ignore_index=True)
    X["Month"] = 1
    r = analyse_frame(X, y)
    assert not r["flags"]["EC6_missing_le_30pct_per_column"]
    assert r["exact_duplicate_rows"] >= 1 and r["conflicting_duplicate_groups"] >= 1
    assert r["flags"]["EC4_time_like_columns_to_review"] and "Month" in r["time_like_columns"]


def test_small_class_fails_ec3_and_hash_is_stable():
    X, y = _frame()
    y.iloc[:190] = "u"
    assert not analyse_frame(X, y)["flags"]["EC3_dimensions_and_min_per_class"]
    assert data_sha256(X, y) == data_sha256(X.copy(), y.copy())


def test_registry_has_12_primary_and_ordered_reserves_without_duplicates():
    prim = [k for k, v in CANDIDATES.items() if v["primary"]]
    assert len(prim) == 12 and len({v["uci_id"] for v in CANDIDATES.values()}) == len(CANDIDATES)
    reserves = sorted((v["order"] for v in CANDIDATES.values() if not v["primary"]))
    assert reserves == list(range(101, 101 + len(reserves)))
