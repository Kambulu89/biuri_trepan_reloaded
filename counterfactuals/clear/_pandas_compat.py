"""Compatibilidade pandas 2.x — substitui DataFrame.append removido."""
from __future__ import annotations

import pandas as pd


def df_append(df, other, ignore_index: bool = False, sort: bool = False):
    """Equivalente a DataFrame.append para pandas >= 2.0."""
    if other is None:
        return df
    if isinstance(other, pd.Series):
        other = other.to_frame().T
    elif not isinstance(other, pd.DataFrame):
        other = pd.DataFrame([other])
    return pd.concat([df, other], ignore_index=ignore_index, sort=sort)
