from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NHANES_MISSING_NUMERIC = {7777, 9999, 77777, 99999, 777777, 999999}
NHANES_MISSING_CATEGORICAL = {7, 9, 77, 99}


def _coerce_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s.mask(s.isin(NHANES_MISSING_NUMERIC))
    return out


def _mean_of_existing(df: pd.DataFrame, cols: list[str], new_col: str) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        df[new_col] = np.nan
        return df
    df[new_col] = df[existing].astype(float).mean(axis=1, skipna=True)
    return df


def make_target(df: pd.DataFrame) -> pd.Series:
    """Binary target from DIQ010.

    DIQ010: Doctor told you have diabetes
      1=Yes, 2=No, 3=Borderline, 7=Refused, 9=Don't know

    We map 1->1, 2->0 and drop others.
    """
    if "DIQ010" not in df.columns:
        raise ValueError("DIQ010 not found; ensure DIQ component is loaded.")

    y = df["DIQ010"].copy()
    y = y.replace({7: np.nan, 9: np.nan, 3: np.nan})
    y = y.map({1: 1, 2: 0})
    return y.astype("float").astype("Int64")


def build_features(df: pd.DataFrame, *, with_labs: bool) -> pd.DataFrame:
    df = _coerce_missing(df)
    out = df.copy()

    # Blood pressure: average available replicates
    out = _mean_of_existing(out, ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"], "BP_SYS_MEAN")
    out = _mean_of_existing(out, ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"], "BP_DIA_MEAN")

    keep = [
        # Demographics
        "RIDAGEYR", "RIAGENDR", "RIDRETH3",
        "DMDEDUC2", "INDFMPIR",
        # Anthropometrics
        "BMXBMI", "BMXWAIST",
        # BP engineered
        "BP_SYS_MEAN", "BP_DIA_MEAN",
        # Lifestyle
        "SMQ020",  # smoked >=100 cigarettes (1/2)
        "ALQ101",  # >=12 drinks in any one year (1/2)
        "PAQ650",  # vigorous work activity (1/2)
        "PAQ665",  # moderate work activity (1/2)
        # Weights
        "WTMEC2YR",
        "NHANES_CYCLE",
    ]

    if with_labs:
        # HbA1c from GHB
        keep.append("LBXGH")

    cols = [c for c in keep if c in out.columns]
    X = out[cols].copy()

    # Many NHANES categoricals are numeric-coded.
    for c in ["RIAGENDR", "RIDRETH3", "DMDEDUC2", "SMQ020", "ALQ101", "PAQ650", "PAQ665"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").mask(X[c].isin(NHANES_MISSING_CATEGORICAL))

    return X


def split_X_y_and_weights(
    df: pd.DataFrame,
    *,
    with_labs: bool,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    y = make_target(df)
    X = build_features(df, with_labs=with_labs)

    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype(int).reset_index(drop=True)

    w = None
    if "WTMEC2YR" in X.columns:
        # 4 cycles -> divide 2-yr MEC weight by 4
        w = X["WTMEC2YR"].astype(float) / 4.0
        X = X.drop(columns=["WTMEC2YR"])

    # Cycle is useful for diagnostics but not a predictor
    if "NHANES_CYCLE" in X.columns:
        X = X.drop(columns=["NHANES_CYCLE"])

    return X, y, w


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
