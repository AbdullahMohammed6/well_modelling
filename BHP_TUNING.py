from __future__ import annotations
import argparse
from typing import Iterable, Sequence
import numpy as np
import pandas as pd


DEFAULT_INPUT = "../input/well_modelling/prod_with_bhp.csv"
DEFAULT_OUTPUT = "../input/well_modelling/prod_with_bhp_tuned.csv"
CALC_COLUMN = "BHP_bar"
ACTIVITY_COLS = ["WOPR", "WWPR", "WGPR", "WWIR", "WGIR"]


def _head_list(x: Iterable, n: int = 5) -> list:
    vals = list(x)
    return vals[: min(n, len(vals))]


def _candidate_measured_columns(columns: Sequence[str]) -> list[str]:
    cols_upper = [c.upper() for c in columns]
    banned = {"BHP_BAR", "BHP_STATUS", "BHP_BAR_SMOOTHED"}
    candidates = []
    for col, up in zip(columns, cols_upper):
        if "BHP" not in up:
            continue
        if up in banned:
            continue
        if "CALC" in up or "CALCULATED" in up:
            continue
        candidates.append(col)
    return candidates


def _pick_measured_column(df: pd.DataFrame) -> str:
    candidates = _candidate_measured_columns(df.columns)
    if not candidates:
        raise ValueError(
            "No measured BHP column found. "
            "Add a column containing measured BHP values (e.g., BHP_MEASURED) "
            "or adjust _candidate_measured_columns()."
        )
    for pref in ("BHP_MEASURED", "BHP_MEAS", "BHP_OBS", "WBHP"):
        for c in candidates:
            if c.upper().startswith(pref):
                return c
    return candidates[0]


def _fit_adjustment(calc: pd.Series, meas: pd.Series) -> tuple[float, float]:
    mask = calc.notna() & meas.notna()
    if not mask.any():
        return (1.0, 0.0)
    c = calc[mask].astype(float)
    m = meas[mask].astype(float)
    finite = np.isfinite(c) & np.isfinite(m)
    if not finite.any():
        return (1.0, 0.0)
    c = c[finite]
    m = m[finite]
    if len(c) >= 2 and c.nunique() > 1:
        try:
            slope, intercept = np.polyfit(c, m, 1)
        except np.linalg.LinAlgError:
            slope = 1.0
            intercept = float(np.nanmedian(m - c))
    else:
        slope = 1.0
        intercept = float(m.iloc[0] - c.iloc[0])
    return float(slope), float(intercept)


def tune_well(df_well: pd.DataFrame, measured_col: str, calc_col: str = CALC_COLUMN) -> pd.Series:
    calc = pd.to_numeric(df_well.get(calc_col), errors="coerce")
    meas = pd.to_numeric(df_well.get(measured_col), errors="coerce")
    meas = meas.where(meas.between(50, 450))
    calc = calc.where(calc.between(100, 450))

    slope, intercept = _fit_adjustment(calc, meas)
    tuned = slope * calc + intercept

    tuned = tuned.where(calc.notna(), meas)
    tuned = tuned.clip(lower=100, upper=450)
    tuned = tuned.rolling(window=5, min_periods=1, center=True).median()
    tuned = tuned.rolling(window=5, min_periods=1, center=True).mean()

    activity_cols = [c for c in ACTIVITY_COLS if c in df_well.columns]
    if activity_cols:
        activity = (
            df_well[activity_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .ne(0.0)
            .any(axis=1)
        )
        active_tuned = tuned.where(activity)
        continuous = (
            active_tuned.interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
            .clip(lower=100, upper=450)
        )
        tuned = tuned.where(~activity, continuous)
    return tuned


def process(input_path: str = DEFAULT_INPUT, output_path: str = DEFAULT_OUTPUT) -> None:
    df = pd.read_csv(input_path, low_memory=False)
    if CALC_COLUMN not in df.columns:
        raise ValueError(f"{CALC_COLUMN} column is required in {input_path}.")

    measured_col = _pick_measured_column(df)
    print(f"Using measured BHP column: {measured_col}")

    df = df.copy()
    df["BHP_bar_tuned"] = (
        df.groupby("WELL")
        .apply(lambda g: tune_well(g, measured_col))
        .reset_index(level=0, drop=True)
    )

    missing_calc = df[CALC_COLUMN].isna().sum()
    missing_meas = df[measured_col].isna().sum()
    print(f"Rows missing calculated BHP: {missing_calc}")
    print(f"Rows missing measured BHP  : {missing_meas}")

    df.to_csv(output_path, index=False)
    print(f"Wrote tuned BHP to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune calculated BHP logs using measured points.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV containing BHP_bar and measured BHP.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV for tuned BHP values.")
    args = parser.parse_args()
    process(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
