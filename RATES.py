# """
# Compute equivalent water rate per well-date using field-specific PVT tables,
# with these rules:

# 1) Property lookup pressure is capped at 400 bar:
#       P_LOOKUP = min(BHP, 400.0)

# 2) Any missing rate values (NaN) in production/injection columns are treated as zero:
#       WOPR, WGPR, WWPR, WGIR, WWIR  -> fillna(0)

# 3) PVT properties are interpolated per FIELD (linear).

# qw = WOPR*(Bo - Bg*Rs) + WGPR*(Bg - Bo*Rv) + WWPR*(Bw) - WGIR*(Bg) - WWIR*(Bw)
# """

# from __future__ import annotations
# import argparse
# from dataclasses import dataclass
# from typing import Dict, Tuple, Optional
# import numpy as np
# import pandas as pd


# REQUIRED_DATA_COLS = ["WELL", "DATE", "BHP_bar_tuned", "WOPR", "WGPR", "WWPR", "WGIR", "WWIR"]
# REQUIRED_MAP_COLS = ["WELL", "FIELD"]
# REQUIRED_PVT_COLS = ["FIELD", "PO", "RS", "BO", "PG", "RV", "BG", "PW", "BW"]

# P_MAX_BAR = 400.0
# RATE_COLS = ["WOPR", "WGPR", "WWPR", "WGIR", "WWIR"]


# def _read_table(path: str) -> pd.DataFrame:
#     """Robust reader for .csv/.tsv/.txt with comma/tab/space separators."""
#     try:
#         return pd.read_csv(path)
#     except Exception:
#         pass
#     try:
#         return pd.read_csv(path, sep="\t")
#     except Exception:
#         pass
#     return pd.read_csv(path, sep=None, engine="python")


# def _assert_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValueError(f"{name} is missing required columns: {missing}. Found: {list(df.columns)}")


# def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
#     out = df.copy()
#     for c in cols:
#         out[c] = pd.to_numeric(out[c], errors="coerce")
#     return out


# def _prep_pvt(pvt: pd.DataFrame) -> pd.DataFrame:
#     pvt = pvt.copy()
#     _assert_columns(pvt, REQUIRED_PVT_COLS, "PVT table")

#     num_cols = [c for c in REQUIRED_PVT_COLS if c != "FIELD"]
#     pvt = _to_numeric(pvt, num_cols)

#     pvt = pvt.dropna(subset=["FIELD"])
#     pvt["FIELD"] = pvt["FIELD"].astype(str)
#     return pvt


# def _interp_with_clip(x_query: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     """
#     1D linear interpolation with constant extrapolation (edge clipping).
#     Removes NaNs from (x,y), sorts by x, averages duplicate x.
#     """
#     m = np.isfinite(x) & np.isfinite(y)
#     x = x[m]
#     y = y[m]
#     if x.size == 0:
#         return np.full_like(x_query, np.nan, dtype=float)

#     order = np.argsort(x)
#     x = x[order]
#     y = y[order]

#     if np.any(np.diff(x) == 0):
#         df = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].mean()
#         x = df["x"].to_numpy()
#         y = df["y"].to_numpy()

#     xq = np.asarray(x_query, dtype=float)
#     out = np.interp(xq, x, y, left=y[0], right=y[-1])
#     out[~np.isfinite(xq)] = np.nan
#     return out


# @dataclass(frozen=True)
# class FieldPVTInterpolators:
#     oil_rs: Tuple[np.ndarray, np.ndarray]  # PO -> RS
#     oil_bo: Tuple[np.ndarray, np.ndarray]  # PO -> BO
#     gas_rv: Tuple[np.ndarray, np.ndarray]  # PG -> RV
#     gas_bg: Tuple[np.ndarray, np.ndarray]  # PG -> BG
#     wat_bw: Tuple[np.ndarray, np.ndarray]  # PW -> BW


# def _build_field_interps(pvt: pd.DataFrame) -> Dict[str, FieldPVTInterpolators]:
#     out: Dict[str, FieldPVTInterpolators] = {}
#     for field, g in pvt.groupby("FIELD", sort=False):
#         f = str(field)
#         out[f] = FieldPVTInterpolators(
#             oil_rs=(g["PO"].to_numpy(float), g["RS"].to_numpy(float)),
#             oil_bo=(g["PO"].to_numpy(float), g["BO"].to_numpy(float)),
#             gas_rv=(g["PG"].to_numpy(float), g["RV"].to_numpy(float)),
#             gas_bg=(g["PG"].to_numpy(float), g["BG"].to_numpy(float)),
#             wat_bw=(g["PW"].to_numpy(float), g["BW"].to_numpy(float)),
#         )
#     return out


# def compute_equivalent_water_rate(
#     data_path: str,
#     well_field_map_path: str,
#     pvt_path: str,
#     out_path: str,
#     date_col: str = "DATE",
# ) -> None:
#     data = _read_table(data_path)
#     wf = _read_table(well_field_map_path)
#     pvt = _prep_pvt(_read_table(pvt_path))

#     _assert_columns(data, REQUIRED_DATA_COLS, "Production/Injection dataset")
#     _assert_columns(wf, REQUIRED_MAP_COLS, "Well-field map")

#     data = data.copy()
#     wf = wf.copy()

#     data["WELL"] = data["WELL"].astype(str)
#     wf["WELL"] = wf["WELL"].astype(str)
#     wf["FIELD"] = wf["FIELD"].astype(str)

#     if date_col not in data.columns:
#         raise ValueError(f"date_col='{date_col}' not found in data columns.")
#     data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

#     # Convert numeric columns
#     data = _to_numeric(data, ["BHP_bar_tuned"] + RATE_COLS)

#     # KEY FIX: Treat missing production/injection rates as zero
#     data[RATE_COLS] = data[RATE_COLS].fillna(0.0)

#     merged = data.merge(wf[["WELL", "FIELD"]], on="WELL", how="left")

#     # Cap pressure at 400 bar for property lookup
#     merged["P_LOOKUP"] = np.minimum(merged["BHP_bar_tuned"].to_numpy(float), P_MAX_BAR)

#     field_interps = _build_field_interps(pvt)

#     merged["RS"] = np.nan
#     merged["BO"] = np.nan
#     merged["RV"] = np.nan
#     merged["BG"] = np.nan
#     merged["BW"] = np.nan

#     for field, idx in merged.groupby("FIELD", dropna=False).groups.items():
#         if field is None or (isinstance(field, float) and np.isnan(field)):
#             continue

#         f = str(field)
#         itp = field_interps.get(f)
#         if itp is None:
#             continue

#         p_lookup = merged.loc[idx, "P_LOOKUP"].to_numpy(float)

#         merged.loc[idx, "RS"] = _interp_with_clip(p_lookup, itp.oil_rs[0], itp.oil_rs[1])
#         merged.loc[idx, "BO"] = _interp_with_clip(p_lookup, itp.oil_bo[0], itp.oil_bo[1])
#         merged.loc[idx, "RV"] = _interp_with_clip(p_lookup, itp.gas_rv[0], itp.gas_rv[1])
#         merged.loc[idx, "BG"] = _interp_with_clip(p_lookup, itp.gas_bg[0], itp.gas_bg[1])
#         merged.loc[idx, "BW"] = _interp_with_clip(p_lookup, itp.wat_bw[0], itp.wat_bw[1])

#     Bo = merged["BO"].to_numpy(float)
#     Bg = merged["BG"].to_numpy(float)
#     Rs = merged["RS"].to_numpy(float)
#     Rv = merged["RV"].to_numpy(float)
#     Bw = merged["BW"].to_numpy(float)

#     WOPR = merged["WOPR"].to_numpy(float)
#     WGPR = merged["WGPR"].to_numpy(float)
#     WWPR = merged["WWPR"].to_numpy(float)
#     WGIR = merged["WGIR"].to_numpy(float)
#     WWIR = merged["WWIR"].to_numpy(float)

#     merged["RATE"] = (
#         (WOPR * (Bo - Bg * Rs))
#         + (WGPR * (Bg - Bo * Rv))
#         + (WWPR * Bw)
#         - (WGIR * Bg)
#         - (WWIR * Bw)
#     )

#     merged.to_csv(out_path, index=False)

#     print(f"Wrote: {out_path}")
#     print(f"Rows: {len(merged)}")
#     print(f"Rows with missing FIELD (no mapping): {int(merged['FIELD'].isna().sum())}")
#     print(f"Rows with RATE = NaN (missing PVT/BHP): {int(merged['RATE'].isna().sum())}")
#     print(f"Pressure cap for lookup applied at: {P_MAX_BAR} bar (see column P_LOOKUP)")
#     print("NaN production/injection rates were treated as 0.0 for WOPR/WGPR/WWPR/WGIR/WWIR.")

# def main():
#     data      = "../input/well_modelling/prod_with_bhp_tuned.csv"
#     mapfile   = "../input/well_modelling/well_field_map.csv"
#     pvt       = "../input/well_modelling/PVT/pvt_data_reduced.csv"
#     out       = "../input/well_modelling/rates.csv"
#     compute_equivalent_water_rate(
#         data_path=data,
#         well_field_map_path=mapfile,
#         out_path=out,
#         pvt_path=pvt,
#         date_col="DATE",
#     )
#     return print("Done")

# if __name__ == "__main__":
#     main()



"""
Compute equivalent water rate per well-date using field-specific PVT tables,
with these rules:

1) Property lookup pressure is capped at 400 bar:
      P_LOOKUP = min(BHP, 400.0)

2) Any missing rate values (NaN) in production/injection columns are treated as zero:
      WOPR, WGPR, WWPR, WGIR, WWIR  -> fillna(0)

3) PVT properties are interpolated per FIELD (linear).

qw = WOPR*(Bo - Bg*Rs) + WGPR*(Bg - Bo*Rv) + WWPR*(Bw) - WGIR*(Bg) - WWIR*(Bw)
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


REQUIRED_DATA_COLS = ["WELL", "DATE", "BHP_bar_tuned", "WOPR", "WGPR", "WWPR", "WGIR", "WWIR"]
REQUIRED_MAP_COLS = ["WELL", "FIELD"]
REQUIRED_PVT_COLS = ["FIELD", "PO", "RS", "BO", "PG", "RV", "BG", "PW", "BW"]

P_MAX_BAR = 400.0
RATE_COLS = ["WOPR", "WGPR", "WWPR", "WGIR", "WWIR"]


def _read_table(path: str) -> pd.DataFrame:
    """Robust reader for .csv/.tsv/.txt with comma/tab/space separators."""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        pass
    try:
        return pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        pass
    return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def _assert_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}. Found: {list(df.columns)}")


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _prep_pvt(pvt: pd.DataFrame) -> pd.DataFrame:
    pvt = pvt.copy()
    _assert_columns(pvt, REQUIRED_PVT_COLS, "PVT table")

    num_cols = [c for c in REQUIRED_PVT_COLS if c != "FIELD"]
    pvt = _to_numeric(pvt, num_cols)

    pvt = pvt.dropna(subset=["FIELD"])
    pvt["FIELD"] = pvt["FIELD"].astype(str)
    return pvt


def _interp_with_clip(x_query: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    1D linear interpolation with constant extrapolation (edge clipping).
    Removes NaNs from (x,y), sorts by x, averages duplicate x.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return np.full_like(x_query, np.nan, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if np.any(np.diff(x) == 0):
        df = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].mean()
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()

    xq = np.asarray(x_query, dtype=float)
    out = np.interp(xq, x, y, left=y[0], right=y[-1])
    out[~np.isfinite(xq)] = np.nan
    return out


@dataclass(frozen=True)
class FieldPVTInterpolators:
    oil_rs: Tuple[np.ndarray, np.ndarray]  # PO -> RS
    oil_bo: Tuple[np.ndarray, np.ndarray]  # PO -> BO
    gas_rv: Tuple[np.ndarray, np.ndarray]  # PG -> RV
    gas_bg: Tuple[np.ndarray, np.ndarray]  # PG -> BG
    wat_bw: Tuple[np.ndarray, np.ndarray]  # PW -> BW


def _build_field_interps(pvt: pd.DataFrame) -> Dict[str, FieldPVTInterpolators]:
    out: Dict[str, FieldPVTInterpolators] = {}
    for field, g in pvt.groupby("FIELD", sort=False):
        f = str(field)
        out[f] = FieldPVTInterpolators(
            oil_rs=(g["PO"].to_numpy(float), g["RS"].to_numpy(float)),
            oil_bo=(g["PO"].to_numpy(float), g["BO"].to_numpy(float)),
            gas_rv=(g["PG"].to_numpy(float), g["RV"].to_numpy(float)),
            gas_bg=(g["PG"].to_numpy(float), g["BG"].to_numpy(float)),
            wat_bw=(g["PW"].to_numpy(float), g["BW"].to_numpy(float)),
        )
    return out


def compute_equivalent_water_rate(
    data_path: str,
    well_field_map_path: str,
    pvt_path: str,
    out_path: str,
    date_col: str = "DATE",
) -> None:
    data = _read_table(data_path)
    wf = _read_table(well_field_map_path)
    pvt = _prep_pvt(_read_table(pvt_path))

    _assert_columns(data, REQUIRED_DATA_COLS, "Production/Injection dataset")
    _assert_columns(wf, REQUIRED_MAP_COLS, "Well-field map")

    data = data.copy()
    wf = wf.copy()

    data["WELL"] = data["WELL"].astype(str).str.strip()
    wf["WELL"] = wf["WELL"].astype(str).str.strip()
    wf["FIELD"] = wf["FIELD"].astype(str).str.strip()

    if date_col not in data.columns:
        raise ValueError(f"date_col='{date_col}' not found in data columns.")
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

    # Convert numeric columns
    data = _to_numeric(data, ["BHP_bar_tuned"] + RATE_COLS)

    # KEY FIX: Treat missing production/injection rates as zero
    data[RATE_COLS] = data[RATE_COLS].fillna(0.0)

    merged = data.merge(wf[["WELL", "FIELD"]], on="WELL", how="left")
    missing_field_mask = merged["FIELD"].isna()
    merged["FIELD_KEY"] = merged["FIELD"].where(~missing_field_mask, "").astype(str)

    # Cap pressure at 400 bar for property lookup
    merged["P_LOOKUP"] = np.minimum(merged["BHP_bar_tuned"].to_numpy(float), P_MAX_BAR)

    field_interps = _build_field_interps(pvt)

    merged["RS"] = np.nan
    merged["BO"] = np.nan
    merged["RV"] = np.nan
    merged["BG"] = np.nan
    merged["BW"] = np.nan

    for field, idx in merged.groupby("FIELD_KEY", dropna=False).groups.items():
        if not field:
            continue

        f = str(field)
        itp = field_interps.get(f)
        if itp is None:
            continue

        p_lookup = merged.loc[idx, "P_LOOKUP"].to_numpy(float)

        merged.loc[idx, "RS"] = _interp_with_clip(p_lookup, itp.oil_rs[0], itp.oil_rs[1])
        merged.loc[idx, "BO"] = _interp_with_clip(p_lookup, itp.oil_bo[0], itp.oil_bo[1])
        merged.loc[idx, "RV"] = _interp_with_clip(p_lookup, itp.gas_rv[0], itp.gas_rv[1])
        merged.loc[idx, "BG"] = _interp_with_clip(p_lookup, itp.gas_bg[0], itp.gas_bg[1])
        merged.loc[idx, "BW"] = _interp_with_clip(p_lookup, itp.wat_bw[0], itp.wat_bw[1])

    Bo = merged["BO"].to_numpy(float)
    Bg = merged["BG"].to_numpy(float)
    Rs = merged["RS"].to_numpy(float)
    Rv = merged["RV"].to_numpy(float)
    Bw = merged["BW"].to_numpy(float)

    WOPR = merged["WOPR"].to_numpy(float)
    WGPR = merged["WGPR"].to_numpy(float)
    WWPR = merged["WWPR"].to_numpy(float)
    WGIR = merged["WGIR"].to_numpy(float)
    WWIR = merged["WWIR"].to_numpy(float)

    merged["RATE"] = (
        (WOPR * (Bo - Bg * Rs))
        + (WGPR * (Bg - Bo * Rv))
        + (WWPR * Bw)
        - (WGIR * Bg)
        - (WWIR * Bw)
    )

    merged.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Rows: {len(merged)}")
    print(f"Rows with missing FIELD (no mapping): {int(missing_field_mask.sum())}")
    print(f"Rows with RATE = NaN (missing PVT/BHP): {int(merged['RATE'].isna().sum())}")
    print(f"Pressure cap for lookup applied at: {P_MAX_BAR} bar (see column P_LOOKUP)")
    print("NaN production/injection rates were treated as 0.0 for WOPR/WGPR/WWPR/WGIR/WWIR.")


def main():
    data = "../input/well_modelling/prod_with_bhp_tuned.csv"
    mapfile = "../input/well_modelling/well_field_map.csv"
    pvt = "../input/well_modelling/PVT/pvt_data_reduced.csv"
    out = "../input/well_modelling/rates.csv"
    compute_equivalent_water_rate(
        data_path=data,
        well_field_map_path=mapfile,
        out_path=out,
        pvt_path=pvt,
        date_col="DATE",
    )
    return print("Done")


if __name__ == "__main__":
    main()
