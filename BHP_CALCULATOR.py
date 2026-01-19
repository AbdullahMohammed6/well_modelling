from __future__ import annotations
import math
import multiprocessing as mp
from typing import Iterable
import numpy as np
import pandas as pd

G = 9.80665
INCH_TO_M = 0.0254
BAR_TO_PA = 1e5
PA_TO_BAR = 1e-5
PA_PER_PSI = 6894.757293168
P_STD_PA = 101325.0
T_STD_K = 288.15
R = 8.314462618
ACTIVITY_COLS = ["WOPR", "WWPR", "WGPR", "WWIR", "WGIR"]
DEFAULT_BHP_BAR = 100.0
DEFAULT_THP_BAR = 100.0
THP_SMOOTH_WINDOW = 3
BHP_SMOOTH_WINDOW = 5


def _head_list(x: Iterable, n: int = 10):
    x = list(x)
    return x[: min(n, len(x))]

def report_mapping_coverage(df: pd.DataFrame, df_name: str, raw_well_col: str) -> None:
    if "WELL_KEY" not in df.columns:
        print(f"{df_name}: no WELL_KEY column; cannot report mapping.")
        return

    unmapped = df[df["WELL_KEY"].isna()]
    if unmapped.empty:
        print(f"{df_name}: mapping OK (0 unmapped rows).")
        return

    raw_vals = unmapped[raw_well_col].dropna().unique()
    print(f"{df_name}: {len(unmapped)} unmapped rows; sample raw wells: {_head_list(raw_vals, 20)}")


def ensure_columns(df: pd.DataFrame, required: list[str | tuple[str, ...]], name: str) -> None:
    missing = []
    for item in required:
        if isinstance(item, tuple):
            if not any(col in df.columns for col in item):
                missing.append(f"one of {list(item)}")
        else:
            if item not in df.columns:
                missing.append(item)

    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def alias_first_available(df: pd.DataFrame, target: str, candidates: list[str]) -> str | None:
    if target in df.columns:
        return target

    for c in candidates:
        if c in df.columns:
            df[target] = df[c]
            return c

    return None


def find_prod_diff_wells(prod: pd.DataFrame, prod_existing: pd.DataFrame | None) -> set[str]:
    if prod_existing is None or prod_existing.empty:
        return set(prod["WELL_KEY"].dropna().unique())

    prod_existing = prod_existing.copy()
    if "DATE" in prod_existing.columns:
        prod_existing["DATE"] = pd.to_datetime(prod_existing["DATE"], errors="coerce")

    keys = ["WELL", "DATE"]
    if "WELL_KEY" in prod.columns and "WELL_KEY" in prod_existing.columns:
        keys.append("WELL_KEY")

    compare_cols = [
        col
        for col in prod.columns
        if col in prod_existing.columns and col not in keys
    ]

    existing_subset = prod_existing[keys + compare_cols].copy()
    merged = prod.merge(
        existing_subset,
        on=keys,
        how="left",
        suffixes=("", "_prev"),
        indicator=True,
    )

    new_mask = merged["_merge"].eq("left_only")
    changed_mask = pd.Series(False, index=merged.index)
    for col in compare_cols:
        prev_col = f"{col}_prev"
        if prev_col not in merged.columns:
            continue
        same = merged[col].eq(merged[prev_col]) | (
            merged[col].isna() & merged[prev_col].isna()
        )
        changed_mask |= ~same

    diff = merged[new_mask | changed_mask]
    return set(diff["WELL_KEY"].dropna().unique())


def build_tubing_intervals_for_well(comp_w: pd.DataFrame, md_limit: float) -> pd.DataFrame:
    if comp_w.empty:
        return pd.DataFrame(columns=["md_top", "md_bot", "tubing_id", "ID_m"])

    comp_w = comp_w.sort_values("RKB_HANGER").reset_index(drop=True)

    intervals = []
    for i in range(len(comp_w)):
        md_top = float(comp_w.loc[i, "RKB_HANGER"])
        md_bot = float(comp_w.loc[i + 1, "RKB_HANGER"]) if i < len(comp_w) - 1 else float(md_limit)

        if md_top >= md_limit:
            continue

        md_bot = min(md_bot, md_limit)
        if md_bot <= md_top:
            continue

        intervals.append(
            {
                "md_top": md_top,
                "md_bot": md_bot,
                "tubing_id": comp_w.loc[i, "RKB_HANGER"],
                "ID_m": float(comp_w.loc[i, "ID_m"]),
            }
        )

    return pd.DataFrame(intervals)


def assign_tubing_to_md(md_mid: float, intervals: pd.DataFrame) -> tuple[object, float]:
    if intervals.empty:
        return (None, np.nan)

    inside = intervals[(md_mid >= intervals["md_top"]) & (md_mid < intervals["md_bot"])]
    if not inside.empty:
        r = inside.iloc[0]
        return (r["tubing_id"], r["ID_m"])

    tops = intervals["md_top"].to_numpy()
    bots = intervals["md_bot"].to_numpy()
    d = np.minimum(np.abs(md_mid - tops), np.abs(md_mid - bots))
    j = int(np.argmin(d))
    r = intervals.iloc[j]
    return (r["tubing_id"], r["ID_m"])


def build_segments_for_well(
    well_key: str,
    traj_w: pd.DataFrame,
    perf_md: float,
    comp_w: pd.DataFrame,
) -> pd.DataFrame:
    traj_w = traj_w.sort_values("MD").dropna(subset=["MD", "TVD"]).reset_index(drop=True)

    min_md = float(traj_w["MD"].iloc[0])
    if perf_md < min_md:
        raise ValueError(f"perf MD ({perf_md}) is above first trajectory MD ({min_md}).")

    traj_w = traj_w[traj_w["MD"] <= perf_md].copy()
    if len(traj_w) < 2:
        raise ValueError("Not enough trajectory points <= perf MD to build segments.")

    intervals = build_tubing_intervals_for_well(comp_w, md_limit=perf_md)

    md_points = set(traj_w["MD"].astype(float).tolist())
    if not intervals.empty:
        md_points.update(intervals["md_top"].astype(float).tolist())
        md_points.update(intervals["md_bot"].astype(float).tolist())
    md_points.add(float(perf_md))

    md_points = np.array(sorted(md_points), dtype=float)
    md_points = md_points[(md_points >= min_md) & (md_points <= perf_md)]
    if md_points.size < 2:
        raise ValueError("Insufficient MD breakpoints after clipping.")

    tvd = np.interp(md_points, traj_w["MD"].to_numpy(dtype=float), traj_w["TVD"].to_numpy(dtype=float))

    segs = []
    for i in range(md_points.size - 1):
        md1, md2 = float(md_points[i]), float(md_points[i + 1])
        tvd1, tvd2 = float(tvd[i]), float(tvd[i + 1])

        dmd = md2 - md1
        if dmd <= 0:
            continue

        dtvd = tvd2 - tvd1
        ratio = np.clip(dtvd / dmd, -1.0, 1.0)
        alpha = float(np.degrees(np.arccos(ratio)))

        md_mid = 0.5 * (md1 + md2)
        tubing_id, id_m = assign_tubing_to_md(md_mid, intervals)

        segs.append(
            {
                "WELL_KEY": well_key,
                "md_top": md1,
                "md_bot": md2,
                "dMD": dmd,
                "dTVD": dtvd,
                "alpha_deg": alpha,
                "md_mid": md_mid,
                "tubing_id": tubing_id,
                "ID_m": id_m,
            }
        )

    return pd.DataFrame(segs)


def _get_first(series: pd.Series, keys, default=np.nan) -> float:
    for k in keys:
        v = series.get(k, np.nan)
        try:
            v = float(v)
        except Exception:
            continue
        if np.isfinite(v):
            return v
    return float(default)


def water_viscosity_mccain_Pa_s(
    T_C: float,
    p_Pa: float,
    salinity_wt_pct: float = 0.0,
) -> float:
    T_F = 1.8 * T_C + 32.0
    if T_F <= 0:
        raise ValueError("Temperature too low for McCain correlation (T_F must be > 0).")

    p_psia = p_Pa / PA_PER_PSI
    S = max(0.0, float(salinity_wt_pct))

    A = 109.574 - 8.40564 * S + 0.313314 * S**2 + 8.72213e-3 * S**3
    B = (
        1.2166
        - 2.63951e-2 * S
        + 6.79461e-4 * S**2
        + 5.47119e-5 * S**3
        - 1.55586e-6 * S**4
    )

    mu_w1_cP = A * (T_F ** (-B))
    corr_p = 0.9994 + 4.0295e-5 * p_psia + 3.1062e-9 * (p_psia**2)
    mu_cP = mu_w1_cP * corr_p

    return mu_cP * 1e-3


def pressure_drop_segment_water_injection(seg: pd.Series, prod_row: pd.Series) -> float:
    rho = 1000.0
    rough = 1e-5

    T_C = _get_first(prod_row, keys=["T_C", "TEMP_C", "TEMP", "T"], default=60.0)

    p_Pa = _get_first(
        prod_row,
        keys=["P_Pa", "PRES_Pa", "PRESSURE_Pa", "BHP_Pa", "WHP_Pa", "P"],
        default=1e5,
    )

    S_wt = _get_first(prod_row, keys=["SALINITY_wt", "SAL_wt", "S_wt_pct"], default=np.nan)
    if not np.isfinite(S_wt):
        S_ppm = _get_first(prod_row, keys=["SALINITY_ppm", "TDS_ppm", "S_ppm"], default=0.0)
        S_wt = max(0.0, float(S_ppm)) / 10000.0

    mu = water_viscosity_mccain_Pa_s(T_C=T_C, p_Pa=p_Pa, salinity_wt_pct=S_wt)

    wopr = float(prod_row.get("WOPR", 0.0))
    wwpr = float(prod_row.get("WWPR", 0.0))
    ww_ir = float(prod_row.get("WWIR", 0.0))
    q_liq_m3_day = (wopr + wwpr) if (wopr + wwpr) > 0 else ww_ir
    q_liq_m3_s = q_liq_m3_day / 86400.0

    dmd = float(seg["dMD"])
    dtvd = float(seg["dTVD"])
    D = float(seg["ID_m"])

    dp_grav = rho * G * dtvd

    if not np.isfinite(D) or D <= 0 or q_liq_m3_s <= 0:
        return dp_grav

    A = np.pi * D * D / 4.0
    v = q_liq_m3_s / A
    Re = rho * v * D / mu if mu > 0 else 0.0

    if Re <= 0:
        return dp_grav

    if Re < 2100:
        f = 64.0 / Re
    else:
        f = (-1.8 * np.log10((rough / (3.7 * D)) ** 1.11 + 6.9 / Re)) ** -2

    dp_fric = f * (dmd / D) * (rho * v * v / 2.0)
    return dp_grav + dp_fric


def gas_pseudocritical_sutton(gas_sg: float) -> tuple[float, float]:
    g = float(gas_sg)
    Tpc_R = 169.2 + 349.5 * g - 74.0 * g * g
    Ppc_psia = 756.8 - 131.0 * g - 3.6 * g * g
    return Ppc_psia, Tpc_R


def z_factor_dak(p_Pa: float, T_K: float, gas_sg: float, max_iter: int = 50) -> float:
    p_psia = p_Pa / PA_PER_PSI
    T_R = T_K * 9.0 / 5.0

    Ppc_psia, Tpc_R = gas_pseudocritical_sutton(gas_sg)
    ppr = p_psia / Ppc_psia
    tpr = T_R / Tpc_R

    A1, A2, A3, A4, A5 = 0.3265, -1.0700, -0.5339, 0.01569, -0.05165
    A6, A7, A8 = 0.5475, -0.7361, 0.1844
    A9, A10, A11 = 0.1056, 0.6134, 0.7210

    rho_r = 0.27 * ppr / max(tpr, 1e-12)
    rho_r = max(rho_r, 1e-12)

    for _ in range(max_iter):
        Tr = tpr
        rr = rho_r

        c1 = A1 + A2 / Tr + A3 / Tr**3 + A4 / Tr**4 + A5 / Tr**5
        c2 = A6 + A7 / Tr + A8 / Tr**2
        c3 = A9 * (A7 / Tr + A8 / Tr**2)
        c4 = A10 / Tr**3

        z = (
            1.0
            + c1 * rr
            + c2 * rr**2
            - c3 * rr**5
            + c4 * (1.0 + A11 * rr**2) * rr**2 * math.exp(-A11 * rr**2)
        )

        z = max(z, 0.05)
        rho_new = 0.27 * ppr / (z * Tr)

        if abs(rho_new - rho_r) / max(rho_r, 1e-12) < 1e-8:
            return float(z)

        rho_r = max(rho_new, 1e-12)

    return float(z)


def gas_viscosity_lee_Pa_s(p_Pa: float, T_K: float, gas_sg: float, z: float) -> float:
    p_psia = p_Pa / PA_PER_PSI
    T_R = T_K * 9.0 / 5.0
    MW = 28.967 * float(gas_sg)

    rho_lbft3 = (p_psia * MW) / (max(z, 1e-8) * 10.7316 * max(T_R, 1e-8))
    rho_gcc = rho_lbft3 * 0.016018463

    K = (9.379 + 0.01607 * MW) * (T_R ** 1.5) / (209.2 + 19.26 * MW + T_R)
    X = 3.448 + 986.4 / T_R + 0.01009 * MW
    Y = 2.447 - 0.2224 * X

    mu_cP = 1e-4 * K * math.exp(X * (rho_gcc ** Y))
    return mu_cP * 1e-3


def gas_density_real(p_Pa: float, T_K: float, gas_sg: float, z: float) -> float:
    M = float(gas_sg) * 28.967e-3
    return (p_Pa * M) / (max(z, 1e-8) * R * max(T_K, 1e-8))


def _haaland_friction_factor(Re: float, eps: float, D: float) -> float:
    if Re < 2100:
        return 64.0 / max(Re, 1e-12)
    return (-1.8 * np.log10((eps / (3.7 * D)) ** 1.11 + 6.9 / Re)) ** -2


def pressure_drop_segment_gas_injection(seg: pd.Series, inj_row: pd.Series) -> float:
    dmd = float(seg["dMD"])
    dtvd = float(seg["dTVD"])
    D = float(seg["ID_m"])
    if not np.isfinite(D) or D <= 0:
        return 0.0

    rough = float(seg.get("rough_m", 1e-5))

    qg_std_sm3_d = _get_first(
        inj_row,
        keys=["WGIR", "QG_INJ_sm3_d", "QGINJ_sm3_d", "QGINJ", "GAS_INJ_RATE", "QG", "QGI"],
        default=0.0,
    )
    qg_std_sm3_s = qg_std_sm3_d / 86400.0

    gas_sg = _get_first(inj_row, keys=["GAS_SG", "SGG", "GG", "gas_sg"], default=0.65)

    Tsurf_C = _get_first(inj_row, keys=["T_SURF_C", "TSURF_C", "TWH_C", "T_SURFACE_C"], default=np.nan)
    Tbh_C = _get_first(inj_row, keys=["T_BH_C", "TBH_C", "TBOTTOM_C", "TDH_C", "T_DOWNHOLE_C"], default=np.nan)

    if np.isfinite(Tsurf_C) and np.isfinite(Tbh_C):
        T_C = 0.5 * (Tsurf_C + Tbh_C)
    else:
        T_C = _get_first(inj_row, keys=["T_C", "TEMP_C", "TEMP", "T"], default=60.0)

    T_K = T_C + 273.15

    p_Pa = _get_first(
        inj_row,
        keys=["P_Pa", "PRES_Pa", "PRESSURE_Pa", "BHP_Pa", "WHP_Pa", "P"],
        default=1e6,
    )

    z = z_factor_dak(p_Pa=p_Pa, T_K=T_K, gas_sg=gas_sg)
    rho = gas_density_real(p_Pa=p_Pa, T_K=T_K, gas_sg=gas_sg, z=z)
    mu = gas_viscosity_lee_Pa_s(p_Pa=p_Pa, T_K=T_K, gas_sg=gas_sg, z=z)

    dp_grav = rho * G * dtvd

    if qg_std_sm3_s <= 0:
        return dp_grav

    p_std = 101325.0
    T_std = 288.15
    z_std = 1.0
    rho_std = gas_density_real(p_Pa=p_std, T_K=T_std, gas_sg=gas_sg, z=z_std)

    m_dot = qg_std_sm3_s * rho_std
    q_act = m_dot / max(rho, 1e-12)

    A = np.pi * D * D / 4.0
    v = q_act / A
    Re = rho * v * D / max(mu, 1e-12)

    f = _haaland_friction_factor(Re=Re, eps=rough, D=D)
    dp_fric = f * (dmd / D) * (rho * v * v / 2.0)

    return dp_grav - dp_fric


def _get_rate(series: pd.Series, key: str) -> float:
    try:
        v = float(series.get(key, 0.0))
    except Exception:
        return 0.0
    return v if np.isfinite(v) else 0.0


def _activity_mask(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ACTIVITY_COLS if c in df.columns]
    if not cols:
        return pd.Series(False, index=df.index)
    return (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .ne(0.0)
        .any(axis=1)
    )


def _smooth_series_time(
    dates: pd.Series,
    values: pd.Series,
    window: int,
) -> pd.Series:
    series = pd.Series(values.to_numpy(), index=dates)
    series = series.interpolate(method="time", limit_direction="both")
    series = series.rolling(window=window, min_periods=1, center=True).mean()
    return pd.Series(series.to_numpy(), index=values.index)


def _smooth_thp_for_well(df_well: pd.DataFrame) -> pd.DataFrame:
    df_well = df_well.sort_values("DATE").copy()
    activity = _activity_mask(df_well)
    thp = pd.to_numeric(df_well["WTHPS"], errors="coerce")
    if activity.any():
        active_thp = thp.where(activity)
        if active_thp.notna().any():
            smoothed = _smooth_series_time(df_well["DATE"], active_thp, THP_SMOOTH_WINDOW)
            thp = thp.where(~activity, smoothed)
        else:
            thp = thp.where(~activity, DEFAULT_THP_BAR)
    df_well["WTHPS"] = thp
    return df_well


def _smooth_bhp_for_well(df_well: pd.DataFrame) -> pd.DataFrame:
    df_well = df_well.sort_values("DATE").copy()
    activity = _activity_mask(df_well)
    bhp = pd.to_numeric(df_well["BHP_bar"], errors="coerce")
    bhp_raw = bhp.copy()
    filled_mask = pd.Series(False, index=df_well.index)
    if activity.any():
        active_bhp = bhp.where(activity)
        if active_bhp.notna().any():
            smoothed = _smooth_series_time(df_well["DATE"], active_bhp, BHP_SMOOTH_WINDOW)
            bhp = bhp.where(~activity, smoothed)
            filled_mask = activity & bhp_raw.isna() & bhp.notna()
        else:
            bhp = bhp.where(~activity, DEFAULT_BHP_BAR)
            filled_mask = activity & bhp_raw.isna()
    df_well["BHP_bar"] = bhp
    if "BHP_STATUS" in df_well.columns:
        df_well["BHP_STATUS"] = df_well["BHP_STATUS"].where(~filled_mask, "DEFAULTED")
    return df_well


def _get_thp_bar(row: pd.Series) -> float:
    thp = _get_first(row, keys=["WTHPS", "WTHP"], default=np.nan)
    try:
        thp = float(thp)
    except Exception:
        thp = np.nan
    return thp


def well_type_from_prod_row(row: pd.Series) -> str:
    wgir = _get_rate(row, "WGIR")
    wwir = _get_rate(row, "WWIR")
    if wgir > 0.0:
        return "GAS_INJECTOR"
    if wwir > 0.0:
        return "WATER_INJECTOR"
    return "PRODUCER"


def api_to_oil_sg(api: float) -> float:
    return 141.5 / (api + 131.5)


def beggs_robinson_dead_oil_mu_pa_s(api: float, T_c: float) -> float:
    T_f = T_c * 9.0 / 5.0 + 32.0
    if T_f <= 0:
        T_f = 1.0
    Z = 3.0324 - 0.02023 * api
    Y = 10.0 ** Z
    x = Y * (T_f ** -1.163)
    mu_cp = (10.0 ** x) - 1.0
    return max(mu_cp, 0.01) * 1e-3


def water_mu_pa_s_simple(T_c: float) -> float:
    T_k = T_c + 273.15
    mu = 2.414e-5 * 10.0 ** (247.8 / (T_k - 140.0))
    return float(np.clip(mu, 1e-5, 5e-3))


def sutton_pseudocritical(sg_gas: float) -> tuple[float, float]:
    Tpc_R = 169.2 + 349.5 * sg_gas - 74.0 * sg_gas ** 2
    Ppc_psia = 756.8 - 131.0 * sg_gas - 3.6 * sg_gas ** 2
    return Tpc_R, Ppc_psia


def lee_gas_mu_pa_s(p_pa: float, T_k: float, sg_gas: float, z: float) -> float:
    if p_pa <= 0 or T_k <= 0:
        return 1e-5

    MW = 28.967 * sg_gas
    MW_kg_per_mol = (28.967e-3) * sg_gas

    z_use = max(z, 0.2)
    rho_g = p_pa * MW_kg_per_mol / (z_use * R * T_k)
    rho_g_gcc = rho_g / 1000.0

    T_R = T_k * 9.0 / 5.0

    K = ((9.379 + 0.01607 * MW) * (T_R ** 1.5)) / (209.2 + 19.26 * MW + T_R)
    X = 3.448 + 986.4 / T_R + 0.01009 * MW
    Y = 2.447 - 0.2224 * X

    mu_cp = 1e-4 * K * np.exp(X * (rho_g_gcc ** Y))
    return float(np.clip(mu_cp * 1e-3, 1e-6, 5e-4))


def haaland_friction_factor_darcy(Re: float, rel_rough: float) -> float:
    if Re <= 0:
        return 0.0
    if Re < 2100:
        return 64.0 / Re
    return float((-1.8 * np.log10((rel_rough / 3.7) ** 1.11 + 6.9 / Re)) ** -2)


def beggs_brill_holdup(C_L: float, Fr_m: float, theta_rad_flow: float, N_lv: float) -> float:
    C_L = float(np.clip(C_L, 1e-9, 1.0))
    Fr_m = max(float(Fr_m), 1e-12)

    L1 = 316.0 * C_L ** 0.302
    L2 = 0.0009252 * C_L ** (-2.4684)
    L3 = 0.1 * C_L ** (-1.4516)
    L4 = 0.5 * C_L ** (-6.738)

    is_segregated = ((C_L < 0.01 and Fr_m < L1) or (C_L >= 0.01 and Fr_m < L2))
    is_transition = (L2 < Fr_m < L3)
    is_intermittent = ((0.01 <= C_L < 0.4 and (L3 < Fr_m <= L1)) or (C_L >= 0.4 and (L3 < Fr_m <= L4)))
    is_distributed = ((C_L < 0.4 and Fr_m >= L4) or (C_L >= 0.4 and Fr_m > L4))

    def E0_for(regime: str) -> float:
        if regime == "segregated":
            a, b, c = 0.98, 0.4846, 0.0868
        elif regime == "intermittent":
            a, b, c = 0.845, 0.5351, 0.0173
        else:
            a, b, c = 1.065, 0.5824, 0.0609
        E0 = a * (C_L ** b) / (Fr_m ** c)
        return float(max(E0, C_L))

    def beta_for(regime: str, uphill: bool) -> float:
        if regime == "distributed":
            return 0.0
        if uphill:
            if regime == "segregated":
                d, e, f, g = 0.011, -3.768, 3.539, -1.614
            else:
                d, e, f, g = 2.96, 0.305, -0.4473, 0.0978
        else:
            d, e, f, g = 4.7, -0.3692, 0.1244, -0.5056

        arg = d * (C_L ** e) * (max(N_lv, 1e-12) ** f) * (Fr_m ** g)
        arg = max(arg, 1e-30)
        return float((1.0 - C_L) * np.log(arg))

    def B_of_theta(beta: float, theta: float) -> float:
        s = np.sin(1.8 * theta)
        return float(1.0 + beta * (s - (1.0 / 3.0) * (s ** 3)))

    uphill = (theta_rad_flow >= 0.0)

    if is_transition:
        A = (L3 - Fr_m) / (L3 - L2)
        B = 1.0 - A

        E_seg = E0_for("segregated")
        E_int = E0_for("intermittent")

        beta_seg = beta_for("segregated", uphill)
        beta_int = beta_for("intermittent", uphill)

        E_seg_t = B_of_theta(beta_seg, theta_rad_flow) * E_seg
        E_int_t = B_of_theta(beta_int, theta_rad_flow) * E_int
        E = A * E_seg_t + B * E_int_t
        return float(np.clip(E, C_L, 1.0))

    if is_segregated:
        reg = "segregated"
    elif is_intermittent:
        reg = "intermittent"
    elif is_distributed:
        reg = "distributed"
    else:
        reg = "intermittent"

    E0 = E0_for(reg)
    beta = beta_for(reg, uphill)
    E = B_of_theta(beta, theta_rad_flow) * E0
    return float(np.clip(E, C_L, 1.0))


def pressure_drop_multiphase(
    seg: pd.Series,
    prod_row: pd.Series,
    p_top_pa: float,
    *,
    rough: float = 1e-5,
    sigma_n_m: float = 0.03,
    t_surf_col: str = "T_SURF_C",
    t_bh_col: str = "T_BH_C",
    gas_sg_col: str = "GAS_SG",
    api_col: str = "OIL_API",
    q_o_col: str = "WOPR",
    q_w_col: str = "WWPR",
    q_g_col: str = "WGPR",
) -> float:
    dmd = float(seg["dMD"])
    dtvd = float(seg["dTVD"])
    D = float(seg["ID_m"])

    if not np.isfinite(D) or D <= 0 or not np.isfinite(dmd) or dmd <= 0:
        return 0.0

    T_surf_c = float(prod_row.get(t_surf_col, np.nan))
    T_bh_c = float(prod_row.get(t_bh_col, np.nan))
    if not np.isfinite(T_surf_c) or not np.isfinite(T_bh_c):
        T_avg_c = 60.0
    else:
        T_avg_c = 0.5 * (T_surf_c + T_bh_c)

    T_avg_k = T_avg_c + 273.15

    gas_sg = float(prod_row.get(gas_sg_col, 0.65))
    api = float(prod_row.get(api_col, 35.0))

    q_o = abs(float(prod_row.get(q_o_col, 0.0))) / 86400.0
    q_w = abs(float(prod_row.get(q_w_col, 0.0))) / 86400.0
    q_l = q_o + q_w

    q_g_std = abs(float(prod_row.get(q_g_col, 0.0))) / 86400.0

    if q_l <= 0.0 and q_g_std <= 0.0:
        rho_fallback = 1000.0
        return rho_fallback * G * dtvd

    A = np.pi * D * D / 4.0

    p_top_pa = max(float(p_top_pa), 1e5)
    z = z_factor_dak(p_top_pa, T_avg_k, gas_sg)
    MW_kg_per_mol = 28.967e-3 * gas_sg
    rho_g = p_top_pa * MW_kg_per_mol / (max(z, 0.2) * R * T_avg_k)

    q_g = q_g_std * (P_STD_PA / p_top_pa) * (max(z, 0.2) / 1.0) * (T_avg_k / T_STD_K)

    v_sl = q_l / A
    v_sg = q_g / A
    v_m = v_sl + v_sg
    if v_m <= 0:
        return 1000.0 * G * dtvd

    C_L = v_sl / v_m
    Fr_m = (v_m ** 2) / (G * D)

    rho_w = 1000.0
    sg_o = api_to_oil_sg(api)
    rho_o = sg_o * 1000.0
    wc = (q_w / q_l) if q_l > 0 else 0.0
    rho_l = (1.0 - wc) * rho_o + wc * rho_w

    mu_w = water_mu_pa_s_simple(T_avg_c)
    mu_o = beggs_robinson_dead_oil_mu_pa_s(api, T_avg_c)
    if q_l > 0:
        xw = wc
        xo = 1.0 - xw
        mu_l = float(np.exp(xo * np.log(max(mu_o, 1e-9)) + xw * np.log(max(mu_w, 1e-9))))
    else:
        mu_l = mu_w

    sin_alpha = float(np.clip(dtvd / dmd, -1.0, 1.0))
    theta_flow = float(np.arcsin(abs(sin_alpha)))

    sigma = max(float(sigma_n_m), 1e-6)
    N_lv = v_sl * (rho_l / (G * sigma)) ** 0.25

    E_L = beggs_brill_holdup(C_L, Fr_m, theta_flow, N_lv)

    rho_m = rho_l * E_L + rho_g * (1.0 - E_L)

    rho_ns = rho_l * C_L + rho_g * (1.0 - C_L)
    mu_g = lee_gas_mu_pa_s(p_top_pa, T_avg_k, gas_sg, z)
    mu_ns = (mu_l ** C_L) * (mu_g ** (1.0 - C_L)) if (mu_l > 0 and mu_g > 0) else max(mu_l, mu_g, 1e-6)

    Re_ns = rho_ns * v_m * D / max(mu_ns, 1e-12)

    rel_rough = rough / D
    f_ns = haaland_friction_factor_darcy(Re_ns, rel_rough)

    y = C_L / max(E_L, 1e-9) ** 2
    if y <= 1.0:
        S = 0.0
    elif 1.0 < y < 1.2:
        S = float(np.log(2.2 * y - 1.2))
    else:
        ln_y = float(np.log(y))
        denom = (-0.0523 + 3.182 * ln_y - 0.8725 * (ln_y ** 2) + 0.01853 * (ln_y ** 4))
        S = float(ln_y / denom) if denom != 0 else 0.0

    f_tp = f_ns * np.exp(S)

    dpdl_fric = f_tp * (rho_ns * v_m * v_m) / (2.0 * D)
    dpdl_elev = rho_m * G * (dtvd / dmd)

    Ek = (rho_m * v_m * v_sg) / max(p_top_pa, 1e5)
    Ek = float(np.clip(Ek, 0.0, 0.95))

    dp_segment = (dpdl_fric + dpdl_elev) * dmd / (1.0 - Ek)
    return float(dp_segment)


def _compute_bhp_for_well(wk: str, prod_rows: pd.DataFrame, segs: pd.DataFrame) -> pd.DataFrame:
    segs_sorted = segs.sort_values("md_top").reset_index(drop=True)

    def compute_row(row: pd.Series) -> float:
        thp_bar = _get_thp_bar(row)
        if not np.isfinite(thp_bar):
            return np.nan

        wtype = well_type_from_prod_row(row)

        thp_pa = float(thp_bar) * BAR_TO_PA
        p_local = thp_pa
        dp_total = 0.0

        for _, seg in segs_sorted.iterrows():
            row_local = row.copy()
            row_local["P_Pa"] = p_local

            if wtype == "GAS_INJECTOR":
                dp = float(pressure_drop_segment_gas_injection(seg, row_local))
            elif wtype == "WATER_INJECTOR":
                dp = float(pressure_drop_segment_water_injection(seg, row_local))
            else:
                dp = float(pressure_drop_multiphase(seg, row_local, p_top_pa=p_local))

            if not np.isfinite(dp):
                return np.nan

            dp_total += dp
            p_local += dp

        bhp_pa = thp_pa + dp_total
        return bhp_pa * PA_TO_BAR

    prod_rows = prod_rows.copy()
    prod_rows["BHP_bar"] = prod_rows.apply(compute_row, axis=1)
    prod_rows["BHP_STATUS"] = np.where(prod_rows["BHP_bar"].isna(), "INSUFFICIENT_DATA", "OK")
    return prod_rows


def _pool_initializer():
    np.seterr(all="warn")


def _process_well(task: tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]):
    """
    Process a single well on one worker: build segments then compute BHP for all rows of that well.
    Returns (wk, segments_df, prod_with_bhp_df, error_message_or_None).
    """
    wk, traj_w, top_perf_w, comp_w, prod_w = task
    try:
        if top_perf_w.empty:
            raise ValueError("No top_perf rows for well.")
        perf_row = top_perf_w.sort_values("MDSTART").iloc[0]
        perf_md = float(perf_row["MDSTART"])

        seg_w = build_segments_for_well(wk, traj_w, perf_md, comp_w)
        if seg_w.empty:
            raise ValueError("No segments generated (empty).")

        prod_bhp = _compute_bhp_for_well(wk, prod_w, seg_w)
        return (wk, seg_w, prod_bhp, None)
    except Exception as e:
        return (wk, None, None, str(e))


def main(processes: int | None = None):
    completion = pd.read_csv("../input/well_modelling/completion.csv", low_memory=False)
    top_perf   = pd.read_csv("../input/well_modelling/topperf.csv",    low_memory=False)
    prod       = pd.read_csv("../input/well_modelling/pypdm_history.csv",       low_memory=False)
    traj       = pd.read_csv("../input/well_modelling/traj.csv",       low_memory=False)
    wellmap    = pd.read_csv("../input/well_modelling/WELLS.names",    low_memory=False)
    try:
        prod_existing = pd.read_csv("../input/well_modelling/prod_with_bhp.csv", low_memory=False)
    except FileNotFoundError:
        prod_existing = None
    try:
        segments_existing = pd.read_csv("../input/well_modelling/segments_geometry.csv", low_memory=False)
    except FileNotFoundError:
        segments_existing = None

    ensure_columns(
        wellmap,
        ["RMS_WELL", "PDM_WELL", ("CSD_WELLBORE", "CSD_WELL")],
        "wellsnames.csv",
    )
    ensure_columns(
        completion,
        ["WELLBORE_ID", "SYMBOL_NAME", "RKB_HANGER", "ID"],
        "completion.csv",
    )
    ensure_columns(top_perf, ["WELL", ("MDSTART", "MD_TOP")], "top_perf.csv")
    ensure_columns(prod, ["WELL", "DATE", ("WTHPS", "WTHP")], "prod.csv")
    ensure_columns(traj, ["WELL", "X", "Y", "TVD", "MD"], "traj.csv")

    alias_first_available(wellmap, "CSD_WELLBORE", ["CSD_WELL"])
    alias_first_available(top_perf, "MDSTART", ["MD_TOP"])
    alias_first_available(prod, "WTHPS", ["WTHP"])

    prod["DATE"] = pd.to_datetime(prod["DATE"], errors="coerce")

    completion = completion.copy()
    completion["ID_m"] = completion["ID"].astype(float) * INCH_TO_M

    completion = completion.merge(
        wellmap[["CSD_WELLBORE"]],
        left_on="WELLBORE_ID",
        right_on="CSD_WELLBORE",
        how="left",
    )
    completion["WELL_KEY"] = completion["CSD_WELLBORE"]

    prod = prod.merge(
        wellmap[["PDM_WELL", "CSD_WELLBORE"]],
        left_on="WELL",
        right_on="PDM_WELL",
        how="left",
    )
    prod["WELL_KEY"] = prod["CSD_WELLBORE"]

    traj = traj.merge(
        wellmap[["RMS_WELL", "CSD_WELLBORE"]],
        left_on="WELL",
        right_on="RMS_WELL",
        how="left",
    )
    traj["WELL_KEY"] = traj["CSD_WELLBORE"]

    top_perf = top_perf.merge(
        wellmap[["RMS_WELL", "CSD_WELLBORE"]],
        left_on="WELL",
        right_on="RMS_WELL",
        how="left",
    )
    top_perf["WELL_KEY"] = top_perf["CSD_WELLBORE"]

    print("=== Mapping coverage ===")
    report_mapping_coverage(prod, "PROD", "WELL")
    report_mapping_coverage(traj, "TRAJ", "WELL")
    report_mapping_coverage(top_perf, "TOP_PERF", "WELL")
    report_mapping_coverage(completion, "COMPLETION", "WELLBORE_ID")

    prod_m = prod.dropna(subset=["WELL_KEY"]).copy()
    prod_m = (
        prod_m.groupby("WELL_KEY", group_keys=False)
        .apply(_smooth_thp_for_well)
        .reset_index(drop=True)
    )
    traj_m = traj.dropna(subset=["WELL_KEY"]).copy()
    top_perf_m = top_perf.dropna(subset=["WELL_KEY"]).copy()
    completion_m = completion.dropna(subset=["WELL_KEY"]).copy()

    diff_wells = find_prod_diff_wells(prod, prod_existing)
    if not diff_wells:
        print("\nNo differences detected between prod input and existing prod_with_bhp.csv.")

    wells_prod = set(prod_m["WELL_KEY"].unique())
    wells_traj = set(traj_m["WELL_KEY"].unique())
    wells_perf = set(top_perf_m["WELL_KEY"].unique())
    wells_comp = set(completion_m["WELL_KEY"].unique())
    valid_wells = (wells_prod & wells_traj & wells_perf & wells_comp) & set(diff_wells)

    print("\n=== Well availability ===")
    print(f"prod wells      : {len(wells_prod)}")
    print(f"traj wells      : {len(wells_traj)}")
    print(f"top_perf wells  : {len(wells_perf)}")
    print(f"completion wells: {len(wells_comp)}")
    print(f"valid wells     : {len(valid_wells)}")

    if diff_wells and not valid_wells:
        print("\nNo wells have sufficient data AFTER mapping for the diff set.")
        print("Sample WELL_KEY from prod     :", _head_list(sorted(wells_prod), 10))
        print("Sample WELL_KEY from traj     :", _head_list(sorted(wells_traj), 10))
        print("Sample WELL_KEY from top_perf :", _head_list(sorted(wells_perf), 10))
        print("Sample WELL_KEY from completion:", _head_list(sorted(wells_comp), 10))

    print("\n=== Building segments + computing BHP in parallel (per well) ===")
    completion_tubing = completion_m[completion_m["SYMBOL_NAME"].astype(str).str.strip().eq("Tubing")].copy()

    # Pre-group per-well data to keep each well's processing on a single worker
    traj_by_wk = {wk: df.copy() for wk, df in traj_m.groupby("WELL_KEY")}
    comp_by_wk = {wk: df.copy() for wk, df in completion_tubing.groupby("WELL_KEY")}
    top_perf_by_wk = {wk: df.copy() for wk, df in top_perf_m.groupby("WELL_KEY")}
    prod_by_wk = {wk: df.copy() for wk, df in prod_m.groupby("WELL_KEY")}

    tasks = []
    for wk in sorted(valid_wells):
        traj_w = traj_by_wk.get(wk)
        comp_w = comp_by_wk.get(wk, pd.DataFrame(columns=completion_tubing.columns))
        top_perf_w = top_perf_by_wk.get(wk, pd.DataFrame(columns=top_perf_m.columns))
        prod_w = prod_by_wk.get(wk, pd.DataFrame(columns=prod_m.columns))
        if traj_w is None or top_perf_w.empty or prod_w.empty:
            continue
        tasks.append((wk, traj_w, top_perf_w, comp_w, prod_w))

    if not tasks and valid_wells:
        raise RuntimeError(
            "No wells remain after filtering for trajectory/top_perf/prod data. "
            "Check mapping coverage and that top_perf/prod contain rows for the same WELL_KEY."
        )

    segments_list = []
    prod_results = []
    skipped = []
    if tasks:
        with mp.Pool(processes=processes, initializer=_pool_initializer) as pool:
            for wk, seg_w, prod_w_bhp, err in pool.imap_unordered(_process_well, tasks):
                if err:
                    skipped.append((wk, err))
                else:
                    segments_list.append(seg_w)
                    prod_results.append(prod_w_bhp)

    if skipped:
        print("Skipped wells (sample up to 20):")
        for wk, msg in skipped[:20]:
            print(f"  {wk}: {msg}")

    segments_list = [df for df in segments_list if not df.empty]

    segments = pd.concat(segments_list, ignore_index=True) if segments_list else pd.DataFrame()
    prod_calc = pd.concat(prod_results, ignore_index=True) if prod_results else pd.DataFrame(columns=list(prod_m.columns) + ["BHP_bar", "BHP_STATUS"])
    if not prod_calc.empty:
        prod_calc = (
            prod_calc.groupby("WELL_KEY", group_keys=False)
            .apply(_smooth_bhp_for_well)
            .reset_index(drop=True)
        )

    prod_out = prod.copy()
    if prod_existing is not None and not prod_existing.empty:
        prod_existing["DATE"] = pd.to_datetime(prod_existing["DATE"], errors="coerce")
        prod_out = prod_out.merge(
            prod_existing[["WELL", "DATE", "WELL_KEY", "BHP_bar", "BHP_STATUS"]],
            on=["WELL", "DATE", "WELL_KEY"],
            how="left",
        )
    if not prod_calc.empty:
        prod_out = prod_out.merge(
            prod_calc[["WELL", "DATE", "WELL_KEY", "BHP_bar", "BHP_STATUS"]],
            on=["WELL", "DATE", "WELL_KEY"],
            how="left",
            suffixes=("", "_new"),
        )
        for col in ["BHP_bar", "BHP_STATUS"]:
            new_col = f"{col}_new"
            if new_col in prod_out.columns:
                prod_out[col] = prod_out[new_col].combine_first(prod_out[col])
                prod_out.drop(columns=[new_col], inplace=True)
    prod_out["BHP_STATUS"] = prod_out["BHP_STATUS"].fillna("INSUFFICIENT_DATA")

    if not segments.empty:
        if segments_existing is not None and not segments_existing.empty:
            segments_existing = segments_existing[~segments_existing["WELL_KEY"].isin(diff_wells)]
            segments = pd.concat([segments_existing, segments], ignore_index=True)
        segments.to_csv("../input/well_modelling/segments_geometry.csv", index=False)
        print("Wrote: segments_geometry.csv")

    prod_out.to_csv("../input/well_modelling/prod_with_bhp.csv", index=False)

    print("\nDone.")
    print("Wrote: prod_with_bhp.csv")


if __name__ == "__main__":
    main()
