"""
Inżyniera cech do modelu prognozy cen FIXING I
"""

import holidays
import numpy as np
import pandas as pd

RANDOM_STATE = 42

START_DATE = "2026-08-01"
END_DATE = "2026-10-07"

LOCAL_TZ = "Europe/Warsaw"

TARGET = "fixing_1_pln_mwh"
DEMAND = "demand_fcst_pse"
PV = "pv_fcst_pse"
WIND = "wind_fcst_pse"
EXCHANGE = "exchange_fcst_pse"
RESLOAD = "resload_fcst_pse"
RES_SHARE = "res_share"

TIME_FEATURES = [f"{fn}_d{k}" for k in (1, 2, 3) for fn in ("sin", "cos")] + [
    "sin_w",
    "cos_w",
    "is_weekend",
    "is_holiday",
    "is_day_off",
]

BASE_FEATURES = ["price_lag_24", *TIME_FEATURES]
LINEAR_EXTRAS = ["lag24_x_dayoff", "pv_x_sin_d1", "pv_x_cos_d1"]
FUNDAMENTALS = [DEMAND, PV, WIND, EXCHANGE, RESLOAD, RES_SHARE]

ALL_COLUMNS = [*BASE_FEATURES, *LINEAR_EXTRAS, *FUNDAMENTALS]

FEATURE_SETS: dict[str, list[str]] = {
    "Z0": [],
    "Z1": [DEMAND, PV, WIND, EXCHANGE],
    "Z2": [RESLOAD, EXCHANGE],
    "Z3": [RESLOAD, RES_SHARE, EXCHANGE],
}


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    idx = pd.DatetimeIndex(data.index)
    data.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return data.sort_index()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    local = data.index.tz_convert(LOCAL_TZ)
    hour = np.asarray(local.hour + local.minute / 60.0, dtype=float)
    dow = np.asarray(local.dayofweek)

    for k in (1, 2, 3):
        angle = 2 * np.pi * k * hour / 24.0
        data[f"sin_d{k}"] = np.sin(angle)
        data[f"cos_d{k}"] = np.cos(angle)

    t_week = dow * 24.0 + hour
    data["sin_w"] = np.sin(2 * np.pi * t_week / 168.0)
    data["cos_w"] = np.cos(2 * np.pi * t_week / 168.0)

    pl_holidays = holidays.country_holidays("PL", years=sorted(set(local.year)))
    local_dates = list(local.date)
    data["is_weekend"] = (dow >= 5).astype(float)
    data["is_holiday"] = np.array([d in pl_holidays for d in local_dates], dtype=float)
    data["is_day_off"] = np.maximum(data["is_weekend"], data["is_holiday"])
    data["delivery_date"] = local_dates
    return data


def add_price_lags(df: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    data = df.copy()
    data["price_lag_24"] = price.reindex(data.index - pd.Timedelta(hours=24)).to_numpy()
    data["price_lag_168"] = price.reindex(data.index - pd.Timedelta(hours=168)).to_numpy()

    dow = np.asarray(data.index.tz_convert(LOCAL_TZ).dayofweek)
    use_week = np.isin(dow, [0, 5, 6])
    data["price_lag_typeday"] = np.where(use_week, data["price_lag_168"], data["price_lag_24"])
    return data


def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if RESLOAD not in data:
        data[RESLOAD] = data[DEMAND] - data[PV] - data[WIND]
    data[RES_SHARE] = (data[PV] + data[WIND]) / data[DEMAND]
    return data


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["lag24_x_dayoff"] = data["price_lag_24"] * data["is_day_off"]
    data["pv_x_sin_d1"] = data[PV] * data["sin_d1"]
    data["pv_x_cos_d1"] = data[PV] * data["cos_d1"]
    return data


def make_training_frame(df: pd.DataFrame, dropna: bool = False) -> pd.DataFrame:
    data = df.copy()
    data = _ensure_utc_index(df)

    price = data[TARGET]
    data = add_time_features(data)
    data = add_price_lags(data, price)
    data = add_fundamental_features(data)
    data = add_interactions(data)

    feature_cols = [c for c in ALL_COLUMNS if c in data.columns]
    data[feature_cols] = data[feature_cols].astype("float64")
    if dropna:
        data = data.dropna(subset=[*ALL_COLUMNS, TARGET])
    return data
