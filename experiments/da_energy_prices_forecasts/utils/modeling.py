from dataclasses import dataclass

import numpy as np
import pandas as pd
from experiments.da_energy_prices_forecasts.utils.features import (
    ALL_COLUMNS,
    BASE_FEATURES,
    FEATURE_SETS,
    LINEAR_EXTRAS,
    LOCAL_TZ,
    RANDOM_STATE,
    RESLOAD,
    TARGET,
)
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

PEAK_HOURS_UTC = (16, 17, 18, 19)

MODEL_INFO = {
    # id: (ops, rodzina funkcja straty, zestawy danych)
    "M1": ("Ridge", "linear", "L2", ["Z0", "Z1", "Z2", "Z3"]),
    "M2": ("HubberRegressor", "linear", "Huber", ["Z1", "Z2"]),
    "M3": ("LGBM Huber", "tree", "Huber", ["Z0", "Z1", "Z2", "Z3"]),
    "M4": ("LGBM l1", "tree", "MAE", ["Z1", "Z3"]),
    "M5": ("ExtraTress", "tree", "L2", ["Z1", "Z3"]),
}

REGISTERED_MODEL_NAME = "da_price_fixing1"


@dataclass(frozen=True)
class Spec:
    model_id: str
    fset: str
    target: str = "raw"  # raw albo asinh

    @property
    def name(self) -> str:
        suffix = "" if self.target == "raw" else f"_{self.target}"
        return f"{self.model_id}_{self.fset}{suffix}"

    @property
    def family(self) -> str:
        return MODEL_INFO[self.model_id][1]


def split_days(days: list, n_tune: int = 35, n_holdout: int = 7):
    days = sorted(days)
    tune = days[:n_tune]
    walk_forward = days[n_tune : len(days) - n_holdout]
    holdout = days[-n_holdout:]
    return tune, walk_forward, holdout


def benchmark_predictions(df: pd.DataFrame, target_days=list) -> dict[str, pd.Series]:
    test = df[df["delivery_date"].isin(target_days)]
    data = {
        "B0_persistance_t24": test["price_lag_24"],
        "B1_persistance_typeday": test["price_lag_typeday"],
    }
    local_hour = df.index.tz_convert(LOCAL_TZ).hour
    piv = df.assign(_h=np.asarray(local_hour)).pivot_table(
        index="delivery_date", columns="_h", values=TARGET, aggfunc="mean"
    )
    keys = pd.MultiIndex.from_arrays(
        [test["delivery_date"].to_numpy(), np.asarray(test.index.tz_convert(LOCAL_TZ).hour)]
    )
    for n in (7, 14):
        prof = piv.rolling(n, min_periods=n).mean().shift(1).stack()
        data[f"B2_profile_{n}d"] = pd.Series(prof.reindex(keys).to_numpy(), index=test.index)
    return data


def compute_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_b0: pd.Series | None = None
) -> dict[str, float]:
    idx = y_pred.dropna().index.intersection(y_true.index)
    y, p = y_true.loc[idx], y_pred.loc[idx]
    err = p - y
    abs_err = err.abs()
    peak = np.isin(idx.hour, PEAK_HOURS_UTC)
    days = pd.Series(np.asarray(idx.tz_convert(LOCAL_TZ).date), index=idx)
    daily = abs_err.groupby(days).mean()

    m = {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "wape": float(abs_err.sum() / y.abs().sum()),
        "bias": float(err.mean()),
        "mae_peak_16_19": float(abs_err[peak].mean()) if peak.any() else float("nan"),
        "bias_peak_16_19": float(err[peak].mean()) if peak.any() else float("nan"),
        "daily_mae_median": float(daily.median()),
        "daily_mae_iqr": float(daily.quantile(0.75) - daily.quantile(0.25)),
        "n_obs": float(len(idx)),
    }
    if y_b0 is not None:
        b0 = y_b0.reindex(idx)
        m["rmae"] = float(m["mae"] / (b0 - y).abs().mean())
    return m


############################ MODELOWANIE#############################


def walk_forward(model, data: pd.DataFrame, target_days: list) -> pd.Series:
    """
    Dla każdej doby D model się uczy na wszystkich dniach przed D, prognozuje 24 godziny dla dnia D.
    Hiperparametry są zamrożone
    """
    preds = []
    for day in target_days:
        train = data[data["delivery_date"] < day]
        test = data[data["delivery_date"] == day]
        if train.empty or test.empty:
            continue
        wf_model = clone(model).fit(train[ALL_COLUMNS], train[TARGET])
        preds.append(pd.Series(wf_model.predict(test[ALL_COLUMNS]), index=test.index))
    return pd.concat(preds).sort_index()


def tune(spec: Spec, data: pd.DataFrame, tune_days: list, n_splits: int = 3):
    """
    Strojenie hiperparametrów.
    Zwraca (best_params, cv_mae)
    """
    sub = data[data["delivery_date"].isin(tune_days)]
    search = GridSearchCV(
        build_estimator(spec),
        param_grid(spec),
        scoring="neg_mean_absolute_error",
        cv=day_cv_splits(sub["delivery_date"], n_splits),
        refit=False,
        n_jobs=-1,
    )
    search.fit(sub[ALL_COLUMNS], sub[TARGET])
    return search.best_params_, float(-search.best_score_)


def param_grid(spec: Spec) -> dict[str, list]:
    p = "regressor__reg__"
    m_id = spec.model_id
    if m_id == "M1":
        return {p + "alpha": list(np.logspace(-2, 3, 20))}
    if m_id == "M2":
        return {p + "epsilon": [1.35, 2.0], p + "alpha": [1e-4, 1e-2, 1.0]}
    if m_id in ("M3", "M4"):
        grid = {
            p + "num_leaves": [4, 8],
            p + "learning_rate": [0.03, 0.1],
            p + "n_estimators": [200, 400],
            p + "min_child_samples": [20, 40],
            p + "reg_lambda": [1, 10],
        }
        if m_id == "M3":
            grid[p + "alpha"] = [50, 100] if spec.target == "raw" else [0.2, 0.5]
        return grid
    if m_id == "M5":
        return {p + "min_samples_leaf": [5, 10, 20], p + "max_features": [0.5, 1.0]}
    raise ValueError(m_id)


def day_cv_splits(dates: pd.Series, n_splits: int = 3):
    dates = pd.Series(np.asarray(dates))
    days = np.array(sorted(dates.unique()))
    splits = []
    for train, validation in TimeSeriesSplit(n_splits=n_splits).split(days):
        train_idx = np.flatnonzero(dates.isin(days[train]).to_numpy())
        validation_idx = np.flatnonzero(dates.isin(days[validation]).to_numpy())
        splits.append((train_idx, validation_idx))
    return splits


def build_estimator(spec: Spec):
    """
    Wybór kolumn -> (skalowanie) -> model -> transformacja celu
    """

    steps = [("pre", _preprocessor(spec))]
    if spec.family == "linear":
        steps.append(("scale", StandardScaler()))
    steps.append(("reg", _regressor(spec)))
    pipe = Pipeline(steps)
    if spec.target == "asinh":
        return TransformedTargetRegressor(
            regressor=pipe, func=asinh100, inverse_func=sinh100, check_inverse=False
        )
    return TransformedTargetRegressor(regressor=pipe, func=None, inverse_func=None)


def asinh100(y):
    return np.arcsinh(np.asarray(y, dtype=float) / 100.0)


def sinh100(z):
    return 100.0 * np.sinh(np.asarray(z, dtype=float))


def model_columns(spec: Spec) -> list[str]:
    cols = [*BASE_FEATURES, *FEATURE_SETS[spec.fset]]
    if spec.family == "linear":
        cols += LINEAR_EXTRAS
    if "pv_fcst_pse" not in cols:
        cols = [c for c in cols if c not in ("pv_x_sin_d1", "pv_x_cos_d1")]
    return cols


def _regressor(spec: Spec):
    m_id = spec.model_id
    if m_id == "M1":
        return Ridge(random_state=RANDOM_STATE)
    if m_id == "M2":
        return HuberRegressor(max_iter=2000)

    lgbm_common = dict(
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )

    if m_id == "M3":
        return LGBMRegressor(objective="huber", **lgbm_common)
    if m_id == "M4":
        return LGBMRegressor(objective="l1", **lgbm_common)
    if m_id == "M5":
        return ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)
    raise ValueError(f"Nieznany model {m_id}")


def _preprocessor(spec: Spec) -> ColumnTransformer:
    cols = model_columns(spec)
    if spec.family == "linear":
        spline_cols = [RESLOAD] if RESLOAD in cols else []
        rest = [c for c in cols if c not in spline_cols]
        transformers = []
        if spline_cols:
            transformers.append(
                (
                    "spline",
                    SplineTransformer(n_knots=5, degree=3, include_bias=False),
                    spline_cols,
                )
            )
        transformers.append(("pass", "passthrough", rest))
        return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return ColumnTransformer(
        [("pass", "passthrough", cols)], remainder="drop", verbose_feature_names_out=False
    )


def default_matrix() -> list[Spec]:
    return [Spec(m_id, fs) for m_id, info in MODEL_INFO.items() for fs in info[3]]


def build_ensemble(specs: list[Spec], params: list[dict]) -> VotingRegressor:
    members = []
    for spec, par in zip(specs, params, strict=True):
        est = build_estimator(spec)
        est.set_params(**par)
        members.append((spec.name, est))
    return VotingRegressor(members)
