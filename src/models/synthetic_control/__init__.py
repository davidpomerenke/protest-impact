from functools import partial
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib.parallel import Parallel, delayed
from scipy import stats
from scipy.optimize import fmin_slsqp
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

from src.cache import cache
from src.features.aggregation import all_regions
from src.paths import processed_data


def loss_w(W, X, y) -> float:
    # from https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html
    # Copyright (c) 2020 Matheus Facure
    return np.sqrt(np.mean((y - X.dot(W)) ** 2))


def get_w_by_interpolation(X, y):
    # from https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html
    # Copyright (c) 2020 Matheus Facure
    w_start = [1 / X.shape[1]] * X.shape[1]
    weights = fmin_slsqp(
        partial(loss_w, X=X, y=y),
        np.array(w_start),
        f_eqcons=lambda x: np.sum(x) - 1,
        bounds=[(0.0, 1.0)] * len(w_start),
        disp=False,
    )
    return weights


def get_w_by_regression(X, y):
    return LinearRegression(fit_intercept=False, positive=True).fit(X, y).coef_


def get_features(df: pd.DataFrame, scale, rolling, idx_pre, date_) -> pd.DataFrame:
    """
    Reformulate the outcome dimensions so that all the boolean queries are positive.
    This should be easier to interpolate.
    """
    df = df[[c for c in df.columns if c.startswith("media_")]]
    df = df.rolling(rolling).mean()
    if scale == "demean":
        df = df - df[idx_pre].mean()
    if scale == "demean_end":
        df = df - df.loc[date_ - pd.Timedelta(days=1)]
    if scale == "log":
        # arcsinh approximates log for x > 1
        # and is also defined for smaller values
        df = np.arcsinh(df)
    if scale == "diff":
        df = np.arcsinh(df).diff(1)
    return df


@cache
def synthetic_control_single(
    region: str,
    date_: pd.Timestamp,
    rolling: int = 1,
    scale: Literal["demean", "demean_end", "log", "diff", None] = "demean",
    pre_period: int = 28,
    post_period: int = 28,
    treatment: str = "occ_protest",
    ignore_group: bool = True,
    ignore_medium: bool = False,
) -> tuple[pd.Series, pd.Series] | None:
    dfs = all_regions(ignore_group=ignore_group, protest_source="acled", positive_queries=True, ignore_medium=ignore_medium)
    df_w = [df for name, df in dfs if name == region][0]
    control_regions = [
        (name, df) for name, df in dfs if df[df.index == date_].iloc[0][treatment] == 0
    ]
    if len(control_regions) == 0:
        print(f"No control regions for {region} on {date_}")
        return None
    idx_pre = (df_w.index >= date_ - pd.Timedelta(days=pre_period)) & (
        df_w.index < date_
    )
    idx_post = (df_w.index >= date_) & (
        df_w.index < date_ + pd.Timedelta(days=post_period)
    )
    idx_all = idx_pre | idx_post
    df_w = get_features(df_w, scale, rolling, idx_pre, date_)
    df_c = [
        get_features(df, scale, rolling, idx_pre, date_) for _, df in control_regions
    ]
    y = df_w[idx_pre].stack().reset_index(level=0, drop=True)
    X = pd.concat(
        [df[idx_pre].stack().reset_index(level=0, drop=True) for df in df_c],
        keys=[name for name, _ in control_regions],
        axis=1,
    )
    X = sm.add_constant(X)
    weights = get_w_by_regression(X, y)
    y_all = df_w[idx_all]
    y_c_all = pd.DataFrame()
    for col in y_all.columns:
        X = pd.concat([df[idx_all][col] for df in df_c], axis=1)
        X = sm.add_constant(X)
        y_c_all[col] = X.values.dot(weights)
    y_c_all.index = y_all.index
    return y_all, y_c_all


@cache(ignore=["n_jobs"])
def compute_synthetic_controls(
    pre_period: int = 28,
    post_period: int = 28,
    rolling: int = 1,
    scale: str | None = "demean",
    treatment: str = "occ_protest",
    ignore_group: bool = True,
    ignore_medium: bool = False,
    n_jobs: int = 4,
):
    dfs = all_regions(ignore_group=ignore_group, protest_source="acled", positive_queries=True,
                       ignore_medium=ignore_medium)
    protest_dates = []
    for name, df in dfs:
        dates = df[df[treatment] == 1].index
        for date_ in dates:
            protest_dates.append((name, date_))
    # maybe actually use lags and steps as pre_period and post_period?
    results = Parallel(n_jobs=n_jobs)(
        delayed(synthetic_control_single)(name, date_, rolling=rolling, scale=scale, treatment=treatment,
                                          ignore_group=ignore_group,
                                           ignore_medium=ignore_medium)
        for name, date_ in tqdm(protest_dates)
    )
    ys, y_cs = [], []
    for (name, date_), result in zip(protest_dates, results):
        if result is None:
            continue
        y, y_c = result
        y.index = (y.index - date_).days
        y_c.index = (y_c.index - date_).days
        y = y.reindex(range(-pre_period, post_period))
        y_c = y_c.reindex(range(-pre_period, post_period))
        ys.append(y)
        y_cs.append(y_c)
    return ys, y_cs


@cache
def synthetic_control(
    target: str | list[str],
    treatment: str,
    lags: Iterable[int],
    steps: Iterable[int],
    cumulative: bool = False,
    ignore_group: bool = False,
    ignore_medium: bool = False,
    positive_queries: bool = True,
    n_jobs: int = 4,
) -> pd.DataFrame:
    # assert treatment == "occ_protest"
    # assert ignore_group
    ys, y_cs = compute_synthetic_controls(
        # pre_period=len([s for s in lags if s < 0]),
        # post_period=len([s for s in steps if s >= 0]),
        treatment=treatment,
        ignore_group=ignore_group,
        ignore_medium=ignore_medium,
        n_jobs=n_jobs,
    )
    col_dfs = dict()
    for col in ys[0].columns:
        y_df = pd.concat([y[col] for y in ys], axis=1)
        y_c_df = pd.concat([y_c[col] for y_c in y_cs], axis=1)
        diff = y_df - y_c_df
        if cumulative:
            diff = diff[diff.index >= 0].cumsum()
        col_dfs[col] = diff
    if not positive_queries:
        for medium in ["online", "print", "combined"]:
            if f"media_{medium}_protest" in col_dfs:
                df1 = col_dfs[f"media_{medium}_all"].rename(columns=lambda c: c.removesuffix("_all"))
                df2 = col_dfs[f"media_{medium}_protest"].rename(columns=lambda c: c.removesuffix("_protest"))
                col_dfs[f"media_{medium}_not_protest"] = df1 - df2
    targets = [target] if isinstance(target, str) else target
    rows = []
    for target in targets:
        for step in steps:
            df = col_dfs[target].loc[step]
            ci_low, ci_high = stats.t.interval(0.95, len(df)-1, loc=df.mean(), scale=stats.sem(df.dropna()))
            rows.append(
                dict(
                    step=step,
                    target=target,
                    predictor=treatment,
                    coef=df.mean(),
                    ci_lower=ci_low,
                    ci_upper=ci_high,
                )
            )
    return pd.DataFrame(rows)
