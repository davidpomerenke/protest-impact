from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.models import RegressionModel
from sklearn.base import BaseEstimator
from sklearn.linear_model import BayesianRidge, Lasso, LassoLarsIC, LinearRegression
from sklearn.preprocessing import OneHotEncoder

from src.cache import cache
from src.features.aggregation import naive_all_regions
from src.models.util.darts_helpers import retrieve_params
from src.models.util.statsmodels_wrapper import SMWrapper


@cache
def regression(
    lags=0,
    steps=7,
    gap=0,
    cumulative=False,
    include_controls=True,
    media_source="mediacloud",
):
    data = naive_all_regions(media_source=media_source)
    if not include_controls:
        # only keep the treatments (occurrence of protests)
        data.x = [df[[c for c in df.columns if c.startswith("occ_")]] for df in data.x]
    sk_ols = SMWrapper(
        sm.OLS, fit_args=dict(cov_type="HC3"), fit_intercept=False
    )  # the intercept can be dropped because the static variable dummies do not drop the first column
    f = ts_results_cumulative if cumulative else ts_results
    results = f(data.y, data.x, model=sk_ols, lags=lags, steps=steps, gap=gap)
    return results


def get_ts_list_with_statics(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    for i, df in enumerate(dfs):
        groups = pd.Series(f"SERIES{i}", index=df.index)
        df["group"] = groups
        df.reset_index(inplace=True)
    df = pd.concat(dfs)
    ts_list = TimeSeries.from_group_dataframe(df, group_cols="group", time_col="index")
    ohe = OneHotEncoder(drop=None)
    transformer = StaticCovariatesTransformer(transformer_cat=ohe)
    ts_list = transformer.fit_transform(ts_list)
    return ts_list


def ts_results(
    y: list[pd.DataFrame],
    x: list[pd.DataFrame],
    model: BaseEstimator,
    lags: int,
    steps: int = 1,
    gap: int = 0,
):
    x = get_ts_list_with_statics(x)  # list of ts
    y = get_ts_list_with_statics(y)  # list of ts
    model = RegressionModel(
        lags=None if lags == 0 else list(range(-lags - gap, -gap)),
        lags_future_covariates=(lags + gap, 1),
        model=model,
        output_chunk_length=steps,
    )
    model.fit(y, future_covariates=x)
    coefs = retrieve_params(model, list(y[0].columns))
    return coefs


def ts_results_cumulative(
    y: list[pd.DataFrame],
    x: list[pd.DataFrame],
    model: BaseEstimator,
    lags: int,
    steps: int = 1,
    gap: int = 0,
):
    dfs = []
    for step in range(0, steps):
        res = _ts_results_cumulative(y, x, model, lags=lags, step=step, gap=gap)
        res["lag"] = res["lag"] + step
        res["step"] = step
        dfs.append(res)
    return pd.concat(dfs)


def _ts_results_cumulative(
    y: list[pd.DataFrame],
    x: list[pd.DataFrame],
    model: BaseEstimator,
    lags: int,
    step: int = 0,
    gap: int = 0,
):
    x = [dfx.copy() for dfx in x]
    y = [dfy.copy() for dfy in y]
    x = get_ts_list_with_statics(x)  # list of ts
    y_rolling = get_ts_list_with_statics(
        [dfy.rolling(step + 1).sum().dropna() for dfy in y]
    )
    y = get_ts_list_with_statics(y)  # list of ts
    model = RegressionModel(
        lags=None,
        lags_future_covariates=list(range(-lags - gap - step, 1 - step)),
        lags_past_covariates=None
        if lags == 0
        else list(range(-lags - gap - step, -gap - step)),
        model=model,
        output_chunk_length=1,
    )
    model.fit(y_rolling, future_covariates=x, past_covariates=y if lags > 0 else None)
    coefs = retrieve_params(model, list(y[0].columns))
    return coefs


def plot_lagged_impact(results: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 4))
    for target in ["protest", "not_protest"]:
        r = results[results["target"] == target]
        ax.plot(r["shift"], r["coef"], label=target, linewidth=1.5)
        if "lower" in r.columns and "upper" in r.columns:
            ax.fill_between(r["shift"], r["upper"], r["lower"], alpha=0.2)
    ax.set_xticks(range(-10, 11, 1))
    ax.set_xlabel("Shift (days)")
    ax.set_ylabel("Coefficient")
    ax.set_title("Naive regression coefficients")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    fig.tight_layout()
    return fig, ax
