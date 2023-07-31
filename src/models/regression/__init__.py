from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from src.cache import cache
from src.features.aggregation import naive_all_regions
from src.features.time_series.lagged_impact import lagged_impact
from src.models.util.darts_helpers import retrieve_params
from src.models.util.statsmodels_wrapper import sk_ols


@cache
def regression(max_lags=0, include_controls=True, n_days=10, media_source="mediacloud"):
    data = naive_all_regions(media_source=media_source)
    if not include_controls:
        # only keep the treatments (occurrence of protests)
        data.x = [df[[c for c in df.columns if c.startswith("occ_")]] for df in data.x]
    ts_ols = partial(ts_results, model=sk_ols, lags=max_lags)
    results = lagged_impact(data.y, data.x, ts_ols)
    return results


def get_ts_list_with_statics(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    value_cols = dfs[0].columns
    for i, df in enumerate(dfs):
        groups = pd.Series(i, index=df.index)
        df["group"] = groups
        df.reset_index(inplace=True)
    df = pd.concat(dfs)
    dummies = pd.get_dummies(df["group"], prefix="SERIES")
    df = pd.concat([df, dummies], axis=1)
    return TimeSeries.from_group_dataframe(
        df,
        group_cols="group",
        time_col="index",
        value_cols=value_cols,
        static_cols=list(dummies.columns),
    )  # this returns a list!


def ts_results(
    y: list[pd.DataFrame], x: list[pd.DataFrame], model: BaseEstimator, lags=14
):
    x = get_ts_list_with_statics(x)  # list of ts
    y = get_ts_list_with_statics(y)  # list of ts
    model = RegressionModel(
        lags=None if lags == 0 else lags,
        lags_future_covariates=(lags, 1),
        model=model,
    )
    model.fit(y, future_covariates=x)
    coefs = retrieve_params(model, list(y[0].columns))
    return coefs


ts_lr = partial(ts_results, model=LinearRegression())

ts_ols = partial(ts_results, model=sk_ols)


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
