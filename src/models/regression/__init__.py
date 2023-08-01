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
from src.models.util.statsmodels_wrapper import SMWrapper, sk_ols


@cache
def regression(lags=0, gap=3, include_controls=True, media_source="mediacloud"):
    data = naive_all_regions(media_source=media_source)
    if not include_controls:
        # only keep the treatments (occurrence of protests)
        data.x = [df[[c for c in df.columns if c.startswith("occ_")]] for df in data.x]
    sk_ols = SMWrapper(
        sm.OLS, fit_args=dict(cov_type="HC3"), fit_intercept=False
    )  # the intercept can be dropped because the static variable dummies do not drop the first column
    results = ts_results(data.y, data.x, model=sk_ols, lags=lags, gap=gap)
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
    gap: int,
):
    x = get_ts_list_with_statics(x)  # list of ts
    y = get_ts_list_with_statics(y)  # list of ts
    model = RegressionModel(
        lags=None if lags == 0 else lags,
        lags_future_covariates=(lags - gap, lags + gap + 1),
        model=model,
        output_chunk_length=1 + gap,
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
