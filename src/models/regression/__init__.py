from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from darts import TimeSeries
from darts.models import RegressionModel
from darts.utils.timeseries_generation import (
    linear_timeseries,
    random_walk_timeseries,
    sine_timeseries,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from src.cache import cache
from src.features.aggregation import naive_all_regions
from src.features.time_series.lagged_impact import lagged_impact


# @cache
def regression(max_lags=0, include_controls=True, media_source="mediacloud"):
    data = naive_all_regions(media_source=media_source)
    results = lagged_impact(data.y, data.x, ts_lr)
    return results


class SMWrapper(BaseEstimator, RegressorMixin):
    """A universal sklearn-style wrapper for statsmodels regressors"""

    def __init__(self, model_class, fit_intercept=True, **fit_args):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.fit_args = fit_args

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit(**self.fit_args)
        self.coef_ = self.results_.params[1:]
        self.intercept_ = self.results_.params[0]
        self.conf_int_ = self.results_.conf_int()[1:]
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


sk_ols = SMWrapper(sm.OLS, fit_args=dict(cov_type="HC3"))


def ts_results(
    y: list[pd.DataFrame], x: list[pd.DataFrame], model: BaseEstimator, lags=14
):
    def merge_with_statics(dfs: list[pd.DataFrame]) -> pd.DataFrame:
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
        )

    x = merge_with_statics(x)
    y = [TimeSeries.from_dataframe(df) for df in y]
    model = RegressionModel(
        lags=None if lags == 0 else lags,
        lags_future_covariates=(lags, 1),
        model=model,
    )
    model.fit(y, future_covariates=x)
    coefs = retrieve_params(model, list(y[0].columns))
    # if (conf_int := retrieve_conf_int(model)) is not None:
    #     conf_ints = [
    #         dict(
    #             target=name,
    #             lower=ci["occurrence"][0][0],
    #             upper=ci["occurrence"][0][1],
    #         )
    #         for name, ci in conf_int.items()
    #     ]
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


def _decode_param_names(
    coefficients: list, feature_names: list[str], return_type: str
) -> pd.DataFrame:
    df = pd.DataFrame({"feature_name": feature_names, return_type: coefficients})
    name_parts = df["feature_name"].str.rsplit("_", n=2)
    df["predictor"] = name_parts.str[0]
    df["lag"] = name_parts.str[2].str.replace("lag", "").astype(int)
    df = df.drop(columns=["feature_name"])
    return df


def _retrieve_params(
    return_type: str, model: RegressionModel, targets: list[str]
) -> dict[str, pd.DataFrame]:
    mm = model.model
    key = return_type + "_"
    if isinstance(mm, MultiOutputRegressor):
        if not hasattr(mm.estimators_[0], key):
            return None
        params = [getattr(estimator, key) for estimator in mm.estimators_]
    else:
        if not hasattr(mm, key):
            return None
        params = getattr(mm, key)
    dfs = []
    for target, param in zip(targets, params):
        df = _decode_param_names(param, model.lagged_feature_names, return_type)
        df["target"] = target
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def retrieve_coefficients(model: RegressionModel, targets: list[str]) -> pd.DataFrame:
    return _retrieve_params("coef", model, targets)


def retrieve_params(model: RegressionModel, targets: list[str]) -> pd.DataFrame:
    coef = _retrieve_params("coef", model, targets)
    conf_int = _retrieve_params("conf_int", model, targets)
    if conf_int is not None:
        coef = coef.merge(conf_int, on=["target", "predictor", "lag"])
    return coef


def test_retrieve_coefficients():
    a = linear_timeseries(length=1000, start_value=0, end_value=10)
    b = sine_timeseries(length=1000, value_frequency=0.01, value_amplitude=5)
    c = random_walk_timeseries(length=1000, mean=0)
    d = random_walk_timeseries(length=1000, mean=0)
    p = sine_timeseries(length=1000, value_frequency=0.03, column_name="sth1")
    f = random_walk_timeseries(length=1000, mean=0, column_name="sth2")

    y = a.stack(b).stack(c).stack(d)
    model = RegressionModel(
        lags=4, lags_past_covariates=4, lags_future_covariates=(4, 1)
    )
    model.fit(y, past_covariates=p, future_covariates=f)
    coefs = retrieve_coefficients(model)

    for target in coefs["target"].unique():
        coef = coefs[coefs["target"] == target]
        assert len(coef) == 5
        assert len(coef.columns) == 6
        assert np.isnan(coef["linear"][0])
        assert np.isnan(coef["sine"][0])
        assert np.isnan(coef["random_walk"][0])
        assert np.isnan(coef["random_walk_1"][0])
        assert np.isnan(coef["sth1"][0])
        assert not np.isnan(coef["sth2"][0])
        same = coefs[(coefs["target"] == target) & (coef["predictor"] == target)]
        assert same.sum() > 0.95
    assert (
        coefs[
            (coefs["target"] == "random_walk")
            & (coef["predictor"] == "random_walk")
            & (coef["lag"] == -1)
        ]
        > 0.8
    )
    assert (
        coefs[
            (coefs["target"] == "random_walk_1")
            & (coef["predictor"] == "random_walk_1")
            & (coef["lag"] == -1)
        ]
        > 0.8
    )
