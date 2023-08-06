from functools import partial

import pandas as pd
from darts.models import RegressionModel
from dowhy import CausalModel
from dowhy.causal_estimators.propensity_score_weighting_estimator import (
    PropensityScoreWeightingEstimator,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.cache import cache
from src.features.aggregation import naive_all_regions
from src.models.regression import get_ts_list_with_statics


class DummyModel(BaseEstimator, RegressorMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X


class DowhyWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        df: pd.DataFrame,
        y: str,
        w: str,
        x: list[str],
        lagged_feature_names: list[
            str
        ] = None,  # this is needed because darts re-initializes the model 🙄😩🤬
        **fit_args
    ) -> None:
        self.df = df
        self.y = y
        self.w = w
        self.x = x
        self.fit_args = fit_args
        self.lagged_feature_names = lagged_feature_names

    def fit(self, X, y):
        assert isinstance(self.lagged_feature_names, list)
        w_col = self.w + "_futcov_lag0"
        x_col = [c for c in self.lagged_feature_names if c != w_col]
        X = pd.DataFrame(X, columns=self.lagged_feature_names)
        y = pd.DataFrame(y, columns=[self.y])
        df = y.join(X)
        self.df = df
        model = CausalModel(
            data=df,
            treatment=w_col,
            outcome=self.y,
            common_causes=x_col,
            graph=None,
        )
        estimand = model.identify_effect()
        self.estimator = PropensityScoreWeightingEstimator(
            estimand, propensity_score_model=LogisticRegression()
        )
        self.estimator.fit(df, **self.fit_args)
        return self

    def predict(self, X):
        pass

    def estimate(self):
        return self.estimator.estimate_effect(data=self.df, target_units="att")


# @cache
def propensity(lags: int, steps: int = 1, model=None):
    data = naive_all_regions()
    # y = [df[["media_online_protest"]].copy() for df in data.y]
    x = get_ts_list_with_statics(data.x)  # list of ts
    y = get_ts_list_with_statics(data.y)  # list of ts
    model_args = dict(
        lags=None if lags == 0 else lags,
        lags_future_covariates=(lags, 1),
        output_chunk_length=steps,
    )
    fit_args = dict(series=y, future_covariates=x)
    feature_names = (
        RegressionModel(model=DummyModel(), **model_args)
        .fit(**fit_args)
        .lagged_feature_names
    )
    model.lagged_feature_names = feature_names
    ts_model = RegressionModel(model=model, **model_args)
    ts_model.fit(**fit_args)
    return ts_model
