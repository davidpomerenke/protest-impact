from functools import partial

import pandas as pd
from darts.models import RegressionModel
from dowhy import CausalModel
from dowhy.causal_estimators.propensity_score_weighting_estimator import (
    PropensityScoreWeightingEstimator,
)
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
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


class DowhyWrapper(MultiOutputMixin, RegressorMixin, BaseEstimator):
    def __init__(
        self, df: pd.DataFrame, y: list[str], w: str, x: list[str], **fit_args
    ) -> None:
        """
        x: should include w, because it will be in the same dataframe
        """
        self.df = df
        self.y = y
        self.w = w
        self.x = x
        self.fit_args = fit_args

    def fit(self, X, y):
        X = pd.DataFrame(X, columns=self.x)
        y = pd.DataFrame(y, columns=self.y)
        df = y.join(X)
        self.df = df
        self.causal_model = CausalModel(
            data=df,
            treatment=self.w,
            outcome=self.y,
            common_causes=[c for c in self.x if c != self.w],
            graph=None,
        )
        self.estimand = self.causal_model.identify_effect()
        self.estimator = PropensityScoreWeightingEstimator(
            self.estimand, propensity_score_model=LogisticRegression()
        ).fit(df, **self.fit_args)
        return self

    def predict(self, X):
        pass

    def estimate(self):
        return self.estimator.estimate_effect(data=self.df, target_units="att")


# @cache
def propensity(w: str, lags: int, steps: int = 1):
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
    propensity_model = DowhyWrapper(
        df=pd.DataFrame(),
        y=y[0].columns,
        w=w + "_futcov_lag0",
        x=feature_names,
    )
    ts_model = RegressionModel(model=propensity_model, **model_args)
    ts_model.fit(**fit_args)
    return ts_model
