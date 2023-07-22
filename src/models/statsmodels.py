from functools import partial

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor


def lag_0_variants(df: pd.DataFrame, cols: list[str]) -> list[str]:
    assert len(cols) > 0
    # assert len([c for c in df.columns if c.rsplit("_", 1)[0] in cols]) == len(cols)
    return df[[c for c in df.columns if c.replace("_lag0", "") in cols]]


def all_lag_variants(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return df[[c for c in df.columns if c.rsplit("_", 1)[0] in cols]]


class Wrapper(BaseEstimator, RegressorMixin):
    """An sklearn-style wrapper for statsmodels regressors"""

    def __init__(
        self,
        model_class: callable,
        fit_intercept: bool = True,
        iv_cols: dict = None,
        fit_kwargs=dict(),
        **kwargs,
    ):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.iv_cols = iv_cols
        self.fit_kwargs = fit_kwargs
        self.kwargs = kwargs

    def fit(self, X, y):
        self._init_model(X, y)
        self.results_ = self.model.fit(**self.fit_kwargs)
        self.coef_ = self._coef()
        self.intercept_ = self.results_.params[0]
        self.lower_ = self._conf_int("lower")
        self.upper_ = self._conf_int("upper")
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


class SMWrapper(Wrapper):
    def _init_model(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model = self.model_class(y, X, **self.kwargs)

    def _coef(self):
        return self.results_.params[1:]

    def _conf_int(self, which: str):
        match which:
            case "lower":
                return self.results_.conf_int()[1:][0]
            case "upper":
                return self.results_.conf_int()[1:][1]
            case _:
                raise ValueError


class LMWrapper(Wrapper):
    def _init_model(self, X, y):
        exog = all_lag_variants(X, self.iv_cols["exog"])
        if self.fit_intercept:
            exog = sm.add_constant(exog)
        endog = lag_0_variants(X, self.iv_cols["endog"])
        instruments = lag_0_variants(X, self.iv_cols["instruments"])
        self.instr_cols = instruments.columns
        self.model = self.model_class(
            dependent=y, endog=endog, exog=exog, instruments=instruments, **self.kwargs
        )

    def _coef(self):
        coef = self.results_.params[1:]
        for col in self.instr_cols:
            # the time series model expects coefficients for all columns in X
            # but with IV, we do not get any for the instruments
            coef[col] = np.nan
        return coef

    def _conf_int(self, which: str):
        conf_int = self.results_.conf_int()[which][1:]
        for col in self.instr_cols:
            # the time series model expects coefficients for all columns in X
            # but with IV, we do not get any for the instruments
            conf_int[col] = np.nan
        return conf_int


SMLinearRegression = partial(
    MultiOutputRegressor, SMWrapper(sm.OLS, fit_kwargs=dict(cov_type="HC3"))
)
