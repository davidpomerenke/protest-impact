from itertools import chain
from typing import Callable

import numpy as np
import pandas as pd
from darts import TimeSeries
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

LagDict = dict[tuple[int, int], list[str]]
Filter = Callable[[pd.DataFrame], pd.DataFrame]


class TimeSeriesRegressor:
    def __init__(
        self,
        model: BaseEstimator,
        y_cols: str | list[str],
        lags: LagDict | None,
    ):
        self.model = model
        self.y_cols = [y_cols] if isinstance(y_cols, str) else y_cols
        assert isinstance(self.y_cols, list)
        assert len(self.y_cols) > 0
        assert isinstance(self.y_cols[0], str)
        self.lags = lags

    def _prepare_data(
        self,
        df: pd.DataFrame,
        filter: Filter = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert set(self.y_cols).issubset(set(df.columns))
        lagged_columns = set(chain(*list(self.lags.values())))
        assert set(df.columns) == lagged_columns.union(set(self.y_cols))
        assert all(
            [k[0] >= 0 and k[1] >= k[0] for k in self.lags.keys()]
        ), "lag boundaries must be 0 or positive and in the right order"
        assert all(
            [k[0] > 0 for k, v in self.lags.items() if set(self.y_cols).intersection(v)]
        ), "y leaked"
        lagged_df = pd.DataFrame(
            {
                f"{column}_lag{lag}": df[column].shift(lag)
                for lags, columns in self.lags.items()
                for lag in range(lags[0], lags[1] + 1)
                for column in columns
            }
        ).dropna()
        assert len(lagged_df) == len(df) - max(
            [k[1] for k in self.lags.keys()], default=0
        )
        if filter is not None:
            lagged_df = filter(lagged_df)
        y = df[self.y_cols].loc[lagged_df.index]
        X_lagged = lagged_df
        return y, X_lagged

    def _fit(
        self,
        y: pd.DataFrame,
        X_lagged: pd.DataFrame,
    ):
        self.model.fit(X_lagged, y)
        self.X_lagged = X_lagged
        self.y = y
        return self

    def fit(self, df: pd.DataFrame, filter: Filter = None):
        assert isinstance(df, pd.DataFrame)
        y, X_lagged = self._prepare_data(df, filter)
        return self._fit(y, X_lagged)

    def fit_multiple(
        self,
        dfs: list[pd.DataFrame],
        static_covariates: pd.DataFrame | str,
        filters: list[Filter] = None,
    ):
        assert isinstance(dfs, (list, tuple))
        assert isinstance(dfs[0], pd.DataFrame)
        assert isinstance(static_covariates, pd.DataFrame) or static_covariates in [
            "same",
            "dummies",
        ]
        filters = filters or [None] * len(dfs)
        data = [self._prepare_data(df, fil) for df, fil in zip(dfs, filters)]
        if static_covariates == "same":
            pass
        elif static_covariates == "dummies":
            static_covariates = pd.get_dummies(
                pd.Series(range(len(dfs))), drop_first=True, prefix="dataset"
            )
            for i, (y, X_lagged) in enumerate(data):
                X_lagged[static_covariates.columns] = static_covariates.iloc[i]
        else:
            raise NotImplementedError
        y, X_lagged = zip(*data)
        return self._fit(pd.concat(y), pd.concat(X_lagged))

    def _predict(self, X_lagged: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.predict(X_lagged), index=X_lagged.index, columns=self.y.columns
        )

    def extract_coefficients(self, estimator, target_names, attr):
        if not hasattr(estimator, attr):
            return None
        coef = getattr(estimator, attr)
        if len(coef.shape) == 1:
            coef = np.array(coef).reshape(1, -1)
        df = (
            pd.DataFrame(
                coef,
                columns=self.X_lagged.columns,
                index=target_names,
            )
            .reset_index()
            .melt(id_vars="index", var_name="variable", value_name=attr[:-1])
            .rename(columns={"index": "target"})
            .assign(
                predictor=lambda df: df.variable.str.rsplit("_", n=1, expand=True)[0],
                lag=lambda df: df.variable.str.rsplit("_", n=1, expand=True)[1]
                .str.replace("lag", "")
                .astype(int),
            )
        )
        cols = ["target", "predictor", "lag", attr[:-1]]
        return df[cols].sort_values(by=cols)

    def _get_coefficients(self, attr):
        if isinstance(self.model, MultiOutputRegressor):
            coefs = [
                self.extract_coefficients(estimator, [name], attr)
                for estimator, name in zip(self.model.estimators_, self.y.columns)
            ]
            return None if any([c is None for c in coefs]) else pd.concat(coefs)
        else:
            return self.extract_coefficients(
                self.model,
                self.y.columns if hasattr(self.y, "columns") else [self.y.name],
                attr,
            )

    def get_coefficients(self):
        coefs = self._get_coefficients("coef_")
        lower = self._get_coefficients("lower_")
        upper = self._get_coefficients("upper_")
        if lower is not None and upper is not None:
            coefs = coefs.merge(lower, on=["target", "predictor", "lag"])
            coefs = coefs.merge(upper, on=["target", "predictor", "lag"])
        return coefs

    def get_most_important_coefficients(self, target: str) -> pd.DataFrame:
        df = self.get_coefficients()
        df = df[df["target"] == target]
        df["coef_"] = df["coef"].abs()
        df = df.sort_values(by="coef_", ascending=False).reset_index(drop=True)
        df = df.drop(columns=["coef_"])
        return df

    def from_darts_regression_model(self, regression_model):
        y = regression_model.training_series.pd_dataframe()
        X_past = (
            regression_model.past_covariate_series.pd_dataframe()
            if regression_model.uses_past_covariates
            else None
        )
        X_future = (
            regression_model.future_covariate_series.pd_dataframe()
            if regression_model.uses_future_covariates
            else None
        )
        return self.fit(y, X_past, X_future)

    def to_darts_regression_model(self, regression_model_class):
        y_ts = TimeSeries.from_dataframe(self.y)
        X_past_ts = (
            TimeSeries.from_dataframe(self.X_lagged)
            if "X_past_lag1" in self.X_lagged.columns
            else None
        )
        X_future_ts = (
            TimeSeries.from_dataframe(self.X_lagged)
            if "X_future_lag0" in self.X_lagged.columns
            else None
        )
        return regression_model_class(
            lags=None if self.lags_ar == 0 else self.lags_ar,
            lags_past_covariates=None if self.lags_past == 0 else self.lags_past,
            lags_future_covariates=None if self.lags_future == 0 else self.lags_future,
            model=self.model,
        ).fit(y_ts, past_covariates=X_past_ts, future_covariates=X_future_ts)


def test_create_lagged_df():
    df = pd.DataFrame({"a": range(10)})
    model = TimeSeriesRegressor(
        LinearRegression(), lags_ar=3, lags_past=3, lags_future=3
    )
    result = model._create_lagged_df(df, 3)
    print(result)
    assert (result.columns == ["a_lag1", "a_lag2", "a_lag3"]).all()
    assert result.shape == (7, 3)
    assert result.loc[3, "a_lag3"] == 0


def test_prepare_data():
    y = pd.DataFrame({"y": range(10)})
    X_past = pd.DataFrame({"X_past": range(10)})
    X_future = pd.DataFrame({"X_future": range(10)})
    model = TimeSeriesRegressor(
        LinearRegression(), lags_ar=3, lags_past=3, lags_future=3
    )
    y_result, X_result = model._prepare_data(y, X_past, X_future)
    print(y_result)
    print(X_result)
    assert y_result.shape == (7, 1)
    assert X_result.shape == (7, 10)
    assert "y_lag3" in X_result.columns
    assert "X_past_lag3" in X_result.columns
    assert "X_future_lag0" in X_result.columns
    assert "X_future_lag3" in X_result.columns
