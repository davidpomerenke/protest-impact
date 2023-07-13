import json
import warnings

import numpy as np
import pandas as pd

# from neuralforecast.losses.pytorch import RMSE, MSE
from pygam import LinearGAM, f, s
from pygam.terms import TermList
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from protest_impact.util import project_root


class GAM(BaseEstimator, RegressorMixin):
    def __init__(self, n_splines=20, spline_order=3, lam=0.6):
        self.gam = None
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.lam = lam

    def fit(self, X, y):
        kwargs = dict(
            n_splines=self.n_splines, spline_order=self.spline_order, lam=self.lam
        )
        splines = [s(i, **kwargs) for i in range(X.shape[1])]
        self.gam = LinearGAM(TermList(*splines))
        self.gam.fit(X, y)
        return self

    def predict(self, X):
        return self.gam.predict(X)


class AutoRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, lags):
        self.regressor = regressor
        self.lags = lags

    def fit(self, X, y):
        self.train_length = len(y)
        y = pd.Series(y, name="y")
        X = pd.DataFrame(X, columns=[f"X_{i}" for i in range(X.shape[1])])
        self.forecaster = ForecasterAutoreg(
            regressor=self.regressor,
            lags=self.lags,
        )
        self.forecaster.fit(y=y, exog=X)
        return self

    def predict(self, X):
        index = pd.RangeIndex(start=self.train_length, stop=self.train_length + len(X))
        X = pd.DataFrame(X, index=index, columns=[f"X_{i}" for i in range(X.shape[1])])
        return self.forecaster.predict(steps=len(X), exog=X).values


class BayesianStructuralTimeSeries(BaseEstimator, RegressorMixin):
    def __init__(self, fit_method="hmc", nseasons=7, prior_level_sd=0.1):
        self.fit_method = fit_method
        self.nseasons = nseasons
        self.prior_level_sd = prior_level_sd

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        from causalimpact import CausalImpact

        y = [pd.Series(self.y_train), pd.Series([-np.inf] * len(X))]
        y = pd.concat(y, ignore_index=True).rename("y")
        _X = [pd.DataFrame(self.X_train), pd.DataFrame(X)]
        _X = pd.concat(_X, ignore_index=True)
        df = pd.concat([y, _X], axis=1)
        split = len(self.y_train)
        ci = CausalImpact(
            df,
            [0, split - 1],
            [split, len(df) - 1],
            model_args=dict(
                nseasons=self.nseasons,
                prior_level_sd=self.prior_level_sd,
                fit_method=self.fit_method,
            ),
        )
        y_pred = ci.inferences.complete_preds_means.values[split:]
        return y_pred


class NeuralEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, horizon, agg_weekly=False, model_name="nbeatsx", max_steps=200):
        self.horizon = horizon
        self.agg_weekly = agg_weekly
        self.model_name = model_name
        self.max_steps = max_steps

    def fit(self, X, y):
        import logging

        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import RMSE
        from neuralforecast.models import NBEATS, NHITS, NBEATSx

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("pytorch").setLevel(logging.ERROR)
        logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
        # warnings.filterwarnings("ignore", category=UserWarning)
        self.X_columns = [f"X_{i}" for i in range(X.shape[1])]
        _model = dict(NBEATS=NBEATS, NHITS=NHITS, NBEATSx=NBEATSx)[self.model_name]
        models = [
            _model(
                h=self.horizon,
                input_size=X.shape[1],
                futr_exog_list=self.X_columns,
                loss=RMSE(),
                scaler_type="standard",
                max_steps=self.max_steps,
            )
        ]
        freq = "W" if self.agg_weekly else "D"
        self.model = NeuralForecast(models=models, freq=freq)
        X_train = pd.DataFrame(X, columns=self.X_columns)
        y_train = pd.Series(y, name="y")
        df_train = pd.concat([y_train, X_train], axis=1)
        df_train["unique_id"] = [1] * len(y_train)
        df_train["ds"] = list(range(-len(y_train), 0))
        self.model.fit(df=df_train)

    def predict(self, X):
        df_test = pd.DataFrame(X, columns=self.X_columns)
        df_test["y"] = np.nan
        df_test["unique_id"] = [1] * len(X)
        df_test["ds"] = list(range(0, len(X)))
        Y_pred = self.model.predict(futr_df=df_test)
        Y_pred = Y_pred[self.model_name].values
        if len(X) > len(Y_pred):
            Y_pred = np.concatenate([[np.nan] * (len(X) - len(Y_pred)), Y_pred])
        return Y_pred


class MeanEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X.mean(axis=1)


with open(project_root / "data" / "regions" / "distance-weights.json") as f:
    distance_weights = json.load(f)


class DistanceWeightsEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, event_region, control_regions, inverse=False):
        self.event_region = event_region
        self.control_regions = control_regions
        self.inverse = inverse

    def fit(self, X, y):
        self.weights = [
            distance_weights[self.event_region][r] for r in self.control_regions
        ]
        if self.inverse:
            self.weights = [1 / w for w in self.weights]
        self.weights = [w / sum(self.weights) for w in self.weights]

    def predict(self, X):
        return X.dot(self.weights)


df_sdi = pd.read_csv(project_root / "data/regions/sociodemographic-indicators.csv")
df_sdi = df_sdi.set_index(df_sdi.columns[0], drop=True)
# the saldo has negative values, drop it for the positive methods
df_sdi = df_sdi.T.drop(columns=["Wanderungssaldo je 10.000 EW"]).T
df_sdi = pd.DataFrame(
    MinMaxScaler().fit_transform(df_sdi.T).T, index=df_sdi.index, columns=df_sdi.columns
)


class SociodemographicWeightsEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, event_region, control_regions, method, sum_to_one=True):
        self.event_region = event_region
        self.control_regions = control_regions
        self.method = method
        self.sum_to_one = sum_to_one

    def fit(self, X, y):
        _X = df_sdi[self.control_regions]
        _y = df_sdi[self.event_region]
        m = len(self.control_regions)
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=ConvergenceWarning)
        if self.method == "corr":
            self.weights = _X.corrwith(_y)
            self.weights = self.weights / self.weights.sum()
        elif self.method == "ols":
            model = LinearRegression(fit_intercept=False, positive=True).fit(_X, _y)
            self.weights = model.coef_
        elif self.method == "pc":
            pipe = make_pipeline(
                PCA(n_components=min(m, 4)), LinearRegression(fit_intercept=False)
            ).fit(_X, _y)
            self.weights = np.dot(
                pipe.steps[0][1].components_.T, pipe.steps[1][1].coef_
            )
        elif self.method == "pls":
            pls = PLSRegression(n_components=min(m, 3)).fit(_X, _y)
            self.weights = pls.coef_
        elif self.method == "nmf":
            pipe = make_pipeline(
                NMF(n_components=min(m, 4), max_iter=1000),
                LinearRegression(fit_intercept=False, positive=True),
            ).fit(_X, _y)
            self.weights = np.dot(
                pipe.steps[0][1].components_.T, pipe.steps[1][1].coef_
            )
        if self.sum_to_one:
            self.weights = self.weights / self.weights.sum()

    def predict(self, X):
        y_pred = np.dot(X, self.weights)
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values
        return y_pred


class SleepingScaler(TransformerMixin, BaseEstimator):
    # This scaler is sleeping and doesn't do anything.
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class MeanScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.mean = X.mean(axis=0)
        assert self.mean.shape[0] == X.shape[1]
        assert np.all(self.mean > 0)
        return self

    def transform(self, X):
        return X / self.mean

    def inverse_transform(self, X):
        return X * np.expand_dims(self.mean, axis=0)
