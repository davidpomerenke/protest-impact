import json
from itertools import product

import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from munch import Munch
from sklearn.linear_model import BayesianRidge, LassoLarsIC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from tqdm.auto import tqdm

from src.cache import cache
from src.features.time_series import get_lagged_df
from src.paths import models


@cache
def objective(params):
    p = Munch(params)
    target = "media_combined_all"
    lagged_df = get_lagged_df(
        target=target,
        lags=p.lags,
        step=6,
        cumulative=True,
        ignore_group=False,
        ignore_medium=True,
        region_dummies=p.region_dummies,
        add_features=p.add_features,
    )
    # cv = TimeSeriesSplit(n_splits=5)
    cv = KFold(n_splits=5, shuffle=True)
    rmses = []
    r2s = []
    maes = []
    biases = []
    for train_index, test_index in cv.split(lagged_df):
        train = lagged_df.iloc[train_index]
        test = lagged_df.iloc[test_index]
        if p.model == "ols":
            model = sm.OLS(
                endog=train[target], exog=sm.add_constant(train.drop(columns=[target]))
            )
            results = model.fit(cov_type="HC3")
            y_pred = results.predict(sm.add_constant(test.drop(columns=[target])))
        elif p.model == "ridge":
            model = BayesianRidge()
            model.fit(train.drop(columns=[target]), train[target])
            y_pred = model.predict(test.drop(columns=[target]))
        elif p.model == "lasso":
            model = LassoLarsIC()
            model.fit(train.drop(columns=[target]), train[target])
            y_pred = model.predict(test.drop(columns=[target]))
        rmse = mean_squared_error(test[target], y_pred) ** 0.5
        r2 = r2_score(test[target], y_pred)
        rmses.append(rmse)
        r2s.append(r2)
        maes.append(np.mean(np.abs(y_pred - test[target])))
        biases.append(np.mean(y_pred - test[target]))
    params["rmse"] = np.mean(rmses)
    params["rmse_std"] = np.std(rmses)
    params["r2"] = np.mean(r2s)
    params["r2_std"] = np.std(r2s)
    params["mae"] = np.mean(maes)
    params["mae_std"] = np.std(maes)
    params["bias"] = np.mean(biases)
    params["bias_std"] = np.std(biases)
    return params


def hyperopt(model, n_jobs=4):
    params = dict(
        lags=[list(range(-i, 1)) for i in range(1, 15, 2)],
        region_dummies=[False],  # True
        add_features=[[], ["diff"], ["diff", "ewm"]],
        model=[model],
    )
    combinations = list(product(*params.values()))
    results = Parallel(n_jobs=n_jobs)(
        delayed(objective)(dict(zip(params.keys(), combination)))
        for combination in tqdm(combinations)
    )
    with open(models / "regression" / f"{model}_params.json", "w") as f:
        json.dump(results, f, indent=2)
    best_result = min(results, key=lambda r: np.abs(r["bias"]))
    return best_result


@cache
def best_regression(ignore_group=False):
    p = Munch(hyperopt("ols"))
    target = "media_combined_all"
    df = get_lagged_df(
        target=target,
        lags=p.lags,
        step=1,
        cumulative=True,
        ignore_group=ignore_group,
        ignore_medium=True,
        region_dummies=p.region_dummies,
        add_features=p.add_features,
    )
    model = sm.OLS(endog=df[target], exog=sm.add_constant(df.drop(columns=[target])))
    results = model.fit(cov_type="HC3")
    return results
