import numpy as np
import pandas as pd
from darts.models import RegressionModel
from darts.utils.timeseries_generation import (
    linear_timeseries,
    random_walk_timeseries,
    sine_timeseries,
)
from sklearn.multioutput import MultiOutputRegressor


def _decode_param_names(
    coefficients: list, feature_names: list[str], return_type: str
) -> pd.DataFrame:
    if return_type == "conf_int":
        df = pd.DataFrame({"feature_name": feature_names})
        df["ci_lower"] = coefficients[:, 0]
        df["ci_upper"] = coefficients[:, 1]
    elif return_type == "coef":
        df = pd.DataFrame({"feature_name": feature_names, "coef": coefficients})
    # deal with static covariates, which do not have lags
    # but have a name like "SERIES_13_statcov_target_global_components"
    df["feature_name"] = df["feature_name"].str.replace(
        "_target_global_components", "_lag0"
    )
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
    targets_ = []
    for i in range(len(params)):
        for target in targets:
            targets_.append((i, target))
    dfs = []
    for (step, target), param in zip(targets_, params):
        df = _decode_param_names(param, model.lagged_feature_names, return_type)
        df["target"] = target
        df["step"] = step
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def retrieve_coefficients(model: RegressionModel, targets: list[str]) -> pd.DataFrame:
    return _retrieve_params("coef", model, targets)


def retrieve_params(model: RegressionModel, targets: list[str]) -> pd.DataFrame:
    coef = _retrieve_params("coef", model, targets)
    conf_int = _retrieve_params("conf_int", model, targets)
    if conf_int is not None:
        coef = coef.merge(conf_int, on=["target", "predictor", "lag", "step"])
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
