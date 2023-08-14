import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dowhy import CausalModel
from dowhy.causal_estimators.instrumental_variable_estimator import (
    InstrumentalVariableEstimator,
)
from linearmodels.iv import IVLIML
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import Binarizer

from src.cache import cache
from src.features.time_series import get_lagged_df
from src.paths import models


def binarize_optimally(x, y):
    x = x.to_numpy().reshape(-1, 1)
    y = y.to_numpy().reshape(-1, 1)
    best_threshold, best_cov = None, None
    for threshold in np.linspace(x.min(), x.max(), 1000):
        X_bin = Binarizer(threshold=threshold).fit_transform(x)
        cov = np.cov(X_bin.flatten(), y.flatten())[0, 1]
        if best_cov is None or abs(cov) > abs(best_cov):
            best_threshold = threshold
            best_cov = cov
        x_bin = Binarizer(threshold=best_threshold).fit_transform(x).flatten()
    return x_bin, best_threshold


def get_data(instrument_prefix="weather_"):
    df = get_lagged_df(
        "media_print_protest",
        include_instruments=True,
        lags=range(-7, 1),
        ignore_group=True,
    )
    instruments = [
        c for c in df.columns if c.startswith(instrument_prefix) and c.endswith("lag0")
    ]
    treatment = "occ_protest_lag0"
    outcome = "media_print_protest"
    confounders = [
        c
        for c in df.columns
        if c not in [outcome, treatment] + instruments
        and not c.startswith("weather_")
        and not c.startswith("covid_")
    ]
    # normalize all instruments
    for instrument in instruments:
        df[instrument] = (df[instrument] - df[instrument].mean()) / df[instrument].std()
    return df, instruments, treatment, outcome, confounders


def get_covariances(instrument_prefix="weather_"):
    df, instruments, treatment, outcome, _ = get_data(instrument_prefix)

    # 1. with continuous instruments

    # get covariances between instruments and treatment
    covs = df[instruments + [treatment]].cov()
    covs_w = covs.loc[instruments, treatment]

    # get covariances between instruments and outcome
    covs = df[instruments + [outcome]].cov()
    covs_y = covs.loc[instruments, outcome]

    covs = pd.concat([covs_w, covs_y], axis=1, keys=["cov_w", "cov_y"])
    covs["wald"] = covs["cov_y"] / covs["cov_w"]
    covs_cont = covs

    bin_df = pd.DataFrame({treatment: df[treatment], outcome: df[outcome]})

    # 2. with binary instruments

    # discretize all instruments optimally
    for instrument in instruments:
        X_bin, threshold = binarize_optimally(df[instrument], df[treatment])
        print(f"{instrument}: {threshold:.3f}")
        bin_df[instrument] = X_bin

    # get covariances between instruments and treatment
    covs = bin_df[instruments + [treatment]].cov()
    covs_w = covs.loc[instruments, treatment]

    # get covariances between instruments and outcome
    covs = bin_df[instruments + [outcome]].cov()
    covs_y = covs.loc[instruments, outcome]

    covs = pd.concat([covs_w, covs_y], axis=1, keys=["cov_w", "cov_y"])
    covs["wald"] = covs["cov_y"] / covs["cov_w"]
    covs_bin = covs

    # 3. overview

    covs = pd.concat([covs_cont, covs_bin], axis=1, keys=["cont", "bin"])

    covs = covs.sort_values(ascending=False, key=abs, by=("cont", "cov_w"))
    return covs


def get_coefficients(instrument_prefix):
    df, instruments, treatment, outcome, confounders = get_data(instrument_prefix)
    params = dict()
    for instr in instruments:
        params[instr] = (
            sm.OLS(df[treatment], sm.add_constant(df[confounders + [instr]]))
            .fit()
            .params[instr]
        )
    params_single = pd.DataFrame.from_dict(params, orient="index", columns=["coef"])
    params_combi = pd.DataFrame(
        sm.OLS(df[treatment], sm.add_constant(df[confounders + instruments]))
        .fit()
        .params,
        columns=["coef"],
    )
    params_combi = params_combi[params_combi.index.str.startswith(instrument_prefix)]
    params = pd.concat([params_single, params_combi], axis=1, keys=["single", "combi"])

    params = params.sort_values(by=("single", "coef"), key=abs, ascending=False)
    return params


def get_rf_params():
    df, instruments, treatment, outcome, confounders = get_data()
    path = models / "instrumental_variable/weather"
    path.mkdir(parents=True, exist_ok=True)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_features": ["sqrt", "log2"],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced"],
    }

    rf = RandomForestClassifier()
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=4, verbose=2, scoring="f1"
    )

    # 1. model only with confounders
    f_params_1 = path / "rf1_params.json"
    if not f_params_1.exists():
        gs1 = grid_search.fit(df[confounders], df[treatment])
        with open(f_params_1, "w") as f:
            json.dump(gs1.best_params_, f)
        with open(path / "rf1_score.json", "w") as f:
            json.dump(gs1.best_score_, f)
        rf1 = gs1.best_params_
    else:
        with open(f_params_1, "r") as f:
            rf1 = json.load(f)

    # 2. model with confounders and instruments
    f_params_2 = path / "rf2_params.json"
    if not f_params_2.exists():
        gs2 = grid_search.fit(df[confounders + instruments], df[treatment])
        with open(f_params_2, "w") as f:
            json.dump(gs2.best_params_, f)
        with open(path / "rf2_score.json", "w") as f:
            json.dump(gs2.best_score_, f)
        rf2 = gs2.best_params_
    else:
        with open(f_params_2, "r") as f:
            rf2 = json.load(f)
    return rf1, rf2


@cache
def check_for_improvement(instrument_prefix="weather_"):
    """
    Does adding the instruments improve the model for predicting the treatment?
    """
    df, instruments, treatment, outcome, confounders = get_data(
        instrument_prefix=instrument_prefix
    )
    rf1, rf2 = get_rf_params()
    classifiers = [
        LogisticRegression(max_iter=1000, solver="saga", class_weight="balanced"),
        RandomForestClassifier(**rf1),
    ]
    tscv = TimeSeriesSplit(n_splits=20)
    results = dict()
    for cls in classifiers:
        cvs1 = cross_val_score(
            cls, df[confounders], df[treatment], cv=tscv, scoring="f1", n_jobs=4
        )
        cvs2 = cross_val_score(
            cls,
            df[confounders + instruments],
            df[treatment],
            cv=tscv,
            scoring="f1",
            n_jobs=4,
        )
        # hypothesis test whether cvs2 is better than cvs1
        p = stats.ttest_rel(cvs2, cvs1, alternative="greater").pvalue
        results[cls.__class__.__name__] = (cvs1, cvs2, p)
    return results


# time series methods:


@cache
def _instrumental_variable(
    target: str, treatment: str, instrument: str, lagged_df: pd.DataFrame
):
    """
    For use with src/models/time_series.py.
    """
    assert (
        treatment == "occ_protest"
    ), "general iv cannot control for other protest groups"
    assert instrument == "weather_prcp"
    treatment_ = treatment + "_lag0"
    instrument_ = instrument + "_lag0"
    lagged_df[instrument_] = lagged_df[instrument_] > 0
    model = CausalModel(
        data=lagged_df,
        treatment=treatment_,
        outcome=target,
        instruments=[instrument_],
    )
    estimand = model.identify_effect(method_name="maximal-adjustment")
    estimator = InstrumentalVariableEstimator(
        estimand,
        iv_instrument_name=instrument_,
        confidence_intervals=True,
    )
    estimator.fit(lagged_df)
    estimate = estimator.estimate_effect(lagged_df, target_units="att")
    ci = estimate.get_confidence_intervals()
    coefs = pd.DataFrame(
        dict(
            coef=[estimate.value],
            predictor=[treatment],
            ci_lower=ci[0],
            ci_upper=ci[1],
        )
    )
    return estimator, coefs


@cache
def _instrumental_variable_liml(
    target: str,
    treatment: str,
    instruments: list[str],
    lagged_df: pd.DataFrame,
    binarize: bool = False,
):
    """
    WIP.
    For use with src/models/time_series.py.
    """
    assert (
        treatment == "occ_protest"
    ), "general iv cannot control for other protest groups"
    assert isinstance(treatment, str)
    treatment_ = treatment + "_lag0"
    all_instruments = [
        c
        for c in lagged_df.columns
        if c.startswith("weather_") or c.startswith("covid_")
    ]
    instruments = [f"{instr}_lag0" for instr in instruments]
    if binarize:
        for instrument in instruments:
            lagged_df[instrument] = binarize_optimally(
                lagged_df[instrument], lagged_df[treatment_]
            )[0]
    confounders = lagged_df[
        [
            c
            for c in lagged_df.columns
            if not c in [target, treatment_] + all_instruments
        ]
    ]
    confounders = sm.add_constant(confounders)
    model = IVLIML(
        dependent=lagged_df[target],
        exog=confounders,
        endog=lagged_df[treatment_],
        instruments=lagged_df[instruments],
    )
    results = model.fit()
    ci = results.conf_int()
    coefs = pd.DataFrame(
        dict(
            coef=[results.params[0]],
            predictor=[treatment],
            ci_lower=ci["lower"][0],
            ci_upper=ci["upper"][0],
            sargan=results.sargan.pval,
        )
    )
    return None, coefs
