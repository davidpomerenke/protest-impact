import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from dowhy import CausalModel
from dowhy.causal_estimators.instrumental_variable_estimator import (
    InstrumentalVariableEstimator,
)
from linearmodels.iv import IV2SLS, IVGMM, IVLIML
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from statsmodels import stats

from src.cache import cache
from src.features.time_series import get_lagged_df
from src.paths import models

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_data(instruments_, loadings=False):
    df = get_lagged_df(
        "media_combined_protest",
        instruments=instruments_,
        lags=range(-7, 1),
        step=6,
        cumulative=True,
        ignore_group=True,
        ignore_medium=True,
        region_dummies=True,
        positive_queries=True,
        return_loadings=loadings,
    )
    if loadings:
        return df
    instruments = [
        c
        for c in df.columns
        if c.startswith("weather_") or c.startswith("covid_") or c.startswith("pc_")
    ]
    treatment = "occ_protest_lag0"
    outcome = "media_combined_protest"
    confounders = [c for c in df.columns if c not in [outcome, treatment] + instruments]
    # normalize all instruments
    for instrument in instruments:
        df[instrument] = (df[instrument] - df[instrument].mean()) / df[instrument].std()
    return df, instruments, treatment, outcome, confounders


def get_covariances(instruments):
    df, instruments, treatment, outcome, _ = get_data(instruments)

    covs = pd.DataFrame()

    # get covariances between instruments and treatment
    covs_w = df[instruments + [treatment]].cov().loc[instruments, treatment]

    # get covariances between instruments and outcome
    covs_y = df[instruments + [outcome]].cov().loc[instruments, outcome]

    covs["wald"] = (covs_y / covs_w).apply(lambda x: round(x, 2))

    # get correlation between instruments and treatment

    covs["corr_w"] = (
        df[instruments + [treatment]]
        .corr()
        .loc[instruments, treatment]
        .apply(lambda x: round(x, 4))
    )

    # get correlation between instruments and outcome

    covs["corr_y"] = (
        df[instruments + [outcome]]
        .corr()
        .loc[instruments, outcome]
        .apply(lambda x: round(x, 4))
    )

    covs = covs[["corr_w", "corr_y", "wald"]]
    covs.index = covs.index.str.replace("_lag0", "")
    return covs


def get_coefficients(instruments_):
    df, instruments, treatment, outcome, confounders = get_data(instruments_)
    loadings_df = None
    results_combi = sm.OLS(
        df[treatment], sm.add_constant(df[confounders + instruments])
    ).fit()
    params = pd.DataFrame(
        dict(coef=results_combi.params, pval=results_combi.pvalues),
        index=results_combi.params.index,
    )
    params.columns = params.columns.str.replace("coef", "coef_w").str.replace(
        "pval", "p"
    )
    # correct for multiple testing with benjamini hochberg
    params["p_bh"] = stats.multitest.multipletests(
        params["p"], alpha=0.05, method="fdr_bh"
    )[1]
    params["p_by"] = stats.multitest.multipletests(
        params["p"], alpha=0.05, method="fdr_by"
    )[1]
    params["p"] = params["p"].apply(lambda x: round(x, 4))
    params["p_by"] = params["p_by"].apply(lambda x: round(x, 4))
    params["p_bh"] = params["p_bh"].apply(lambda x: round(x, 4))
    params["coef_w"] = params["coef_w"].apply(lambda x: round(x, 5))
    for instr in instruments:
        results = sm.OLS(df[treatment], sm.add_constant(df[[instr]])).fit()
        params.loc[instr, "fstat"] = results.fvalue.round(2)
    params = params[params.index.isin(instruments)]
    params.index = params.index.str.replace("_lag0", "")
    # params = params[["coef_w", "p", "p_bh", "p_by", "fstat"]]
    return params, loadings_df


def pc_vis(instruments, numbers, dfs=slice(None)):
    loadings_dfs = get_data(instruments, loadings=True)
    for loadings_df in loadings_dfs[dfs]:
        plt.figure(figsize=(15, 3))
        loadings = loadings_df.iloc[numbers]
        loadings.columns = (
            loadings.columns.str.replace("weather_", "")
            .str.replace("covid_", "")
            .str.replace("_lag0", "")
        )
        sns.heatmap(loadings.abs(), annot=loadings, cmap="Blues", linewidths=0.5)
        plt.title("PCA Component Loadings")
        plt.ylabel("Principal Components")
        plt.xlabel("Original Features")
        plt.show()


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
    lagged_df: pd.DataFrame,
    iv_instruments: list[str] = ["pc_resid_9"],
):
    """
    WIP.
    For use with src/models/time_series.py.
    """
    assert isinstance(treatment, str)
    treatment_ = treatment + "_lag0"
    all_instruments = [
        c
        for c in lagged_df.columns
        if (c.startswith("weather_") or c.startswith("covid_") or c.startswith("pc_"))
        and not "season" in c
    ]
    confounders = lagged_df[
        [
            c
            for c in lagged_df.columns
            # if "region" in c
            if not c in [target, treatment_] + all_instruments
        ]
    ]
    confounders = sm.add_constant(confounders)
    model = IVLIML(
        dependent=lagged_df[target],
        exog=confounders,
        endog=lagged_df[treatment_],
        instruments=lagged_df[iv_instruments],
    )
    results = model.fit()
    ci = results.conf_int()
    first_stage_model = IV2SLS(
        dependent=lagged_df[treatment_],
        exog=pd.concat([confounders, lagged_df[iv_instruments]], axis=1),
        endog=None,
        instruments=None,
    )
    first_stage_results = first_stage_model.fit()
    first_stage_ci = first_stage_results.conf_int()
    first_stage_solo_results = IV2SLS(
        dependent=lagged_df[treatment_],
        exog=lagged_df[iv_instruments],
        endog=None,
        instruments=None,
    ).fit()
    coefs = pd.DataFrame(
        dict(
            predictor=[treatment],
            coef=[results.params[treatment_]],
            ci_lower=ci["lower"][treatment_],
            ci_upper=ci["upper"][treatment_],
            # wooldridge=results.wooldridge_overid.pval,
            # anderson_rubin=results.anderson_rubin.pval,
            first_stage_coef=[first_stage_results.params[iv_instruments[0]]],
            first_stage_ci_lower=first_stage_ci["lower"][iv_instruments[0]],
            first_stage_ci_upper=first_stage_ci["upper"][iv_instruments[0]],
            first_stage_fstat=first_stage_solo_results.f_statistic.stat,
        )
    )
    return coefs
